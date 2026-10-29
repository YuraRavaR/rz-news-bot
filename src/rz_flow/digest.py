"""The Digest: one Events channel Post listing the coming weekend's events.

Built from stored Published events with an Event date in the weekend (Friday to
Sunday, Europe/Warsaw) and the weekend's Calendar entries, and posted at most once per
weekend (ADR 0001): a record is written as Publishing before sending and becomes
Published on success.

Calendar entries are translated, filtered and matched against the news events in one
Gemini request per Digest (none without entries), about one a week (ADR 0002). When
the Calendar cannot be read or that request fails, the Digest goes out with news events
only and the admin is alerted.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog

from rz_flow.ai import GeminiAIFilter
from rz_flow.city_calendar import read_weekend_calendar
from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig
from rz_flow.models import CalendarEntry, DatedEvent
from rz_flow.storage import StorageProtocol
from rz_flow.telegram import (
    _MAX_MESSAGE_LEN,
    _STAGING_POST_BANNER,
    TelegramPublisher,
    TelegramRateLimitedError,
    TelegramRejectedError,
    TelegramUnreachableError,
    _html_escape,
    format_event_date,
)

logger = structlog.get_logger(__name__)

# The weekend is the city's weekend
_DIGEST_TIMEZONE = ZoneInfo("Europe/Warsaw")

_FRIDAY = 4


_UA_WEEKEND_DAYS = ("Пʼятниця", "Субота", "Неділя")

_DRY_RUN_BANNER = (
    "<b>🧪 DRY RUN{mode}</b>\n"
    "<i>Попередній перегляд Дайджесту: у канал нічого не надіслано.</i>\n\n"
)

# A near-empty Digest is not worth a Post
MIN_DIGEST_ITEMS = 3


def _proves_not_delivered(exc: Exception) -> bool:
    """Failures that prove the Digest did not go out (ADR 0001): no connection, a 429,
    another 4xx. Anything else, a timeout after sending or a 5xx above all, may have
    delivered it.
    """
    if isinstance(exc, TelegramRejectedError):
        return 400 <= exc.status_code < 500
    return isinstance(exc, TelegramUnreachableError | TelegramRateLimitedError)


def digest_weekend(today: date) -> tuple[date, date]:
    """Friday and Sunday of the coming weekend: the current one from Friday to Sunday."""
    # Monday to Thursday look ahead to Friday; Saturday and Sunday look back to it
    friday = today + timedelta(days=_FRIDAY - today.weekday())
    return friday, friday + timedelta(days=2)


def today_in_warsaw() -> date:
    return datetime.now(_DIGEST_TIMEZONE).date()


@dataclass(frozen=True)
class DigestItem:
    title: str
    url: str
    start: date
    end: date | None
    # Calendar entries are the first to go when the Digest is too long
    from_calendar: bool = False


def _events_post_url(channel_id: str, message_id: int) -> str | None:
    """Link to a Post in the Events channel; None when the ID gives no link."""
    if channel_id.startswith("-100"):
        return f"https://t.me/c/{channel_id.removeprefix('-100')}/{message_id}"
    if channel_id.startswith("@"):
        return f"https://t.me/{channel_id.removeprefix('@')}/{message_id}"
    return None


def _digest_item(event: DatedEvent, events_channel_id: str) -> DigestItem:
    url = None
    if event.tg_events_message_id is not None:
        url = _events_post_url(events_channel_id, event.tg_events_message_id)
    return DigestItem(
        title=event.ua_title or event.title_pl,
        url=url or event.url,
        start=event.event_start,
        end=event.event_end,
    )


def _calendar_item(entry: CalendarEntry, ua_title: str) -> DigestItem:
    return DigestItem(
        title=ua_title, url=entry.url, start=entry.start, end=entry.end, from_calendar=True
    )


def _first_day(item: DigestItem, friday: date) -> date:
    """The weekend day an item is listed under: the first one it covers."""
    return max(item.start, friday)


def _render(friday: date, items: list[DigestItem], dropped: int, banner: str) -> str:
    sunday = friday + timedelta(days=2)
    lines = [f"{banner}<b>🗓 Вихідні в Жешові, {format_event_date(friday, sunday)}</b>"]
    for offset, day_name in enumerate(_UA_WEEKEND_DAYS):
        day = friday + timedelta(days=offset)
        day_items = [i for i in items if _first_day(i, friday) == day]
        if not day_items:
            continue
        lines.append("")
        lines.append(f"<b>{day_name}, {format_event_date(day, None)}</b>")
        for item in day_items:
            link = f'<a href="{_html_escape(item.url)}">{_html_escape(item.title)}</a>'
            if item.end is not None:
                link += f" ({format_event_date(item.start, item.end)})"
            lines.append(f"• {link}")
    if dropped:
        lines.append("")
        lines.append(f"…і ще {dropped}")
    return "\n".join(lines)


def build_digest(friday: date, items: list[DigestItem], banner: str = "") -> tuple[str, int]:
    """The Digest's HTML and how many items it lists.

    A header, then each weekend day that has items. Each item is listed once, under
    the first weekend day it covers; a multi-day event shows its range. The text is
    always one message: past Telegram's limit, Calendar entries are dropped first, then
    news events from the end, and a last line counts them.
    """
    ordered = sorted(items, key=lambda i: _first_day(i, friday))  # stable: keeps input order
    # Positions in the order they are dropped: Calendar entries, then news, each from the end
    drop_order = sorted(range(len(ordered)), key=lambda n: (not ordered[n].from_calendar, -n))
    for dropped in range(len(ordered)):
        gone = set(drop_order[:dropped])
        shown = [item for n, item in enumerate(ordered) if n not in gone]
        text = _render(friday, shown, dropped, banner)
        if len(text) <= _MAX_MESSAGE_LEN:
            return text, len(shown)
    return _render(friday, [], len(ordered), banner), 0


async def run_digest(
    settings: Settings,
    flow_config: FlowConfig,
    storage: StorageProtocol,
    *,
    today: date | None = None,
    dry_run: bool = False,
    staging: bool = False,
) -> int:
    """Build and post the coming weekend's Digest; returns the CLI exit code."""
    friday, sunday = digest_weekend(today or today_in_warsaw())
    try:
        return await _run_digest(settings, flow_config, storage, friday, sunday, dry_run, staging)
    except Exception as exc:
        # An unexpected failure, storage above all; a Publishing record left behind
        # still blocks a second Digest
        logger.exception("digest_fatal_error", weekend=friday.isoformat(), error=str(exc))
        if not dry_run and settings.is_production:
            await _alert(
                settings,
                f"Дайджест на вихідні {format_event_date(friday, sunday)} не зібрано: "
                f"{type(exc).__name__}: {exc}",
            )
        return 1


async def _run_digest(
    settings: Settings,
    flow_config: FlowConfig,
    storage: StorageProtocol,
    friday: date,
    sunday: date,
    dry_run: bool,
    staging: bool,
) -> int:
    log = logger.bind(weekend=friday.isoformat(), dry_run=dry_run, staging=staging)

    events_channel_id = settings.events_telegram_chat_id(staging=staging)
    if events_channel_id is None:
        log.warning("digest_skipped", reason="events_channel_not_configured")
        return 0

    existing = await storage.get_digest(friday)
    if existing is not None:
        log.info("digest_already_exists", status=existing.status.value)
        return 0

    events = await storage.published_events_between(friday, sunday)
    items = [_digest_item(e, events_channel_id) for e in events]
    items += await _calendar_items(settings, flow_config, friday, sunday, events)
    weekend = format_event_date(friday, sunday)
    if len(items) < MIN_DIGEST_ITEMS:
        log.info("digest_skipped", reason="too_few_items", items=len(items))
        await _alert(
            settings,
            f"Дайджест на вихідні {weekend} не опубліковано: подій з датою і записів "
            f"календаря лише {len(items)}, а потрібно щонайменше {MIN_DIGEST_ITEMS}.",
        )
        return 0
    if dry_run:
        return await _send_dry_run(settings, friday, items, staging=staging)

    text, shown = build_digest(friday, items, banner=_STAGING_POST_BANNER if staging else "")
    publisher = TelegramPublisher(
        bot_token=settings.telegram_bot_token, channel_id=events_channel_id
    )
    if not await storage.start_digest(friday, item_count=shown):
        log.info("digest_already_exists")
        return 0
    try:
        message_id = await publisher.send_digest(text)
    except Exception as exc:
        if _proves_not_delivered(exc):
            log.error("digest_not_delivered", error_type=type(exc).__name__, error=str(exc))
            await storage.delete_digest(friday)
            await _alert(
                settings,
                f"Дайджест на вихідні {weekend} не опубліковано ({type(exc).__name__}: "
                f"{exc}). Запис видалено, тож його можна запустити ще раз вручну.",
            )
            return 1
        log.error("digest_delivery_unknown", error_type=type(exc).__name__, error=str(exc))
        await _alert(
            settings,
            f"Дайджест на вихідні {weekend}: невідомо, чи він дійшов "
            f"({type(exc).__name__}: {exc}). Перевір канал подій: повторно його не "
            "надішлемо.",
        )
        return 1

    try:
        await storage.mark_digest_published(friday, message_id)
    except Exception as exc:
        # The record stays Publishing, which still blocks a second Digest
        log.error(
            "digest_record_not_updated",
            message_id=message_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        await _alert(
            settings,
            f"Дайджест на вихідні {weekend} опубліковано, але запис не оновлено "
            f"({type(exc).__name__}: {exc}). Повторно його не надішлемо.",
        )
        return 1
    log.info("digest_published", message_id=message_id, items=shown, dropped=len(items) - shown)
    return 0


async def _calendar_items(
    settings: Settings,
    flow_config: FlowConfig,
    friday: date,
    sunday: date,
    events: list[DatedEvent],
) -> list[DigestItem]:
    """The weekend's Calendar entries the AI kept: interesting, and not the same event as
    a news event. None, with an admin alert, when the Calendar or the AI request fails.
    """
    log = logger.bind(weekend=friday.isoformat())
    news_only = "Дайджест вийде лише з подіями з новин."
    reading = await read_weekend_calendar(
        flow_config.digest, friday, timeout=settings.scraper_timeout
    )
    if reading.failure is not None:
        await _alert(settings, f"{reading.failure} {news_only}")
        return []
    if not reading.entries:
        return []

    try:
        ai = GeminiAIFilter(
            api_key=settings.gemini_api_key, model=flow_config.pipeline.gemini_model
        )
        verdicts = {v.id: v for v in await ai.judge_calendar(reading.entries, events)}
    except Exception as exc:
        log.error("digest_calendar_not_judged", error_type=type(exc).__name__, error=str(exc))
        weekend = format_event_date(friday, sunday)
        await _alert(
            settings,
            f"Записи календаря подій на вихідні {weekend} не оцінено "
            f"({type(exc).__name__}: {exc}). {news_only}",
        )
        return []

    unjudged = [e.id for e in reading.entries if e.id not in verdicts]
    if unjudged:
        # Without a Ukrainian title an entry cannot be listed
        log.warning("digest_calendar_entries_unjudged", entry_ids=unjudged)
    kept = [
        _calendar_item(entry, verdict.ua_title)
        for entry in reading.entries
        if (verdict := verdicts.get(entry.id)) is not None
        and verdict.is_interesting
        and verdict.same_as is None
    ]
    log.info("digest_calendar_judged", entries=len(reading.entries), kept=len(kept))
    return kept


async def _send_dry_run(
    settings: Settings, friday: date, items: list[DigestItem], *, staging: bool
) -> int:
    """Send the Digest's text to the admin chat instead of the channel; writes nothing."""
    log = logger.bind(weekend=friday.isoformat(), dry_run=True, staging=staging)
    if not settings.telegram_admin_chat_id:
        log.warning("digest_dry_run_not_sent", reason="telegram_admin_chat_id_unset")
        return 0
    banner = _DRY_RUN_BANNER.format(mode=" STAGING" if staging else "")
    text, shown = build_digest(friday, items, banner=banner)
    admin = TelegramPublisher(
        bot_token=settings.telegram_bot_token, channel_id=settings.telegram_admin_chat_id
    )
    try:
        message_id = await admin.send_digest(text)
    except Exception as exc:
        log.error("digest_dry_run_send_failed", error_type=type(exc).__name__, error=str(exc))
        return 1
    log.info("digest_dry_run_sent", message_id=message_id, items=shown)
    return 0


async def _alert(settings: Settings, message: str) -> None:
    """Alert the admin chat; never falls back to a channel."""
    if not settings.telegram_admin_chat_id:
        logger.warning(
            "digest_alert_not_sent", reason="telegram_admin_chat_id_unset", alert=message
        )
        return
    admin = TelegramPublisher(
        bot_token=settings.telegram_bot_token,
        channel_id=settings.telegram_channel_id,
        admin_chat_id=settings.telegram_admin_chat_id,
    )
    await admin.send_alert(_html_escape(message))
