"""The Weekly report: what storage says about the previous Monday–Sunday.

Sent to the admin chat by ``rz-flow --weekly-report``. It only reads storage: no
Gemini, no scraping, no writes. Each Article counts in the week of its last write
(``seen_at``, Europe/Warsaw days), so an Error retried next week moves there.
"""

from datetime import date, datetime, time, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

import structlog

from rz_flow.config import Settings
from rz_flow.digest import today_in_warsaw
from rz_flow.models import Decision, DigestRecord, DigestStatus, PeriodCounts
from rz_flow.sources import source_names_by_prefix
from rz_flow.storage import _JUDGED_OUTCOMES, StorageProtocol
from rz_flow.telegram import TelegramPublisher, _html_escape, format_event_date

logger = structlog.get_logger(__name__)

_REPORT_TIMEZONE = ZoneInfo("Europe/Warsaw")

# Gemini's free-tier requests per day for the configured model
GEMINI_DAILY_LIMIT = 500

# The day Event dates reached production (#59); Published events before it have none
EVENT_DATES_SINCE = date(2026, 10, 29)

_UA_WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "нд")

# Outcome columns of a Source line, in order
_SOURCE_COLUMNS = (
    ("опубліковано", Decision.POSTED),
    ("пропущено", Decision.SKIPPED),
    ("повторів", Decision.REPEAT),
    ("помилок", Decision.ERROR),
    ("непідтверджених", Decision.UNCONFIRMED),
)


def report_week(today: date) -> date:
    """Monday of the last full Monday–Sunday week before ``today``."""
    return today - timedelta(days=today.weekday() + 7)


def _start_of(day: date) -> datetime:
    return datetime.combine(day, time(), tzinfo=_REPORT_TIMEZONE)


def _published(counts: PeriodCounts) -> int:
    return sum(by_outcome.get(Decision.POSTED, 0) for by_outcome in counts.by_prefix.values())


def _judged(counts: PeriodCounts) -> int:
    """AI-judged Articles: about one Gemini request each."""
    return sum(
        by_outcome.get(outcome, 0)
        for by_outcome in counts.by_prefix.values()
        for outcome in _JUDGED_OUTCOMES
    )


def _source_line(name: str, by_outcome: dict[Decision, int]) -> str:
    bits = [f"нових {sum(by_outcome.values())}"]
    bits += [f"{label} {by_outcome.get(outcome, 0)}" for label, outcome in _SOURCE_COLUMNS]
    publishing = by_outcome.get(Decision.PUBLISHING, 0)
    if publishing:
        bits.append(f"публікується {publishing}")
    return f"<b>{_html_escape(name)}</b>: " + " · ".join(bits)


def _published_line(week: PeriodCounts, previous: PeriodCounts) -> str:
    published, before = _published(week), _published(previous)
    line = f"Опубліковано: {published}, минулого тижня {before}"
    if published * 2 < before:
        return f"⚠️ {line}: менше ніж половина"
    return line


def _events_lines(week: PeriodCounts, monday: date) -> list[str]:
    posted, undated = week.events_posted, week.events_posted_undated
    if not posted:
        return ["Події: опубліковано 0"]
    lines = [f"Події: опубліковано {posted}, без дати {undated} ({undated * 100 // posted}%)"]
    if undated and monday < EVENT_DATES_SINCE:
        lines.append(
            f"<i>Дати подій зберігаються з {format_event_date(EVENT_DATES_SINCE, None)}, "
            "тож старіші події рахуються без дати.</i>"
        )
    return lines


def _gemini_line(days: list[PeriodCounts], monday: date) -> str:
    judged = [_judged(day) for day in days]
    busiest = max(range(len(judged)), key=lambda i: judged[i])
    day = monday + timedelta(days=busiest)
    return (
        f"Gemini (оцінка): у найзавантаженіший день ({_UA_WEEKDAYS[busiest]} "
        f"{format_event_date(day, None)}) {judged[busiest]} з {GEMINI_DAILY_LIMIT} запитів. "
        "<i>Рахуються AI-оцінені статті; повторні спроби та evals не враховано.</i>"
    )


def _digest_line(record: DigestRecord | None, friday: date) -> str:
    weekend = format_event_date(friday, friday + timedelta(days=2))
    if record is None:
        return f"Дайджест {weekend}: не опубліковано, запису немає"
    if record.status == DigestStatus.PUBLISHED:
        return f"Дайджест {weekend}: опубліковано, подій {record.item_count}"
    return (
        f"⚠️ Дайджест {weekend}: доставку не підтверджено (подій {record.item_count}), "
        "перевір канал подій"
    )


def build_weekly_report(
    monday: date,
    week: PeriodCounts,
    previous: PeriodCounts,
    days: list[PeriodCounts],
    digest: DigestRecord | None,
    source_names: dict[str, str],
    banner: str = "",
) -> list[str]:
    """The report's HTML segments, packed into messages by the caller.

    ``days`` are the week's seven days from Monday; ``source_names`` maps the ID
    prefixes of the registered Sources to their names, which are always listed.
    """
    sunday = monday + timedelta(days=6)
    segments = [
        f"{banner}<b>📊 Тижневий звіт, {format_event_date(monday, sunday)}</b>",
        _published_line(week, previous),
        "",
        "<b>Джерела</b>",
    ]
    prefixes = list(source_names) + sorted(set(week.by_prefix) - set(source_names))
    segments += [
        _source_line(source_names.get(p, p or "без префікса"), week.by_prefix.get(p, {}))
        for p in prefixes
    ]
    segments += ["", *_events_lines(week, monday), _gemini_line(days, monday)]
    segments.append(_digest_line(digest, monday + timedelta(days=4)))
    return segments


def _banner(dry_run: bool, staging: bool) -> str:
    modes = [mode for mode, on in (("DRY RUN", dry_run), ("STAGING", staging)) if on]
    return f"<b>🧪 {' '.join(modes)}</b>\n\n" if modes else ""


async def run_weekly_report(
    settings: Settings,
    storage: StorageProtocol,
    *,
    today: date | None = None,
    dry_run: bool = False,
    staging: bool = False,
) -> int:
    """Send the previous week's report to the admin chat; returns the CLI exit code."""
    monday = report_week(today or today_in_warsaw())
    log = logger.bind(week=monday.isoformat(), dry_run=dry_run, staging=staging)
    if not settings.telegram_admin_chat_id:
        # Never falls back to a channel: the report is for the admin only
        log.warning("weekly_report_not_sent", reason="telegram_admin_chat_id_unset")
        return 0
    admin = TelegramPublisher(
        bot_token=settings.telegram_bot_token,
        channel_id=settings.telegram_channel_id,
        admin_chat_id=settings.telegram_admin_chat_id,
    )

    try:
        week_starts = [_start_of(monday + timedelta(days=i)) for i in range(8)]
        week = await storage.period_counts(week_starts[0], week_starts[7])
        previous = await storage.period_counts(
            _start_of(monday - timedelta(days=7)), week_starts[0]
        )
        days = [await storage.period_counts(a, b) for a, b in pairwise(week_starts)]
        digest = await storage.get_digest(monday + timedelta(days=4))
    except Exception as exc:
        log.exception("weekly_report_failed", error=str(exc))
        if not dry_run and settings.is_production:
            await admin.send_alert(
                _html_escape(f"Тижневий звіт не зібрано: {type(exc).__name__}: {exc}")
            )
        return 1

    segments = build_weekly_report(
        monday,
        week,
        previous,
        days,
        digest,
        source_names_by_prefix(),
        banner=_banner(dry_run, staging),
    )
    return 0 if await admin.send_weekly_report(segments) else 1
