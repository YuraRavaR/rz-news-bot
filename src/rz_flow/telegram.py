"""Telegram Bot API publisher.

Uses httpx directly (no python-telegram-bot library) because:
  - The bot only sends messages — no receiving, no polling
  - Fewer dependencies = simpler setup, smaller Docker image
  - Great for learning how Bot API actually works

Handles:
  - HTML parse mode (bold titles, links)
  - Rate limit (HTTP 429) with Retry-After header respect
  - General retry with exponential backoff
"""

import asyncio
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rz_flow.models import AIDecision, Article, PublishResult

if TYPE_CHECKING:
    from rz_flow.pipeline import ArticleRunEntry, PipelineStats

logger = structlog.get_logger(__name__)

# Telegram Bot API base URL
_TG_API_BASE = "https://api.telegram.org/bot{token}"

# Maximum message length Telegram allows (HTML entities count toward it)
_MAX_MESSAGE_LEN = 4096

# Maximum articles shown in the run report before truncation
_MAX_REPORT_ARTICLES = 20

# Maximum rows in the "remaining queue" section (post cap / quota tail)
_MAX_REMAINING_IN_REPORT = 15

# Per-field cap for AI snippets in the admin HTML report (many articles × long text)
_MAX_ADMIN_AI_SNIPPET = 420


def _admin_snippet(text: str, max_len: int = _MAX_ADMIN_AI_SNIPPET) -> str:
    """Single-line-ish snippet for Telegram HTML (avoids huge admin messages)."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1] + "…"

# Template for a channel post — {hashtag} is empty string when category is "inne"
_POST_TEMPLATE = """\
<b>{title}</b>

{summary}

<a href="{url}">Деталі на {domain}</a>{hashtag}"""

# Maps CategoryTag values → Ukrainian hashtags shown at the end of a post.
# "inne" maps to empty string so no hashtag line is appended for uncategorised content.
_CATEGORY_HASHTAGS: dict[str, str] = {
    "koncert": "#концерт",
    "festyn": "#фестиваль",
    "sport": "#спорт",
    "komunikacja": "#транспорт",
    "inne": "",
}


# Prepended to every channel post when running with --staging (HTML, Telegram-safe)
_STAGING_POST_BANNER = (
    "<b>🧪 STAGING</b>\n"
    "<i>Чернетковий канал — це не продакшен-публікація.</i>\n\n"
)


def _build_message(article: Article, decision: AIDecision, *, staging: bool = False) -> str:
    """Render the Telegram HTML message from article + AI decision."""
    # HTML-escape any stray < > & in user content (titles from the site)
    title = _html_escape(decision.ua_title)
    summary = _html_escape(decision.ua_summary)
    url = article.url
    domain = urlparse(url).netloc  # e.g. "rzeszow24.info" or "rzeszow-news.pl"

    raw_hashtag = _CATEGORY_HASHTAGS.get(decision.category_tag.value, "")
    hashtag = f"\n\n{raw_hashtag}" if raw_hashtag else ""

    text = _POST_TEMPLATE.format(
        title=title,
        summary=summary,
        url=url,
        domain=domain,
        hashtag=hashtag,
    )

    if staging:
        text = _STAGING_POST_BANNER + text

    # Fit Telegram limit without breaking HTML (body can be long)
    if len(text) > _MAX_MESSAGE_LEN:
        text = _truncate_telegram_html(text)

    return text


def _html_escape(text: str) -> str:
    """Minimal HTML escaping for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate_telegram_html(text: str, max_len: int = _MAX_MESSAGE_LEN - 8) -> str:
    """Fit HTML into Telegram's length limit without splitting tags mid-string.

    A naive slice after ``join()`` can break inside ``<blockquote>``, ``<a>``,
    etc., which yields ``400 Bad Request: can't parse entities: Unclosed end tag``.
    """
    if len(text) <= max_len:
        return text

    notice = "\n\n<i>… trimmed (Telegram length limit).</i>"

    def _build(cut: int) -> str:
        chunk = text[: max(0, min(cut, len(text)))]
        chunk = re.sub(r"<[^>\n]*$", "", chunk)

        def _missing_closes(open_re: str, close_lit: str) -> int:
            o = len(re.findall(open_re, chunk, flags=re.I))
            cl = len(re.findall(re.escape(close_lit), chunk, flags=re.I))
            return max(0, o - cl)

        tail = ""
        for _ in range(_missing_closes(r"<blockquote\b", "</blockquote>")):
            tail += "</blockquote>"
        for _ in range(_missing_closes(r"<b\b", "</b>")):
            tail += "</b>"
        for _ in range(_missing_closes(r"<i\b", "</i>")):
            tail += "</i>"
        for _ in range(_missing_closes(r"<a\s", "</a>")):
            tail += "</a>"
        return chunk + tail + notice

    lo, hi = 1, len(text)
    best = _build(1)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _build(mid)
        if len(candidate) <= max_len:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def format_run_report_clock(now_utc: datetime, display_timezone: str | None) -> str:
    """Wall clock for the admin run-report header (UTC or a configured IANA zone)."""
    if not display_timezone:
        return f'{now_utc.strftime("%d.%m %H:%M")} UTC'
    local = now_utc.astimezone(ZoneInfo(display_timezone))
    abbr = local.tzname()
    if not abbr:
        abbr = local.strftime("%z")
    return f"{local.strftime('%d.%m %H:%M')} {abbr}"


def _format_run_report_elapsed(seconds: int) -> str:
    """Human-readable duration for the admin run report (e.g. 2m 18s, 45s)."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {sec}s"


def _article_run_report_lines(entry: "ArticleRunEntry") -> list[str]:
    """HTML lines for one article row in the admin run report (Telegram HTML)."""
    if entry.report_icon:
        icon = entry.report_icon
    else:
        decision_icons = {"posted": "✅", "skipped": "⏭", "error": "❌"}
        icon = decision_icons.get(entry.decision.value, "❓")
        if entry.decision.value == "posted" and entry.posted_to_events:
            icon = "✅📅"
    score_str = f"{entry.score:.1f}" if entry.score is not None else "-"
    title_plain = entry.ua_title or entry.title_pl
    title_esc = _html_escape(title_plain)
    url = (entry.article_url or "").strip()
    if url:
        href = _html_escape(url)
        head = f"  {icon} {score_str} · <a href=\"{href}\">{title_esc}</a>"
    else:
        head = f"  {icon} {score_str} · {title_esc}"
    if entry.error_msg:
        err = _html_escape(entry.error_msg)
        head += f" (<i>{err}</i>)"
    out: list[str] = [head]
    detail_lines: list[str] = []
    if entry.decision.value == "posted":
        channel_label = "новини + події" if entry.posted_to_events else "новини"
        detail_lines.append(f"<b>Канал</b>: {channel_label}")
    if entry.ai_reason:
        detail_lines.append(f"<b>AI</b>: {_html_escape(_admin_snippet(entry.ai_reason))}")
    if entry.ai_ua_summary:
        detail_lines.append(
            f"<b>UA summary</b>: {_html_escape(_admin_snippet(entry.ai_ua_summary))}"
        )
    if detail_lines:
        inner = "Details\n" + "\n".join(detail_lines)
        out.append(f"<blockquote expandable>\n{inner}\n</blockquote>")
    return out


def _collect_run_report_segments(
    stats: "PipelineStats",
    dry_run: bool,
    report_display_timezone: str | None,
    staging: bool,
) -> list[str]:
    """Return report body as segments (join with ``\\n`` = full report; pack for Telegram separately)."""
    from rz_flow.pipeline import ArticleRunEntry

    now_line = format_run_report_clock(datetime.now(UTC), report_display_timezone)
    mode_parts: list[str] = []
    if dry_run:
        mode_parts.append("DRY RUN")
    if staging:
        mode_parts.append("STAGING")
    mode = f" [{' '.join(mode_parts)}]" if mode_parts else ""
    flags = ""
    if stats.quota_exhausted:
        flags += " ⚠️ QUOTA"
    if stats.post_cap_reached:
        flags += " 🔒 CAP"

    head_lines: list[str] = [f"<b>📊 Rz-Flow{mode}{flags}</b> - {now_line}"]

    elapsed_fmt = _format_run_report_elapsed(stats.elapsed_s)
    model_raw = (stats.report_gemini_model or "").strip()
    meta_inner: list[str] = [f"⏱\ufe0f {elapsed_fmt}"]
    if model_raw:
        model_esc = _html_escape(model_raw)
        min_esc = _html_escape(f"{stats.report_ai_min_score:.1f}")
        meta_inner.append(model_esc)
        meta_inner.append(f"min score {min_esc}")
    meta_inner.append(
        f"{stats.total_scraped} scraped -> {stats.new_articles} new -> "
        f"posted {stats.posted} · skipped {stats.skipped} · errors {stats.errors}"
    )
    rq_n = len(stats.remaining_queued)
    if rq_n:
        meta_inner.append(f"not evaluated this run: {rq_n}")
    head_lines.append("<blockquote>" + "\n".join(meta_inner) + "</blockquote>")

    all_source_names = sorted(set(stats.source_scraped) | set(stats.source_new))
    if all_source_names:
        head_lines.append("<b>Sources</b>")
        for name in all_source_names:
            scraped = stats.source_scraped.get(name, 0)
            new = stats.source_new.get(name, 0)
            tail = f": {new} new of {scraped}"
            url = stats.source_urls.get(name, "").strip()
            if url:
                safe_href = _html_escape(url)
                safe_label = _html_escape(name)
                head_lines.append(f'  <a href="{safe_href}">{safe_label}</a>{tail}')
            else:
                head_lines.append(f"  {_html_escape(name)}{tail}")

    segments: list[str] = ["\n".join(head_lines)]

    log = stats.article_log
    if not log:
        segments.append(f"<i>No new articles</i> (scraped={stats.total_scraped})")
    else:
        shown = log[:_MAX_REPORT_ARTICLES]
        by_source: dict[str, list[ArticleRunEntry]] = defaultdict(list)
        for entry in shown:
            src_key = entry.source_name.strip() if entry.source_name.strip() else "-"
            by_source[src_key].append(entry)

        ordered_sources: list[str] = []
        for n in all_source_names:
            if n in by_source and n not in ordered_sources:
                ordered_sources.append(n)
        for n in sorted(by_source.keys()):
            if n not in ordered_sources:
                ordered_sources.append(n)

        segments.append("<b>Articles</b>")
        for src in ordered_sources:
            entries = by_source.get(src, [])
            if not entries:
                continue
            # One Telegram segment per article so packing never splits mid-<blockquote>.
            for entry in entries:
                body = "\n".join(_article_run_report_lines(entry))
                segments.append(f"<b>{_html_escape(src)}</b>\n{body}")
        if len(log) > _MAX_REPORT_ARTICLES:
            segments.append(f"<i>… {len(log) - _MAX_REPORT_ARTICLES} more articles</i>")

    if stats.remaining_queued:
        reason = (stats.remaining_stop_reason or "").strip()
        if reason == "post_cap":
            intro = (
                "Post limit reached for this run; below are articles Gemini has not "
                "evaluated in this run."
            )
        elif reason == "quota":
            intro = (
                "Gemini quota exhausted; below are the next articles in the list that did not "
                "start in this run (they will be retried on the next run)."
            )
        else:
            intro = "Not processed in this run:"
        segments.append("<b>Pending evaluation</b>")
        segments.append(f"<i>{_html_escape(intro)}</i>")
        shown_rq = stats.remaining_queued[:_MAX_REMAINING_IN_REPORT]
        for b in shown_rq:
            title_esc = _html_escape(b.title_pl)
            url = (b.url or "").strip()
            src_bit = f"{_html_escape(b.source_name.strip())} · " if b.source_name.strip() else ""
            if url:
                href = _html_escape(url)
                segments.append(f"  ⏳ {src_bit}<a href=\"{href}\">{title_esc}</a>")
            else:
                segments.append(f"  ⏳ {src_bit}{title_esc}")
        if len(stats.remaining_queued) > _MAX_REMAINING_IN_REPORT:
            more = len(stats.remaining_queued) - _MAX_REMAINING_IN_REPORT
            segments.append(f"<i>… {more} more</i>")

    summary_bits = [
        f"posted={stats.posted}",
        f"skipped={stats.skipped}",
        f"errors={stats.errors}",
    ]
    n_queued = len(stats.remaining_queued)
    if n_queued:
        summary_bits.append(f"not_evaluated_this_run={n_queued}")
    segments.append("<b>Summary</b> · " + " · ".join(summary_bits))
    return segments


def _pack_run_report_segments_for_telegram(
    segments: list[str],
    max_len: int = 4000,
) -> list[str]:
    """Split segments into 1+ HTML messages under Telegram's length limit (atomic segments)."""
    safe = [
        s if len(s) <= max_len else _truncate_telegram_html(s, max_len=max_len) for s in segments
    ]
    chunks: list[str] = []
    buf: list[str] = []
    bl = 0
    part = 1

    def _emit_buf() -> None:
        nonlocal buf, bl
        if not buf:
            return
        raw = "\n".join(buf)
        if len(raw) > max_len:
            raw = _truncate_telegram_html(raw, max_len=max_len)
        chunks.append(raw)
        buf = []
        bl = 0

    for seg in safe:
        sep = 1 if buf else 0
        if buf and bl + sep + len(seg) > max_len:
            _emit_buf()
            part += 1
            hdr = f"<b>Rz-Flow</b> <i>(Part {part})</i>\n"
            room = max(120, max_len - len(hdr))
            body = seg if len(seg) <= room else _truncate_telegram_html(seg, max_len=room)
            buf = [hdr + body]
            bl = len(buf[0])
        else:
            if buf:
                bl += sep
            buf.append(seg)
            bl += len(seg)
    _emit_buf()
    return chunks


def _build_run_report(
    stats: "PipelineStats",
    dry_run: bool,
    report_display_timezone: str | None = None,
    staging: bool = False,
) -> str:
    """Format the full admin run report (single string; tests / previews)."""
    segments = _collect_run_report_segments(stats, dry_run, report_display_timezone, staging)
    return "\n".join(segments)


class TelegramPublisher:
    """Publishes messages to a Telegram channel via Bot API."""

    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        admin_chat_id: str | None = None,
        report_display_timezone: str | None = None,
        mark_channel_posts_staging: bool = False,
    ) -> None:
        self._base_url = _TG_API_BASE.format(token=bot_token)
        self._channel_id = channel_id
        # Alerts go to admin_chat_id when set, otherwise fall back to the public channel.
        self._admin_chat_configured = bool((admin_chat_id or "").strip())
        self._alert_chat_id = admin_chat_id or channel_id
        self._report_display_timezone = report_display_timezone
        self._mark_channel_posts_staging = mark_channel_posts_staging

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def publish(self, article: Article, decision: AIDecision) -> PublishResult:
        """Send a message to the Telegram channel.

        Returns PublishResult with the message_id assigned by Telegram.
        Raises httpx.HTTPStatusError on persistent failures.
        """
        text = _build_message(
            article,
            decision,
            staging=self._mark_channel_posts_staging,
        )

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self._channel_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": False,
                },
            )

            # Handle rate limiting: respect Retry-After header
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "5"))
                logger.warning("rate_limited", retry_after=retry_after)
                await asyncio.sleep(retry_after)
                response.raise_for_status()  # trigger retry via tenacity

            if not response.is_success:
                # Log Telegram's actual error description before raising
                tg_error = response.json().get("description", response.text)
                logger.error(
                    "telegram_api_error",
                    status_code=response.status_code,
                    tg_description=tg_error,
                    chat_id=self._channel_id,
                )
                response.raise_for_status()

        data = response.json()
        message_id: int = data["result"]["message_id"]

        return PublishResult(
            article_id=article.id,
            message_id=message_id,
            chat_id=self._channel_id,
        )

    def _admin_send_target(self) -> str:
        """Log label for where alerts/reports go (no chat IDs)."""
        return "admin_chat" if self._admin_chat_configured else "channel"

    async def send_alert(self, message: str) -> None:
        """Send a plain-text alert to the admin chat (or channel if no admin chat configured)."""
        target = self._admin_send_target()
        try:
            logger.info("admin_alert_sending", target=target)
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._base_url}/sendMessage",
                    json={
                        "chat_id": self._alert_chat_id,
                        "text": f"⚠️ Rz-Flow alert:\n{message}",
                        "parse_mode": "HTML",
                    },
                )
            try:
                data: dict = response.json()
            except ValueError:
                data = {}
            if not response.is_success:
                desc = data.get("description", response.text) if isinstance(data, dict) else response.text
                logger.error(
                    "alert_send_failed",
                    target=target,
                    http_status=response.status_code,
                    tg_description=str(desc),
                )
                return
            if not isinstance(data, dict) or not data.get("ok"):
                logger.error(
                    "alert_send_failed",
                    target=target,
                    tg_description=str(data.get("description", "")) if isinstance(data, dict) else "",
                )
                return
            msg_id = data.get("result", {}).get("message_id") if isinstance(data.get("result"), dict) else None
            logger.info("admin_alert_sent", target=target, message_id=msg_id)
        except Exception as exc:
            # Alert failures should never crash the pipeline
            logger.error("alert_send_failed", target=target, error=str(exc))

    async def send_run_report(
        self,
        stats: "PipelineStats",
        dry_run: bool = False,
        staging: bool = False,
    ) -> None:
        """Send a structured run summary to the admin chat.

        Sent after every pipeline run (success, quota, or partial failure).
        Silently swallows errors so a reporting failure never affects the main run.
        """
        target = self._admin_send_target()
        try:
            segments = _collect_run_report_segments(
                stats,
                dry_run,
                self._report_display_timezone,
                staging=staging,
            )
            chunks = _pack_run_report_segments_for_telegram(segments)
            logger.info(
                "run_report_sending",
                dry_run=dry_run,
                staging=staging,
                target=target,
                report_chunks=len(chunks),
            )
            last_msg_id: int | None = None
            async with httpx.AsyncClient(timeout=10.0) as client:
                for i, text in enumerate(chunks):
                    if i:
                        await asyncio.sleep(0.35)
                    response = await client.post(
                        f"{self._base_url}/sendMessage",
                        json={
                            "chat_id": self._alert_chat_id,
                            "text": text,
                            "parse_mode": "HTML",
                            "disable_web_page_preview": True,
                        },
                    )
                    try:
                        data = response.json()
                    except ValueError:
                        data = {}
                    if not response.is_success:
                        desc = (
                            data.get("description", response.text) if isinstance(data, dict) else response.text
                        )
                        logger.error(
                            "run_report_send_failed",
                            target=target,
                            dry_run=dry_run,
                            chunk_index=i,
                            http_status=response.status_code,
                            tg_description=str(desc),
                        )
                        return
                    if not isinstance(data, dict) or not data.get("ok"):
                        logger.error(
                            "run_report_send_failed",
                            target=target,
                            dry_run=dry_run,
                            chunk_index=i,
                            tg_description=str(data.get("description", "")) if isinstance(data, dict) else "",
                        )
                        return
                    last_msg_id = (
                        data.get("result", {}).get("message_id")
                        if isinstance(data.get("result"), dict)
                        else None
                    )
            logger.info(
                "run_report_sent",
                dry_run=dry_run,
                target=target,
                message_id=last_msg_id,
                report_chunks=len(chunks),
            )
        except Exception as exc:
            logger.error("run_report_send_failed", target=target, dry_run=dry_run, error=str(exc))
