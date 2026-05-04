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
    from rz_flow.pipeline import PipelineStats

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


def _build_message(article: Article, decision: AIDecision) -> str:
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

    # Truncate gracefully if somehow over Telegram limit
    if len(text) > _MAX_MESSAGE_LEN:
        text = text[: _MAX_MESSAGE_LEN - 3] + "…"

    return text


def _html_escape(text: str) -> str:
    """Minimal HTML escaping for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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


def _build_run_report(
    stats: "PipelineStats",
    dry_run: bool,
    report_display_timezone: str | None = None,
    staging: bool = False,
) -> str:
    """Format a concise HTML run-report message for the admin chat."""
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

    lines: list[str] = [f"<b>📊 Rz-Flow{mode}{flags}</b> - {now_line}"]

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
        meta_inner.append(f"queued (not started this run): {rq_n}")
    lines.append("<blockquote>" + "\n".join(meta_inner) + "</blockquote>")

    # Sources section
    all_source_names = sorted(
        set(stats.source_scraped) | set(stats.source_new)
    )
    if all_source_names:
        lines.append("<b>Sources</b>")
        for name in all_source_names:
            scraped = stats.source_scraped.get(name, 0)
            new = stats.source_new.get(name, 0)
            tail = f": {new} new of {scraped}"
            url = stats.source_urls.get(name, "").strip()
            if url:
                safe_href = _html_escape(url)
                safe_label = _html_escape(name)
                lines.append(f'  <a href="{safe_href}">{safe_label}</a>{tail}')
            else:
                lines.append(f"  {_html_escape(name)}{tail}")

    # Articles section — grouped by source (same order as Sources when possible)
    log = stats.article_log
    if not log:
        lines.append(f"<i>No new articles</i> (scraped={stats.total_scraped})")
    else:
        from rz_flow.pipeline import ArticleRunEntry

        lines.append("<b>Articles</b>")
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

        def _article_report_lines(entry: ArticleRunEntry) -> list[str]:
            if entry.report_icon:
                icon = entry.report_icon
            else:
                decision_icons = {"posted": "✅", "skipped": "⏭", "error": "❌"}
                icon = decision_icons.get(entry.decision.value, "❓")
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
            if entry.ai_reason:
                detail_lines.append(
                    f"<b>AI</b>: {_html_escape(_admin_snippet(entry.ai_reason))}"
                )
            if entry.ai_ua_summary:
                detail_lines.append(
                    f"<b>UA summary</b>: {_html_escape(_admin_snippet(entry.ai_ua_summary))}"
                )
            if detail_lines:
                # Collapsed preview shows only "Details"; full AI + UA after expand.
                inner = "Details\n" + "\n".join(detail_lines)
                out.append(f"<blockquote expandable>\n{inner}\n</blockquote>")
            return out

        for src in ordered_sources:
            entries = by_source.get(src, [])
            if not entries:
                continue
            lines.append(f"<b>{_html_escape(src)}</b>")
            for entry in entries:
                lines.extend(_article_report_lines(entry))
        if len(log) > _MAX_REPORT_ARTICLES:
            lines.append(f"<i>… {len(log) - _MAX_REPORT_ARTICLES} more articles</i>")

    # New articles never started (stopped by post cap or Gemini quota)
    if stats.remaining_queued:
        reason = (stats.remaining_stop_reason or "").strip()
        if reason == "post_cap":
            intro = (
                "Ліміт постів за прогоном; нижче — новини з черги, які Gemini ще не оцінював у цьому прогоні."
            )
        elif reason == "quota":
            intro = (
                "Квота Gemini; нижче — наступні у списку новин, які не стартували в цьому прогоні "
                "(будуть знову в черзі на наступному)."
            )
        else:
            intro = "Не оброблені в цьому прогоні (черга):"
        lines.append("<b>У черзі</b>")
        lines.append(f"<i>{_html_escape(intro)}</i>")
        shown_rq = stats.remaining_queued[:_MAX_REMAINING_IN_REPORT]
        for b in shown_rq:
            title_esc = _html_escape(b.title_pl)
            url = (b.url or "").strip()
            src_bit = f"{_html_escape(b.source_name.strip())} · " if b.source_name.strip() else ""
            if url:
                href = _html_escape(url)
                lines.append(f"  ⏳ {src_bit}<a href=\"{href}\">{title_esc}</a>")
            else:
                lines.append(f"  ⏳ {src_bit}{title_esc}")
        if len(stats.remaining_queued) > _MAX_REMAINING_IN_REPORT:
            more = len(stats.remaining_queued) - _MAX_REMAINING_IN_REPORT
            lines.append(f"<i>… ще {more}</i>")

    # Summary (same numbers as funnel; kept for quick scan at end of message)
    lines.append(
        f"<b>Summary</b> · posted={stats.posted} · skipped={stats.skipped} · "
        f"errors={stats.errors}"
    )

    text = "\n".join(lines)
    if len(text) > _MAX_MESSAGE_LEN:
        text = text[: _MAX_MESSAGE_LEN - 3] + "…"
    return text


class TelegramPublisher:
    """Publishes messages to a Telegram channel via Bot API."""

    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        admin_chat_id: str | None = None,
        report_display_timezone: str | None = None,
    ) -> None:
        self._base_url = _TG_API_BASE.format(token=bot_token)
        self._channel_id = channel_id
        # Alerts go to admin_chat_id when set, otherwise fall back to the public channel.
        self._admin_chat_configured = bool((admin_chat_id or "").strip())
        self._alert_chat_id = admin_chat_id or channel_id
        self._report_display_timezone = report_display_timezone

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
        text = _build_message(article, decision)

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
            text = _build_run_report(stats, dry_run, self._report_display_timezone)
            logger.info("run_report_sending", dry_run=dry_run, target=target)
            async with httpx.AsyncClient(timeout=10.0) as client:
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
                data: dict = response.json()
            except ValueError:
                data = {}
            if not response.is_success:
                desc = data.get("description", response.text) if isinstance(data, dict) else response.text
                logger.error(
                    "run_report_send_failed",
                    target=target,
                    dry_run=dry_run,
                    http_status=response.status_code,
                    tg_description=str(desc),
                )
                return
            if not isinstance(data, dict) or not data.get("ok"):
                logger.error(
                    "run_report_send_failed",
                    target=target,
                    dry_run=dry_run,
                    tg_description=str(data.get("description", "")) if isinstance(data, dict) else "",
                )
                return
            msg_id = data.get("result", {}).get("message_id") if isinstance(data.get("result"), dict) else None
            logger.info("run_report_sent", dry_run=dry_run, target=target, message_id=msg_id)
        except Exception as exc:
            logger.error("run_report_send_failed", target=target, dry_run=dry_run, error=str(exc))
