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


def _build_run_report(stats: "PipelineStats", dry_run: bool) -> str:
    """Format a concise HTML run-report message for the admin chat."""
    now = datetime.now(UTC).strftime("%d.%m %H:%M")
    mode = " [DRY RUN]" if dry_run else ""
    flags = ""
    if stats.quota_exhausted:
        flags += " ⚠️ QUOTA"
    if stats.post_cap_reached:
        flags += " 🔒 CAP"

    lines: list[str] = [f"<b>📊 Rz-Flow{mode}{flags}</b> — {now} UTC\n"]

    # Sources section
    all_source_names = sorted(
        set(stats.source_scraped) | set(stats.source_new)
    )
    if all_source_names:
        lines.append("<b>Sources:</b>")
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
        lines.append("")

    # Articles section — grouped by source (same order as Sources when possible)
    log = stats.article_log
    if not log:
        lines.append(f"No new articles (scraped={stats.total_scraped})")
    else:
        from rz_flow.pipeline import ArticleRunEntry

        lines.append("<b>Articles:</b>")
        shown = log[:_MAX_REPORT_ARTICLES]
        by_source: dict[str, list[ArticleRunEntry]] = defaultdict(list)
        for entry in shown:
            src_key = entry.source_name.strip() if entry.source_name.strip() else "—"
            by_source[src_key].append(entry)

        ordered_sources: list[str] = []
        for n in all_source_names:
            if n in by_source and n not in ordered_sources:
                ordered_sources.append(n)
        for n in sorted(by_source.keys()):
            if n not in ordered_sources:
                ordered_sources.append(n)

        def _article_line(entry: ArticleRunEntry) -> str:
            if entry.report_icon:
                icon = entry.report_icon
            else:
                decision_icons = {"posted": "✅", "skipped": "⏭", "error": "❌"}
                icon = decision_icons.get(entry.decision.value, "❓")
            score_str = f"{entry.score:.1f}" if entry.score is not None else "—"
            title = _html_escape(entry.ua_title or entry.title_pl)
            if entry.error_msg:
                err = _html_escape(entry.error_msg)
                return f"  {icon} {score_str} — {title} (<i>{err}</i>)"
            return f"  {icon} {score_str} — {title}"

        for src in ordered_sources:
            entries = by_source.get(src, [])
            if not entries:
                continue
            lines.append(f"<b>{_html_escape(src)}</b>")
            for entry in entries:
                lines.append(_article_line(entry))
        if len(log) > _MAX_REPORT_ARTICLES:
            lines.append(f"<i>… {len(log) - _MAX_REPORT_ARTICLES} more articles</i>")
        lines.append("")

    # Summary line
    lines.append(
        f"<b>Summary:</b> posted={stats.posted}  skipped={stats.skipped}  errors={stats.errors}"
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
    ) -> None:
        self._base_url = _TG_API_BASE.format(token=bot_token)
        self._channel_id = channel_id
        # Alerts go to admin_chat_id when set, otherwise fall back to the public channel.
        self._alert_chat_id = admin_chat_id or channel_id

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

    async def send_alert(self, message: str) -> None:
        """Send a plain-text alert to the admin chat (or channel if no admin chat configured)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self._base_url}/sendMessage",
                    json={
                        "chat_id": self._alert_chat_id,
                        "text": f"⚠️ Rz-Flow alert:\n{message}",
                        "parse_mode": "HTML",
                    },
                )
        except Exception as exc:
            # Alert failures should never crash the pipeline
            logger.error("alert_send_failed", error=str(exc))

    async def send_run_report(self, stats: "PipelineStats", dry_run: bool = False) -> None:
        """Send a structured run summary to the admin chat.

        Sent after every pipeline run (success, quota, or partial failure).
        Silently swallows errors so a reporting failure never affects the main run.
        """
        try:
            text = _build_run_report(stats, dry_run)
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self._base_url}/sendMessage",
                    json={
                        "chat_id": self._alert_chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
        except Exception as exc:
            logger.error("run_report_send_failed", error=str(exc))
