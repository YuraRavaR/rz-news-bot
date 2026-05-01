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

logger = structlog.get_logger(__name__)

# Telegram Bot API base URL
_TG_API_BASE = "https://api.telegram.org/bot{token}"

# Maximum message length Telegram allows (HTML entities count toward it)
_MAX_MESSAGE_LEN = 4096

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
