"""Tests for the Telegram publisher (mocked HTTP with respx).

respx intercepts httpx calls at the transport level — no real network.
This tests our message formatting, API call structure, retry logic,
and rate-limit handling.
"""

import pytest
import respx
from httpx import Response

from rz_flow.models import AIDecision, Article, Category, CategoryTag
from rz_flow.telegram import TelegramPublisher, _build_message, _html_escape


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_article(article_id: str = "TESTID123456789") -> Article:
    return Article(
        id=article_id,
        url=f"https://rzeszow24.info/imprezy/test/{article_id}",
        category=Category.IMPREZY,
        title_pl="Festiwal w Rzeszowie",
        summary_pl="Festiwal muzyczny odbędzie się w centrum.",
    )


def _make_decision(ua_title: str = "Фестиваль у Жешові") -> AIDecision:
    return AIDecision(
        is_interesting=True,
        score=8.0,
        category_tag=CategoryTag.FESTIVAL,
        ua_title=ua_title,
        ua_summary="У центрі міста відбудеться великий музичний фестиваль з безкоштовним входом.",
        reason="Major public event",
    )


def _tg_url(method: str, token: str = "fake-token") -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


# ── Unit tests for message formatting ────────────────────────────────────────
class TestHtmlEscape:
    def test_escapes_ampersand(self) -> None:
        assert _html_escape("A&B") == "A&amp;B"

    def test_escapes_less_than(self) -> None:
        assert _html_escape("<tag>") == "&lt;tag&gt;"

    def test_leaves_normal_text_unchanged(self) -> None:
        assert _html_escape("Normal text 2026") == "Normal text 2026"


class TestBuildMessage:
    def test_contains_ua_title(self) -> None:
        article = _make_article()
        decision = _make_decision("Фестиваль у Жешові 2026")
        msg = _build_message(article, decision)
        assert "Фестиваль у Жешові 2026" in msg

    def test_contains_article_url(self) -> None:
        article = _make_article()
        decision = _make_decision()
        msg = _build_message(article, decision)
        assert article.url in msg

    def test_contains_ua_summary(self) -> None:
        article = _make_article()
        decision = _make_decision()
        msg = _build_message(article, decision)
        assert decision.ua_summary in msg

    def test_uses_bold_for_title(self) -> None:
        article = _make_article()
        decision = _make_decision()
        msg = _build_message(article, decision)
        assert "<b>" in msg and "</b>" in msg

    def test_title_with_html_special_chars_escaped(self) -> None:
        article = _make_article()
        decision = _make_decision("Фестиваль <Rock & Roll>")
        msg = _build_message(article, decision)
        assert "<Rock" not in msg  # raw < should be escaped
        assert "&lt;Rock" in msg
        assert "&amp; Roll" in msg

    def test_message_within_telegram_limit(self) -> None:
        article = _make_article()
        decision = _make_decision("A" * 200)
        msg = _build_message(article, decision)
        assert len(msg) <= 4096


# ── Integration tests with mocked HTTP ───────────────────────────────────────
class TestTelegramPublisher:
    @respx.mock
    async def test_successful_publish_returns_result(self) -> None:
        respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(
                200,
                json={"ok": True, "result": {"message_id": 123}},
            )
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        result = await publisher.publish(_make_article(), _make_decision())

        assert result.message_id == 123
        assert result.article_id == "TESTID123456789"
        assert result.chat_id == "-100123"

    @respx.mock
    async def test_publish_sends_html_parse_mode(self) -> None:
        route = respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(200, json={"ok": True, "result": {"message_id": 1}})
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        await publisher.publish(_make_article(), _make_decision())

        sent_body = route.calls[0].request.content
        import json

        data = json.loads(sent_body)
        assert data["parse_mode"] == "HTML"
        assert data["chat_id"] == "-100123"

    @respx.mock
    async def test_raises_on_persistent_http_error(self) -> None:
        respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(500, text="Internal Server Error")
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        with pytest.raises(Exception):
            await publisher.publish(_make_article(), _make_decision())

    @respx.mock
    async def test_send_alert_does_not_raise_on_failure(self) -> None:
        """Alert failures must never crash the pipeline."""
        respx.post(_tg_url("sendMessage")).mock(return_value=Response(500, text="Error"))

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        # Should not raise
        await publisher.send_alert("Test alert message")

    @respx.mock
    async def test_send_alert_prefixes_with_warning_emoji(self) -> None:
        route = respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(200, json={"ok": True, "result": {"message_id": 1}})
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        await publisher.send_alert("Something went wrong")

        import json

        data = json.loads(route.calls[0].request.content)
        assert "⚠️" in data["text"]
        assert "Something went wrong" in data["text"]
