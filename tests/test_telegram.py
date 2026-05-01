"""Tests for the Telegram publisher (mocked HTTP with respx).

respx intercepts httpx calls at the transport level — no real network.
This tests our message formatting, API call structure, retry logic,
and rate-limit handling.
"""

from unittest.mock import AsyncMock, patch

import httpx
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

    def test_link_label_uses_domain_from_article_url(self) -> None:
        """QW-1: link label should reflect the actual source domain, not a hardcoded string."""
        article = Article(
            id="rzn/some-slug",
            url="https://rzeszow-news.pl/some-slug/",
            category=Category.IMPREZY,
            title_pl="Test",
            summary_pl="",
        )
        decision = _make_decision()
        msg = _build_message(article, decision)
        assert "rzeszow-news.pl" in msg
        assert "rzeszow24.info" not in msg

    def test_link_label_rzeszow24_domain(self) -> None:
        """Articles from rzeszow24.info should show that domain in the link label."""
        article = _make_article()  # URL is rzeszow24.info
        decision = _make_decision()
        msg = _build_message(article, decision)
        assert "rzeszow24.info" in msg

    def test_includes_hashtag_for_festival(self) -> None:
        """QW-8: festival category tag should add #фестиваль hashtag."""
        article = _make_article()
        decision = _make_decision()  # CategoryTag.FESTIVAL
        msg = _build_message(article, decision)
        assert "#фестиваль" in msg

    def test_includes_hashtag_for_concert(self) -> None:
        """QW-8: concert category tag should add #концерт hashtag."""
        article = _make_article()
        decision = AIDecision(
            is_interesting=True,
            score=8.0,
            category_tag=CategoryTag.CONCERT,
            ua_title="Концерт",
            ua_summary="Опис.",
            reason="Concert",
        )
        msg = _build_message(article, decision)
        assert "#концерт" in msg

    def test_no_hashtag_for_inne(self) -> None:
        """QW-8: 'inne' (other) category should produce no hashtag line."""
        article = _make_article()
        decision = AIDecision(
            is_interesting=True,
            score=7.5,
            category_tag=CategoryTag.OTHER,
            ua_title="Новина",
            ua_summary="Опис.",
            reason="General news",
        )
        msg = _build_message(article, decision)
        assert "#" not in msg

    def test_message_truncated_when_over_4096_chars(self) -> None:
        """Very long summaries must be truncated to fit Telegram's 4096-char limit."""
        article = _make_article()
        decision = _make_decision(ua_title="Т" * 200)
        # Override summary to push message well over the limit
        long_decision = AIDecision(
            is_interesting=True,
            score=8.0,
            category_tag=CategoryTag.FESTIVAL,
            ua_title="Т" * 200,
            ua_summary="С" * 4000,
            reason="test",
        )
        msg = _build_message(article, long_decision)
        assert len(msg) <= 4096
        assert msg.endswith("…")


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

    @patch("rz_flow.telegram.asyncio.sleep", new_callable=AsyncMock)
    @respx.mock
    async def test_raises_on_persistent_http_error(self, _mock_sleep: AsyncMock) -> None:
        respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(500, json={"ok": False, "description": "Internal Server Error"})
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        with pytest.raises(httpx.HTTPStatusError):
            await publisher.publish(_make_article(), _make_decision())

    @patch("rz_flow.telegram.asyncio.sleep", new_callable=AsyncMock)
    @respx.mock
    async def test_publish_respects_retry_after_header_on_429(
        self, mock_sleep: AsyncMock
    ) -> None:
        """429 response: asyncio.sleep is called with the Retry-After value, then retried."""
        call_count = 0

        def side_effect(request: httpx.Request) -> Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Response(429, headers={"Retry-After": "3"})
            return Response(200, json={"ok": True, "result": {"message_id": 77}})

        respx.post(_tg_url("sendMessage")).mock(side_effect=side_effect)

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        result = await publisher.publish(_make_article(), _make_decision())

        assert result.message_id == 77
        mock_sleep.assert_any_call(3)

    @patch("rz_flow.telegram.asyncio.sleep", new_callable=AsyncMock)
    @respx.mock
    async def test_publish_logs_and_raises_on_json_error_response(
        self, _mock_sleep: AsyncMock
    ) -> None:
        """Non-success response with JSON body: error is logged and HTTPStatusError raised."""
        respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(
                400,
                json={"ok": False, "description": "Bad Request: chat not found"},
            )
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        with pytest.raises(httpx.HTTPStatusError):
            await publisher.publish(_make_article(), _make_decision())

    @respx.mock
    async def test_send_alert_swallows_network_exception(self) -> None:
        """A network-level error in send_alert must never propagate."""
        respx.post(_tg_url("sendMessage")).mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100123")
        # Must not raise
        await publisher.send_alert("Test alert")

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

    @respx.mock
    async def test_send_alert_uses_admin_chat_id_when_set(self) -> None:
        """QW-6: alerts go to admin_chat_id, not the public channel."""
        import json

        route = respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(200, json={"ok": True, "result": {"message_id": 1}})
        )

        publisher = TelegramPublisher(
            bot_token="fake-token",
            channel_id="-100channel",
            admin_chat_id="99999admin",
        )
        await publisher.send_alert("Crash!")

        data = json.loads(route.calls[0].request.content)
        assert data["chat_id"] == "99999admin"

    @respx.mock
    async def test_send_alert_falls_back_to_channel_when_no_admin_chat(self) -> None:
        """QW-6: when admin_chat_id is unset, alert falls back to the public channel."""
        import json

        route = respx.post(_tg_url("sendMessage")).mock(
            return_value=Response(200, json={"ok": True, "result": {"message_id": 1}})
        )

        publisher = TelegramPublisher(bot_token="fake-token", channel_id="-100channel")
        await publisher.send_alert("Crash!")

        data = json.loads(route.calls[0].request.content)
        assert data["chat_id"] == "-100channel"
