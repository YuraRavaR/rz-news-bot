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

from rz_flow.models import AIDecision, Article, Category, CategoryTag, Decision
from rz_flow.telegram import (
    TelegramPublisher,
    _build_message,
    _html_escape,
    format_run_report_clock,
)


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


class TestFormatRunReportClock:
    def test_utc_when_timezone_unset(self) -> None:
        from datetime import UTC, datetime

        now = datetime(2026, 5, 3, 11, 5, tzinfo=UTC)
        assert format_run_report_clock(now, None) == "03.05 11:05 UTC"

    def test_warsaw_summer_time(self) -> None:
        from datetime import UTC, datetime

        now = datetime(2026, 5, 3, 11, 5, tzinfo=UTC)
        assert format_run_report_clock(now, "Europe/Warsaw") == "03.05 13:05 CEST"


class TestBuildRunReport:
    def test_source_lines_use_clickable_links_when_url_known(self) -> None:
        """Admin report: each source name links to its configured base_url."""
        from rz_flow.pipeline import ArticleRunEntry, PipelineStats
        from rz_flow.telegram import _build_run_report

        stats = PipelineStats(
            dry_run=True,
            source_scraped={"rzeszow-news.pl": 15, "rzeszow24/najnowsze": 17},
            source_new={"rzeszow-news.pl": 1, "rzeszow24/najnowsze": 0},
            source_urls={
                "rzeszow-news.pl": "https://rzeszow-news.pl",
                "rzeszow24/najnowsze": "https://rzeszow24.info/najnowsze",
            },
            article_log=[
                ArticleRunEntry(
                    article_id="rzn/x",
                    title_pl="T",
                    ua_title="Ukr",
                    score=8.0,
                    decision=Decision.POSTED,
                    source_name="rzeszow-news.pl",
                    article_url="https://rzeszow-news.pl/wiadomosci/slug/",
                    ai_reason="Local relevance high",
                    ai_ua_summary="Короткий опис для каналу.",
                )
            ],
            posted=1,
        )
        text = _build_run_report(stats, dry_run=True)
        assert 'href="https://rzeszow-news.pl"' in text
        assert " UTC\n" in text
        assert 'href="https://rzeszow24.info/najnowsze"' in text
        assert "<b>rzeszow-news.pl</b>" in text
        assert 'href="https://rzeszow-news.pl/wiadomosci/slug/"' in text
        assert "Local relevance high" in text
        assert "Короткий опис для каналу." in text
        assert "<blockquote expandable>" in text

    def test_run_report_header_uses_display_timezone_when_set(self) -> None:
        from rz_flow.pipeline import PipelineStats
        from rz_flow.telegram import _build_run_report

        text = _build_run_report(PipelineStats(), dry_run=False, report_display_timezone="Europe/Warsaw")
        first = text.split("\n", 1)[0]
        assert " UTC" not in first
        assert "Rz-Flow" in first

    def test_run_report_uses_report_icon_when_set(self) -> None:
        from rz_flow.pipeline import ArticleRunEntry, PipelineStats
        from rz_flow.telegram import _build_run_report

        stats = PipelineStats(
            article_log=[
                ArticleRunEntry(
                    article_id="x",
                    title_pl="T",
                    ua_title=None,
                    score=None,
                    decision=Decision.SKIPPED,
                    error_msg="quota message",
                    report_icon="⏸",
                    article_url="https://example.com/news/1",
                )
            ],
        )
        text = _build_run_report(stats, dry_run=False)
        assert "⏸" in text
        assert "quota message" in text
        assert "<b>—</b>" in text  # grouped under unknown source when source_name empty
        assert 'href="https://example.com/news/1"' in text


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
