"""Pipeline tests across consecutive Runs, with the real Telegram publisher.

These drive ``Pipeline.run()`` from the outside and assert what a channel reader or
the admin would notice: which Posts reached which channel, what the next Run does,
and what the run report shows. Article fetching is patched and the AI filter is
mocked; the Bot API is mocked with respx, so the publisher's own failure handling
runs end to end.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig, PipelineConfig, SourceConfig
from rz_flow.models import AIDecision, Article, Category, CategoryTag, Decision, StopSignal
from rz_flow.pipeline import Pipeline, PipelineStats
from rz_flow.storage import InMemoryStorage
from rz_flow.telegram import _build_run_report

_TOKEN = "fake:token"
_MAIN = "-100111"
_EVENTS = "-100333"
_SEND_URL = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        telegram_bot_token=_TOKEN,
        telegram_channel_id=_MAIN,
        telegram_admin_chat_id=None,
        telegram_events_channel_id=_EVENTS,
        telegram_staging_channel_id=None,
        telegram_staging_events_channel_id=None,
        gemini_api_key="fake-key",
        turso_database_url="libsql://fake.turso.io",
        turso_auth_token="fake-token",
        ai_min_score=7.0,
    )


def _flow_config() -> FlowConfig:
    return FlowConfig(
        sources=[
            SourceConfig(
                scraper="NajnowszeScraper",
                base_url="https://rzeszow24.info/najnowsze",
                max_articles=5,
            )
        ],
        pipeline=PipelineConfig(
            max_posts_per_run=10,
            inter_ai_delay_seconds=0.0,
            inter_post_delay_seconds=0.0,
        ),
    )


def _article(article_id: str) -> Article:
    return Article(
        id=article_id,
        url=f"https://rzeszow24.info/imprezy/{article_id}",
        category=Category.IMPREZY,
        title_pl="Koncert w Rzeszowie",
        summary_pl="Opis koncertu.",
        source_name="rzeszow24/najnowsze",
    )


def _event_decision(is_event: bool = True) -> AIDecision:
    return AIDecision(
        is_interesting=True,
        is_event=is_event,
        score=8.0,
        category_tag=CategoryTag.CONCERT,
        ua_title="Концерт у Жешові",
        ua_summary="Великий концерт у центрі міста.",
        reason="Public event",
    )


@dataclass
class FakeBotApi:
    """respx-backed Bot API: records Posts per channel and fails channels on demand."""

    posts: list[str] = field(default_factory=list)  # chat_id of every sendMessage, in order
    # chat_id → HTTP status to answer with, or a transport exception to raise
    failing: dict[str, int | Exception] = field(default_factory=dict)
    _next_id: int = 1

    def handle(self, request: httpx.Request) -> httpx.Response:
        chat_id = json.loads(request.content)["chat_id"]
        self.posts.append(chat_id)
        failure = self.failing.get(chat_id)
        if isinstance(failure, Exception):
            raise failure
        if failure is not None:
            return httpx.Response(failure, json={"ok": False, "description": "failure"})
        self._next_id += 1
        return httpx.Response(200, json={"ok": True, "result": {"message_id": self._next_id}})


@pytest.fixture
def bot_api() -> Iterator[FakeBotApi]:
    api = FakeBotApi()
    with respx.mock(assert_all_called=False) as router:
        router.post(_SEND_URL).mock(side_effect=api.handle)
        yield api


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


async def _run(
    storage: InMemoryStorage,
    articles: list[Article],
    decision: AIDecision,
    *,
    dry_run: bool = False,
) -> PipelineStats:
    """One Run over ``articles`` on shared storage, every Article judged ``decision``."""
    with (
        patch("rz_flow.pipeline.fetch_articles", new_callable=AsyncMock) as fetch,
        patch("rz_flow.pipeline.GeminiAIFilter") as ai_cls,
    ):
        fetch.return_value = (articles, {"rzeszow24/najnowsze": len(articles)})
        ai = MagicMock()
        ai.evaluate = AsyncMock(return_value=decision)
        ai_cls.return_value = ai
        pipeline = Pipeline(settings=_settings(), storage=storage, flow_config=_flow_config())
        return await pipeline.run(dry_run=dry_run)


class TestMainChannelFirstEventsAfter:
    async def test_event_article_posts_to_main_channel_before_events_channel(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        stats = await _run(storage, [_article("EVT_1")], _event_decision())

        assert bot_api.posts == [_MAIN, _EVENTS]
        assert stats.posted == 1
        assert stats.article_log[0].posted_to_events is True

    async def test_main_channel_failure_never_calls_events_channel(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        bot_api.failing[_MAIN] = 400

        stats = await _run(storage, [_article("EVT_2")], _event_decision())

        assert _EVENTS not in bot_api.posts
        assert stats.posted == 0
        assert stats.errors == 1
        record = storage.get_record("EVT_2")
        assert record is not None
        assert record.decision == Decision.ERROR

    async def test_events_channel_failure_still_publishes_and_is_never_retried(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        bot_api.failing[_EVENTS] = 500

        run1 = await _run(storage, [_article("EVT_3")], _event_decision())

        # One main Post, one events attempt — no in-process retry of the events Post.
        assert bot_api.posts == [_MAIN, _EVENTS]
        assert run1.posted == 1
        assert run1.errors == 0
        record = storage.get_record("EVT_3")
        assert record is not None
        assert record.decision == Decision.POSTED
        entry = run1.article_log[0]
        assert entry.posted_to_events is False
        assert entry.events_error is not None
        report = _build_run_report(run1, dry_run=False)
        assert "Канал подій" in report

        bot_api.failing.clear()
        run2 = await _run(storage, [_article("EVT_3")], _event_decision())

        # The Article is Settled: the second Run posts it to neither channel.
        assert bot_api.posts == [_MAIN, _EVENTS]
        assert run2.new_articles == 0


class TestUncertainDeliveryIsNeverReposted:
    @pytest.mark.parametrize(
        "failure",
        [httpx.ReadTimeout("timed out"), 502],
        ids=["read_timeout", "5xx"],
    )
    async def test_uncertain_main_post_stays_publishing_and_stops_the_run(
        self, bot_api: FakeBotApi, storage: InMemoryStorage, failure: int | Exception
    ) -> None:
        bot_api.failing[_MAIN] = failure
        articles = [_article("UNSURE_1"), _article("UNSURE_2")]

        stats = await _run(storage, articles, _event_decision(is_event=False))

        # One send, no retry, and the second Article is never reached.
        assert bot_api.posts == [_MAIN]
        assert stats.stop_signal is StopSignal.PUBLISHING_BLOCKED
        assert stats.errors == 0
        # Oldest first: UNSURE_2 is the one tried, UNSURE_1 is never reached.
        tried = storage.get_record("UNSURE_2")
        assert tried is not None
        assert tried.decision == Decision.PUBLISHING
        assert storage.get_record("UNSURE_1") is None
        assert stats.article_log[0].decision == Decision.PUBLISHING
        assert [b.article_id for b in stats.remaining_queued] == ["UNSURE_1"]
        assert "PUBLISHING BLOCKED" in _build_run_report(stats, dry_run=False)

    @pytest.mark.parametrize("still_scraped", [True, False])
    async def test_next_run_makes_it_unconfirmed_alerts_once_and_posts_nothing(
        self, bot_api: FakeBotApi, storage: InMemoryStorage, still_scraped: bool
    ) -> None:
        article = _article("UNSURE_3")
        bot_api.failing[_MAIN] = httpx.ReadTimeout("timed out")
        await _run(storage, [article], _event_decision(is_event=False))
        bot_api.failing.clear()
        posts_after_run1 = list(bot_api.posts)

        run2 = await _run(
            storage, [article] if still_scraped else [], _event_decision(is_event=False)
        )

        assert bot_api.posts == posts_after_run1
        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.UNCONFIRMED
        assert len([a for a in run2.alerts if article.url in a]) == 1

        run3 = await _run(storage, [article], _event_decision(is_event=False))

        assert not [a for a in run3.alerts if article.url in a]
        assert bot_api.posts == posts_after_run1

    async def test_dry_run_leaves_publishing_untouched(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        article = _article("UNSURE_4")
        await storage.mark_publishing(article)

        stats = await _run(storage, [article], _event_decision(is_event=False), dry_run=True)

        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.PUBLISHING
        assert stats.alerts == []
        assert bot_api.posts == []

    async def test_dry_run_writes_no_publishing_record(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        stats = await _run(
            storage, [_article("DRY_1")], _event_decision(is_event=False), dry_run=True
        )

        assert stats.posted == 1
        assert storage.get_record("DRY_1") is None
        assert bot_api.posts == []
