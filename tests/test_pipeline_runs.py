"""Pipeline tests across consecutive Runs, with the real Telegram publisher.

These drive ``Pipeline.run()`` from the outside and assert what a channel reader or
the admin would notice: which Posts reached which channel, what the next Run does,
and what the run report shows. Article fetching is patched and the AI filter is
mocked; the Bot API is mocked with respx, so the publisher's own failure handling
runs end to end.
"""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from rz_flow.ai import GeminiUnavailableError
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


_RZ24 = "rzeszow24/najnowsze"
_RZN = "rzeszow-news.pl"


def _flow_config(rz24_window: int = 5, rzn_window: int | None = None) -> FlowConfig:
    sources = [
        SourceConfig(
            scraper="NajnowszeScraper",
            base_url="https://rzeszow24.info/najnowsze",
            max_articles=rz24_window,
        )
    ]
    if rzn_window is not None:
        sources.append(
            SourceConfig(
                scraper="RzeszowNewsScraper",
                base_url="https://rzeszow-news.pl/",
                max_articles=rzn_window,
            )
        )
    return FlowConfig(
        sources=sources,
        pipeline=PipelineConfig(
            gemini_model="gemini-test-model",
            max_posts_per_run=10,
            inter_ai_delay_seconds=0.0,
            inter_post_delay_seconds=0.0,
        ),
    )


def _article(article_id: str, title: str | None = None) -> Article:
    """A rzeszow24 Article; each ID gets its own title unless one is given."""
    return Article(
        id=article_id,
        url=f"https://rzeszow24.info/imprezy/{title_slug(title or article_id)}/{article_id}",
        category=Category.IMPREZY,
        title_pl=title or f"Koncert w Rzeszowie {article_id}",
        summary_pl="Opis koncertu.",
        source_name=_RZ24,
    )


def _rzn_article(slug: str, title: str | None = None) -> Article:
    """A rzeszow-news.pl Article, whose ID is its slug."""
    return Article(
        id=f"rzn/{slug}",
        url=f"https://rzeszow-news.pl/{slug}/",
        category=Category.WIADOMOSCI,
        title_pl=title or f"Wiadomość {slug}",
        summary_pl="Opis.",
        source_name=_RZN,
    )


def title_slug(title: str) -> str:
    return "-".join(title.lower().split())


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
    # chat_id → responses served first, one per request, before ``failing`` applies
    scripted: dict[str, list[httpx.Response]] = field(default_factory=dict)
    _next_id: int = 1

    def handle(self, request: httpx.Request) -> httpx.Response:
        chat_id = json.loads(request.content)["chat_id"]
        self.posts.append(chat_id)
        if self.scripted.get(chat_id):
            return self.scripted[chat_id].pop(0)
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
    decision: AIDecision | None = None,
    *,
    dry_run: bool = False,
    evaluate: AsyncMock | None = None,
    flow_config: FlowConfig | None = None,
) -> PipelineStats:
    """One Run over ``articles`` on shared storage.

    Every Article is judged ``decision`` unless an ``evaluate`` mock is given, which
    also lets a test count how often Gemini was asked.
    """
    with (
        patch("rz_flow.pipeline.fetch_articles", new_callable=AsyncMock) as fetch,
        patch("rz_flow.pipeline.GeminiAIFilter") as ai_cls,
    ):
        fetch.return_value = (articles, dict(Counter(a.source_name for a in articles)))
        ai = MagicMock()
        ai.evaluate = evaluate or AsyncMock(return_value=decision)
        ai_cls.return_value = ai
        pipeline = Pipeline(
            settings=_settings(), storage=storage, flow_config=flow_config or _flow_config()
        )
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


def _rate_limited(retry_after: int) -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "ok": False,
            "description": "Too Many Requests",
            "parameters": {"retry_after": retry_after},
        },
    )


class TestDefinitelyNotDeliveredIsDeferred:
    async def test_large_retry_after_defers_stops_and_next_run_posts(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        bot_api.scripted[_MAIN] = [_rate_limited(120)]
        articles = [_article("RL_1"), _article("RL_2")]
        with patch("rz_flow.telegram._sleep", new_callable=AsyncMock) as tg_wait:
            run1 = await _run(storage, articles, _event_decision(is_event=False))

        tg_wait.assert_not_awaited()
        assert bot_api.posts == [_MAIN]
        assert run1.stop_signal is StopSignal.TELEGRAM_RATE_LIMITED
        assert run1.deferred == 1
        assert run1.article_log[0].decision == Decision.DEFERRED
        assert storage.get_record("RL_1") is None
        assert storage.get_record("RL_2") is None
        assert [b.article_id for b in run1.remaining_queued] == ["RL_1"]
        assert run1.remaining_stop_reason == "telegram_rate_limited"

        run2 = await _run(storage, articles, _event_decision(is_event=False))

        assert run2.posted == 2
        assert bot_api.posts == [_MAIN, _MAIN, _MAIN]

    async def test_small_retry_after_waits_once_then_posts_once(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        bot_api.scripted[_MAIN] = [_rate_limited(3)]
        with patch("rz_flow.telegram._sleep", new_callable=AsyncMock) as tg_wait:
            stats = await _run(storage, [_article("RL_3")], _event_decision(is_event=False))

        tg_wait.assert_awaited_once_with(3.0)
        assert bot_api.posts == [_MAIN, _MAIN]
        assert stats.posted == 1
        record = storage.get_record("RL_3")
        assert record is not None
        assert record.decision == Decision.POSTED

    async def test_connect_error_defers_stops_and_next_run_posts(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        bot_api.failing[_MAIN] = httpx.ConnectError("Connection refused")
        articles = [_article("CONN_1"), _article("CONN_2")]

        run1 = await _run(storage, articles, _event_decision(is_event=False))

        assert run1.stop_signal is StopSignal.PUBLISHING_BLOCKED
        assert run1.deferred == 1
        assert run1.errors == 0
        assert storage.get_record("CONN_1") is None
        assert storage.get_record("CONN_2") is None
        assert [b.article_id for b in run1.remaining_queued] == ["CONN_1"]

        bot_api.failing.clear()
        posts_before = len(bot_api.posts)
        run2 = await _run(storage, articles, _event_decision(is_event=False))

        assert run2.posted == 2
        assert bot_api.posts[posts_before:] == [_MAIN, _MAIN]

    async def test_error_article_deferred_keeps_its_attempt_count(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        article = _article("CONN_ERR")
        await storage.save_decision(article, Decision.ERROR)
        bot_api.failing[_MAIN] = httpx.ConnectError("Connection refused")

        stats = await _run(storage, [article], _event_decision(is_event=False))

        assert stats.deferred == 1
        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.ERROR
        assert record.error_attempts == 1


class TestRejectedPostBecomesError:
    async def test_400_sends_once_and_records_error(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        bot_api.failing[_MAIN] = 400

        stats = await _run(storage, [_article("BAD_1")], _event_decision(is_event=False))

        assert bot_api.posts == [_MAIN]
        assert stats.errors == 1
        assert stats.stop_signal is StopSignal.CONTINUE
        record = storage.get_record("BAD_1")
        assert record is not None
        assert record.decision == Decision.ERROR
        assert record.error_attempts == 1
        assert "TelegramRejectedError" in (stats.article_log[0].error_msg or "")

    async def test_attempts_continue_from_before_the_run_until_settled(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        article = _article("BAD_2")
        await storage.save_decision(article, Decision.ERROR)
        bot_api.failing[_MAIN] = 400

        await _run(storage, [article], _event_decision(is_event=False))

        record = storage.get_record(article.id)
        assert record is not None
        assert record.error_attempts == 2
        assert await storage.filter_new_ids([article.id]) == [article.id]

        await _run(storage, [article], _event_decision(is_event=False))

        record = storage.get_record(article.id)
        assert record is not None
        assert record.error_attempts == 3
        # Default budget is 3 attempts: the Article is now Settled.
        assert await storage.filter_new_ids([article.id]) == []


class TestStorageFailuresNeverDuplicate:
    async def test_unwritable_publishing_record_sends_nothing_and_stops(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [_article("DB_1"), _article("DB_2")]
        with patch.object(
            storage, "mark_publishing", AsyncMock(side_effect=ConnectionError("turso down"))
        ):
            run1 = await _run(storage, articles, _event_decision(is_event=False))

        assert bot_api.posts == []
        assert run1.stop_signal is StopSignal.PUBLISHING_BLOCKED
        assert run1.deferred == 1
        assert run1.article_log[0].decision == Decision.DEFERRED
        assert [b.article_id for b in run1.remaining_queued] == ["DB_1"]

        run2 = await _run(storage, articles, _event_decision(is_event=False))

        assert run2.posted == 2

    async def test_lost_save_after_post_alerts_and_next_run_marks_unconfirmed(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        article = _article("DB_3")
        with (
            patch.object(
                storage, "save_decision", AsyncMock(side_effect=ConnectionError("turso down"))
            ),
            patch("rz_flow.pipeline._SAVE_RETRY_DELAY_S", 0.0),
        ):
            run1 = await _run(storage, [article], _event_decision(is_event=False))

        assert bot_api.posts == [_MAIN]
        assert run1.posted == 1
        alerts = [a for a in run1.alerts if article.url in a]
        assert len(alerts) == 1
        assert "Повторної публікації не буде" in alerts[0]

        run2 = await _run(storage, [article], _event_decision(is_event=False))

        assert bot_api.posts == [_MAIN]
        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.UNCONFIRMED
        assert len([a for a in run2.alerts if article.url in a]) == 1


class TestGeminiModelFromConfig:
    async def test_evaluation_and_run_report_use_the_configured_model(
        self, bot_api: FakeBotApi, storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
        with (
            patch("rz_flow.pipeline.fetch_articles", new_callable=AsyncMock) as fetch,
            patch("rz_flow.pipeline.GeminiAIFilter") as ai_cls,
        ):
            fetch.return_value = ([_article("MODEL_1")], {"rzeszow24/najnowsze": 1})
            ai_cls.return_value.evaluate = AsyncMock(return_value=_event_decision())
            pipeline = Pipeline(settings=_settings(), storage=storage, flow_config=_flow_config())
            stats = await pipeline.run()

        assert ai_cls.call_args.kwargs["model"] == "gemini-test-model"
        report = _build_run_report(stats, dry_run=False)
        assert "gemini-test-model" in report
        assert "gemini-from-env" not in report


def _source_line(report: str, source_name: str) -> str:
    """The run report's line for one Source in the Sources block."""
    lines = [line for line in report.splitlines() if f">{source_name}</a>" in line]
    assert len(lines) == 1, f"expected one Sources line for {source_name}"
    return lines[0]


class TestSourceWindowWarning:
    """A Source whose whole window is new may have had older Articles cut off."""

    async def test_full_window_of_new_articles_is_flagged(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [_article(f"WIN_{i}") for i in range(3)]

        stats = await _run(storage, articles, _event_decision(), flow_config=_flow_config(3))

        line = _source_line(_build_run_report(stats, dry_run=False), _RZ24)
        assert "⚠️" in line
        assert "max_articles=3" in line

    async def test_fewer_articles_than_the_window_are_not_flagged(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [_article(f"FEW_{i}") for i in range(2)]

        stats = await _run(storage, articles, _event_decision(), flow_config=_flow_config(3))

        assert "⚠️" not in _source_line(_build_run_report(stats, dry_run=False), _RZ24)

    async def test_full_window_with_an_article_already_in_storage_is_not_flagged(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        seen = _article("SEEN_0")
        await _run(storage, [seen], _event_decision(), flow_config=_flow_config(3))

        articles = [seen, _article("SEEN_1"), _article("SEEN_2")]
        stats = await _run(storage, articles, _event_decision(), flow_config=_flow_config(3))

        assert "⚠️" not in _source_line(_build_run_report(stats, dry_run=False), _RZ24)

    async def test_full_window_with_an_error_article_being_retried_is_not_flagged(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        retried = _article("RETRY_0")
        await storage.save_decision(retried, Decision.ERROR)

        articles = [retried, _article("RETRY_1"), _article("RETRY_2")]
        stats = await _run(storage, articles, _event_decision(), flow_config=_flow_config(3))

        # The Error Article is new again, but it proves the window overlapped a Run
        assert stats.new_articles == 3
        assert "⚠️" not in _source_line(_build_run_report(stats, dry_run=False), _RZ24)

    async def test_other_sources_are_unaffected(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [_article(f"MIX_{i}") for i in range(3)] + [_rzn_article("mix-a")]

        stats = await _run(
            storage, articles, _event_decision(), flow_config=_flow_config(3, rzn_window=3)
        )

        report = _build_run_report(stats, dry_run=False)
        assert "⚠️" in _source_line(report, _RZ24)
        assert "⚠️" not in _source_line(report, _RZN)

    async def test_dry_run_also_flags_a_full_window(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [_article(f"DRYWIN_{i}") for i in range(3)]

        stats = await _run(
            storage, articles, _event_decision(), dry_run=True, flow_config=_flow_config(3)
        )

        assert "max_articles=3" in _source_line(_build_run_report(stats, dry_run=True), _RZ24)
        assert storage.count() == 0


def _boring_decision() -> AIDecision:
    return _event_decision(is_event=False).model_copy(
        update={"is_interesting": False, "score": 2.0}
    )


class TestRepeatInOneRun:
    """An Article about a Story the AI already judged in this Run skips Gemini."""

    async def test_same_normalised_title_is_judged_once_and_posted_once(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        # Oldest first: the last listed Article is the one judged first.
        later = _article("TITLE_B", "Koncert, w RZESZOWIE!")
        earlier = _article("TITLE_A", "Koncert w Rzeszowie")
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(storage, [later, earlier], evaluate=evaluate)

        assert evaluate.await_count == 1
        assert bot_api.posts == [_MAIN]
        assert stats.posted == 1
        assert stats.repeats == 1
        record = storage.get_record(later.id)
        assert record is not None
        assert record.decision == Decision.REPEAT
        assert record.repeat_of == earlier.id

    async def test_run_report_links_each_repeat_to_its_original_and_counts_it_apart(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        later = _article("REP_B", "Utrudnienia na Podkarpackiej")
        earlier = _article("REP_A", "Utrudnienia na Podkarpackiej")

        stats = await _run(storage, [later, earlier], _event_decision(is_event=False))

        assert stats.skipped == 0
        report = _build_run_report(stats, dry_run=False)
        repeat_lines = [line for line in report.splitlines() if "🔁" in line]
        assert len(repeat_lines) == 1
        assert f'href="{later.url}"' in repeat_lines[0]
        assert f'href="{earlier.url}"' in repeat_lines[0]
        assert "repeats 1" in report
        assert "skipped 0" in report

    async def test_same_rzeszow24_slug_with_a_new_title_is_a_repeat(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        slug = "remont-mostu-zamkowego"
        later = _article("SLUG_B").model_copy(
            update={"url": f"https://rzeszow24.info/wiadomosci/{slug}/SLUG_B"}
        )
        earlier = _article("SLUG_A").model_copy(
            update={"url": f"https://rzeszow24.info/wiadomosci/{slug}/SLUG_A"}
        )
        assert later.title_pl != earlier.title_pl
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(storage, [later, earlier], evaluate=evaluate)

        assert evaluate.await_count == 1
        assert bot_api.posts == [_MAIN]
        record = storage.get_record(later.id)
        assert record is not None
        assert record.decision == Decision.REPEAT
        assert record.repeat_of == earlier.id
        assert stats.repeats == 1

    async def test_same_slug_on_two_sources_is_not_a_repeat(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        slug = "dni-rzeszowa"
        rz24 = _article("CROSS_1").model_copy(
            update={"url": f"https://rzeszow24.info/imprezy/{slug}/CROSS_1"}
        )
        rzn = _rzn_article(slug)
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(
            storage, [rz24, rzn], evaluate=evaluate, flow_config=_flow_config(rzn_window=5)
        )

        assert evaluate.await_count == 2
        assert stats.repeats == 0
        assert stats.posted == 2

    async def test_title_match_works_across_sources(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        rz24 = _article("XSRC_1", "Pożar hali przy Lubelskiej")
        rzn = _rzn_article("pozar-hali", "Pożar hali przy Lubelskiej")
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(
            storage, [rz24, rzn], evaluate=evaluate, flow_config=_flow_config(rzn_window=5)
        )

        assert evaluate.await_count == 1
        assert stats.repeats == 1
        assert bot_api.posts == [_MAIN]

    async def test_skipped_original_still_produces_a_repeat(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        later = _article("SKIP_B", "Kradzież roweru")
        earlier = _article("SKIP_A", "Kradzież roweru")
        evaluate = AsyncMock(return_value=_boring_decision())

        stats = await _run(storage, [later, earlier], evaluate=evaluate)

        assert evaluate.await_count == 1
        assert stats.skipped == 1
        assert stats.repeats == 1
        assert bot_api.posts == []

    @pytest.mark.parametrize(
        "first_failure",
        [RuntimeError("broken"), GeminiUnavailableError("503")],
        ids=["error", "deferred"],
    )
    async def test_failed_original_is_never_an_original(
        self, bot_api: FakeBotApi, storage: InMemoryStorage, first_failure: Exception
    ) -> None:
        later = _article("FAIL_B", "Koncert Dżemu")
        earlier = _article("FAIL_A", "Koncert Dżemu")
        evaluate = AsyncMock(side_effect=[first_failure, _event_decision(is_event=False)])

        stats = await _run(storage, [later, earlier], evaluate=evaluate)

        assert evaluate.await_count == 2
        assert stats.repeats == 0
        assert stats.posted == 1
        record = storage.get_record(later.id)
        assert record is not None
        assert record.decision == Decision.POSTED

    async def test_repeat_is_settled_on_the_next_run(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [_article("SET_B", "Mecz Resovii"), _article("SET_A", "Mecz Resovii")]
        await _run(storage, articles, _event_decision(is_event=False))
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        run2 = await _run(storage, articles, evaluate=evaluate)

        assert run2.new_articles == 0
        assert evaluate.await_count == 0
        assert bot_api.posts == [_MAIN]

    async def test_dry_run_reports_repeats_and_writes_nothing(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        later = _article("DRYREP_B", "Festyn na Baranówce")
        earlier = _article("DRYREP_A", "Festyn na Baranówce")
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(storage, [later, earlier], evaluate=evaluate, dry_run=True)

        assert evaluate.await_count == 1
        assert stats.repeats == 1
        assert "🔁" in _build_run_report(stats, dry_run=True)
        assert storage.count() == 0
        assert bot_api.posts == []


class TestRepeatAcrossRuns:
    """Originals also include AI-judged Articles from the last 72 hours."""

    async def test_story_judged_in_an_earlier_run_is_a_repeat(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        first = _article("ACROSS_A", "Nowe rondo na Krakowskiej")
        await _run(storage, [first], _event_decision(is_event=False))
        republished = _article("ACROSS_B", "Nowe rondo na Krakowskiej")
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(storage, [first, republished], evaluate=evaluate)

        assert evaluate.await_count == 0
        assert bot_api.posts == [_MAIN]
        assert stats.repeats == 1
        assert stats.article_log[0].original is not None
        assert stats.article_log[0].original.article_id == first.id
        record = storage.get_record(republished.id)
        assert record is not None
        assert record.repeat_of == first.id

    async def test_original_older_than_72_hours_is_not_matched(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        first = _article("OLD_A", "Nowe rondo na Krakowskiej")
        await _run(storage, [first], _event_decision(is_event=False))
        record = storage.get_record(first.id)
        assert record is not None
        storage._records[first.id] = record.model_copy(
            update={"seen_at": datetime.now(UTC) - timedelta(hours=73)}
        )
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(
            storage, [_article("OLD_B", "Nowe rondo na Krakowskiej")], evaluate=evaluate
        )

        assert evaluate.await_count == 1
        assert stats.repeats == 0

    async def test_dry_run_matches_stored_originals_too(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        first = _article("DRYX_A", "Nowe rondo na Krakowskiej")
        await _run(storage, [first], _event_decision(is_event=False))
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(
            storage,
            [_article("DRYX_B", "Nowe rondo na Krakowskiej")],
            evaluate=evaluate,
            dry_run=True,
        )

        assert evaluate.await_count == 0
        assert stats.repeats == 1
        assert storage.get_record("DRYX_B") is None

    async def test_no_storage_read_without_new_articles(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        article = _article("NOREAD_A")
        await _run(storage, [article], _event_decision(is_event=False))
        read = AsyncMock(wraps=storage.recent_judged_articles)

        with patch.object(storage, "recent_judged_articles", read):
            stats = await _run(storage, [article], _event_decision(is_event=False))

        assert stats.new_articles == 0
        read.assert_not_awaited()


class TestNearMatchHint:
    """A similar title never blocks an Article but leaves a hint for the admin."""

    async def test_similar_title_goes_to_gemini_and_carries_a_hint(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        later = _article("NEAR_B", "Nowe rondo na Krakowskiej otwarte")
        earlier = _article("NEAR_A", "Nowe rondo na ulicy Krakowskiej")
        evaluate = AsyncMock(return_value=_event_decision(is_event=False))

        stats = await _run(storage, [later, earlier], evaluate=evaluate)

        assert evaluate.await_count == 2
        assert stats.posted == 2
        assert stats.repeats == 0
        report = _build_run_report(stats, dry_run=False)
        hint_lines = [line for line in report.splitlines() if "схожа" in line]
        assert len(hint_lines) == 1
        assert f'href="{later.url}"' in hint_lines[0]
        assert f'href="{earlier.url}"' in hint_lines[0]
        assert "0.67" in hint_lines[0]

    async def test_title_below_the_threshold_gets_no_hint(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [
            _article("FAR_B", "Mecz Resovii w hali Podpromie"),
            _article("FAR_A", "Koncert w filharmonii"),
        ]

        stats = await _run(storage, articles, _event_decision(is_event=False))

        assert "схожа" not in _build_run_report(stats, dry_run=False)

    async def test_exact_match_is_a_repeat_without_a_hint(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        articles = [_article("EXACT_B", "Koncert Dżemu"), _article("EXACT_A", "Koncert Dżemu")]

        stats = await _run(storage, articles, _event_decision(is_event=False))

        assert stats.repeats == 1
        assert "схожа" not in _build_run_report(stats, dry_run=False)

    async def test_hint_also_points_at_stored_articles_in_a_dry_run(
        self, bot_api: FakeBotApi, storage: InMemoryStorage
    ) -> None:
        stored = _article("NEARDRY_A", "Nowe rondo na ulicy Krakowskiej")
        await _run(storage, [stored], _event_decision(is_event=False))
        records_before = storage.all_records()

        stats = await _run(
            storage,
            [_article("NEARDRY_B", "Nowe rondo na Krakowskiej otwarte")],
            _event_decision(is_event=False),
            dry_run=True,
        )

        report = _build_run_report(stats, dry_run=True)
        assert [line for line in report.splitlines() if "схожа" in line and stored.url in line]
        assert storage.all_records() == records_before
