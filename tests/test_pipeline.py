"""Tests for the pipeline orchestrator.

All external dependencies (scraper, AI, Telegram) are mocked with
unittest.mock.AsyncMock so we test the orchestration logic in isolation:
  - Does it call AI for new articles only?
  - Does it save skipped AND posted articles to storage?
  - Does it continue after a single article error?
  - Does dry-run skip publishing?
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from rz_flow.ai import GeminiQuotaExhaustedError, GeminiServerError
from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig, PipelineConfig, SourceConfig
from rz_flow.models import AIDecision, Article, Category, CategoryTag, Decision
from rz_flow.pipeline import Pipeline, PipelineStats
from rz_flow.storage import InMemoryStorage


def _make_settings() -> Settings:
    return Settings(
        telegram_bot_token="fake:token",
        telegram_channel_id="-100123",
        gemini_api_key="fake-key",
        turso_database_url="libsql://fake.turso.io",
        turso_auth_token="fake-token",
        ai_min_score=7.0,
    )


def _make_article(article_id: str, category: Category = Category.IMPREZY) -> Article:
    return Article(
        id=article_id,
        url=f"https://rzeszow24.info/imprezy/{article_id}",
        category=category,
        title_pl="Festiwal w Rzeszowie",
        summary_pl="Opis festiwalu.",
    )


def _make_interesting_decision(score: float = 8.0) -> AIDecision:
    return AIDecision(
        is_interesting=True,
        score=score,
        category_tag=CategoryTag.FESTIVAL,
        ua_title="Фестиваль у Жешові",
        ua_summary="Великий фестиваль у центрі міста.",
        reason="Great public event",
    )


def _make_boring_decision(score: float = 3.0) -> AIDecision:
    return AIDecision(
        is_interesting=False,
        score=score,
        category_tag=CategoryTag.OTHER,
        ua_title="Кримінальна новина",
        ua_summary="Деталі злочину.",
        reason="Criminal news, not suitable",
    )


def _make_flow_config(max_posts_per_run: int = 10) -> FlowConfig:
    return FlowConfig(
        sources=[
            SourceConfig(
                scraper="NajnowszeScraper",
                base_url="https://rzeszow24.info/najnowsze",
                max_articles=5,
            )
        ],
        pipeline=PipelineConfig(
            max_posts_per_run=max_posts_per_run,
            # Zero delays so tests run at full speed (production uses 5s/2s for rate limiting)
            inter_ai_delay_seconds=0.0,
            inter_post_delay_seconds=0.0,
        ),
    )


def _build_pipeline(storage: InMemoryStorage, max_posts_per_run: int = 10) -> Pipeline:
    return Pipeline(
        settings=_make_settings(),
        storage=storage,
        flow_config=_make_flow_config(max_posts_per_run=max_posts_per_run),
    )


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


class TestPipelineRunBasic:
    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_posts_interesting_article(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        article = _make_article("NEW_ARTICLE_12345")
        mock_fetch.return_value = ([article], {"rzeszow24/najnowsze": 1})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock(return_value=_make_interesting_decision())

        mock_tg = MagicMock()
        mock_tg_cls.return_value = mock_tg
        publish_result = MagicMock()
        publish_result.message_id = 99
        mock_tg.publish = AsyncMock(return_value=publish_result)

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.posted == 1
        assert stats.skipped == 0
        assert stats.errors == 0

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_skips_boring_article(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        article = _make_article("BORING_ARTICLE_123")
        mock_fetch.return_value = ([article], {"rzeszow24/najnowsze": 1})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock(return_value=_make_boring_decision())

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.skipped == 1
        assert stats.posted == 0

        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.SKIPPED

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_skips_article_below_min_score(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """is_interesting=True but score < threshold → skip."""
        article = _make_article("LOWSCORE_ARTICLE_1")
        mock_fetch.return_value = ([article], {"rzeszow24/najnowsze": 1})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        low_score = _make_interesting_decision(score=5.0)
        low_score = low_score.model_copy(update={"is_interesting": True})
        mock_ai.evaluate = AsyncMock(return_value=low_score)

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.skipped == 1

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_continues_after_single_article_error(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """One article failing AI should not stop processing of the next."""
        articles = [_make_article("FAIL_ARTICLE_12345"), _make_article("GOOD_ARTICLE_12345")]
        mock_fetch.return_value = (articles, {"rzeszow24/najnowsze": 2})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock(
            side_effect=[
                Exception("Gemini error"),
                _make_interesting_decision(),
            ]
        )

        mock_tg = MagicMock()
        mock_tg_cls.return_value = mock_tg
        publish_result = MagicMock()
        publish_result.message_id = 10
        mock_tg.publish = AsyncMock(return_value=publish_result)

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.errors == 1
        assert stats.posted == 1

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_dry_run_does_not_call_publish(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        article = _make_article("DRY_RUN_ARTICLE_1")
        mock_fetch.return_value = ([article], {"rzeszow24/najnowsze": 1})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock(return_value=_make_interesting_decision())

        mock_tg = MagicMock()
        mock_tg_cls.return_value = mock_tg
        mock_tg.publish = AsyncMock()

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run(dry_run=True)

        assert stats.dry_run is True
        assert stats.posted == 1
        mock_tg.publish.assert_not_called()
        # Dry-run must NOT persist anything — article should be retryable next run
        assert storage.get_record("DRY_RUN_ARTICLE_1") is None

    @patch("rz_flow.pipeline.fetch_articles")
    async def test_returns_zero_stats_when_no_articles(
        self,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        mock_fetch.return_value = ([], {})

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.total_scraped == 0
        assert stats.new_articles == 0

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_skips_already_seen_articles(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        article = _make_article("ALREADY_SEEN_1234")
        # Pre-populate storage so article is "seen"
        await storage.save_decision(article, Decision.POSTED)

        mock_fetch.return_value = ([article], {"rzeszow24/najnowsze": 1})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock()  # Should never be called

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.new_articles == 0
        mock_ai.evaluate.assert_not_called()


    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_stops_on_quota_exhausted_and_does_not_save(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """When daily quota is exhausted, the pipeline stops and articles are NOT saved.

        This ensures unsaved articles are retried on the next pipeline run
        (they'll pass filter_new_ids again since they're not in storage).
        """
        articles = [
            _make_article("QUOTA_ART_1"),
            _make_article("QUOTA_ART_2"),
            _make_article("QUOTA_ART_3"),
        ]
        mock_fetch.return_value = (articles, {"rzeszow24/najnowsze": 3})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        # First article hits daily quota
        mock_ai.evaluate = AsyncMock(side_effect=GeminiQuotaExhaustedError("quota gone"))

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.quota_exhausted is True
        # No articles should be saved in storage — they'll be retried next run
        assert storage.get_record("QUOTA_ART_1") is None
        assert storage.get_record("QUOTA_ART_2") is None
        # AI was only called once (stopped immediately after first quota error)
        assert mock_ai.evaluate.call_count == 1
        # Admin run report: quota row appears in article_log without DB save
        assert len(stats.article_log) == 1
        assert stats.article_log[0].article_id == "QUOTA_ART_1"
        assert stats.article_log[0].report_icon == "⏸"

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_processes_articles_before_quota_exhausted(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """Articles processed before quota exhaustion are saved normally."""
        articles = [
            _make_article("GOOD_ART_1"),
            _make_article("QUOTA_ART_2"),
        ]
        mock_fetch.return_value = (articles, {"rzeszow24/najnowsze": 2})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai

        mock_tg = MagicMock()
        mock_tg_cls.return_value = mock_tg
        publish_result = MagicMock()
        publish_result.message_id = 42
        mock_tg.publish = AsyncMock(return_value=publish_result)

        mock_ai.evaluate = AsyncMock(
            side_effect=[
                _make_interesting_decision(),  # first article succeeds
                GeminiQuotaExhaustedError("quota gone"),  # second hits quota
            ]
        )

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        assert stats.quota_exhausted is True
        assert stats.posted == 1
        # First article was saved, second was NOT
        assert storage.get_record("GOOD_ART_1") is not None
        assert storage.get_record("QUOTA_ART_2") is None
        assert len(stats.article_log) == 2
        assert stats.article_log[0].decision == Decision.POSTED
        assert stats.article_log[1].article_id == "QUOTA_ART_2"
        assert stats.article_log[1].report_icon == "⏸"


class TestPipelinePostCap:
    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_stops_at_post_cap(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """QW-7: pipeline stops after max_posts_per_run posts are made."""
        articles = [_make_article(f"ART_{i:04d}_XYZABCDE") for i in range(5)]
        mock_fetch.return_value = (articles, {"rzeszow24/najnowsze": 5})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        # All articles are interesting
        mock_ai.evaluate = AsyncMock(return_value=_make_interesting_decision())

        mock_tg = MagicMock()
        mock_tg_cls.return_value = mock_tg
        publish_result = MagicMock()
        publish_result.message_id = 1
        mock_tg.publish = AsyncMock(return_value=publish_result)

        # Cap at 2 posts
        pipeline = _build_pipeline(storage, max_posts_per_run=2)
        stats = await pipeline.run()

        assert stats.posted == 2
        assert stats.post_cap_reached is True
        # Only 2 articles should be published despite 5 being available
        assert mock_tg.publish.call_count == 2

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_cap_not_triggered_when_below_limit(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """QW-7: post_cap_reached stays False when posts are within the limit."""
        articles = [_make_article(f"ART_{i:04d}_XYZABCDE") for i in range(3)]
        mock_fetch.return_value = (articles, {"rzeszow24/najnowsze": 3})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock(return_value=_make_interesting_decision())

        mock_tg = MagicMock()
        mock_tg_cls.return_value = mock_tg
        publish_result = MagicMock()
        publish_result.message_id = 1
        mock_tg.publish = AsyncMock(return_value=publish_result)

        # Cap at 5, but only 3 articles — cap should not be triggered
        pipeline = _build_pipeline(storage, max_posts_per_run=5)
        stats = await pipeline.run()

        assert stats.posted == 3
        assert stats.post_cap_reached is False


class TestPipelineBranchCoverage:
    """Covers branches missed by the happy-path tests above."""

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_fetch_articles_fatal_error_propagates(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """When fetch_articles itself raises, the exception propagates out of run()."""
        mock_fetch.side_effect = RuntimeError("DNS lookup failed")

        pipeline = _build_pipeline(storage)
        with pytest.raises(RuntimeError, match="DNS lookup failed"):
            await pipeline.run()

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_gemini_server_error_skips_without_saving_continues(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
        storage: InMemoryStorage,
    ) -> None:
        """GeminiServerError (503) is transient: article NOT saved, pipeline continues."""
        articles = [_make_article("SERVER_ERR_ART_01X"), _make_article("GOOD_ART_X123456")]
        mock_fetch.return_value = (articles, {"rzeszow24/najnowsze": 2})

        server_err = genai_errors.ServerError(
            503, {"error": {"code": 503, "message": "Service Unavailable"}}
        )

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock(
            side_effect=[server_err, _make_interesting_decision()]
        )

        mock_tg = MagicMock()
        mock_tg_cls.return_value = mock_tg
        publish_result = MagicMock()
        publish_result.message_id = 5
        mock_tg.publish = AsyncMock(return_value=publish_result)

        pipeline = _build_pipeline(storage)
        stats = await pipeline.run()

        # GeminiServerError → "continue" (not counted as error)
        assert stats.errors == 0
        assert stats.posted == 1
        # Article that hit ServerError is NOT saved — will be retried next run
        assert storage.get_record("SERVER_ERR_ART_01X") is None
        # Second article WAS processed and saved
        assert storage.get_record("GOOD_ART_X123456") is not None
        assert len(stats.article_log) == 2
        assert stats.article_log[0].report_icon == "🔄"
        assert stats.article_log[1].decision == Decision.POSTED

    @patch("rz_flow.pipeline.fetch_articles")
    @patch("rz_flow.pipeline.GeminiAIFilter")
    @patch("rz_flow.pipeline.TelegramPublisher")
    async def test_save_decision_failure_logs_but_does_not_crash(
        self,
        mock_tg_cls: MagicMock,
        mock_ai_cls: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        """If save_decision throws, the pipeline logs the error and keeps running."""
        article = _make_article("SAVE_FAIL_ART_01XX")
        mock_fetch.return_value = ([article], {"rzeszow24/najnowsze": 1})

        mock_ai = MagicMock()
        mock_ai_cls.return_value = mock_ai
        mock_ai.evaluate = AsyncMock(return_value=_make_boring_decision())

        broken_storage = MagicMock()
        broken_storage.filter_new_ids = AsyncMock(return_value=[article.id])
        broken_storage.save_decision = AsyncMock(side_effect=RuntimeError("DB connection lost"))

        pipeline = Pipeline(
            settings=_make_settings(),
            storage=broken_storage,
            flow_config=_make_flow_config(),
        )

        # Must not raise despite storage failure
        stats = await pipeline.run()
        assert stats.skipped == 1


class TestPipelineStats:
    def test_str_representation_no_dry_run(self) -> None:
        stats = PipelineStats(total_scraped=10, new_articles=5, posted=3, skipped=2)
        assert "posted=3" in str(stats)
        assert "DRY RUN" not in str(stats)

    def test_str_representation_dry_run(self) -> None:
        stats = PipelineStats(dry_run=True, posted=5)
        assert "DRY RUN" in str(stats)

    def test_str_representation_quota_exhausted(self) -> None:
        stats = PipelineStats(quota_exhausted=True, posted=2, skipped=1)
        assert "QUOTA EXHAUSTED" in str(stats)

    def test_str_representation_post_cap(self) -> None:
        """QW-7: POST CAP flag appears in string representation."""
        stats = PipelineStats(post_cap_reached=True, posted=5)
        assert "POST CAP" in str(stats)
