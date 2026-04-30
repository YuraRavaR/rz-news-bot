"""Main pipeline orchestrator.

Coordinates all stages: scrape → filter new → AI evaluate → publish → save.

Design principles:
  - Each article is processed independently (no batch AI calls)
  - Errors on individual articles don't stop the pipeline
  - All results (posted, skipped, error) are persisted in storage
  - Dry-run mode processes everything but skips Telegram publishing
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import structlog

from rz_flow.ai import GeminiAIFilter, GeminiQuotaExhaustedError
from rz_flow.config import Settings
from rz_flow.models import AIDecision, Article, Decision
from rz_flow.scraper import fetch_articles
from rz_flow.storage import StorageProtocol
from rz_flow.telegram import TelegramPublisher

logger = structlog.get_logger(__name__)

# Delay between consecutive Telegram messages to avoid hitting rate limits
_INTER_POST_DELAY_SECONDS = 2.0

# Delay between Gemini API calls — free tier allows 15 RPM = 4 sec minimum.
# We use 5 sec for safety margin.
_INTER_AI_DELAY_SECONDS = 5.0


@dataclass
class PipelineStats:
    """Summary statistics returned after a pipeline run."""

    total_scraped: int = 0
    new_articles: int = 0
    posted: int = 0
    skipped: int = 0
    errors: int = 0
    dry_run: bool = False
    quota_exhausted: bool = False  # True if Gemini daily quota was hit

    def __str__(self) -> str:
        mode = " [DRY RUN]" if self.dry_run else ""
        quota = " [QUOTA EXHAUSTED]" if self.quota_exhausted else ""
        return (
            f"Pipeline run{mode}{quota}: "
            f"scraped={self.total_scraped}, "
            f"new={self.new_articles}, "
            f"posted={self.posted}, "
            f"skipped={self.skipped}, "
            f"errors={self.errors}"
        )


@dataclass
class Pipeline:
    """Assembles and runs the full Rz-Flow pipeline."""

    settings: Settings
    storage: StorageProtocol
    ai_filter: GeminiAIFilter = field(init=False)
    publisher: TelegramPublisher = field(init=False)

    def __post_init__(self) -> None:
        self.ai_filter = GeminiAIFilter(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gemini_model,
        )
        self.publisher = TelegramPublisher(
            bot_token=self.settings.telegram_bot_token,
            channel_id=self.settings.telegram_channel_id,
        )

    async def run(self, dry_run: bool = False) -> PipelineStats:
        """Execute the full pipeline and return statistics.

        Args:
            dry_run: If True, evaluate articles but do NOT publish to Telegram.
        """
        stats = PipelineStats(dry_run=dry_run)
        _start = time.monotonic()

        source = urlparse(self.settings.scraper_base_url).hostname or self.settings.scraper_base_url

        # ── Stage 1: Scrape ───────────────────────────────────────────────────
        log = logger.bind(dry_run=dry_run)
        log.info(
            "pipeline_started",
            source=source,
            model=self.settings.gemini_model,
            min_score=self.settings.ai_min_score,
        )
        try:
            all_articles = await fetch_articles(self.settings)
        except Exception as exc:
            log.exception("scrape_failed", error=str(exc))
            raise  # Fatal — can't continue without articles

        stats.total_scraped = len(all_articles)
        log.info("scrape_complete", total=stats.total_scraped)

        if not all_articles:
            log.info("no_articles_found")
            return stats

        # ── Stage 2: Filter already-seen articles ─────────────────────────────
        all_ids = [a.id for a in all_articles]
        new_ids_set = set(await self.storage.filter_new_ids(all_ids))
        new_articles = [a for a in all_articles if a.id in new_ids_set]

        stats.new_articles = len(new_articles)
        seen_count = stats.total_scraped - stats.new_articles
        log.info("filter_complete", new=stats.new_articles, seen=seen_count)

        if not new_articles:
            log.info("no_new_articles")
            return stats

        # ── Stage 3: AI evaluate + publish ────────────────────────────────────
        for i, article in enumerate(new_articles):
            quota_hit = await self._process_article(article, stats, dry_run)
            if quota_hit:
                # Daily quota exhausted — no point processing remaining articles.
                # They will be retried on the next pipeline run (not saved as error).
                log.warning(
                    "quota_exhausted_stopping",
                    processed=i + 1,
                    remaining=len(new_articles) - i - 1,
                )
                stats.quota_exhausted = True
                break

            is_last = i == len(new_articles) - 1
            if not is_last:
                # Respect Gemini free-tier rate limit (15 RPM)
                await asyncio.sleep(_INTER_AI_DELAY_SECONDS)
            if not dry_run and stats.posted > 0:
                await asyncio.sleep(_INTER_POST_DELAY_SECONDS)

        elapsed_s = round(time.monotonic() - _start)
        log.info(
            "pipeline_complete",
            posted=stats.posted,
            skipped=stats.skipped,
            errors=stats.errors,
            elapsed_s=elapsed_s,
        )
        return stats

    async def _process_article(
        self,
        article: Article,
        stats: PipelineStats,
        dry_run: bool,
    ) -> bool:
        """Process a single article: AI evaluate → maybe publish → save.

        Returns:
            True if Gemini daily quota was exhausted (caller should stop loop).
            False in all other cases (success, skip, transient error).
        """
        ai_decision: AIDecision | None = None
        decision = Decision.ERROR
        tg_message_id: int | None = None
        quota_exhausted = False

        log = logger.bind(article_id=article.id, category=article.category.value)

        try:
            # Stage 3a: AI evaluation
            ai_decision = await self.ai_filter.evaluate(article)
            log.info(
                "ai_evaluated",
                score=ai_decision.score,
                is_interesting=ai_decision.is_interesting,
                reason=ai_decision.reason,
            )

            # Stage 3b: Apply threshold and publish
            if ai_decision.is_interesting and ai_decision.score >= self.settings.ai_min_score:
                if dry_run:
                    log.info(
                        "dry_run_would_publish",
                        score=ai_decision.score,
                        ua_title=ai_decision.ua_title,
                    )
                    stats.posted += 1
                    decision = Decision.POSTED
                else:
                    result = await self.publisher.publish(article, ai_decision)
                    tg_message_id = result.message_id
                    log.info(
                        "published",
                        score=ai_decision.score,
                        tg_message_id=tg_message_id,
                        ua_title=ai_decision.ua_title,
                    )
                    stats.posted += 1
                    decision = Decision.POSTED
            else:
                log.info("skipped", score=ai_decision.score, reason=ai_decision.reason)
                stats.skipped += 1
                decision = Decision.SKIPPED

        except GeminiQuotaExhaustedError:
            # Do NOT save as error — we'll retry this article on the next run
            # (it won't appear in filter_new_ids since it's not persisted)
            quota_exhausted = True
            return quota_exhausted

        except Exception as exc:
            log.exception("article_error", error=str(exc))
            stats.errors += 1
            decision = Decision.ERROR

        # Stage 3c: Always save the result (even errors), so we don't retry
        try:
            await self.storage.save_decision(
                article=article,
                decision=decision,
                ai_decision=ai_decision,
                tg_message_id=tg_message_id,
            )
        except Exception as save_exc:
            log.error("save_failed", error=str(save_exc))

        return quota_exhausted
