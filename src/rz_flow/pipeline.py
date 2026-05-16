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
from collections import Counter, deque
from dataclasses import dataclass, field

import structlog
from google.genai.errors import ServerError as GeminiServerError

from rz_flow.ai import GeminiAIFilter, GeminiQuotaExhaustedError
from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig
from rz_flow.models import AIDecision, Article, Decision
from rz_flow.scraper import fetch_articles
from rz_flow.sources import get_active_sources
from rz_flow.storage import StorageProtocol
from rz_flow.telegram import TelegramPublisher

logger = structlog.get_logger(__name__)


def _order_new_articles_round_robin_oldest_first(articles: list[Article]) -> list[Article]:
    """Reorder new articles: round-robin across sources, oldest DOM row first per source.

    ``fetch_articles`` concatenates each source in config order (newest-first DOM per
    source). We reverse each source's slice for oldest-first, then interleave sources
    so one article from source 1, one from source 2, … until all queues are empty.
    """
    if not articles:
        return []
    order_keys: list[str] = []
    by_key: dict[str, list[Article]] = {}
    for a in articles:
        key = a.source_name.strip()
        if key not in by_key:
            by_key[key] = []
            order_keys.append(key)
        by_key[key].append(a)
    queues: dict[str, deque[Article]] = {
        k: deque(reversed(v)) for k, v in by_key.items()
    }
    out: list[Article] = []
    while True:
        took_any = False
        for key in order_keys:
            q = queues[key]
            if q:
                out.append(q.popleft())
                took_any = True
        if not took_any:
            break
    return out


@dataclass(frozen=True)
class RemainingArticleBrief:
    """New article not started in this run (queue behind post-cap / quota stop)."""

    article_id: str
    title_pl: str
    url: str
    source_name: str = ""


@dataclass
class ArticleRunEntry:
    """Per-article result collected during the run for admin reporting."""

    article_id: str
    title_pl: str
    ua_title: str | None
    score: float | None
    decision: Decision
    error_msg: str | None = None
    # When set, admin run report uses this icon instead of mapping from decision
    # (e.g. quota / 503 — not persisted, retry next run).
    report_icon: str | None = None
    # Scraper source label (e.g. rzeszow24/najnowsze) for grouped admin report
    source_name: str = ""
    # Original article URL (admin report link, including skipped)
    article_url: str = ""
    # Gemini rationale / Ukrainian summary when evaluation ran
    ai_reason: str | None = None
    ai_ua_summary: str | None = None
    # True when the article was also published to the events channel
    posted_to_events: bool = False


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
    post_cap_reached: bool = False  # True if max_posts_per_run was hit
    # New articles never started this run (indices after the stop); empty if none
    remaining_queued: list[RemainingArticleBrief] = field(default_factory=list)
    # Why the queue tail was skipped: "" | "post_cap" | "quota"
    remaining_stop_reason: str = ""
    # Per-source counts for admin reporting
    source_scraped: dict[str, int] = field(default_factory=dict)
    source_new: dict[str, int] = field(default_factory=dict)
    # source name → scrape URL (for clickable admin run report)
    source_urls: dict[str, str] = field(default_factory=dict)
    # Per-article log for admin run report
    article_log: list[ArticleRunEntry] = field(default_factory=list)
    # Admin run-report extras (set in Pipeline.run)
    elapsed_s: int = 0
    report_gemini_model: str = ""
    report_ai_min_score: float = 0.0

    def __str__(self) -> str:
        mode = " [DRY RUN]" if self.dry_run else ""
        quota = " [QUOTA EXHAUSTED]" if self.quota_exhausted else ""
        cap = " [POST CAP]" if self.post_cap_reached else ""
        return (
            f"Pipeline run{mode}{quota}{cap}: "
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
    flow_config: FlowConfig
    use_staging_channel: bool = False
    ai_filter: GeminiAIFilter = field(init=False)
    publisher: TelegramPublisher = field(init=False)
    # None when TELEGRAM_EVENTS_CHANNEL_ID is not configured — events still post to main channel
    events_publisher: TelegramPublisher | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.ai_filter = GeminiAIFilter(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gemini_model,
        )
        publish_id = self.settings.publish_telegram_chat_id(staging=self.use_staging_channel)
        self.publisher = TelegramPublisher(
            bot_token=self.settings.telegram_bot_token,
            channel_id=publish_id,
            report_display_timezone=self.flow_config.pipeline.report_display_timezone,
            mark_channel_posts_staging=self.use_staging_channel,
        )
        events_id = self.settings.events_telegram_chat_id(staging=self.use_staging_channel)
        if events_id:
            self.events_publisher = TelegramPublisher(
                bot_token=self.settings.telegram_bot_token,
                channel_id=events_id,
                report_display_timezone=self.flow_config.pipeline.report_display_timezone,
                mark_channel_posts_staging=self.use_staging_channel,
            )
            logger.info("events_channel_configured", events_channel_id=events_id)
        else:
            logger.info("events_channel_not_configured")

    async def run(self, dry_run: bool = False) -> PipelineStats:
        """Execute the full pipeline and return statistics.

        Args:
            dry_run: If True, evaluate articles but do NOT publish to Telegram and do NOT
                persist decisions to Turso. With ``use_staging_channel=True``, still uses
                staging Turso for ``filter_new_ids`` reads only.
        """
        stats = PipelineStats(dry_run=dry_run)
        stats.report_gemini_model = self.settings.gemini_model
        stats.report_ai_min_score = self.settings.ai_min_score
        _start = time.monotonic()

        def _stamp_elapsed() -> None:
            stats.elapsed_s = max(0, round(time.monotonic() - _start))

        active = self.flow_config.enabled_sources
        stats.source_urls = {
            s.name: src.base_url
            for s, src in zip(get_active_sources(self.flow_config), active)
        }

        # ── Stage 1: Scrape ───────────────────────────────────────────────────
        log = logger.bind(dry_run=dry_run)
        log.info(
            "pipeline_started",
            sources=[
                {"name": s.name, "base_url": src.base_url, "max_articles": src.max_articles}
                for s, src in zip(get_active_sources(self.flow_config), active)
            ],
            model=self.settings.gemini_model,
            min_score=self.settings.ai_min_score,
        )
        try:
            all_articles, source_scraped = await fetch_articles(self.settings, self.flow_config)
        except Exception as exc:
            log.exception("pipeline_fatal_error", error=str(exc))
            raise

        stats.total_scraped = len(all_articles)
        stats.source_scraped = source_scraped

        if not all_articles:
            log.info("no_articles_found")
            _stamp_elapsed()
            return stats

        # ── Stage 2: Filter already-seen articles ─────────────────────────────
        all_ids = [a.id for a in all_articles]
        new_ids_set = set(await self.storage.filter_new_ids(all_ids))
        new_articles = [a for a in all_articles if a.id in new_ids_set]
        new_articles = _order_new_articles_round_robin_oldest_first(new_articles)

        stats.new_articles = len(new_articles)
        seen_count = stats.total_scraped - stats.new_articles
        # Per-source new-article counts for the admin run report
        stats.source_new = dict(Counter(a.source_name for a in new_articles))
        log.info("filter_complete", new=stats.new_articles, seen=seen_count)

        if not new_articles:
            log.info("no_new_articles")
            _stamp_elapsed()
            return stats

        # ── Stage 3: AI evaluate + publish ────────────────────────────────────
        for i, article in enumerate(new_articles):
            stop_signal = await self._process_article(article, stats, dry_run)

            if stop_signal == "quota_exhausted":
                tail = new_articles[i + 1 :]
                if tail:
                    stats.remaining_queued = [
                        RemainingArticleBrief(
                            article_id=a.id,
                            title_pl=a.title_pl,
                            url=a.url,
                            source_name=a.source_name,
                        )
                        for a in tail
                    ]
                    stats.remaining_stop_reason = "quota"
                log.warning(
                    "quota_exhausted_stopping",
                    processed=i + 1,
                    remaining=len(new_articles) - i - 1,
                )
                stats.quota_exhausted = True
                break

            if stop_signal == "cap_reached":
                tail = new_articles[i + 1 :]
                if tail:
                    stats.remaining_queued = [
                        RemainingArticleBrief(
                            article_id=a.id,
                            title_pl=a.title_pl,
                            url=a.url,
                            source_name=a.source_name,
                        )
                        for a in tail
                    ]
                    stats.remaining_stop_reason = "post_cap"
                log.info(
                    "post_cap_reached",
                    cap=self.flow_config.pipeline.max_posts_per_run,
                    remaining=len(new_articles) - i - 1,
                )
                stats.post_cap_reached = True
                break

            is_last = i == len(new_articles) - 1
            if not is_last:
                # Respect Gemini free-tier rate limit (15 RPM)
                await asyncio.sleep(self.flow_config.pipeline.inter_ai_delay_seconds)
            if not dry_run and stats.posted > 0:
                await asyncio.sleep(self.flow_config.pipeline.inter_post_delay_seconds)

        _stamp_elapsed()
        log.info(
            "pipeline_complete",
            posted=stats.posted,
            skipped=stats.skipped,
            errors=stats.errors,
            elapsed_s=stats.elapsed_s,
        )
        return stats

    async def _process_article(
        self,
        article: Article,
        stats: PipelineStats,
        dry_run: bool,
    ) -> str:
        """Process a single article: AI evaluate → maybe publish → save.

        Routing logic:
          - is_interesting + is_event + events_publisher configured
              → post to events channel first, then post to main channel
          - is_interesting + (not is_event OR events_publisher not configured)
              → post to main channel only
          - not is_interesting → skip both channels

        Events channel failure is non-fatal: if the events channel post fails,
        the error is logged and we still proceed to post on the main channel.
        Main channel failure IS fatal for the article (marked as ERROR).

        Returns a stop-signal string for the caller:
            "continue"        — normal; keep processing remaining articles
            "quota_exhausted" — Gemini daily quota hit; stop immediately (article not saved)
            "cap_reached"     — max_posts_per_run reached after this post; stop the loop
        """
        ai_decision: AIDecision | None = None
        decision = Decision.ERROR
        tg_message_id: int | None = None
        tg_events_message_id: int | None = None
        posted_to_events = False
        error_msg: str | None = None

        log = logger.bind(article_id=article.id, category=article.category.value)

        try:
            # Stage 3a: AI evaluation
            log.info("ai_processing", title=article.title_pl)
            ai_decision = await self.ai_filter.evaluate(article)
            log.info(
                "ai_evaluated",
                score=ai_decision.score,
                is_interesting=ai_decision.is_interesting,
                is_event=ai_decision.is_event,
                reason=ai_decision.reason,
            )

            # Stage 3b: Apply threshold and publish
            should_publish = ai_decision.is_interesting and ai_decision.score >= self.settings.ai_min_score
            route_to_events = should_publish and ai_decision.is_event and self.events_publisher is not None

            if should_publish:
                if dry_run:
                    channels = ["events", "main"] if route_to_events else ["main"]
                    log.info(
                        "dry_run_would_publish",
                        score=ai_decision.score,
                        ua_title=ai_decision.ua_title,
                        channels=channels,
                    )
                    stats.posted += 1
                    decision = Decision.POSTED
                    posted_to_events = route_to_events
                else:
                    # Events channel first — failure is non-fatal
                    if route_to_events:
                        assert self.events_publisher is not None  # narrowing for type checker
                        try:
                            events_result = await self.events_publisher.publish(article, ai_decision)
                            tg_events_message_id = events_result.message_id
                            posted_to_events = True
                            log.info(
                                "published_to_events_channel",
                                tg_events_message_id=tg_events_message_id,
                            )
                        except Exception as events_exc:
                            # Non-fatal: log and continue to publish on the main channel
                            log.error(
                                "events_channel_publish_failed",
                                error=str(events_exc),
                                article_id=article.id,
                            )

                    # Main channel — failure IS fatal for the article
                    result = await self.publisher.publish(article, ai_decision)
                    tg_message_id = result.message_id
                    log.info(
                        "published",
                        score=ai_decision.score,
                        tg_message_id=tg_message_id,
                        ua_title=ai_decision.ua_title,
                        posted_to_events=posted_to_events,
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
            stats.article_log.append(
                ArticleRunEntry(
                    article_id=article.id,
                    title_pl=article.title_pl,
                    ua_title=None,
                    score=None,
                    decision=Decision.SKIPPED,
                    error_msg="Gemini daily quota exhausted — not saved, will retry on next run",
                    report_icon="⏸",
                    source_name=article.source_name,
                    article_url=article.url,
                )
            )
            return "quota_exhausted"

        except GeminiServerError as exc:
            # 503 UNAVAILABLE after retries — transient server overload.
            # Do NOT save to DB so the article is retried on the next pipeline run.
            log.warning("gemini_unavailable_skipping", error=str(exc))
            stats.article_log.append(
                ArticleRunEntry(
                    article_id=article.id,
                    title_pl=article.title_pl,
                    ua_title=None,
                    score=None,
                    decision=Decision.SKIPPED,
                    error_msg=f"Gemini temporarily unavailable (503): {exc}",
                    report_icon="🔄",
                    source_name=article.source_name,
                    article_url=article.url,
                )
            )
            return "continue"

        except Exception as exc:
            log.exception("article_error", error=str(exc))
            stats.errors += 1
            decision = Decision.ERROR
            error_msg = f"{type(exc).__name__}: {exc}"

        # Append to the per-run article log for the admin run report
        stats.article_log.append(
            ArticleRunEntry(
                article_id=article.id,
                title_pl=article.title_pl,
                ua_title=ai_decision.ua_title if ai_decision else None,
                score=ai_decision.score if ai_decision else None,
                decision=decision,
                error_msg=error_msg,
                source_name=article.source_name,
                article_url=article.url,
                ai_reason=ai_decision.reason if ai_decision else None,
                ai_ua_summary=ai_decision.ua_summary if ai_decision else None,
                posted_to_events=posted_to_events,
            )
        )

        # Stage 3c: Save result so we don't retry — skipped in dry_run (no side effects)
        if not dry_run:
            try:
                await self.storage.save_decision(
                    article=article,
                    decision=decision,
                    ai_decision=ai_decision,
                    tg_message_id=tg_message_id,
                    tg_events_message_id=tg_events_message_id,
                )
            except Exception as save_exc:
                log.error("save_failed", error=str(save_exc))

        # Signal cap after the post is saved so the article is not retried next run
        cap = self.flow_config.pipeline.max_posts_per_run
        if decision == Decision.POSTED and stats.posted >= cap:
            return "cap_reached"

        return "continue"
