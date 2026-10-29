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
from datetime import UTC, datetime, timedelta

import structlog

from rz_flow.ai import GeminiAIFilter
from rz_flow.config import Settings
from rz_flow.dedup import NearMatch, Original, SlugOf, StoryIndex
from rz_flow.errors import DeferralError, PublishingRecordError
from rz_flow.flow_config import FlowConfig, SourceConfig
from rz_flow.models import AIDecision, Article, Decision, PriorRecord, StopSignal
from rz_flow.scraper import fetch_articles
from rz_flow.sources import ScraperSource, get_active_sources
from rz_flow.storage import StorageProtocol
from rz_flow.telegram import TelegramDeliveryUnknownError, TelegramPublisher

logger = structlog.get_logger(__name__)

# Storage writes are retried in-process: the alternative to a successful write is a
# duplicate post on the next run, so it is worth a few seconds.
_SAVE_MAX_ATTEMPTS = 3
_SAVE_RETRY_DELAY_S = 1.0

# How far back stored AI-judged Articles count as originals for Repeats
_REPEAT_WINDOW = timedelta(hours=72)

# Early stops of the Run → the reason recorded for the "not reached" list in the
# run report. A new stop reason needs an entry here and wording in the report.
_REMAINING_STOP_REASONS: dict[StopSignal, str] = {
    StopSignal.QUOTA_EXHAUSTED: "quota",
    StopSignal.CAP_REACHED: "post_cap",
    StopSignal.PUBLISHING_BLOCKED: "publishing_blocked",
    StopSignal.TELEGRAM_RATE_LIMITED: "telegram_rate_limited",
}


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
    queues: dict[str, deque[Article]] = {k: deque(reversed(v)) for k, v in by_key.items()}
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


def _original(article: Article) -> Original:
    return Original(article_id=article.id, url=article.url, title_pl=article.title_pl)


def _slug_finder(sources: list[ScraperSource]) -> SlugOf:
    """url → (Source name, slug) from the first Source that recognises the URL."""

    def slug_of(url: str) -> tuple[str, str] | None:
        for source in sources:
            slug = source.slug(url)
            if slug:
                return source.name, slug
        return None

    return slug_of


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
    # Set when the main channel Post went out but the events channel Post failed
    events_error: str | None = None
    # For a Repeat: the Article the AI already judged about the same Story
    original: Original | None = None
    # Not a Repeat, but its title is similar to an original's (report-only hint)
    near_match: NearMatch | None = None


@dataclass
class PipelineStats:
    """Summary statistics returned after a pipeline run."""

    total_scraped: int = 0
    new_articles: int = 0
    posted: int = 0
    skipped: int = 0
    errors: int = 0
    # Articles not sent to Gemini because it already judged their Story
    repeats: int = 0
    dry_run: bool = False
    quota_exhausted: bool = False  # True if Gemini daily quota was hit
    post_cap_reached: bool = False  # True if max_posts_per_run was hit
    # Why the Run stopped early; CONTINUE when it went through every new Article
    stop_signal: StopSignal = StopSignal.CONTINUE
    # New articles never started this run (indices after the stop); empty if none
    remaining_queued: list[RemainingArticleBrief] = field(default_factory=list)
    # Why the queue tail was skipped: "" | "post_cap" | "quota" | "publishing_blocked"
    remaining_stop_reason: str = ""
    # Per-source counts for admin reporting
    source_scraped: dict[str, int] = field(default_factory=dict)
    source_new: dict[str, int] = field(default_factory=dict)
    # source name → scrape URL (for clickable admin run report)
    source_urls: dict[str, str] = field(default_factory=dict)
    # source name → max_articles, for Sources whose whole window was new Articles
    # (older ones may have been cut off; the window may need to grow)
    source_window_full: dict[str, int] = field(default_factory=dict)
    # Per-article log for admin run report
    article_log: list[ArticleRunEntry] = field(default_factory=list)
    # Articles left for the next Run by a failure that was not their fault
    deferred: int = 0
    # Published Articles the AI judged events, and how many of them have no Event date
    events_posted: int = 0
    events_without_date: int = 0
    # Operational warnings worth pushing to the admin chat immediately, separate from
    # the run report (e.g. a source that went silent, a post that may have duplicated).
    alerts: list[str] = field(default_factory=list)
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
            f"repeats={self.repeats}, "
            f"errors={self.errors}, "
            f"deferred={self.deferred}"
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
            model=self.flow_config.pipeline.gemini_model,
            min_score=self.settings.ai_min_score,
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
                show_event_date=True,
            )
            logger.info("events_channel_configured", events_channel_id=events_id)
        elif self.use_staging_channel:
            logger.warning("staging_events_channel_not_configured")
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
        stats.report_gemini_model = self.flow_config.pipeline.gemini_model
        stats.report_ai_min_score = self.settings.ai_min_score
        _start = time.monotonic()

        def _stamp_elapsed() -> None:
            stats.elapsed_s = max(0, round(time.monotonic() - _start))

        active = self.flow_config.enabled_sources
        # Instantiate scrapers once — this pairing drives both the log line and the
        # clickable source links in the admin run report.
        source_pairs = list(zip(get_active_sources(self.flow_config), active, strict=True))
        stats.source_urls = {s.name: src.base_url for s, src in source_pairs}

        # ── Stage 1: Scrape ───────────────────────────────────────────────────
        log = logger.bind(dry_run=dry_run)
        log.info(
            "pipeline_started",
            sources=[
                {"name": s.name, "base_url": src.base_url, "max_articles": src.max_articles}
                for s, src in source_pairs
            ],
            model=self.flow_config.pipeline.gemini_model,
            min_score=self.settings.ai_min_score,
        )
        # Before scraping, so a Publishing Article is resolved even after it has
        # dropped off its Source's listing page.
        if not dry_run:
            await self._resolve_stale_publishing(stats, log)

        try:
            all_articles, source_scraped = await fetch_articles(self.settings, self.flow_config)
        except Exception as exc:
            log.exception("pipeline_fatal_error", error=str(exc))
            raise

        stats.total_scraped = len(all_articles)
        stats.source_scraped = source_scraped
        self._check_source_health(stats, source_pairs, log)

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
        await self._check_source_windows(stats, source_pairs, new_articles, log)

        if not new_articles:
            log.info("no_new_articles")
            _stamp_elapsed()
            return stats

        # ── Stage 3: Repeat check + AI evaluate + publish ─────────────────────
        # Originals are Articles the AI judged in the last 72 hours (read once, oldest
        # first) and earlier in this Run, so of two Articles about one Story the
        # earlier one is the original.
        stories = StoryIndex(_slug_finder([s for s, _ in source_pairs]))
        since = datetime.now(UTC) - _REPEAT_WINDOW
        for judged in await self.storage.recent_judged_articles(since):
            stories.add(Original(article_id=judged.id, url=judged.url, title_pl=judged.title_pl))
        for i, article in enumerate(new_articles):
            original = stories.find_original(article)
            if original is not None:
                await self._record_repeat(article, original, stats, dry_run)
                continue

            stop_signal = await self._process_article(
                article, stats, dry_run, stories, stories.near_match(article)
            )

            if stop_signal is not StopSignal.CONTINUE:
                self._stop_early(stats, stop_signal, new_articles[i + 1 :], log)
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
            repeats=stats.repeats,
            errors=stats.errors,
            elapsed_s=stats.elapsed_s,
        )
        return stats

    def _stop_early(
        self,
        stats: PipelineStats,
        stop_signal: StopSignal,
        tail: list[Article],
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """Record why the Run stopped and which new Articles it never reached."""
        stats.stop_signal = stop_signal
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
            stats.remaining_stop_reason = _REMAINING_STOP_REASONS[stop_signal]
        if stop_signal is StopSignal.QUOTA_EXHAUSTED:
            stats.quota_exhausted = True
        elif stop_signal is StopSignal.CAP_REACHED:
            stats.post_cap_reached = True
        log.warning(
            "run_stopped_early",
            reason=stop_signal.value,
            remaining=len(tail),
            cap=self.flow_config.pipeline.max_posts_per_run,
        )

    async def _resolve_stale_publishing(
        self, stats: PipelineStats, log: structlog.stdlib.BoundLogger
    ) -> None:
        """Turn Publishing records left by an earlier Run into Unconfirmed, alerting once.

        Their Post may already be in the Main channel, so they are never posted again;
        the admin checks the channel by hand.
        """
        for article in await self.storage.resolve_stale_publishing():
            log.warning("publishing_unconfirmed", article_id=article.id, url=article.url)
            stats.alerts.append(
                f"Не вдалося підтвердити публікацію статті «{article.title_pl}»: "
                f"{article.url}. Перевірте основний канал: якщо поста там немає, "
                "опублікуйте його вручну. Бот більше не публікуватиме цю статтю."
            )

    @staticmethod
    def _check_source_health(
        stats: PipelineStats,
        source_pairs: list[tuple[ScraperSource, SourceConfig]],
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """Flag enabled sources that returned nothing.

        A layout change on the news site makes the parser return an empty list, which
        otherwise looks exactly like a quiet news day: the run succeeds, exits 0, and
        the channel just stops updating. Surfacing it is the only way to notice.
        """
        dead = [s.name for s, _ in source_pairs if stats.source_scraped.get(s.name, 0) == 0]
        if not dead:
            return
        log.warning("source_returned_no_articles", sources=dead)
        urls = ", ".join(f"{name} ({stats.source_urls.get(name, '?')})" for name in dead)
        stats.alerts.append(
            f"Джерело не повернуло жодної статті: {urls}. "
            "Ймовірно змінилась верстка сайту або він недоступний — перевірте парсер."
        )

    async def _check_source_windows(
        self,
        stats: PipelineStats,
        source_pairs: list[tuple[ScraperSource, SourceConfig]],
        new_articles: list[Article],
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """Flag Sources that returned their full window, none of it already in storage.

        A listing page shows only its newest ``max_articles`` Articles; if more than
        that appeared since the last Run, the older ones were never seen. An Error
        Article being retried counts as new but proves the window overlapped a Run,
        so storage is asked once, and only when some window looks all new.
        """
        candidates = {
            source.name: src.max_articles
            for source, src in source_pairs
            if stats.source_scraped.get(source.name, 0) == src.max_articles
            and stats.source_new.get(source.name, 0) == src.max_articles
        }
        if not candidates:
            return
        window_ids = [a.id for a in new_articles if a.source_name in candidates]
        stored = await self.storage.existing_ids(window_ids)
        seen_sources = {a.source_name for a in new_articles if a.id in stored}
        stats.source_window_full = {
            name: size for name, size in candidates.items() if name not in seen_sources
        }
        if stats.source_window_full:
            log.warning("source_window_full", sources=stats.source_window_full)

    async def _record_repeat(
        self,
        article: Article,
        original: Original,
        stats: PipelineStats,
        dry_run: bool,
    ) -> None:
        """Settle an Article whose Story the AI already judged, without asking it again."""
        log = logger.bind(article_id=article.id, category=article.category.value)
        log.info("repeat", title=article.title_pl, original_id=original.article_id)
        stats.repeats += 1
        stats.article_log.append(
            ArticleRunEntry(
                article_id=article.id,
                title_pl=article.title_pl,
                ua_title=None,
                score=None,
                decision=Decision.REPEAT,
                source_name=article.source_name,
                article_url=article.url,
                original=original,
            )
        )
        if not dry_run:
            await self._save_decision_with_retry(
                article=article,
                decision=Decision.REPEAT,
                ai_decision=None,
                tg_message_id=None,
                tg_events_message_id=None,
                stats=stats,
                log=log,
                repeat_of=original.article_id,
            )

    async def _process_article(
        self,
        article: Article,
        stats: PipelineStats,
        dry_run: bool,
        stories: StoryIndex,
        near_match: NearMatch | None,
    ) -> StopSignal:
        """Process a single article: AI evaluate → maybe publish → save.

        Routing logic:
          - is_interesting + is_event + events_publisher configured
              → post to main channel first, then to events channel
          - is_interesting + (not is_event OR events_publisher not configured)
              → post to main channel only
          - not is_interesting → skip both channels

        Main channel failure IS fatal for the article (marked as ERROR), and the
        events channel is never called, so the next attempt starts clean.
        Events channel failure is non-fatal: the article is still Published, the
        failure goes into the run report, and the events Post is never retried —
        not in-process (a retry after an unanswered request could duplicate it) and
        not on later runs (the article is Settled).

        Failure modes fall into two groups:
          - Not the article's fault (any DeferralError): the article is Deferred.
            Nothing is written to storage, so it resurfaces untouched on the next run;
            the error's stop signal may end the run early.
          - Everything else: recorded as ERROR with an incremented attempt counter.
            Storage keeps returning it as "new" until the retry budget runs out, so a
            one-off network blip cannot retire an article permanently.
        """
        ai_decision: AIDecision | None = None
        decision = Decision.ERROR
        tg_message_id: int | None = None
        tg_events_message_id: int | None = None
        posted_to_events = False
        events_error: str | None = None
        error_msg: str | None = None
        # Set once the Publishing record is written: what to restore on a Deferral
        marked_publishing = False
        previous_record: PriorRecord | None = None

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
            should_publish = (
                ai_decision.is_interesting and ai_decision.score >= self.settings.ai_min_score
            )
            route_to_events = (
                should_publish and ai_decision.is_event and self.events_publisher is not None
            )

            if should_publish:
                if dry_run:
                    channels = ["main", "events"] if route_to_events else ["main"]
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
                    # Write-ahead record: if delivery turns out uncertain, it stays
                    # Publishing and the Article is never posted again (ADR 0001).
                    try:
                        previous_record = await self.storage.mark_publishing(article)
                    except Exception as exc:
                        raise PublishingRecordError(f"{type(exc).__name__}: {exc}") from exc
                    marked_publishing = True

                    # Main channel first — failure IS fatal for the article
                    result = await self.publisher.publish(article, ai_decision)
                    tg_message_id = result.message_id
                    stats.posted += 1
                    decision = Decision.POSTED

                    # Events channel after it — failure is non-fatal, never retried
                    if route_to_events:
                        assert self.events_publisher is not None  # narrowing for type checker
                        try:
                            events_result = await self.events_publisher.publish_once(
                                article, ai_decision
                            )
                            tg_events_message_id = events_result.message_id
                            posted_to_events = True
                            log.info(
                                "published_to_events_channel",
                                tg_events_message_id=tg_events_message_id,
                            )
                        except Exception as events_exc:
                            events_error = f"{type(events_exc).__name__}: {events_exc}"
                            log.error("events_channel_publish_failed", error=str(events_exc))

                    log.info(
                        "published",
                        score=ai_decision.score,
                        tg_message_id=tg_message_id,
                        ua_title=ai_decision.ua_title,
                        posted_to_events=posted_to_events,
                    )
            else:
                log.info("skipped", score=ai_decision.score, reason=ai_decision.reason)
                stats.skipped += 1
                decision = Decision.SKIPPED

        except DeferralError as exc:
            # Not the article's fault: do NOT save — it won't be in storage, so
            # filter_new_ids returns it again on the next run.
            log.warning(
                "article_deferred",
                error_type=type(exc).__name__,
                error=str(exc),
                stop_signal=exc.stop_signal.value,
            )
            if marked_publishing:
                # The Post definitely did not go out: storage goes back to exactly
                # its state before this Run, attempt count included.
                await self._release_publishing(article, previous_record, log)
            stats.deferred += 1
            stats.article_log.append(
                ArticleRunEntry(
                    article_id=article.id,
                    title_pl=article.title_pl,
                    ua_title=ai_decision.ua_title if ai_decision else None,
                    score=ai_decision.score if ai_decision else None,
                    decision=Decision.DEFERRED,
                    error_msg=exc.report_message,
                    report_icon=exc.report_icon,
                    source_name=article.source_name,
                    article_url=article.url,
                    ai_reason=ai_decision.reason if ai_decision else None,
                    ai_ua_summary=ai_decision.ua_summary if ai_decision else None,
                    near_match=near_match,
                )
            )
            return exc.stop_signal

        except TelegramDeliveryUnknownError as exc:
            # The Post may be in the Main channel: leave the record Publishing, never
            # send again, and stop — an unhealthy Telegram would only pile up more.
            log.error("main_channel_delivery_unknown", error=str(exc))
            stories.add(_original(article))
            stats.article_log.append(
                ArticleRunEntry(
                    article_id=article.id,
                    title_pl=article.title_pl,
                    ua_title=ai_decision.ua_title if ai_decision else None,
                    score=ai_decision.score if ai_decision else None,
                    decision=Decision.PUBLISHING,
                    error_msg=f"delivery unknown, check the main channel ({exc})",
                    source_name=article.source_name,
                    article_url=article.url,
                    ai_reason=ai_decision.reason if ai_decision else None,
                    ai_ua_summary=ai_decision.ua_summary if ai_decision else None,
                    near_match=near_match,
                )
            )
            return StopSignal.PUBLISHING_BLOCKED

        except Exception as exc:
            log.exception("article_error", error=str(exc))
            stats.errors += 1
            decision = Decision.ERROR
            error_msg = f"{type(exc).__name__}: {exc}"

        if decision == Decision.POSTED and ai_decision is not None and ai_decision.is_event:
            stats.events_posted += 1
            if ai_decision.event_start is None:
                stats.events_without_date += 1

        # The AI judged it: later Articles about the same Story are Repeats. An Error
        # is never an original, so its Story is not lost if the Article failed.
        if decision in (Decision.POSTED, Decision.SKIPPED):
            stories.add(_original(article))

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
                events_error=events_error,
                near_match=near_match,
            )
        )

        # Stage 3c: Save result so we don't retry — skipped in dry_run (no side effects)
        if not dry_run:
            await self._save_decision_with_retry(
                article=article,
                decision=decision,
                ai_decision=ai_decision,
                tg_message_id=tg_message_id,
                tg_events_message_id=tg_events_message_id,
                stats=stats,
                log=log,
            )

        # Signal cap after the post is saved so the article is not retried next run
        cap = self.flow_config.pipeline.max_posts_per_run
        if decision == Decision.POSTED and stats.posted >= cap:
            return StopSignal.CAP_REACHED

        return StopSignal.CONTINUE

    async def _release_publishing(
        self,
        article: Article,
        previous: PriorRecord | None,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """Undo the Publishing record of a Deferred Article.

        If this fails too, the record stays Publishing and the next Run turns it
        Unconfirmed: the Article is lost to automation, but never posted twice.
        """
        try:
            await self.storage.release_publishing(article.id, previous)
        except Exception as exc:
            log.error("release_publishing_failed", error=str(exc))

    async def _save_decision_with_retry(
        self,
        *,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None,
        tg_message_id: int | None,
        tg_events_message_id: int | None,
        stats: PipelineStats,
        log: structlog.stdlib.BoundLogger,
        repeat_of: str | None = None,
    ) -> None:
        """Persist the outcome, retrying transient storage failures.

        After a Post the record is Publishing, so even a lost write cannot cause a
        Duplicate: the next Run turns it Unconfirmed. Retrying still matters, because
        an Unconfirmed Article needs the admin to check the channel by hand.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _SAVE_MAX_ATTEMPTS + 1):
            try:
                await self.storage.save_decision(
                    article=article,
                    decision=decision,
                    ai_decision=ai_decision,
                    tg_message_id=tg_message_id,
                    tg_events_message_id=tg_events_message_id,
                    repeat_of=repeat_of,
                )
                return
            except Exception as save_exc:
                last_exc = save_exc
                log.warning("save_failed", error=str(save_exc), attempt=attempt)
                if attempt < _SAVE_MAX_ATTEMPTS:
                    await asyncio.sleep(_SAVE_RETRY_DELAY_S * attempt)

        log.error("save_failed_permanently", error=str(last_exc), article_id=article.id)
        if decision == Decision.POSTED:
            stats.alerts.append(
                f"Пост опубліковано, але результат не збережено в БД: {article.url} "
                f"({type(last_exc).__name__}: {last_exc}). "
                "Наступний запуск позначить статтю як непідтверджену. "
                "Повторної публікації не буде."
            )
        else:
            stats.alerts.append(
                f"Не вдалося зберегти результат обробки {article.url} "
                f"({type(last_exc).__name__}: {last_exc})."
            )
