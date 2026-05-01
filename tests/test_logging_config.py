"""Tests for logging configuration."""

from unittest.mock import patch

import pytest
import structlog

from rz_flow.logging_config import (
    _PrettyRenderer,
    _drop_ai_evaluated,
    configure_logging,
)


# ── configure_logging smoke tests ─────────────────────────────────────────────


def test_configure_logging_production() -> None:
    """configure_logging(production) runs without error and sets up structlog."""
    configure_logging("production")
    log = structlog.get_logger("test")
    log.info("test_event", env="production")


def test_configure_logging_development() -> None:
    """configure_logging(development) uses ConsoleRenderer."""
    configure_logging("development")
    log = structlog.get_logger("test")
    log.info("test_event", env="development")


def test_configure_logging_default_is_production() -> None:
    """Default environment is production."""
    configure_logging()
    log = structlog.get_logger("test")
    log.info("test_event")


def test_configure_logging_uses_pretty_renderer_when_tty() -> None:
    """When stdout is a TTY, structlog is configured with _PrettyRenderer (not JSON)."""
    with patch("rz_flow.logging_config.sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = True
        configure_logging("production")

    config = structlog.get_config()
    processor_types = [type(p) for p in config["processors"]]
    assert _PrettyRenderer in processor_types, (
        "Expected _PrettyRenderer in processors for TTY, got: "
        + str([t.__name__ for t in processor_types])
    )
    assert structlog.processors.JSONRenderer not in processor_types
    # Restore a valid non-TTY config so subsequent tests get a working logger
    configure_logging("production")


def test_configure_logging_uses_json_renderer_when_not_tty() -> None:
    """When stdout is not a TTY, structlog is configured with JSONRenderer (not PrettyRenderer)."""
    with patch("rz_flow.logging_config.sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = False
        configure_logging("production")

    config = structlog.get_config()
    processor_types = [type(p) for p in config["processors"]]
    assert structlog.processors.JSONRenderer in processor_types, (
        "Expected JSONRenderer in processors for non-TTY, got: "
        + str([t.__name__ for t in processor_types])
    )
    assert _PrettyRenderer not in processor_types
    # Restore a valid config so subsequent tests get a working logger
    configure_logging("production")


# ── _drop_ai_evaluated ────────────────────────────────────────────────────────


class TestDropAiEvaluated:
    def test_drops_ai_evaluated_event(self) -> None:
        """ai_evaluated events are dropped (they're too noisy for TTY output)."""
        with pytest.raises(structlog.DropEvent):
            _drop_ai_evaluated(None, "info", {"event": "ai_evaluated", "score": 8.5})

    def test_passes_through_other_events(self) -> None:
        event_dict = {"event": "published", "ua_title": "Test"}
        result = _drop_ai_evaluated(None, "info", event_dict)
        assert result["event"] == "published"

    def test_passes_through_warning_events(self) -> None:
        event_dict = {"event": "rate_limited", "retry_after": 10}
        result = _drop_ai_evaluated(None, "warning", event_dict)
        assert result is event_dict


# ── _PrettyRenderer ───────────────────────────────────────────────────────────


class TestPrettyRenderer:
    """Direct unit tests of _PrettyRenderer.__call__ / _format.

    We instantiate _PrettyRenderer directly instead of going through structlog
    to keep tests simple and fast.
    """

    def _call(self, event: str, level: str = "info", **ctx: object) -> str:
        renderer = _PrettyRenderer()
        event_dict = {
            "level": level,
            "timestamp": "2026-05-01T12:00:00.000000Z",
            "event": event,
            **ctx,
        }
        return renderer(None, level, event_dict)

    # ── pipeline_started ──────────────────────────────────────────────────────

    def test_pipeline_started_contains_pipeline_header(self) -> None:
        result = self._call(
            "pipeline_started",
            sources=[],
            model="gemini-2.0-flash",
            min_score=7.0,
        )
        assert "Pipeline" in result

    def test_pipeline_started_lists_source_names(self) -> None:
        sources = [
            {"name": "rzeszow24/najnowsze", "base_url": "https://rzeszow24.info/najnowsze", "max_articles": 5}
        ]
        result = self._call("pipeline_started", sources=sources, model="m", min_score=7)
        assert "rzeszow24/najnowsze" in result

    def test_pipeline_started_empty_sources(self) -> None:
        result = self._call("pipeline_started", sources=[], model="gemini", min_score=7)
        assert "Pipeline" in result

    # ── scrape_* events ───────────────────────────────────────────────────────

    def test_scrape_started_shows_source_count(self) -> None:
        result = self._call("scrape_started", count=2)
        assert "2" in result

    def test_scrape_source_start_shows_name(self) -> None:
        result = self._call("scrape_source_start", name="rzeszow24/najnowsze", url="https://rzeszow24.info")
        assert "rzeszow24/najnowsze" in result

    def test_scrape_source_done_shows_found_count(self) -> None:
        result = self._call("scrape_source_done", name="rzeszow24/najnowsze", found=8)
        assert "8" in result

    def test_scrape_done_shows_total(self) -> None:
        result = self._call("scrape_done", total=15)
        assert "15" in result

    # ── filter_complete ───────────────────────────────────────────────────────

    def test_filter_complete_shows_new_count(self) -> None:
        result = self._call("filter_complete", new=3, seen=7)
        assert "3" in result

    def test_filter_complete_hides_seen_when_zero(self) -> None:
        result = self._call("filter_complete", new=5, seen=0)
        assert "already seen" not in result

    # ── no_new_articles / no_articles_found ──────────────────────────────────

    def test_no_new_articles_shows_nothing_to_do(self) -> None:
        result = self._call("no_new_articles")
        assert "Nothing to do" in result

    def test_no_articles_found_shows_nothing_to_do(self) -> None:
        result = self._call("no_articles_found")
        assert "Nothing to do" in result

    # ── published ────────────────────────────────────────────────────────────

    def test_published_shows_ua_title(self) -> None:
        result = self._call(
            "published",
            ua_title="Фестиваль у Жешові",
            category="imprezy",
            score=8.5,
            tg_message_id=42,
        )
        assert "Фестиваль у Жешові" in result

    def test_published_shows_message_id(self) -> None:
        result = self._call(
            "published",
            ua_title="Title",
            category="imprezy",
            score=8.0,
            tg_message_id=99,
        )
        assert "msg#99" in result

    def test_published_truncates_long_title(self) -> None:
        long_title = "А" * 100
        result = self._call(
            "published",
            ua_title=long_title,
            category="imprezy",
            score=8.0,
            tg_message_id=1,
        )
        assert "…" in result

    # ── dry_run_would_publish ─────────────────────────────────────────────────

    def test_dry_run_would_publish_shows_dry_run_label(self) -> None:
        result = self._call(
            "dry_run_would_publish",
            ua_title="Нова подія",
            category="imprezy",
            score=8.0,
        )
        assert "DRY RUN" in result

    # ── ai_processing ─────────────────────────────────────────────────────────

    def test_ai_processing_shows_title(self) -> None:
        result = self._call(
            "ai_processing",
            title="Festiwal Muzyczny w Rzeszowie",
            category="imprezy",
        )
        assert "Festiwal" in result

    # ── skipped ──────────────────────────────────────────────────────────────

    def test_skipped_shows_score_and_reason(self) -> None:
        result = self._call("skipped", category="imprezy", score=3.0, reason="Criminal news")
        assert "score=3.0" in result
        assert "Criminal" in result

    # ── pipeline_complete ─────────────────────────────────────────────────────

    def test_pipeline_complete_shows_all_counters(self) -> None:
        result = self._call(
            "pipeline_complete", posted=5, skipped=2, errors=0, elapsed_s=30
        )
        assert "posted=5" in result
        assert "skipped=2" in result
        assert "errors=0" in result

    def test_pipeline_complete_shows_elapsed_time(self) -> None:
        result = self._call(
            "pipeline_complete", posted=1, skipped=0, errors=0, elapsed_s=75
        )
        # 75s = 1m 15s
        assert "1m" in result

    def test_pipeline_complete_elapsed_seconds_only_for_under_minute(self) -> None:
        result = self._call(
            "pipeline_complete", posted=1, skipped=0, errors=0, elapsed_s=45
        )
        assert "45s" in result

    # ── gemini_unavailable_skipping ───────────────────────────────────────────

    def test_gemini_unavailable_skipping_shows_warning(self) -> None:
        result = self._call("gemini_unavailable_skipping", error="503 Service Unavailable")
        assert "Gemini 503" in result

    # ── quota_exhausted_stopping ──────────────────────────────────────────────

    def test_quota_exhausted_stopping_shows_counts(self) -> None:
        result = self._call("quota_exhausted_stopping", processed=3, remaining=7)
        assert "3" in result
        assert "7" in result

    # ── db_initialized ────────────────────────────────────────────────────────

    def test_db_initialized_shows_database_label(self) -> None:
        result = self._call("db_initialized")
        assert "Database" in result

    # ── generic fallback cases ────────────────────────────────────────────────

    def test_error_level_generic_event_shows_event_name(self) -> None:
        result = self._call("something_failed", level="error", error="oops")
        assert "something_failed" in result

    def test_warning_level_generic_event_shows_event_name(self) -> None:
        result = self._call("rate_limited", level="warning", retry_after=30)
        assert "rate_limited" in result

    def test_info_level_unknown_event_generic_fallback(self) -> None:
        result = self._call("some_unknown_event", key="value")
        assert "some_unknown_event" in result

    # ── __call__ internals ────────────────────────────────────────────────────

    def test_dry_run_key_not_in_output(self) -> None:
        """dry_run context key is silently removed before rendering."""
        result = self._call(
            "pipeline_started",
            dry_run=True,
            sources=[],
            model="m",
            min_score=7,
        )
        assert "dry_run" not in result

    def test_exc_info_appended_below_main_line(self) -> None:
        renderer = _PrettyRenderer()
        event_dict = {
            "level": "error",
            "timestamp": "2026-05-01T12:00:00Z",
            "event": "crash",
            "exception": "Traceback (most recent call last):\n  File x.py\nValueError: oops",
        }
        result = renderer(None, "error", event_dict)
        assert "ValueError" in result
        assert "\n" in result  # exc_info is on separate indented lines

    def test_timestamp_prefix_uses_time_portion(self) -> None:
        result = self._call("published", ua_title="T", category="c", score=8.0, tg_message_id=1)
        # Timestamp "2026-05-01T12:00:00.000000Z" → time portion "12:00:00"
        assert "12:00:00" in result

    def test_article_id_shown_in_error_level(self) -> None:
        renderer = _PrettyRenderer()
        event_dict = {
            "level": "error",
            "timestamp": "2026-05-01T12:00:00Z",
            "event": "article_error",
            "article_id": "rz24/some-article-id",
            "error": "AI failed",
        }
        result = renderer(None, "error", event_dict)
        assert "article_id" in result

    def test_article_id_hidden_in_info_level(self) -> None:
        """article_id is dropped for info-level events to reduce noise."""
        renderer = _PrettyRenderer()
        event_dict = {
            "level": "info",
            "timestamp": "2026-05-01T12:00:00Z",
            "event": "ai_processing",
            "article_id": "rz24/some-article-id",
            "title": "Test Title",
            "category": "imprezy",
        }
        result = renderer(None, "info", event_dict)
        assert "article_id" not in result
