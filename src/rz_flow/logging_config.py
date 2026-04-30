"""Structlog configuration.

Structlog produces structured JSON logs in production (GitHub Actions)
and human-friendly colorized logs in development.

Why structlog over stdlib logging:
  - All log entries are key-value pairs → easy to grep, parse, send to Loki
  - Context binding: add article_id once, appears in all subsequent log lines
  - Same API in prod and dev, just different renderers

Usage:
    import structlog
    log = structlog.get_logger(__name__)
    log.info("article_processed", article_id="abc", score=8.5)
"""

import logging
import sys

import structlog


def configure_logging(app_env: str = "production") -> None:
    """Configure structlog for the given environment.

    Call once at application startup (from main.py).
    """
    is_dev = app_env == "development"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_dev:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,  # allow reconfiguration in tests
    )

    # Also configure stdlib logging (httpx, google-genai use it)
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        level=logging.WARNING,
    )
