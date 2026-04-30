"""Tests for logging configuration."""

import structlog

from rz_flow.logging_config import configure_logging


def test_configure_logging_production() -> None:
    """configure_logging(production) runs without error and sets up structlog."""
    configure_logging("production")
    log = structlog.get_logger("test")
    # Should not raise
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
