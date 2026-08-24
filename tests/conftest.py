"""Shared pytest fixtures and helpers."""

import pathlib
from collections.abc import Iterator
from typing import Final

import pytest
from tenacity import wait_none

FIXTURES_DIR: Final = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_retry_backoff() -> Iterator[None]:
    """Collapse tenacity backoff to zero so retry tests don't actually sleep.

    Waiting is configured on the decorator at import time, so it has to be swapped on
    the retry object itself rather than patched at the module level.
    """
    from rz_flow.ai import GeminiAIFilter
    from rz_flow.scraper import _fetch_one
    from rz_flow.telegram import TelegramPublisher

    retryables = [GeminiAIFilter.evaluate, TelegramPublisher.publish, _fetch_one]
    original = [fn.retry.wait for fn in retryables]  # type: ignore[attr-defined]
    for fn in retryables:
        fn.retry.wait = wait_none()  # type: ignore[attr-defined]
    yield
    for fn, wait in zip(retryables, original, strict=True):
        fn.retry.wait = wait  # type: ignore[attr-defined]


@pytest.fixture
def najnowsze_html() -> str:
    return (FIXTURES_DIR / "najnowsze_sample.html").read_text(encoding="utf-8")


@pytest.fixture
def rzeszow_news_html() -> str:
    return (FIXTURES_DIR / "rzeszow_news_sample.html").read_text(encoding="utf-8")


# ── Archived fixtures (used by archived parser tests) ─────────────────────────


@pytest.fixture
def imprezy_html() -> str:
    return (FIXTURES_DIR / "imprezy_sample.html").read_text(encoding="utf-8")


@pytest.fixture
def wiadomosci_html() -> str:
    return (FIXTURES_DIR / "wiadomosci_sample.html").read_text(encoding="utf-8")
