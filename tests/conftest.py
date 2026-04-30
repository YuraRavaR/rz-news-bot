"""Shared pytest fixtures and helpers."""

import pathlib
from typing import Final

import pytest

FIXTURES_DIR: Final = pathlib.Path(__file__).parent / "fixtures"


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
