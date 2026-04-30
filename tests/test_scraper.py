"""Tests for the async scraper (mocked HTTP with respx).

Key AQA lesson: we mock the HTTP layer with respx so the test:
  - runs offline (no real network)
  - is fast and deterministic
  - tests the integration between scraper + parser + sources
"""

import pathlib

import pytest
import respx
from httpx import Response

from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig, PipelineConfig, SourceConfig
from rz_flow.scraper import fetch_articles

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _make_settings(**overrides) -> Settings:
    """Build minimal settings with fakes for required secret fields."""
    defaults = dict(
        telegram_bot_token="fake:token",
        telegram_channel_id="-100123",
        gemini_api_key="fake-key",
        turso_database_url="libsql://fake.turso.io",
        turso_auth_token="fake-token",
    )
    return Settings(**{**defaults, **overrides})


def _make_flow_config(max_articles: int = 10) -> FlowConfig:
    """Build a FlowConfig matching the mocked URLs used in tests."""
    return FlowConfig(
        sources=[
            SourceConfig(
                scraper="NajnowszeScraper",
                base_url="https://rzeszow24.info/najnowsze",
                max_articles=max_articles,
            ),
            SourceConfig(
                scraper="RzeszowNewsScraper",
                base_url="https://rzeszow-news.pl/",
                max_articles=max_articles,
            ),
        ],
        pipeline=PipelineConfig(),
    )


def _mock_all_sources(najnowsze_html: str, rzeszow_news_html: str) -> None:
    """Register respx mocks for all active sources."""
    respx.get("https://rzeszow24.info/najnowsze").mock(
        return_value=Response(200, text=najnowsze_html)
    )
    respx.get("https://rzeszow-news.pl/").mock(return_value=Response(200, text=rzeszow_news_html))


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_returns_articles() -> None:
    """Scraper fetches all active sources and returns a non-empty article list."""
    najnowsze = (FIXTURES_DIR / "najnowsze_sample.html").read_text()
    rn = (FIXTURES_DIR / "rzeszow_news_sample.html").read_text()
    _mock_all_sources(najnowsze, rn)

    articles = await fetch_articles(_make_settings(), _make_flow_config())

    assert len(articles) > 0


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_deduplicates() -> None:
    """If the same article ID appears in multiple sources, it's returned once."""
    najnowsze = (FIXTURES_DIR / "najnowsze_sample.html").read_text()
    rn = (FIXTURES_DIR / "rzeszow_news_sample.html").read_text()
    _mock_all_sources(najnowsze, rn)

    articles = await fetch_articles(_make_settings(), _make_flow_config())
    ids = [a.id for a in articles]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_raises_on_http_error() -> None:
    """Scraper propagates HTTP errors so the pipeline can handle them."""
    respx.get("https://rzeszow24.info/najnowsze").mock(return_value=Response(503))
    respx.get("https://rzeszow-news.pl/").mock(return_value=Response(200, text="<html></html>"))

    with pytest.raises(Exception):
        await fetch_articles(_make_settings(), _make_flow_config())


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_respects_max_articles_limit() -> None:
    """max_articles per source limits how many articles are returned."""
    najnowsze = (FIXTURES_DIR / "najnowsze_sample.html").read_text()
    rn = (FIXTURES_DIR / "rzeszow_news_sample.html").read_text()
    _mock_all_sources(najnowsze, rn)

    articles = await fetch_articles(_make_settings(), _make_flow_config(max_articles=3))

    # 3 per source × 2 sources = 6 max (assuming no cross-source deduplication)
    assert len(articles) <= 6
