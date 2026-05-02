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

    articles, _ = await fetch_articles(_make_settings(), _make_flow_config())

    assert len(articles) > 0


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_deduplicates() -> None:
    """If the same article ID appears in multiple sources, it's returned once."""
    najnowsze = (FIXTURES_DIR / "najnowsze_sample.html").read_text()
    rn = (FIXTURES_DIR / "rzeszow_news_sample.html").read_text()
    _mock_all_sources(najnowsze, rn)

    articles, _ = await fetch_articles(_make_settings(), _make_flow_config())
    ids = [a.id for a in articles]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_continues_when_one_source_fails() -> None:
    """A 503 from one source is logged and skipped; the other source still runs."""
    rn = (FIXTURES_DIR / "rzeszow_news_sample.html").read_text()
    respx.get("https://rzeszow24.info/najnowsze").mock(return_value=Response(503))
    respx.get("https://rzeszow-news.pl/").mock(return_value=Response(200, text=rn))

    # Should NOT raise — partial success is acceptable
    articles, source_scraped = await fetch_articles(_make_settings(), _make_flow_config())
    # Articles from rzeszow-news.pl are returned even though rzeszow24 failed
    assert len(articles) > 0
    # All returned IDs carry the rzn/ prefix from RzeszowNewsScraper
    assert all(a.id.startswith("rzn/") for a in articles)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_ids_carry_source_prefix() -> None:
    """Article IDs are prefixed with the source identifier to prevent cross-source collisions."""
    najnowsze = (FIXTURES_DIR / "najnowsze_sample.html").read_text()
    rn = (FIXTURES_DIR / "rzeszow_news_sample.html").read_text()
    _mock_all_sources(najnowsze, rn)

    articles, _ = await fetch_articles(_make_settings(), _make_flow_config())

    rz24_articles = [a for a in articles if a.id.startswith("rz24/")]
    rzn_articles = [a for a in articles if a.id.startswith("rzn/")]
    assert len(rz24_articles) > 0, "Expected articles with rz24/ prefix from NajnowszeScraper"
    assert len(rzn_articles) > 0, "Expected articles with rzn/ prefix from RzeszowNewsScraper"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_respects_max_articles_limit() -> None:
    """max_articles per source limits how many articles are returned."""
    najnowsze = (FIXTURES_DIR / "najnowsze_sample.html").read_text()
    rn = (FIXTURES_DIR / "rzeszow_news_sample.html").read_text()
    _mock_all_sources(najnowsze, rn)

    articles, _ = await fetch_articles(_make_settings(), _make_flow_config(max_articles=3))

    # 3 per source × 2 sources = 6 max (assuming no cross-source deduplication)
    assert len(articles) <= 6
