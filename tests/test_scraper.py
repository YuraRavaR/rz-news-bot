"""Tests for the async scraper (mocked HTTP with respx).

Key AQA lesson: we mock the HTTP layer with respx so the test:
  - runs offline (no real network)
  - is fast and deterministic
  - tests the integration between scraper + parser
"""

import pathlib

import pytest
import respx
from httpx import Response

from rz_flow.config import Settings
from rz_flow.models import Category
from rz_flow.scraper import fetch_articles

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _make_settings() -> Settings:
    """Build minimal settings with fakes for required secret fields."""
    return Settings(
        telegram_bot_token="fake:token",
        telegram_channel_id="-100123",
        gemini_api_key="fake-key",
        turso_database_url="libsql://fake.turso.io",
        turso_auth_token="fake-token",
    )


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_returns_both_categories() -> None:
    """Scraper fetches imprezy and wiadomosci, returns articles from both."""
    imprezy_html = (FIXTURES_DIR / "imprezy_sample.html").read_text()
    wiadomosci_html = (FIXTURES_DIR / "wiadomosci_sample.html").read_text()

    respx.get("https://rzeszow24.info/imprezy/").mock(return_value=Response(200, text=imprezy_html))
    respx.get("https://rzeszow24.info/wiadomosci/").mock(
        return_value=Response(200, text=wiadomosci_html)
    )

    settings = _make_settings()
    articles = await fetch_articles(settings)

    assert len(articles) > 0

    categories = {a.category for a in articles}
    assert Category.IMPREZY in categories
    assert Category.WIADOMOSCI in categories


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_deduplicates_across_categories() -> None:
    """If the same article ID appears in two responses, it's returned once."""
    # Use the same HTML for both endpoints (simulates duplicated content)
    imprezy_html = (FIXTURES_DIR / "imprezy_sample.html").read_text()

    respx.get("https://rzeszow24.info/imprezy/").mock(return_value=Response(200, text=imprezy_html))
    respx.get("https://rzeszow24.info/wiadomosci/").mock(
        return_value=Response(200, text=imprezy_html)
    )

    articles = await fetch_articles(_make_settings())
    ids = [a.id for a in articles]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_raises_on_http_error() -> None:
    """scraper propagates HTTP errors so the pipeline can handle them."""
    respx.get("https://rzeszow24.info/imprezy/").mock(return_value=Response(503))
    respx.get("https://rzeszow24.info/wiadomosci/").mock(
        return_value=Response(200, text="<html></html>")
    )

    with pytest.raises(Exception):
        await fetch_articles(_make_settings())


@pytest.mark.asyncio
@respx.mock
async def test_fetch_articles_respects_max_articles_limit() -> None:
    """scraper_max_articles setting limits how many articles are returned per category."""
    imprezy_html = (FIXTURES_DIR / "imprezy_sample.html").read_text()
    wiadomosci_html = (FIXTURES_DIR / "wiadomosci_sample.html").read_text()

    respx.get("https://rzeszow24.info/imprezy/").mock(return_value=Response(200, text=imprezy_html))
    respx.get("https://rzeszow24.info/wiadomosci/").mock(
        return_value=Response(200, text=wiadomosci_html)
    )

    settings = _make_settings()
    settings = settings.model_copy(update={"scraper_max_articles": 3})
    articles = await fetch_articles(settings)

    # At most 3 imprezy + 3 wiadomosci = 6 total
    assert len(articles) <= 6
