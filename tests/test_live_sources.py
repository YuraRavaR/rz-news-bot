"""Canary tests against the real news sites.

Deselected by default (`-m 'not live'` in pyproject) and run on a schedule instead.

Why these exist: every other parser test runs against HTML fixtures captured in May
2026. When a source changes its markup, those fixtures keep passing while production
silently returns zero articles and the channel just stops updating. Only a request to
the live site catches that.

A failure here means the parser needs updating, not that the code regressed. Refresh
the fixtures in tests/fixtures/ from the live page while you fix it.

Run manually:
    uv run pytest -m live --no-cov
"""

from collections.abc import AsyncIterator

import httpx
import pytest

from rz_flow.flow_config import load_flow_config
from rz_flow.scraper import _HEADERS
from rz_flow.sources import ScraperSource, get_active_sources

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

_TIMEOUT = 30.0

_SOURCES = get_active_sources(load_flow_config())
_SOURCE_IDS = [s.name for s in _SOURCES]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers=_HEADERS,
        http2=True,
    ) as c:
        yield c


@pytest.mark.parametrize("source", _SOURCES, ids=_SOURCE_IDS)
async def test_source_still_yields_articles(
    source: ScraperSource, client: httpx.AsyncClient
) -> None:
    """The live page must still parse into at least one article.

    Zero articles means either the site is blocking us or its markup changed —
    both are silent content loss in production.
    """
    articles = await source.fetch(client, max_articles=5)

    assert articles, (
        f"{source.name} returned 0 articles from {source.url} — "
        "the page layout probably changed, or the request was blocked"
    )


@pytest.mark.parametrize("source", _SOURCES, ids=_SOURCE_IDS)
async def test_parsed_articles_are_usable(
    source: ScraperSource, client: httpx.AsyncClient
) -> None:
    """Parsing can 'succeed' while producing empty titles or broken URLs."""
    articles = await source.fetch(client, max_articles=5)

    for article in articles:
        assert article.title_pl.strip(), f"empty title for {article.url}"
        assert article.url.startswith("http"), f"malformed URL: {article.url}"
        assert "/" in article.id, f"missing source prefix in ID: {article.id}"
