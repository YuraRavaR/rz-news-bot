"""Scraper sources for rzeszow24.info.

Active source
-------------
NajnowszeScraper — scrapes /najnowsze (all-category latest news feed).

Archived sources (kept for reference, not in ACTIVE_SOURCES)
-------------------------------------------------------------
CategoryPageScraper — scrapes /imprezy/ and /wiadomosci/ category pages.
  These pages update infrequently and tend to surface older articles,
  so they were replaced by the unified /najnowsze feed.
"""

from __future__ import annotations

import httpx

from rz_flow.models import Article, Category
from rz_flow.parser import parse_category_page, parse_najnowsze_page


class NajnowszeScraper:
    """Fetches the /najnowsze (latest) feed from rzeszow24.info."""

    name = "rzeszow24/najnowsze"
    ID_PREFIX = "rz24"

    def __init__(self, base_url: str = "https://rzeszow24.info/najnowsze") -> None:
        self.url = base_url

    async def fetch(
        self,
        client: httpx.AsyncClient,
        max_articles: int,
    ) -> list[Article]:
        response = await client.get(self.url)
        response.raise_for_status()
        articles = parse_najnowsze_page(response.text)[:max_articles]
        return [a.model_copy(update={"id": f"{self.ID_PREFIX}/{a.id}"}) for a in articles]


# ── Archived ──────────────────────────────────────────────────────────────────


class CategoryPageScraper:
    """Scrapes explicit category pages: /imprezy/ and /wiadomosci/.

    Archived — replaced by NajnowszeScraper which covers both categories
    from a single, more frequently updated endpoint.
    """

    name = "rzeszow24/categories"

    _CATEGORIES: list[Category] = [Category.IMPREZY, Category.WIADOMOSCI]

    def __init__(self, base_url: str = "https://rzeszow24.info") -> None:
        self._base_url = base_url

    async def fetch(
        self,
        client: httpx.AsyncClient,
        max_articles: int,
    ) -> list[Article]:
        all_articles: list[Article] = []
        seen_ids: set[str] = set()

        for category in self._CATEGORIES:
            url = f"{self._base_url}/{category.value}/"
            response = await client.get(url)
            response.raise_for_status()

            parsed = parse_category_page(response.text, category)
            new = [a for a in parsed if a.id not in seen_ids]
            seen_ids.update(a.id for a in new)
            all_articles.extend(new[:max_articles])

        return all_articles
