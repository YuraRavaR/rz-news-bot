"""Scraper source for rzeszow24.info.

NajnowszeScraper scrapes /najnowsze, the all-category latest news feed.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from rz_flow.models import Article
from rz_flow.parser import parse_najnowsze_page


class NajnowszeScraper:
    """Fetches the /najnowsze (latest) feed from rzeszow24.info."""

    name = "rzeszow24/najnowsze"
    ID_PREFIX = "rz24"

    def __init__(self, base_url: str = "https://rzeszow24.info/najnowsze") -> None:
        self.url = base_url

    def slug(self, url: str) -> str | None:
        """The path segment before the trailing ID: /{category}/{slug}/{ID}.

        rzeszow24 republishes a Story under a new ID, often keeping the slug.
        """
        parsed = urlparse(url)
        if not parsed.netloc.endswith("rzeszow24.info"):
            return None
        segments = [s for s in parsed.path.split("/") if s]
        return segments[-2] if len(segments) >= 3 else None

    async def fetch(
        self,
        client: httpx.AsyncClient,
        max_articles: int,
    ) -> list[Article]:
        response = await client.get(self.url)
        response.raise_for_status()
        articles = parse_najnowsze_page(response.text)[:max_articles]
        return [a.model_copy(update={"id": f"{self.ID_PREFIX}/{a.id}"}) for a in articles]
