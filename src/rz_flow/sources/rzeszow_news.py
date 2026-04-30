"""Scraper source for rzeszow-news.pl.

Scrapes the homepage, collecting articles from the small td_module_10 cards
(thumbnail + title + excerpt).  Each card maps to one Article; category
defaults to WIADOMOSCI and the AI decides relevance from the content.
"""

from __future__ import annotations

import httpx

from rz_flow.models import Article
from rz_flow.parser import parse_rzeszow_news_page


class RzeszowNewsScraper:
    """Fetches the rzeszow-news.pl homepage and parses td_module_10 cards."""

    name = "rzeszow-news.pl/homepage"

    def __init__(self, base_url: str = "https://rzeszow-news.pl/") -> None:
        self.url = base_url

    async def fetch(
        self,
        client: httpx.AsyncClient,
        max_articles: int,
    ) -> list[Article]:
        response = await client.get(self.url)
        response.raise_for_status()
        return parse_rzeszow_news_page(response.text)[:max_articles]
