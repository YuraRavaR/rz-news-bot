"""Base protocol for all scraper sources.

Each source is responsible for fetching and parsing articles from one URL.
The protocol is intentionally thin: a name for logging and a fetch() method.

To add a new source:
  1. Create src/rz_flow/sources/<site>.py implementing ScraperSource.
  2. Add an instance to ACTIVE_SOURCES in src/rz_flow/sources/__init__.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import httpx

    from rz_flow.models import Article


@runtime_checkable
class ScraperSource(Protocol):
    """Contract every scraper source must satisfy."""

    name: str
    """Human-readable identifier used in logs, e.g. 'rzeszow24/najnowsze' or 'rzeszow-news.pl'."""

    url: str
    """Actual URL the scraper fetches."""

    async def fetch(
        self,
        client: httpx.AsyncClient,
        max_articles: int,
    ) -> list[Article]:
        """Fetch, parse and return up to *max_articles* articles.

        Args:
            client: Shared async HTTP client (headers/timeouts pre-configured).
            max_articles: Upper bound on returned items.

        Raises:
            httpx.HTTPError: on network or HTTP-status failures.
        """
        ...
