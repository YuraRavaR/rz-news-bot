"""Async HTTP scraper for rzeszow24.info category pages."""

import httpx

from rz_flow.config import Settings
from rz_flow.models import Article, Category
from rz_flow.parser import parse_category_page

# Full browser-like headers to avoid 403 blocks from WAF/anti-bot rules.
# rzeszow24.info blocks requests that look like bots (non-browser User-Agent,
# missing Accept-Encoding, etc.). These headers mimic a real Chrome on macOS.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7,uk;q=0.6",
    # Accept-Encoding is intentionally omitted — httpx sets it automatically
    # and handles brotli/gzip decompression via the `brotli` package.
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_CATEGORIES: list[Category] = [Category.IMPREZY, Category.WIADOMOSCI]


async def fetch_articles(settings: Settings) -> list[Article]:
    """Fetch and parse articles from all configured categories.

    Returns a combined deduplicated list (imprezy first, then wiadomosci).
    Raises httpx.HTTPError on network failures (caller handles retry/logging).
    """
    all_articles: list[Article] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(
        timeout=settings.scraper_timeout,
        follow_redirects=True,
        headers=_HEADERS,
        http2=True,  # Chrome uses HTTP/2 — helps bypass some WAFs
    ) as client:
        for category in _CATEGORIES:
            url = f"{settings.scraper_base_url}/{category.value}/"
            response = await client.get(url)
            response.raise_for_status()

            parsed = parse_category_page(response.text, category)
            # Deduplicate across categories (unlikely but safe)
            new_articles = [a for a in parsed if a.id not in seen_ids]
            seen_ids.update(a.id for a in new_articles)
            all_articles.extend(new_articles[: settings.scraper_max_articles])

    return all_articles
