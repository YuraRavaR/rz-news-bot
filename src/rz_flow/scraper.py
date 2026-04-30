"""Async HTTP orchestrator: fetches articles from all active sources.

Which sources run (and with which limits) is controlled by FlowConfig loaded
from config.yaml.  To add or change what is scraped, edit that file and the
sources registry — this file stays stable.
"""

from __future__ import annotations

import httpx
import structlog

from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig
from rz_flow.models import Article
from rz_flow.sources import get_active_sources

logger = structlog.get_logger(__name__)

# Full browser-like headers to avoid 403 blocks from WAF/anti-bot rules.
# rzeszow24.info blocks requests that look like bots (non-browser User-Agent,
# missing Accept-Encoding, etc.). These headers mimic a real Chrome on macOS.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
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


async def fetch_articles(settings: Settings, config: FlowConfig) -> list[Article]:
    """Fetch and deduplicate articles from all active sources.

    Each source is instantiated from FlowConfig and receives its own
    per-source article limit.  Results are deduplicated across sources by
    article ID.

    Raises:
        httpx.HTTPError: propagated from any source on network/HTTP failures.
    """
    all_articles: list[Article] = []
    seen_ids: set[str] = set()

    enabled = config.enabled_sources
    sources = get_active_sources(config)

    logger.info("scrape_started", count=len(sources))

    async with httpx.AsyncClient(
        timeout=settings.scraper_timeout,
        follow_redirects=True,
        headers=_HEADERS,
        http2=True,  # Chrome uses HTTP/2 — helps bypass some WAFs
    ) as client:
        for source, src_cfg in zip(sources, enabled):
            logger.info("scrape_source_start", name=source.name, url=source.url)
            articles = await source.fetch(client, src_cfg.max_articles)
            new = [a for a in articles if a.id not in seen_ids]
            seen_ids.update(a.id for a in new)
            all_articles.extend(new)
            logger.info("scrape_source_done", name=source.name, found=len(articles))

    logger.info("scrape_done", total=len(all_articles))
    return all_articles
