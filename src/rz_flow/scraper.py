"""Async HTTP orchestrator: fetches articles from all active sources.

Which sources run (and with which limits) is controlled by FlowConfig loaded
from config.yaml.  To add or change what is scraped, edit that file and the
sources registry — this file stays stable.
"""

from __future__ import annotations

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
        "Chrome/136.0.0.0 Safari/537.36"
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

@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=20),
    reraise=True,
)
async def _fetch_one(
    source: object,
    client: httpx.AsyncClient,
    max_articles: int,
) -> list[Article]:
    """Thin wrapper around source.fetch that tenacity can decorate."""
    fetch = getattr(source, "fetch")
    return await fetch(client, max_articles)  # type: ignore[no-any-return]


async def fetch_articles(settings: Settings, config: FlowConfig) -> list[Article]:
    """Fetch and deduplicate articles from all active sources.

    Each source is fetched independently — a timeout or network error on one
    source is logged and skipped so the remaining sources still run.
    Results are deduplicated across sources by article ID.

    Returns an empty list only when every source fails.
    """
    all_articles: list[Article] = []
    seen_ids: set[str] = set()
    failed_sources: list[str] = []

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
            log = logger.bind(name=source.name)
            log.info("scrape_source_start", url=source.url)
            try:
                articles = await _fetch_one(source, client, src_cfg.max_articles)
            except Exception as exc:
                log.warning("scrape_source_failed", error=str(exc))
                failed_sources.append(source.name)
                continue

            new = [a for a in articles if a.id not in seen_ids]
            seen_ids.update(a.id for a in new)
            all_articles.extend(new)
            log.info("scrape_source_done", found=len(articles))

    if failed_sources:
        logger.warning("scrape_partial_failure", failed=failed_sources)

    logger.info("scrape_done", total=len(all_articles))
    return all_articles
