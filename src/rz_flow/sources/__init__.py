"""Scraper sources registry.

``get_active_sources(config)`` builds the list of enabled scrapers from
FlowConfig (loaded from config.yaml).

To add a new site:
  1. Create sources/<site>.py with a scraper class.
  2. Import the class here and add it to ``_SCRAPER_REGISTRY``.
  3. Add an entry in config.yaml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rz_flow.sources.base import ScraperSource

if TYPE_CHECKING:
    from rz_flow.flow_config import FlowConfig
from rz_flow.sources.rzeszow24 import NajnowszeScraper
from rz_flow.sources.rzeszow_news import RzeszowNewsScraper

_SCRAPER_REGISTRY: dict[str, type] = {
    "NajnowszeScraper": NajnowszeScraper,
    "RzeszowNewsScraper": RzeszowNewsScraper,
}


def get_active_sources(config: FlowConfig) -> list[ScraperSource]:
    """Return instantiated scrapers for all enabled sources in config."""
    sources = []
    for src in config.enabled_sources:
        cls = _SCRAPER_REGISTRY.get(src.scraper)
        if cls is None:
            raise ValueError(
                f"Unknown scraper '{src.scraper}' in config.yaml. "
                f"Available: {', '.join(_SCRAPER_REGISTRY)}"
            )
        sources.append(cls(base_url=src.base_url))
    return sources


__all__ = ["ScraperSource", "get_active_sources"]
