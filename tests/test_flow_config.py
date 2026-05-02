"""Tests for YAML-based flow configuration loading and validation."""

import pathlib
import textwrap

import pytest
import respx
from httpx import Response

from rz_flow.flow_config import FlowConfig, PipelineConfig, SourceConfig, load_flow_config
from rz_flow.sources import get_active_sources
from rz_flow.sources.rzeszow24 import CategoryPageScraper

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# ── load_flow_config ──────────────────────────────────────────────────────────


def test_load_flow_config_reads_yaml(tmp_path) -> None:
    """load_flow_config parses a valid YAML file into a FlowConfig."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        sources:
          - scraper: NajnowszeScraper
            base_url: https://rzeszow24.info/najnowsze
            max_articles: 7
            enabled: true
        pipeline:
          inter_ai_delay_seconds: 3.0
          inter_post_delay_seconds: 1.5
        """)
    )
    config = load_flow_config(str(cfg_file))

    assert len(config.sources) == 1
    assert config.sources[0].scraper == "NajnowszeScraper"
    assert config.sources[0].max_articles == 7
    assert config.pipeline.inter_ai_delay_seconds == 3.0
    assert config.pipeline.inter_post_delay_seconds == 1.5


def test_load_flow_config_uses_env_var(tmp_path, monkeypatch) -> None:
    """FLOW_CONFIG_PATH env var overrides the default path."""
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        sources:
          - scraper: RzeszowNewsScraper
            base_url: https://rzeszow-news.pl/
            max_articles: 10
        """)
    )
    monkeypatch.setenv("FLOW_CONFIG_PATH", str(cfg_file))
    config = load_flow_config()
    assert config.sources[0].scraper == "RzeszowNewsScraper"


def test_load_flow_config_missing_file_raises(tmp_path) -> None:
    """FileNotFoundError is raised when the config file does not exist."""
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_flow_config(str(tmp_path / "config.yaml"))


def test_load_flow_config_pipeline_defaults(tmp_path) -> None:
    """pipeline section is optional; defaults are applied when omitted."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        sources:
          - scraper: NajnowszeScraper
            base_url: https://rzeszow24.info/najnowsze
        """)
    )
    config = load_flow_config(str(cfg_file))
    assert config.pipeline.inter_ai_delay_seconds == 5.0
    assert config.pipeline.inter_post_delay_seconds == 2.0


# ── SourceConfig validation ───────────────────────────────────────────────────


def test_source_config_max_articles_must_be_positive() -> None:
    with pytest.raises(Exception):
        SourceConfig(scraper="NajnowszeScraper", base_url="https://example.com", max_articles=0)


# ── FlowConfig.enabled_sources ────────────────────────────────────────────────


def test_enabled_sources_filters_disabled() -> None:
    config = FlowConfig(
        sources=[
            SourceConfig(scraper="NajnowszeScraper", base_url="https://a.com", enabled=True),
            SourceConfig(scraper="RzeszowNewsScraper", base_url="https://b.com", enabled=False),
        ]
    )
    assert len(config.enabled_sources) == 1
    assert config.enabled_sources[0].scraper == "NajnowszeScraper"


# ── get_active_sources ────────────────────────────────────────────────────────


def test_get_active_sources_returns_correct_instances() -> None:
    config = FlowConfig(
        sources=[
            SourceConfig(scraper="NajnowszeScraper", base_url="https://rzeszow24.info/najnowsze"),
            SourceConfig(scraper="RzeszowNewsScraper", base_url="https://rzeszow-news.pl/"),
        ]
    )
    sources = get_active_sources(config)
    assert len(sources) == 2
    assert sources[0].name == "rzeszow24/najnowsze"
    assert sources[1].name == "rzeszow-news.pl"


def test_get_active_sources_uses_base_url() -> None:
    """Scraper url attribute matches the base_url from config."""
    config = FlowConfig(
        sources=[
            SourceConfig(scraper="NajnowszeScraper", base_url="https://custom.example.com/feed"),
        ]
    )
    sources = get_active_sources(config)
    assert sources[0].url == "https://custom.example.com/feed"


def test_get_active_sources_unknown_scraper_raises() -> None:
    config = FlowConfig(
        sources=[
            SourceConfig(scraper="NonExistentScraper", base_url="https://example.com"),
        ]
    )
    with pytest.raises(ValueError, match="NonExistentScraper"):
        get_active_sources(config)


# ── CategoryPageScraper (archived) ────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_category_page_scraper_fetch() -> None:
    """CategoryPageScraper fetches /imprezy/ and /wiadomosci/ and returns articles."""
    import httpx

    imprezy_html = (FIXTURES_DIR / "imprezy_sample.html").read_text(encoding="utf-8")
    wiadomosci_html = (FIXTURES_DIR / "wiadomosci_sample.html").read_text(encoding="utf-8")
    respx.get("https://rzeszow24.info/imprezy/").mock(
        return_value=Response(200, text=imprezy_html)
    )
    respx.get("https://rzeszow24.info/wiadomosci/").mock(
        return_value=Response(200, text=wiadomosci_html)
    )

    scraper = CategoryPageScraper(base_url="https://rzeszow24.info")
    async with httpx.AsyncClient() as client:
        articles = await scraper.fetch(client, max_articles=5)

    assert isinstance(articles, list)
