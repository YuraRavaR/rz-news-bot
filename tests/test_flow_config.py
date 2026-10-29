"""Tests for YAML-based flow configuration loading and validation."""

import pathlib
import textwrap

import pytest

from rz_flow.flow_config import (
    FlowConfig,
    PipelineConfig,
    SourceConfig,
    load_flow_config,
)
from rz_flow.sources import get_active_sources

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── load_flow_config ──────────────────────────────────────────────────────────


def test_repo_root_config_yaml_loads() -> None:
    """Committed config.yaml must parse (catches schema drift in CI and locally)."""
    cfg_path = REPO_ROOT / "config.yaml"
    config = load_flow_config(str(cfg_path))
    assert len(config.sources) >= 1
    assert config.pipeline.max_posts_per_run >= 1
    assert config.pipeline.gemini_model
    assert config.digest.calendar_url.startswith("https://")


def _write_config(tmp_path: pathlib.Path, pipeline_yaml: str) -> str:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "sources:\n"
        "  - scraper: NajnowszeScraper\n"
        "    base_url: https://rzeszow24.info/najnowsze\n" + pipeline_yaml,
        encoding="utf-8",
    )
    return str(cfg_file)


def test_gemini_model_is_read_from_config(tmp_path: pathlib.Path) -> None:
    path = _write_config(tmp_path, "pipeline:\n  gemini_model: gemini-test-model\n")

    assert load_flow_config(path).pipeline.gemini_model == "gemini-test-model"


@pytest.mark.parametrize(
    "pipeline_yaml",
    [
        "",
        "pipeline:\n  max_posts_per_run: 3\n",
        "pipeline:\n  gemini_model: ''\n",
        "pipeline:\n  gemini_model: '   '\n",
    ],
    ids=["no_pipeline_section", "model_missing", "model_empty", "model_blank"],
)
def test_config_without_a_gemini_model_fails_to_load(
    tmp_path: pathlib.Path, pipeline_yaml: str
) -> None:
    path = _write_config(tmp_path, pipeline_yaml)

    with pytest.raises(ValueError, match="gemini_model"):
        load_flow_config(path)


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
          gemini_model: gemini-test-model
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
        pipeline:
          gemini_model: gemini-test-model
        """)
    )
    monkeypatch.setenv("FLOW_CONFIG_PATH", str(cfg_file))
    config = load_flow_config()
    assert config.sources[0].scraper == "RzeszowNewsScraper"


def test_digest_section_is_read_from_config(tmp_path: pathlib.Path) -> None:
    path = _write_config(
        tmp_path,
        "pipeline:\n  gemini_model: gemini-test-model\n"
        "digest:\n  calendar_url: https://calendar.example/index.php\n  enabled: false\n",
    )

    digest = load_flow_config(path).digest
    assert digest.calendar_url == "https://calendar.example/index.php"
    assert digest.enabled is False


def test_digest_section_defaults_to_the_city_calendar(tmp_path: pathlib.Path) -> None:
    path = _write_config(tmp_path, "pipeline:\n  gemini_model: gemini-test-model\n")

    digest = load_flow_config(path).digest
    assert digest.calendar_url == "https://erzeszow.pl/index.php"
    assert digest.enabled is True


def test_load_flow_config_missing_file_raises(tmp_path) -> None:
    """FileNotFoundError is raised when the config file does not exist."""
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_flow_config(str(tmp_path / "config.yaml"))


def test_load_flow_config_pipeline_defaults(tmp_path) -> None:
    """Every pipeline option except the model has a default."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        sources:
          - scraper: NajnowszeScraper
            base_url: https://rzeszow24.info/najnowsze
        pipeline:
          gemini_model: gemini-test-model
        """)
    )
    config = load_flow_config(str(cfg_file))
    assert config.pipeline.inter_ai_delay_seconds == 5.0
    assert config.pipeline.inter_post_delay_seconds == 2.0
    assert config.pipeline.report_display_timezone is None


def test_pipeline_report_display_timezone_accepts_valid_iana(tmp_path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        sources:
          - scraper: NajnowszeScraper
            base_url: https://rzeszow24.info/najnowsze
        pipeline:
          gemini_model: gemini-test-model
          report_display_timezone: Europe/Warsaw
        """)
    )
    config = load_flow_config(str(cfg_file))
    assert config.pipeline.report_display_timezone == "Europe/Warsaw"


def test_pipeline_report_display_timezone_rejects_invalid(tmp_path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        sources:
          - scraper: NajnowszeScraper
            base_url: https://rzeszow24.info/najnowsze
        pipeline:
          gemini_model: gemini-test-model
          report_display_timezone: Not/AZone
        """)
    )
    with pytest.raises(ValueError, match="Invalid IANA"):
        load_flow_config(str(cfg_file))


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
        ],
        pipeline=PipelineConfig(gemini_model="gemini-test-model"),
    )
    assert len(config.enabled_sources) == 1
    assert config.enabled_sources[0].scraper == "NajnowszeScraper"


# ── get_active_sources ────────────────────────────────────────────────────────


def test_get_active_sources_returns_correct_instances() -> None:
    config = FlowConfig(
        sources=[
            SourceConfig(scraper="NajnowszeScraper", base_url="https://rzeszow24.info/najnowsze"),
            SourceConfig(scraper="RzeszowNewsScraper", base_url="https://rzeszow-news.pl/"),
        ],
        pipeline=PipelineConfig(gemini_model="gemini-test-model"),
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
        ],
        pipeline=PipelineConfig(gemini_model="gemini-test-model"),
    )
    sources = get_active_sources(config)
    assert sources[0].url == "https://custom.example.com/feed"


def test_get_active_sources_unknown_scraper_raises() -> None:
    config = FlowConfig(
        sources=[
            SourceConfig(scraper="NonExistentScraper", base_url="https://example.com"),
        ],
        pipeline=PipelineConfig(gemini_model="gemini-test-model"),
    )
    with pytest.raises(ValueError, match="NonExistentScraper"):
        get_active_sources(config)
