"""YAML-based flow configuration.

Loads config.yaml (or the path in FLOW_CONFIG_PATH env var) and validates it
with Pydantic.  Contains source definitions (which scrapers run, with which
URLs and per-source article limits) and pipeline timing settings.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

_DEFAULT_CONFIG_PATH = "config.yaml"


class SourceConfig(BaseModel):
    """Configuration for a single scraper source."""

    scraper: str
    base_url: str
    max_articles: int = 5
    enabled: bool = True

    @field_validator("max_articles")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_articles must be at least 1")
        return v


class PipelineConfig(BaseModel):
    """Timing settings for the processing pipeline."""

    inter_ai_delay_seconds: float = 5.0
    inter_post_delay_seconds: float = 2.0


class FlowConfig(BaseModel):
    """Top-level flow configuration loaded from config.yaml."""

    sources: list[SourceConfig]
    pipeline: PipelineConfig = PipelineConfig()

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]


def load_flow_config(path: str | None = None) -> FlowConfig:
    """Load and validate flow config from a YAML file.

    The path is resolved in order:
      1. ``path`` argument (if provided)
      2. ``FLOW_CONFIG_PATH`` environment variable
      3. ``config.yaml`` in the current working directory

    Raises:
        FileNotFoundError: if the resolved path does not exist.
        ValueError: if the YAML content is invalid.
    """
    resolved = path or os.environ.get("FLOW_CONFIG_PATH") or _DEFAULT_CONFIG_PATH
    config_path = Path(resolved)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Flow config file not found: {config_path.resolve()}\n"
            "Create config.yaml in the project root or set FLOW_CONFIG_PATH."
        )

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return FlowConfig.model_validate(raw or {})
