"""YAML-based flow configuration.

Loads config.yaml (or the path in FLOW_CONFIG_PATH env var) and validates it
with Pydantic.  Contains source definitions (which scrapers run, with which
URLs and per-source article limits), the Gemini model, pipeline timing,
optional report clock timezone, and the Digest's Calendar.
"""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, field_validator

from rz_flow.storage import DEFAULT_MAX_ERROR_ATTEMPTS

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
    """Model, pacing, caps and admin-report display settings for the pipeline."""

    gemini_model: str = Field(
        description=(
            "Gemini model used for every evaluation. It decides the daily free-tier "
            "quota (ADR 0002), so it is committed here rather than set per environment."
        ),
    )
    inter_ai_delay_seconds: float = 5.0
    inter_post_delay_seconds: float = 2.0
    max_posts_per_run: int = 5
    max_error_attempts: int = Field(
        default=DEFAULT_MAX_ERROR_ATTEMPTS,
        ge=1,
        description=(
            "How many times a failing article is retried on later runs before it is "
            "retired permanently. 1 = never retry."
        ),
    )
    report_display_timezone: str | None = Field(
        default=None,
        description=(
            "IANA timezone for the admin Telegram run-report header (e.g. Europe/Warsaw). "
            "Omit or null for UTC."
        ),
    )

    @field_validator("gemini_model")
    @classmethod
    def _non_empty_model(cls, v: str) -> str:
        name = v.strip()
        if not name:
            raise ValueError("gemini_model must name a Gemini model")
        return name

    @field_validator("report_display_timezone")
    @classmethod
    def _validate_report_display_timezone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        name = v.strip()
        if not name:
            return None
        try:
            ZoneInfo(name)
        except Exception as exc:
            raise ValueError(f"Invalid IANA timezone: {name!r}") from exc
        return name


class DigestConfig(BaseModel):
    """The city's events Calendar, read only for the Digest."""

    calendar_url: str = Field(
        default="https://erzeszow.pl/index.php",
        description="Endpoint of the Calendar's day picker (one request per weekend day).",
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Read the Calendar. False turns it off with a config change, no code release: "
            "no request is made and the Digest goes out with news events only."
        ),
    )


class FlowConfig(BaseModel):
    """Top-level flow configuration loaded from config.yaml."""

    sources: list[SourceConfig]
    # Validated even when the section is missing, so the error names the required
    # pipeline.gemini_model instead of just the absent section.
    pipeline: PipelineConfig = Field(default_factory=dict, validate_default=True)
    digest: DigestConfig = Field(default_factory=DigestConfig)

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
