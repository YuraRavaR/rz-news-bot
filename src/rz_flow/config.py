"""Application configuration loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings are read from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(..., description="Telegram Bot API token from @BotFather")
    telegram_channel_id: str = Field(..., description="Target channel ID (e.g. -1001234567890)")

    # ── Gemini AI ─────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google AI Studio API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")

    # ── Turso (libsql) ────────────────────────────────────────────────────────
    turso_database_url: str = Field(..., description="libsql:// URL from turso db show")
    turso_auth_token: str = Field(..., description="Auth token from turso db tokens create")

    # ── Scraper ───────────────────────────────────────────────────────────────
    scraper_base_url: str = Field(
        default="https://rzeszow24.info",
        description="Base URL of the news site",
    )
    scraper_timeout: float = Field(default=15.0, description="HTTP request timeout in seconds")
    scraper_max_articles: int = Field(
        default=5,
        description="Max articles to fetch per category per run",
    )

    # ── AI filtering ──────────────────────────────────────────────────────────
    ai_min_score: float = Field(
        default=7.0,
        ge=0,
        le=10,
        description="Minimum Gemini score (0–10) to publish",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = Field(default="production", description="'production' or 'development'")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
