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
    telegram_admin_chat_id: str | None = Field(
        default=None,
        description="Private chat ID for crash/quota alerts (falls back to channel if unset)",
    )
    telegram_staging_channel_id: str | None = Field(
        default=None,
        description="Optional staging channel ID (used with CLI --staging)",
    )
    telegram_events_channel_id: str | None = Field(
        default=None,
        description=(
            "Channel ID for the dedicated events channel. "
            "When set, articles with is_event=True are posted here in addition to the main channel. "
            "When unset, all posts go only to the main channel."
        ),
    )
    telegram_staging_events_channel_id: str | None = Field(
        default=None,
        description="Staging variant of TELEGRAM_EVENTS_CHANNEL_ID (used with CLI --staging)",
    )

    # ── Gemini AI ─────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google AI Studio API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")

    # ── Turso (libsql) ────────────────────────────────────────────────────────
    turso_database_url: str = Field(..., description="libsql:// URL from turso db show")
    turso_auth_token: str = Field(..., description="Auth token from turso db tokens create")
    turso_staging_database_url: str | None = Field(
        default=None,
        description="Separate Turso DB for --staging runs (dedup isolated from production)",
    )
    turso_staging_auth_token: str | None = Field(
        default=None,
        description="Auth token for turso_staging_database_url",
    )

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

    def publish_telegram_chat_id(self, *, staging: bool) -> str:
        """Chat ID for pipeline publishes: production channel or staging channel."""
        if not staging:
            return self.telegram_channel_id.strip()
        sid = (self.telegram_staging_channel_id or "").strip()
        if not sid:
            msg = "TELEGRAM_STAGING_CHANNEL_ID is required for staging runs"
            raise ValueError(msg)
        return sid

    def events_telegram_chat_id(self, *, staging: bool) -> str | None:
        """Chat ID for the events channel, or None if not configured.

        In staging mode, prefers TELEGRAM_STAGING_EVENTS_CHANNEL_ID; falls back to
        the production events channel ID so staging runs still exercise dual-channel
        logic even without a dedicated staging events channel.
        Returns None when no events channel is configured at all — callers treat this
        as "post to main channel only".
        """
        if staging:
            sid = (self.telegram_staging_events_channel_id or "").strip()
            if sid:
                return sid
        prod = (self.telegram_events_channel_id or "").strip()
        return prod or None

    def staging_turso_credentials(self) -> tuple[str, str]:
        """Return (database_url, auth_token) for staging Turso; raises if incomplete."""
        url = (self.turso_staging_database_url or "").strip()
        token = (self.turso_staging_auth_token or "").strip()
        if not url or not token:
            msg = (
                "TURSO_STAGING_DATABASE_URL and TURSO_STAGING_AUTH_TOKEN are required "
                "for staging runs (use a separate DB from production)"
            )
            raise ValueError(msg)
        return url, token

    def staging_config_errors(self) -> list[str]:
        """Human-readable list of missing staging env vars (empty if staging is runnable)."""
        missing: list[str] = []
        if not (self.telegram_staging_channel_id or "").strip():
            missing.append("TELEGRAM_STAGING_CHANNEL_ID")
        if not (self.turso_staging_database_url or "").strip():
            missing.append("TURSO_STAGING_DATABASE_URL")
        if not (self.turso_staging_auth_token or "").strip():
            missing.append("TURSO_STAGING_AUTH_TOKEN")
        return missing


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
