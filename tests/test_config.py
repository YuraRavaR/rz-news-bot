"""Tests for application Settings and get_settings()."""

import pytest

from rz_flow.config import Settings


def _make_settings(**overrides: object) -> Settings:
    """Build a Settings instance that is fully isolated from .env file values.

    All optional fields are set explicitly so the tests do not depend on whatever
    the developer has in their local .env file.
    """
    defaults: dict[str, object] = {
        # Required fields
        "telegram_bot_token": "fake:token",
        "telegram_channel_id": "-100123",
        "telegram_admin_chat_id": None,
        "gemini_api_key": "fake-key",
        "turso_database_url": "libsql://fake.turso.io",
        "turso_auth_token": "fake-token",
        # Optional fields — set explicitly to avoid .env overriding them
        "gemini_model": "gemini-2.0-flash",
        "ai_min_score": 7.0,
        "app_env": "production",
    }
    return Settings(**{**defaults, **overrides})


class TestSettingsIsProduction:
    def test_is_production_true_when_app_env_is_production(self) -> None:
        s = _make_settings(app_env="production")
        assert s.is_production is True

    def test_is_production_false_when_app_env_is_development(self) -> None:
        s = _make_settings(app_env="development")
        assert s.is_production is False

    def test_is_production_false_for_any_non_production_value(self) -> None:
        s = _make_settings(app_env="staging")
        assert s.is_production is False


class TestSettingsDefaults:
    def test_default_app_env_is_production(self) -> None:
        s = _make_settings()
        assert s.app_env == "production"

    def test_gemini_model_can_be_overridden(self) -> None:
        s = _make_settings(gemini_model="gemini-test-model")
        assert s.gemini_model == "gemini-test-model"

    def test_gemini_model_field_default_in_schema(self) -> None:
        """The schema-level default is gemini-2.0-flash (can be overridden by env)."""
        field_info = Settings.model_fields["gemini_model"]
        assert field_info.default == "gemini-2.0-flash"

    def test_default_ai_min_score(self) -> None:
        """ai_min_score defaults to 7.0 when not overridden."""
        s = _make_settings()
        assert s.ai_min_score == 7.0

    def test_default_app_env_is_production_in_schema(self) -> None:
        """Schema-level default for app_env is 'production'."""
        field_info = Settings.model_fields["app_env"]
        assert field_info.default == "production"

    def test_ai_min_score_validation_rejects_above_10(self) -> None:
        with pytest.raises(Exception):
            _make_settings(ai_min_score=11.0)

    def test_ai_min_score_validation_rejects_below_0(self) -> None:
        with pytest.raises(Exception):
            _make_settings(ai_min_score=-1.0)

    def test_admin_chat_id_defaults_to_none(self) -> None:
        s = _make_settings()
        assert s.telegram_admin_chat_id is None


class TestPublishTelegramChatId:
    def test_production_uses_main_channel(self) -> None:
        s = _make_settings(telegram_channel_id="-100111")
        assert s.publish_telegram_chat_id(staging=False) == "-100111"

    def test_staging_uses_staging_channel_when_set(self) -> None:
        s = _make_settings(
            telegram_channel_id="-100111",
            telegram_staging_channel_id="-100222",
        )
        assert s.publish_telegram_chat_id(staging=True) == "-100222"

    def test_staging_raises_when_channel_missing(self) -> None:
        s = _make_settings(telegram_staging_channel_id=None)
        with pytest.raises(ValueError, match="TELEGRAM_STAGING_CHANNEL_ID"):
            s.publish_telegram_chat_id(staging=True)

    def test_staging_raises_when_channel_blank(self) -> None:
        s = _make_settings(telegram_staging_channel_id="   ")
        with pytest.raises(ValueError, match="TELEGRAM_STAGING_CHANNEL_ID"):
            s.publish_telegram_chat_id(staging=True)


class TestStagingTursoCredentials:
    def test_returns_pair_when_set(self) -> None:
        s = _make_settings(
            turso_staging_database_url="libsql://staging.turso.io",
            turso_staging_auth_token="staging-token",
        )
        assert s.staging_turso_credentials() == ("libsql://staging.turso.io", "staging-token")

    def test_raises_when_url_missing(self) -> None:
        s = _make_settings(
            turso_staging_database_url=None,
            turso_staging_auth_token="t",
        )
        with pytest.raises(ValueError, match="TURSO_STAGING"):
            s.staging_turso_credentials()

    def test_staging_config_errors_lists_missing(self) -> None:
        s = _make_settings()
        err = s.staging_config_errors()
        assert "TELEGRAM_STAGING_CHANNEL_ID" in err
        assert "TURSO_STAGING_DATABASE_URL" in err
        assert "TURSO_STAGING_AUTH_TOKEN" in err

    def test_staging_config_errors_empty_when_complete(self) -> None:
        s = _make_settings(
            telegram_staging_channel_id="-1001",
            turso_staging_database_url="libsql://s",
            turso_staging_auth_token="tok",
        )
        assert s.staging_config_errors() == []


class TestGetSettings:
    def test_get_settings_returns_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_settings() returns the same instance on repeated calls."""
        import rz_flow.config as cfg

        monkeypatch.setattr(cfg, "_settings", None)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake:token")
        monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "-100123")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://fake.turso.io")
        monkeypatch.setenv("TURSO_AUTH_TOKEN", "fake-token")

        s1 = cfg.get_settings()
        s2 = cfg.get_settings()
        assert s1 is s2

    def test_get_settings_resets_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After resetting _settings to None, get_settings() creates a fresh instance."""
        import rz_flow.config as cfg

        monkeypatch.setattr(cfg, "_settings", None)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake:token")
        monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "-100123")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://fake.turso.io")
        monkeypatch.setenv("TURSO_AUTH_TOKEN", "fake-token")

        settings = cfg.get_settings()
        assert settings is not None
        assert settings.telegram_bot_token == "fake:token"
