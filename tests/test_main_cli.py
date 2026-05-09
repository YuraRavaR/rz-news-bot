"""Smoke tests for rz_flow.main._async_main (no real Turso/Telegram)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import rz_flow.main as main_mod
from tests.test_config import _make_settings


@pytest.mark.asyncio
async def test_async_main_init_db_only_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """init-db path initializes storage and exits before pipeline."""
    mock_storage = MagicMock()
    mock_storage.init = AsyncMock()
    mock_storage.close = AsyncMock()
    monkeypatch.setattr(main_mod, "get_settings", lambda: _make_settings())
    monkeypatch.setattr(main_mod, "create_storage", lambda **kwargs: mock_storage)

    code = await main_mod._async_main(init_db_only=True)

    assert code == 0
    mock_storage.init.assert_awaited_once()
    mock_storage.close.assert_awaited_once()
