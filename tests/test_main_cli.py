"""Tests for main CLI staging guards (no real Telegram or Turso)."""

import pytest

import rz_flow.main as main_mod
from tests.test_config import _make_settings


@pytest.mark.asyncio
async def test_async_main_rejects_staging_with_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_mod, "get_settings", lambda: _make_settings())
    code = await main_mod._async_main(dry_run=True, staging=True)
    assert code == 2


@pytest.mark.asyncio
async def test_async_main_staging_returns_2_when_staging_env_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_mod, "get_settings", lambda: _make_settings())
    code = await main_mod._async_main(staging=True)
    assert code == 2
