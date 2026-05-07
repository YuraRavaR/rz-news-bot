"""Tests for admin run-report branching in rz_flow.main._async_main."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import rz_flow.main as main_mod
from rz_flow.pipeline import PipelineStats
from rz_flow.storage import InMemoryStorage
from tests.test_config import _make_settings
from tests.test_pipeline import _make_flow_config


@pytest.mark.asyncio
async def test_async_main_skips_send_run_report_when_no_admin_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without TELEGRAM_ADMIN_CHAT_ID equivalent, send_run_report must not run."""
    storage = InMemoryStorage()
    pipeline_inst = MagicMock()
    pipeline_inst.run = AsyncMock(return_value=PipelineStats())

    publishers: list[MagicMock] = []

    def _publisher_factory(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.send_run_report = AsyncMock()
        m.send_alert = AsyncMock()
        m.publish = AsyncMock()
        publishers.append(m)
        return m

    monkeypatch.setattr(main_mod, "get_settings", lambda: _make_settings(telegram_admin_chat_id=None))
    monkeypatch.setattr(main_mod, "create_storage", lambda **kw: storage)
    monkeypatch.setattr(main_mod, "load_flow_config", _make_flow_config)
    monkeypatch.setattr(
        main_mod,
        "Pipeline",
        MagicMock(side_effect=lambda *a, **k: pipeline_inst),
    )
    monkeypatch.setattr(main_mod, "TelegramPublisher", MagicMock(side_effect=_publisher_factory))

    code = await main_mod._async_main(dry_run=False)

    assert code == 0
    pipeline_inst.run.assert_awaited_once_with(dry_run=False)
    # Pipeline is mocked, so only main's TelegramPublisher is constructed.
    admin_publisher = publishers[-1]
    admin_publisher.send_run_report.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_main_awaits_send_run_report_when_admin_chat_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With admin chat id set, send_run_report runs once after a successful pipeline."""
    storage = InMemoryStorage()
    pipeline_inst = MagicMock()
    stats = PipelineStats(posted=1, skipped=0, errors=0)
    pipeline_inst.run = AsyncMock(return_value=stats)

    publishers: list[MagicMock] = []

    def _publisher_factory(*args: object, **kwargs: object) -> MagicMock:
        m = MagicMock()
        m.send_run_report = AsyncMock()
        m.send_alert = AsyncMock()
        m.publish = AsyncMock()
        publishers.append(m)
        return m

    monkeypatch.setattr(
        main_mod,
        "get_settings",
        lambda: _make_settings(telegram_admin_chat_id="999001"),
    )
    monkeypatch.setattr(main_mod, "create_storage", lambda **kw: storage)
    monkeypatch.setattr(main_mod, "load_flow_config", _make_flow_config)
    monkeypatch.setattr(
        main_mod,
        "Pipeline",
        MagicMock(side_effect=lambda *a, **k: pipeline_inst),
    )
    monkeypatch.setattr(main_mod, "TelegramPublisher", MagicMock(side_effect=_publisher_factory))

    code = await main_mod._async_main(dry_run=False)

    assert code == 0
    admin_publisher = publishers[-1]
    admin_publisher.send_run_report.assert_awaited_once_with(stats, dry_run=False)
