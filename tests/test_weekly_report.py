"""The Weekly report from the outside: ``rz-flow --weekly-report`` on seeded storage.

Storage is an InMemoryStorage seeded across two weeks, the Bot API is mocked with
respx, and "today" is injected. Tests assert what the admin would notice: which chat
got which message and what it says.
"""

from collections.abc import Awaitable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

import httpx
import pytest
import respx

import rz_flow.main as main_mod
from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig, PipelineConfig, SourceConfig
from rz_flow.models import AIDecision, Article, Category, CategoryTag, Decision
from rz_flow.storage import InMemoryStorage
from tests.test_digest import _ADMIN, _SEND_URL, _settings
from tests.test_pipeline_runs import FakeBotApi

# The report sent on Monday 26.10 covers Monday 19.10 to Sunday 25.10 (Warsaw)
_MONDAY = date(2026, 10, 26)
# Noon in Warsaw on a day of the report week (CEST until 25.10, UTC+2)
_TUESDAY_NOON = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
_FRIDAY_NOON = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)
_PREVIOUS_WEEK = datetime(2026, 10, 14, 10, 0, tzinfo=UTC)


def _flow_config() -> FlowConfig:
    return FlowConfig(
        sources=[
            SourceConfig(scraper="NajnowszeScraper", base_url="https://rzeszow24.info"),
            SourceConfig(scraper="RzeszowNewsScraper", base_url="https://rzeszow-news.pl"),
        ],
        pipeline=PipelineConfig(gemini_model="gemini-test-model"),
    )


async def _seed(
    storage: InMemoryStorage,
    article_id: str,
    decision: Decision,
    when: datetime,
    *,
    is_event: bool = False,
    event_start: date | None = None,
) -> None:
    """Store one Article as a Run would have, last written at ``when``."""
    article = Article(
        id=article_id,
        url=f"https://example.com/{article_id}",
        category=Category.WIADOMOSCI,
        title_pl=f"Artykuł {article_id}",
    )
    ai = AIDecision(
        is_interesting=decision == Decision.POSTED,
        is_event=is_event,
        score=8.0,
        category_tag=CategoryTag.OTHER,
        ua_title="Стаття",
        ua_summary="Опис.",
        reason="Test",
        event_start=event_start,
    )
    judged = decision in (Decision.POSTED, Decision.SKIPPED)
    await storage.save_decision(article, decision, ai if judged else None)
    record = storage.get_record(article_id)
    assert record is not None
    storage._records[article_id] = record.model_copy(update={"seen_at": when})


class RunReport(Protocol):
    def __call__(
        self,
        today: date = _MONDAY,
        *,
        dry_run: bool = False,
        staging: bool = False,
        settings: Settings | None = None,
    ) -> Awaitable[int]: ...


@pytest.fixture
def bot_api() -> Iterator[FakeBotApi]:
    api = FakeBotApi()
    with respx.mock(assert_all_called=False) as router:
        router.post(_SEND_URL).mock(side_effect=api.handle)
        yield api


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def run_report(monkeypatch: pytest.MonkeyPatch, storage: InMemoryStorage) -> RunReport:
    """``rz-flow --weekly-report`` with "today" injected, on the seeded storage."""

    async def _run(
        today: date = _MONDAY,
        *,
        dry_run: bool = False,
        staging: bool = False,
        settings: Settings | None = None,
    ) -> int:
        monkeypatch.setattr(main_mod, "get_settings", lambda: settings or _settings())
        monkeypatch.setattr(main_mod, "load_flow_config", _flow_config)
        monkeypatch.setattr(main_mod, "create_storage", lambda **kw: storage)
        return await main_mod._async_main(
            weekly_report=True, dry_run=dry_run, staging=staging, today=today
        )

    return _run


def _report(bot_api: FakeBotApi) -> str:
    """Everything the admin chat got, as one text."""
    return "\n".join(bot_api.texts[_ADMIN])


class TestPerSource:
    async def test_counts_each_outcome_of_the_week_per_source(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        for article_id, decision in [
            ("rz24/1", Decision.POSTED),
            ("rz24/2", Decision.POSTED),
            ("rz24/3", Decision.SKIPPED),
            ("rz24/4", Decision.REPEAT),
            ("rz24/5", Decision.ERROR),
            ("rz24/6", Decision.UNCONFIRMED),
            ("rzn/1", Decision.SKIPPED),
        ]:
            await _seed(storage, article_id, decision, _TUESDAY_NOON)
        # The week before is not this week's
        await _seed(storage, "rz24/OLD", Decision.POSTED, _PREVIOUS_WEEK)

        code = await run_report()

        assert code == 0
        assert bot_api.posts == [_ADMIN]
        report = _report(bot_api)
        assert report.startswith("<b>📊 Тижневий звіт, 19–25.10</b>")
        assert (
            "<b>rzeszow24/najnowsze</b>: нових 6 · опубліковано 2 · пропущено 1 · "
            "повторів 1 · помилок 1 · непідтверджених 1"
        ) in report
        assert (
            "<b>rzeszow-news.pl</b>: нових 1 · опубліковано 0 · пропущено 1 · "
            "повторів 0 · помилок 0 · непідтверджених 0"
        ) in report

    async def test_an_enabled_source_with_nothing_this_week_is_listed_with_zeros(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _seed(storage, "rz24/1", Decision.POSTED, _TUESDAY_NOON)

        await run_report()

        assert (
            "<b>rzeszow-news.pl</b>: нових 0 · опубліковано 0 · пропущено 0 · "
            "повторів 0 · помилок 0 · непідтверджених 0"
        ) in _report(bot_api)

    async def test_the_week_is_monday_to_sunday_in_warsaw(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        # 00:30 on Monday 19.10 in Warsaw is still Sunday 18.10 in UTC
        await _seed(
            storage, "rz24/FIRST", Decision.POSTED, datetime(2026, 10, 18, 22, 30, tzinfo=UTC)
        )
        # 00:30 on Monday 26.10 in Warsaw (CET from 25.10) is in the next week
        await _seed(
            storage, "rz24/NEXT", Decision.SKIPPED, datetime(2026, 10, 25, 23, 30, tzinfo=UTC)
        )

        await run_report()

        assert "<b>rzeszow24/najnowsze</b>: нових 1 · опубліковано 1 · пропущено 0" in _report(
            bot_api
        )

    async def test_any_day_reports_the_previous_full_week(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await run_report(today=date(2026, 11, 1))  # a Sunday

        assert _report(bot_api).startswith("<b>📊 Тижневий звіт, 19–25.10</b>")


class TestEventsAndQuota:
    async def test_events_posted_and_those_without_an_event_date(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _seed(
            storage,
            "rz24/E1",
            Decision.POSTED,
            _TUESDAY_NOON,
            is_event=True,
            event_start=date(2026, 10, 24),
        )
        for article_id in ("rz24/E2", "rz24/E3", "rzn/E4"):
            await _seed(storage, article_id, Decision.POSTED, _TUESDAY_NOON, is_event=True)
        # A Skipped event is not a posted one
        await _seed(storage, "rz24/E5", Decision.SKIPPED, _TUESDAY_NOON, is_event=True)

        await run_report()

        assert "Події: опубліковано 4, без дати 3 (75%)" in _report(bot_api)

    async def test_a_week_from_before_event_dates_says_older_events_have_none(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _seed(storage, "rz24/E1", Decision.POSTED, _TUESDAY_NOON, is_event=True)

        await run_report()

        assert "Дати подій зберігаються з 29.10" in _report(bot_api)

    async def test_a_week_after_event_dates_has_no_such_note(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _seed(
            storage,
            "rz24/E1",
            Decision.POSTED,
            datetime(2026, 11, 3, 10, 0, tzinfo=UTC),
            is_event=True,
        )

        await run_report(today=date(2026, 11, 9))

        report = _report(bot_api)
        assert "Події: опубліковано 1, без дати 1 (100%)" in report
        assert "Дати подій зберігаються" not in report

    async def test_busiest_day_estimate_counts_ai_judged_articles(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _seed(storage, "rz24/T1", Decision.POSTED, _TUESDAY_NOON)
        for i, decision in enumerate(
            [Decision.POSTED, Decision.SKIPPED, Decision.UNCONFIRMED, Decision.SKIPPED]
        ):
            await _seed(storage, f"rz24/F{i}", decision, _FRIDAY_NOON)
        # Neither a Repeat nor an Error is counted: no Gemini answer was stored for them
        await _seed(storage, "rz24/T_REPEAT", Decision.REPEAT, _TUESDAY_NOON)
        await _seed(storage, "rz24/T_ERROR", Decision.ERROR, _TUESDAY_NOON)
        for i in range(3):
            await _seed(storage, f"rz24/R{i}", Decision.REPEAT, _TUESDAY_NOON + timedelta(hours=1))

        await run_report()

        report = _report(bot_api)
        assert "у найзавантаженіший день (пт 23.10) 4 з 500 запитів" in report
        assert "оцінка" in report
        assert "повторні спроби та evals не враховано" in report

    async def test_no_gemini_request_is_made(
        self, run_report: RunReport, storage: InMemoryStorage
    ) -> None:
        await _seed(storage, "rz24/1", Decision.POSTED, _TUESDAY_NOON)
        with respx.mock(assert_all_called=False) as router:
            telegram = router.post(_SEND_URL).mock(
                return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
            )
            everything_else = router.route().mock(return_value=httpx.Response(500))

            code = await run_report()

        assert code == 0
        assert telegram.called
        assert not everything_else.called


class TestDigestStatus:
    _WEEKEND_FRIDAY = date(2026, 10, 23)

    async def test_a_published_digest_with_its_item_count(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await storage.start_digest(self._WEEKEND_FRIDAY, item_count=7)
        await storage.mark_digest_published(self._WEEKEND_FRIDAY, message_id=55)

        await run_report()

        assert "Дайджест 23–25.10: опубліковано, подій 7" in _report(bot_api)

    async def test_a_digest_left_publishing_is_a_warning(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await storage.start_digest(self._WEEKEND_FRIDAY, item_count=5)

        await run_report()

        assert (
            "⚠️ Дайджест 23–25.10: доставку не підтверджено (подій 5), перевір канал подій"
        ) in _report(bot_api)

    async def test_no_digest_recorded(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        # Another weekend's Digest is not this week's
        await storage.start_digest(date(2026, 10, 30), item_count=5)

        await run_report()

        assert "Дайджест 23–25.10: не опубліковано, запису немає" in _report(bot_api)


class TestPublishedTrend:
    async def _published(self, storage: InMemoryStorage, this_week: int, last_week: int) -> None:
        for i in range(this_week):
            await _seed(storage, f"rz24/NOW{i}", Decision.POSTED, _TUESDAY_NOON)
        for i in range(last_week):
            await _seed(storage, f"rz24/BEFORE{i}", Decision.POSTED, _PREVIOUS_WEEK)

    async def test_a_fall_by_more_than_half_is_a_warning(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await self._published(storage, this_week=4, last_week=9)

        await run_report()

        assert "⚠️ Опубліковано: 4, минулого тижня 9: менше ніж половина" in _report(bot_api)

    @pytest.mark.parametrize(("this_week", "last_week"), [(5, 10), (9, 9), (12, 3), (0, 0)])
    async def test_no_warning_otherwise(
        self,
        run_report: RunReport,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        this_week: int,
        last_week: int,
    ) -> None:
        await self._published(storage, this_week, last_week)

        await run_report()

        report = _report(bot_api)
        assert f"Опубліковано: {this_week}, минулого тижня {last_week}" in report
        assert "⚠️ Опубліковано" not in report
        assert "менше ніж половина" not in report


class TestModes:
    async def test_dry_run_adds_a_banner(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _seed(storage, "rz24/1", Decision.POSTED, _TUESDAY_NOON)

        code = await run_report(dry_run=True)

        assert code == 0
        assert bot_api.posts == [_ADMIN]
        assert _report(bot_api).startswith("<b>🧪 DRY RUN</b>\n\n<b>📊 Тижневий звіт, 19–25.10</b>")
        assert "<b>rzeszow24/najnowsze</b>: нових 1 · опубліковано 1" in _report(bot_api)

    async def test_staging_reads_the_staging_database(
        self,
        monkeypatch: pytest.MonkeyPatch,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
    ) -> None:
        await _seed(storage, "rz24/1", Decision.POSTED, _TUESDAY_NOON)
        opened: list[str] = []

        def _create_storage(**kw: object) -> InMemoryStorage:
            opened.append(str(kw["database_url"]))
            return storage

        monkeypatch.setattr(main_mod, "get_settings", _settings)
        monkeypatch.setattr(main_mod, "load_flow_config", _flow_config)
        monkeypatch.setattr(main_mod, "create_storage", _create_storage)

        code = await main_mod._async_main(weekly_report=True, staging=True, today=_MONDAY)

        assert code == 0
        assert opened == ["libsql://staging.turso.io"]
        assert bot_api.posts == [_ADMIN]
        assert _report(bot_api).startswith("<b>🧪 STAGING</b>\n\n")

    async def test_without_an_admin_chat_nothing_is_sent(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _seed(storage, "rz24/1", Decision.POSTED, _TUESDAY_NOON)

        code = await run_report(settings=_settings(telegram_admin_chat_id=None))

        assert code == 0
        assert bot_api.posts == []

    async def test_a_failed_send_exits_non_zero(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        bot_api.failing[_ADMIN] = 400

        assert await run_report() == 1

    async def test_a_long_report_is_split_into_several_messages(
        self, run_report: RunReport, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        # Rows under many unknown prefixes give one Source line each
        for i in range(120):
            await _seed(storage, f"source{i:03d}/1", Decision.POSTED, _TUESDAY_NOON)

        code = await run_report()

        assert code == 0
        assert len(bot_api.texts[_ADMIN]) > 1
        assert all(len(text) <= 4096 for text in bot_api.texts[_ADMIN])
        report = _report(bot_api)
        assert "<b>source000</b>: нових 1" in report
        assert "<b>source119</b>: нових 1" in report
        assert "Дайджест 23–25.10" in bot_api.texts[_ADMIN][-1]

    async def test_a_storage_failure_alerts_the_admin_and_exits_non_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        run_report: RunReport,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
    ) -> None:
        async def _broken(*args: object) -> None:
            raise RuntimeError("database is locked")

        monkeypatch.setattr(storage, "period_counts", _broken)

        code = await run_report()

        assert code == 1
        assert bot_api.posts == [_ADMIN]
        assert "Тижневий звіт не зібрано: RuntimeError: database is locked" in _report(bot_api)
