"""The Digest from the outside: ``rz-flow --digest`` on seeded storage.

Storage is an InMemoryStorage seeded with Published events, the Calendar's per-day
endpoint and the Bot API are mocked with respx, the AI filter is mocked, and "today"
is injected. Tests assert what a reader or the admin would
notice: which chat got which message, and what storage says afterwards.
"""

import json
from collections.abc import Awaitable, Iterator
from datetime import date
from typing import Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import rz_flow.main as main_mod
from rz_flow.ai import GeminiQuotaExhaustedError, GeminiUnavailableError
from rz_flow.config import Settings
from rz_flow.flow_config import DigestConfig, FlowConfig, PipelineConfig, SourceConfig
from rz_flow.models import (
    AIDecision,
    Article,
    CalendarVerdict,
    Category,
    CategoryTag,
    Decision,
    DigestStatus,
)
from rz_flow.storage import InMemoryStorage
from tests.test_pipeline_runs import FakeBotApi

_TOKEN = "fake:token"
_MAIN = "-100111"
_ADMIN = "-100222"
_EVENTS = "-1003330001"
_STAGING_MAIN = "-100444"
_STAGING_EVENTS = "-1005550002"
_SEND_URL = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
_CALENDAR_URL = "https://erzeszow.pl/index.php"
_CALENDAR_ENTRY_URL = "https://erzeszow.pl/41-miasto-rzeszow/6241-kalendarz-imprez/{id}-e.html"

_WEDNESDAY = date(2026, 10, 28)
_FRIDAY = date(2026, 10, 30)
_SATURDAY = date(2026, 10, 31)
_SUNDAY = date(2026, 11, 1)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "telegram_bot_token": _TOKEN,
        "telegram_channel_id": _MAIN,
        "telegram_admin_chat_id": _ADMIN,
        "telegram_events_channel_id": _EVENTS,
        "telegram_staging_channel_id": _STAGING_MAIN,
        "telegram_staging_events_channel_id": _STAGING_EVENTS,
        "gemini_api_key": "fake-key",
        "turso_database_url": "libsql://fake.turso.io",
        "turso_auth_token": "fake-token",
        "turso_staging_database_url": "libsql://staging.turso.io",
        "turso_staging_auth_token": "fake-staging-token",
        "ai_min_score": 7.0,
        "app_env": "production",
    }
    return Settings(_env_file=None, **{**values, **overrides})  # type: ignore[call-arg]


def _flow_config(*, calendar: bool) -> FlowConfig:
    return FlowConfig(
        sources=[SourceConfig(scraper="NajnowszeScraper", base_url="https://rzeszow24.info")],
        pipeline=PipelineConfig(gemini_model="gemini-test-model"),
        digest=DigestConfig(calendar_url=_CALENDAR_URL, enabled=calendar),
    )


async def _publish_event(
    storage: InMemoryStorage,
    article_id: str,
    start: date | None,
    end: date | None = None,
    *,
    title: str | None = None,
    events_message_id: int | None = 10,
    decision: Decision = Decision.POSTED,
) -> Article:
    """Seed one Article the AI judged an event, as a Run would have stored it."""
    article = Article(
        id=f"rz24/{article_id}",
        url=f"https://rzeszow24.info/imprezy/{article_id}",
        category=Category.IMPREZY,
        title_pl=f"Wydarzenie {article_id}",
        source_name="rzeszow24/najnowsze",
    )
    ai = AIDecision(
        is_interesting=True,
        is_event=True,
        score=8.0,
        category_tag=CategoryTag.CONCERT,
        ua_title=title or f"Подія {article_id}",
        ua_summary="Опис.",
        reason="Event",
        event_start=start,
        event_end=end,
    )
    await storage.save_decision(
        article, decision, ai, tg_message_id=1, tg_events_message_id=events_message_id
    )
    return article


def _calendar_day(*entries: tuple[int, str, str]) -> httpx.Response:
    """A per-day Calendar response listing (id, Polish title, dates) entries."""
    articles = "".join(
        f'<article><h2><a href="/41-miasto-rzeszow/6241-kalendarz-imprez/{i}-e.html">'
        f'{title}</a></h2><div class="modal__event--data">{dates}</div></article>'
        for i, title, dates in entries
    )
    html = f'<div id="events-oneday"></div><div id="events-multidays">{articles}</div>'
    return httpx.Response(200, text=json.dumps({"html": html}))


def _mock_calendar(router: respx.MockRouter, *friday_entries: tuple[int, str, str]) -> None:
    """The Calendar of the 30.10–1.11 weekend: *friday_entries* listed on Friday, and
    nothing else on Saturday or Sunday.
    """
    days = [("10-2026", "30", friday_entries), ("10-2026", "31", ()), ("11-2026", "1", ())]
    for month, day, entries in days:
        router.get(
            _CALENDAR_URL,
            params={"action": "disp-wydarzenia-dzien", "dataw": month, "dzien": day},
        ).mock(return_value=_calendar_day(*entries))


def _verdict(
    entry_id: int, ua_title: str, *, interesting: bool = True, same_as: str | None = None
) -> CalendarVerdict:
    return CalendarVerdict(
        id=entry_id, ua_title=ua_title, is_interesting=interesting, same_as=same_as
    )


class RunDigest(Protocol):
    def __call__(
        self,
        today: date,
        *,
        dry_run: bool = False,
        staging: bool = False,
        settings: Settings | None = None,
        calendar: bool = False,
    ) -> Awaitable[int]: ...


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock_router:
        yield mock_router


@pytest.fixture
def bot_api(router: respx.MockRouter) -> FakeBotApi:
    api = FakeBotApi()
    router.post(_SEND_URL).mock(side_effect=api.handle)
    return api


@pytest.fixture
def judge() -> Iterator[AsyncMock]:
    """The AI filter's Digest request; answers with no verdicts unless a test says."""
    with patch("rz_flow.digest.GeminiAIFilter") as ai_cls:
        ai = MagicMock()
        ai.judge_calendar = AsyncMock(return_value=[])
        ai_cls.return_value = ai
        yield ai.judge_calendar


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def run_digest(
    monkeypatch: pytest.MonkeyPatch, storage: InMemoryStorage, judge: AsyncMock
) -> RunDigest:
    """``rz-flow --digest`` with "today" injected, on the seeded storage; the Calendar
    is read only when a test asks for it.
    """

    async def _run(
        today: date,
        *,
        dry_run: bool = False,
        staging: bool = False,
        settings: Settings | None = None,
        calendar: bool = False,
    ) -> int:
        monkeypatch.setattr(main_mod, "get_settings", lambda: settings or _settings())
        monkeypatch.setattr(main_mod, "load_flow_config", lambda: _flow_config(calendar=calendar))
        monkeypatch.setattr(main_mod, "create_storage", lambda **kw: storage)
        return await main_mod._async_main(
            digest=True, dry_run=dry_run, staging=staging, today=today
        )

    return _run


async def _three_saturday_events(storage: InMemoryStorage) -> None:
    for article_id in ("A", "B", "C"):
        await _publish_event(storage, article_id, _SATURDAY)


class TestWindow:
    async def test_on_a_weekday_the_digest_covers_the_coming_weekend(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)

        code = await run_digest(_WEDNESDAY)

        assert code == 0
        assert bot_api.posts == [_EVENTS]
        assert bot_api.texts[_EVENTS][0].startswith("<b>🗓 Вихідні в Жешові, 30.10–1.11</b>")
        record = await storage.get_digest(_FRIDAY)
        assert record is not None
        assert (record.status, record.item_count) == (DigestStatus.PUBLISHED, 3)
        assert record.tg_message_id is not None

    async def test_on_saturday_the_digest_covers_the_current_weekend(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)

        await run_digest(_SATURDAY)

        assert bot_api.texts[_EVENTS][0].startswith("<b>🗓 Вихідні в Жешові, 30.10–1.11</b>")
        assert await storage.get_digest(_FRIDAY) is not None

    async def test_on_monday_the_digest_covers_the_next_weekend(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        for article_id in ("A", "B", "C"):
            await _publish_event(storage, article_id, date(2026, 11, 7))

        await run_digest(date(2026, 11, 2))

        assert bot_api.texts[_EVENTS][0].startswith("<b>🗓 Вихідні в Жешові, 6–8.11</b>")
        assert await storage.get_digest(date(2026, 11, 6)) is not None


class TestContent:
    async def test_items_overlapping_the_weekend_listed_once_under_their_first_day(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _publish_event(
            storage,
            "EXHIBITION",
            date(2026, 10, 28),
            date(2026, 10, 30),
            title="Виставка",
            events_message_id=11,
        )
        await _publish_event(
            storage, "FESTIVAL", _FRIDAY, _SUNDAY, title="Фестиваль", events_message_id=12
        )
        await _publish_event(storage, "CONCERT", _SATURDAY, title="Концерт", events_message_id=13)
        await _publish_event(
            storage, "FAIR", _SUNDAY, title="Ярмарок & <базар>", events_message_id=14
        )
        await _publish_event(storage, "THURSDAY", date(2026, 10, 29), title="Четвер")
        await _publish_event(storage, "MONDAY", date(2026, 11, 2), title="Понеділок")
        await _publish_event(storage, "UNDATED", None, title="Без дати")
        await _publish_event(
            storage, "SKIPPED", _SATURDAY, title="Пропущена", decision=Decision.SKIPPED
        )

        await run_digest(_WEDNESDAY)

        assert bot_api.texts[_EVENTS] == [
            "<b>🗓 Вихідні в Жешові, 30.10–1.11</b>\n"
            "\n"
            "<b>Пʼятниця, 30.10</b>\n"
            '• <a href="https://t.me/c/3330001/11">Виставка</a> (28–30.10)\n'
            '• <a href="https://t.me/c/3330001/12">Фестиваль</a> (30.10–1.11)\n'
            "\n"
            "<b>Субота, 31.10</b>\n"
            '• <a href="https://t.me/c/3330001/13">Концерт</a>\n'
            "\n"
            "<b>Неділя, 1.11</b>\n"
            '• <a href="https://t.me/c/3330001/14">Ярмарок &amp; &lt;базар&gt;</a>'
        ]

    async def test_item_without_an_events_channel_post_links_to_its_article(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)
        await _publish_event(
            storage, "NO_POST", _SATURDAY, title="Без поста", events_message_id=None
        )

        await run_digest(_WEDNESDAY)

        text = bot_api.texts[_EVENTS][0]
        assert '<a href="https://rzeszow24.info/imprezy/NO_POST">Без поста</a>' in text
        assert text.count("https://t.me/c/3330001/10") == 3

    async def test_digest_has_link_previews_off(
        self, run_digest: RunDigest, storage: InMemoryStorage
    ) -> None:
        await _three_saturday_events(storage)
        with respx.mock() as router:
            route = router.post(_SEND_URL).mock(
                return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 5}})
            )
            await run_digest(_WEDNESDAY)

        body = json.loads(route.calls.last.request.content)
        assert (body["parse_mode"], body["link_preview_options"]) == ("HTML", {"is_disabled": True})


class TestMinimum:
    async def test_fewer_than_three_items_posts_nothing_and_alerts_the_admin(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _publish_event(storage, "A", _SATURDAY)
        await _publish_event(storage, "B", _SUNDAY)

        code = await run_digest(_WEDNESDAY)

        assert code == 0
        assert bot_api.posts == [_ADMIN]
        assert "2" in bot_api.texts[_ADMIN][0]
        assert "30.10–1.11" in bot_api.texts[_ADMIN][0]
        assert await storage.get_digest(_FRIDAY) is None


class TestAtMostOnce:
    async def test_second_run_for_the_same_weekend_posts_nothing(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)
        await run_digest(_WEDNESDAY)
        await _publish_event(storage, "LATE", _SUNDAY)

        code = await run_digest(_SATURDAY)

        assert code == 0
        assert bot_api.posts == [_EVENTS]
        record = await storage.get_digest(_FRIDAY)
        assert record is not None
        assert (record.status, record.item_count) == (DigestStatus.PUBLISHED, 3)

    @pytest.mark.parametrize(
        "failure",
        [
            500,
            502,
            200,
            httpx.ReadTimeout("read timed out"),
            httpx.WriteTimeout("write timed out"),
        ],
        ids=["500", "502", "200-not-ok", "read-timeout", "write-timeout"],
    )
    async def test_uncertain_delivery_keeps_the_record_and_a_rerun_posts_nothing(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        failure: int | Exception,
    ) -> None:
        await _three_saturday_events(storage)
        bot_api.failing[_EVENTS] = failure

        code = await run_digest(_WEDNESDAY)

        assert code == 1
        record = await storage.get_digest(_FRIDAY)
        assert record is not None
        assert record.status == DigestStatus.PUBLISHING
        assert len(bot_api.texts[_ADMIN]) == 1
        assert "перевір" in bot_api.texts[_ADMIN][0].lower()

        del bot_api.failing[_EVENTS]
        await run_digest(_WEDNESDAY)

        assert bot_api.posts == [_EVENTS, _ADMIN]

    @pytest.mark.parametrize(
        "failure",
        [400, 403, 429, httpx.ConnectError("no route"), httpx.ConnectTimeout("connect")],
        ids=["400", "403", "429", "connect-error", "connect-timeout"],
    )
    async def test_definite_non_delivery_lets_a_rerun_post(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        failure: int | Exception,
    ) -> None:
        await _three_saturday_events(storage)
        bot_api.failing[_EVENTS] = failure

        code = await run_digest(_WEDNESDAY)

        assert code == 1
        assert await storage.get_digest(_FRIDAY) is None
        assert len(bot_api.texts[_ADMIN]) == 1

        del bot_api.failing[_EVENTS]
        code = await run_digest(_WEDNESDAY)

        assert code == 0
        assert bot_api.posts == [_EVENTS, _ADMIN, _EVENTS]
        record = await storage.get_digest(_FRIDAY)
        assert record is not None
        assert record.status == DigestStatus.PUBLISHED

    async def test_digest_sent_but_not_marked_published_stays_publishing_and_alerts(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _three_saturday_events(storage)

        async def _storage_down(weekend: date, message_id: int) -> None:
            raise ConnectionError("turso down")

        monkeypatch.setattr(storage, "mark_digest_published", _storage_down)

        code = await run_digest(_WEDNESDAY)

        assert code == 1
        assert bot_api.posts == [_EVENTS, _ADMIN]
        record = await storage.get_digest(_FRIDAY)
        assert record is not None
        assert record.status == DigestStatus.PUBLISHING

    async def test_storage_failure_before_sending_posts_nothing_and_alerts(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _storage_down(start: date, end: date) -> None:
            raise ConnectionError("turso down")

        monkeypatch.setattr(storage, "published_events_between", _storage_down)

        code = await run_digest(_WEDNESDAY)

        assert code == 1
        assert bot_api.posts == [_ADMIN]
        assert "ConnectionError" in bot_api.texts[_ADMIN][0]


class TestLength:
    async def test_overlong_digest_is_one_message_ending_with_the_dropped_count(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        total = 60
        for n in range(total):
            day = (_FRIDAY, _SATURDAY, _SUNDAY)[n % 3]
            await _publish_event(storage, f"E{n:02d}", day, title=f"Подія {n:02d} " + "х" * 80)

        await run_digest(_WEDNESDAY)

        assert bot_api.posts == [_EVENTS]
        text = bot_api.texts[_EVENTS][0]
        assert len(text) <= 4096
        shown = text.count("<a href=")
        last_line = text.rsplit("\n", 1)[-1]
        assert last_line == f"…і ще {total - shown}"
        assert 0 < shown < total
        # Dropped from the end: Sunday's last item goes before Friday's first
        assert "Подія 00" in text
        assert "Подія 59" not in text
        record = await storage.get_digest(_FRIDAY)
        assert record is not None
        assert record.item_count == shown


class TestModes:
    async def test_dry_run_sends_the_text_to_the_admin_chat_and_writes_nothing(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)

        code = await run_digest(_WEDNESDAY, dry_run=True)

        assert code == 0
        assert bot_api.posts == [_ADMIN]
        text = bot_api.texts[_ADMIN][0]
        assert "DRY RUN" in text.split("\n", 1)[0]
        assert "🗓 Вихідні в Жешові, 30.10–1.11" in text
        assert await storage.get_digest(_FRIDAY) is None

    async def test_dry_run_without_an_admin_chat_sends_nothing(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)

        code = await run_digest(
            _WEDNESDAY, dry_run=True, settings=_settings(telegram_admin_chat_id=None)
        )

        assert code == 0
        assert bot_api.posts == []
        assert await storage.get_digest(_FRIDAY) is None

    async def test_staging_posts_to_the_staging_events_channel_only(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)

        code = await run_digest(_WEDNESDAY, staging=True)

        assert code == 0
        assert bot_api.posts == [_STAGING_EVENTS]
        text = bot_api.texts[_STAGING_EVENTS][0]
        assert "STAGING" in text.split("\n", 1)[0]
        assert "https://t.me/c/5550002/10" in text
        assert _EVENTS not in bot_api.texts

    async def test_staging_without_a_staging_events_channel_never_falls_back(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)

        code = await run_digest(
            _WEDNESDAY,
            staging=True,
            settings=_settings(telegram_staging_events_channel_id=None),
        )

        assert code == 0
        assert bot_api.posts == []
        assert await storage.get_digest(_FRIDAY) is None

    async def test_without_an_events_channel_the_digest_is_skipped(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi
    ) -> None:
        await _three_saturday_events(storage)

        code = await run_digest(_WEDNESDAY, settings=_settings(telegram_events_channel_id=None))

        assert code == 0
        assert bot_api.posts == []
        assert await storage.get_digest(_FRIDAY) is None


class TestCalendar:
    async def test_interesting_entries_listed_in_ukrainian_others_and_repeats_dropped(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
    ) -> None:
        await _publish_event(storage, "LIGHTS", _FRIDAY, _SUNDAY, title="Фестиваль світла")
        await _publish_event(storage, "B", _SATURDAY)
        _mock_calendar(
            router,
            (501, "Koncert jesienny", "2026-10-30"),
            (502, "Sesja Rady Miasta", "2026-10-30"),
            (503, "Festiwal Światła", "2026-10-30 - 2026-11-01"),
        )
        judge.return_value = [
            _verdict(501, "Осінній концерт"),
            _verdict(502, "Сесія міськради", interesting=False),
            _verdict(503, "Фестиваль світла (календар)", same_as="rz24/LIGHTS"),
        ]

        code = await run_digest(_WEDNESDAY, calendar=True)

        assert code == 0
        assert judge.await_count == 1
        entries, news_events = judge.await_args.args
        assert [e.id for e in entries] == [501, 502, 503]
        assert [e.id for e in news_events] == ["rz24/LIGHTS", "rz24/B"]
        assert bot_api.posts == [_EVENTS]
        text = bot_api.texts[_EVENTS][0]
        assert (
            "<b>Пʼятниця, 30.10</b>\n"
            '• <a href="https://t.me/c/3330001/10">Фестиваль світла</a> (30.10–1.11)\n'
            f'• <a href="{_CALENDAR_ENTRY_URL.format(id=501)}">Осінній концерт</a>\n'
        ) in text
        assert "Сесія міськради" not in text
        assert "(календар)" not in text
        record = await storage.get_digest(_FRIDAY)
        assert record is not None
        assert record.item_count == 3

    async def test_no_gemini_request_when_the_calendar_is_off(
        self, run_digest: RunDigest, storage: InMemoryStorage, bot_api: FakeBotApi, judge: AsyncMock
    ) -> None:
        await _three_saturday_events(storage)

        await run_digest(_WEDNESDAY)

        assert judge.await_count == 0
        assert bot_api.posts == [_EVENTS]

    async def test_no_gemini_request_without_entries_for_the_weekend(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
    ) -> None:
        await _three_saturday_events(storage)
        # A months-long exhibition that opened before this weekend
        _mock_calendar(router, (601, "Wystawa", "2026-09-01 - 2027-01-31"))

        await run_digest(_WEDNESDAY, calendar=True)

        assert judge.await_count == 0
        assert bot_api.posts == [_EVENTS]

    async def test_calendar_entries_count_towards_the_minimum(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
    ) -> None:
        await _publish_event(storage, "A", _SATURDAY)
        _mock_calendar(router, (701, "Koncert", "2026-10-30"), (702, "Kino", "2026-10-31"))
        judge.return_value = [_verdict(701, "Концерт"), _verdict(702, "Кіно")]

        await run_digest(_WEDNESDAY, calendar=True)

        assert bot_api.posts == [_EVENTS]
        text = bot_api.texts[_EVENTS][0]
        assert (
            "<b>Пʼятниця, 30.10</b>\n"
            f'• <a href="{_CALENDAR_ENTRY_URL.format(id=701)}">Концерт</a>\n'
            "\n"
            "<b>Субота, 31.10</b>\n"
            '• <a href="https://t.me/c/3330001/10">Подія A</a>\n'
            f'• <a href="{_CALENDAR_ENTRY_URL.format(id=702)}">Кіно</a>'
        ) in text

    @pytest.mark.parametrize(
        "failure",
        [
            GeminiQuotaExhaustedError("daily quota"),
            GeminiUnavailableError("503"),
            ValueError("Empty Gemini response"),
        ],
        ids=["quota-exhausted", "unavailable", "bad-answer"],
    )
    async def test_ai_failure_yields_a_news_only_digest_and_an_alert(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
        failure: Exception,
    ) -> None:
        await _three_saturday_events(storage)
        _mock_calendar(router, (801, "Koncert", "2026-10-30"))
        judge.side_effect = failure

        code = await run_digest(_WEDNESDAY, calendar=True)

        assert code == 0
        assert judge.await_count == 1
        assert sorted(bot_api.posts) == sorted([_EVENTS, _ADMIN])
        assert "календар" not in bot_api.texts[_EVENTS][0].lower()
        assert bot_api.texts[_EVENTS][0].count("<a href=") == 3
        alert = bot_api.texts[_ADMIN][0]
        assert type(failure).__name__ in alert
        assert "лише з подіями з новин" in alert

    async def test_calendar_failure_yields_a_news_only_digest_and_an_alert(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
    ) -> None:
        await _three_saturday_events(storage)
        router.get(_CALENDAR_URL).mock(return_value=httpx.Response(500))

        code = await run_digest(_WEDNESDAY, calendar=True)

        assert code == 0
        assert judge.await_count == 0
        assert bot_api.texts[_EVENTS][0].count("<a href=") == 3
        alert = bot_api.texts[_ADMIN][0]
        assert "Календар подій" in alert
        assert "лише з подіями з новин" in alert

    async def test_overlong_digest_drops_calendar_entries_before_any_news_event(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
    ) -> None:
        news = 25
        for n in range(news):
            await _publish_event(storage, f"E{n:02d}", _SUNDAY, title=f"Подія {n:02d} " + "х" * 80)
        calendar = 10
        _mock_calendar(
            router, *[(900 + n, f"Wydarzenie {n}", "2026-10-30") for n in range(calendar)]
        )
        judge.return_value = [
            _verdict(900 + n, f"Календарна {n} " + "к" * 80) for n in range(calendar)
        ]

        await run_digest(_WEDNESDAY, calendar=True)

        text = bot_api.texts[_EVENTS][0]
        assert len(text) <= 4096
        assert all(f"Подія {n:02d} " in text for n in range(news))
        shown_calendar = text.count("kalendarz-imprez")
        assert 0 < shown_calendar < calendar
        # Dropped from the end: the first Calendar entries stay
        assert "Календарна 0 " in text
        assert f"Календарна {calendar - 1} " not in text
        assert text.rsplit("\n", 1)[-1] == f"…і ще {calendar - shown_calendar}"

    async def test_overlong_digest_without_room_for_calendar_entries_cuts_news_from_the_end(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
    ) -> None:
        news = 60
        for n in range(news):
            day = (_FRIDAY, _SATURDAY, _SUNDAY)[n % 3]
            await _publish_event(storage, f"E{n:02d}", day, title=f"Подія {n:02d} " + "х" * 80)
        _mock_calendar(router, (1101, "Koncert", "2026-10-30"), (1102, "Kino", "2026-10-30"))
        judge.return_value = [_verdict(1101, "Концерт"), _verdict(1102, "Кіно")]

        await run_digest(_WEDNESDAY, calendar=True)

        text = bot_api.texts[_EVENTS][0]
        assert len(text) <= 4096
        assert "kalendarz-imprez" not in text
        shown = text.count("<a href=")
        assert 0 < shown < news
        assert "Подія 00" in text
        assert "Подія 59" not in text
        assert text.rsplit("\n", 1)[-1] == f"…і ще {news + 2 - shown}"

    async def test_dry_run_still_makes_the_one_gemini_request(
        self,
        run_digest: RunDigest,
        storage: InMemoryStorage,
        bot_api: FakeBotApi,
        router: respx.MockRouter,
        judge: AsyncMock,
    ) -> None:
        await _three_saturday_events(storage)
        _mock_calendar(router, (1001, "Koncert", "2026-10-30"))
        judge.return_value = [_verdict(1001, "Концерт")]

        code = await run_digest(_WEDNESDAY, dry_run=True, calendar=True)

        assert code == 0
        assert judge.await_count == 1
        assert bot_api.posts == [_ADMIN]
        assert "Концерт" in bot_api.texts[_ADMIN][0]
        assert await storage.get_digest(_FRIDAY) is None
