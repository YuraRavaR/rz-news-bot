"""The Digest from the outside: ``rz-flow --digest`` on seeded storage.

Storage is an InMemoryStorage seeded with Published events, the Bot API is mocked
with respx, and "today" is injected. Tests assert what a reader or the admin would
notice: which chat got which message, and what storage says afterwards.
"""

import json
from collections.abc import Awaitable, Iterator
from datetime import date
from typing import Protocol

import httpx
import pytest
import respx

import rz_flow.main as main_mod
from rz_flow.config import Settings
from rz_flow.flow_config import FlowConfig, PipelineConfig, SourceConfig
from rz_flow.models import AIDecision, Article, Category, CategoryTag, Decision, DigestStatus
from rz_flow.storage import InMemoryStorage
from tests.test_pipeline_runs import FakeBotApi

_TOKEN = "fake:token"
_MAIN = "-100111"
_ADMIN = "-100222"
_EVENTS = "-1003330001"
_STAGING_MAIN = "-100444"
_STAGING_EVENTS = "-1005550002"
_SEND_URL = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"

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


def _flow_config() -> FlowConfig:
    return FlowConfig(
        sources=[SourceConfig(scraper="NajnowszeScraper", base_url="https://rzeszow24.info")],
        pipeline=PipelineConfig(gemini_model="gemini-test-model"),
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


class RunDigest(Protocol):
    def __call__(
        self,
        today: date,
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
def run_digest(monkeypatch: pytest.MonkeyPatch, storage: InMemoryStorage) -> RunDigest:
    """``rz-flow --digest`` with "today" injected, on the seeded storage."""

    async def _run(
        today: date,
        *,
        dry_run: bool = False,
        staging: bool = False,
        settings: Settings | None = None,
    ) -> int:
        monkeypatch.setattr(main_mod, "get_settings", lambda: settings or _settings())
        monkeypatch.setattr(main_mod, "load_flow_config", _flow_config)
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
