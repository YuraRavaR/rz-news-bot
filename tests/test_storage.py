"""Tests for the storage layer.

Two levels:
  - InMemoryStorage covers protocol behaviour with zero I/O.
  - TursoStorage runs against a local libsql file, so the actual SQL (DDL, migrations,
    the IN (?) builder, the UPSERT) is executed rather than assumed. Same client
    library as production, no network and no credentials.
"""

import pathlib
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta

import pytest

from rz_flow import storage as storage_module
from rz_flow.models import AIDecision, Article, Category, CategoryTag, Decision
from rz_flow.storage import InMemoryStorage, TursoStorage, create_storage

_HOUR = timedelta(hours=1)

# The original May 2026 schema, before ua_*/is_event/error_attempts existed.
_LEGACY_POSTS_TABLE = """
CREATE TABLE posts (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    category      TEXT NOT NULL,
    title_pl      TEXT NOT NULL,
    seen_at       TEXT NOT NULL,
    decision      TEXT NOT NULL,
    ai_score      REAL,
    ai_reason     TEXT,
    tg_message_id INTEGER
)
"""


def _make_article(article_id: str = "TESTID123456789") -> Article:
    return Article(
        id=article_id,
        url=f"https://rzeszow24.info/imprezy/test/{article_id}",
        category=Category.IMPREZY,
        title_pl="Test Article",
        summary_pl="Test summary.",
    )


def _make_ai_decision(score: float = 8.0, is_event: bool = True) -> AIDecision:
    return AIDecision(
        is_interesting=True,
        is_event=is_event,
        score=score,
        category_tag=CategoryTag.FESTIVAL,
        ua_title="Тестова Стаття",
        ua_summary="Короткий опис події.",
        reason="Interesting local event",
    )


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


class TestInMemoryStorageInit:
    async def test_init_does_not_raise(self, storage: InMemoryStorage) -> None:
        await storage.init()
        assert storage.count() == 0

    async def test_close_does_not_raise(self, storage: InMemoryStorage) -> None:
        await storage.close()


class TestFilterNewIds:
    async def test_returns_all_ids_when_empty(self, storage: InMemoryStorage) -> None:
        ids = ["AAA", "BBB", "CCC"]
        result = await storage.filter_new_ids(ids)
        assert result == ids

    async def test_returns_empty_list_for_empty_input(self, storage: InMemoryStorage) -> None:
        result = await storage.filter_new_ids([])
        assert result == []

    async def test_filters_out_seen_ids(self, storage: InMemoryStorage) -> None:
        article = _make_article("SEEN_ID_123456789")
        await storage.save_decision(article, Decision.POSTED)

        result = await storage.filter_new_ids(["SEEN_ID_123456789", "NEW_ID_987654321"])
        assert result == ["NEW_ID_987654321"]

    async def test_preserves_order_of_new_ids(self, storage: InMemoryStorage) -> None:
        ids = ["FIRST123456789AB", "SECOND12345678AB", "THIRD1234567890A"]
        result = await storage.filter_new_ids(ids)
        assert result == ids


class TestSaveDecision:
    async def test_save_posted_with_ai_decision(self, storage: InMemoryStorage) -> None:
        article = _make_article()
        ai = _make_ai_decision(score=8.5)

        await storage.save_decision(article, Decision.POSTED, ai, tg_message_id=42)

        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.POSTED
        assert record.ai_score == 8.5
        assert record.tg_message_id == 42

    async def test_save_stores_ua_fields(self, storage: InMemoryStorage) -> None:
        """QW-4: Ukrainian AI output (ua_title, ua_summary, category_tag) is persisted."""
        article = _make_article()
        ai = _make_ai_decision()

        await storage.save_decision(article, Decision.POSTED, ai)

        record = storage.get_record(article.id)
        assert record is not None
        assert record.ua_title == ai.ua_title
        assert record.ua_summary == ai.ua_summary
        assert record.category_tag == ai.category_tag.value

    async def test_save_ua_fields_none_without_ai(self, storage: InMemoryStorage) -> None:
        """QW-4: ua fields are None when no AI decision is provided (e.g. error case)."""
        article = _make_article()
        await storage.save_decision(article, Decision.ERROR)

        record = storage.get_record(article.id)
        assert record is not None
        assert record.ua_title is None
        assert record.ua_summary is None
        assert record.category_tag is None

    async def test_save_skipped_without_ai(self, storage: InMemoryStorage) -> None:
        article = _make_article()
        await storage.save_decision(article, Decision.SKIPPED)

        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.SKIPPED
        assert record.ai_score is None
        assert record.tg_message_id is None

    async def test_save_error_decision(self, storage: InMemoryStorage) -> None:
        article = _make_article()
        await storage.save_decision(article, Decision.ERROR)

        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.ERROR

    async def test_overwrite_existing_record(self, storage: InMemoryStorage) -> None:
        """INSERT OR REPLACE semantics: re-saving an ID updates the record."""
        article = _make_article()
        await storage.save_decision(article, Decision.ERROR)
        await storage.save_decision(article, Decision.POSTED, _make_ai_decision())

        assert storage.count() == 1
        record = storage.get_record(article.id)
        assert record is not None
        assert record.decision == Decision.POSTED

    async def test_all_records_returns_all_saved(self, storage: InMemoryStorage) -> None:
        for i in range(3):
            article = _make_article(f"ARTICLE_ID_{i:05d}XYZABC")
            await storage.save_decision(article, Decision.POSTED)

        assert storage.count() == 3
        assert len(storage.all_records()) == 3

    async def test_filter_excludes_after_save(self, storage: InMemoryStorage) -> None:
        """Once saved, article should not appear in filter_new_ids results."""
        article = _make_article("UNIQUE_ID_12345678")
        await storage.save_decision(article, Decision.SKIPPED)

        new_ids = await storage.filter_new_ids([article.id])
        assert new_ids == []


class TestStorageProtocolCompliance:
    """Verify InMemoryStorage satisfies StorageProtocol at runtime."""

    def test_implements_protocol(self, storage: InMemoryStorage) -> None:
        from rz_flow.storage import StorageProtocol

        assert isinstance(storage, StorageProtocol)


class TestTursoStorageNormalizeUrl:
    """_normalize_url is a pure static method — no DB connection needed."""

    def test_converts_libsql_scheme_to_https(self) -> None:
        url = TursoStorage._normalize_url("libsql://my-db.turso.io")
        assert url == "https://my-db.turso.io"

    def test_leaves_https_url_unchanged(self) -> None:
        url = TursoStorage._normalize_url("https://my-db.turso.io")
        assert url == "https://my-db.turso.io"

    def test_leaves_other_schemes_unchanged(self) -> None:
        url = TursoStorage._normalize_url("wss://my-db.turso.io")
        assert url == "wss://my-db.turso.io"

    def test_conversion_preserves_path_and_query(self) -> None:
        url = TursoStorage._normalize_url("libsql://my-db.turso.io/some/path?key=val")
        assert url == "https://my-db.turso.io/some/path?key=val"


class TestCreateStorage:
    def test_create_storage_returns_turso_storage(self) -> None:
        storage = create_storage(
            database_url="libsql://fake.turso.io",
            auth_token="fake-token",
        )
        assert isinstance(storage, TursoStorage)

    def test_create_storage_normalizes_url(self) -> None:
        """create_storage should apply URL normalization internally."""
        storage = create_storage(
            database_url="libsql://fake.turso.io",
            auth_token="fake-token",
        )
        # Internal URL should be https:// after normalization
        assert storage._url.startswith("https://")

    def test_create_storage_passes_retry_budget(self) -> None:
        storage = create_storage(
            database_url="libsql://fake.turso.io",
            auth_token="fake-token",
            max_error_attempts=7,
        )
        assert storage._max_error_attempts == 7


# ── Error retry budget (shared semantics) ─────────────────────────────────────
class TestErrorRetryBudget:
    """A transient failure must not retire an article; a persistent one must."""

    async def test_errored_article_is_offered_again(self) -> None:
        storage = InMemoryStorage(max_error_attempts=3)
        article = _make_article("RETRY_ME_12345678")

        await storage.save_decision(article, Decision.ERROR)

        assert await storage.filter_new_ids([article.id]) == [article.id]

    async def test_article_retires_once_budget_is_spent(self) -> None:
        storage = InMemoryStorage(max_error_attempts=3)
        article = _make_article("GIVE_UP_123456789")

        for _ in range(3):
            await storage.save_decision(article, Decision.ERROR)

        assert await storage.filter_new_ids([article.id]) == []

    async def test_attempts_accumulate_across_saves(self) -> None:
        storage = InMemoryStorage()
        article = _make_article("COUNTER_123456789")

        await storage.save_decision(article, Decision.ERROR)
        await storage.save_decision(article, Decision.ERROR)

        record = storage.get_record(article.id)
        assert record is not None
        assert record.error_attempts == 2

    async def test_success_resets_attempts_and_settles(self) -> None:
        storage = InMemoryStorage(max_error_attempts=3)
        article = _make_article("RECOVERED_1234567")

        await storage.save_decision(article, Decision.ERROR)
        await storage.save_decision(article, Decision.POSTED, _make_ai_decision())

        record = storage.get_record(article.id)
        assert record is not None
        assert record.error_attempts == 0
        assert await storage.filter_new_ids([article.id]) == []

    async def test_budget_of_one_never_retries(self) -> None:
        storage = InMemoryStorage(max_error_attempts=1)
        article = _make_article("NO_RETRY_12345678")

        await storage.save_decision(article, Decision.ERROR)

        assert await storage.filter_new_ids([article.id]) == []

    async def test_posted_and_skipped_are_never_reoffered(self) -> None:
        storage = InMemoryStorage()
        posted = _make_article("POSTED_1234567890")
        skipped = _make_article("SKIPPED_123456789")

        await storage.save_decision(posted, Decision.POSTED, _make_ai_decision())
        await storage.save_decision(skipped, Decision.SKIPPED, _make_ai_decision())

        assert await storage.filter_new_ids([posted.id, skipped.id]) == []


# ── TursoStorage against a real local libsql database ─────────────────────────
@pytest.fixture
async def turso(tmp_path: pathlib.Path) -> AsyncIterator[TursoStorage]:
    """TursoStorage backed by a temp SQLite file — exercises the real SQL."""
    storage = TursoStorage(
        database_url=f"file:{tmp_path / 'test.db'}",
        auth_token="",
        max_error_attempts=3,
    )
    await storage.init()
    yield storage
    await storage.close()


class TestTursoStorageSql:
    async def test_init_is_idempotent(self, turso: TursoStorage) -> None:
        """init() runs on every start, so re-running it must be harmless."""
        await turso.init()
        await turso.init()

        assert await turso.filter_new_ids(["ANY_ID_123456789"]) == ["ANY_ID_123456789"]

    async def test_round_trip_posted_article(self, turso: TursoStorage) -> None:
        article = _make_article("ROUNDTRIP_1234567")
        ai = _make_ai_decision(score=9.5)

        await turso.save_decision(
            article, Decision.POSTED, ai, tg_message_id=11, tg_events_message_id=22
        )

        row = await _fetch_row(turso, article.id)
        assert row["decision"] == Decision.POSTED.value
        assert row["ai_score"] == 9.5
        assert row["tg_message_id"] == 11
        assert row["tg_events_message_id"] == 22
        assert row["ua_title"] == ai.ua_title
        assert row["ua_summary"] == ai.ua_summary
        assert row["category_tag"] == ai.category_tag.value
        assert row["is_event"] == 1

    async def test_save_without_ai_decision_writes_nulls(self, turso: TursoStorage) -> None:
        article = _make_article("NO_AI_12345678901")

        await turso.save_decision(article, Decision.ERROR)

        row = await _fetch_row(turso, article.id)
        assert row["ai_score"] is None
        assert row["ua_title"] is None
        assert row["is_event"] is None

    async def test_filter_new_ids_excludes_settled(self, turso: TursoStorage) -> None:
        seen = _make_article("SEEN_123456789012")
        await turso.save_decision(seen, Decision.POSTED, _make_ai_decision())

        result = await turso.filter_new_ids([seen.id, "FRESH_12345678901"])

        assert result == ["FRESH_12345678901"]

    async def test_filter_new_ids_preserves_input_order(self, turso: TursoStorage) -> None:
        ids = ["AAA_1234567890123", "BBB_1234567890123", "CCC_1234567890123"]
        assert await turso.filter_new_ids(ids) == ids

    async def test_filter_new_ids_handles_empty_input(self, turso: TursoStorage) -> None:
        assert await turso.filter_new_ids([]) == []

    async def test_filter_new_ids_handles_large_batch(self, turso: TursoStorage) -> None:
        """The IN (?) list is built from the batch size — check it holds up."""
        ids = [f"BULK_{i:012d}" for i in range(200)]
        assert await turso.filter_new_ids(ids) == ids

    async def test_upsert_overwrites_without_duplicating(self, turso: TursoStorage) -> None:
        article = _make_article("UPSERT_1234567890")

        await turso.save_decision(article, Decision.ERROR)
        await turso.save_decision(article, Decision.POSTED, _make_ai_decision())

        client = turso._get_client()
        result = await client.execute("SELECT COUNT(*) FROM posts WHERE id = ?", [article.id])
        assert result.rows[0][0] == 1

    async def test_error_attempts_increment_in_sql(self, turso: TursoStorage) -> None:
        """UPSERT must accumulate the counter — INSERT OR REPLACE would reset it."""
        article = _make_article("SQL_COUNTER_12345")

        await turso.save_decision(article, Decision.ERROR)
        assert (await _fetch_row(turso, article.id))["error_attempts"] == 1

        await turso.save_decision(article, Decision.ERROR)
        assert (await _fetch_row(turso, article.id))["error_attempts"] == 2

    async def test_error_attempts_reset_on_success_in_sql(self, turso: TursoStorage) -> None:
        article = _make_article("SQL_RESET_1234567")

        await turso.save_decision(article, Decision.ERROR)
        await turso.save_decision(article, Decision.POSTED, _make_ai_decision())

        assert (await _fetch_row(turso, article.id))["error_attempts"] == 0

    async def test_retryable_error_still_counts_as_new(self, turso: TursoStorage) -> None:
        article = _make_article("SQL_RETRY_1234567")
        await turso.save_decision(article, Decision.ERROR)

        assert await turso.filter_new_ids([article.id]) == [article.id]

    async def test_exhausted_error_stops_being_new(self, turso: TursoStorage) -> None:
        article = _make_article("SQL_EXHAUST_12345")
        for _ in range(3):
            await turso.save_decision(article, Decision.ERROR)

        assert await turso.filter_new_ids([article.id]) == []

    async def test_close_is_safe_to_call_twice(self, turso: TursoStorage) -> None:
        await turso.close()
        await turso.close()


class TestTursoStorageMigrations:
    """Existing databases predate several columns — init() must upgrade them in place."""

    async def test_adds_columns_to_legacy_table(self, tmp_path: pathlib.Path) -> None:
        db_path = f"file:{tmp_path / 'legacy.db'}"
        storage = TursoStorage(database_url=db_path, auth_token="")
        await storage._get_client().execute(_LEGACY_POSTS_TABLE)

        await storage.init()

        # A full save must now succeed against the upgraded table.
        await storage.save_decision(
            _make_article("MIGRATED_12345678"), Decision.POSTED, _make_ai_decision()
        )
        row = await _fetch_row(storage, "MIGRATED_12345678")
        assert row["ua_title"] is not None
        assert row["error_attempts"] == 0
        await storage.close()

    async def test_event_date_columns_migrate_onto_a_legacy_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        storage = TursoStorage(database_url=f"file:{tmp_path / 'legacy.db'}", auth_token="")
        await storage._get_client().execute(_LEGACY_POSTS_TABLE)
        await storage.init()

        await storage.save_decision(
            _make_article("DATED_LEGACY_1234"), Decision.POSTED, _dated_decision()
        )

        assert await _event_date(storage, "DATED_LEGACY_1234") == (
            date(2026, 10, 24),
            date(2026, 10, 26),
        )
        await storage.close()

    async def test_legacy_rows_survive_the_upgrade(self, tmp_path: pathlib.Path) -> None:
        storage = TursoStorage(database_url=f"file:{tmp_path / 'legacy.db'}", auth_token="")
        client = storage._get_client()
        await client.execute(_LEGACY_POSTS_TABLE)
        await client.execute(
            "INSERT INTO posts (id, url, category, title_pl, seen_at, decision) "
            "VALUES ('rz24/OLD', 'https://rzeszow24.info/x/OLD', 'wiadomosci', 'Stary', "
            "'2026-05-01T00:00:00+00:00', 'posted')"
        )

        await storage.init()

        row = await _fetch_row(storage, "rz24/OLD")
        assert row["title_pl"] == "Stary"
        assert row["decision"] == "posted"
        assert row["error_attempts"] == 0
        assert await storage.filter_new_ids(["rz24/OLD"]) == []
        await storage.close()

    async def test_second_init_runs_no_alter_table(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage = TursoStorage(database_url=f"file:{tmp_path / 'current.db'}", auth_token="")
        await storage.init()
        client = storage._get_client()
        executed: list[str] = []
        real_execute = client.execute

        async def _spy(sql: str, *args: object) -> object:
            executed.append(sql)
            return await real_execute(sql, *args)

        monkeypatch.setattr(client, "execute", _spy)

        await storage.init()

        assert not [sql for sql in executed if "ALTER TABLE" in sql.upper()]
        await storage.close()

    async def test_a_failing_migration_raises(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage = TursoStorage(database_url=f"file:{tmp_path / 'broken.db'}", auth_token="")
        await storage.init()
        # SQLite never lets ALTER TABLE add a PRIMARY KEY column.
        monkeypatch.setattr(
            storage_module,
            "_COLUMN_MIGRATIONS",
            [*storage_module._COLUMN_MIGRATIONS, ("broken", "INTEGER PRIMARY KEY")],
        )

        with pytest.raises(Exception, match="PRIMARY KEY"):
            await storage.init()
        await storage.close()

    async def test_fresh_database_gets_the_full_schema(self, tmp_path: pathlib.Path) -> None:
        storage = TursoStorage(database_url=f"file:{tmp_path / 'fresh.db'}", auth_token="")
        await storage.init()

        result = await storage._get_client().execute("PRAGMA table_info(posts)")
        columns = {row[1] for row in result.rows}

        assert columns == {
            "id",
            "url",
            "category",
            "title_pl",
            "seen_at",
            "decision",
            "ai_score",
            "ai_reason",
            "is_event",
            "tg_message_id",
            "tg_events_message_id",
            "ua_title",
            "ua_summary",
            "category_tag",
            "error_attempts",
            "repeat_of",
            "event_start",
            "event_end",
        }
        await storage.close()

    async def test_backfills_source_prefix_on_legacy_ids(self, tmp_path: pathlib.Path) -> None:
        """QW-2 data migration: un-prefixed IDs get their source prefix."""
        db_path = f"file:{tmp_path / 'prefix.db'}"
        storage = TursoStorage(database_url=db_path, auth_token="")
        await storage.init()
        client = storage._get_client()
        await client.execute(
            """
            INSERT INTO posts (id, url, category, title_pl, seen_at, decision)
            VALUES ('OLDSLUG123456789', 'https://rzeszow24.info/imprezy/x/OLDSLUG123456789',
                    'imprezy', 'Stary', '2026-05-01T00:00:00+00:00', 'posted')
            """
        )

        await storage.init()  # re-run migrations

        result = await client.execute("SELECT id FROM posts")
        assert result.rows[0][0] == "rz24/OLDSLUG123456789"
        await storage.close()

    async def test_prefix_backfill_is_idempotent(self, tmp_path: pathlib.Path) -> None:
        """Migrations run on every start — a prefixed ID must not gain a second one."""
        db_path = f"file:{tmp_path / 'idempotent.db'}"
        storage = TursoStorage(database_url=db_path, auth_token="")
        await storage.init()
        client = storage._get_client()
        await client.execute(
            """
            INSERT INTO posts (id, url, category, title_pl, seen_at, decision)
            VALUES ('rz24/ALREADY12345678',
                    'https://rzeszow24.info/imprezy/x/ALREADY12345678',
                    'imprezy', 'Juz', '2026-05-01T00:00:00+00:00', 'posted')
            """
        )

        await storage.init()
        await storage.init()

        result = await client.execute("SELECT id FROM posts")
        assert result.rows[0][0] == "rz24/ALREADY12345678"
        await storage.close()


async def _fetch_row(storage: TursoStorage, article_id: str) -> dict[str, object]:
    """Read one posts row as a column->value mapping."""
    client = storage._get_client()
    result = await client.execute("SELECT * FROM posts WHERE id = ?", [article_id])
    assert len(result.rows) == 1, f"expected exactly one row for {article_id}"
    return dict(zip(result.columns, result.rows[0], strict=True))


# ── Contract: both storages behave the same for Publishing / Unconfirmed ──────
@pytest.fixture(params=["memory", "turso"])
async def any_storage(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> AsyncIterator[InMemoryStorage | TursoStorage]:
    """Each contract test runs once per storage implementation."""
    if request.param == "memory":
        yield InMemoryStorage(max_error_attempts=3)
        return
    storage = TursoStorage(
        database_url=f"file:{tmp_path / 'contract.db'}",
        auth_token="",
        max_error_attempts=3,
    )
    await storage.init()
    yield storage
    await storage.close()


async def _outcome(
    storage: InMemoryStorage | TursoStorage, article_id: str
) -> tuple[Decision, int] | None:
    """(outcome, error attempts) of one Article, or None when there is no record."""
    if isinstance(storage, InMemoryStorage):
        record = storage.get_record(article_id)
        return (record.decision, record.error_attempts) if record else None
    client = storage._get_client()
    result = await client.execute(
        "SELECT decision, error_attempts FROM posts WHERE id = ?", [article_id]
    )
    if not result.rows:
        return None
    return Decision(result.rows[0][0]), int(result.rows[0][1])


class TestPublishingContract:
    async def test_mark_publishing_new_article(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("PUB_NEW_123456789")

        previous = await any_storage.mark_publishing(article)

        assert previous is None
        assert await _outcome(any_storage, article.id) == (Decision.PUBLISHING, 0)

    async def test_mark_publishing_error_article_keeps_attempts(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("PUB_ERR_123456789")
        await any_storage.save_decision(article, Decision.ERROR)
        await any_storage.save_decision(article, Decision.ERROR)

        previous = await any_storage.mark_publishing(article)

        assert previous is Decision.ERROR
        assert await _outcome(any_storage, article.id) == (Decision.PUBLISHING, 2)

    async def test_publishing_is_settled(self, any_storage: InMemoryStorage | TursoStorage) -> None:
        article = _make_article("PUB_SETTLED_12345")
        await any_storage.mark_publishing(article)

        assert await any_storage.filter_new_ids([article.id]) == []

    async def test_resolve_stale_publishing_turns_it_unconfirmed(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        stale = _make_article("PUB_STALE_1234567")
        posted = _make_article("PUB_POSTED_123456")
        await any_storage.mark_publishing(stale)
        await any_storage.save_decision(posted, Decision.POSTED, _make_ai_decision())

        resolved = await any_storage.resolve_stale_publishing()

        assert [(a.id, a.url, a.title_pl) for a in resolved] == [
            (stale.id, stale.url, stale.title_pl)
        ]
        assert await _outcome(any_storage, stale.id) == (Decision.UNCONFIRMED, 0)
        assert await _outcome(any_storage, posted.id) == (Decision.POSTED, 0)
        # Resolved once: a second pass finds nothing.
        assert await any_storage.resolve_stale_publishing() == []

    async def test_unconfirmed_is_settled(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("UNCONF_SETTLED_12")
        await any_storage.mark_publishing(article)
        await any_storage.resolve_stale_publishing()

        assert await any_storage.filter_new_ids([article.id]) == []

    async def test_published_over_publishing_resets_attempts(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("PUB_DONE_12345678")
        await any_storage.save_decision(article, Decision.ERROR)
        await any_storage.mark_publishing(article)

        await any_storage.save_decision(
            article, Decision.POSTED, _make_ai_decision(), tg_message_id=7
        )

        assert await _outcome(any_storage, article.id) == (Decision.POSTED, 0)

    async def test_release_publishing_new_article_removes_the_record(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("REL_NEW_123456789")
        previous = await any_storage.mark_publishing(article)

        await any_storage.release_publishing(article.id, previous)

        assert await _outcome(any_storage, article.id) is None
        assert await any_storage.filter_new_ids([article.id]) == [article.id]

    async def test_release_publishing_error_article_restores_error_and_attempts(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("REL_ERR_123456789")
        await any_storage.save_decision(article, Decision.ERROR)
        await any_storage.save_decision(article, Decision.ERROR)
        previous = await any_storage.mark_publishing(article)

        await any_storage.release_publishing(article.id, previous)

        assert await _outcome(any_storage, article.id) == (Decision.ERROR, 2)
        assert await any_storage.filter_new_ids([article.id]) == [article.id]

    async def test_error_over_publishing_increments_preserved_attempts(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("ERR_OVER_PUB_1234")
        await any_storage.save_decision(article, Decision.ERROR)
        await any_storage.mark_publishing(article)

        await any_storage.save_decision(article, Decision.ERROR, _make_ai_decision())

        assert await _outcome(any_storage, article.id) == (Decision.ERROR, 2)


class TestRepeatContract:
    async def test_repeat_is_settled_and_keeps_its_original(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("REPEAT_123456789")

        await any_storage.save_decision(article, Decision.REPEAT, repeat_of="ORIGINAL_1234567")

        assert await any_storage.filter_new_ids([article.id]) == []
        assert await _outcome(any_storage, article.id) == (Decision.REPEAT, 0)
        assert await _repeat_of(any_storage, article.id) == "ORIGINAL_1234567"


async def _repeat_of(storage: InMemoryStorage | TursoStorage, article_id: str) -> str | None:
    if isinstance(storage, InMemoryStorage):
        record = storage.get_record(article_id)
        return record.repeat_of if record else None
    result = await storage._get_client().execute(
        "SELECT repeat_of FROM posts WHERE id = ?", [article_id]
    )
    return result.rows[0][0] if result.rows else None


async def _set_last_write(
    storage: InMemoryStorage | TursoStorage, article_id: str, when: datetime
) -> None:
    """Backdate one record's last write."""
    if isinstance(storage, InMemoryStorage):
        record = storage.get_record(article_id)
        assert record is not None
        storage._records[article_id] = record.model_copy(update={"seen_at": when})
        return
    await storage._get_client().execute(
        "UPDATE posts SET seen_at = ? WHERE id = ?", [when.isoformat(), article_id]
    )


class TestRecentJudgedArticlesContract:
    """The AI-judged rows of the last 72 hours: the originals a new Article may repeat."""

    async def test_returns_judged_outcomes_and_never_error_or_repeat(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        await any_storage.save_decision(_make_article("J_POSTED"), Decision.POSTED)
        await any_storage.save_decision(_make_article("J_SKIPPED"), Decision.SKIPPED)
        await any_storage.save_decision(_make_article("J_ERROR"), Decision.ERROR)
        await any_storage.save_decision(
            _make_article("J_REPEAT"), Decision.REPEAT, repeat_of="J_POSTED"
        )
        await any_storage.mark_publishing(_make_article("J_UNCONFIRMED"))
        await any_storage.resolve_stale_publishing()
        await any_storage.mark_publishing(_make_article("J_PUBLISHING"))

        rows = await any_storage.recent_judged_articles(since=datetime.now(UTC) - _HOUR)

        assert {r.id for r in rows} == {"J_POSTED", "J_SKIPPED", "J_UNCONFIRMED", "J_PUBLISHING"}
        posted = next(r for r in rows if r.id == "J_POSTED")
        assert posted.url == _make_article("J_POSTED").url
        assert posted.title_pl == _make_article("J_POSTED").title_pl

    async def test_window_includes_its_boundary_and_nothing_older(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        since = datetime(2026, 10, 23, 12, 0, 0, 500000, tzinfo=UTC)
        for article_id, when in [
            ("W_AT_BOUNDARY", since),
            ("W_INSIDE", since + _HOUR),
            ("W_JUST_OUTSIDE", since - timedelta(microseconds=1)),
            ("W_OLD", since - 24 * _HOUR),
        ]:
            await any_storage.save_decision(_make_article(article_id), Decision.POSTED)
            await _set_last_write(any_storage, article_id, when)

        rows = await any_storage.recent_judged_articles(since=since)

        assert {r.id for r in rows} == {"W_AT_BOUNDARY", "W_INSIDE"}
        # Oldest first, so the earliest Article about a Story is its original
        assert [r.id for r in rows] == ["W_AT_BOUNDARY", "W_INSIDE"]


class TestExistingIdsContract:
    async def test_returns_ids_with_any_record_including_retryable_errors(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        await any_storage.save_decision(_make_article("E_POSTED"), Decision.POSTED)
        await any_storage.save_decision(_make_article("E_ERROR"), Decision.ERROR)
        await any_storage.mark_publishing(_make_article("E_PUBLISHING"))

        found = await any_storage.existing_ids(["E_POSTED", "E_ERROR", "E_PUBLISHING", "E_NEW"])

        assert found == {"E_POSTED", "E_ERROR", "E_PUBLISHING"}

    async def test_empty_input_returns_nothing(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        assert await any_storage.existing_ids([]) == set()


def _dated_decision(
    start: date | None = date(2026, 10, 24), end: date | None = date(2026, 10, 26)
) -> AIDecision:
    return _make_ai_decision().model_copy(update={"event_start": start, "event_end": end})


async def _event_date(
    storage: InMemoryStorage | TursoStorage, article_id: str
) -> tuple[date | None, date | None]:
    if isinstance(storage, InMemoryStorage):
        record = storage.get_record(article_id)
        assert record is not None
        return record.event_start, record.event_end
    result = await storage._get_client().execute(
        "SELECT event_start, event_end FROM posts WHERE id = ?", [article_id]
    )
    start, end = result.rows[0]
    return (
        date.fromisoformat(start) if start else None,
        date.fromisoformat(end) if end else None,
    )


class TestEventDateContract:
    @pytest.mark.parametrize("decision", [Decision.POSTED, Decision.SKIPPED])
    async def test_event_date_is_stored_with_an_ai_judged_row(
        self, any_storage: InMemoryStorage | TursoStorage, decision: Decision
    ) -> None:
        article = _make_article("DATED_123456789")

        await any_storage.save_decision(article, decision, _dated_decision())

        assert await _event_date(any_storage, article.id) == (
            date(2026, 10, 24),
            date(2026, 10, 26),
        )

    async def test_single_day_event_has_no_end(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("SINGLE_DAY_12345")

        await any_storage.save_decision(
            article, Decision.POSTED, _dated_decision(date(2026, 10, 31), None)
        )

        assert await _event_date(any_storage, article.id) == (date(2026, 10, 31), None)

    async def test_row_without_ai_decision_has_no_event_date(
        self, any_storage: InMemoryStorage | TursoStorage
    ) -> None:
        article = _make_article("NO_DATE_12345678")
        await any_storage.save_decision(article, Decision.POSTED, _dated_decision())

        await any_storage.save_decision(article, Decision.ERROR)

        assert await _event_date(any_storage, article.id) == (None, None)
