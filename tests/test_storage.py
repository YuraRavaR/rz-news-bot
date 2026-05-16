"""Tests for the storage layer.

Uses InMemoryStorage so tests never need a real Turso connection.
This verifies the business logic of the storage protocol without
any I/O or credentials.
"""

import pytest

from rz_flow.models import AIDecision, Article, Category, CategoryTag, Decision
from rz_flow.storage import InMemoryStorage, TursoStorage, create_storage


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
