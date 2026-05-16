"""Storage layer: Turso (libsql) client and InMemoryStorage for tests.

Design pattern: StorageProtocol defines the interface.
  - TursoStorage  → production (Turso cloud SQLite)
  - InMemoryStorage → tests (zero setup, zero cost)

This means tests NEVER touch a real database, and switching to Postgres later
is as simple as writing a new class that implements StorageProtocol.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import libsql_client

from rz_flow.models import AIDecision, Article, Decision, PostRecord

# ── SQL DDL ───────────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS posts (
    id                    TEXT PRIMARY KEY,
    url                   TEXT NOT NULL,
    category              TEXT NOT NULL,
    title_pl              TEXT NOT NULL,
    seen_at               TEXT NOT NULL,
    decision              TEXT NOT NULL,
    ai_score              REAL,
    ai_reason             TEXT,
    is_event              INTEGER,
    tg_message_id         INTEGER,
    tg_events_message_id  INTEGER,
    ua_title              TEXT,
    ua_summary            TEXT,
    category_tag          TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_posts_seen_at ON posts(seen_at DESC);
"""

# Schema migrations — ALTER TABLE statements.
# SQLite does not support "ADD COLUMN IF NOT EXISTS", so errors are silently suppressed:
# a "duplicate column name" error means the column already exists and we can move on.
_SCHEMA_MIGRATIONS = [
    "ALTER TABLE posts ADD COLUMN ua_title TEXT",
    "ALTER TABLE posts ADD COLUMN ua_summary TEXT",
    "ALTER TABLE posts ADD COLUMN category_tag TEXT",
    "ALTER TABLE posts ADD COLUMN is_event INTEGER",
    "ALTER TABLE posts ADD COLUMN tg_events_message_id INTEGER",
]

# Data migrations — one-time UPDATE statements that backfill existing rows.
# These are idempotent (WHERE clause ensures they only touch un-migrated rows)
# and must NOT be suppressed — a failure here means data is inconsistent.
#
# QW-2 introduced source-prefixed IDs (rz24/..., rzn/...).
# Existing DB records were stored without a prefix, so we backfill them here
# to prevent the bot from re-processing already-seen articles after the upgrade.
_DATA_MIGRATIONS = [
    "UPDATE posts SET id = 'rz24/' || id WHERE url LIKE '%rzeszow24.info%' AND id NOT LIKE 'rz24/%'",
    "UPDATE posts SET id = 'rzn/' || id WHERE url LIKE '%rzeszow-news.pl%' AND id NOT LIKE 'rzn/%'",
]


# ── Protocol (interface) ──────────────────────────────────────────────────────
@runtime_checkable
class StorageProtocol(Protocol):
    """Interface that all storage backends must implement."""

    async def init(self) -> None:
        """Create tables / run migrations if needed."""
        ...

    async def filter_new_ids(self, article_ids: list[str]) -> list[str]:
        """Return only IDs that have NOT been seen before."""
        ...

    async def save_decision(
        self,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None = None,
        tg_message_id: int | None = None,
        tg_events_message_id: int | None = None,
    ) -> None:
        """Persist the processing result for one article."""
        ...

    async def close(self) -> None:
        """Release resources (connection pool, etc.)."""
        ...


# ── Turso (production) ────────────────────────────────────────────────────────
class TursoStorage:
    """Turso / libsql-backed storage — used in production."""

    def __init__(self, database_url: str, auth_token: str) -> None:
        self._url = self._normalize_url(database_url)
        self._token = auth_token
        self._client: libsql_client.Client | None = None

    @staticmethod
    def _normalize_url(database_url: str) -> str:
        """Normalize Turso URL for libsql-client transport compatibility.

        Some Turso endpoints reject websocket upgrade (wss) with HTTP 505.
        Converting libsql:// to https:// forces the HTTP transport, which
        works reliably with the same auth token.
        """
        if database_url.startswith("libsql://"):
            return database_url.replace("libsql://", "https://", 1)
        return database_url

    def _get_client(self) -> libsql_client.Client:
        if self._client is None:
            self._client = libsql_client.create_client(
                url=self._url,
                auth_token=self._token,
            )
        return self._client

    async def init(self) -> None:
        client = self._get_client()
        await client.execute(_CREATE_TABLE)
        await client.execute(_CREATE_INDEX)
        await self._migrate(client)

    async def _migrate(self, client: libsql_client.Client) -> None:
        """Apply schema and data migrations for existing databases."""
        for sql in _SCHEMA_MIGRATIONS:
            with contextlib.suppress(Exception):  # column already exists — ignore
                await client.execute(sql)
        for sql in _DATA_MIGRATIONS:
            await client.execute(sql)  # must succeed — no suppression

    async def filter_new_ids(self, article_ids: list[str]) -> list[str]:
        if not article_ids:
            return []

        client = self._get_client()
        placeholders = ", ".join("?" * len(article_ids))
        result = await client.execute(
            f"SELECT id FROM posts WHERE id IN ({placeholders})",
            article_ids,
        )
        seen_ids = {row[0] for row in result.rows}
        return [aid for aid in article_ids if aid not in seen_ids]

    async def save_decision(
        self,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None = None,
        tg_message_id: int | None = None,
        tg_events_message_id: int | None = None,
    ) -> None:
        client = self._get_client()
        await client.execute(
            """
            INSERT OR REPLACE INTO posts
              (id, url, category, title_pl, seen_at, decision,
               ai_score, ai_reason, is_event, tg_message_id, tg_events_message_id,
               ua_title, ua_summary, category_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                article.id,
                article.url,
                article.category.value,
                article.title_pl,
                datetime.now(UTC).isoformat(),
                decision.value,
                ai_decision.score if ai_decision else None,
                ai_decision.reason if ai_decision else None,
                int(ai_decision.is_event) if ai_decision else None,
                tg_message_id,
                tg_events_message_id,
                ai_decision.ua_title if ai_decision else None,
                ai_decision.ua_summary if ai_decision else None,
                ai_decision.category_tag.value if ai_decision else None,
            ],
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# ── InMemoryStorage (tests) ───────────────────────────────────────────────────
class InMemoryStorage:
    """In-memory storage for tests — no Turso connection required."""

    def __init__(self) -> None:
        self._records: dict[str, PostRecord] = {}

    async def init(self) -> None:
        pass  # nothing to create

    async def filter_new_ids(self, article_ids: list[str]) -> list[str]:
        return [aid for aid in article_ids if aid not in self._records]

    async def save_decision(
        self,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None = None,
        tg_message_id: int | None = None,
        tg_events_message_id: int | None = None,
    ) -> None:
        self._records[article.id] = PostRecord(
            id=article.id,
            url=article.url,
            category=article.category,
            title_pl=article.title_pl,
            decision=decision,
            ai_score=ai_decision.score if ai_decision else None,
            ai_reason=ai_decision.reason if ai_decision else None,
            is_event=ai_decision.is_event if ai_decision else None,
            tg_message_id=tg_message_id,
            tg_events_message_id=tg_events_message_id,
            ua_title=ai_decision.ua_title if ai_decision else None,
            ua_summary=ai_decision.ua_summary if ai_decision else None,
            category_tag=ai_decision.category_tag.value if ai_decision else None,
        )

    async def close(self) -> None:
        pass

    # ── Helpers used in tests ─────────────────────────────────────────────────
    def all_records(self) -> list[PostRecord]:
        return list(self._records.values())

    def get_record(self, article_id: str) -> PostRecord | None:
        return self._records.get(article_id)

    def count(self) -> int:
        return len(self._records)


def create_storage(database_url: str, auth_token: str) -> TursoStorage:
    """Factory that creates a TursoStorage (used in main pipeline)."""
    return TursoStorage(database_url=database_url, auth_token=auth_token)
