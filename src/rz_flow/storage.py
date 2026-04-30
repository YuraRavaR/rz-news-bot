"""Storage layer: Turso (libsql) client and InMemoryStorage for tests.

Design pattern: StorageProtocol defines the interface.
  - TursoStorage  → production (Turso cloud SQLite)
  - InMemoryStorage → tests (zero setup, zero cost)

This means tests NEVER touch a real database, and switching to Postgres later
is as simple as writing a new class that implements StorageProtocol.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import libsql_client

from rz_flow.models import AIDecision, Article, Decision, PostRecord

# ── SQL DDL ───────────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    category        TEXT NOT NULL,
    title_pl        TEXT NOT NULL,
    seen_at         TEXT NOT NULL,
    decision        TEXT NOT NULL,
    ai_score        REAL,
    ai_reason       TEXT,
    tg_message_id   INTEGER
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_posts_seen_at ON posts(seen_at DESC);
"""


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
    ) -> None:
        client = self._get_client()
        await client.execute(
            """
            INSERT OR REPLACE INTO posts
              (id, url, category, title_pl, seen_at, decision, ai_score, ai_reason, tg_message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                tg_message_id,
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
    ) -> None:
        self._records[article.id] = PostRecord(
            id=article.id,
            url=article.url,
            category=article.category,
            title_pl=article.title_pl,
            decision=decision,
            ai_score=ai_decision.score if ai_decision else None,
            ai_reason=ai_decision.reason if ai_decision else None,
            tg_message_id=tg_message_id,
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
