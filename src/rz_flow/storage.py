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

from rz_flow.models import AIDecision, Article, Decision, PostRecord, UnconfirmedArticle

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
    category_tag          TEXT,
    error_attempts        INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_posts_seen_at ON posts(seen_at DESC);
"""

# Schema migrations: columns added after the first release, as (name, definition).
# At init the current columns are read with PRAGMA table_info and only the missing
# ones are added. No error is suppressed: a failing ALTER TABLE stops the Run.
_COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("ua_title", "TEXT"),
    ("ua_summary", "TEXT"),
    ("category_tag", "TEXT"),
    ("is_event", "INTEGER"),
    ("tg_events_message_id", "INTEGER"),
    ("error_attempts", "INTEGER NOT NULL DEFAULT 0"),
]

# Data migrations — one-time UPDATE statements that backfill existing rows.
# These are idempotent (WHERE clause ensures they only touch un-migrated rows)
# and must NOT be suppressed — a failure here means data is inconsistent.
#
# QW-2 introduced source-prefixed IDs (rz24/..., rzn/...).
# Existing DB records were stored without a prefix, so we backfill them here
# to prevent the bot from re-processing already-seen articles after the upgrade.
_DATA_MIGRATIONS = [
    "UPDATE posts SET id = 'rz24/' || id "
    "WHERE url LIKE '%rzeszow24.info%' AND id NOT LIKE 'rz24/%'",
    "UPDATE posts SET id = 'rzn/' || id WHERE url LIKE '%rzeszow-news.pl%' AND id NOT LIKE 'rzn/%'",
]


# How many times an article may fail before we stop retrying it. A transient blip
# (network, malformed AI JSON) must not retire an article permanently, but a genuinely
# broken one must not be retried forever either.
DEFAULT_MAX_ERROR_ATTEMPTS = 3


# ── Protocol (interface) ──────────────────────────────────────────────────────
@runtime_checkable
class StorageProtocol(Protocol):
    """Interface that all storage backends must implement."""

    async def init(self) -> None:
        """Create tables / run migrations if needed."""
        ...

    async def filter_new_ids(self, article_ids: list[str]) -> list[str]:
        """Return IDs that still need processing.

        That means never-seen articles *plus* previously errored ones that have
        not yet used up their retry budget.
        """
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

    async def mark_publishing(self, article: Article) -> Decision | None:
        """Record the Article as Publishing before its Main channel Post.

        With no row, inserts a minimal one. Over an Error row, changes only the
        outcome: the attempt count and every other column stay untouched.
        Returns the previous outcome (None or ERROR).
        """
        ...

    async def release_publishing(self, article_id: str, previous: Decision | None) -> None:
        """Undo ``mark_publishing`` after a Post that definitely did not go out.

        ``previous`` is what ``mark_publishing`` returned: with None the row is
        deleted, with ERROR it goes back to Error with its attempt count unchanged.
        """
        ...

    async def resolve_stale_publishing(self) -> list[UnconfirmedArticle]:
        """Turn every Publishing row into Unconfirmed and return those Articles.

        Runs once at the start of every non-dry Run, before scraping. Safe only
        because production Runs never overlap.
        """
        ...

    async def close(self) -> None:
        """Release resources (connection pool, etc.)."""
        ...


# ── Turso (production) ────────────────────────────────────────────────────────
class TursoStorage:
    """Turso / libsql-backed storage — used in production."""

    def __init__(
        self,
        database_url: str,
        auth_token: str,
        max_error_attempts: int = DEFAULT_MAX_ERROR_ATTEMPTS,
    ) -> None:
        self._url = self._normalize_url(database_url)
        self._token = auth_token
        self._max_error_attempts = max_error_attempts
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
        """Add missing columns, then apply data migrations, for existing databases."""
        result = await client.execute("PRAGMA table_info(posts)")
        existing = {row[1] for row in result.rows}
        for name, definition in _COLUMN_MIGRATIONS:
            if name not in existing:
                await client.execute(f"ALTER TABLE posts ADD COLUMN {name} {definition}")
        for sql in _DATA_MIGRATIONS:
            await client.execute(sql)  # must succeed — no suppression

    async def filter_new_ids(self, article_ids: list[str]) -> list[str]:
        if not article_ids:
            return []

        client = self._get_client()
        placeholders = ", ".join("?" * len(article_ids))
        # "Settled" = posted, skipped, publishing, unconfirmed, or errored past the
        # retry budget. Anything else in this batch still needs work, including rows
        # we are about to retry.
        result = await client.execute(
            f"""
            SELECT id FROM posts
            WHERE id IN ({placeholders})
              AND NOT (decision = ? AND COALESCE(error_attempts, 0) < ?)
            """,
            [*article_ids, Decision.ERROR.value, self._max_error_attempts],
        )
        settled_ids = {row[0] for row in result.rows}
        return [aid for aid in article_ids if aid not in settled_ids]

    async def save_decision(
        self,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None = None,
        tg_message_id: int | None = None,
        tg_events_message_id: int | None = None,
    ) -> None:
        client = self._get_client()
        is_error = decision == Decision.ERROR
        # UPSERT rather than INSERT OR REPLACE so error_attempts can accumulate across
        # runs — REPLACE deletes the old row and would reset the counter every time.
        await client.execute(
            """
            INSERT INTO posts
              (id, url, category, title_pl, seen_at, decision,
               ai_score, ai_reason, is_event, tg_message_id, tg_events_message_id,
               ua_title, ua_summary, category_tag, error_attempts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                url                  = excluded.url,
                category             = excluded.category,
                title_pl             = excluded.title_pl,
                seen_at              = excluded.seen_at,
                decision             = excluded.decision,
                ai_score             = excluded.ai_score,
                ai_reason            = excluded.ai_reason,
                is_event             = excluded.is_event,
                tg_message_id        = excluded.tg_message_id,
                tg_events_message_id = excluded.tg_events_message_id,
                ua_title             = excluded.ua_title,
                ua_summary           = excluded.ua_summary,
                category_tag         = excluded.category_tag,
                error_attempts       = CASE
                    WHEN excluded.decision = ?
                    THEN COALESCE(posts.error_attempts, 0) + 1
                    ELSE 0
                END
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
                1 if is_error else 0,
                Decision.ERROR.value,
            ],
        )

    async def mark_publishing(self, article: Article) -> Decision | None:
        client = self._get_client()
        result = await client.execute("SELECT decision FROM posts WHERE id = ?", [article.id])
        previous = Decision(result.rows[0][0]) if result.rows else None
        await client.execute(
            """
            INSERT INTO posts (id, url, category, title_pl, seen_at, decision, error_attempts)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(id) DO UPDATE SET decision = excluded.decision
            """,
            [
                article.id,
                article.url,
                article.category.value,
                article.title_pl,
                datetime.now(UTC).isoformat(),
                Decision.PUBLISHING.value,
            ],
        )
        return previous

    async def release_publishing(self, article_id: str, previous: Decision | None) -> None:
        client = self._get_client()
        if previous is None:
            await client.execute(
                "DELETE FROM posts WHERE id = ? AND decision = ?",
                [article_id, Decision.PUBLISHING.value],
            )
        else:
            await client.execute(
                "UPDATE posts SET decision = ? WHERE id = ? AND decision = ?",
                [previous.value, article_id, Decision.PUBLISHING.value],
            )

    async def resolve_stale_publishing(self) -> list[UnconfirmedArticle]:
        client = self._get_client()
        result = await client.execute(
            "SELECT id, url, title_pl FROM posts WHERE decision = ? ORDER BY seen_at",
            [Decision.PUBLISHING.value],
        )
        stale = [UnconfirmedArticle(id=r[0], url=r[1], title_pl=r[2]) for r in result.rows]
        if stale:
            await client.execute(
                "UPDATE posts SET decision = ? WHERE decision = ?",
                [Decision.UNCONFIRMED.value, Decision.PUBLISHING.value],
            )
        return stale

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# ── InMemoryStorage (tests) ───────────────────────────────────────────────────
class InMemoryStorage:
    """In-memory storage for tests — no Turso connection required.

    Mirrors TursoStorage semantics, including the error retry budget.
    """

    def __init__(self, max_error_attempts: int = DEFAULT_MAX_ERROR_ATTEMPTS) -> None:
        self._records: dict[str, PostRecord] = {}
        self._max_error_attempts = max_error_attempts

    async def init(self) -> None:
        pass  # nothing to create

    async def filter_new_ids(self, article_ids: list[str]) -> list[str]:
        return [aid for aid in article_ids if not self._is_settled(aid)]

    def _is_settled(self, article_id: str) -> bool:
        record = self._records.get(article_id)
        if record is None:
            return False
        if record.decision != Decision.ERROR:
            return True
        return record.error_attempts >= self._max_error_attempts

    async def save_decision(
        self,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None = None,
        tg_message_id: int | None = None,
        tg_events_message_id: int | None = None,
    ) -> None:
        previous = self._records.get(article.id)
        prior_attempts = previous.error_attempts if previous else 0
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
            error_attempts=prior_attempts + 1 if decision == Decision.ERROR else 0,
        )

    async def mark_publishing(self, article: Article) -> Decision | None:
        previous = self._records.get(article.id)
        if previous is not None:
            self._records[article.id] = previous.model_copy(
                update={"decision": Decision.PUBLISHING}
            )
            return previous.decision
        self._records[article.id] = PostRecord(
            id=article.id,
            url=article.url,
            category=article.category,
            title_pl=article.title_pl,
            decision=Decision.PUBLISHING,
        )
        return None

    async def release_publishing(self, article_id: str, previous: Decision | None) -> None:
        record = self._records.get(article_id)
        if record is None or record.decision != Decision.PUBLISHING:
            return
        if previous is None:
            del self._records[article_id]
        else:
            self._records[article_id] = record.model_copy(update={"decision": previous})

    async def resolve_stale_publishing(self) -> list[UnconfirmedArticle]:
        stale = [r for r in self._records.values() if r.decision == Decision.PUBLISHING]
        stale.sort(key=lambda r: r.seen_at)
        for record in stale:
            self._records[record.id] = record.model_copy(update={"decision": Decision.UNCONFIRMED})
        return [UnconfirmedArticle(id=r.id, url=r.url, title_pl=r.title_pl) for r in stale]

    async def close(self) -> None:
        pass

    # ── Helpers used in tests ─────────────────────────────────────────────────
    def all_records(self) -> list[PostRecord]:
        return list(self._records.values())

    def get_record(self, article_id: str) -> PostRecord | None:
        return self._records.get(article_id)

    def count(self) -> int:
        return len(self._records)


def create_storage(
    database_url: str,
    auth_token: str,
    max_error_attempts: int = DEFAULT_MAX_ERROR_ATTEMPTS,
) -> TursoStorage:
    """Factory that creates a TursoStorage (used in main pipeline)."""
    return TursoStorage(
        database_url=database_url,
        auth_token=auth_token,
        max_error_attempts=max_error_attempts,
    )
