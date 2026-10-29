"""Storage layer: Turso (libsql) client and InMemoryStorage for tests.

Design pattern: StorageProtocol defines the interface.
  - TursoStorage  → production (Turso cloud SQLite)
  - InMemoryStorage → tests (zero setup, zero cost)

This means tests NEVER touch a real database, and switching to Postgres later
is as simple as writing a new class that implements StorageProtocol.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Protocol, runtime_checkable

import libsql_client

from rz_flow.models import (
    AIDecision,
    Article,
    DatedEvent,
    Decision,
    DigestRecord,
    DigestStatus,
    JudgedArticle,
    PostRecord,
    UnconfirmedArticle,
)

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
    error_attempts        INTEGER NOT NULL DEFAULT 0,
    repeat_of             TEXT,
    event_start           TEXT,
    event_end             TEXT
);
"""

# One row per weekend (keyed by its Friday): the Digest is posted at most once
_CREATE_DIGESTS_TABLE = """
CREATE TABLE IF NOT EXISTS digests (
    weekend        TEXT PRIMARY KEY,
    status         TEXT NOT NULL,
    tg_message_id  INTEGER,
    item_count     INTEGER NOT NULL,
    written_at     TEXT NOT NULL
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
    ("repeat_of", "TEXT"),
    # Event date as ISO YYYY-MM-DD; rows from before it have none (not backfilled)
    ("event_start", "TEXT"),
    ("event_end", "TEXT"),
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


# Outcomes the AI reached for an Article: such Articles are originals for Repeats.
# Error and Repeat rows never are (the AI never judged them).
_JUDGED_OUTCOMES = (
    Decision.POSTED,
    Decision.SKIPPED,
    Decision.PUBLISHING,
    Decision.UNCONFIRMED,
)

# How many times an article may fail before we stop retrying it. A transient blip
# (network, malformed AI JSON) must not retire an article permanently, but a genuinely
# broken one must not be retried forever either.
DEFAULT_MAX_ERROR_ATTEMPTS = 3


def _iso_or_none(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


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

    async def existing_ids(self, article_ids: list[str]) -> set[str]:
        """IDs that have a record of any outcome, retryable Errors included."""
        ...

    async def save_decision(
        self,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None = None,
        tg_message_id: int | None = None,
        tg_events_message_id: int | None = None,
        repeat_of: str | None = None,
    ) -> None:
        """Persist the processing result for one article.

        ``repeat_of`` is the original's ID, set only for a Repeat.
        """
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

    async def recent_judged_articles(self, since: datetime) -> list[JudgedArticle]:
        """AI-judged Articles (Published, Skipped, Publishing, Unconfirmed) whose
        last write is at or after ``since``, oldest first.
        """
        ...

    async def published_events_between(self, start: date, end: date) -> list[DatedEvent]:
        """Published events whose Event date overlaps ``start..end`` (both included),
        by first day then ID. A missing end means a single day; undated events are
        left out.
        """
        ...

    async def get_digest(self, weekend: date) -> DigestRecord | None:
        """The Digest record of the weekend starting on Friday ``weekend``, if any."""
        ...

    async def start_digest(self, weekend: date, item_count: int) -> bool:
        """Record the weekend's Digest as Publishing before it is sent.

        Returns False, writing nothing, when the weekend already has a record.
        """
        ...

    async def mark_digest_published(self, weekend: date, message_id: int) -> None:
        """The Digest went out: record it Published with its message ID."""
        ...

    async def delete_digest(self, weekend: date) -> None:
        """Drop the weekend's record after a Digest that definitely did not go out,
        so a manual re-run can post it.
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
        await client.execute(_CREATE_DIGESTS_TABLE)
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
        # "Settled" = posted, skipped, repeat, publishing, unconfirmed, or errored past the
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

    async def existing_ids(self, article_ids: list[str]) -> set[str]:
        if not article_ids:
            return set()
        placeholders = ", ".join("?" * len(article_ids))
        result = await self._get_client().execute(
            f"SELECT id FROM posts WHERE id IN ({placeholders})", article_ids
        )
        return {row[0] for row in result.rows}

    async def save_decision(
        self,
        article: Article,
        decision: Decision,
        ai_decision: AIDecision | None = None,
        tg_message_id: int | None = None,
        tg_events_message_id: int | None = None,
        repeat_of: str | None = None,
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
               ua_title, ua_summary, category_tag, error_attempts, repeat_of,
               event_start, event_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                repeat_of            = excluded.repeat_of,
                event_start          = excluded.event_start,
                event_end            = excluded.event_end,
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
                repeat_of,
                _iso_or_none(ai_decision.event_start) if ai_decision else None,
                _iso_or_none(ai_decision.event_end) if ai_decision else None,
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

    async def recent_judged_articles(self, since: datetime) -> list[JudgedArticle]:
        client = self._get_client()
        placeholders = ", ".join("?" * len(_JUDGED_OUTCOMES))
        # seen_at is an ISO-8601 UTC string, so string order is time order and the
        # seen_at index serves the range
        result = await client.execute(
            f"""
            SELECT id, url, title_pl FROM posts
            WHERE seen_at >= ? AND decision IN ({placeholders})
            ORDER BY seen_at
            """,
            [
                since.astimezone(UTC).isoformat(timespec="microseconds"),
                *(d.value for d in _JUDGED_OUTCOMES),
            ],
        )
        return [JudgedArticle(id=r[0], url=r[1], title_pl=r[2]) for r in result.rows]

    async def published_events_between(self, start: date, end: date) -> list[DatedEvent]:
        result = await self._get_client().execute(
            """
            SELECT id, url, title_pl, ua_title, event_start, event_end, tg_events_message_id
            FROM posts
            WHERE decision = ? AND is_event = 1 AND event_start IS NOT NULL
              AND event_start <= ? AND COALESCE(event_end, event_start) >= ?
            ORDER BY event_start, id
            """,
            [Decision.POSTED.value, end.isoformat(), start.isoformat()],
        )
        return [
            DatedEvent(
                id=r[0],
                url=r[1],
                title_pl=r[2],
                ua_title=r[3],
                event_start=date.fromisoformat(r[4]),
                event_end=date.fromisoformat(r[5]) if r[5] else None,
                tg_events_message_id=r[6],
            )
            for r in result.rows
        ]

    async def get_digest(self, weekend: date) -> DigestRecord | None:
        result = await self._get_client().execute(
            "SELECT status, tg_message_id, item_count, written_at FROM digests WHERE weekend = ?",
            [weekend.isoformat()],
        )
        if not result.rows:
            return None
        status, message_id, item_count, written_at = result.rows[0]
        return DigestRecord(
            weekend=weekend,
            status=DigestStatus(status),
            tg_message_id=message_id,
            item_count=item_count,
            written_at=datetime.fromisoformat(written_at),
        )

    async def start_digest(self, weekend: date, item_count: int) -> bool:
        result = await self._get_client().execute(
            """
            INSERT INTO digests (weekend, status, item_count, written_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(weekend) DO NOTHING
            """,
            [
                weekend.isoformat(),
                DigestStatus.PUBLISHING.value,
                item_count,
                datetime.now(UTC).isoformat(),
            ],
        )
        return bool(result.rows_affected == 1)

    async def mark_digest_published(self, weekend: date, message_id: int) -> None:
        await self._get_client().execute(
            "UPDATE digests SET status = ?, tg_message_id = ?, written_at = ? WHERE weekend = ?",
            [
                DigestStatus.PUBLISHED.value,
                message_id,
                datetime.now(UTC).isoformat(),
                weekend.isoformat(),
            ],
        )

    async def delete_digest(self, weekend: date) -> None:
        await self._get_client().execute(
            "DELETE FROM digests WHERE weekend = ?", [weekend.isoformat()]
        )

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
        self._digests: dict[date, DigestRecord] = {}
        self._max_error_attempts = max_error_attempts

    async def init(self) -> None:
        pass  # nothing to create

    async def filter_new_ids(self, article_ids: list[str]) -> list[str]:
        return [aid for aid in article_ids if not self._is_settled(aid)]

    async def existing_ids(self, article_ids: list[str]) -> set[str]:
        return {aid for aid in article_ids if aid in self._records}

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
        repeat_of: str | None = None,
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
            repeat_of=repeat_of,
            event_start=ai_decision.event_start if ai_decision else None,
            event_end=ai_decision.event_end if ai_decision else None,
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

    async def recent_judged_articles(self, since: datetime) -> list[JudgedArticle]:
        judged = [
            r
            for r in self._records.values()
            if r.decision in _JUDGED_OUTCOMES and r.seen_at >= since
        ]
        judged.sort(key=lambda r: r.seen_at)
        return [JudgedArticle(id=r.id, url=r.url, title_pl=r.title_pl) for r in judged]

    async def published_events_between(self, start: date, end: date) -> list[DatedEvent]:
        events = [
            DatedEvent(
                id=r.id,
                url=r.url,
                title_pl=r.title_pl,
                ua_title=r.ua_title,
                event_start=r.event_start,
                event_end=r.event_end,
                tg_events_message_id=r.tg_events_message_id,
            )
            for r in self._records.values()
            if r.decision == Decision.POSTED
            and r.is_event
            and r.event_start is not None
            and r.event_start <= end
            and (r.event_end or r.event_start) >= start
        ]
        events.sort(key=lambda e: (e.event_start, e.id))
        return events

    async def get_digest(self, weekend: date) -> DigestRecord | None:
        return self._digests.get(weekend)

    async def start_digest(self, weekend: date, item_count: int) -> bool:
        if weekend in self._digests:
            return False
        self._digests[weekend] = DigestRecord(
            weekend=weekend, status=DigestStatus.PUBLISHING, item_count=item_count
        )
        return True

    async def mark_digest_published(self, weekend: date, message_id: int) -> None:
        record = self._digests.get(weekend)
        if record is not None:
            self._digests[weekend] = record.model_copy(
                update={
                    "status": DigestStatus.PUBLISHED,
                    "tg_message_id": message_id,
                    "written_at": datetime.now(UTC),
                }
            )

    async def delete_digest(self, weekend: date) -> None:
        self._digests.pop(weekend, None)

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
