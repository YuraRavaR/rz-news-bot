"""Shared data models (Pydantic).

These models act as the contracts between pipeline stages.
They are intentionally simple for MVP — extend via subclassing, not editing.
"""

import re
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Self

import structlog
from pydantic import BaseModel, Field, model_validator

logger = structlog.get_logger(__name__)

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class Category(StrEnum):
    IMPREZY = "imprezy"
    WIADOMOSCI = "wiadomosci"


class Article(BaseModel):
    """Raw article extracted from a Rzeszów news source."""

    id: str = Field(..., description="Unique slug/ID from the URL tail")
    url: str = Field(..., description="Full article URL")
    category: Category
    title_pl: str = Field(..., description="Original Polish title")
    summary_pl: str = Field(default="", description="Lead paragraph or excerpt in Polish")
    source_name: str = Field(
        default="",
        description="Source label for logs/reports (e.g. 'rzeszow24/najnowsze', 'rzeszow-news.pl')",
    )


class CategoryTag(StrEnum):
    CONCERT = "koncert"
    FESTIVAL = "festyn"
    SPORT = "sport"
    TRANSPORT = "komunikacja"
    OTHER = "inne"


class AIDecision(BaseModel):
    """Structured response from Gemini AI."""

    is_interesting: bool
    is_event: bool = Field(
        ...,
        description=(
            "True if this is a specific upcoming public event "
            "(concert, festival, exhibition, sports match, fair, cultural/community gathering). "
            "False for general news, infrastructure, transport, or any non-event content."
        ),
    )
    score: float = Field(..., ge=0, le=10, description="Relevance score for Rzeszów residents")
    category_tag: CategoryTag
    ua_title: str = Field(..., description="Ukrainian translation of the title")
    ua_summary: str = Field(..., description="2–3 sentence Ukrainian summary")
    reason: str = Field(..., description="Short explanation for the decision (used in logs)")
    # The Event date: first and last day of an event; a single-day event has no end.
    # A bad date from the AI becomes "no date" and never makes the Article an Error.
    event_start: date | None = None
    event_end: date | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_event_dates(cls, data: object) -> object:
        """Both dates or neither: one bad value makes the whole Event date unknown."""
        if not isinstance(data, dict):
            return data
        raw = {name: data.get(name) for name in ("event_start", "event_end")}
        parsed = {name: _parse_iso_date(value) for name, value in raw.items()}
        if any(value is not None and parsed[name] is None for name, value in raw.items()):
            logger.warning(
                "event_date_dropped",
                reason="unparseable",
                **{name: repr(value) for name, value in raw.items()},
            )
            parsed = {"event_start": None, "event_end": None}
        return {**data, **parsed}

    @model_validator(mode="after")
    def _check_event_dates(self) -> Self:
        if self.event_start is None and self.event_end is None:
            return self
        reason = None
        if not self.is_event:
            reason = "not_an_event"
        elif self.event_start is None:
            reason = "end_without_start"
        elif self.event_end is not None and self.event_end < self.event_start:
            reason = "end_before_start"
        if reason is not None:
            logger.warning(
                "event_date_dropped",
                reason=reason,
                event_start=str(self.event_start),
                event_end=str(self.event_end),
            )
            self.event_start = self.event_end = None
        elif self.event_end == self.event_start:
            self.event_end = None
        return self


def _parse_iso_date(value: object) -> date | None:
    """A ``YYYY-MM-DD`` string (or a date) as a date; None for anything else."""
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


class Decision(StrEnum):
    POSTED = "posted"
    SKIPPED = "skipped"
    ERROR = "error"
    # Written before the Main channel Post; stays when delivery is uncertain.
    PUBLISHING = "publishing"
    # A Publishing record that a later Run found unresolved. Final.
    UNCONFIRMED = "unconfirmed"
    # Not sent to the AI: it already judged another Article about the same Story.
    # Final; the record keeps the original's ID.
    REPEAT = "repeat"
    # Run-report only: a Deferred Article is never written to storage.
    DEFERRED = "deferred"


class StopSignal(StrEnum):
    """Why the per-article loop should stop (or not) after processing one article."""

    CONTINUE = "continue"
    QUOTA_EXHAUSTED = "quota_exhausted"
    CAP_REACHED = "cap_reached"
    # Telegram unreachable, delivery uncertain, or the Publishing record unwritable
    PUBLISHING_BLOCKED = "publishing_blocked"
    # Telegram asked to wait longer than we are willing to, or rate limited us twice
    TELEGRAM_RATE_LIMITED = "telegram_rate_limited"


class PostRecord(BaseModel):
    """Record stored in Turso after processing an article."""

    id: str
    url: str
    category: Category
    title_pl: str
    seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decision: Decision
    ai_score: float | None = None
    ai_reason: str | None = None
    is_event: bool | None = None
    tg_message_id: int | None = None
    tg_events_message_id: int | None = None
    ua_title: str | None = None
    ua_summary: str | None = None
    category_tag: str | None = None
    error_attempts: int = Field(
        default=0,
        description="Consecutive failed processing attempts; reset to 0 on success",
    )
    repeat_of: str | None = Field(
        default=None,
        description="For a Repeat, the ID of the original Article",
    )
    event_start: date | None = None
    event_end: date | None = None


class UnconfirmedArticle(BaseModel):
    """A Publishing Article turned Unconfirmed at Run start; the admin is told once."""

    id: str
    url: str
    title_pl: str


class JudgedArticle(BaseModel):
    """A stored Article the AI judged: a possible original for later Repeats."""

    id: str
    url: str
    title_pl: str


class PublishResult(BaseModel):
    """Result returned by the Telegram publisher."""

    article_id: str
    message_id: int
    chat_id: str


class DatedEvent(BaseModel):
    """A Published event with an Event date: one possible Digest item."""

    id: str
    url: str
    title_pl: str
    ua_title: str | None
    event_start: date
    # None for a single-day event
    event_end: date | None
    # None when the Events channel Post failed or was never sent
    tg_events_message_id: int | None


class CalendarEntry(BaseModel):
    """One entry of the city's events Calendar: a possible Digest item, not an Article."""

    # The numeric ID in the entry's URL; the same entry is listed on every day it covers
    id: int
    url: str
    title_pl: str
    start: date
    # None for a single-day entry
    end: date | None


class DigestStatus(StrEnum):
    # Written before the Digest is sent; stays when delivery is uncertain
    PUBLISHING = "publishing"
    PUBLISHED = "published"


class DigestRecord(BaseModel):
    """The one Digest of a weekend, keyed by its Friday."""

    weekend: date
    status: DigestStatus
    tg_message_id: int | None = None
    item_count: int
    written_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PeriodCounts(BaseModel):
    """Articles whose last write falls in a period: the Weekly report's numbers."""

    # ID prefix ("rz24") → outcome → Articles; outcomes with none are left out
    by_prefix: dict[str, dict[Decision, int]] = Field(default_factory=dict)
    events_posted: int = 0
    # Published events with no Event date, which includes all written before Event dates
    events_posted_undated: int = 0
