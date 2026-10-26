"""Shared data models (Pydantic).

These models act as the contracts between pipeline stages.
They are intentionally simple for MVP — extend via subclassing, not editing.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


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


class Decision(StrEnum):
    POSTED = "posted"
    SKIPPED = "skipped"
    ERROR = "error"
    # Written before the Main channel Post; stays when delivery is uncertain.
    PUBLISHING = "publishing"
    # A Publishing record that a later Run found unresolved. Final.
    UNCONFIRMED = "unconfirmed"
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


class UnconfirmedArticle(BaseModel):
    """A Publishing Article turned Unconfirmed at Run start; the admin is told once."""

    id: str
    url: str
    title_pl: str


class PublishResult(BaseModel):
    """Result returned by the Telegram publisher."""

    article_id: str
    message_id: int
    chat_id: str
