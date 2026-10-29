"""The city's events Calendar (erzeszow.pl), read only for the Digest.

The Calendar is not a Source: it never goes through a Run, the Repeat check or
storage. (Not named ``calendar``, which would shadow the standard library.)

It is read through the per-day request its day picker makes, once for each day of
the weekend. A failure is never raised: it comes back in the CalendarReading, so the
Digest can go out without Calendar entries and alert the admin.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rz_flow.flow_config import DigestConfig
from rz_flow.models import CalendarEntry
from rz_flow.parser import parse_calendar_day
from rz_flow.scraper import _HEADERS

logger = structlog.get_logger(__name__)

# Longer entries (exhibitions, series) count only in the weekend they open
_MAX_SHORT_ENTRY_DAYS = 3

# The day picker's request: the list of one day as HTML inside JSON
_DAY_ACTION = "disp-wydarzenia-dzien"
_CALENDAR_ID = "1"


@dataclass(frozen=True)
class CalendarReading:
    """The weekend's Calendar entries, or why they could not be read."""

    entries: list[CalendarEntry]
    # Plain-text admin alert (Ukrainian; escape it before an HTML message). None when the
    # Calendar was read or is disabled
    failure: str | None = None


def in_weekend(entry: CalendarEntry, friday: date) -> bool:
    """Whether the Digest of the weekend starting on *friday* should consider *entry*.

    A short entry is kept when it overlaps the weekend; a long one only when it starts
    within it, so a months-long exhibition appears in its opening weekend alone.
    """
    sunday = friday + timedelta(days=2)
    last_day = entry.end or entry.start
    if (last_day - entry.start).days + 1 <= _MAX_SHORT_ENTRY_DAYS:
        return entry.start <= sunday and last_day >= friday
    return friday <= entry.start <= sunday


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=20),
    reraise=True,
)
async def _fetch_day(client: httpx.AsyncClient, url: str, day: date) -> list[CalendarEntry]:
    """One day of the Calendar, as its day picker requests it."""
    response = await client.get(
        url,
        params={
            "action": _DAY_ACTION,
            "event_id": _CALENDAR_ID,
            "dataw": f"{day.month:02d}-{day.year}",
            "dzien": str(day.day),
        },
    )
    response.raise_for_status()
    return parse_calendar_day(response.text, url)


async def read_weekend_calendar(
    config: DigestConfig, friday: date, *, timeout: float
) -> CalendarReading:
    """The Calendar entries the Digest of the weekend starting on *friday* should consider.

    Entries listed on several days are merged by their ID. A request or parse failure on
    any day, or no entries on all three days, is a failure with no entries: a Calendar
    that lists nothing for a whole weekend has most likely changed its markup.
    """
    if not config.enabled:
        logger.info("calendar_disabled")
        return CalendarReading(entries=[])

    sunday = friday + timedelta(days=2)
    weekend = f"{friday:%d.%m}–{sunday:%d.%m.%Y}"
    log = logger.bind(weekend=friday.isoformat())
    try:
        listed = await _read_weekend_days(config.calendar_url, friday, timeout)
    except Exception as exc:
        log.warning("calendar_read_failed", error_type=type(exc).__name__, error=str(exc))
        return CalendarReading(
            entries=[],
            failure=f"Календар подій на {weekend} не прочитано ({type(exc).__name__}: {exc}).",
        )

    if not listed:
        log.warning("calendar_empty")
        return CalendarReading(
            entries=[],
            failure=f"Календар подій не має жодного запису на {weekend}: "
            "можливо, змінилася розмітка сайту.",
        )
    kept = [e for e in listed if in_weekend(e, friday)]
    log.info("calendar_read", listed=len(listed), kept=len(kept))
    return CalendarReading(entries=kept)


async def _read_weekend_days(url: str, friday: date, timeout: float) -> list[CalendarEntry]:
    """Every entry listed on Friday, Saturday or Sunday, once each; raises on any day."""
    listed: dict[int, CalendarEntry] = {}
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=_HEADERS, http2=True
    ) as client:
        for offset in range(3):
            for entry in await _fetch_day(client, url, friday + timedelta(days=offset)):
                listed.setdefault(entry.id, entry)
    return list(listed.values())
