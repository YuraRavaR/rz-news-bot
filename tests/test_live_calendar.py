"""Canary test against the city's events Calendar (erzeszow.pl).

Deselected by default (`-m 'not live'` in pyproject) and run in the daily canary
workflow, like tests/test_live_sources.py. A failure means the Calendar's day-picker
endpoint or its markup changed: update parse_calendar_day and refresh the frozen
responses in tests/fixtures/calendar/ from the live endpoint.

Run manually:
    uv run pytest -m live --no-cov tests/test_live_calendar.py
"""

import pytest

from rz_flow.city_calendar import read_weekend_calendar
from rz_flow.digest import digest_weekend, today_in_warsaw
from rz_flow.flow_config import load_flow_config

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

_TIMEOUT = 30.0


async def test_calendar_still_lists_the_coming_weekend() -> None:
    """The coming weekend's three days must be read and list at least one entry.

    Long exhibitions run through almost every weekend, so an empty Calendar means the
    request was blocked or the markup changed.
    """
    config = load_flow_config().digest
    friday, _ = digest_weekend(today_in_warsaw())

    reading = await read_weekend_calendar(
        config.model_copy(update={"enabled": True}), friday, timeout=_TIMEOUT
    )

    assert reading.failure is None, reading.failure
    for entry in reading.entries:
        assert entry.title_pl.strip(), f"empty title for {entry.url}"
        assert entry.url.startswith("https://"), f"malformed URL: {entry.url}"
