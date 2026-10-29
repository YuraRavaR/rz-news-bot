"""Tests for the Calendar reader (HTTP mocked with respx, frozen per-day responses)."""

import json
from datetime import date

import httpx
import pytest
import respx
from httpx import Response

from rz_flow.city_calendar import CalendarReading, in_weekend, read_weekend_calendar
from rz_flow.flow_config import DigestConfig
from rz_flow.models import CalendarEntry
from tests.conftest import calendar_day_body

FRIDAY = date(2026, 10, 30)  # weekend 30.10–1.11.2026
CALENDAR_URL = "https://erzeszow.pl/index.php"
CONFIG = DigestConfig(calendar_url=CALENDAR_URL, enabled=True)
WEEKEND_DAYS = [("10-2026", "30"), ("10-2026", "31"), ("11-2026", "1")]


def _entry(start: date, end: date | None = None, entry_id: int = 1) -> CalendarEntry:
    return CalendarEntry(
        id=entry_id,
        url=f"https://erzeszow.pl/41-miasto-rzeszow/6241-kalendarz-imprez/{entry_id}-e.html",
        title_pl=f"Wydarzenie {entry_id}",
        start=start,
        end=end,
    )


class TestInWeekend:
    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (date(2026, 10, 31), None),
            (date(2026, 10, 29), date(2026, 10, 31)),
            (date(2026, 11, 1), date(2026, 11, 3)),
            (date(2026, 10, 30), date(2026, 11, 1)),
        ],
        ids=["single_day_saturday", "thu_to_sat", "sun_to_tue", "the_whole_weekend"],
    )
    def test_short_entry_overlapping_the_weekend_is_kept(
        self, start: date, end: date | None
    ) -> None:
        assert in_weekend(_entry(start, end), FRIDAY)

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (date(2026, 10, 29), None),
            (date(2026, 11, 2), None),
            (date(2026, 10, 27), date(2026, 10, 29)),
        ],
        ids=["thursday_before", "monday_after", "tue_to_thu_before"],
    )
    def test_short_entry_outside_the_weekend_is_dropped(
        self, start: date, end: date | None
    ) -> None:
        assert not in_weekend(_entry(start, end), FRIDAY)

    @pytest.mark.parametrize(
        "start",
        [date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 1)],
        ids=["friday", "saturday", "sunday"],
    )
    def test_long_entry_is_kept_in_its_opening_weekend(self, start: date) -> None:
        assert in_weekend(_entry(start, date(2027, 1, 31)), FRIDAY)

    @pytest.mark.parametrize(
        "start",
        [date(2026, 10, 28), date(2026, 6, 30), date(2026, 11, 2)],
        ids=["wednesday_before", "months_before", "monday_after"],
    )
    def test_long_entry_outside_its_opening_weekend_is_dropped(self, start: date) -> None:
        assert not in_weekend(_entry(start, date(2027, 1, 31)), FRIDAY)

    def test_four_days_is_long(self) -> None:
        """Thursday to Sunday lasts four days, so it counts only from its opening day."""
        assert not in_weekend(_entry(date(2026, 10, 29), date(2026, 11, 1)), FRIDAY)


def _mock_day(month: str, day: str, response: Response | Exception) -> respx.Route:
    """Mock one day's request; mocking the same day again replaces the earlier route."""
    route = respx.get(
        CALENDAR_URL,
        params={"action": "disp-wydarzenia-dzien", "dataw": month, "dzien": day},
    )
    if isinstance(response, Exception):
        return route.mock(side_effect=response)
    return route.mock(return_value=response)


def _mock_frozen_weekend() -> list[respx.Route]:
    return [
        _mock_day(month, day, Response(200, text=calendar_day_body(iso)))
        for (month, day), iso in zip(
            WEEKEND_DAYS, ["2026-10-30", "2026-10-31", "2026-11-01"], strict=True
        )
    ]


def _day_body(*entries: tuple[int, str]) -> str:
    """A per-day response listing (id, dates) entries."""
    articles = "".join(
        f'<article><h2><a href="/41-miasto-rzeszow/6241-kalendarz-imprez/{i}-e.html">'
        f'Wydarzenie {i}</a></h2><div class="modal__event--data">{dates}</div></article>'
        for i, dates in entries
    )
    html = f'<div id="events-oneday"></div><div id="events-multidays">{articles}</div>'
    return json.dumps({"html": html})


async def _read() -> CalendarReading:
    return await read_weekend_calendar(CONFIG, FRIDAY, timeout=5.0)


class TestReadWeekendCalendar:
    @respx.mock
    async def test_reads_each_weekend_day_and_keeps_weekend_entries(self) -> None:
        routes = _mock_frozen_weekend()

        reading = await _read()

        assert reading.failure is None
        assert all(r.call_count == 1 for r in routes)
        # The three one-day entries; every long exhibition opened before this weekend
        assert [e.id for e in reading.entries] == [153244, 151427, 153540]

    @respx.mock
    async def test_entry_listed_on_several_days_is_one_entry(self) -> None:
        festival = (7, "2026-10-30 - 2026-11-01")
        _mock_day("10-2026", "30", Response(200, text=_day_body(festival)))
        _mock_day("10-2026", "31", Response(200, text=_day_body(festival, (8, "2026-10-31"))))
        _mock_day("11-2026", "1", Response(200, text=_day_body(festival)))

        reading = await _read()

        assert reading.failure is None
        assert [(e.id, e.start, e.end) for e in reading.entries] == [
            (7, date(2026, 10, 30), date(2026, 11, 1)),
            (8, date(2026, 10, 31), None),
        ]

    @respx.mock
    async def test_long_entry_is_kept_in_its_opening_weekend(self) -> None:
        opening = (9, "2026-10-31 - 2027-01-31")
        _mock_day("10-2026", "30", Response(200, text=_day_body()))
        _mock_day("10-2026", "31", Response(200, text=_day_body(opening)))
        _mock_day(
            "11-2026", "1", Response(200, text=_day_body(opening, (10, "2026-06-30 - 2026-11-30")))
        )

        reading = await _read()

        assert [e.id for e in reading.entries] == [9]

    @respx.mock
    async def test_weekend_entries_all_filtered_out_is_not_a_failure(self) -> None:
        old = (10, "2026-06-30 - 2026-11-30")
        for month, day in WEEKEND_DAYS:
            _mock_day(month, day, Response(200, text=_day_body(old)))

        reading = await _read()

        assert reading == CalendarReading(entries=[], failure=None)

    @pytest.mark.parametrize(
        "response",
        [Response(503), Response(404)],
        ids=["server_error", "not_found"],
    )
    @respx.mock
    async def test_http_error_is_a_failure(self, response: Response) -> None:
        _mock_frozen_weekend()
        _mock_day("10-2026", "31", response)

        reading = await _read()

        assert reading.entries == []
        assert reading.failure is not None
        assert str(response.status_code) in reading.failure

    @respx.mock
    async def test_network_error_is_a_failure(self) -> None:
        _mock_frozen_weekend()
        _mock_day("11-2026", "1", httpx.ConnectTimeout("timed out"))

        reading = await _read()

        assert reading.entries == []
        assert reading.failure is not None
        assert "ConnectTimeout" in reading.failure

    @respx.mock
    async def test_one_timeout_is_retried(self) -> None:
        _mock_frozen_weekend()
        route = respx.get(
            CALENDAR_URL,
            params={"action": "disp-wydarzenia-dzien", "dataw": "10-2026", "dzien": "30"},
        ).mock(
            side_effect=[
                httpx.ReadTimeout("timed out"),
                Response(200, text=calendar_day_body("2026-10-30")),
            ]
        )

        reading = await _read()

        assert reading.failure is None
        assert route.call_count == 2
        assert 153244 in [e.id for e in reading.entries]

    @respx.mock
    async def test_unparseable_body_is_a_failure(self) -> None:
        _mock_frozen_weekend()
        _mock_day("10-2026", "30", Response(200, text="<html>Przerwa techniczna</html>"))

        reading = await _read()

        assert reading.entries == []
        assert reading.failure is not None
        assert "CalendarParseError" in reading.failure

    @respx.mock
    async def test_no_entries_on_any_day_is_a_failure(self) -> None:
        for month, day in WEEKEND_DAYS:
            _mock_day(month, day, Response(200, text=_day_body()))

        reading = await _read()

        assert reading.entries == []
        assert reading.failure is not None

    async def test_disabled_makes_no_request(self, respx_mock: respx.MockRouter) -> None:
        reading = await read_weekend_calendar(
            DigestConfig(calendar_url=CALENDAR_URL, enabled=False), FRIDAY, timeout=5.0
        )

        assert reading == CalendarReading(entries=[], failure=None)
        assert not respx_mock.calls
