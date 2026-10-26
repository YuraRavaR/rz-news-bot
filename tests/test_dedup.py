"""Tests for title normalisation used to spot Repeats."""

import pytest

from rz_flow.dedup import normalize_title


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # case
        ("Koncert W RZESZOWIE", "koncert w rzeszowie"),
        # punctuation
        ("Koncert, w Rzeszowie!", "koncert w rzeszowie"),
        ("„Wielki” koncert: (relacja) — 2026", "wielki koncert relacja 2026"),
        ("Rzeszów-Jasionka...", "rzeszow jasionka"),
        # whitespace
        ("  Koncert \t w\n\nRzeszowie  ", "koncert w rzeszowie"),
        # Polish diacritics, ł included
        ("Zażółć gęślą jaźń", "zazolc gesla jazn"),
        ("ŁÓDŹ Śródmieście", "lodz srodmiescie"),
        # nothing left
        ("!!! ...", ""),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected
