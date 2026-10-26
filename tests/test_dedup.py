"""Tests for title normalisation and similarity used to spot Repeats and near matches."""

import pytest

from rz_flow.dedup import normalize_title, title_similarity


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


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Koncert w Rzeszowie", "Koncert w Rzeszowie", 1.0),
        # normalised first: case, punctuation and diacritics do not count
        ("Koncert, w RZESZÓWIE!", "koncert w rzeszowie", 1.0),
        # 4 shared tokens of 6 distinct
        ("Nowe rondo na ulicy Krakowskiej", "Nowe rondo na Krakowskiej otwarte", 4 / 6),
        # repeated tokens count once
        ("Rzeszów Rzeszów Rzeszów", "Rzeszów", 1.0),
        ("Koncert w filharmonii", "Mecz Resovii", 0.0),
        ("", "Koncert", 0.0),
        ("!!!", "...", 0.0),
    ],
)
def test_title_similarity(a: str, b: str, expected: float) -> None:
    assert title_similarity(a, b) == pytest.approx(expected)
