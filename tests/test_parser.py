"""Tests for the HTML parser (offline, no network).

These tests run against real saved HTML fixtures — fast and deterministic.
This is a key AQA concept: separate I/O (scraper) from logic (parser),
so the logic can be tested without network access.
"""

import json
from datetime import date

import pytest

from rz_flow.models import Article, Category
from rz_flow.parser import (
    CalendarParseError,
    _clean_title,
    _extract_id,
    parse_calendar_day,
    parse_najnowsze_page,
    parse_rzeszow_news_page,
)
from tests.conftest import calendar_day_body

_CALENDAR_BASE = "https://erzeszow.pl/index.php"


class TestExtractId:
    def test_extracts_id_from_standard_url(self) -> None:
        url = "https://rzeszow24.info/imprezy/healthy-day/vVfqNNrtDBNYDiO0dtfM"
        assert _extract_id(url) == "vVfqNNrtDBNYDiO0dtfM"

    def test_extracts_id_with_trailing_slash(self) -> None:
        url = "https://rzeszow24.info/wiadomosci/some-slug/utZhdWP2F2qWIA7hetcZ/"
        assert _extract_id(url) == "utZhdWP2F2qWIA7hetcZ"

    def test_returns_none_for_category_root(self) -> None:
        assert _extract_id("https://rzeszow24.info/imprezy/") is None

    def test_returns_none_for_short_tail(self) -> None:
        assert _extract_id("https://rzeszow24.info/imprezy/abc/") is None


class TestCleanTitle:
    def test_strips_whitespace(self) -> None:
        assert _clean_title("  Hello  ") == "Hello"

    def test_deduplicates_repeated_text(self) -> None:
        doubled = "Festival w RzeszowFestival w Rzeszow"
        assert _clean_title(doubled) == "Festival w Rzeszow"

    def test_leaves_normal_title_unchanged(self) -> None:
        title = "Trening z Anną Lewandowską w Rzeszowie"
        assert _clean_title(title) == title


class TestParseNajnowszePage:
    """Tests for the /najnowsze unified feed parser."""

    def test_returns_articles_from_fixture(self, najnowsze_html: str) -> None:
        articles = parse_najnowsze_page(najnowsze_html)
        assert len(articles) > 0

    def test_all_articles_have_required_fields(self, najnowsze_html: str) -> None:
        articles = parse_najnowsze_page(najnowsze_html)
        for a in articles:
            assert isinstance(a, Article)
            assert a.id, f"Empty id for {a.url}"
            assert a.url.startswith("https://"), f"Bad URL: {a.url}"
            assert a.title_pl, f"Empty title for {a.id}"
            assert a.category in (Category.IMPREZY, Category.WIADOMOSCI)

    def test_article_ids_are_unique(self, najnowsze_html: str) -> None:
        articles = parse_najnowsze_page(najnowsze_html)
        ids = [a.id for a in articles]
        assert len(ids) == len(set(ids)), "Duplicate article IDs found"

    def test_no_sponsored_content(self, najnowsze_html: str) -> None:
        articles = parse_najnowsze_page(najnowsze_html)
        for a in articles:
            assert "sponsorowany" not in a.title_pl.lower()
            assert "materiał promocyjny" not in a.title_pl.lower()

    def test_tile_card_parsed(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/test-event/ABCDEFGHIJKLMNOP">
          <div class="image-tile-overlay">
            <h3 class="image-tile-overlay__title">Test Festival 2026</h3>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 1
        assert articles[0].id == "ABCDEFGHIJKLMNOP"
        assert articles[0].title_pl == "Test Festival 2026"
        assert articles[0].category == Category.IMPREZY
        assert articles[0].summary_pl == ""

    def test_listing_item_parsed(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/wiadomosci/news-slug/XYZ1234567890ABC">
          <div class="news-listing-item__wrapper">
            <p class="news-listing-item__text">
              <strong>Breaking News Title</strong>
              This is the lead paragraph with details.
            </p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 1
        assert articles[0].id == "XYZ1234567890ABC"
        assert articles[0].title_pl == "Breaking News Title"
        assert articles[0].category == Category.WIADOMOSCI
        assert "lead paragraph" in articles[0].summary_pl

    def test_mixed_categories_both_returned(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/fest/IMPREZYID123456789">
          <div class="image-tile-overlay">
            <h3 class="image-tile-overlay__title">Festival</h3>
          </div>
        </a>
        <a href="https://rzeszow24.info/wiadomosci/news/WIADOMID123456789">
          <div class="news-listing-item__wrapper">
            <p class="news-listing-item__text">
              <strong>City News</strong> Some details.
            </p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 2
        categories = {a.category for a in articles}
        assert Category.IMPREZY in categories
        assert Category.WIADOMOSCI in categories

    def test_unknown_category_url_skipped(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/nieobslugiwana-sekcja/artykul/UNKNOWNCAT12345678">
          <div class="image-tile-overlay">
            <h3 class="image-tile-overlay__title">Should Be Skipped</h3>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 0

    def test_sport_urls_parsed_as_wiadomosci(self) -> None:
        """Sport items appear on /najnowsze with /sport/ URLs — must not be dropped."""
        html = """
        <html><body>
        <a href="https://rzeszow24.info/sport/kontrakt/SPORTARTICLE123456">
          <div class="news-listing-item__wrapper">
            <p class="news-listing-item__text">
              <strong>Resovia extends contract</strong>
              Lead text here.
            </p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 1
        assert articles[0].id == "SPORTARTICLE123456"
        assert articles[0].category == Category.WIADOMOSCI
        assert articles[0].title_pl == "Resovia extends contract"

    def test_sponsored_content_excluded(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/wiadomosci/ad/SPONSOREDID123456789">
          <div class="news-listing-item__wrapper">
            <p class="news-listing-item__text">
              <span class="badge">sponsorowany</span>
              <strong>Buy Our Product</strong>
            </p>
          </div>
        </a>
        </body></html>
        """
        assert parse_najnowsze_page(html) == []

    def test_empty_html_returns_empty_list(self) -> None:
        assert parse_najnowsze_page("<html><body></body></html>") == []


class TestParseRzeszowNewsPage:
    """Tests for the rzeszow-news.pl td_module_10 card parser."""

    MINIMAL_CARD = """
    <html><body>
    <div class="td_module_10 td_module_wrap td-animation-stack">
      <div class="td-module-thumb">
        <a href="https://rzeszow-news.pl/majowka-w-lancucie-nocne-zwiedzanie/" rel="bookmark">
          <img src="img.jpg" alt="test">
        </a>
      </div>
      <div class="item-details">
        <h3 class="entry-title td-module-title">
          <a href="https://rzeszow-news.pl/majowka-w-lancucie-nocne-zwiedzanie/" rel="bookmark">
            Majówka w Łańcucie: nocne zwiedzanie
          </a>
        </h3>
        <div class="td-excerpt">
          Już w najbliższą sobotę zamek otworzy podwoje w nocnej scenerii...
        </div>
      </div>
    </div>
    </body></html>
    """

    SPONSORED_CARD = """
    <html><body>
    <div class="td_module_10 td_module_wrap td-animation-stack">
      <div class="item-details">
        <h3 class="entry-title td-module-title">
          <a href="https://rzeszow-news.pl/oferta-pracy-pomoc-kuchenna/">
            Oferta pracy: Pomoc kuchenna w Rzeszowie
          </a>
        </h3>
        <div class="td-excerpt">Firma poszukuje osoby na stanowisko pomoc kuchenna...</div>
      </div>
    </div>
    </body></html>
    """

    def test_returns_articles_from_fixture(self, rzeszow_news_html: str) -> None:
        articles = parse_rzeszow_news_page(rzeszow_news_html)
        assert len(articles) > 0

    def test_all_articles_have_required_fields(self, rzeszow_news_html: str) -> None:
        articles = parse_rzeszow_news_page(rzeszow_news_html)
        for a in articles:
            assert isinstance(a, Article)
            assert a.id, f"Empty id for {a.url}"
            assert a.url.startswith("https://rzeszow-news.pl/"), f"Bad URL: {a.url}"
            assert a.title_pl, f"Empty title for {a.id}"
            assert a.category == Category.WIADOMOSCI

    def test_article_ids_are_unique(self, rzeszow_news_html: str) -> None:
        articles = parse_rzeszow_news_page(rzeszow_news_html)
        ids = [a.id for a in articles]
        assert len(ids) == len(set(ids)), "Duplicate article IDs found"

    def test_articles_have_summaries(self, rzeszow_news_html: str) -> None:
        articles = parse_rzeszow_news_page(rzeszow_news_html)
        with_summary = [a for a in articles if a.summary_pl]
        assert len(with_summary) > 0, "Expected articles with summary_pl"

    def test_summaries_stripped_of_ellipsis(self, rzeszow_news_html: str) -> None:
        articles = parse_rzeszow_news_page(rzeszow_news_html)
        for a in articles:
            assert not a.summary_pl.endswith("..."), (
                f"Summary still has ellipsis: {a.summary_pl[-30:]}"
            )

    def test_minimal_card_parsed(self) -> None:
        articles = parse_rzeszow_news_page(self.MINIMAL_CARD)
        assert len(articles) == 1
        assert articles[0].id == "majowka-w-lancucie-nocne-zwiedzanie"
        assert articles[0].title_pl == "Majówka w Łańcucie: nocne zwiedzanie"
        assert articles[0].category == Category.WIADOMOSCI
        assert "zamek" in articles[0].summary_pl

    def test_sponsored_job_ads_excluded(self) -> None:
        articles = parse_rzeszow_news_page(self.SPONSORED_CARD)
        assert len(articles) == 0

    def test_empty_html_returns_empty_list(self) -> None:
        assert parse_rzeszow_news_page("<html><body></body></html>") == []


class TestParserEdgeCases:
    """Guard-clause and deduplication edge cases without fixture dependency."""

    # ── parse_najnowsze_page deduplication ────────────────────────────────────

    def test_najnowsze_deduplicates_same_id_appearing_as_tile_and_listing(self) -> None:
        """If the same article ID appears in both tile and listing sections, it's returned once."""
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/event/DUPLICATE12345678">
          <div class="image-tile-overlay">
            <h3 class="image-tile-overlay__title">Festival Tile</h3>
          </div>
        </a>
        <a href="https://rzeszow24.info/imprezy/event/DUPLICATE12345678">
          <div class="news-listing-item__wrapper">
            <p class="news-listing-item__text">
              <strong>Festival Listing</strong> Some lead text here.
            </p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 1
        assert articles[0].id == "DUPLICATE12345678"

    def test_najnowsze_skips_tile_with_no_title_element(self) -> None:
        """Tile card without h3 and without mobile fallback is skipped."""
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/event/NOTITLEID123456789">
          <div class="image-tile-overlay">
            <div class="image-tile-overlay__wrapper">
              <!-- No h3 here, no mobile fallback -->
            </div>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 0

    def test_najnowsze_uses_mobile_title_fallback(self) -> None:
        """Falls back to .image-tile-overlay-mobile p when the h3 title is absent."""
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/event/MOBILEID1234567890">
          <div class="image-tile-overlay">
            <!-- No h3 -->
          </div>
          <div class="image-tile-overlay-mobile">
            <p>Mobile Title Text</p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 1
        assert articles[0].title_pl == "Mobile Title Text"

    def test_najnowsze_listing_skips_item_with_no_strong_tag(self) -> None:
        """Listing item without <strong> (no title) is skipped."""
        html = """
        <html><body>
        <a href="https://rzeszow24.info/wiadomosci/news/NOSTRONGID1234567">
          <div class="news-listing-item__wrapper">
            <p class="news-listing-item__text">
              Just plain text without strong tag.
            </p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_najnowsze_page(html)
        assert len(articles) == 0

    # ── parse_rzeszow_news_page deduplication ──────────────────────────────────

    def test_rzeszow_news_deduplicates_same_slug(self) -> None:
        """If two cards share the same slug, only one article is returned."""
        html = """
        <html><body>
        <div class="td_module_10 td_module_wrap">
          <h3 class="entry-title td-module-title">
            <a href="https://rzeszow-news.pl/duplicate-slug-article/">First</a>
          </h3>
        </div>
        <div class="td_module_10 td_module_wrap">
          <h3 class="entry-title td-module-title">
            <a href="https://rzeszow-news.pl/duplicate-slug-article/">Second</a>
          </h3>
        </div>
        </body></html>
        """
        articles = parse_rzeszow_news_page(html)
        assert len(articles) == 1

    def test_rzeszow_news_skips_card_without_h3(self) -> None:
        html = """
        <html><body>
        <div class="td_module_10 td_module_wrap">
          <!-- No h3.entry-title -->
          <div class="td-excerpt">Some text</div>
        </div>
        </body></html>
        """
        articles = parse_rzeszow_news_page(html)
        assert len(articles) == 0

    def test_rzeszow_news_skips_card_with_non_matching_url(self) -> None:
        """Cards with URLs not matching the rzeszow-news.pl slug pattern are skipped."""
        html = """
        <html><body>
        <div class="td_module_10 td_module_wrap">
          <h3 class="entry-title td-module-title">
            <a href="https://other-site.com/article/">External link</a>
          </h3>
        </div>
        </body></html>
        """
        articles = parse_rzeszow_news_page(html)
        assert len(articles) == 0


# ── erzeszow.pl Calendar ──────────────────────────────────────────────────────


class TestParseCalendarDay:
    def test_single_day_entry(self) -> None:
        entries = parse_calendar_day(calendar_day_body("2026-10-31"), _CALENDAR_BASE)

        concert = next(e for e in entries if e.id == 151427)
        assert concert.title_pl == "COSMOS - Koniec Lata | Koncert"
        assert concert.start == date(2026, 10, 31)
        assert concert.end is None
        assert concert.url == (
            "https://erzeszow.pl/41-miasto-rzeszow/6241-kalendarz-imprez/"
            "151427-cosmos-koniec-lata-_-koncert.html"
        )

    def test_multi_day_entry_has_its_range(self) -> None:
        entries = parse_calendar_day(calendar_day_body("2026-10-30"), _CALENDAR_BASE)

        exhibition = next(e for e in entries if e.id == 146386)
        assert exhibition.title_pl == "Wystawa czasowa: „Pejzaż w malarstwie nowożytnym”"
        assert exhibition.start == date(2026, 6, 30)
        assert exhibition.end == date(2026, 11, 30)

    def test_reads_both_tabs(self) -> None:
        """Friday 30.10.2026 lists 1 one-day entry and 20 multi-day ones."""
        entries = parse_calendar_day(calendar_day_body("2026-10-30"), _CALENDAR_BASE)

        assert len(entries) == 21
        assert sum(e.end is None for e in entries) == 1

    def test_entry_listed_twice_in_one_day_is_returned_once(self) -> None:
        article = _calendar_article("/x/7-a.html", "A", "2026-10-30")
        body = json.dumps(
            {
                "html": f'<div id="events-oneday">{article}</div>'
                f'<div id="events-multidays">{article}</div>'
            }
        )

        assert [e.id for e in parse_calendar_day(body, _CALENDAR_BASE)] == [7]

    def test_entries_sharing_a_title_stay_apart(self) -> None:
        """Two workshop series with one title have their own IDs, so two entries."""
        entries = parse_calendar_day(calendar_day_body("2026-11-01"), _CALENDAR_BASE)

        assert {151745, 151748} <= {e.id for e in entries}

    def test_day_without_one_day_entries(self) -> None:
        entries = parse_calendar_day(calendar_day_body("2026-11-01"), _CALENDAR_BASE)

        assert entries
        assert all(e.end is not None for e in entries)

    def test_id_is_the_number_before_the_slug(self) -> None:
        body = _calendar_body(
            _calendar_article(
                "/41-miasto-rzeszow/6241-kalendarz-imprez/153244-wystawa-2026-_-instalacja.html",
                "Wystawa",
                "2026-10-30",
            )
        )

        [entry] = parse_calendar_day(body, _CALENDAR_BASE)
        assert entry.id == 153244

    def test_same_start_and_end_is_a_single_day(self) -> None:
        body = _calendar_body(_calendar_article("/x/1-a.html", "A", "2026-10-30 - 2026-10-30"))

        [entry] = parse_calendar_day(body, _CALENDAR_BASE)
        assert entry.end is None

    def test_entry_without_link_numeric_id_or_date_is_skipped(self) -> None:
        body = _calendar_body(
            _calendar_article("/x/no-id.html", "No ID", "2026-10-30")
            + _calendar_article("/x/2-no-date.html", "No date", "wkrótce")
            + _calendar_article("/x/3-bad-date.html", "Bad date", "2026-13-45")
            + _calendar_article("/x/4-kept.html", "Kept", "2026-10-30")
            + "<article><h2>No link</h2></article>"
        )

        assert [e.id for e in parse_calendar_day(body, _CALENDAR_BASE)] == [4]

    @pytest.mark.parametrize(
        "body",
        [
            "<html><body>Przerwa techniczna</body></html>",
            json.dumps({"error": "no such action"}),
            json.dumps({"html": None}),
            json.dumps(["html"]),
            json.dumps({"html": "<div>Strona główna</div>"}),
        ],
        ids=["html_page", "no_html_key", "html_not_a_string", "not_an_object", "no_event_list"],
    )
    def test_unexpected_body_raises(self, body: str) -> None:
        with pytest.raises(CalendarParseError):
            parse_calendar_day(body, _CALENDAR_BASE)


def _calendar_article(href: str, title: str, dates: str) -> str:
    return f"""
    <article>
      <h2 class="modal__event--title"><a href="{href}">{title}</a></h2>
      <div class="modal__event--data"><div><span>Data wydarzenia:</span>{dates}</div></div>
    </article>
    """


def _calendar_body(articles: str) -> str:
    html = f'<div id="events-oneday">{articles}</div><div id="events-multidays"></div>'
    return json.dumps({"html": html})
