"""Tests for the HTML parser (offline, no network).

These tests run against real saved HTML fixtures — fast and deterministic.
This is a key AQA concept: separate I/O (scraper) from logic (parser),
so the logic can be tested without network access.
"""

from rz_flow.models import Article, Category
from rz_flow.parser import (
    _clean_title,
    _extract_id,
    parse_category_page,
    parse_najnowsze_page,
    parse_rzeszow_news_page,
)


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


class TestParseImprezyPage:
    def test_returns_list_of_articles(self, imprezy_html: str) -> None:
        articles = parse_category_page(imprezy_html, Category.IMPREZY)
        assert isinstance(articles, list)
        assert len(articles) > 0

    def test_all_articles_have_required_fields(self, imprezy_html: str) -> None:
        articles = parse_category_page(imprezy_html, Category.IMPREZY)
        for article in articles:
            assert isinstance(article, Article)
            assert article.id, f"Empty id for {article.url}"
            assert article.url.startswith("https://"), f"Bad URL: {article.url}"
            assert article.title_pl, f"Empty title for {article.id}"
            assert article.category == Category.IMPREZY

    def test_article_ids_are_unique(self, imprezy_html: str) -> None:
        articles = parse_category_page(imprezy_html, Category.IMPREZY)
        ids = [a.id for a in articles]
        assert len(ids) == len(set(ids)), "Duplicate article IDs found"

    def test_urls_contain_imprezy_path(self, imprezy_html: str) -> None:
        articles = parse_category_page(imprezy_html, Category.IMPREZY)
        for a in articles:
            assert "/imprezy/" in a.url, f"Wrong category URL: {a.url}"

    def test_no_sponsored_content(self, imprezy_html: str) -> None:
        articles = parse_category_page(imprezy_html, Category.IMPREZY)
        for a in articles:
            assert "sponsorowany" not in a.title_pl.lower()
            assert "materiał promocyjny" not in a.title_pl.lower()

    def test_at_least_5_articles(self, imprezy_html: str) -> None:
        """Sanity check — listing page should have a reasonable number of articles."""
        articles = parse_category_page(imprezy_html, Category.IMPREZY)
        assert len(articles) >= 5


class TestParseWiadomosciPage:
    def test_returns_articles_with_summaries(self, wiadomosci_html: str) -> None:
        articles = parse_category_page(wiadomosci_html, Category.WIADOMOSCI)
        assert len(articles) > 0
        # At least some articles should have a summary (from news-listing-item)
        with_summary = [a for a in articles if a.summary_pl]
        assert len(with_summary) > 0, "Expected some articles with summary_pl"

    def test_all_articles_are_wiadomosci_category(self, wiadomosci_html: str) -> None:
        articles = parse_category_page(wiadomosci_html, Category.WIADOMOSCI)
        for a in articles:
            assert a.category == Category.WIADOMOSCI

    def test_summaries_do_not_contain_title_text(self, wiadomosci_html: str) -> None:
        """Summary should be the lead text, not a repetition of the title."""
        articles = parse_category_page(wiadomosci_html, Category.WIADOMOSCI)
        for a in articles:
            if a.summary_pl:
                # Summary should not start with the full title
                assert not a.summary_pl.startswith(a.title_pl), (
                    f"Summary looks like a title repeat for {a.id}"
                )

    def test_summaries_stripped_of_ellipsis(self, wiadomosci_html: str) -> None:
        articles = parse_category_page(wiadomosci_html, Category.WIADOMOSCI)
        for a in articles:
            assert not a.summary_pl.endswith("(...)"), (
                f"Summary still has ellipsis: {a.summary_pl[-30:]}"
            )


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


class TestParseMinimalHtml:
    """Unit tests with hand-crafted minimal HTML — no fixture dependency."""

    MINIMAL_IMPREZY = """
    <html><body>
    <a href="https://rzeszow24.info/imprezy/test-event/ABCDEFGHIJKLMNOP" target="_self">
      <div class="image-tile-overlay">
        <div class="image-tile-overlay__wrapper">
          <h3 class="image-tile-overlay__title">Test Festival 2026</h3>
        </div>
      </div>
    </a>
    </body></html>
    """

    MINIMAL_NEWS = """
    <html><body>
    <a href="https://rzeszow24.info/wiadomosci/news-slug/XYZ1234567890ABC" target="_self">
      <div class="news-listing-item__wrapper">
        <p class="news-listing-item__text">
          <strong>Breaking News Title</strong>
          This is the lead paragraph with details.
        </p>
      </div>
    </a>
    </body></html>
    """

    def test_minimal_tile_card_parsed(self) -> None:
        articles = parse_category_page(self.MINIMAL_IMPREZY, Category.IMPREZY)
        assert len(articles) == 1
        assert articles[0].id == "ABCDEFGHIJKLMNOP"
        assert articles[0].title_pl == "Test Festival 2026"
        assert articles[0].summary_pl == ""

    def test_minimal_news_listing_parsed(self) -> None:
        articles = parse_category_page(self.MINIMAL_NEWS, Category.WIADOMOSCI)
        assert len(articles) == 1
        assert articles[0].id == "XYZ1234567890ABC"
        assert articles[0].title_pl == "Breaking News Title"
        assert "lead paragraph" in articles[0].summary_pl

    def test_sponsored_link_excluded(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/sponsored/SPONSOREDID123456">
          <div class="image-tile-overlay">
            <span class="badge">sponsorowany</span>
            <h3 class="image-tile-overlay__title">Buy Our Product</h3>
          </div>
        </a>
        </body></html>
        """
        articles = parse_category_page(html, Category.IMPREZY)
        assert len(articles) == 0

    def test_empty_html_returns_empty_list(self) -> None:
        assert parse_category_page("<html><body></body></html>", Category.IMPREZY) == []


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
        """When h3.image-tile-overlay__title is absent, falls back to .image-tile-overlay-mobile p."""
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

    # ── parse_category_page deduplication ─────────────────────────────────────

    def test_category_page_deduplicates_same_id_across_tile_and_listing(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/event/CATDUPID12345678AB" target="_self">
          <div class="image-tile-overlay">
            <h3 class="image-tile-overlay__title">Festival Tile</h3>
          </div>
        </a>
        <a href="https://rzeszow24.info/imprezy/event/CATDUPID12345678AB" target="_self">
          <div class="news-listing-item__wrapper">
            <p class="news-listing-item__text">
              <strong>Festival Listing</strong> Lead text.
            </p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_category_page(html, Category.IMPREZY)
        assert len(articles) == 1
        assert articles[0].id == "CATDUPID12345678AB"

    def test_category_page_uses_mobile_title_fallback(self) -> None:
        """parse_category_page also falls back to mobile title when h3 is absent."""
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/event/CATMOBILEID12345678" target="_self">
          <div class="image-tile-overlay">
            <!-- No h3 title -->
          </div>
          <div class="image-tile-overlay-mobile">
            <p>Mobile Category Title</p>
          </div>
        </a>
        </body></html>
        """
        articles = parse_category_page(html, Category.IMPREZY)
        assert len(articles) == 1
        assert articles[0].title_pl == "Mobile Category Title"

    def test_category_page_skips_tile_with_no_title_at_all(self) -> None:
        html = """
        <html><body>
        <a href="https://rzeszow24.info/imprezy/event/EMPTYTILEID12345678" target="_self">
          <div class="image-tile-overlay">
            <!-- No h3, no mobile -->
          </div>
        </a>
        </body></html>
        """
        articles = parse_category_page(html, Category.IMPREZY)
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
