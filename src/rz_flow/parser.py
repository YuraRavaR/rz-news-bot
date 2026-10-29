"""HTML parsers for Rzeszów news sites.

rzeszow24.info
--------------
  parse_najnowsze_page(html)      — /najnowsze unified feed

rzeszow-news.pl
---------------
  parse_rzeszow_news_page(html)   — homepage td_module_10 cards

erzeszow.pl (the Calendar, read only for the Digest)
----------------------------------------------------
  parse_calendar_day(body, base_url) — one day's JSON response from the day picker
"""

import json
import re
from datetime import date
from typing import Final
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from rz_flow.models import Article, CalendarEntry, Category

# Regex to extract the unique article ID from the URL tail
# URL pattern: https://rzeszow24.info/{category}/{slug}/{ID}
_ID_RE: Final = re.compile(r"/([A-Za-z0-9_-]{15,30})/?$")


def _extract_id(url: str) -> str | None:
    """Extract the article's unique ID from its URL."""
    match = _ID_RE.search(url)
    return match.group(1) if match else None


def _clean_title(raw: str) -> str:
    """Remove duplicated text that sometimes appears (title repeated twice) and strip tags."""
    raw = raw.strip()
    mid = len(raw) // 2
    # Heuristic: if first half equals second half, deduplicate
    if len(raw) > 20 and raw[:mid].strip() == raw[mid:].strip():
        return raw[:mid].strip()
    return raw


def _is_sponsored(link: Tag) -> bool:
    """Return True if the article is a sponsored/promotional piece."""
    text = link.get_text(separator=" ", strip=True).lower()
    badges = link.find_all(class_=re.compile(r"badge"))
    badge_text = " ".join(b.get_text(strip=True).lower() for b in badges)
    keywords = ("sponsorowany", "sponsored", "materiał promocyjny", "promowane", "reklama")
    return any(kw in text or kw in badge_text for kw in keywords)


def _category_from_url(href: str) -> Category | None:
    """Infer article category from its URL path segment."""
    if "/imprezy/" in href:
        return Category.IMPREZY
    if "/wiadomosci/" in href:
        return Category.WIADOMOSCI
    # /najnowsze mixes in sport articles, grouped with news for AI scoring
    if "/sport/" in href:
        return Category.WIADOMOSCI
    return None


def parse_najnowsze_page(html: str) -> list[Article]:
    """Parse the /najnowsze (latest news) feed and return unique Article objects.

    Handles both image-tile and news-listing card types.
    Category is inferred from each article's URL (imprezy, wiadomosci, sport).
    Articles whose URL does not match a known category are skipped.
    Ignores sponsored content.
    """
    soup = BeautifulSoup(html, "lxml")
    articles: dict[str, Article] = {}

    # ── 1. Image tile cards ───────────────────────────────────────────────────
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        category = _category_from_url(href)
        if category is None:
            continue
        if _is_sponsored(link):
            continue

        tile = link.find(class_="image-tile-overlay")
        if not tile:
            continue

        article_id = _extract_id(href)
        if not article_id or article_id in articles:
            continue

        title_el = tile.find("h3", class_="image-tile-overlay__title")
        if not title_el:
            mobile = link.find(class_="image-tile-overlay-mobile")
            title_el = mobile.find("p") if mobile else None

        if not title_el:
            continue

        title = _clean_title(title_el.get_text(strip=True))
        if not title:
            continue

        articles[article_id] = Article(
            id=article_id,
            url=href,
            category=category,
            title_pl=title,
            summary_pl="",
        )

    # ── 2. Text listing items ─────────────────────────────────────────────────
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        category = _category_from_url(href)
        if category is None:
            continue
        if _is_sponsored(link):
            continue

        text_block = link.find(class_="news-listing-item__text")
        if not text_block:
            continue

        article_id = _extract_id(href)
        if not article_id or article_id in articles:
            continue

        title_el = text_block.find("strong")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        if not title:
            continue

        title_el.extract()
        lead = text_block.get_text(strip=True).lstrip(".,: ").strip()
        lead = re.sub(r"\s*\(\.{3}\)\s*$", "", lead).strip()

        articles[article_id] = Article(
            id=article_id,
            url=href,
            category=category,
            title_pl=title,
            summary_pl=lead,
        )

    return list(articles.values())


# ── rzeszow-news.pl ───────────────────────────────────────────────────────────

# Regex to extract the slug-based ID from rzeszow-news.pl article URLs.
# URL pattern: https://rzeszow-news.pl/{slug}/
_RN_SLUG_RE: Final = re.compile(r"rzeszow-news\.pl/([^/]+)/?$")

_RN_SPONSORED_KEYWORDS: Final = (
    "reklama",
    "sponsored",
    "materiał sponsorowany",
    "oferta pracy",
)


def _is_rn_sponsored(title: str, excerpt: str) -> bool:
    """Return True if the article looks like a job ad or sponsored post."""
    combined = (title + " " + excerpt).lower()
    return any(kw in combined for kw in _RN_SPONSORED_KEYWORDS)


def parse_rzeszow_news_page(html: str) -> list[Article]:
    """Parse rzeszow-news.pl homepage and return articles from td_module_10 cards.

    Each card contains a thumbnail, title link, and an excerpt paragraph.
    Category defaults to WIADOMOSCI — the AI evaluates actual content.
    Sponsored/job-listing cards are filtered out.
    """
    soup = BeautifulSoup(html, "lxml")
    articles: dict[str, Article] = {}

    for card in soup.find_all(class_="td_module_10"):
        title_el = card.find("h3", class_="entry-title")
        if not title_el:
            continue

        link = title_el.find("a", href=True)
        if not link:
            continue

        href = link.get("href", "")
        slug_match = _RN_SLUG_RE.search(href)
        if not slug_match:
            continue

        article_id = slug_match.group(1)
        if article_id in articles:
            continue

        title = link.get_text(strip=True)
        if not title:
            continue

        excerpt_el = card.find(class_="td-excerpt")
        excerpt = excerpt_el.get_text(strip=True) if excerpt_el else ""
        # Strip trailing CMS ellipsis ("...")
        excerpt = re.sub(r"\s*\.\.\.\s*$", "", excerpt).strip()

        if _is_rn_sponsored(title, excerpt):
            continue

        articles[article_id] = Article(
            id=article_id,
            url=href,
            category=Category.WIADOMOSCI,
            title_pl=title,
            summary_pl=excerpt,
        )

    return list(articles.values())


# ── erzeszow.pl Calendar ──────────────────────────────────────────────────────

# URL pattern: /41-miasto-rzeszow/6241-kalendarz-imprez/{ID}-{slug}.html
_CALENDAR_ID_RE: Final = re.compile(r"/(\d+)-[^/]*\.html$")
_CALENDAR_DATE_RE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")


class CalendarParseError(ValueError):
    """A Calendar response that does not have the expected shape."""


def _calendar_entry(article: Tag, base_url: str) -> CalendarEntry | None:
    link = article.select_one("h2 a[href]")
    dates_el = article.select_one(".modal__event--data")
    if link is None or dates_el is None:
        return None
    url = urljoin(base_url, str(link["href"]))
    id_match = _CALENDAR_ID_RE.search(url)
    title = link.get_text(strip=True)
    dates = _CALENDAR_DATE_RE.findall(dates_el.get_text(" ", strip=True))
    if id_match is None or not title or not dates:
        return None
    try:
        start = date.fromisoformat(dates[0])
        end = date.fromisoformat(dates[1]) if len(dates) > 1 else None
    except ValueError:
        return None
    return CalendarEntry(
        id=int(id_match.group(1)),
        url=url,
        title_pl=title,
        start=start,
        end=end if end != start else None,
    )


def parse_calendar_day(body: str, base_url: str) -> list[CalendarEntry]:
    """Parse one day of the Calendar: the JSON its day picker gets, holding the list's HTML.

    The list has two tabs, one-day and multi-day entries; both are returned, each entry
    once. An entry without a title, a date or a numeric ID in its URL is skipped.

    Raises:
        CalendarParseError: the body is not that JSON, or the HTML has neither tab.
    """
    try:
        html = json.loads(body)["html"]
    except (ValueError, TypeError, KeyError) as exc:
        raise CalendarParseError(f"not a Calendar day response: {exc}") from exc
    if not isinstance(html, str):
        raise CalendarParseError("not a Calendar day response: html is not a string")

    soup = BeautifulSoup(html, "lxml")
    panes = soup.select("#events-oneday, #events-multidays")
    if not panes:
        raise CalendarParseError("no event list in the Calendar day response")

    entries: dict[int, CalendarEntry] = {}
    for pane in panes:
        for article in pane.find_all("article"):
            entry = _calendar_entry(article, base_url)
            if entry is not None and entry.id not in entries:
                entries[entry.id] = entry
    return list(entries.values())
