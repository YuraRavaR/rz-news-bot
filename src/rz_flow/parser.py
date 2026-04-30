"""HTML parser for rzeszow24.info article listings.

Parses two types of article cards found on category pages:
  1. image-tile-overlay  — featured tiles (title only)
  2. news-listing-item   — text list with title + lead paragraph
"""

import re
from typing import Final

from bs4 import BeautifulSoup, Tag

from rz_flow.models import Article, Category

# Regex to extract the unique article ID from the URL tail
# URL pattern: https://rzeszow24.info/{category}/{slug}/{ID}
_ID_RE: Final = re.compile(r"/([A-Za-z0-9_-]{15,30})/?$")

# Category badge text → Category enum (the site uses Polish names)
_BADGE_MAP: Final[dict[str, Category]] = {
    "imprezy": Category.IMPREZY,
    "wiadomości": Category.WIADOMOSCI,
    "wiadomosci": Category.WIADOMOSCI,
    "sport": Category.WIADOMOSCI,  # sport is grouped with news for AI scoring
    "podkarpacie": Category.WIADOMOSCI,
    "praca rzeszów": Category.WIADOMOSCI,
    "polska i świat": Category.WIADOMOSCI,
}


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


def parse_category_page(html: str, category: Category) -> list[Article]:
    """Parse a category listing page and return unique Article objects.

    Works on both /imprezy/ and /wiadomosci/ page HTML.
    Deduplicates by article ID. Ignores sponsored content.
    """
    soup = BeautifulSoup(html, "lxml")
    articles: dict[str, Article] = {}

    # ── 1. Image tile cards (featured, top section) ──────────────────────────
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        if f"/{category.value}/" not in href:
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
            # Fall back to mobile title
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

    # ── 2. Text listing items (regular list, has lead paragraph) ─────────────
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        if f"/{category.value}/" not in href:
            continue
        if _is_sponsored(link):
            continue

        text_block = link.find(class_="news-listing-item__text")
        if not text_block:
            continue

        article_id = _extract_id(href)
        if not article_id or article_id in articles:
            continue

        # Title is in the <strong> tag; summary is the surrounding text
        title_el = text_block.find("strong")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        if not title:
            continue

        # Extract lead text: everything after </strong> in the paragraph
        title_el.extract()
        lead = text_block.get_text(strip=True).lstrip(".,: ").strip()
        # Remove trailing ellipsis marker "(…)" or "(...)"
        lead = re.sub(r"\s*\(\.{3}\)\s*$", "", lead).strip()

        articles[article_id] = Article(
            id=article_id,
            url=href,
            category=category,
            title_pl=title,
            summary_pl=lead,
        )

    return list(articles.values())
