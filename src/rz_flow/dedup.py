"""Repeat detection: has the AI already judged another Article about the same Story?

Checked before Gemini, so a Repeat costs no quota. Two Articles are about the same
Story when their normalised titles are equal (across all Sources), or when they come
from the same Source and share a slug.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rz_flow.models import Article

# Letters NFKD does not decompose into a base letter plus a combining mark
_UNDECOMPOSABLE = str.maketrans({"ł": "l"})
_NON_WORD = re.compile(r"[\W_]+")


def normalize_title(title: str) -> str:
    """Casefold, fold Polish diacritics, drop punctuation, collapse whitespace."""
    folded = title.casefold().translate(_UNDECOMPOSABLE)
    decomposed = unicodedata.normalize("NFKD", folded)
    letters = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_NON_WORD.sub(" ", letters).split())


# url → (Source name, slug), or None when no Source recognises the URL
SlugOf = Callable[[str], tuple[str, str] | None]


@dataclass(frozen=True)
class Original:
    """An AI-judged Article that later Articles about the same Story repeat."""

    article_id: str
    url: str
    title_pl: str


class StoryIndex:
    """Originals of one Run, looked up by normalised title and by (Source, slug)."""

    def __init__(self, slug_of: SlugOf) -> None:
        self._slug_of = slug_of
        self._by_title: dict[str, Original] = {}
        self._by_slug: dict[tuple[str, str], Original] = {}

    def add(self, original: Original) -> None:
        """Make ``original`` matchable; the first original for a key wins."""
        title = normalize_title(original.title_pl)
        if title:
            self._by_title.setdefault(title, original)
        slug = self._slug_of(original.url)
        if slug is not None:
            self._by_slug.setdefault(slug, original)

    def find_original(self, article: Article) -> Original | None:
        """The original ``article`` repeats, if any (an Article never repeats itself)."""
        candidates = [self._by_title.get(normalize_title(article.title_pl))]
        slug = self._slug_of(article.url)
        if slug is not None:
            candidates.append(self._by_slug.get(slug))
        for original in candidates:
            if original is not None and original.article_id != article.id:
                return original
        return None
