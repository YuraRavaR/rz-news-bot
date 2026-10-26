"""Repeat detection: has the AI already judged another Article about the same Story?

Checked before Gemini, so a Repeat costs no quota. Two Articles are about the same
Story when their normalised titles are equal (across all Sources), or when they come
from the same Source and share a slug. A merely similar title (token Jaccard) is a
near match: it never blocks an Article and only adds a hint to the run report.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rz_flow.models import Article

# A near match at or above this token Jaccard similarity is hinted in the run report.
# It never blocks an Article: it is evidence for whether a smarter dedup is worth it.
NEAR_MATCH_THRESHOLD = 0.5

# Letters NFKD does not decompose into a base letter plus a combining mark
_UNDECOMPOSABLE = str.maketrans({"ł": "l"})
_NON_WORD = re.compile(r"[\W_]+")


def normalize_title(title: str) -> str:
    """Casefold, fold Polish diacritics, drop punctuation, collapse whitespace."""
    folded = title.casefold().translate(_UNDECOMPOSABLE)
    decomposed = unicodedata.normalize("NFKD", folded)
    letters = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_NON_WORD.sub(" ", letters).split())


def title_similarity(a: str, b: str) -> float:
    """Token Jaccard similarity of two titles after normalisation (0.0 to 1.0)."""
    tokens_a = set(normalize_title(a).split())
    tokens_b = set(normalize_title(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


# url → (Source name, slug), or None when no Source recognises the URL
SlugOf = Callable[[str], tuple[str, str] | None]


@dataclass(frozen=True)
class Original:
    """An AI-judged Article that later Articles about the same Story repeat."""

    article_id: str
    url: str
    title_pl: str


@dataclass(frozen=True)
class NearMatch:
    """The most similar original of an Article that is not a Repeat; never stored."""

    original: Original
    similarity: float


class StoryIndex:
    """The pool of originals for one Run: stored AI-judged Articles of the last 72 hours
    plus those judged earlier in this Run, looked up by normalised title, by
    (Source, slug) and by title similarity.
    """

    def __init__(self, slug_of: SlugOf) -> None:
        self._slug_of = slug_of
        self._by_title: dict[str, Original] = {}
        self._by_slug: dict[tuple[str, str], Original] = {}
        self._originals: list[Original] = []

    def add(self, original: Original) -> None:
        """Make ``original`` matchable; the first original for a key wins."""
        self._originals.append(original)
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

    def near_match(self, article: Article) -> NearMatch | None:
        """The most similar original, if it reaches ``NEAR_MATCH_THRESHOLD``.

        Ties go to the earliest original.
        """
        best: NearMatch | None = None
        for original in self._originals:
            if original.article_id == article.id:
                continue
            similarity = title_similarity(article.title_pl, original.title_pl)
            if similarity >= NEAR_MATCH_THRESHOLD and (
                best is None or similarity > best.similarity
            ):
                best = NearMatch(original=original, similarity=similarity)
        return best
