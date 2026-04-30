"""Structlog configuration.

Two output modes, auto-selected:
  - TTY (local terminal): compact human-friendly lines; ai_evaluated is
    silently dropped because its info surfaces in published/skipped.
  - Non-TTY (GitHub Actions, pipe): structured JSON for machine parsing.

Error/warning events always show full context + traceback regardless of mode.

Usage:
    import structlog
    log = structlog.get_logger(__name__)
    log.info("published", ua_title="...", score=8.5)
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from urllib.parse import urlparse

import structlog

# ── ANSI colour helpers ────────────────────────────────────────────────────────
_R = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"


def _drop_ai_evaluated(logger: object, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop ai_evaluated — its data surfaces in the published/skipped line."""
    if event_dict.get("event") == "ai_evaluated":
        raise structlog.DropEvent()
    return event_dict


class _PrettyRenderer:
    """Single-line human-friendly renderer for TTY output.

    Info events  → compact, formatted line (no noise).
    Warn/error   → full context + traceback attached below.
    """

    def __call__(self, logger: object, method: str, event_dict: dict[str, Any]) -> str:
        level = event_dict.pop("level", "info")
        ts = event_dict.pop("timestamp", "")
        event = event_dict.pop("event", "?")
        exc_info = event_dict.pop("exception", None)

        # dry_run is implicit context — never useful inline
        event_dict.pop("dry_run", None)

        # article_id shown only on errors (keep it in dict for now)
        article_id = event_dict.pop("article_id", None)
        if level in ("error", "critical") and article_id:
            event_dict["article_id"] = article_id

        time_str = ts[11:19] if len(ts) >= 19 else ts
        prefix = f"{_DIM}{time_str}{_R} │ "

        msg = self._format(event, event_dict, level)
        line = prefix + msg

        if exc_info:
            indented = "\n".join(f"           {ln}" for ln in exc_info.splitlines())
            line += f"\n{indented}"

        return line

    def _format(self, event: str, ctx: dict[str, Any], level: str) -> str:  # noqa: C901 PLR0911
        match event:
            case "pipeline_started":
                sources = ctx.get("sources", [])
                model = ctx.get("model", "")
                min_score = ctx.get("min_score", "")
                header_parts = ["🚀 Pipeline"]
                if model:
                    header_parts.append(f"{_DIM}model={_R}{model}")
                if min_score != "":
                    header_parts.append(f"{_DIM}min_score={_R}{min_score}")
                header = "  ·  ".join(header_parts)
                if isinstance(sources, list) and sources:
                    rows = []
                    for src in sources:
                        host = urlparse(src.get("base_url", "")).netloc or src.get("base_url", "")
                        name = src.get("name", "")
                        max_a = src.get("max_articles", "?")
                        rows.append(f"\n           {_DIM}• {name}  {host}  max={max_a}{_R}")
                    return header + "".join(rows)
                return header

            case "scrape_started":
                n = ctx.get("count", "?")
                src_word = "джерело" if n == 1 else "джерела" if n in (2, 3, 4) else "джерел"
                return f"{_DIM}🔎 Скрапінг почався  ({n} {src_word}){_R}"

            case "scrape_source_start":
                name = ctx.get("name", "")
                url = ctx.get("url", "")
                return f"{_DIM}   ↓ {name}  {url}{_R}"

            case "scrape_source_done":
                name = ctx.get("name", "")
                found = ctx.get("found", "?")
                return f"{_DIM}   ✓ {name}  знайдено {found}{_R}"

            case "scrape_done":
                total = ctx.get("total", "?")
                return f"📰 Скрапінг завершено  всього {total} статей"

            case "filter_complete":
                new = ctx.get("new", "?")
                seen = ctx.get("seen", 0)
                seen_str = f", {seen} already seen" if seen else ""
                return f"🔍 {new} new{seen_str}"

            case "no_new_articles" | "no_articles_found":
                return f"{_DIM}💤 Nothing to do{_R}"

            case "published":
                title = ctx.get("ua_title", "")
                cat = ctx.get("category", "")
                tg_id = ctx.get("tg_message_id", "")
                score = ctx.get("score", "")
                score_str = f" score={score}" if score != "" else ""
                t = title[:65] + "…" if len(title) > 65 else title
                return f'{_GREEN}✅{_R} [{cat}]{score_str}  {_BOLD}"{t}"{_R}  → msg#{tg_id}'

            case "dry_run_would_publish":
                title = ctx.get("ua_title", "")
                cat = ctx.get("category", "")
                score = ctx.get("score", "")
                score_str = f" score={score}" if score != "" else ""
                t = title[:65] + "…" if len(title) > 65 else title
                return f'{_CYAN}🔵{_R} [DRY RUN] [{cat}]{score_str}  "{t}"'

            case "ai_processing":
                title = ctx.get("title", "")
                cat = ctx.get("category", "")
                t = title[:70] + "…" if len(title) > 70 else title
                cat_str = f"[{cat}] " if cat else ""
                return f"{_DIM}🤖 {cat_str}{t}{_R}"

            case "skipped":
                cat = ctx.get("category", "")
                score = ctx.get("score", "?")
                reason = ctx.get("reason", "")
                r = reason[:75] + "…" if len(reason) > 75 else reason
                return f"⏭  [{cat}] score={score}  {_DIM}{r}{_R}"

            case "pipeline_complete":
                posted = ctx.get("posted", 0)
                skipped = ctx.get("skipped", 0)
                errors = ctx.get("errors", 0)
                elapsed_s = ctx.get("elapsed_s")
                err_color = _RED if errors else _R
                elapsed_str = ""
                if elapsed_s is not None:
                    m, s = divmod(int(elapsed_s), 60)
                    elapsed_str = f"  ·  {_DIM}{m}m {s}s{_R}" if m else f"  ·  {_DIM}{s}s{_R}"
                return (
                    f"🏁 Done  "
                    f"{_GREEN}posted={posted}{_R}  "
                    f"skipped={skipped}  "
                    f"{err_color}errors={errors}{_R}"
                    f"{elapsed_str}"
                )

            case "gemini_unavailable_skipping":
                error = ctx.get("error", "")
                first_line = error.split("\n")[0]
                rest = "\n".join(
                    f"           {_DIM}{ln}{_R}"
                    for ln in error.split("\n")[1:]
                    if ln.strip()
                )
                detail = f"\n{rest}" if rest else ""
                return (
                    f"{_YELLOW}⚠️  Gemini 503 — skipping, retry next run{_R}"
                    f"  {_DIM}{first_line}{_R}{detail}"
                )

            case "quota_exhausted_stopping":
                p = ctx.get("processed", "?")
                r = ctx.get("remaining", "?")
                return f"{_YELLOW}⚠️  Gemini quota exhausted{_R}  processed={p}  remaining={r}"

            case "db_initialized":
                return "🗄️  Database initialized"

            case _:
                if level in ("error", "critical"):
                    error = ctx.pop("error", "")
                    details = "  ".join(f"{k}={v}" for k, v in ctx.items())
                    suffix = f"  {_DIM}{details}{_R}" if details else ""
                    err_str = f"  {error}" if error else ""
                    return f"{_RED}❌ {event}{_R}{err_str}{suffix}"

                if level == "warning":
                    details = "  ".join(f"{k}={v}" for k, v in ctx.items())
                    suffix = f"  {_DIM}{details}{_R}" if details else ""
                    return f"{_YELLOW}⚠️  {event}{_R}{suffix}"

                # Generic info fallback
                details = "  ".join(f"{k}={v}" for k, v in ctx.items())
                suffix = f"  {_DIM}{details}{_R}" if details else ""
                return f"• {event}{suffix}"


def configure_logging(app_env: str = "production") -> None:
    """Configure structlog.  Mode is auto-detected from stdout:

    - TTY  → pretty human-friendly renderer
    - Pipe → JSON renderer (GitHub Actions, log collectors)
    """
    is_tty = sys.stdout.isatty()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_tty:
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            _drop_ai_evaluated,
            structlog.processors.format_exc_info,
            _PrettyRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,  # allow reconfiguration in tests
    )

    # stdlib logging (httpx, google-genai, etc.)
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        level=logging.WARNING,
    )
