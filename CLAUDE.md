# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Rz-Flow: a scheduled batch job (not a long-running bot) that scrapes Rzeszów local news, has Gemini score/translate each article into Ukrainian, and posts the good ones to a Telegram channel. Runs on GitHub Actions cron (`publish.yml`); state lives in Turso (cloud SQLite). Python 3.12 (pinned in `.python-version`), managed with `uv` (pinned to 0.11.8 in every workflow). Dependabot opens weekly grouped update PRs against `stage`.

## Commands

```bash
uv sync --dev                                   # install
uv run pytest                                   # tests (coverage on, live tests deselected via addopts)
uv run pytest tests/test_pipeline.py::test_name --no-cov   # single test
uv run pytest -m live --no-cov                  # canary tests against the real news sites
uv run ruff check src/ tests/                   # lint (CI runs this)
uv run ruff format src/ tests/                  # CI runs --check
uv run pre-commit install                       # hooks run ruff/mypy from uv.lock
uv run mypy                                     # strict mode, configured in pyproject

uv run rz-flow --dry-run                        # evaluate with Gemini, no Telegram posts, no DB writes
uv run rz-flow --staging [--dry-run]            # staging channel + staging Turso DB
uv run rz-flow --init-db [--staging]            # create/migrate tables and exit
```

Running the CLI needs a filled `.env` (see `.env.example`); set `APP_ENV=development` locally to suppress crash/quota alerts. Coverage must stay ≥ 70% (`fail_under`). Ruff enforces `ANN` (type annotations) in `src/` but not in `tests/`.

## Constraints

Free tiers only, no paid services (see `docs/adr/0002-free-tier-only.md`). Check every change for the quota it spends, Gemini daily requests above all. Domain terms are in `GLOSSARY.md`, decisions in `docs/adr/`.

## Branching

Work happens on `stage`; changes reach `main` through PRs (`stage` → `main`). `main` is protected and a pre-commit hook (`no-commit-to-branch`) refuses commits to it. CI (`ci.yml`) runs ruff check, ruff format --check, mypy, pytest on every non-`main` push and on PRs.

## Architecture

Flow: `main.py` → `Pipeline.run()` in `pipeline.py`, which goes through these stages:

0. **Resolve stale Publishing** (non-dry Runs only): `resolve_stale_publishing` turns Publishing rows left by an earlier Run into Unconfirmed and alerts the admin once per Article.

1. **Scrape** (`scraper.py`): `fetch_articles` runs every enabled source from `config.yaml` one after another, sharing one client with browser-like headers over HTTP/2. A failing source is isolated and the others continue. `Pipeline._check_source_health` raises an admin alert when an enabled source returns 0 articles, because a markup change otherwise looks just like a quiet news day.
2. **Dedup**: `storage.filter_new_ids` returns IDs not yet settled. Articles are then reordered round-robin across sources, oldest first. `Pipeline._check_source_windows` flags (in the run report) every Source that returned its full `max_articles` window with none of it already in storage (retryable Errors count as stored; `storage.existing_ids` is read only when some window looks all new).
3. **Repeat check + evaluate + publish**, one article at a time. First `StoryIndex.find_original` (`dedup.py`) looks for an original: an Article the AI judged earlier in this Run (Published, Skipped or Publishing; never Error or Deferred) or a stored AI-judged row whose last write is within 72 hours (`storage.recent_judged_articles`, read once per Run and only when there are new Articles: Published, Skipped, Publishing, Unconfirmed) with the same normalised title (any Source) or the same slug from the same Source (`ScraperSource.slug`). A match is a Repeat: no Gemini call, no Post, saved as `REPEAT` with `repeat_of` = the original's ID, listed in the run report next to its original. Without a match, a title with token Jaccard ≥ 0.5 (`NEAR_MATCH_THRESHOLD`) against the same pool still goes to Gemini but gets a near-match hint in the run report (never stored). Otherwise `GeminiAIFilter.evaluate` (structured output → `AIDecision`), then `mark_publishing` and `TelegramPublisher.publish` to the main channel (that failure is fatal for the article). Event articles then go to the events channel with a single attempt (`publish_once`); that failure is non-fatal, shown in the run report and never retried. If `mark_publishing` fails, nothing is sent and the Run stops. Any `StopSignal` other than `CONTINUE` (post cap, Gemini quota, publishing blocked, Telegram rate limited) stops the loop through `Pipeline._stop_early`.
4. **Persist**: `save_decision` with in-process retries. If a post is published but never saved, the row stays Publishing (the next Run makes it Unconfirmed) and an admin alert says no Duplicate will follow.

After the run, `main.py` sends `stats.alerts` and a run report to `TELEGRAM_ADMIN_CHAT_ID` and writes a GitHub step summary.

### Retry/persistence semantics (the subtle part)

- **Deferred: never persisted, so the article retries next run untouched:** every subclass of `DeferralError` (`errors.py`): `GeminiQuotaExhaustedError`, `GeminiUnavailableError` (5xx after retries), `GeminiResponseIncompleteError` (truncated JSON), `TelegramRateLimitedError` (429 asking for more than 60 s, or a second 429 after one wait), `TelegramUnreachableError` (connect error/timeout, pool timeout after a few attempts). These are not the article's fault. A Deferred Article whose Publishing record was already written gets it released (`release_publishing`): storage goes back to its state before the Run, attempt count included. Each one carries its run-report icon, message and `StopSignal`; `Pipeline` handles them all in one `except`. Early stops go through `Pipeline._stop_early`, and a new stop reason needs an entry in `_REMAINING_STOP_REASONS`.
- **At-most-once posting (ADR 0001):** before the Main channel Post, `mark_publishing` records the Article as `PUBLISHING` (over an `ERROR` row only the outcome changes, attempts are kept). When delivery is uncertain (`TelegramDeliveryUnknownError`: read/write timeout, read error, protocol error, 5xx) the row stays `PUBLISHING` and the Run stops with `StopSignal.PUBLISHING_BLOCKED`. Every non-dry Run starts with `resolve_stale_publishing`, which turns `PUBLISHING` rows into `UNCONFIRMED` and alerts the admin once per Article. Both outcomes are Settled.
- **Any other exception** saves the article as `ERROR` and increments `error_attempts` (an UPSERT, so the counter accumulates). `filter_new_ids` keeps treating ERROR rows as new until `error_attempts >= max_error_attempts`.
- `--dry-run` reads from storage but never writes to it. `--staging` switches both the Telegram channel and the Turso DB, so production dedup is never touched.
- `TursoStorage` and `InMemoryStorage` both implement `StorageProtocol` and must behave the same way. Tests use `InMemoryStorage` and never touch a real DB. Schema changes go in `_COLUMN_MIGRATIONS` in `storage.py` (`(name, definition)` pairs; init reads `PRAGMA table_info` and adds only the missing columns, errors not suppressed) or `_DATA_MIGRATIONS` (idempotent `UPDATE`s, errors not suppressed). There is no migration framework.

### Configuration split

- `config.py` (`Settings`, pydantic-settings): secrets and env vars (`AI_MIN_SCORE`, `SCRAPER_TIMEOUT`, `APP_ENV`, …).
- `flow_config.py`: loads `config.yaml`, which defines the sources list (scraper class name, URL, `max_articles`, `enabled`), the Gemini model (`pipeline.gemini_model`, required; there is no env override) and pipeline pacing and caps. The same committed file is used in CI.

### Adding a news source

1. Create `sources/<site>.py` with a class that satisfies the `ScraperSource` protocol (`name`, `url`, `slug(url)`, `async fetch(client, max_articles)`), with the HTML parsing in `parser.py`.
2. Give it a unique `ID_PREFIX`. Article IDs are stored as `<prefix>/<id>` (`rz24/…`, `rzn/…`) and serve as the dedup key, so changing a prefix re-posts everything from that source.
3. Register it in `_SCRAPER_REGISTRY` in `sources/__init__.py`.
4. Add an entry to `config.yaml`.

### Gemini (`ai.py`)

Uses the response schema to get structured output (score 0–10, `ua_title`, `ua_summary`, `is_event`, `category_tag`, and the Event date as nullable `event_start`/`event_end`). The user prompt carries the Run date in Europe/Warsaw so relative dates resolve; `AIDecision` turns a bad Event date into none instead of failing. Errors are sorted into transient (exponential backoff), per-minute 429 (waits for Gemini's own `retryDelay`), daily quota (raised and stops the run) and truncated output. Publishing requires `is_interesting` and `score >= AI_MIN_SCORE`.

### Telegram (`telegram.py`)

Talks to the Bot API directly through httpx, with no bot library. Messages use HTML parse mode, so any site- or AI-sourced text must pass through `_html_escape`. Messages are capped at 4096 characters. Longer text is truncated in a way that keeps the HTML tags balanced. The same class sends channel posts, admin alerts and run reports.

## Testing notes

- HTTP is mocked with `respx`. `asyncio_mode = "auto"`, so async tests need no decorator.
- `conftest.py` has an autouse fixture that sets tenacity's wait to zero on `GeminiAIFilter.evaluate`, `TelegramPublisher._publish_reaching_telegram` and `scraper._fetch_one`. A new `@retry`-decorated function needs adding to that list, or its tests will sleep.
- Parser tests run against frozen HTML in `tests/fixtures/`. When the daily canary workflow (`canary.yml`, `-m live`) fails, the site's markup changed: update the parser and refresh the fixtures from the live page.
- User-facing strings (posts, admin alerts) are in Ukrainian; logs and code are in English. Logging goes through structlog (`logger.bind(...)`, snake_case event names), with JSON output in CI and pretty output in a TTY.

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues for `YuraRavaR/rz-news-bot`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
