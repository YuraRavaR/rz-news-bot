# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Rz-Flow: a scheduled batch job (not a long-running bot) that scrapes Rzeszów local news, has Gemini score/translate each article into Ukrainian, and posts the good ones to a Telegram channel. Runs on GitHub Actions cron (`publish.yml`); state lives in Turso (cloud SQLite). Python 3.12, managed with `uv`.

## Commands

```bash
uv sync --dev                                   # install
uv run pytest                                   # tests (coverage on, live tests deselected via addopts)
uv run pytest tests/test_pipeline.py::test_name --no-cov   # single test
uv run pytest -m live --no-cov                  # canary tests against the real news sites
uv run ruff check src/ tests/                   # lint (CI runs this)
uv run ruff format src/ tests/
uv run mypy                                     # strict mode, configured in pyproject

uv run rz-flow --dry-run                        # evaluate with Gemini, no Telegram posts, no DB writes
uv run rz-flow --staging [--dry-run]            # staging channel + staging Turso DB
uv run rz-flow --init-db [--staging]            # create/migrate tables and exit
```

Running the CLI needs a filled `.env` (see `.env.example`); set `APP_ENV=development` locally to suppress crash/quota alerts. Coverage must stay ≥ 70% (`fail_under`). Ruff enforces `ANN` (type annotations) in `src/` but not in `tests/`.

## Constraints

Free tiers only, no paid services (see `docs/adr/0002-free-tier-only.md`). Check every change for the quota it spends, Gemini daily requests above all. Domain terms are in `GLOSSARY.md`, decisions in `docs/adr/`.

## Branching

Work happens on `stage`; changes reach `main` through PRs (`stage` → `main`). `main` is protected and a pre-commit hook (`no-commit-to-branch`) refuses commits to it. CI (`ci.yml`) runs ruff, mypy, pytest on every non-`main` push and on PRs.

## Architecture

Flow: `main.py` → `Pipeline.run()` in `pipeline.py`, which goes through these stages:

1. **Scrape** (`scraper.py`): `fetch_articles` runs every enabled source from `config.yaml` one after another, sharing one client with browser-like headers over HTTP/2. A failing source is isolated and the others continue. `Pipeline._check_source_health` raises an admin alert when an enabled source returns 0 articles, because a markup change otherwise looks just like a quiet news day.
2. **Dedup**: `storage.filter_new_ids` returns IDs not yet settled. Articles are then reordered round-robin across sources, oldest first.
3. **Evaluate + publish**, one article at a time: `GeminiAIFilter.evaluate` (structured output → `AIDecision`), then `TelegramPublisher.publish`. Event articles go to the events channel first (that failure is non-fatal), then to the main channel (that failure is fatal for the article). The loop stops early on `StopSignal.CAP_REACHED` (`max_posts_per_run`) or `StopSignal.QUOTA_EXHAUSTED`.
4. **Persist**: `save_decision` with in-process retries. If a post is published but never saved, an admin alert warns that a duplicate post is likely.

After the run, `main.py` sends `stats.alerts` and a run report to `TELEGRAM_ADMIN_CHAT_ID` and writes a GitHub step summary.

### Retry/persistence semantics (the subtle part)

- **Never persisted, so the article retries next run untouched:** `GeminiQuotaExhaustedError`, `GeminiServerError` (503), `GeminiResponseIncompleteError` (truncated JSON). These are not the article's fault.
- **Any other exception** saves the article as `ERROR` and increments `error_attempts` (an UPSERT, so the counter accumulates). `filter_new_ids` keeps treating ERROR rows as new until `error_attempts >= max_error_attempts`.
- `--dry-run` reads from storage but never writes to it. `--staging` switches both the Telegram channel and the Turso DB, so production dedup is never touched.
- `TursoStorage` and `InMemoryStorage` both implement `StorageProtocol` and must behave the same way. Tests use `InMemoryStorage` and never touch a real DB. Schema changes go in `_SCHEMA_MIGRATIONS` in `storage.py` (`ALTER TABLE`, duplicate-column errors suppressed) or `_DATA_MIGRATIONS` (idempotent `UPDATE`s, errors not suppressed). There is no migration framework.

### Configuration split

- `config.py` (`Settings`, pydantic-settings): secrets and env vars (`AI_MIN_SCORE`, `SCRAPER_TIMEOUT`, `APP_ENV`, …). `scraper_base_url` and `scraper_max_articles` are legacy fields that nothing reads.
- `flow_config.py`: loads `config.yaml`, which defines the sources list (scraper class name, URL, `max_articles`, `enabled`) and pipeline pacing and caps. The same committed file is used in CI.

### Adding a news source

1. Create `sources/<site>.py` with a class that satisfies the `ScraperSource` protocol (`name`, `url`, `async fetch(client, max_articles)`), with the HTML parsing in `parser.py`.
2. Give it a unique `ID_PREFIX`. Article IDs are stored as `<prefix>/<id>` (`rz24/…`, `rzn/…`) and serve as the dedup key, so changing a prefix re-posts everything from that source.
3. Register it in `_SCRAPER_REGISTRY` in `sources/__init__.py`. The docstrings that mention `ACTIVE_SOURCES` are stale.
4. Add an entry to `config.yaml`.

`CategoryPageScraper` and `parse_category_page` are archived (not registered) but still have tests and fixtures.

### Gemini (`ai.py`)

Uses the response schema to get structured output (score 0–10, `ua_title`, `ua_summary`, `is_event`, `category_tag`). Errors are sorted into transient (exponential backoff), per-minute 429 (waits for Gemini's own `retryDelay`), daily quota (raised and stops the run) and truncated output. Publishing requires `is_interesting` and `score >= AI_MIN_SCORE`.

### Telegram (`telegram.py`)

Talks to the Bot API directly through httpx, with no bot library. Messages use HTML parse mode, so any site- or AI-sourced text must pass through `_html_escape`. Messages are capped at 4096 characters. Longer text is truncated in a way that keeps the HTML tags balanced. The same class sends channel posts, admin alerts and run reports.

## Testing notes

- HTTP is mocked with `respx`. `asyncio_mode = "auto"`, so async tests need no decorator.
- `conftest.py` has an autouse fixture that sets tenacity's wait to zero on `GeminiAIFilter.evaluate`, `TelegramPublisher.publish` and `scraper._fetch_one`. A new `@retry`-decorated function needs adding to that list, or its tests will sleep.
- Parser tests run against frozen HTML in `tests/fixtures/`. When the daily canary workflow (`canary.yml`, `-m live`) fails, the site's markup changed: update the parser and refresh the fixtures from the live page.
- User-facing strings (posts, admin alerts) are in Ukrainian; logs and code are in English. Logging goes through structlog (`logger.bind(...)`, snake_case event names), with JSON output in CI and pretty output in a TTY.

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues for `YuraRavaR/rz-news-bot`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
