# Rz-Flow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pipeline that scrapes local news, scores each article with Gemini AI for relevance, and publishes only the most useful ones to a Telegram channel — translated into Ukrainian.

**Sources:** [rzeszow24.info](https://rzeszow24.info) · [rzeszow-news.pl](https://rzeszow-news.pl)
**Published in:** Ukrainian
**Schedule:** every 3 hours, 06:00–21:00 CET (GitHub Actions cron)

> **Content disclaimer:** This project aggregates publicly available news headlines and URLs from third-party websites. All original content belongs to the respective publishers. This bot does not reproduce full article text — it links back to the original source. Provided for educational and informational purposes only.

---

## How It Works

```
rzeszow24.info  ─┐
                  ├─► scraper ──► parser ──► Turso (dedup) ──► Gemini AI ──► Telegram
rzeszow-news.pl ─┘
```

1. **Scraper** fetches the latest articles from both news sources using browser-like headers over HTTP/2 — with automatic retry on timeouts
2. **Parser** extracts structured `Article` objects from raw HTML
3. **Turso** (SQLite in the cloud) filters out articles already seen in previous runs
4. **Gemini AI** scores each article (0–10) for relevance to Rzeszów residents, translates the title and summary into Ukrainian, flags whether it's a specific upcoming **event**, and assigns a category tag (concert / festival / sport / transport / other)
5. Articles scoring **≥ 7** are published to the Telegram channel — with a hashtag derived from the category tag and a link back to the original source (label follows the actual source domain); the rest are saved as `skipped`
6. Articles flagged as an **event** are additionally cross-posted to a dedicated events channel, if `TELEGRAM_EVENTS_CHANNEL_ID` is configured (failure to post there is non-fatal — the main-channel post still goes out)
7. Publishing stops early once `max_posts_per_run` is reached, so a backlog never floods the channel in one run — the rest is retried on the next run
8. All results (posted / skipped / error) are persisted so articles are never processed twice
9. After every run, a structured admin report (elapsed time, per-source counts, per-article log) is sent to `TELEGRAM_ADMIN_CHAT_ID` if configured

If a source times out, the pipeline continues with the remaining sources. Gemini quota exhaustion is handled gracefully — unprocessed articles are retried on the next run.

---

## Project Structure

```
src/rz_flow/
├── config.py          — environment-based settings (pydantic-settings)
├── flow_config.py     — sources + pipeline options loaded from config.yaml
├── models.py          — typed data models: Article, AIDecision, PostRecord
├── sources/           — pluggable scrapers (NajnowszeScraper, RzeszowNewsScraper)
├── scraper.py         — async HTTP orchestrator with per-source error isolation
├── parser.py          — HTML → Article[] (BeautifulSoup + lxml)
├── ai.py              — Gemini structured-output wrapper with retry & quota handling
├── storage.py         — TursoStorage (production) + InMemoryStorage (tests)
├── telegram.py        — Telegram Bot API publisher
├── pipeline.py        — pipeline orchestrator: scrape → filter → AI → publish → save
├── digest.py          — the weekend Digest for the Events channel (--digest)
├── weekly_report.py   — last week's report to the admin chat (--weekly-report)
├── logging_config.py  — pretty TTY renderer locally, JSON for CI
└── main.py            — CLI entry point (--dry-run, --staging, --init-db, --digest, --weekly-report)
```

---

## Local Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- A [Turso](https://turso.tech) account (free tier is enough)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com/apikey)

### 1. Install dependencies

```bash
uv sync --dev
```

### 2. Configure secrets

```bash
cp .env.example .env
# Fill in all variables — instructions inside the file
```

| Variable | Where to get it |
|---|---|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) → `/newbot` |
| `TELEGRAM_CHANNEL_ID` | [@userinfobot](https://t.me/userinfobot) or `/getUpdates` |
| `TELEGRAM_ADMIN_CHAT_ID` | Your personal chat ID (optional — enables per-run admin reports) |
| `TELEGRAM_EVENTS_CHANNEL_ID` | (Optional) Dedicated channel for event articles (concerts, festivals, etc.) — when set, `is_event` articles are cross-posted here **in addition to** the main channel |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `TURSO_DATABASE_URL` | `turso db show rz-flow --url` |
| `TURSO_AUTH_TOKEN` | `turso db tokens create rz-flow` |
| `TELEGRAM_STAGING_CHANNEL_ID` | (Optional) Second channel for `uv run rz-flow --staging` |
| `TELEGRAM_STAGING_EVENTS_CHANNEL_ID` | (Optional) Staging variant of `TELEGRAM_EVENTS_CHANNEL_ID`; when unset, staging posts event articles to the staging channel only (never to the production events channel) |
| `TURSO_STAGING_DATABASE_URL` | (Optional) Separate DB URL — **required** with staging flag |
| `TURSO_STAGING_AUTH_TOKEN` | (Optional) Token for staging DB |

### 3. Create the database

```bash
turso db create rz-flow        # one-time
uv run rz-flow --init-db
```

For a **staging** database (used with `--staging` only):

```bash
turso db create rz-flow-staging
uv run rz-flow --init-db --staging
```

### 4. Dry run (no Telegram publishing)

```bash
uv run rz-flow --dry-run
```

### 5. Staging channel (preview posts without touching production dedup)

Requires `TELEGRAM_STAGING_CHANNEL_ID`, `TURSO_STAGING_DATABASE_URL`, and `TURSO_STAGING_AUTH_TOKEN` in `.env`.

```bash
uv run rz-flow --staging
```

**Staging + dry run:** same staging Turso for reads (which articles are “new”), but nothing is written to the DB and nothing is posted to Telegram — good for a safe rehearsal:

```bash
uv run rz-flow --staging --dry-run
```

### 6. Production run

```bash
uv run rz-flow
```

### 7. Weekend Digest

Posts one Digest of the coming weekend's (Friday–Sunday, Europe/Warsaw) Published events with an Event date to the Events channel, at most once per weekend, and exits. With fewer than three events nothing is posted and the admin gets an alert. `--dry-run` sends the text to the admin chat and writes nothing; `--staging` uses the staging DB and staging Events channel.

```bash
uv run rz-flow --digest --dry-run
uv run rz-flow --digest
```

### 8. Weekly report

Sends a report about the previous Monday–Sunday (Europe/Warsaw) to the admin chat and exits: per Source, how many Articles were new, Published, Skipped, Repeats, Errors and Unconfirmed; events posted and how many had no Event date; an estimate of Gemini requests on the busiest day against the free-tier 500; whether that week's Digest went out; and a ⚠️ when Published fell by more than half against the week before. It only reads storage (no Gemini, no scraping). `--dry-run` adds a banner; `--staging` reads the staging DB.

```bash
uv run rz-flow --weekly-report --dry-run
uv run rz-flow --weekly-report
```

---

## Configuration

Edit `config.yaml` to control sources, pipeline pacing, caps, and (optionally) how the admin Telegram run report shows the clock:

```yaml
sources:
  - scraper: NajnowszeScraper
    base_url: https://rzeszow24.info/najnowsze
    max_articles: 20
    enabled: true

  - scraper: RzeszowNewsScraper
    base_url: https://rzeszow-news.pl
    max_articles: 15
    enabled: true

pipeline:
  inter_ai_delay_seconds: 5.0     # pause between Gemini calls (free tier: 15 RPM)
  inter_post_delay_seconds: 2.0   # pause after each Telegram post
  max_posts_per_run: 5            # cap successful posts per run
  max_error_attempts: 3           # retries across runs before an article is dropped
  report_display_timezone: Europe/Warsaw   # optional IANA name; omit or null → UTC in report header
```

`max_error_attempts` guards against a transient failure (network blip, malformed AI response) permanently retiring an article: a failed article stays in the queue and is retried on later runs until the budget is spent. Set it to `1` to never retry.

`report_display_timezone` must be a valid IANA zone if set (same validation as at startup). GitHub Actions uses the same committed file, so CI and local runs stay aligned without extra env vars.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `AI_MIN_SCORE` | `7` | Minimum Gemini score (0–10) to publish |
| `SCRAPER_TIMEOUT` | `15` | HTTP timeout per source (seconds) |
| `APP_ENV` | `production` | Set to `development` to suppress crash/quota Telegram alerts (used for local runs); use CLI `--dry-run` to skip real publishing |

---

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/rz_flow/
```

Install the pre-commit hooks once so lint, formatting and type errors are caught before they reach CI. The hooks run ruff and mypy through `uv run`, so they use the versions pinned in `uv.lock`, and `.python-version` pins Python 3.12 to match CI:

```bash
uv run pre-commit install
```

### Canary tests

Parser tests run against HTML fixtures, which stay green even after a source site changes its markup. The `live` tests hit the real sites to catch that drift. They are excluded from `pytest` by default and run on a daily schedule:

```bash
uv run pytest -m live --no-cov
```

A failure means the parser needs updating — refresh the fixtures in `tests/fixtures/` as part of the fix.

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | push (non-`main`), PR | ruff check, ruff format --check, mypy, pytest on Python 3.12 |
| `publish.yml` | schedule + manual | The actual bot run |
| `canary.yml` | daily + manual | Live scrape of both sources |

`publish.yml` can be triggered manually with optional `dry_run` and `staging` flags (`workflow_dispatch` inputs).

**Required GitHub Secrets** (`Settings → Secrets and variables → Actions`):

| Secret | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token |
| `TELEGRAM_CHANNEL_ID` | Target channel ID |
| `TELEGRAM_ADMIN_CHAT_ID` | Admin chat ID for run reports (optional) |
| `TELEGRAM_EVENTS_CHANNEL_ID` | Dedicated events channel ID (optional) |
| `GEMINI_API_KEY` | Google AI Studio key |
| `TURSO_DATABASE_URL` | Turso database URL |
| `TURSO_AUTH_TOKEN` | Turso auth token |
| `TELEGRAM_STAGING_CHANNEL_ID` | Staging channel ID — required only for the manual `staging: true` dispatch input |
| `TELEGRAM_STAGING_EVENTS_CHANNEL_ID` | Staging events channel ID (optional) |
| `TURSO_STAGING_DATABASE_URL` | Staging Turso DB URL — required only for `staging: true` |
| `TURSO_STAGING_AUTH_TOKEN` | Staging Turso auth token — required only for `staging: true` |
| `HEALTHCHECK_URL` | healthchecks.io ping URL (optional) — dead-man's switch for production Runs, see below |

**Dead-man's switch.** When `HEALTHCHECK_URL` is set, production Runs that are not dry runs ping healthchecks.io: `/start` first, the bare URL on success, `/fail` when any step fails. Staging and dry-run Runs never ping. A failed ping never fails the Run. Set the check up once by hand: cron schedule identical to `publish.yml` (`23 4,7,10,13,16,19 * * *`), timezone UTC, grace period 3 h, and the Telegram integration pointed at the admin chat (email if the free plan has no Telegram). healthchecks.io then alerts when a scheduled Run does not happen, hangs, or fails, including failures before the Python code starts.

---
