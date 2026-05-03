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
4. **Gemini AI** scores each article (0–10) for relevance to Rzeszów residents, translates the title and body into Ukrainian
5. Articles scoring **≥ 7** are published to the Telegram channel; the rest are saved as `skipped`
6. All results (posted / skipped / error) are persisted so articles are never processed twice

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
├── logging_config.py  — pretty TTY renderer locally, JSON for CI
└── main.py            — CLI entry point (--dry-run, --init-db)
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
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | e.g. `gemini-2.0-flash` or `gemini-2.0-flash-lite` |
| `TURSO_DATABASE_URL` | `turso db show rz-flow --url` |
| `TURSO_AUTH_TOKEN` | `turso db tokens create rz-flow` |

### 3. Create the database

```bash
turso db create rz-flow        # one-time
uv run rz-flow --init-db
```

### 4. Dry run (no Telegram publishing)

```bash
uv run rz-flow --dry-run
```

### 5. Production run

```bash
uv run rz-flow
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
  report_display_timezone: Europe/Warsaw   # optional IANA name; omit or null → UTC in report header
```

`report_display_timezone` must be a valid IANA zone if set (same validation as at startup). GitHub Actions uses the same committed file, so CI and local runs stay aligned without extra env vars.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `AI_MIN_SCORE` | `7` | Minimum Gemini score (0–10) to publish |
| `SCRAPER_TIMEOUT` | `15` | HTTP timeout per source (seconds) |
| `APP_ENV` | `production` | Set to `development` to disable real publishing |

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

---

## CI/CD (GitHub Actions)

The workflow (`.github/workflows/publish.yml`) runs automatically on the schedule and can be triggered manually with an optional `dry_run` flag.

**Required GitHub Secrets** (`Settings → Secrets and variables → Actions`):

| Secret | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token |
| `TELEGRAM_CHANNEL_ID` | Target channel ID |
| `TELEGRAM_ADMIN_CHAT_ID` | Admin chat ID for run reports (optional) |
| `GEMINI_API_KEY` | Google AI Studio key |
| `GEMINI_MODEL` | Model name (optional, falls back to `gemini-2.0-flash`) |
| `TURSO_DATABASE_URL` | Turso database URL |
| `TURSO_AUTH_TOKEN` | Turso auth token |

---

