# Rz-Flow

Автоматизований Telegram-канал з подіями та новинами міста (Польща).

Джерело: [rzeszow24.info](https://rzeszow24.info)  
Публікується: українською мовою  
Частота: кожні 3 години (GitHub Actions cron)

## Як це працює

```
rzeszow24.info → scraper → parser → Turso (фільтр дублів)
    → Gemini AI (оцінка + переклад) → Telegram-канал
```

1. Скрапер завантажує категорії `/imprezy/` (події) та `/wiadomosci/` (новини)
2. Парсер витягує статті у типізовані моделі
3. Turso фільтрує вже оброблені статті
4. Gemini 2.0 Flash оцінює цікавість (score 0–10) і перекладає українською
5. Статті з `score >= 7` публікуються у Telegram-канал

## Структура проєкту

```
src/rz_flow/
├── config.py      — налаштування через env-змінні (pydantic-settings)
├── models.py      — типи даних: Article, AIDecision
├── scraper.py     — async HTTP-запити до rzeszow24.info
├── parser.py      — HTML → Article[]
├── ai.py          — Gemini wrapper (structured output)
├── storage.py     — Turso клієнт + InMemoryStorage для тестів
├── telegram.py    — Bot API publisher
├── pipeline.py    — оркестратор пайплайну
└── main.py        — точка входу
```

## Локальний запуск

### 1. Встановити uv (якщо немає)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Встановити залежності

```bash
uv sync --dev
```

### 3. Налаштувати секрети

```bash
cp .env.example .env
# Відкрий .env та заповни всі змінні (інструкція всередині)
```

Що потрібно отримати:

| Змінна | Де взяти |
|--------|----------|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) → `/newbot` |
| `TELEGRAM_CHANNEL_ID` | [@userinfobot](https://t.me/userinfobot) або `/getUpdates` |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `TURSO_DATABASE_URL` | `turso db show rz-flow --url` |
| `TURSO_AUTH_TOKEN` | `turso db tokens create rz-flow` |

### 4. Ініціалізувати БД

```bash
uv run rz-flow --init-db
```

### 5. Тестовий запуск (без публікації в канал)

```bash
uv run rz-flow --dry-run
```

### 6. Реальний запуск

```bash
uv run rz-flow
```

## Тести

```bash
# Всі тести
uv run pytest

# Тільки швидкі (без моків HTTP)
uv run pytest -m "not integration"

# З виводом логів
uv run pytest -s
```

## Linting / типізація

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/rz_flow/
```

## CI/CD (GitHub Actions)

Пайплайн запускається автоматично **кожні 3 години** і при ручному тригері.

Необхідні GitHub Secrets (Settings → Secrets and variables → Actions):

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHANNEL_ID`
- `GEMINI_API_KEY`
- `TURSO_DATABASE_URL`
- `TURSO_AUTH_TOKEN`

## Налаштування

| Змінна | За замовчуванням | Опис |
|--------|-----------------|------|
| `AI_MIN_SCORE` | `7` | Мінімальний score Gemini для публікації |
| `APP_ENV` | `production` | Оточення (`development` вимикає реальну публікацію) |

## Roadmap

- [ ] Зображення з оригіналу (`sendPhoto`)
- [ ] Щоденний дайджест топ-5 подій
- [ ] Адмін-команди: `/threshold`, `/pause`, `/stats`
- [ ] Дашборд на Streamlit
- [ ] Семантична дедуплікація через Gemini Embeddings
- [ ] Підтримка кількох джерел
