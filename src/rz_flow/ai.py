"""Gemini AI wrapper: article relevance scoring + Ukrainian translation.

Uses Gemini's structured output (response_schema) to get a guaranteed
JSON response that maps directly to the AIDecision Pydantic model.
This avoids brittle regex parsing of LLM text output.

Error handling strategy — three distinct cases:
  1. Transient (network timeout, 5xx)   → retry with exponential backoff
  2. Per-minute rate limit (429 RPM)    → retry after Retry-After delay
  3. Daily quota exhausted (429 + PerDay limit=0) → raise GeminiQuotaExhaustedError
     The pipeline catches this and stops immediately — no point retrying
     until tomorrow when the quota resets.
"""

import asyncio
import json
import re

import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai.errors import ServerError as GeminiServerError
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rz_flow.models import AIDecision, Article, CategoryTag

logger = structlog.get_logger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────
class GeminiQuotaExhaustedError(Exception):
    """Daily/project quota for Gemini API is fully exhausted.

    This is NOT retryable during the current run — quota resets at midnight UTC.
    The pipeline should stop processing remaining articles to avoid noise in logs.
    """

    pass


class GeminiRateLimitError(Exception):
    """Per-minute (RPM) rate limit hit — retrying after a delay will work."""

    def __init__(self, message: str, retry_after: float = 30.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after(error: genai_errors.ClientError) -> float:
    """Extract retryDelay seconds from a Gemini 429 error response."""
    try:
        details = error.details or {}
        error_body = details.get("error", {})
        for detail in error_body.get("details", []):
            if "retryDelay" in detail:
                delay_str = detail["retryDelay"]  # e.g. "27s" or "27.5s"
                match = re.search(r"(\d+(?:\.\d+)?)", delay_str)
                if match:
                    return float(match.group(1))
    except Exception:
        pass
    return 30.0  # safe default


def _is_daily_quota_exhausted(error: genai_errors.ClientError) -> bool:
    """Return True if the daily project quota is at 0 (not just per-minute)."""
    try:
        details = error.details or {}
        error_body = details.get("error", {})
        for detail in error_body.get("details", []):
            violations = detail.get("violations", [])
            for v in violations:
                if "PerDay" in v.get("quotaId", ""):
                    return True
    except Exception:
        pass
    return False


def _classify_gemini_error(exc: Exception) -> Exception:
    """Convert a raw Gemini ClientError into a domain-specific exception."""
    if not isinstance(exc, genai_errors.ClientError):
        return exc
    if exc.code != 429:
        return exc

    if _is_daily_quota_exhausted(exc):
        return GeminiQuotaExhaustedError(
            "Gemini daily quota exhausted — will reset at midnight UTC. "
            "Stopping pipeline to avoid unnecessary retries."
        )

    retry_after = _parse_retry_after(exc)
    return GeminiRateLimitError(str(exc), retry_after=retry_after)

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
Ти — редактор Telegram-каналу "Rzeszów для своїх"
для україномовних мешканців міста Жешув (Польща).

Твоє завдання: переглянути новину з польського міського порталу
та вирішити, чи варто публікувати її в каналі.

КРИТЕРІЇ ПУБЛІКАЦІЇ (score 7–10):
✅ Публічні події: концерти, фестивалі, ярмарки, спортивні заходи, виставки
✅ Важлива міська інфраструктура: нові об'єкти, ремонти доріг/комунікацій
✅ Корисна інформація для мешканців: транспорт, школи, лікарні, послуги
✅ Цікаве місцеве культурне / спортивне життя

НЕ ПУБЛІКУВАТИ (score 0–4):
❌ Кримінальні новини, ДТП, вбивства
❌ Некрологи та трагічні події
❌ Рекламні матеріали та спонсорський контент
❌ Скандали та чутки
❌ Загальнопольська або міжнародна політика (без зв'язку з Жешовом)
❌ Новини про людей, нерелевантні широкій аудиторії

НЕЙТРАЛЬНИЙ КОНТЕНТ (score 5–6): якщо сумніваєшся.

Відповідай ЛИШЕ валідним JSON згідно із заданою схемою. Без пояснень поза JSON."""

_USER_PROMPT_TEMPLATE = """Категорія: {category}
Заголовок (польською): {title}
Лід-абзац (польською): {summary}

Оціни та переклади українською."""


def _build_response_schema() -> dict[str, object]:
    """Build the JSON Schema that Gemini will enforce for its response."""
    return {
        "type": "object",
        "properties": {
            "is_interesting": {
                "type": "boolean",
                "description": "True if score >= 7 and article should be published",
            },
            "score": {
                "type": "number",
                "description": "Relevance score 0–10 for Rzeszów Ukrainian community",
            },
            "category_tag": {
                "type": "string",
                "enum": [tag.value for tag in CategoryTag],
                "description": "Content category tag",
            },
            "ua_title": {
                "type": "string",
                "description": "Ukrainian translation of the title (concise, engaging)",
            },
            "ua_summary": {
                "type": "string",
                "description": "2–3 sentence Ukrainian summary for Telegram post",
            },
            "reason": {
                "type": "string",
                "description": "One-sentence explanation of the decision (for logs)",
            },
        },
        "required": ["is_interesting", "score", "category_tag", "ua_title", "ua_summary", "reason"],
    }


class GeminiAIFilter:
    """Wraps the Gemini API client with retry and structured output."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._schema = _build_response_schema()

    @retry(
        # Retry per-minute rate limits AND server-side 5xx (e.g. 503 high demand).
        # GeminiQuotaExhaustedError is NOT retried — it propagates up immediately.
        retry=retry_if_exception_type((GeminiRateLimitError, GeminiServerError)),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def evaluate(self, article: Article) -> AIDecision:
        """Send article to Gemini for evaluation and translation.

        Raises:
            GeminiQuotaExhaustedError: daily quota is 0 — stop the pipeline.
            GeminiRateLimitError: per-minute limit hit — retried automatically.
            ValueError: empty or unparseable Gemini response.
        """
        user_content = _USER_PROMPT_TEMPLATE.format(
            category=article.category.value,
            title=article.title_pl,
            summary=article.summary_pl or "(немає)",
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=self._schema,
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )
        except genai_errors.ClientError as exc:
            classified = _classify_gemini_error(exc)
            if isinstance(classified, GeminiRateLimitError):
                logger.warning(
                    "gemini_rate_limited",
                    article_id=article.id,
                    retry_after=classified.retry_after,
                )
                # Sleep for the exact delay Gemini asked for before tenacity retries
                await asyncio.sleep(classified.retry_after)
                raise classified
            if isinstance(classified, GeminiQuotaExhaustedError):
                logger.error("gemini_quota_exhausted", article_id=article.id)
                raise classified
            raise  # other 4xx/5xx — re-raise original

        raw_text = response.text
        if not raw_text:
            raise ValueError(f"Empty Gemini response for article {article.id}")

        data = json.loads(raw_text)
        return AIDecision(**data)
