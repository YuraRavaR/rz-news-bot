"""Tests for the Gemini AI filter.

We NEVER call the real Gemini API in tests — instead we use unittest.mock
to patch the client. This ensures:
  - Tests run offline (zero cost, zero rate limits)
  - We test our code (prompt building, response parsing, retry logic)
    rather than testing Google's API
  - Tests are fast and deterministic
"""

import json
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import RetryCallState

from rz_flow.ai import (
    GeminiAIFilter,
    GeminiQuotaExhaustedError,
    GeminiRateLimitError,
    GeminiResponseIncompleteError,
    _build_response_schema,
    _build_system_prompt,
    _classify_gemini_error,
    _gemini_retry_wait,
    _is_daily_quota_exhausted,
    _is_truncated,
    _parse_retry_after,
)
from rz_flow.models import Article, Category, CategoryTag


def _make_article(
    article_id: str = "TEST_ID_12345678",
    category: Category = Category.IMPREZY,
) -> Article:
    return Article(
        id=article_id,
        url=f"https://rzeszow24.info/imprezy/test/{article_id}",
        category=category,
        title_pl="Festiwal Muzyczny w Rzeszowie 2026",
        summary_pl="W centrum Rzeszowa odbędzie się festiwal muzyczny.",
    )


def _make_gemini_response(payload: dict[str, object]) -> MagicMock:
    """Build a mock Gemini response with .text set to JSON string."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    return mock_response


VALID_AI_RESPONSE = {
    "is_interesting": True,
    "is_event": True,
    "score": 8.5,
    "category_tag": "festyn",
    "ua_title": "Музичний Фестиваль у Жешові 2026",
    "ua_summary": "У центрі міста відбудеться великий музичний фестиваль.",
    "reason": "Popular public event relevant to all Rzeszów residents",
}

SKIP_AI_RESPONSE = {
    "is_interesting": False,
    "is_event": False,
    "score": 2.0,
    "category_tag": "inne",
    "ua_title": "Злочинець затриманий поліцією",
    "ua_summary": "Поліція затримала підозрюваного.",
    "reason": "Criminal news, not relevant for the channel",
}


class TestGeminiAIFilterEvaluate:
    @patch("rz_flow.ai.genai.Client")
    async def test_returns_ai_decision_for_interesting_article(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_response(VALID_AI_RESPONSE)
        )

        ai = GeminiAIFilter(api_key="fake-key")
        decision = await ai.evaluate(_make_article())

        assert decision.is_interesting is True
        assert decision.score == 8.5
        assert decision.category_tag == CategoryTag.FESTIVAL
        assert "Музичний" in decision.ua_title
        assert decision.ua_summary != ""
        assert decision.reason != ""

    @patch("rz_flow.ai.genai.Client")
    async def test_returns_skip_decision_for_crime_news(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_response(SKIP_AI_RESPONSE)
        )

        ai = GeminiAIFilter(api_key="fake-key")
        article = _make_article()
        article = article.model_copy(update={"title_pl": "Napad na bank w Rzeszowie"})
        decision = await ai.evaluate(article)

        assert decision.is_interesting is False
        assert decision.score < 7

    @patch("rz_flow.ai.genai.Client")
    async def test_raises_on_empty_gemini_response(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        empty_response = MagicMock()
        empty_response.text = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=empty_response)

        ai = GeminiAIFilter(api_key="fake-key")

        with pytest.raises(ValueError, match="Empty Gemini response"):
            await ai.evaluate(_make_article())

    @patch("rz_flow.ai.genai.Client")
    async def test_raises_on_invalid_json(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        bad_response = MagicMock()
        bad_response.text = "not json at all"
        mock_client.aio.models.generate_content = AsyncMock(return_value=bad_response)

        ai = GeminiAIFilter(api_key="fake-key")
        with pytest.raises(json.JSONDecodeError):
            await ai.evaluate(_make_article())

    @patch("rz_flow.ai.genai.Client")
    async def test_passes_article_content_to_gemini(self, mock_client_cls: MagicMock) -> None:
        """Verify that article title and summary are included in the prompt."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_response(VALID_AI_RESPONSE)
        )

        ai = GeminiAIFilter(api_key="fake-key")
        article = _make_article()
        await ai.evaluate(article)

        call_kwargs = mock_client.aio.models.generate_content.call_args
        # The contents argument should include the article title
        contents_arg = call_kwargs.kwargs.get("contents") or call_kwargs.args[1]
        assert article.title_pl in contents_arg

    @patch("rz_flow.ai.genai.Client")
    async def test_retries_on_per_minute_rate_limit(
        self, mock_client_cls: MagicMock
    ) -> None:
        """GeminiRateLimitError (per-minute 429) is retried automatically."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Build a fake 429 ClientError that looks like a per-minute limit (no PerDay)
        fake_429 = genai_errors.ClientError(
            429,
            {"error": {"code": 429, "details": [{"retryDelay": "10s"}]}},
        )
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[fake_429, _make_gemini_response(VALID_AI_RESPONSE)]
        )

        ai = GeminiAIFilter(api_key="fake-key")
        decision = await ai.evaluate(_make_article())

        assert decision.is_interesting is True
        # Should have been called twice (first fail → retry → success)
        assert mock_client.aio.models.generate_content.call_count == 2

    @patch("rz_flow.ai.genai.Client")
    async def test_raises_quota_exhausted_for_daily_limit(
        self, mock_client_cls: MagicMock
    ) -> None:
        """Daily quota exhaustion raises GeminiQuotaExhaustedError — NOT retried."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        fake_daily_429 = genai_errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "details": [
                        {
                            "violations": [
                                {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                            ]
                        }
                    ],
                }
            },
        )
        mock_client.aio.models.generate_content = AsyncMock(side_effect=fake_daily_429)

        ai = GeminiAIFilter(api_key="fake-key")
        with pytest.raises(GeminiQuotaExhaustedError):
            await ai.evaluate(_make_article())

        # Must NOT retry — only one API call attempt
        assert mock_client.aio.models.generate_content.call_count == 1

    @patch("rz_flow.ai.genai.Client")
    async def test_raises_after_max_retries_on_rate_limit(
        self, mock_client_cls: MagicMock
    ) -> None:
        """After 3 per-minute rate limit attempts, GeminiRateLimitError is re-raised."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        fake_429 = genai_errors.ClientError(
            429,
            {"error": {"code": 429, "details": [{"retryDelay": "5s"}]}},
        )
        mock_client.aio.models.generate_content = AsyncMock(side_effect=fake_429)

        ai = GeminiAIFilter(api_key="fake-key")
        with pytest.raises(GeminiRateLimitError):
            await ai.evaluate(_make_article())

        assert mock_client.aio.models.generate_content.call_count == 3

    @patch("rz_flow.ai.genai.Client")
    async def test_reraises_non_rate_limit_client_error(
        self, mock_client_cls: MagicMock
    ) -> None:
        """4xx errors other than 429 (e.g. 400 invalid request) are re-raised as-is."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        fake_400 = genai_errors.ClientError(
            400, {"error": {"code": 400, "message": "INVALID_ARGUMENT"}}
        )
        mock_client.aio.models.generate_content = AsyncMock(side_effect=fake_400)

        ai = GeminiAIFilter(api_key="fake-key")
        with pytest.raises(genai_errors.ClientError):
            await ai.evaluate(_make_article())

        assert mock_client.aio.models.generate_content.call_count == 1


class TestGeminiHelpers:
    """Unit tests for module-level helper functions (no AI calls needed)."""

    # ── _parse_retry_after ────────────────────────────────────────────────────

    def test_parse_retry_after_extracts_integer_seconds(self) -> None:
        err = genai_errors.ClientError(
            429, {"error": {"code": 429, "details": [{"retryDelay": "27s"}]}}
        )
        assert _parse_retry_after(err) == 27.0

    def test_parse_retry_after_handles_decimal_seconds(self) -> None:
        err = genai_errors.ClientError(
            429, {"error": {"code": 429, "details": [{"retryDelay": "10.5s"}]}}
        )
        assert _parse_retry_after(err) == 10.5

    def test_parse_retry_after_returns_default_when_no_retry_delay(self) -> None:
        """When retryDelay is absent in the details, default 30s is used."""
        err = genai_errors.ClientError(
            429, {"error": {"code": 429, "details": [{"quotaId": "some-quota"}]}}
        )
        assert _parse_retry_after(err) == 30.0

    def test_parse_retry_after_returns_default_on_empty_details(self) -> None:
        err = genai_errors.ClientError(429, {"error": {"code": 429, "details": []}})
        assert _parse_retry_after(err) == 30.0

    # ── _is_daily_quota_exhausted ─────────────────────────────────────────────

    def test_is_daily_quota_exhausted_true_for_per_day_violation(self) -> None:
        err = genai_errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "details": [
                        {
                            "violations": [
                                {
                                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                                }
                            ]
                        }
                    ],
                }
            },
        )
        assert _is_daily_quota_exhausted(err) is True

    def test_is_daily_quota_exhausted_false_for_per_minute_limit(self) -> None:
        err = genai_errors.ClientError(
            429,
            {"error": {"code": 429, "details": [{"retryDelay": "10s"}]}},
        )
        assert _is_daily_quota_exhausted(err) is False

    def test_is_daily_quota_exhausted_false_for_empty_violations(self) -> None:
        err = genai_errors.ClientError(
            429,
            {"error": {"code": 429, "details": [{"violations": []}]}},
        )
        assert _is_daily_quota_exhausted(err) is False

    # ── _classify_gemini_error ────────────────────────────────────────────────

    def test_classify_non_client_error_returned_unchanged(self) -> None:
        """Non-ClientError exceptions are returned as-is without classification."""
        original = RuntimeError("network timeout")
        result = _classify_gemini_error(original)
        assert result is original

    def test_classify_429_with_per_day_quota_returns_quota_exhausted(self) -> None:
        err = genai_errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "details": [
                        {
                            "violations": [
                                {"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}
                            ]
                        }
                    ],
                }
            },
        )
        result = _classify_gemini_error(err)
        assert isinstance(result, GeminiQuotaExhaustedError)

    def test_classify_429_with_retry_delay_returns_rate_limit(self) -> None:
        err = genai_errors.ClientError(
            429, {"error": {"code": 429, "details": [{"retryDelay": "15s"}]}}
        )
        result = _classify_gemini_error(err)
        assert isinstance(result, GeminiRateLimitError)
        assert result.retry_after == 15.0

    def test_classify_non_429_client_error_returned_unchanged(self) -> None:
        err = genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad request"}})
        result = _classify_gemini_error(err)
        assert result is err


class TestGeminiRetryWait:
    """Waiting is tenacity's job now — evaluate() no longer sleeps on its own."""

    @staticmethod
    def _state(exc: Exception | None, attempt: int = 1) -> RetryCallState:
        state = RetryCallState(retry_object=None, fn=None, args=(), kwargs={})
        state.attempt_number = attempt
        outcome: Future = Future()
        if exc is not None:
            outcome.set_exception(exc)
        else:
            outcome.set_result(None)
        state.outcome = outcome
        return state

    def test_honours_gemini_retry_delay(self) -> None:
        wait = _gemini_retry_wait(self._state(GeminiRateLimitError("429", retry_after=45.0)))
        assert wait == 45.0

    def test_caps_absurd_retry_delay(self) -> None:
        wait = _gemini_retry_wait(self._state(GeminiRateLimitError("429", retry_after=9999.0)))
        assert wait <= 60.0

    def test_never_waits_less_than_exponential_floor(self) -> None:
        """A tiny retryDelay must not defeat backoff on repeated failures."""
        wait = _gemini_retry_wait(self._state(GeminiRateLimitError("429", retry_after=0.1)))
        assert wait >= 5.0

    def test_falls_back_to_exponential_for_other_errors(self) -> None:
        wait = _gemini_retry_wait(self._state(RuntimeError("503")))
        assert 0 < wait <= 60.0


class TestTruncationDetection:
    """A response cut off at the token ceiling yields unparseable JSON."""

    def _response(self, finish_reason: object) -> MagicMock:
        candidate = MagicMock()
        candidate.finish_reason = finish_reason
        response = MagicMock()
        response.candidates = [candidate]
        return response

    def test_detects_max_tokens_enum(self) -> None:
        assert _is_truncated(self._response(types.FinishReason.MAX_TOKENS)) is True

    def test_detects_max_tokens_string(self) -> None:
        assert _is_truncated(self._response("MAX_TOKENS")) is True

    def test_normal_stop_is_not_truncated(self) -> None:
        assert _is_truncated(self._response(types.FinishReason.STOP)) is False

    def test_missing_candidates_is_not_truncated(self) -> None:
        response = MagicMock()
        response.candidates = None
        assert _is_truncated(response) is False

    @patch("rz_flow.ai.genai.Client")
    async def test_evaluate_raises_incomplete_on_truncation(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        truncated = self._response(types.FinishReason.MAX_TOKENS)
        truncated.text = '{"is_interesting": true, "ua_ti'
        mock_client.aio.models.generate_content = AsyncMock(return_value=truncated)

        ai = GeminiAIFilter(api_key="fake-key")
        with pytest.raises(GeminiResponseIncompleteError):
            await ai.evaluate(_make_article())

    @patch("rz_flow.ai.genai.Client")
    async def test_truncation_is_not_retried(self, mock_client_cls: MagicMock) -> None:
        """Retrying the same prompt would truncate again — the next run is the fix."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        truncated = self._response("MAX_TOKENS")
        truncated.text = "{"
        mock_client.aio.models.generate_content = AsyncMock(return_value=truncated)

        ai = GeminiAIFilter(api_key="fake-key")
        with pytest.raises(GeminiResponseIncompleteError):
            await ai.evaluate(_make_article())

        assert mock_client.aio.models.generate_content.call_count == 1


class TestPromptThreshold:
    """The publish threshold lives in Settings — the prompt must not hardcode its own."""

    def test_prompt_uses_configured_threshold(self) -> None:
        assert "score 8–10" in _build_system_prompt(8.0)
        assert "score 7–10" not in _build_system_prompt(8.0)

    def test_prompt_renders_whole_numbers_without_decimal(self) -> None:
        assert "score 7–10" in _build_system_prompt(7.0)

    def test_prompt_keeps_fractional_thresholds(self) -> None:
        assert "score 7.5–10" in _build_system_prompt(7.5)

    def test_schema_description_matches_threshold(self) -> None:
        schema = _build_response_schema(8.0)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert "score >= 8" in properties["is_interesting"]["description"]

    @patch("rz_flow.ai.genai.Client")
    async def test_filter_sends_configured_threshold_to_gemini(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_gemini_response(VALID_AI_RESPONSE)
        )

        ai = GeminiAIFilter(api_key="fake-key", min_score=9.0)
        await ai.evaluate(_make_article())

        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert "score 9–10" in config.system_instruction
