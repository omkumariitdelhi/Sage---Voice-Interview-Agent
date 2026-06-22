"""The mandatory forced-failure test (spec acceptance criterion).

For each of the three call types, mocks the underlying litellm function to
raise `litellm.RateLimitError` exactly once, then return a valid mocked
response on the 2nd call. Proves:
  (a) the gateway's retry loop actually fires and the call eventually
      returns the mocked success payload,
  (b) the underlying provider-call mock was invoked twice (so this is a real
      retry, not a no-op),
  (c) a structured log line with outcome="retried" was emitted for the
      failed attempt, followed by outcome="success" for the attempt that
      succeeded — proving the structured-logging contract Phase 7 depends on.

All three call types (llm, stt, tts) are proven, exceeding the spec's "at
least one" hard minimum.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import litellm
import pytest

import app.gateway.llm as llm_module
import app.gateway.stt as stt_module
import app.gateway.tts as tts_module


def _rate_limit_error() -> litellm.RateLimitError:
    # litellm.RateLimitError (and its OpenAI parent) require message/
    # llm_provider/model kwargs to construct cleanly across versions.
    return litellm.RateLimitError(
        message="rate limited (forced for test)",
        llm_provider="openrouter",
        model="test-model",
    )


def _log_records_as_dicts(caplog: pytest.LogCaptureFixture) -> list[dict]:
    records = []
    for record in caplog.records:
        if record.name == "app.gateway":
            records.append(json.loads(record.getMessage()))
    return records


def test_llm_complete_retries_then_succeeds(monkeypatch, with_dummy_keys, caplog):
    success_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))]
    )
    mock_completion = MagicMock(
        side_effect=[_rate_limit_error(), success_response]
    )
    monkeypatch.setattr(llm_module.litellm, "completion", mock_completion)

    with caplog.at_level(logging.INFO, logger="app.gateway"):
        result = llm_module.complete([{"role": "user", "content": "hi"}])

    assert result == "recovered"
    assert mock_completion.call_count == 2

    logs = _log_records_as_dicts(caplog)
    outcomes = [entry["outcome"] for entry in logs if entry["call_type"] == "llm"]
    assert "retried" in outcomes
    assert outcomes[-1] == "success"
    for entry in logs:
        assert {"call_type", "model", "latency_ms", "outcome", "attempt"} <= entry.keys()


def test_stt_transcribe_retries_then_succeeds(monkeypatch, with_dummy_keys, caplog):
    success_response = SimpleNamespace(text="recovered transcript")
    mock_transcription = MagicMock(
        side_effect=[_rate_limit_error(), success_response]
    )
    monkeypatch.setattr(stt_module.litellm, "transcription", mock_transcription)

    with caplog.at_level(logging.INFO, logger="app.gateway"):
        result = stt_module.transcribe(b"fake-audio-bytes")

    assert result == "recovered transcript"
    assert mock_transcription.call_count == 2

    logs = _log_records_as_dicts(caplog)
    outcomes = [entry["outcome"] for entry in logs if entry["call_type"] == "stt"]
    assert "retried" in outcomes
    assert outcomes[-1] == "success"


def test_tts_synthesize_retries_then_succeeds(monkeypatch, with_dummy_keys, caplog):
    # tts.py calls Deepgram's REST endpoint directly via httpx (no native
    # litellm Deepgram speech provider exists in the installed version — see
    # tts.py's module docstring). A 429 response is translated by tts.py's
    # `_raise_for_response` into `litellm.RateLimitError`, which is the same
    # transient-exception type `app/gateway/retry.py` already retries on for
    # every other call type.
    rate_limited_response = httpx.Response(
        status_code=429,
        content=b"",
        request=httpx.Request("POST", tts_module._DEEPGRAM_SPEAK_URL),
    )
    success_response = httpx.Response(
        status_code=200,
        content=b"recovered-mp3-bytes",
        request=httpx.Request("POST", tts_module._DEEPGRAM_SPEAK_URL),
    )
    mock_post = MagicMock(side_effect=[rate_limited_response, success_response])
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    with caplog.at_level(logging.INFO, logger="app.gateway"):
        result = tts_module.synthesize("hello")

    assert result == b"recovered-mp3-bytes"
    assert mock_post.call_count == 2

    logs = _log_records_as_dicts(caplog)
    outcomes = [entry["outcome"] for entry in logs if entry["call_type"] == "tts"]
    assert "retried" in outcomes
    assert outcomes[-1] == "success"


@pytest.mark.asyncio
async def test_async_acomplete_retries_then_succeeds(monkeypatch, with_dummy_keys, caplog):
    success_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="async recovered"))]
    )
    mock_acompletion = AsyncMock(side_effect=[_rate_limit_error(), success_response])
    monkeypatch.setattr(llm_module.litellm, "acompletion", mock_acompletion)

    with caplog.at_level(logging.INFO, logger="app.gateway"):
        result = await llm_module.acomplete([{"role": "user", "content": "hi"}])

    assert result == "async recovered"
    assert mock_acompletion.await_count == 2

    logs = _log_records_as_dicts(caplog)
    outcomes = [entry["outcome"] for entry in logs if entry["call_type"] == "llm"]
    assert "retried" in outcomes
    assert outcomes[-1] == "success"


def test_all_attempts_exhausted_raises_all_providers_failed(monkeypatch, with_dummy_keys):
    from app.gateway.exceptions import GatewayAllProvidersFailedError

    mock_completion = MagicMock(side_effect=_rate_limit_error())
    monkeypatch.setattr(llm_module.litellm, "completion", mock_completion)

    with pytest.raises(GatewayAllProvidersFailedError):
        llm_module.complete([{"role": "user", "content": "hi"}])

    # call_with_retry's default max_attempts is 3.
    assert mock_completion.call_count == 3


def test_auth_error_is_not_retried(monkeypatch, with_dummy_keys):
    from app.gateway.exceptions import GatewayAuthError

    auth_error = litellm.AuthenticationError(
        message="bad key (forced for test)",
        llm_provider="openrouter",
        model="test-model",
    )
    mock_completion = MagicMock(side_effect=auth_error)
    monkeypatch.setattr(llm_module.litellm, "completion", mock_completion)

    with pytest.raises(GatewayAuthError):
        llm_module.complete([{"role": "user", "content": "hi"}])

    # Auth errors must fail fast — exactly one attempt, never retried.
    assert mock_completion.call_count == 1
