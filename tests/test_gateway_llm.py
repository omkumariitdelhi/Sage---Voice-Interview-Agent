"""Tests for app/gateway/llm.py — fully mocked, zero real API keys/network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.gateway.llm as llm_module
from app.gateway.exceptions import GatewayAllProvidersFailedError, GatewayAuthError


def _fake_model_response(text: str) -> SimpleNamespace:
    """Mimic the shape of litellm.types.utils.ModelResponse far enough for
    llm.py's `response.choices[0].message.content` access."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_complete_happy_path(monkeypatch, with_dummy_keys):
    mock_completion = MagicMock(return_value=_fake_model_response("hello there"))
    monkeypatch.setattr(llm_module.litellm, "completion", mock_completion)

    result = llm_module.complete([{"role": "user", "content": "hi"}])

    assert result == "hello there"
    mock_completion.assert_called_once()
    _, kwargs = mock_completion.call_args
    assert kwargs["model"] == with_dummy_keys.openrouter_model
    assert kwargs["api_key"] == "dummy-openrouter-key"
    assert kwargs["timeout"] == 30


def test_complete_raises_gateway_auth_error_when_no_key(with_no_keys):
    with pytest.raises(GatewayAuthError):
        llm_module.complete([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_acomplete_happy_path(monkeypatch, with_dummy_keys):
    mock_acompletion = AsyncMock(return_value=_fake_model_response("async hello"))
    monkeypatch.setattr(llm_module.litellm, "acompletion", mock_acompletion)

    result = await llm_module.acomplete([{"role": "user", "content": "hi"}])

    assert result == "async hello"
    mock_acompletion.assert_awaited_once()


def test_complete_retries_on_empty_content_then_succeeds(monkeypatch, with_dummy_keys):
    """Regression test for a real bug found via live testing (not the mocked
    suite): the free-tier openrouter/openai/gpt-oss-120b:free model
    occasionally returns choices[0].message.content=None alongside an
    "Unmapped finish_reason 'error'" litellm warning. Before the
    _require_content guard, that None propagated unchecked into
    evaluation.py/feedback.py's json.loads(None), raising an uncaught
    TypeError (not the json.JSONDecodeError they already catch) and
    crashing the request instead of degrading gracefully. This proves the
    gateway itself now retries on empty content like any other transient
    failure."""
    mock_completion = MagicMock(
        side_effect=[
            _fake_model_response(None),
            _fake_model_response(""),
            _fake_model_response("recovered on the third attempt"),
        ]
    )
    monkeypatch.setattr(llm_module.litellm, "completion", mock_completion)

    result = llm_module.complete([{"role": "user", "content": "hi"}])

    assert result == "recovered on the third attempt"
    assert mock_completion.call_count == 3


def test_complete_raises_all_providers_failed_when_content_persistently_empty(
    monkeypatch, with_dummy_keys
):
    mock_completion = MagicMock(return_value=_fake_model_response(None))
    monkeypatch.setattr(llm_module.litellm, "completion", mock_completion)

    with pytest.raises(GatewayAllProvidersFailedError):
        llm_module.complete([{"role": "user", "content": "hi"}])

    assert mock_completion.call_count == 3


@pytest.mark.asyncio
async def test_acomplete_retries_on_empty_content_then_succeeds(monkeypatch, with_dummy_keys):
    mock_acompletion = AsyncMock(
        side_effect=[
            _fake_model_response(None),
            _fake_model_response("recovered async"),
        ]
    )
    monkeypatch.setattr(llm_module.litellm, "acompletion", mock_acompletion)

    result = await llm_module.acomplete([{"role": "user", "content": "hi"}])

    assert result == "recovered async"
    assert mock_acompletion.await_count == 2
