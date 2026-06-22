"""Tests for app/gateway/stt.py — fully mocked, zero real API keys/network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.gateway.stt as stt_module
from app.gateway.exceptions import GatewayAuthError


def _fake_transcription_response(text: str) -> SimpleNamespace:
    """Mimic litellm.types.utils.TranscriptionResponse far enough for
    stt.py's `response.text` access."""
    return SimpleNamespace(text=text)


def test_transcribe_happy_path(monkeypatch, with_dummy_keys):
    mock_transcription = MagicMock(
        return_value=_fake_transcription_response("the quick brown fox")
    )
    monkeypatch.setattr(stt_module.litellm, "transcription", mock_transcription)

    result = stt_module.transcribe(b"fake-audio-bytes")

    assert result == "the quick brown fox"
    mock_transcription.assert_called_once()
    _, kwargs = mock_transcription.call_args
    assert kwargs["model"] == with_dummy_keys.deepgram_model
    assert kwargs["api_key"] == "dummy-deepgram-key"
    assert kwargs["file"] == b"fake-audio-bytes"


def test_transcribe_raises_gateway_auth_error_when_no_key(with_no_keys):
    with pytest.raises(GatewayAuthError):
        stt_module.transcribe(b"fake-audio-bytes")


@pytest.mark.asyncio
async def test_atranscribe_happy_path(monkeypatch, with_dummy_keys):
    mock_atranscription = AsyncMock(
        return_value=_fake_transcription_response("async transcript")
    )
    monkeypatch.setattr(stt_module.litellm, "atranscription", mock_atranscription)

    result = await stt_module.atranscribe(b"fake-audio-bytes")

    assert result == "async transcript"
    mock_atranscription.assert_awaited_once()
