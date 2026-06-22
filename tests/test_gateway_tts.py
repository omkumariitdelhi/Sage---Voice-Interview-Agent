"""Tests for app/gateway/tts.py — fully mocked, zero real API keys/network.

tts.py calls Deepgram's /v1/speak REST endpoint directly via httpx (litellm
has no native Deepgram speech provider in the installed version — see
tts.py's module docstring), so these tests mock httpx.Client.post /
httpx.AsyncClient.post rather than a litellm function.

``stream_synthesize`` (added for streaming TTS — see
.claude/loop-state/streaming-tts/spec.md) uses
``websockets.sync.client.connect`` instead, so its tests mock
``app.gateway.tts.ws_connect`` (the module's own imported name) with a
``MagicMock`` context manager whose ``.send()``/``.recv()`` are scripted.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from websockets.exceptions import InvalidStatus

import app.gateway.tts as tts_module
from app.gateway.exceptions import GatewayAuthError, GatewayError


def _fake_speech_response(audio_bytes: bytes, status_code: int = 200) -> httpx.Response:
    """A real httpx.Response so tts.py's `.content`/`.raise_for_status()`
    access works exactly as it would against a live call."""
    return httpx.Response(
        status_code=status_code,
        content=audio_bytes,
        request=httpx.Request("POST", tts_module._DEEPGRAM_SPEAK_URL),
    )


def test_synthesize_happy_path(monkeypatch, with_dummy_keys):
    mock_post = MagicMock(return_value=_fake_speech_response(b"fake-mp3-bytes"))
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result = tts_module.synthesize("hello, this is the interviewer speaking")

    assert result == b"fake-mp3-bytes"
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["params"] == {"model": with_dummy_keys.deepgram_tts_model}
    assert kwargs["headers"]["Authorization"] == "Token dummy-deepgram-key"
    assert kwargs["json"] == {"text": "hello, this is the interviewer speaking"}


def test_synthesize_with_explicit_encoding_matches_stream_format(monkeypatch, with_dummy_keys):
    """Regression test: the one-shot fallback in app/web/server.py calls
    synthesize(text, encoding="linear16", sample_rate=24000) so its bytes
    are interchangeable with stream_synthesize()'s format — these params
    must reach Deepgram as query params alongside `model`, not collide with
    it."""
    mock_post = MagicMock(return_value=_fake_speech_response(b"fake-pcm-bytes"))
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    result = tts_module.synthesize(
        "hello", encoding="linear16", sample_rate=24000
    )

    assert result == b"fake-pcm-bytes"
    _, kwargs = mock_post.call_args
    assert kwargs["params"] == {
        "model": with_dummy_keys.deepgram_tts_model,
        "encoding": "linear16",
        "sample_rate": 24000,
    }


def test_synthesize_raises_gateway_auth_error_when_no_key(with_no_keys):
    with pytest.raises(GatewayAuthError):
        tts_module.synthesize("hello")


@pytest.mark.asyncio
async def test_asynthesize_happy_path(monkeypatch, with_dummy_keys):
    mock_post = AsyncMock(return_value=_fake_speech_response(b"async-mp3-bytes"))
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await tts_module.asynthesize("hello async")

    assert result == b"async-mp3-bytes"
    mock_post.assert_awaited_once()


# ---------------------------------------------------------------------------
# stream_synthesize — streaming TTS over a (mocked) WebSocket connection.
# See .claude/loop-state/streaming-tts/spec.md.
# ---------------------------------------------------------------------------


def _make_mock_websocket(recv_sequence: list) -> MagicMock:
    """A MagicMock standing in for the ``ClientConnection`` returned by
    ``websockets.sync.client.connect``, usable as a context manager (so
    ``with ws_connect(...) as websocket:`` works unchanged) whose
    ``.recv()`` plays back ``recv_sequence`` in order, one item per call."""
    mock_ws = MagicMock()
    mock_ws.__enter__ = MagicMock(return_value=mock_ws)
    mock_ws.__exit__ = MagicMock(return_value=False)
    mock_ws.recv = MagicMock(side_effect=list(recv_sequence))
    return mock_ws


def _metadata_message() -> str:
    return json.dumps({"type": "Metadata", "request_id": "abc123"})


def _flushed_message() -> str:
    return json.dumps({"type": "Flushed"})


def test_stream_synthesize_yields_chunks_in_order_and_sends_close(
    monkeypatch, with_dummy_keys
):
    chunks = [b"chunk-one", b"chunk-two", b"chunk-three"]
    recv_sequence = [_metadata_message(), *chunks, _flushed_message()]
    mock_ws = _make_mock_websocket(recv_sequence)
    mock_connect = MagicMock(return_value=mock_ws)
    monkeypatch.setattr(tts_module, "ws_connect", mock_connect)

    result = list(tts_module.stream_synthesize("hello, this is the interviewer"))

    assert result == chunks

    # Speak + Flush were sent before any chunk was read.
    sent_payloads = [json.loads(call.args[0]) for call in mock_ws.send.call_args_list]
    assert sent_payloads[0] == {"type": "Speak", "text": "hello, this is the interviewer"}
    assert sent_payloads[1] == {"type": "Flush"}
    # Close is sent last, after Flushed was observed.
    assert sent_payloads[-1] == {"type": "Close"}

    # Connected with the expected URL shape and auth header scheme
    # (confirmed live: "Authorization: Token <key>" — see
    # backend-self-check.md).
    connect_args, connect_kwargs = mock_connect.call_args
    url = connect_args[0]
    assert url.startswith("wss://api.deepgram.com/v1/speak")
    assert "encoding=linear16" in url
    assert "sample_rate=24000" in url
    assert connect_kwargs["additional_headers"]["Authorization"] == "Token dummy-deepgram-key"


def test_stream_synthesize_stops_at_flushed_without_reading_further(
    monkeypatch, with_dummy_keys
):
    """recv() is never called again after Flushed — the generator must stop
    reading, not keep pulling messages past the end of this clip."""
    chunks = [b"only-chunk"]
    recv_sequence = [_metadata_message(), *chunks, _flushed_message()]
    mock_ws = _make_mock_websocket(recv_sequence)
    monkeypatch.setattr(tts_module, "ws_connect", MagicMock(return_value=mock_ws))

    result = list(tts_module.stream_synthesize("short text"))

    assert result == chunks
    assert mock_ws.recv.call_count == len(recv_sequence)


def test_stream_synthesize_raises_gateway_auth_error_when_no_key(with_no_keys):
    with pytest.raises(GatewayAuthError):
        list(tts_module.stream_synthesize("hello"))


def test_stream_synthesize_raises_gateway_auth_error_on_401_handshake(
    monkeypatch, with_dummy_keys
):
    response = MagicMock()
    response.status_code = 401
    monkeypatch.setattr(
        tts_module,
        "ws_connect",
        MagicMock(side_effect=InvalidStatus(response)),
    )

    with pytest.raises(GatewayAuthError):
        list(tts_module.stream_synthesize("hello"))


def test_stream_synthesize_raises_gateway_error_on_connection_failure(
    monkeypatch, with_dummy_keys
):
    monkeypatch.setattr(
        tts_module,
        "ws_connect",
        MagicMock(side_effect=OSError("connection refused")),
    )

    with pytest.raises(GatewayError):
        list(tts_module.stream_synthesize("hello"))
