"""Text-to-speech access via a direct HTTP client to Deepgram's Aura TTS API.

Only module allowed to call Deepgram's speak endpoint directly — everything
else imports ``synthesize``/``asynthesize`` from here.

Implementation path chosen, and why (verified against the installed
litellm==1.89.2 via direct introspection, not memory/docs-only):

1. ``pip show litellm`` -> version ``1.89.2``.
2. ``.venv/Lib/site-packages/litellm/llms/deepgram/`` contains only an
   ``audio_transcription`` submodule (mirrors ``stt.py``'s STT usage) — there
   is **no** ``audio_speech``/``text_to_speech`` submodule for Deepgram.
3. ``model_prices_and_context_window_backup.json`` (the installed package's
   own model registry) lists 34 ``deepgram/*`` entries — every single one is
   ``mode: audio_transcription``. Zero ``deepgram/aura*`` entries exist, and
   zero Deepgram entries of any kind have ``mode: audio_speech``. The
   providers that DO have ``audio_speech`` entries are: openai, azure,
   elevenlabs, vertex_ai, gemini, runwayml, minimax, aws_polly — deepgram is
   absent from that list entirely.
4. ``litellm.get_llm_provider("deepgram/aura-2-thalia-en")`` resolves to
   ``("aura-2-thalia-en", "deepgram", None, None)`` — but this is *only*
   prefix-string resolution (anything shaped ``"deepgram/<rest>"`` resolves
   the same way) and does NOT imply real dispatch support.
5. Reading ``inspect.getsource(litellm.speech)`` directly confirms the actual
   dispatch ``elif custom_llm_provider == "...":`` branches cover exactly:
   openai, azure, elevenlabs, vertex_ai/vertex_ai_beta, gemini, runwayml,
   minimax, aws_polly. **There is no ``deepgram`` branch.** Calling
   ``litellm.speech(model="deepgram/...")`` would fall through with no
   matching provider and fail — it is not a supported path in this version.

Conclusion: LiteLLM does **not** support Deepgram as a ``speech()``/
``aspeech()`` provider in the installed version, despite Deepgram itself
having a real Aura TTS product. This module therefore implements a minimal,
well-tested direct HTTP client against Deepgram's current REST API instead
of (incorrectly) routing through ``litellm.speech``.

Verified endpoint shape (via WebFetch against the live
developers.deepgram.com/reference/text-to-speech/speak-request docs page,
2026-06-21, not memory):
    POST https://api.deepgram.com/v1/speak?model=<model>
    Headers: "Authorization: Token <DEEPGRAM_API_KEY>", "Content-Type: application/json"
    Body:    {"text": "<text>"}
    Response: raw audio bytes in the body (default encoding: mp3) on 200.
    Errors: Deepgram returns 401 for bad/missing key, 4xx/5xx otherwise.

The gateway still routes through ``app/gateway/retry.py`` for consistent
retry/backoff/structured-logging behavior with ``llm.py``/``stt.py``, and
still raises the same typed exceptions
(``GatewayAuthError``/``GatewayAllProvidersFailedError``/``GatewayError``)
from ``app/gateway/exceptions.py``. Since there is no ``litellm`` exception
to catch here (this is a raw ``httpx`` call), this module maps HTTP status
codes onto the same transient/auth taxonomy ``retry.py``'s ``TRANSIENT_EXCEPTIONS``
expects by raising the equivalent ``litellm`` exception types itself — this
keeps ``call_with_retry``/``acall_with_retry`` provider-agnostic and avoids
duplicating retry logic here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import litellm
from websockets.sync.client import connect as ws_connect
from websockets.exceptions import InvalidStatus, WebSocketException

from app.config import config
from app.gateway.exceptions import GatewayAuthError, GatewayError
from app.gateway.retry import acall_with_retry, call_with_retry

_DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
_DEEPGRAM_SPEAK_WS_URL = "wss://api.deepgram.com/v1/speak"
_DEFAULT_TIMEOUT_S = 30
_STREAM_ENCODING = "linear16"
_STREAM_SAMPLE_RATE = 24000

# Latency fix: the original implementation opened a brand-new httpx.Client
# (`with httpx.Client(...) as client:`) on every single call, paying a fresh
# DNS lookup + TCP + TLS handshake to api.deepgram.com on every interviewer
# turn. litellm's own STT path avoids this (it caches/reuses httpx clients
# internally — see `get_async_httpx_client`/`_get_httpx_client` in
# litellm/llms/custom_httpx/http_handler.py), so this hand-rolled TTS client
# is brought in line with that pattern: one process-lifetime client per sync/
# async call style, reused across every `synthesize`/`asynthesize` call.
#
# `keepalive_expiry` is raised well above httpx's 5s default because
# interview turns are naturally tens of seconds apart (record + think time);
# a 5s-expiry pooled connection would already be dead by the next turn,
# silently defeating the whole point of reuse. Even on a cold connection,
# reusing the same client/SSLContext still gets TLS session-resumption
# benefits that a brand-new httpx.Client() each call cannot get.
_KEEPALIVE_LIMITS = httpx.Limits(max_keepalive_connections=20, keepalive_expiry=120.0)

_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(timeout=_DEFAULT_TIMEOUT_S, limits=_KEEPALIVE_LIMITS)
    return _sync_client


def _get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_S, limits=_KEEPALIVE_LIMITS)
    return _async_client


def _raise_for_response(response: httpx.Response, *, model: str) -> None:
    """Translate a non-2xx Deepgram HTTP response into the same litellm
    exception types ``app/gateway/retry.py`` already knows how to classify
    (transient vs. auth vs. unexpected), so retry behavior is identical
    across every call type without duplicating retry logic here."""
    if response.status_code == 401 or response.status_code == 403:
        raise litellm.AuthenticationError(
            message=f"Deepgram TTS authentication failed (HTTP {response.status_code})",
            llm_provider="deepgram",
            model=model,
        )
    if response.status_code == 429:
        raise litellm.RateLimitError(
            message="Deepgram TTS rate limited (HTTP 429)",
            llm_provider="deepgram",
            model=model,
        )
    if response.status_code >= 500:
        raise litellm.ServiceUnavailableError(
            message=f"Deepgram TTS server error (HTTP {response.status_code})",
            llm_provider="deepgram",
            model=model,
        )
    response.raise_for_status()


def synthesize(
    text: str,
    *,
    encoding: str | None = None,
    sample_rate: int | None = None,
    **kwargs,
) -> bytes:
    """Convert ``text`` to speech audio bytes via Deepgram Aura's REST
    endpoint. Defaults to mp3 (Deepgram's REST default) when ``encoding``/
    ``sample_rate`` are omitted, matching this function's original behavior
    exactly.

    ``encoding``/``sample_rate`` are explicit named params (not folded into
    ``**kwargs``) specifically so a caller can request the SAME raw-PCM
    format ``stream_synthesize()`` produces — see
    ``app/web/server.py``'s one-shot fallback, which calls
    ``synthesize(text, encoding="linear16", sample_rate=24000)`` for exactly
    this reason: the fallback must be byte-format-interchangeable with the
    streaming path, since the browser-side player doesn't know which path
    served it.

    Raises:
        GatewayAuthError: missing/invalid DEEPGRAM_API_KEY.
        GatewayAllProvidersFailedError: transient failures exhausted retries.
        GatewayError: any other unexpected failure.
    """
    if not config.deepgram_api_key:
        raise GatewayAuthError(
            "DEEPGRAM_API_KEY is not set; cannot call the TTS provider."
        )

    model = config.deepgram_tts_model
    params: dict[str, str | int] = {"model": model}
    if encoding is not None:
        params["encoding"] = encoding
    if sample_rate is not None:
        params["sample_rate"] = sample_rate

    def _call():
        client = _get_sync_client()
        try:
            response = client.post(
                _DEEPGRAM_SPEAK_URL,
                params=params,
                headers={
                    "Authorization": f"Token {config.deepgram_api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise litellm.Timeout(
                message=f"Deepgram TTS request timed out: {exc}",
                llm_provider="deepgram",
                model=model,
            ) from exc
        except httpx.ConnectError as exc:
            raise litellm.APIConnectionError(
                message=f"Deepgram TTS connection failed: {exc}",
                llm_provider="deepgram",
                model=model,
            ) from exc
        _raise_for_response(response, model=model)
        return response

    response = call_with_retry(_call, call_type="tts", model=model)
    return response.content


async def asynthesize(text: str, **kwargs) -> bytes:
    """Async counterpart of :func:`synthesize`."""
    if not config.deepgram_api_key:
        raise GatewayAuthError(
            "DEEPGRAM_API_KEY is not set; cannot call the TTS provider."
        )

    model = config.deepgram_tts_model

    async def _call():
        client = _get_async_client()
        try:
            response = await client.post(
                _DEEPGRAM_SPEAK_URL,
                params={"model": model},
                headers={
                    "Authorization": f"Token {config.deepgram_api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise litellm.Timeout(
                message=f"Deepgram TTS request timed out: {exc}",
                llm_provider="deepgram",
                model=model,
            ) from exc
        except httpx.ConnectError as exc:
            raise litellm.APIConnectionError(
                message=f"Deepgram TTS connection failed: {exc}",
                llm_provider="deepgram",
                model=model,
            ) from exc
        _raise_for_response(response, model=model)
        return response

    response = await acall_with_retry(_call, call_type="tts", model=model)
    return response.content


def stream_synthesize(text: str) -> Iterator[bytes]:
    """Stream ``text`` to speech as raw PCM chunks, AS Deepgram generates
    them, via Deepgram's real-time TTS WebSocket endpoint.

    Unlike ``synthesize()``/``asynthesize()`` (REST, full-clip-then-return),
    this is a sync generator: each ``yield`` happens the moment a binary
    audio chunk arrives off the wire, before the rest of the clip has even
    been generated server-side — this is what lets a caller (the web layer)
    start relaying audio to the browser well before the full reply is ready.

    Output format is fixed at raw 16-bit little-endian PCM, mono, 24000 Hz
    (``linear16``) — the only encodings this endpoint supports are
    ``linear16``/``mulaw``/``alaw``, never mp3, unlike the REST endpoint used
    by ``synthesize()``. Callers that need the two to produce interchangeable
    bytes (see ``app/web/server.py``'s one-shot fallback) must account for
    this format difference themselves; it is intentional, not an oversight.

    Protocol (confirmed live against the real API, not docs-only — see
    ``.claude/loop-state/streaming-tts/backend-self-check.md``):
        1. Connect to ``wss://api.deepgram.com/v1/speak`` with
           ``model``/``encoding``/``sample_rate`` query params and an
           ``Authorization: Token <key>`` header (confirmed live: the
           no-prefix scheme the spec flagged as a fallback 401s; the
           "Token "-prefixed scheme used by the REST call also works here).
        2. Send ``{"type": "Speak", "text": ...}`` then ``{"type": "Flush"}``.
        3. Server replies with one ``Metadata`` JSON message (ignored here),
           then binary PCM chunks (yielded one at a time as they arrive),
           then a ``Flushed`` JSON message marking the end of this clip.
        4. Send ``{"type": "Close"}`` and let the ``with`` block close the
           socket — happens even if the caller's consuming loop raises,
           since this is a generator driven by a context manager.

    Raises:
        GatewayAuthError: missing API key, or the WS handshake itself was
            rejected with HTTP 401/403 (bad/invalid key).
        GatewayError: any other failure to establish or maintain the WS
            connection (DNS/TCP failure, unexpected close, protocol error).
            Per spec.md, callers should only fall back to one-shot
            ``synthesize()`` if this is raised BEFORE any chunk has been
            yielded — once at least one chunk has reached the caller there
            is nothing sane to fall back to mid-stream (documented
            limitation, not a bug).
    """
    if not config.deepgram_api_key:
        raise GatewayAuthError(
            "DEEPGRAM_API_KEY is not set; cannot call the TTS provider."
        )

    model = config.deepgram_tts_model
    url = (
        f"{_DEEPGRAM_SPEAK_WS_URL}?model={model}"
        f"&encoding={_STREAM_ENCODING}&sample_rate={_STREAM_SAMPLE_RATE}"
    )
    headers = {"Authorization": f"Token {config.deepgram_api_key}"}

    try:
        with ws_connect(url, additional_headers=headers) as websocket:
            websocket.send(json.dumps({"type": "Speak", "text": text}))
            websocket.send(json.dumps({"type": "Flush"}))

            try:
                while True:
                    message = websocket.recv()
                    if isinstance(message, bytes):
                        yield message
                        continue
                    # Text frames are JSON control messages: "Metadata"
                    # (ignored — informational only) and "Flushed" (the
                    # signal this clip is fully delivered; stop reading).
                    payload = json.loads(message)
                    if payload.get("type") == "Flushed":
                        break
            finally:
                # Always attempt a clean Close handshake, even if the loop
                # above exited via an exception from the caller (e.g. the
                # consumer broke out of the generator) — the `with` block
                # still guarantees the socket itself closes either way.
                try:
                    websocket.send(json.dumps({"type": "Close"}))
                except WebSocketException:
                    pass
    except InvalidStatus as exc:
        status_code = exc.response.status_code
        if status_code in (401, 403):
            raise GatewayAuthError(
                f"tts: Deepgram streaming TTS authentication failed "
                f"(HTTP {status_code})"
            ) from exc
        raise GatewayError(
            f"tts: Deepgram streaming TTS handshake rejected "
            f"(HTTP {status_code})"
        ) from exc
    except WebSocketException as exc:
        raise GatewayError(
            f"tts: Deepgram streaming TTS WebSocket error: {exc}"
        ) from exc
    except OSError as exc:
        raise GatewayError(
            f"tts: Deepgram streaming TTS connection failed: {exc}"
        ) from exc
