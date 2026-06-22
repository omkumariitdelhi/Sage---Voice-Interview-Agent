"""Speech-to-text access via LiteLLM + Deepgram.

Only module allowed to call ``litellm.transcription``/``litellm.atranscription``
directly — everything else imports ``transcribe``/``atranscribe`` from here.

Provider note (verified against the installed litellm==1.89.2 via direct
introspection — ``litellm.get_llm_provider("deepgram/nova-2")`` resolves to
``custom_llm_provider="deepgram"``, and
``model_prices_and_context_window_backup.json`` lists ``deepgram/nova-2``
with ``mode: audio_transcription``, ``litellm_provider: deepgram``):
LiteLLM ships **native** Deepgram transcription support. ``transcribe``/
``atranscribe`` call ``litellm.transcription``/``litellm.atranscription``
with ``model="deepgram/<model>"``, which dispatches to Deepgram's
prerecorded-audio endpoint with the configured ``DEEPGRAM_API_KEY``. TTS
(``tts.py``) also uses Deepgram (Aura) but, unlike STT, calls Deepgram's
REST API directly via ``httpx`` rather than through ``litellm`` — the
installed litellm version has no native Deepgram speech provider; see
``tts.py``'s module docstring for the verified details. Both call types
share the same ``DEEPGRAM_API_KEY``.
"""

from __future__ import annotations

import os

import litellm

from app.config import config
from app.gateway.exceptions import GatewayAuthError
from app.gateway.retry import acall_with_retry, call_with_retry

# Note: litellm.transcription/atranscription expose `max_retries`, not
# `num_retries` (confirmed by signature introspection against the installed
# litellm==1.89.2 — see plan.md §1). Disabled here for the same
# don't-double-retry reason as llm.py.
_NATIVE_RETRIES_DISABLED = 0
_DEFAULT_TIMEOUT_S = 30

AudioInput = bytes | str | os.PathLike


def transcribe(audio: AudioInput, **kwargs) -> str:
    """Transcribe ``audio`` (raw bytes or a file path) to text.

    Raises:
        GatewayAuthError: missing/invalid DEEPGRAM_API_KEY.
        GatewayAllProvidersFailedError: transient failures exhausted retries.
        GatewayError: any other unexpected failure.
    """
    if not config.deepgram_api_key:
        raise GatewayAuthError(
            "DEEPGRAM_API_KEY is not set; cannot call the STT provider."
        )

    model = config.deepgram_model

    def _call():
        return litellm.transcription(
            model=model,
            file=audio,
            api_key=config.deepgram_api_key,
            timeout=_DEFAULT_TIMEOUT_S,
            max_retries=_NATIVE_RETRIES_DISABLED,
            **kwargs,
        )

    response = call_with_retry(_call, call_type="stt", model=model)
    return response.text


async def atranscribe(audio: AudioInput, **kwargs) -> str:
    """Async counterpart of :func:`transcribe`."""
    if not config.deepgram_api_key:
        raise GatewayAuthError(
            "DEEPGRAM_API_KEY is not set; cannot call the STT provider."
        )

    model = config.deepgram_model

    async def _call():
        return await litellm.atranscription(
            model=model,
            file=audio,
            api_key=config.deepgram_api_key,
            timeout=_DEFAULT_TIMEOUT_S,
            max_retries=_NATIVE_RETRIES_DISABLED,
            **kwargs,
        )

    response = await acall_with_retry(_call, call_type="stt", model=model)
    return response.text
