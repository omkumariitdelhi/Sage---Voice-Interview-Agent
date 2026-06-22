"""Interviewer LLM access via LiteLLM + OpenRouter.

This module is the *only* place in the app allowed to call
``litellm.completion``/``litellm.acompletion`` directly — every other module
(LangGraph nodes, FastAPI routes, etc.) must import ``complete``/``acomplete``
from here. That single chokepoint is what makes "no inline litellm calls
outside app/gateway/" grep-checkable.
"""

from __future__ import annotations

import litellm

from app.config import config
from app.gateway.exceptions import GatewayAuthError
from app.gateway.retry import acall_with_retry, call_with_retry

# Native retry disabled (0) — app/gateway/retry.py owns retry/backoff so we
# don't double-retry or duplicate backoff timing.
_NATIVE_RETRIES_DISABLED = 0
_DEFAULT_TIMEOUT_S = 30


def _require_content(response, *, model: str):
    """Raise a retryable transient error if the provider returned a response
    with no actual completion text.

    Found live (not in mocked tests) against the free-tier
    ``openrouter/openai/gpt-oss-120b:free`` model: it occasionally returns
    ``choices[0].message.content = None`` alongside an "Unmapped
    finish_reason 'error'" warning from litellm -- a provider-side hiccup,
    not a malformed-JSON case. Before this guard, that ``None`` propagated
    unchecked into every caller (``evaluation.py``'s/``feedback.py``'s
    ``json.loads(None)`` raised an uncaught ``TypeError``, not the
    ``json.JSONDecodeError`` they already catch and fall back on) and
    crashed the request instead of degrading gracefully. Raising
    ``litellm.ServiceUnavailableError`` here (a member of
    ``retry.py``'s ``TRANSIENT_EXCEPTIONS``) makes this participate in the
    existing retry loop, and a still-empty response after retries surfaces
    as the same typed ``GatewayAllProvidersFailedError`` every caller already
    catches -- fixing this once at the gateway chokepoint restores graceful
    degradation everywhere without touching any caller.
    """
    content = response.choices[0].message.content
    if not content:
        raise litellm.ServiceUnavailableError(
            message="Provider returned an empty/null completion (no content)",
            llm_provider="openrouter",
            model=model,
        )
    return content


def complete(messages: list[dict], **kwargs) -> str:
    """Send ``messages`` to the configured interviewer LLM; return the text.

    Raises:
        GatewayAuthError: missing/invalid OPENROUTER_API_KEY.
        GatewayAllProvidersFailedError: transient failures exhausted retries
            (including a persistently empty/null completion -- see
            :func:`_require_content`).
        GatewayError: any other unexpected failure.
    """
    if not config.openrouter_api_key:
        raise GatewayAuthError(
            "OPENROUTER_API_KEY is not set; cannot call the interviewer LLM."
        )

    model = config.openrouter_model

    def _call():
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=config.openrouter_api_key,
            timeout=_DEFAULT_TIMEOUT_S,
            num_retries=_NATIVE_RETRIES_DISABLED,
            **kwargs,
        )
        _require_content(response, model=model)
        return response

    response = call_with_retry(_call, call_type="llm", model=model)
    return response.choices[0].message.content


async def acomplete(messages: list[dict], **kwargs) -> str:
    """Async counterpart of :func:`complete`. Preferred call path — this
    gateway will be invoked from a FastAPI app (Phase 6)."""
    if not config.openrouter_api_key:
        raise GatewayAuthError(
            "OPENROUTER_API_KEY is not set; cannot call the interviewer LLM."
        )

    model = config.openrouter_model

    async def _call():
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            api_key=config.openrouter_api_key,
            timeout=_DEFAULT_TIMEOUT_S,
            num_retries=_NATIVE_RETRIES_DISABLED,
            **kwargs,
        )
        _require_content(response, model=model)
        return response

    response = await acall_with_retry(_call, call_type="llm", model=model)
    return response.choices[0].message.content
