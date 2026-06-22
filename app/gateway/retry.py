"""Shared retry-with-backoff helper used by llm.py, stt.py, and tts.py.

Centralizing this logic in one place (rather than copy-pasting a retry loop
into each of the three call-type modules) means the retry/backoff/structured
-logging behavior is implemented once, tested once via the forced-failure
tests, and guaranteed identical across call types.

Design notes:
- The gateway owns retries itself rather than relying solely on LiteLLM's
  native ``num_retries``/``max_retries`` kwargs, because (a) those kwargs are
  named inconsistently across call types in the installed litellm==1.89.2
  (``num_retries`` for ``completion``, ``max_retries`` for
  ``transcription``/``speech``), and (b) the SDK's internal retry does not
  give us a hook to emit our own structured "retried" log line per attempt.
  Call sites pass the native retry kwarg as 0/disabled and let this helper
  be the single source of truth for retry behavior.
- Auth errors are never retried — a bad key will not become a good key.
- Only a known set of transient LiteLLM exceptions trigger a retry; anything
  else is wrapped and raised immediately (no bare `except`, ever).
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, TypeVar

import litellm

from app.gateway.exceptions import (
    GatewayAllProvidersFailedError,
    GatewayAuthError,
    GatewayError,
)
from app.gateway.logging_utils import log_call

T = TypeVar("T")

# Exceptions considered transient and therefore retryable. All are
# OpenAI-compatible types re-exported on the `litellm` namespace, confirmed
# present in litellm==1.89.2 via direct introspection (see plan.md §1).
TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
)


def call_with_retry(
    fn: Callable[[], T],
    *,
    call_type: str,
    model: str,
    max_attempts: int = 3,
    base_delay: float = 0.01,
) -> T:
    """Run ``fn()`` with retry-on-transient-failure and structured logging.

    ``fn`` must be a zero-argument callable wrapping the actual LiteLLM call
    (e.g. ``lambda: litellm.completion(...)``). Returns whatever ``fn()``
    returns on success.

    Raises:
        GatewayAuthError: on ``litellm.AuthenticationError`` — never retried.
        GatewayAllProvidersFailedError: when every attempt has been
            exhausted on a transient error.
        GatewayError: on any other, non-transient exception.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        try:
            result = fn()
        except litellm.AuthenticationError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            log_call(
                call_type, model, latency_ms, "error", attempt=attempt,
                error_type="AuthenticationError",
            )
            raise GatewayAuthError(
                f"{call_type}: authentication failed against provider for "
                f"model={model!r}"
            ) from exc
        except TRANSIENT_EXCEPTIONS as exc:
            latency_ms = (time.monotonic() - start) * 1000
            last_error = exc
            if attempt < max_attempts:
                log_call(
                    call_type, model, latency_ms, "retried", attempt=attempt,
                    error_type=type(exc).__name__,
                )
                time.sleep(base_delay * attempt)
                continue
            log_call(
                call_type, model, latency_ms, "error", attempt=attempt,
                error_type=type(exc).__name__,
            )
            raise GatewayAllProvidersFailedError(
                f"{call_type}: all {max_attempts} attempts failed for "
                f"model={model!r}",
                last_error=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - intentionally typed+wrapped, not swallowed
            latency_ms = (time.monotonic() - start) * 1000
            log_call(
                call_type, model, latency_ms, "error", attempt=attempt,
                error_type=type(exc).__name__,
            )
            raise GatewayError(
                f"{call_type}: unexpected error calling model={model!r}: {exc}"
            ) from exc
        else:
            latency_ms = (time.monotonic() - start) * 1000
            outcome = "success" if attempt == 1 else "retried"
            # Final successful attempt is logged as "success" so Phase 7 can
            # distinguish "succeeded after retry" (multiple "retried" lines
            # followed by one "success" line) from a clean first-try success.
            log_call(call_type, model, latency_ms, "success", attempt=attempt)
            return result

    # Unreachable: the loop above always returns or raises. Kept only to
    # satisfy static analysis that this function never falls through silently.
    raise GatewayAllProvidersFailedError(
        f"{call_type}: exhausted retries for model={model!r}", last_error=last_error
    )


async def acall_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    call_type: str,
    model: str,
    max_attempts: int = 3,
    base_delay: float = 0.01,
) -> T:
    """Async counterpart of :func:`call_with_retry`. ``fn`` is a
    zero-argument callable returning an awaitable (e.g.
    ``lambda: litellm.acompletion(...)``).
    """
    import asyncio

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        try:
            result = await fn()
        except litellm.AuthenticationError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            log_call(
                call_type, model, latency_ms, "error", attempt=attempt,
                error_type="AuthenticationError",
            )
            raise GatewayAuthError(
                f"{call_type}: authentication failed against provider for "
                f"model={model!r}"
            ) from exc
        except TRANSIENT_EXCEPTIONS as exc:
            latency_ms = (time.monotonic() - start) * 1000
            last_error = exc
            if attempt < max_attempts:
                log_call(
                    call_type, model, latency_ms, "retried", attempt=attempt,
                    error_type=type(exc).__name__,
                )
                await asyncio.sleep(base_delay * attempt)
                continue
            log_call(
                call_type, model, latency_ms, "error", attempt=attempt,
                error_type=type(exc).__name__,
            )
            raise GatewayAllProvidersFailedError(
                f"{call_type}: all {max_attempts} attempts failed for "
                f"model={model!r}",
                last_error=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - intentionally typed+wrapped, not swallowed
            latency_ms = (time.monotonic() - start) * 1000
            log_call(
                call_type, model, latency_ms, "error", attempt=attempt,
                error_type=type(exc).__name__,
            )
            raise GatewayError(
                f"{call_type}: unexpected error calling model={model!r}: {exc}"
            ) from exc
        else:
            latency_ms = (time.monotonic() - start) * 1000
            log_call(call_type, model, latency_ms, "success", attempt=attempt)
            return result

    raise GatewayAllProvidersFailedError(
        f"{call_type}: exhausted retries for model={model!r}", last_error=last_error
    )
