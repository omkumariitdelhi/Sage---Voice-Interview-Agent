"""Typed exceptions raised by the gateway.

Callers elsewhere in the app (LangGraph nodes in Phase 4, FastAPI routes in
Phase 6) should depend on *these* exceptions, not on LiteLLM's internal
exception types directly. That indirection insulates the rest of the app from
a future LiteLLM exception-hierarchy change and gives every call type
(llm/stt/tts) one consistent failure surface.
"""

from __future__ import annotations


class GatewayError(Exception):
    """Base class for all gateway errors."""


class GatewayAuthError(GatewayError):
    """Raised for missing/invalid credentials. Never retried — a bad key
    will not become a good key on the next attempt."""


class GatewayTransientError(GatewayError):
    """Raised for a single transient failure (rate limit, connection error,
    timeout). Internal to the retry loop; usually caught and retried rather
    than propagated, but exposed for callers that want to handle a single
    attempt's failure explicitly."""


class GatewayAllProvidersFailedError(GatewayError):
    """Raised when every retry attempt has been exhausted without success.

    Carries the last underlying exception in ``last_error`` for diagnostics
    and logging.
    """

    def __init__(self, message: str, last_error: Exception | None = None) -> None:
        super().__init__(message)
        self.last_error = last_error
