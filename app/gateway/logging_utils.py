"""Structured logging for every gateway call.

Phase 7's latency analysis reads these log lines directly, so the field
names below are a frozen contract — do not rename without updating both this
module's docstring and ``self-check.md``.

Required fields on every line (always present):
    call_type   - one of "llm", "stt", "tts"
    model       - the literal model string passed to LiteLLM
    latency_ms  - float, wall-clock milliseconds for the call attempt/loop
    outcome     - one of "success", "retried", "error"
    attempt     - int, the attempt number this log line reflects (1-based)

Additional fields (``**extra``) may be merged in (e.g. ``error_type`` on a
failure line) but the five fields above are always present and never removed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("app.gateway")

# Library default: don't force a handler/format on the host application.
# A consumer (e.g. the future FastAPI app in Phase 6) configures handlers;
# tests use pytest's `caplog` fixture which captures regardless of handlers.
logger.addHandler(logging.NullHandler())


def log_call(
    call_type: str,
    model: str,
    latency_ms: float,
    outcome: str,
    attempt: int = 1,
    **extra: Any,
) -> None:
    """Emit one structured JSON log line for a gateway call attempt.

    Always logs at INFO for "success"/"retried" outcomes and at WARNING for
    "error" outcomes, so log-level filtering still surfaces failures.
    """
    payload: dict[str, Any] = {
        "call_type": call_type,
        "model": model,
        "latency_ms": round(latency_ms, 3),
        "outcome": outcome,
        "attempt": attempt,
        **extra,
    }
    message = json.dumps(payload, default=str)
    if outcome == "error":
        logger.warning(message)
    else:
        logger.info(message)
