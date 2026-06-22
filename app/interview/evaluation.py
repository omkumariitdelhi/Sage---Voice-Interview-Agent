"""Grades a candidate's latest answer against the current question.

Calls the LLM gateway (never ``litellm`` directly — see
``app/gateway/llm.py``'s own docstring for why that chokepoint exists) with a
JSON-mode prompt, parses the result with Pydantic, and degrades gracefully
(never raises) on malformed output or a gateway failure — the interview must
be able to continue/finish even if one grading call fails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from app.gateway.exceptions import GatewayError
from app.gateway.llm import complete
from app.interview.prompts import evaluation_prompt
from app.interview.state import AnswerVerdict, TurnRecord
from app.json_mode import strip_json_fences
from app.retrieval.schema import QAEntry

logger = logging.getLogger(__name__)

_FALLBACK_REASONING = "malformed-llm-output-fallback"
_GATEWAY_FAILURE_REASONING = "gateway-call-failed-fallback"

# The instruction appended on the single re-ask-on-malformed-JSON retry.
_REASK_INSTRUCTION = (
    "Your previous response was not valid JSON matching the required "
    "schema. Return ONLY the JSON object, with no markdown fences and no "
    "extra commentary."
)


class _EvaluationResultModel(BaseModel):
    """Pydantic parse/validation target for the LLM's JSON verdict."""

    verdict: AnswerVerdict
    missed_key_points: list[str] = []
    reasoning: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    """Public return type of :func:`evaluate_answer`."""

    verdict: AnswerVerdict
    missed_key_points: list[str] = field(default_factory=list)
    reasoning: str = ""


def _parse(raw_text: str) -> _EvaluationResultModel | None:
    """Best-effort JSON parse + Pydantic validation. Returns ``None`` (never
    raises) on any failure so the caller can decide retry-vs-fallback."""
    try:
        data = json.loads(strip_json_fences(raw_text))
    except json.JSONDecodeError:
        return None
    try:
        return _EvaluationResultModel.model_validate(data)
    except ValidationError:
        return None


def evaluate_answer(
    entry: QAEntry, candidate_answer: str, history: list[TurnRecord]
) -> EvaluationResult:
    """Grade ``candidate_answer`` against ``entry`` using the gateway LLM.

    Flow: build the JSON-mode prompt -> call the gateway -> parse. On a
    malformed (non-JSON or schema-invalid) response, retry once with a
    sharper "return ONLY JSON" instruction appended to the same message
    history. If the retry also fails to parse, fall back to a safe "weak"
    verdict and log a warning — this function never raises to its caller.

    A :class:`~app.gateway.exceptions.GatewayError` (auth failure, all
    providers exhausted, etc.) is also caught here and degrades to the same
    safe fallback, with a logged warning, so one bad/unavailable LLM call
    cannot crash the whole interview session.
    """
    messages = evaluation_prompt(entry, candidate_answer, history)

    try:
        raw_text = complete(messages, response_format={"type": "json_object"})
    except GatewayError as exc:
        logger.warning(
            "evaluate_answer: gateway call failed for qa_id=%s (%s); "
            "falling back to verdict='weak'.",
            entry.id,
            exc,
        )
        return EvaluationResult(
            verdict="weak", missed_key_points=[], reasoning=_GATEWAY_FAILURE_REASONING
        )

    parsed = _parse(raw_text)
    if parsed is None:
        # Single re-ask-on-malformed-JSON retry, per spec.
        retry_messages = messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": _REASK_INSTRUCTION},
        ]
        try:
            raw_text_retry = complete(
                retry_messages, response_format={"type": "json_object"}
            )
            parsed = _parse(raw_text_retry)
        except GatewayError as exc:
            logger.warning(
                "evaluate_answer: gateway call failed on retry for qa_id=%s "
                "(%s); falling back to verdict='weak'.",
                entry.id,
                exc,
            )
            return EvaluationResult(
                verdict="weak",
                missed_key_points=[],
                reasoning=_GATEWAY_FAILURE_REASONING,
            )

        if parsed is None:
            logger.warning(
                "evaluate_answer: malformed LLM JSON for qa_id=%s after "
                "retry; falling back to verdict='weak'. Raw text: %r",
                entry.id,
                raw_text_retry,
            )
            return EvaluationResult(
                verdict="weak", missed_key_points=[], reasoning=_FALLBACK_REASONING
            )

    return EvaluationResult(
        verdict=parsed.verdict,
        missed_key_points=list(parsed.missed_key_points),
        reasoning=parsed.reasoning,
    )
