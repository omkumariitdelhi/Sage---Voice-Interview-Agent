"""Generates the final structured feedback report from a session transcript.

Mirrors :mod:`app.interview.evaluation`'s gateway-call + JSON-parse + retry +
graceful-fallback pattern, but produces a whole-session
:class:`~app.interview.state.FeedbackReport` instead of a per-answer verdict.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from app.gateway.exceptions import GatewayError
from app.gateway.llm import complete
from app.guardrails.feedback_guard import guard_feedback_report
from app.interview.prompts import feedback_prompt
from app.interview.state import FeedbackReport, QuestionFeedback, TurnRecord
from app.json_mode import strip_json_fences
from app.retrieval.schema import QAEntry

logger = logging.getLogger(__name__)

_REASK_INSTRUCTION = (
    "Your previous response was not valid JSON matching the required "
    "schema. Return ONLY the JSON object, with no markdown fences and no "
    "extra commentary."
)


def _parse(raw_text: str) -> FeedbackReport | None:
    try:
        data = json.loads(strip_json_fences(raw_text))
    except json.JSONDecodeError:
        return None
    try:
        return FeedbackReport.model_validate(data)
    except ValidationError:
        return None


def _fallback_report(
    question_verdicts: list[tuple[str, str]], entries_by_id: dict[str, QAEntry]
) -> FeedbackReport:
    """Degraded-but-still-structurally-valid report built with no LLM call,
    used when the gateway is unavailable or returns unparseable JSON twice in
    a row. Derives a verdict per question from ``question_verdicts`` — the
    graph's own append-only (qa_id, final_verdict) ledger (see
    ``app/interview/state.py``) — rather than leaving the report empty."""
    per_question = [
        QuestionFeedback(
            qa_id=qid,
            topic=entries_by_id[qid].topic if qid in entries_by_id else "unknown",
            verdict=verdict,  # type: ignore[arg-type]
            note="Automated fallback note: LLM feedback generation unavailable.",
        )
        for qid, verdict in question_verdicts
    ]
    return FeedbackReport(
        per_question=per_question,
        overall_strengths=[],
        overall_improvements=[
            "Feedback generation degraded to a fallback report; no LLM-authored summary available."
        ],
    )


def generate_feedback(
    transcript: list[TurnRecord],
    entries_by_id: dict[str, QAEntry],
    question_verdicts: list[tuple[str, str]],
) -> FeedbackReport:
    """Produce a :class:`FeedbackReport` covering every question actually
    asked in ``transcript``, via the gateway LLM.

    ``question_verdicts`` (the graph's append-only (qa_id, final_verdict)
    ledger) is only used by the no-LLM fallback path — the primary path asks
    the LLM to derive its own per-question verdict/note from the transcript.

    Flow mirrors :func:`app.interview.evaluation.evaluate_answer`: JSON-mode
    prompt -> parse -> one re-ask-on-malformed-JSON retry -> safe structural
    fallback (never raises) on a gateway failure or persistent malformed
    output.

    **Phase 5 addition:** every successfully-*parsed* report (both the
    first-try and the retry-succeeded branches below) is passed through
    :func:`app.guardrails.feedback_guard.guard_feedback_report` before being
    returned — an ADDITIONAL layer on top of this existing flow, not a
    replacement for it. That guard (1) re-validates the report against the
    ``FeedbackReport`` schema via a real Guardrails ``Guard`` (catching
    schema-violating-but-valid JSON, which ``_parse()``'s own
    ``pydantic.ValidationError`` catch already guards against today, but the
    spec wants this enforced as a Guardrails-checked property too) and, if
    that fails, degrades to ``_fallback_report(...)`` exactly like the
    malformed-JSON branches below; (2) on schema success, surgically
    redacts any per-question note or overall-summary item that leaks that
    question's ``ideal_answer``. The ``_fallback_report()`` path itself is
    never passed through the guard — it is already a static, leak-free
    template that does not reference ``ideal_answer`` (see its docstring).
    """
    messages = feedback_prompt(transcript, entries_by_id)

    try:
        raw_text = complete(messages, response_format={"type": "json_object"})
    except GatewayError as exc:
        logger.warning(
            "generate_feedback: gateway call failed (%s); using fallback report.",
            exc,
        )
        return _fallback_report(question_verdicts, entries_by_id)

    parsed = _parse(raw_text)
    if parsed is None:
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
                "generate_feedback: gateway call failed on retry (%s); "
                "using fallback report.",
                exc,
            )
            return _fallback_report(question_verdicts, entries_by_id)

        if parsed is None:
            logger.warning(
                "generate_feedback: malformed LLM JSON after retry; using "
                "fallback report. Raw text: %r",
                raw_text_retry,
            )
            return _fallback_report(question_verdicts, entries_by_id)

    guarded = guard_feedback_report(parsed, entries_by_id)
    if guarded is None:
        logger.warning(
            "generate_feedback: parsed JSON was valid but violated the "
            "FeedbackReport schema under Guardrails re-validation; using "
            "fallback report."
        )
        return _fallback_report(question_verdicts, entries_by_id)
    return guarded
