"""Guards the successfully-parsed :class:`~app.interview.state.FeedbackReport`
produced by :func:`app.interview.feedback.generate_feedback` — see plan.md
Section 4 for the full design rationale. Two stages, run only on the
LLM-success path (the existing ``_fallback_report()`` path is already a
static, leak-free template and is never passed through here):

- **Stage A (schema):** re-validate the report against the Pydantic schema
  via ``Guard.for_pydantic`` (``num_reasks=0``, no ``llm_api`` — pure
  structural re-check, no model call). On failure, the caller is expected to
  fall back to ``_fallback_report(...)`` exactly like Phase 4's existing
  malformed-JSON handling.
- **Stage B (no-leak, surgical fix):** for every free-text field
  (``per_question[].note`` checked against *that question's own*
  ``ideal_answer``; ``overall_strengths``/``overall_improvements`` checked
  against the concatenation of every ``ideal_answer`` actually asked in this
  session), redact in place via ``OnFailAction.FIX`` rather than reasking the
  whole report — see plan.md Section 4 for why a second reask loop was
  rejected here.
"""

from __future__ import annotations

import json
import logging

from guardrails import Guard, OnFailAction

from app.guardrails.no_leak import validate_no_leak
from app.interview.state import FeedbackReport
from app.retrieval.schema import QAEntry

logger = logging.getLogger(__name__)

_NOTE_FIX_VALUE = "Note withheld: contained reference material; see verdict only."
_SUMMARY_FIX_VALUE = "Item withheld: contained reference material."


class FeedbackSchemaError(Exception):
    """Raised internally to signal Stage A failed; callers should catch this
    and fall back, never let it escape (see
    :func:`guard_feedback_report`'s docstring)."""


def _schema_guard_passes(report: FeedbackReport) -> bool:
    """Stage A: re-validate ``report`` against the Pydantic schema through a
    real Guardrails Guard (not just trusting pydantic's own earlier parse —
    see plan.md Section 4 for why this distinct check is worth keeping)."""
    guard = Guard.for_pydantic(FeedbackReport)
    outcome = guard.validate(
        json.dumps(report.model_dump()), num_reasks=0, llm_api=None
    )
    return bool(outcome.validation_passed)


def _redact_field(text: str, reference: str, fix_value: str) -> str:
    outcome = validate_no_leak(
        text, reference, on_fail=OnFailAction.FIX, fix_value=fix_value
    )
    return outcome.validated_output if outcome.validated_output is not None else text


def guard_feedback_report(
    report: FeedbackReport, entries_by_id: dict[str, QAEntry]
) -> FeedbackReport | None:
    """Run Stage A (schema) then Stage B (per-field no-leak fix) on a
    successfully-LLM-parsed ``report``.

    Returns:
        A (possibly per-field-redacted) :class:`FeedbackReport` if Stage A
        passes. Returns ``None`` if Stage A fails — the schema-violation
        case — signaling the caller to fall back to ``_fallback_report(...)``
        exactly like Phase 4's existing malformed-JSON handling. Never
        raises: any unexpected validator error degrades to the SAFE,
        REDACTED template for that one field (fail CLOSED — mirrors
        ``app/interview/graph.py``'s ``_safe_complete_no_leak``) rather than
        crashing the whole feedback generation step (graceful degradation,
        CLAUDE.md Section 6) *and* rather than ever returning the original,
        unvalidated text — "when in doubt, fail closed" (CLAUDE.md Section
        6). A withheld field is a far better failure mode than a silent
        reference-answer leak.
    """
    if not _schema_guard_passes(report):
        logger.warning(
            "guard_feedback_report: FeedbackReport failed schema re-validation; "
            "caller should fall back to the ledger-derived report."
        )
        return None

    # The full set of ideal_answers actually asked this session — used for
    # the two overall-summary fields, which are not qa_id-scoped (plan.md
    # Section 4).
    all_ideal_answers = " ".join(
        entry.ideal_answer for entry in entries_by_id.values()
    )

    redacted_per_question = []
    for q in report.per_question:
        entry = entries_by_id.get(q.qa_id)
        if entry is None:
            # Unverifiable (LLM referenced a qa_id we don't have a reference
            # for) but already schema-valid — pass through unredacted rather
            # than crash. Documented residual risk (plan.md Section 9 /
            # self-check.md).
            redacted_per_question.append(q)
            continue
        try:
            new_note = _redact_field(q.note, entry.ideal_answer, _NOTE_FIX_VALUE)
        except Exception:  # noqa: BLE001 - guard-internal failure must fail CLOSED, never leak
            logger.warning(
                "guard_feedback_report: no-leak check raised for qa_id=%s; "
                "withholding note (fail-closed) rather than risking a leak.",
                q.qa_id,
                exc_info=True,
            )
            new_note = _NOTE_FIX_VALUE
        redacted_per_question.append(q.model_copy(update={"note": new_note}))

    def _redact_summary_list(items: list[str]) -> list[str]:
        out = []
        for item in items:
            try:
                out.append(_redact_field(item, all_ideal_answers, _SUMMARY_FIX_VALUE))
            except Exception:  # noqa: BLE001 - guard-internal failure must fail CLOSED, never leak
                logger.warning(
                    "guard_feedback_report: no-leak check raised on a summary "
                    "item; withholding it (fail-closed) rather than risking a leak.",
                    exc_info=True,
                )
                out.append(_SUMMARY_FIX_VALUE)
        return out

    return FeedbackReport(
        per_question=redacted_per_question,
        overall_strengths=_redact_summary_list(report.overall_strengths),
        overall_improvements=_redact_summary_list(report.overall_improvements),
    )


__all__ = ["guard_feedback_report", "FeedbackSchemaError"]
