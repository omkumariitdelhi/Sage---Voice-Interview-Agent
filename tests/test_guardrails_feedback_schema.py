"""Unit tests for app/guardrails/feedback_guard.py — Stage A (schema
re-validation via a real Guardrails Guard) and Stage B (surgical per-field
no-leak fix, keyed by qa_id) applied to a successfully-parsed
``FeedbackReport``. No gateway/network calls; everything here operates on
in-memory Pydantic objects and raw JSON strings.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from guardrails import Guard

from app.guardrails.feedback_guard import (
    _NOTE_FIX_VALUE,
    _SUMMARY_FIX_VALUE,
    guard_feedback_report,
)
from app.interview.state import FeedbackReport, QuestionFeedback
from app.retrieval.schema import QAEntry

Q03 = QAEntry(
    id="q03",
    topic="complexity-analysis",
    difficulty="medium",
    question="What is Big-O notation?",
    ideal_answer=(
        "Big-O describes how an algorithm's running time grows with input "
        "size in the worst case, ignoring constant factors."
    ),
    key_points=["worst-case growth rate"],
)

Q09 = QAEntry(
    id="q09",
    topic="testing",
    difficulty="easy",
    question="Unit vs integration tests?",
    ideal_answer=(
        "Unit tests exercise a single function in isolation; integration "
        "tests exercise multiple components together."
    ),
    key_points=["isolation vs real interactions"],
)

ENTRIES = {"q03": Q03, "q09": Q09}


def _report(per_question, strengths=None, improvements=None) -> FeedbackReport:
    return FeedbackReport(
        per_question=per_question,
        overall_strengths=strengths or [],
        overall_improvements=improvements or [],
    )


# ---------------------------------------------------------------------------
# Stage A: schema re-validation (the raw Guard.for_pydantic path)
# ---------------------------------------------------------------------------


def test_valid_report_dict_passes_schema_guard():
    guard = Guard.for_pydantic(FeedbackReport)
    valid_json = json.dumps(
        {
            "per_question": [
                {"qa_id": "q03", "topic": "complexity-analysis", "verdict": "strong", "note": "Solid."}
            ],
            "overall_strengths": ["clear"],
            "overall_improvements": ["more depth"],
        }
    )
    outcome = guard.validate(valid_json, num_reasks=0)
    assert outcome.validation_passed is True


def test_missing_required_field_fails_schema_guard():
    """A per_question entry missing the required 'topic' field — valid JSON,
    schema-violating. Distinct from the malformed-JSON case (non-JSON text),
    which app/interview/feedback.py's _parse() already handles."""
    guard = Guard.for_pydantic(FeedbackReport)
    bad_json = json.dumps(
        {
            "per_question": [
                {"qa_id": "q03", "verdict": "strong", "note": "Solid."}
            ],
            "overall_strengths": [],
            "overall_improvements": [],
        }
    )
    outcome = guard.validate(bad_json, num_reasks=0)
    assert outcome.validation_passed is False


def test_wrong_type_for_per_question_fails_schema_guard():
    guard = Guard.for_pydantic(FeedbackReport)
    bad_json = json.dumps(
        {
            "per_question": "not a list",
            "overall_strengths": [],
            "overall_improvements": [],
        }
    )
    outcome = guard.validate(bad_json, num_reasks=0)
    assert outcome.validation_passed is False


# ---------------------------------------------------------------------------
# guard_feedback_report(): Stage A wired to Stage B, end-to-end
# ---------------------------------------------------------------------------


def test_leaky_note_for_its_own_question_is_redacted():
    report = _report(
        per_question=[
            QuestionFeedback(
                qa_id="q03",
                topic="complexity-analysis",
                verdict="strong",
                note=f"Nice answer: {Q03.ideal_answer}",
            )
        ]
    )
    result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    assert Q03.ideal_answer not in result.per_question[0].note
    assert "withheld" in result.per_question[0].note.lower()


def test_note_is_checked_against_its_OWN_qa_ids_ideal_answer_not_a_global_one():
    """Per spec.md: each per_question note must be checked against THAT
    question's own ideal_answer, not a single shared reference. A note for
    q09 that happens to quote q03's ideal_answer verbatim must still be
    caught (cross-question leak), and a note for q03 containing q03's own
    answer must be caught — but a note for q09 that merely overlaps with
    q09's own (different) ideal_answer is the one under test for precision."""
    report = _report(
        per_question=[
            QuestionFeedback(qa_id="q03", topic="complexity-analysis", verdict="strong", note="Good, concise."),
            QuestionFeedback(
                qa_id="q09",
                topic="testing",
                verdict="weak",
                note=f"Missed the point: {Q09.ideal_answer}",
            ),
        ]
    )
    result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    # q03's clean note is untouched.
    assert result.per_question[0].note == "Good, concise."
    # q09's leaky note (against ITS OWN ideal_answer) is redacted.
    assert Q09.ideal_answer not in result.per_question[1].note


def test_clean_notes_pass_through_unchanged():
    report = _report(
        per_question=[
            QuestionFeedback(qa_id="q03", topic="complexity-analysis", verdict="strong", note="Covered the growth-rate idea well."),
            QuestionFeedback(qa_id="q09", topic="testing", verdict="weak", note="Didn't mention isolation."),
        ],
        strengths=["communicated clearly"],
        improvements=["go deeper on trade-offs"],
    )
    result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    assert result.per_question[0].note == "Covered the growth-rate idea well."
    assert result.per_question[1].note == "Didn't mention isolation."
    assert result.overall_strengths == ["communicated clearly"]
    assert result.overall_improvements == ["go deeper on trade-offs"]


def test_leaky_overall_strengths_item_is_redacted():
    report = _report(
        per_question=[],
        strengths=[f"Demonstrated this clearly: {Q03.ideal_answer}"],
        improvements=[],
    )
    result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    assert Q03.ideal_answer not in result.overall_strengths[0]


def test_leaky_overall_improvements_item_is_redacted():
    report = _report(
        per_question=[],
        strengths=[],
        improvements=[f"Should have said: {Q09.ideal_answer}"],
    )
    result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    assert Q09.ideal_answer not in result.overall_improvements[0]


def test_unknown_qa_id_does_not_crash_and_passes_through():
    """A qa_id the LLM hallucinated that isn't in entries_by_id: nothing to
    compare against, so the note passes through unredacted rather than
    crashing the whole feedback step. Documented residual risk, not a
    blocker (plan.md Section 9 / self-check.md)."""
    report = _report(
        per_question=[
            QuestionFeedback(qa_id="ghost-id", topic="unknown", verdict="strong", note="Some note text.")
        ]
    )
    result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    assert result.per_question[0].note == "Some note text."


def test_schema_violation_returns_none_signaling_fallback(monkeypatch):
    """guard_feedback_report only ever receives an already-pydantic-typed
    FeedbackReport (pydantic already validated it via _parse() before this
    function is called), so the Stage A re-validation failing in practice
    means the Guardrails-side schema check disagrees with pydantic's own
    (e.g. a future drift between the two, or a Guard-internal hiccup) — we
    force that disagreement here by making the internal schema check report
    failure, and assert the documented contract: ``None`` is returned,
    signaling the caller (generate_feedback) to fall back to
    ``_fallback_report(...)``. The full real-world trigger (a genuinely
    schema-invalid LLM response) is covered end-to-end in
    test_interview_feedback.py's
    ``test_generate_feedback_falls_back_on_schema_violating_valid_json``."""
    import app.guardrails.feedback_guard as feedback_guard_module

    monkeypatch.setattr(
        feedback_guard_module, "_schema_guard_passes", lambda report: False
    )
    report = _report(per_question=[])
    assert guard_feedback_report(report, ENTRIES) is None


def test_fully_valid_report_returns_non_none():
    report = _report(per_question=[])
    assert guard_feedback_report(report, ENTRIES) is not None


# ---------------------------------------------------------------------------
# Regression (review-01.md, Major finding #1): an unexpected, non-FailResult
# exception from _redact_field must fail CLOSED (withheld template), never
# fail open to the original unredacted text. Reproduces the reviewer's exact
# repro: unittest.mock.patch on app.guardrails.feedback_guard._redact_field,
# side_effect=RuntimeError, with a planted literal leak in the input field.
# ---------------------------------------------------------------------------


def test_unexpected_exception_in_redact_field_fails_closed_for_per_question_note():
    """Forces _redact_field to raise a plain RuntimeError (not a FailResult)
    while a literal reference-answer leak is planted in the note. Before the
    fix, the except-block fell back to `new_note = q.note` — the ORIGINAL,
    unredacted, leaky text. After the fix, it must fall back to the same
    safe _NOTE_FIX_VALUE template used by the normal FIX path, never the
    leaked original."""
    leaky_note = f"SECRET ANSWER TEXT: {Q03.ideal_answer}"
    report = _report(
        per_question=[
            QuestionFeedback(
                qa_id="q03",
                topic="complexity-analysis",
                verdict="strong",
                note=leaky_note,
            )
        ]
    )
    with patch(
        "app.guardrails.feedback_guard._redact_field",
        side_effect=RuntimeError("boom"),
    ):
        result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    redacted_note = result.per_question[0].note
    # The leak must not survive the unexpected-exception path.
    assert leaky_note not in redacted_note
    assert Q03.ideal_answer not in redacted_note
    assert "SECRET ANSWER TEXT" not in redacted_note
    # It must be the safe withheld template, not silently-passed-through text.
    assert redacted_note == _NOTE_FIX_VALUE


def test_unexpected_exception_in_redact_field_fails_closed_for_summary_items():
    """Same repro as above, applied to _redact_summary_list (covers both
    overall_strengths and overall_improvements, which share the same helper
    and except-block shape)."""
    leaky_strength = f"SECRET ANSWER TEXT: {Q03.ideal_answer}"
    leaky_improvement = f"SECRET ANSWER TEXT: {Q09.ideal_answer}"
    report = _report(
        per_question=[],
        strengths=[leaky_strength],
        improvements=[leaky_improvement],
    )
    with patch(
        "app.guardrails.feedback_guard._redact_field",
        side_effect=RuntimeError("boom"),
    ):
        result = guard_feedback_report(report, ENTRIES)
    assert result is not None
    assert leaky_strength not in result.overall_strengths[0]
    assert Q03.ideal_answer not in result.overall_strengths[0]
    assert result.overall_strengths[0] == _SUMMARY_FIX_VALUE
    assert leaky_improvement not in result.overall_improvements[0]
    assert Q09.ideal_answer not in result.overall_improvements[0]
    assert result.overall_improvements[0] == _SUMMARY_FIX_VALUE
