"""Unit tests for app/interview/feedback.py and app/interview/evaluation.py —
isolated from the full graph (see test_interview_graph.py for end-to-end
coverage). Every gateway call is mocked; zero network calls.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.gateway.exceptions import GatewayAllProvidersFailedError
from app.interview.evaluation import evaluate_answer
from app.interview.feedback import generate_feedback
from app.interview.state import FeedbackReport, TurnRecord
from app.retrieval.schema import QAEntry

SAMPLE_ENTRY = QAEntry(
    id="q01",
    topic="behavioral",
    difficulty="easy",
    question="Tell me about a challenge.",
    ideal_answer="SECRET_REFERENCE_TEXT must never leak.",
    key_points=["clarified scope", "validated solution"],
)


def _transcript_fixture() -> list[TurnRecord]:
    return [
        TurnRecord(speaker="interviewer", text="Hi, let's start.", qa_id=None, verdict=None),
        TurnRecord(speaker="interviewer", text="Tell me about a challenge.", qa_id="q01", verdict=None),
        TurnRecord(speaker="candidate", text="I clarified scope and validated my fix.", qa_id="q01", verdict=None),
        TurnRecord(speaker="interviewer", text="What's an array vs linked list?", qa_id="q02", verdict=None),
        TurnRecord(speaker="candidate", text="Arrays are contiguous.", qa_id="q02", verdict=None),
    ]


# ---------------------------------------------------------------------------
# evaluate_answer
# ---------------------------------------------------------------------------


def test_evaluate_answer_parses_well_formed_json():
    with patch(
        "app.interview.evaluation.complete",
        return_value=json.dumps(
            {"verdict": "strong", "missed_key_points": [], "reasoning": "good"}
        ),
    ):
        result = evaluate_answer(SAMPLE_ENTRY, "a great answer", [])

    assert result.verdict == "strong"
    assert result.missed_key_points == []


def test_evaluate_answer_retries_once_on_malformed_json_then_succeeds():
    responses = iter(["not json at all", json.dumps({"verdict": "weak", "missed_key_points": ["x"]})])
    with patch("app.interview.evaluation.complete", side_effect=lambda *a, **k: next(responses)) as mock_complete:
        result = evaluate_answer(SAMPLE_ENTRY, "an answer", [])

    assert result.verdict == "weak"
    assert result.missed_key_points == ["x"]
    assert mock_complete.call_count == 2  # one retry, exactly as spec'd


def test_evaluate_answer_falls_back_to_weak_after_persistent_malformed_json():
    with patch("app.interview.evaluation.complete", return_value="still not json"):
        result = evaluate_answer(SAMPLE_ENTRY, "an answer", [])

    assert result.verdict == "weak"
    assert result.reasoning == "malformed-llm-output-fallback"


def test_evaluate_answer_falls_back_to_weak_on_gateway_error():
    with patch(
        "app.interview.evaluation.complete",
        side_effect=GatewayAllProvidersFailedError("boom"),
    ):
        result = evaluate_answer(SAMPLE_ENTRY, "an answer", [])

    assert result.verdict == "weak"
    assert result.reasoning == "gateway-call-failed-fallback"


def test_evaluate_answer_prompt_never_leaks_ideal_answer_into_a_redirect_or_followup():
    """Sanity check the prompt builders evaluate_answer uses internally
    don't accidentally route ideal_answer into candidate-facing text — the
    grading prompt itself is allowed to contain it (never shown to the
    candidate), but this asserts evaluate_answer's RETURN VALUE (what the
    rest of the graph consumes) never echoes it."""
    with patch(
        "app.interview.evaluation.complete",
        return_value=json.dumps(
            {"verdict": "wrong", "missed_key_points": ["clarified scope"]}
        ),
    ):
        result = evaluate_answer(SAMPLE_ENTRY, "an answer", [])

    assert SAMPLE_ENTRY.ideal_answer not in result.reasoning
    assert all(SAMPLE_ENTRY.ideal_answer not in pt for pt in result.missed_key_points)


# ---------------------------------------------------------------------------
# generate_feedback
# ---------------------------------------------------------------------------


def test_generate_feedback_produces_structured_report_covering_every_question():
    transcript = _transcript_fixture()
    entries_by_id = {
        "q01": SAMPLE_ENTRY,
        "q02": QAEntry(
            id="q02",
            topic="data-structures",
            difficulty="easy",
            question="Array vs linked list?",
            ideal_answer="...",
            key_points=["O(1) access"],
        ),
    }
    question_verdicts = [("q01", "strong"), ("q02", "weak")]

    with patch(
        "app.interview.feedback.complete",
        return_value=json.dumps(
            {
                "per_question": [
                    {"qa_id": "q01", "topic": "behavioral", "verdict": "strong", "note": "Good structure."},
                    {"qa_id": "q02", "topic": "data-structures", "verdict": "weak", "note": "Missed cache point."},
                ],
                "overall_strengths": ["clear communication"],
                "overall_improvements": ["more depth"],
            }
        ),
    ):
        report = generate_feedback(transcript, entries_by_id, question_verdicts)

    assert isinstance(report, FeedbackReport)
    assert {q.qa_id for q in report.per_question} == {"q01", "q02"}
    assert report.overall_strengths == ["clear communication"]
    assert report.overall_improvements == ["more depth"]


def test_generate_feedback_falls_back_to_ledger_derived_report_on_gateway_error():
    transcript = _transcript_fixture()
    entries_by_id = {"q01": SAMPLE_ENTRY}
    question_verdicts = [("q01", "strong")]

    with patch(
        "app.interview.feedback.complete",
        side_effect=GatewayAllProvidersFailedError("boom"),
    ):
        report = generate_feedback(transcript, entries_by_id, question_verdicts)

    assert isinstance(report, FeedbackReport)
    assert len(report.per_question) == 1
    assert report.per_question[0].qa_id == "q01"
    assert report.per_question[0].verdict == "strong"
    assert "fallback" in report.overall_improvements[0].lower()


def test_generate_feedback_falls_back_after_persistent_malformed_json():
    transcript = _transcript_fixture()
    entries_by_id = {"q01": SAMPLE_ENTRY}
    question_verdicts = [("q01", "weak")]

    with patch("app.interview.feedback.complete", return_value="not json"):
        report = generate_feedback(transcript, entries_by_id, question_verdicts)

    assert isinstance(report, FeedbackReport)
    assert report.per_question[0].verdict == "weak"


def test_generate_feedback_retries_once_then_succeeds():
    responses = iter(
        [
            "garbage",
            json.dumps(
                {
                    "per_question": [
                        {"qa_id": "q01", "topic": "behavioral", "verdict": "strong", "note": "ok"}
                    ],
                    "overall_strengths": [],
                    "overall_improvements": [],
                }
            ),
        ]
    )
    with patch(
        "app.interview.feedback.complete", side_effect=lambda *a, **k: next(responses)
    ) as mock_complete:
        report = generate_feedback(_transcript_fixture(), {"q01": SAMPLE_ENTRY}, [("q01", "strong")])

    assert mock_complete.call_count == 2
    assert report.per_question[0].qa_id == "q01"


# ---------------------------------------------------------------------------
# Phase 5: guard_feedback_report wiring (no-leak + schema re-validation)
# ---------------------------------------------------------------------------


def test_generate_feedback_redacts_a_leaky_per_question_note():
    """Gateway mocked to return valid JSON whose per_question[0].note
    contains q01's own literal ideal_answer. The FeedbackReport returned by
    generate_feedback() must NOT contain that literal text in the note —
    proves the guard is wired into the real generate_feedback() call site,
    not just unit-tested against guard_feedback_report() in isolation."""
    transcript = _transcript_fixture()
    entries_by_id = {"q01": SAMPLE_ENTRY}
    question_verdicts = [("q01", "strong")]

    with patch(
        "app.interview.feedback.complete",
        return_value=json.dumps(
            {
                "per_question": [
                    {
                        "qa_id": "q01",
                        "topic": "behavioral",
                        "verdict": "strong",
                        "note": f"Nailed it: {SAMPLE_ENTRY.ideal_answer}",
                    }
                ],
                "overall_strengths": ["clear communication"],
                "overall_improvements": ["more depth"],
            }
        ),
    ):
        report = generate_feedback(transcript, entries_by_id, question_verdicts)

    assert isinstance(report, FeedbackReport)
    assert SAMPLE_ENTRY.ideal_answer not in report.per_question[0].note


def test_generate_feedback_falls_back_on_schema_violating_valid_json():
    """Gateway mocked to return JSON that is syntactically valid but
    violates the FeedbackReport schema (per_question entry missing the
    required 'topic' field). Distinct from
    test_generate_feedback_falls_back_after_persistent_malformed_json, which
    feeds non-JSON text — this feeds JSON that parses but fails
    pydantic/Guardrails schema validation. Must degrade to the
    ledger-derived fallback report, never propagate the invalid object."""
    transcript = _transcript_fixture()
    entries_by_id = {"q01": SAMPLE_ENTRY}
    question_verdicts = [("q01", "weak")]

    # This raw text is valid JSON but `per_question[0]` is missing the
    # required `topic` field, so json.loads() succeeds (passing _parse()'s
    # json.JSONDecodeError guard) but FeedbackReport.model_validate() inside
    # _parse() raises pydantic.ValidationError, returning None from _parse().
    # Since _parse() already catches this on the FIRST attempt, the retry
    # path fires (matching the existing malformed-JSON retry flow) — feed
    # the same schema-violating shape on the retry too, so the retry also
    # fails to parse and the ledger-derived fallback is used. This proves
    # call site 2's overall graceful-degradation behavior is preserved
    # end-to-end for a schema-violating (not just non-JSON) response.
    bad_shape = json.dumps(
        {
            "per_question": [{"qa_id": "q01", "verdict": "weak", "note": "Missing topic field."}],
            "overall_strengths": [],
            "overall_improvements": [],
        }
    )
    with patch("app.interview.feedback.complete", return_value=bad_shape):
        report = generate_feedback(transcript, entries_by_id, question_verdicts)

    assert isinstance(report, FeedbackReport)
    # Fell back to the ledger-derived report (same shape as the existing
    # malformed-JSON fallback test asserts).
    assert report.per_question[0].qa_id == "q01"
    assert report.per_question[0].verdict == "weak"
    assert "fallback" in report.overall_improvements[0].lower()


def test_generate_feedback_leak_check_is_scoped_per_qa_id():
    """Two questions in one report: q01's note leaks q01's OWN ideal_answer
    (must be redacted), q02's note is a clean, original note (must pass
    through unchanged) — proves the per-qa_id scoping (not a single global
    reference) survives the real generate_feedback() call site."""
    transcript = _transcript_fixture()
    q02_entry = QAEntry(
        id="q02",
        topic="data-structures",
        difficulty="easy",
        question="Array vs linked list?",
        ideal_answer="Arrays give O(1) random access; linked lists give O(1) insertion at a known node.",
        key_points=["O(1) access"],
    )
    entries_by_id = {"q01": SAMPLE_ENTRY, "q02": q02_entry}
    question_verdicts = [("q01", "strong"), ("q02", "weak")]

    with patch(
        "app.interview.feedback.complete",
        return_value=json.dumps(
            {
                "per_question": [
                    {
                        "qa_id": "q01",
                        "topic": "behavioral",
                        "verdict": "strong",
                        "note": f"Exactly matches the reference: {SAMPLE_ENTRY.ideal_answer}",
                    },
                    {
                        "qa_id": "q02",
                        "topic": "data-structures",
                        "verdict": "weak",
                        "note": "Mentioned arrays but not linked lists.",
                    },
                ],
                "overall_strengths": [],
                "overall_improvements": [],
            }
        ),
    ):
        report = generate_feedback(transcript, entries_by_id, question_verdicts)

    notes_by_qid = {q.qa_id: q.note for q in report.per_question}
    assert SAMPLE_ENTRY.ideal_answer not in notes_by_qid["q01"]
    assert notes_by_qid["q02"] == "Mentioned arrays but not linked lists."
