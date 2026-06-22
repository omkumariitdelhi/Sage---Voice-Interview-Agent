"""Runnable example: demonstrates the Phase 5 no-leak guardrail catching a
deliberately leaky mocked gateway response, both for a live interviewer turn
and for an end-of-interview feedback note.

Run from the project root (module form, matching this project's other
examples):

    .venv\\Scripts\\python.exe -m examples.run_guardrails_example

Makes zero real network calls — both gateway call sites
(``app.interview.graph.complete`` for interviewer turns,
``app.interview.feedback.complete`` for the feedback report) are
monkeypatched with a fake that returns the literal reference answer
verbatim on the first attempt, exactly the failure mode the guard exists to
catch.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from app.guardrails.feedback_guard import guard_feedback_report
from app.guardrails.no_leak import is_leaking
from app.interview.graph import build_graph
from app.interview.state import FeedbackReport, QuestionFeedback, initial_state
from app.retrieval.store import get_by_sequence_index


def demo_live_turn_guard() -> None:
    print("=== Demo 1: live interviewer-turn no-leak guard ===")
    compiled, total_questions, max_follow_ups = build_graph()
    entry = get_by_sequence_index(0)
    print(f"Question q-index 0 ({entry.id}): {entry.question!r}")
    print(f"Its reference ideal_answer (must never reach the candidate verbatim):")
    print(f"  {entry.ideal_answer!r}\n")

    leak_then_safe = iter(
        [
            entry.ideal_answer,  # deliberate leak on the first gateway call
            "Let's think about it differently — what's the first step you'd take?",
        ]
    )

    def _leaky_then_safe_complete(messages, **kwargs):
        return next(leak_then_safe)

    with patch("app.interview.graph.complete", side_effect=_leaky_then_safe_complete), patch(
        "app.interview.evaluation.complete",
        return_value=json.dumps({"verdict": "strong", "missed_key_points": []}),
    ):
        config = {"configurable": {"thread_id": "guardrails-example-session"}}
        state = initial_state(total_questions=total_questions, max_follow_ups=max_follow_ups)
        result = compiled.invoke(state, config)

    interviewer_turn = next(
        t for t in result["transcript"] if t["speaker"] == "interviewer" and t["qa_id"]
    )
    print(f"Mocked gateway's FIRST (leaky) response: {entry.ideal_answer!r}")
    print(f"What actually reached the transcript:    {interviewer_turn['text']!r}")
    assert entry.ideal_answer not in interviewer_turn["text"], (
        "GUARD FAILED: the literal ideal_answer leaked into the transcript."
    )
    print("PASS: the literal reference answer never reached the candidate.\n")


def demo_feedback_note_guard() -> None:
    print("=== Demo 2: feedback-note no-leak guard ===")
    entry = get_by_sequence_index(0)
    leaky_report = FeedbackReport(
        per_question=[
            QuestionFeedback(
                qa_id=entry.id,
                topic=entry.topic,
                verdict="strong",
                note=f"Excellent — you basically said: {entry.ideal_answer}",
            )
        ],
        overall_strengths=["Clear communicator"],
        overall_improvements=["None"],
    )
    print(f"LLM-authored note (deliberately leaky): {leaky_report.per_question[0].note!r}")

    guarded = guard_feedback_report(leaky_report, {entry.id: entry})
    assert guarded is not None
    print(f"Guarded note actually returned:         {guarded.per_question[0].note!r}")
    assert entry.ideal_answer not in guarded.per_question[0].note
    print("PASS: the literal reference answer never reached the feedback report.\n")


def demo_is_leaking_directly() -> None:
    print("=== Demo 3: is_leaking() — exact vs paraphrase vs near-verbatim ===")
    entry = get_by_sequence_index(0)
    paraphrase = (
        "I'd first nail down exactly what's broken and what constraints I'm "
        "working under, then split it into smaller pieces, weigh a couple "
        "of different approaches against each other, check my fix actually "
        "works with tests, and think about what I'd tweak next time."
    )
    print(f"Literal ideal_answer vs itself  -> is_leaking = {is_leaking(entry.ideal_answer, entry.ideal_answer)}")
    print(f"Genuine original paraphrase     -> is_leaking = {is_leaking(paraphrase, entry.ideal_answer)}")


if __name__ == "__main__":
    demo_live_turn_guard()
    demo_feedback_note_guard()
    demo_is_leaking_directly()
