"""Integration tests proving the no-leak guard added in Phase 5 is actually
WIRED into app/interview/graph.py's interviewer_turn node — not just unit
tested against app/guardrails/no_leak.py in isolation. Drives the real graph
with the gateway mocked at its real call site
(``app.interview.graph.complete``), per this project's existing testing
convention (see tests/test_interview_graph.py's own module docstring).
"""

from __future__ import annotations

import shutil
from unittest.mock import patch

import pytest

from app.interview.graph import build_graph
from app.interview.state import initial_state
from app.retrieval.ingest import build_store
from app.retrieval.store import get_by_sequence_index

THREAD_CONFIG = lambda thread_id: {"configurable": {"thread_id": thread_id}}  # noqa: E731


@pytest.fixture
def ingested_store(tmp_path):
    """Mirrors tests/test_interview_graph.py's own fixture: a fresh Chroma
    store built from the real data/qa_bank.yaml (10 entries) in an isolated
    temp directory."""
    chroma_dir = tmp_path / "chroma_store"
    build_store(chroma_dir=chroma_dir)
    yield chroma_dir
    shutil.rmtree(chroma_dir, ignore_errors=True)


def _eval_json(verdict: str, missed=None) -> str:
    import json

    return json.dumps(
        {"verdict": verdict, "missed_key_points": missed or [], "reasoning": "mock"}
    )


def test_leaky_first_response_is_caught_and_reasked_to_a_safe_rephrase(ingested_store):
    """Gateway mocked to return the literal ideal_answer verbatim on the
    FIRST call, and a safe, non-leaking rephrase on the reask (second) call.
    The transcript must end up with the safe rephrase, not the leak — proves
    the reask path is actually exercised by the real node, not just unit
    tested in isolation."""
    entry = get_by_sequence_index(0, chroma_dir=ingested_store)
    responses = iter(
        [
            entry.ideal_answer,  # first attempt: a deliberate full leak
            "Let's think about it differently — what's the first step you'd take?",
        ]
    )

    def _mock_complete(messages, **kwargs):
        return next(responses)

    with patch("app.interview.graph.complete", side_effect=_mock_complete), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("leak-then-safe-reask")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        result = compiled.invoke(state, config)

        interviewer_turns = [
            t for t in result["transcript"] if t["speaker"] == "interviewer" and t["qa_id"]
        ]
        assert len(interviewer_turns) == 1
        final_text = interviewer_turns[0]["text"]
        assert entry.ideal_answer not in final_text
        assert final_text == "Let's think about it differently — what's the first step you'd take?"


def test_leaky_response_on_both_attempts_falls_back_to_static_key_points_redirect(ingested_store):
    """Gateway mocked to return the literal ideal_answer verbatim on BOTH
    the first call and the reask. The node must never raise and must fall
    back to the static, key_points-only template — never containing the
    ideal_answer text — proving the terminal fallback path is real and
    reachable through the actual node, not just the helper function in
    isolation."""
    entry = get_by_sequence_index(0, chroma_dir=ingested_store)

    def _always_leaky(messages, **kwargs):
        return entry.ideal_answer

    with patch("app.interview.graph.complete", side_effect=_always_leaky), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("leak-then-leak-fallback")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        result = compiled.invoke(state, config)

        interviewer_turns = [
            t for t in result["transcript"] if t["speaker"] == "interviewer" and t["qa_id"]
        ]
        assert len(interviewer_turns) == 1
        final_text = interviewer_turns[0]["text"]
        assert entry.ideal_answer not in final_text
        # The static fallback template references key_points, never the
        # full ideal_answer.
        for kp in entry.key_points:
            assert kp in final_text or "key aspects" in final_text


def test_non_leaky_first_response_is_returned_unchanged_no_reask(ingested_store):
    """The common/happy case: gateway returns a clean response on the first
    call. Must be returned as-is, with no second (reask) gateway call —
    proves the guard does not add a reask round-trip when there's nothing
    to catch."""
    call_count = {"n": 0}

    def _clean_response(messages, **kwargs):
        call_count["n"] += 1
        return "Sure — walk me through a recent challenging problem you tackled."

    with patch("app.interview.graph.complete", side_effect=_clean_response), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("clean-no-reask")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        result = compiled.invoke(state, config)

        interviewer_turns = [
            t for t in result["transcript"] if t["speaker"] == "interviewer" and t["qa_id"]
        ]
        assert interviewer_turns[0]["text"] == (
            "Sure — walk me through a recent challenging problem you tackled."
        )
        assert call_count["n"] == 1  # no reask triggered
