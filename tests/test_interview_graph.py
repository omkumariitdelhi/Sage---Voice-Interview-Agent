"""Tests for app/interview/graph.py — the LangGraph interview state machine.

Every test mocks the LLM gateway at its two call sites
(``app.interview.graph.complete`` for interviewer turns,
``app.interview.evaluation.complete`` for grading) — zero real API keys,
zero network calls, per spec.md. The Chroma store is built fresh into a
``tmp_path`` for every test (mirrors tests/test_retrieval_store.py's own
fixture pattern) so tests never depend on/mutate the project's real
``.chroma`` directory.
"""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

import pytest
import yaml
from langgraph.types import Command

from app.interview.graph import build_graph
from app.interview.state import initial_state, max_turns_ceiling
from app.retrieval.ingest import DEFAULT_QA_BANK_PATH, build_store

THREAD_CONFIG = lambda thread_id: {"configurable": {"thread_id": thread_id}}  # noqa: E731


@pytest.fixture
def ingested_store(tmp_path):
    """A fresh Chroma store built from the real data/qa_bank.yaml (10
    entries), in an isolated temp directory — mirrors
    tests/test_retrieval_store.py's own fixture."""
    chroma_dir = tmp_path / "chroma_store"
    build_store(chroma_dir=chroma_dir)
    yield chroma_dir
    shutil.rmtree(chroma_dir, ignore_errors=True)


@pytest.fixture
def small_dataset(tmp_path):
    """A tiny, isolated 2-question dataset + matching Chroma store, with a
    custom max_follow_ups_per_question, so tests can drive a full session
    quickly and prove the bound is data-driven (FLOW-05 + the data-driven
    acceptance criterion) without depending on the real 10-question bank."""
    data = {
        "domain": "Test Domain",
        "intro": "Welcome to the test interview.",
        "closing": "Thanks, that's everything.",
        "max_follow_ups_per_question": 1,
        "questions": [
            {
                "id": "t01",
                "topic": "topic-one",
                "difficulty": "easy",
                "question": "Test question one?",
                "ideal_answer": "SECRET_REFERENCE_ANSWER_ONE should never leak verbatim.",
                "key_points": ["point one a", "point one b"],
            },
            {
                "id": "t02",
                "topic": "topic-two",
                "difficulty": "easy",
                "question": "Test question two?",
                "ideal_answer": "SECRET_REFERENCE_ANSWER_TWO should never leak verbatim.",
                "key_points": ["point two a", "point two b"],
            },
        ],
    }
    dataset_path = tmp_path / "small_qa.yaml"
    dataset_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    chroma_dir = tmp_path / "small_chroma"
    build_store(qa_bank_path=dataset_path, chroma_dir=chroma_dir)

    yield dataset_path, chroma_dir
    shutil.rmtree(chroma_dir, ignore_errors=True)


def _mock_turn_text(messages, **kwargs):
    """Stand-in for app.interview.graph.complete — returns a short,
    deterministic 'interviewer said something' string derived from the
    prompt, never echoing any ideal_answer (the prompts themselves never
    include it for these call sites; this fake just mirrors that)."""
    return "MOCK_INTERVIEWER_TURN"


def _eval_json(verdict: str, missed: list[str] | None = None) -> str:
    return json.dumps(
        {"verdict": verdict, "missed_key_points": missed or [], "reasoning": "mock"}
    )


# ---------------------------------------------------------------------------
# Happy path: full scripted session reaches a terminal state.
# ---------------------------------------------------------------------------


def test_full_session_happy_path_reaches_terminal_state(ingested_store):
    """Scripted: greet -> ask q0 -> strong -> advance -> ... -> wrap_up ->
    feedback, with the LLM gateway fully mocked. Must reach phase == 'done'
    with no manual intervention and no unhandled exception."""
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ), patch("app.interview.feedback.complete") as mock_feedback:
        mock_feedback.return_value = json.dumps(
            {
                "per_question": [],
                "overall_strengths": ["clear"],
                "overall_improvements": ["none"],
            }
        )
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("happy-path")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        result = compiled.invoke(state, config)
        for _ in range(total_q):
            snapshot = compiled.get_state(config)
            if not snapshot.next:
                break
            result = compiled.invoke(Command(resume="a strong, complete answer"), config)

        assert result["phase"] == "done"
        assert result["feedback"] is not None
        assert result["qa_index"] == total_q
        assert not compiled.get_state(config).next  # no pending interrupt left


# ---------------------------------------------------------------------------
# Weak answer -> follow-up turn, qa_index unchanged, follow_up_count++.
# ---------------------------------------------------------------------------


def test_weak_answer_triggers_follow_up_same_question(ingested_store):
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete",
        return_value=_eval_json("weak", ["missed point"]),
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("weak-answer")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        result = compiled.invoke(Command(resume="an incomplete answer"), config)

        assert result["qa_index"] == 0  # did NOT advance
        assert result["follow_up_count"] == 1  # incremented
        assert result["last_verdict"] == "weak"
        assert result["phase"] == "follow_up"
        # The graph must be paused again, waiting for the next reply to the
        # SAME question.
        snapshot = compiled.get_state(config)
        assert snapshot.next == ("await_candidate",)


# ---------------------------------------------------------------------------
# Wrong answer -> scaffold prompt grounded in missed_key_points, no leaked
# ideal_answer substring.
# ---------------------------------------------------------------------------


def test_wrong_answer_scaffold_does_not_leak_ideal_answer(ingested_store):
    from app.interview.prompts import follow_up_message
    from app.retrieval.store import get_by_sequence_index

    entry = get_by_sequence_index(0, chroma_dir=ingested_store)

    # Build the actual follow-up prompt the graph would send and render it
    # through the same mock-turn-text path used elsewhere, but the real
    # acceptance criterion is about the PROMPT CONTENT sent to the model,
    # not a literal LLM call (mocked) — assert the prompt itself never
    # contains the literal ideal_answer text.
    messages = follow_up_message(
        entry, "a wrong answer", ["missed key point"], attempt_number=1
    )
    full_prompt_text = " ".join(m["content"] for m in messages)
    assert entry.ideal_answer not in full_prompt_text

    # Drive the graph end-to-end for a wrong answer and check the emitted
    # interviewer turn (using a realistic mocked turn-text fixture standing
    # in for what the LLM would produce from that prompt) also never
    # contains the ideal_answer substring.
    def realistic_follow_up_text(messages, **kwargs):
        return f"Can you say more about {entry.topic}? Think about the key aspects involved."

    with patch(
        "app.interview.graph.complete", side_effect=realistic_follow_up_text
    ), patch(
        "app.interview.evaluation.complete",
        return_value=_eval_json("wrong", ["missed key point"]),
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("wrong-answer")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        result = compiled.invoke(Command(resume="a materially incorrect answer"), config)

        follow_up_turn = result["transcript"][-1]
        assert entry.ideal_answer not in follow_up_turn["text"]
        assert result["last_verdict"] == "wrong"
        assert result["qa_index"] == 0


# ---------------------------------------------------------------------------
# Off-topic redirect: does not advance qa_index; shares the follow_up_count
# budget (the documented bookkeeping rule from plan.md Section 4).
# ---------------------------------------------------------------------------


def test_off_topic_redirect_does_not_advance_and_shares_followup_budget(ingested_store):
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("off_topic")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("off-topic")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        result = compiled.invoke(Command(resume="totally unrelated reply"), config)

        assert result["qa_index"] == 0  # did not advance
        assert result["last_verdict"] == "off_topic"
        assert result["phase"] == "redirect"
        # The documented rule (plan.md Section 4): off-topic consumes the
        # SAME follow_up_count budget as a weak/wrong answer.
        assert result["follow_up_count"] == 1


def test_redirect_prompt_includes_candidate_answer(ingested_store):
    """The redirect prompt must include the candidate's literal off-topic
    reply so the redirect can acknowledge what they actually said, not just
    generically restate the question."""
    captured: list[list[dict]] = []

    def _capture(messages, **kwargs):
        captured.append(messages)
        return "MOCK_TURN"

    with patch("app.interview.graph.complete", side_effect=_capture), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("off_topic")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("redirect-includes-answer")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        compiled.invoke(Command(resume="totally unrelated reply about my weekend"), config)

    redirect_call_text = " ".join(m["content"] for m in captured[-1])
    assert "totally unrelated reply about my weekend" in redirect_call_text


def test_off_topic_forever_on_one_question_forces_advance_at_cap(small_dataset):
    """Off-topic-forever on a SINGLE question must still force-advance once
    follow_up_count reaches max_follow_ups_per_question — identical bound
    shape to the weak-answer case, proving the shared-budget rule
    end-to-end. With this fixture's max_follow_ups_per_question=1,
    route_verdict's `follow_up_count >= max_follow_ups` check means the
    very first off-topic reply already exhausts the budget and forces an
    immediate advance (1 redirect chance total, not 1 *extra* chance)."""
    dataset_path, chroma_dir = small_dataset
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("off_topic")
    ):
        compiled, total_q, max_fu = build_graph(
            dataset_path=dataset_path, chroma_dir=chroma_dir
        )
        assert max_fu == 1  # this fixture's dataset sets max_follow_ups_per_question: 1
        config = THREAD_CONFIG("off-topic-forever-one-question")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        # follow_up_count goes 0 -> 1, which already equals max_fu(1), so
        # route_verdict forces an immediate advance — qa_index moves to 1
        # after just one off-topic reply.
        result = compiled.invoke(Command(resume="off topic reply 1"), config)
        assert result["qa_index"] == 1
        assert result["follow_up_count"] == 0  # reset for the next question
        assert result["question_verdicts"][0] == ("t01", "off_topic")


# ---------------------------------------------------------------------------
# FLOW-05: the bounded-termination hard ceiling. The single most important
# test in this phase.
# ---------------------------------------------------------------------------


def test_bounded_termination_all_wrong_forever(small_dataset):
    """Every single answer to every question is scripted as 'wrong' forever.
    The graph must still reach phase == 'done' within the provable hard
    ceiling MAX_TURNS = total_questions * (max_follow_ups + 1), and never
    loop past it."""
    dataset_path, chroma_dir = small_dataset
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("wrong")
    ), patch("app.interview.feedback.complete") as mock_feedback:
        mock_feedback.return_value = json.dumps(
            {"per_question": [], "overall_strengths": [], "overall_improvements": []}
        )
        compiled, total_q, max_fu = build_graph(
            dataset_path=dataset_path, chroma_dir=chroma_dir
        )
        ceiling = max_turns_ceiling(total_questions=total_q, max_follow_ups=max_fu)
        assert ceiling == total_q * (max_fu + 1)

        config = THREAD_CONFIG("all-wrong-forever")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        result = compiled.invoke(state, config)
        # Generous but FINITE turn budget for the test loop itself — if the
        # graph has a bug that makes it loop forever, this raises/asserts
        # rather than hanging the test runner.
        turn_iterations = 0
        max_test_iterations = ceiling + 5
        while True:
            snapshot = compiled.get_state(config)
            if not snapshot.next:
                break
            turn_iterations += 1
            assert turn_iterations <= max_test_iterations, (
                "Graph did not halt within the provable ceiling + safety "
                "margin — bounded-termination property is broken."
            )
            result = compiled.invoke(Command(resume="this is always wrong"), config)

        assert result["phase"] == "done"
        assert result["feedback"] is not None
        assert result["total_turns"] <= ceiling
        assert result["qa_index"] == total_q  # every question was force-advanced through


# ---------------------------------------------------------------------------
# Resume-across-invoke: state survives across SEPARATE invoke() calls via
# thread_id. The other most-important test in this phase.
# ---------------------------------------------------------------------------


def test_resume_across_separate_invoke_calls(ingested_store):
    """Calls invoke() for the first turn, then in a SEPARATE invoke() call
    (using only the thread_id + a freshly re-fetched checkpoint state, never
    a held-over Python object) resumes with the next candidate answer —
    proving the graph is callable once per HTTP request, not just in one
    long-lived loop."""
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("resume-across-invoke")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        # --- "Request 1": first invoke() call ---
        compiled.invoke(state, config)
        snapshot_after_first = compiled.get_state(config)
        assert snapshot_after_first.next == ("await_candidate",), (
            "Graph must be paused awaiting the candidate's reply after the "
            "first invoke() call."
        )
        transcript_len_after_first = len(snapshot_after_first.values["transcript"])
        qa_index_after_first = snapshot_after_first.values["qa_index"]
        follow_up_count_after_first = snapshot_after_first.values["follow_up_count"]

        # Simulate a brand-new process/request boundary: drop all local
        # references except the config (thread_id) and the compiled graph
        # object itself (a fresh request handler would rebuild the graph via
        # build_graph() again and only carry the thread_id forward — the
        # compiled graph object is stateless; only the checkpointer holds
        # state, keyed by thread_id).
        del state, snapshot_after_first

        # --- "Request 2": a SEPARATE invoke() call, same thread_id ---
        resumed_snapshot = compiled.get_state(config)
        assert resumed_snapshot.values["qa_index"] == qa_index_after_first
        assert (
            len(resumed_snapshot.values["transcript"]) == transcript_len_after_first
        )

        result = compiled.invoke(Command(resume="my strong and complete answer"), config)

        assert result["qa_index"] == qa_index_after_first + 1  # advanced (strong verdict)
        assert result["follow_up_count"] == 0
        assert len(result["transcript"]) == transcript_len_after_first + 2  # +candidate +next question
        # The candidate's answer from "request 2" actually made it into the
        # transcript that was accumulated starting in "request 1" — proving
        # continuity through the checkpointer, not in-process object reuse.
        assert any(
            turn["text"] == "my strong and complete answer" for turn in result["transcript"]
        )


def test_build_once_many_sessions_pattern(ingested_store):
    """Documents-by-proof the CORRECT usage pattern for Phase 6 (review-01.md
    Finding 1, Major): build_graph() is called exactly ONCE here, and the
    SAME compiled graph object then drives several independent interview
    sessions (distinct thread_ids), interleaved, via invoke()/
    Command(resume=...). This is exactly the "module-level singleton built
    once at server startup, reused by every request handler" pattern
    documented in build_graph()'s docstring and plan.md Section 7.1 — NOT
    a fresh build_graph() per session/request (which would NOT share state,
    since MemorySaver is instantiated fresh inside build_graph() on every
    call).

    Proves two things at once:
    1. One compiled graph object correctly serves multiple, fully
       independent sessions (no cross-contamination between thread_ids),
       interleaved turn-by-turn rather than run-to-completion one at a time.
    2. Each session resumes correctly across separate invoke() calls on
       that one shared object, all the way to a terminal state with its
       own distinct feedback report.
    """
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ), patch("app.interview.feedback.complete") as mock_feedback:
        mock_feedback.return_value = json.dumps(
            {"per_question": [], "overall_strengths": [], "overall_improvements": []}
        )

        # ---- exactly ONE build_graph() call for this whole test ----
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)

        session_ids = ["server-session-A", "server-session-B", "server-session-C"]
        configs = {sid: THREAD_CONFIG(sid) for sid in session_ids}

        # Start all three sessions on the SAME compiled object, interleaved
        # (simulating three concurrent "requests" against one long-lived
        # server process) — start A, then B, then C, rather than finishing
        # one before starting the next.
        for sid in session_ids:
            compiled.invoke(
                initial_state(total_questions=total_q, max_follow_ups=max_fu),
                configs[sid],
            )

        # Each session must independently be paused at qa_index == 0 right
        # now — proves starting B/C did not disturb A's already-checkpointed
        # state (no cross-contamination from sharing one compiled object).
        for sid in session_ids:
            snapshot = compiled.get_state(configs[sid])
            assert snapshot.values["qa_index"] == 0
            assert snapshot.next == ("await_candidate",)

        # Drive each session to completion via Command(resume=...) on the
        # SAME compiled object, interleaving turns across sessions (answer
        # one turn of A, then one of B, then one of C, repeat) rather than
        # finishing each session before touching the next.
        done = {sid: False for sid in session_ids}
        safety_cap = total_q * 5  # generous but finite, mirrors other tests' style
        rounds = 0
        while not all(done.values()):
            rounds += 1
            assert rounds <= safety_cap, "Sessions failed to reach 'done' within a bounded number of rounds."
            for sid in session_ids:
                if done[sid]:
                    continue
                snapshot = compiled.get_state(configs[sid])
                if not snapshot.next:
                    done[sid] = True
                    continue
                result = compiled.invoke(
                    Command(resume=f"a strong, complete answer from {sid}"),
                    configs[sid],
                )
                if result.get("phase") == "done":
                    done[sid] = True

        # Each session must have reached its OWN terminal state, with its
        # own feedback and its own full qa_index progression — none of them
        # leaked state into another despite being driven on one shared
        # compiled graph object and interleaved turn-by-turn.
        for sid in session_ids:
            final_state = compiled.get_state(configs[sid]).values
            assert final_state["phase"] == "done"
            assert final_state["feedback"] is not None
            assert final_state["qa_index"] == total_q
            assert not compiled.get_state(configs[sid]).next
            # Every transcript line attributed to this session's candidate
            # answers must reference only this session's id — no other
            # session's answer text leaked in.
            for turn in final_state["transcript"]:
                if turn["speaker"] == "candidate":
                    assert turn["text"] == f"a strong, complete answer from {sid}"


def test_resume_with_wrong_thread_id_does_not_see_other_session_state(ingested_store):
    """Sanity check on thread isolation: two different thread_ids never
    leak state into each other (a prerequisite for "callable once per HTTP
    request" to be safe for concurrent sessions)."""
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config_a = THREAD_CONFIG("session-a")
        config_b = THREAD_CONFIG("session-b")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config_a)
        compiled.invoke(Command(resume="answer for session a"), config_a)

        # Session B has never been invoked yet — its state must not exist /
        # must not reflect session A's progress.
        snapshot_b = compiled.get_state(config_b)
        assert snapshot_b.values == {} or snapshot_b.values.get("qa_index", 0) == 0


# ---------------------------------------------------------------------------
# max_follow_ups_per_question is data-driven, not hard-coded.
# ---------------------------------------------------------------------------


def test_max_follow_ups_is_data_driven(tmp_path):
    """A temp dataset with max_follow_ups_per_question: 1 (instead of the
    real bank's 2) must change the graph's bound with zero code changes."""
    data = {
        "domain": "Test",
        "intro": "Hi.",
        "closing": "Bye.",
        "max_follow_ups_per_question": 1,
        "questions": [
            {
                "id": "x01",
                "topic": "x",
                "difficulty": "easy",
                "question": "Q?",
                "ideal_answer": "A.",
                "key_points": ["p"],
            }
        ],
    }
    dataset_path = tmp_path / "one_followup.yaml"
    dataset_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    chroma_dir = tmp_path / "one_followup_chroma"
    build_store(qa_bank_path=dataset_path, chroma_dir=chroma_dir)

    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("weak")
    ):
        compiled, total_q, max_fu = build_graph(
            dataset_path=dataset_path, chroma_dir=chroma_dir
        )
        assert max_fu == 1
        config = THREAD_CONFIG("data-driven-cap")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        result = compiled.invoke(Command(resume="weak answer 1"), config)
        # follow_up_count hit max_fu(1) already -> forced advance, NOT a
        # second follow-up loop.
        assert result["qa_index"] == 1
        assert result["follow_up_count"] == 0

    shutil.rmtree(chroma_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Static import-hygiene checks (no litellm / no direct YAML parsing under
# app/interview/).
# ---------------------------------------------------------------------------


def test_no_litellm_import_under_app_interview():
    from pathlib import Path

    interview_dir = Path(__file__).resolve().parents[1] / "app" / "interview"
    offending = []
    for py_file in interview_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "import litellm" in text or "from litellm" in text:
            offending.append(str(py_file))
    assert offending == [], f"Found direct litellm imports under app/interview/: {offending}"


def test_no_direct_yaml_module_import_under_app_interview():
    """app/interview/ must never import the `yaml` module directly — all
    dataset parsing goes through app.retrieval.loader."""
    from pathlib import Path

    interview_dir = Path(__file__).resolve().parents[1] / "app" / "interview"
    offending = []
    for py_file in interview_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "import yaml" in text:
            offending.append(str(py_file))
    assert offending == [], f"Found direct yaml imports under app/interview/: {offending}"


def test_wrong_verdict_mid_question_uses_teach_prompt_not_follow_up(ingested_store):
    """A 'wrong' verdict (still on the same, open question) must route
    through teach_wrong_answer_message, not the plain weak-answer hint —
    distinguish them via teach_wrong_answer_message's unique system-prompt
    marker phrase ("materially incorrect"), which follow_up_message's
    system prompt never contains."""
    captured: list[list[dict]] = []

    def _capture(messages, **kwargs):
        captured.append(messages)
        return "MOCK_TURN"

    with patch("app.interview.graph.complete", side_effect=_capture), patch(
        "app.interview.evaluation.complete",
        return_value=_eval_json("wrong", ["missed point"]),
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("wrong-uses-teach-prompt")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        compiled.invoke(Command(resume="a materially incorrect answer"), config)

    follow_up_call_text = " ".join(m["content"] for m in captured[-1])
    assert "materially incorrect" in follow_up_call_text
    # The candidate's literal answer must be in the prompt so the teaching
    # reply can react to specifics, not generic advice.
    assert "a materially incorrect answer" in follow_up_call_text


def test_weak_verdict_mid_question_uses_plain_follow_up_not_teach(ingested_store):
    """Contrast case: 'weak' must still use the gentler follow_up_message,
    never the teach prompt's marker phrase."""
    captured: list[list[dict]] = []

    def _capture(messages, **kwargs):
        captured.append(messages)
        return "MOCK_TURN"

    with patch("app.interview.graph.complete", side_effect=_capture), patch(
        "app.interview.evaluation.complete",
        return_value=_eval_json("weak", ["missed point"]),
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("weak-uses-plain-follow-up")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        compiled.invoke(Command(resume="an incomplete answer"), config)

    follow_up_call_text = " ".join(m["content"] for m in captured[-1])
    assert "materially incorrect" not in follow_up_call_text
    assert "an incomplete answer" in follow_up_call_text


def test_recap_fires_when_advancing_after_non_strong_verdict(small_dataset):
    """Advancing to question 2 after question 1 was closed with a non-strong
    final verdict (here: weak, follow-up budget exhausted at max_fu=1) must
    build the next interviewer_turn via recap_and_ask_message — proven by
    the previous question's ideal_answer (a deliberately unique marker
    string, SECRET_REFERENCE_ANSWER_ONE) showing up in the prompt sent to
    the gateway. This is the ONE intentional, scoped exception to "never
    leak ideal_answer into a candidate-facing prompt" — see prompts.py's
    module docstring."""
    dataset_path, chroma_dir = small_dataset
    captured: list[list[dict]] = []

    def _capture(messages, **kwargs):
        captured.append(messages)
        return "MOCK_TURN"

    with patch("app.interview.graph.complete", side_effect=_capture), patch(
        "app.interview.evaluation.complete",
        return_value=_eval_json("weak", ["missed point"]),
    ):
        compiled, total_q, max_fu = build_graph(
            dataset_path=dataset_path, chroma_dir=chroma_dir
        )
        assert max_fu == 1
        config = THREAD_CONFIG("recap-after-weak-close")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        result = compiled.invoke(Command(resume="a weak answer"), config)

    assert result["qa_index"] == 1  # force-advanced past question 1
    recap_call_text = " ".join(m["content"] for m in captured[-1])
    assert "SECRET_REFERENCE_ANSWER_ONE" in recap_call_text
    assert "a weak answer" in recap_call_text


def test_no_recap_when_advancing_after_strong_verdict(small_dataset):
    """Contrast case: advancing after a STRONG final verdict must ask the
    next question plainly — no recap, no ideal_answer in the prompt."""
    dataset_path, chroma_dir = small_dataset
    captured: list[list[dict]] = []

    def _capture(messages, **kwargs):
        captured.append(messages)
        return "MOCK_TURN"

    with patch("app.interview.graph.complete", side_effect=_capture), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ):
        compiled, total_q, max_fu = build_graph(
            dataset_path=dataset_path, chroma_dir=chroma_dir
        )
        config = THREAD_CONFIG("no-recap-after-strong-close")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        compiled.invoke(state, config)
        result = compiled.invoke(Command(resume="a strong, complete answer"), config)

    assert result["qa_index"] == 1
    next_question_call_text = " ".join(m["content"] for m in captured[-1])
    assert "SECRET_REFERENCE_ANSWER_ONE" not in next_question_call_text


def test_interviewer_turn_strips_markdown_before_storing(ingested_store):
    """Regression test: if the LLM emits markdown despite _SYSTEM_PERSONA's
    explicit instruction not to, the transcript text actually stored (and
    therefore what TTS would synthesize) must have it stripped — this is
    the defensive backstop in app/interview/speech_text.py applied at
    interviewer_turn's single chokepoint."""

    def _markdown_turn_text(messages, **kwargs):
        return "**What's** the difference between an `array` and a linked list?"

    with patch("app.interview.graph.complete", side_effect=_markdown_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ):
        compiled, total_q, max_fu = build_graph(chroma_dir=ingested_store)
        config = THREAD_CONFIG("markdown-stripped")
        state = initial_state(total_questions=total_q, max_follow_ups=max_fu)

        result = compiled.invoke(state, config)

    stored_text = result["transcript"][-1]["text"]
    assert "*" not in stored_text
    assert "`" not in stored_text
    assert "What's the difference between an array and a linked list?" in stored_text


def test_real_qa_bank_path_is_unchanged_default():
    """Sanity check that the real project dataset is still the default the
    graph wires up to when no override is given (exercised indirectly by
    every other test passing dataset_path explicitly; this asserts the
    default constant itself still resolves to the real file)."""
    assert DEFAULT_QA_BANK_PATH.name == "qa_bank.yaml"
    assert DEFAULT_QA_BANK_PATH.exists()
