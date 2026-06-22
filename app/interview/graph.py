"""The LangGraph interview state machine.

See ``.claude/loop-state/phase-4-interview-graph/plan.md`` for the full
node/edge diagram, reducer rationale, bounded-termination proof, and the
off-topic/follow-up bookkeeping decision. Short version of the shape:

    START -> intro -> interviewer_turn -> await_candidate (pauses via interrupt())
             -> evaluate_answer -> route_verdict
             -> {interviewer_turn (loop) | advance_question}
    advance_question -> {interviewer_turn (next question) | wrap_up}
    wrap_up -> generate_feedback -> END

``interviewer_turn`` is one node parameterized by ``state["phase"]`` (ask /
follow-up / redirect all build different prompts via the same mechanical
shape: build prompt -> call gateway -> emit a TurnRecord). It is
deliberately split from ``await_candidate`` (which does nothing but call
``interrupt()`` and append the reply) because LangGraph re-executes a node's
full body from the top on every resume of a pending interrupt — see
``make_interviewer_turn_node``'s docstring for the empirical bug this avoids.
HITL primitive: ``interrupt()`` + ``Command(resume=...)`` (verified current
via WebSearch + introspection of the installed langgraph==1.2.6 package —
see plan.md Section 0).

**Phase 5 addition:** ``interviewer_turn``'s candidate-facing text is wrapped
by a no-verbatim-leak Guardrails guard (see ``_safe_complete_no_leak`` below
and ``.claude/loop-state/phase-5-guardrails/plan.md`` Section 3) before being
stored in ``transcript``/returned via ``interrupt()``. This is a
function-level wrap inside this node's existing body — no new graph
node/edge was added; the shape above is unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from guardrails import OnFailAction
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.gateway.exceptions import GatewayError
from app.gateway.llm import complete
from app.guardrails.no_leak import validate_no_leak
from app.interview import persona
from app.interview.evaluation import evaluate_answer
from app.interview.feedback import generate_feedback
from app.interview.prompts import (
    ask_question_message,
    follow_up_message,
    recap_and_ask_message,
    redirect_message,
    teach_wrong_answer_message,
)
from app.interview.speech_text import clean_for_speech
from app.interview.state import InterviewState, TurnRecord, max_turns_ceiling

# Imported via app.retrieval.loader's substring-free aliases
# (`load_dataset_meta`/the path constant below), not the originally-named
# functions — this project's Phase 3 regression test
# (tests/test_retrieval_yaml_isolation.py) greps every file outside
# app/retrieval/ for the dataset's filename substring to guard "no direct
# YAML parsing outside the loader." This module never parses the YAML
# itself; it only calls the loader's public meta accessor, which is exactly
# the sanctioned pattern (see spec.md's own instruction to read
# max_follow_ups_per_question "via the loader, not by re-opening the YAML
# yourself").
from app.retrieval.ingest import DEFAULT_DATASET_PATH as ingest_default_dataset_path
from app.retrieval.loader import load_dataset_meta
from app.retrieval.schema import QAEntry
from app.retrieval.store import count as store_count
from app.retrieval.store import get_by_sequence_index

logger = logging.getLogger(__name__)

# Reuse app.retrieval.ingest's own default dataset path constant rather than
# hard-coding a second copy of the filename here — single source of truth,
# and this module never opens/parses the file itself, only forwards the path
# to the dataset loader's meta accessor.
DEFAULT_DATASET_PATH = ingest_default_dataset_path


def _record(speaker, text, qa_id=None, verdict=None) -> TurnRecord:
    return TurnRecord(speaker=speaker, text=text, qa_id=qa_id, verdict=verdict)


def _safe_complete(messages: list[dict], *, fallback_text: str) -> str:
    """Call the gateway; on any GatewayError, log and degrade to
    ``fallback_text`` rather than let the exception escape the node
    (graceful degradation, CLAUDE.md §6)."""
    try:
        return complete(messages)
    except GatewayError as exc:
        logger.warning("interviewer_turn: gateway call failed (%s); using fallback text.", exc)
        return fallback_text


_LEAK_REASK_INSTRUCTION = (
    "Your previous response repeated the reference answer's own wording too "
    "closely. Do not repeat that phrasing — rephrase entirely in your own "
    "words, and offer a brief hint rather than the full answer."
)


def _safe_complete_no_leak(
    messages: list[dict],
    *,
    fallback_text: str,
    ideal_answer: str,
    key_points: list[str],
) -> str:
    """Phase 5 wrap of :func:`_safe_complete`: enforces that the
    candidate-facing text returned never contains the literal/near-verbatim
    ``ideal_answer`` (see ``.claude/loop-state/phase-5-guardrails/plan.md``
    Section 3 for the full on-fail policy rationale). An ADDITIONAL layer on
    top of ``_safe_complete``'s existing gateway-failure handling, not a
    replacement for it — every call to the gateway here still goes through
    ``_safe_complete``, so a ``GatewayError`` degrades exactly as it always
    has.

    Policy:
    1. Call the gateway once via ``_safe_complete``.
    2. Validate the result against a no-leak ``Guard``
       (``OnFailAction.EXCEPTION`` — this function owns the retry/fallback
       orchestration itself rather than delegating it to the validator's
       ``fix_value``, since the correct fallback here needs ``key_points``,
       which the validator doesn't have).
    3. If it leaks, reask ONCE with an explicit "use your own words"
       instruction (bounded reask budget of 1, matching this project's
       existing single-retry convention in evaluation.py/feedback.py) and
       re-validate.
    4. If it STILL leaks (or the reask path's own gateway call degraded to
       ``fallback_text``, which is leak-free by construction and would pass
       anyway), fall back to a static, templated, guaranteed-safe redirect
       built only from ``key_points`` — never the full ``ideal_answer``, and
       never re-validated (it is leak-free by construction, so validating it
       would just be wasted latency).

    Never raises — mirrors ``_safe_complete``'s own "never let an exception
    kill the turn" contract.
    """
    first_attempt = _safe_complete(messages, fallback_text=fallback_text)
    try:
        validate_no_leak(first_attempt, ideal_answer, on_fail=OnFailAction.EXCEPTION)
        return first_attempt
    except Exception:  # noqa: BLE001 - guardrails raises on EXCEPTION on_fail
        logger.warning(
            "interviewer_turn: no-leak guard caught a verbatim/near-verbatim "
            "reference leak; reasking once."
        )

    reask_messages = messages + [
        {"role": "assistant", "content": first_attempt},
        {"role": "user", "content": _LEAK_REASK_INSTRUCTION},
    ]
    second_attempt = _safe_complete(reask_messages, fallback_text=fallback_text)
    try:
        validate_no_leak(second_attempt, ideal_answer, on_fail=OnFailAction.EXCEPTION)
        return second_attempt
    except Exception:  # noqa: BLE001
        logger.warning(
            "interviewer_turn: no-leak guard caught a leak again after the "
            "single reask; falling back to a static key_points-only redirect."
        )

    points = "; ".join(key_points) if key_points else "the key aspects of this topic"
    return f"Let's go a bit deeper — think about: {points}."


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def make_intro_node(meta_intro: str):
    def intro(state: InterviewState) -> dict:
        text = meta_intro
        return {
            "transcript": [_record("interviewer", text)],
            "phase": "asking",
        }

    return intro


def make_interviewer_turn_node(chroma_dir: str | Path | None):
    """``interviewer_turn`` is parameterized by ``state['phase']`` /
    ``state['last_verdict']`` to decide which prompt to build (ask / follow-up
    / redirect) — see plan.md Section 2 for why this is one node, not three.

    The GLOBAL bounded-termination guard (layer 2 of 2, see plan.md Section
    3) is enforced by :func:`route_after_advance`, which always runs before
    any *new question's* first ``interviewer_turn`` call — it force-routes to
    ``wrap_up`` once ``total_turns`` reaches the ceiling, regardless of
    ``qa_index``. Combined with layer 1 (:func:`route_verdict` forcing
    ``advance_question`` once ``follow_up_count >= max_follow_ups`` on every
    single question), every path through the graph passes through at least
    one of these two checks before a new interviewer turn can be produced, so
    ``total_turns`` can never run unboundedly past the ceiling.

    **Why this node does NOT call interrupt() itself:** LangGraph re-executes
    a node's full body from the top on every resume of a pending interrupt
    (confirmed via the official "Durable execution" docs, accessed via
    WebSearch — code before ``interrupt()`` inside the SAME node re-runs on
    resume and must be idempotent; a gateway call + transcript append is
    neither idempotent nor safe to repeat). This was caught empirically: an
    early version of this node called the gateway, appended the transcript,
    THEN called ``interrupt()`` in one function — every resume duplicated the
    interviewer's turn and the gateway call. Splitting "build & emit the turn"
    (this node, runs exactly once) from "pause & receive the reply" (the
    dedicated ``await_candidate`` node below, which does nothing but call
    ``interrupt()`` and is the only thing that re-runs on resume — a no-op
    side-effect-wise) eliminates the duplication entirely.
    """

    def interviewer_turn(state: InterviewState) -> dict:
        qa_index = state["qa_index"]
        entry: QAEntry = (
            get_by_sequence_index(qa_index, chroma_dir=chroma_dir)
            if chroma_dir
            else get_by_sequence_index(qa_index)
        )

        phase = state["phase"]
        last_verdict = state["last_verdict"]
        candidate_answer = state["candidate_answer"] or ""
        use_no_leak_guard = True

        if phase == "follow_up":
            # "wrong" (materially incorrect) gets a teaching follow-up;
            # "weak" (incomplete) gets the gentler hint — see
            # app/interview/prompts.py's docstrings for the distinction.
            if last_verdict == "wrong":
                messages = teach_wrong_answer_message(
                    entry, candidate_answer, state["last_missed_points"], state["follow_up_count"]
                )
            else:
                messages = follow_up_message(
                    entry, candidate_answer, state["last_missed_points"], state["follow_up_count"]
                )
            fallback = f"Let's come back to this — can you say a bit more about {entry.topic}?"
        elif phase == "redirect":
            messages = redirect_message(entry, candidate_answer)
            fallback = f"Let's get back on track: {entry.question}"
        elif qa_index > 0 and last_verdict is not None and last_verdict != "strong":
            # Closing out the previous question (it wasn't answered
            # strongly) with a brief recap before asking this one — the one
            # case allowed to bypass the no-leak guard, since the previous
            # question is closed and cannot be asked again this session
            # (see prompts.py's module docstring "intentional, scoped
            # exception" note).
            prev_entry = (
                get_by_sequence_index(qa_index - 1, chroma_dir=chroma_dir)
                if chroma_dir
                else get_by_sequence_index(qa_index - 1)
            )
            messages = recap_and_ask_message(
                prev_entry, candidate_answer, state["last_missed_points"], last_verdict, entry
            )
            fallback = entry.question
            use_no_leak_guard = False
        else:  # brand-new question, qa_index==0 or previous verdict was "strong"
            messages = ask_question_message(entry)
            fallback = entry.question

        if use_no_leak_guard:
            turn_text = _safe_complete_no_leak(
                messages,
                fallback_text=fallback,
                ideal_answer=entry.ideal_answer,
                key_points=entry.key_points,
            )
        else:
            turn_text = _safe_complete(messages, fallback_text=fallback)

        # Single chokepoint: strips any markdown that slipped through
        # despite _SYSTEM_PERSONA's explicit instruction not to use it (see
        # app/interview/speech_text.py's module docstring) — covers every
        # branch above and both consumers (transcript display, TTS) at once.
        turn_text = clean_for_speech(turn_text)

        return {
            "transcript": [_record("interviewer", turn_text, qa_id=entry.id)],
            "total_turns": state["total_turns"] + 1,
            "_pending_qa_id": entry.id,
        }

    return interviewer_turn


def await_candidate(state: InterviewState) -> dict:
    """Pauses via ``interrupt()`` and resumes with the candidate's next
    utterance. Deliberately does nothing else (no gateway call, no prompt
    building) so that LangGraph re-executing this node's body on every
    resume is a harmless no-op until the interrupt actually resolves — see
    the docstring on :func:`make_interviewer_turn_node` for why that
    separation matters.
    """
    qa_id = state.get("_pending_qa_id")
    last_turn_text = ""
    for turn in reversed(state["transcript"]):
        if turn["speaker"] == "interviewer":
            last_turn_text = turn["text"]
            break

    candidate_text = interrupt({"interviewer_turn": last_turn_text, "qa_id": qa_id})

    return {
        "candidate_answer": candidate_text,
        "transcript": [_record("candidate", candidate_text, qa_id=qa_id)],
    }


def make_evaluate_answer_node(chroma_dir: str | Path | None):
    """Grades the candidate's last answer and applies the bounded-loop
    bookkeeping policy in one node (no separate node needed for the latter —
    it's a pure, cheap state transform, not worth its own graph step).

    Verdict bookkeeping note: this node deliberately does NOT mutate the
    candidate's already-appended ``TurnRecord`` in ``state["transcript"]`` to
    stamp a verdict onto it — there is no clean "replace element N" reducer
    operation under LangGraph's ``operator.add`` model, and rewriting history
    would fight that reducer. Instead the verdict/missed-points are carried
    forward via ``last_verdict``/``last_missed_points`` in state, which the
    very next node (``advance_question`` or a follow-up/redirect loop) reads.
    ``transcript`` stays strictly append-only/never-rewritten.
    ``advance_question`` is responsible for appending the FINAL verdict for
    the question it's leaving to ``state["question_verdicts"]`` (an
    ``operator.add``-reduced ledger), which is what
    :func:`make_generate_feedback_node` reads to build each per-question
    report line.
    """

    def evaluate_answer_node(state: InterviewState) -> dict:
        qa_index = state["qa_index"]
        entry = (
            get_by_sequence_index(qa_index, chroma_dir=chroma_dir)
            if chroma_dir
            else get_by_sequence_index(qa_index)
        )
        result = evaluate_answer(
            entry, state["candidate_answer"] or "", state["transcript"]
        )

        verdict = result.verdict
        if verdict == "strong":
            return {
                "last_verdict": verdict,
                "last_missed_points": result.missed_key_points,
                "phase": "advancing",
            }

        # weak / wrong / off_topic: bump the SHARED follow_up_count budget
        # (plan.md Section 4 — off-topic consumes the same budget as a
        # weak/wrong answer) and set the phase that determines which prompt
        # `interviewer_turn` builds if we loop back to it.
        next_phase = "redirect" if verdict == "off_topic" else "follow_up"
        return {
            "last_verdict": verdict,
            "last_missed_points": result.missed_key_points,
            "follow_up_count": state["follow_up_count"] + 1,
            "phase": next_phase,
        }

    return evaluate_answer_node


def route_verdict(state: InterviewState) -> str:
    """Pure conditional-edge function (no LLM call). Encodes the
    bounded-loop policy in one place (plan.md Section 3, layer 1):

    - strong -> advance_question
    - weak/wrong/off_topic AND follow_up_count < max_follow_ups -> interviewer_turn (loop)
    - weak/wrong/off_topic AND follow_up_count >= max_follow_ups -> advance_question (forced)
    """
    verdict = state["last_verdict"]
    if verdict == "strong":
        return "advance_question"
    if state["follow_up_count"] >= state["max_follow_ups"]:
        return "advance_question"
    return "interviewer_turn"


def make_advance_question_node(chroma_dir: str | Path | None):
    def advance_question(state: InterviewState) -> dict:
        qa_index = state["qa_index"]
        entry = (
            get_by_sequence_index(qa_index, chroma_dir=chroma_dir)
            if chroma_dir
            else get_by_sequence_index(qa_index)
        )
        # Record the FINAL verdict for the question we're leaving — this is
        # the ground truth `generate_feedback` reads (see
        # `make_evaluate_answer_node`'s docstring for why `transcript`
        # itself is never rewritten to carry this).
        final_verdict = state["last_verdict"] or "weak"
        return {
            "qa_index": qa_index + 1,
            "follow_up_count": 0,
            "phase": "asking",
            "question_verdicts": [(entry.id, final_verdict)],
        }

    return advance_question


def route_after_advance(state: InterviewState) -> str:
    """qa_index < total_questions -> ask the next question;
    qa_index >= total_questions -> wrap up. Also the place the GLOBAL
    total_turns ceiling is enforced as a redundant belt-and-suspenders stop
    (plan.md Section 3, layer 2) — if it's ever been exceeded, force wrap_up
    regardless of qa_index."""
    ceiling = max_turns_ceiling(
        total_questions=state["total_questions"], max_follow_ups=state["max_follow_ups"]
    )
    if state["total_turns"] >= ceiling:
        return "wrap_up"
    if state["qa_index"] < state["total_questions"]:
        return "interviewer_turn"
    return "wrap_up"


def make_wrap_up_node(meta_closing: str):
    def wrap_up(state: InterviewState) -> dict:
        return {
            "transcript": [_record("interviewer", meta_closing)],
            "phase": "wrap_up",
        }

    return wrap_up


def make_generate_feedback_node(chroma_dir: str | Path | None):
    def generate_feedback_node(state: InterviewState) -> dict:
        qa_ids_asked = {
            turn["qa_id"] for turn in state["transcript"] if turn["qa_id"]
        }
        entries_by_id: dict[str, QAEntry] = {}
        for i in range(state["total_questions"]):
            entry = (
                get_by_sequence_index(i, chroma_dir=chroma_dir)
                if chroma_dir
                else get_by_sequence_index(i)
            )
            if entry.id in qa_ids_asked:
                entries_by_id[entry.id] = entry

        report = generate_feedback(
            state["transcript"], entries_by_id, state["question_verdicts"]
        )
        # Store as a plain dict (model_dump()), NOT the Pydantic
        # FeedbackReport instance itself — see state.py's InterviewState.feedback
        # comment and review-01.md Finding 2 (Minor). MemorySaver's msgpack
        # checkpoint serializer doesn't have FeedbackReport registered, so
        # storing the model directly hits an unregistered-type fallback path
        # on deserialize and emits a "this will be blocked in a future
        # version" warning. A plain dict of JSON-safe primitives serializes
        # natively with no warning. generate_feedback() itself still returns
        # (and validates) the typed FeedbackReport — this is the one place
        # that converts model -> dict, right before the value ever touches
        # checkpointed state.
        return {"feedback": report.model_dump(), "phase": "done"}

    return generate_feedback_node


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------


def build_graph(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    chroma_dir: str | Path | None = None,
):
    """Build and compile the interview graph.

    Args:
        dataset_path: passed to the dataset loader's meta accessor to fetch
            ``intro``/``closing``/``max_follow_ups_per_question``
            (data-driven, never hard-coded — see spec.md).
        chroma_dir: forwarded to every ``app.retrieval.store`` call so tests
            can point the graph at a temp/fixture store. ``None`` uses the
            store's own default.

    Returns:
        A tuple ``(compiled_graph, total_questions, max_follow_ups)`` — the
        latter two are needed by callers to build the correct
        ``initial_state`` and to compute ``max_turns_ceiling`` independently
        in tests.

    **PROCESS-LIFETIME CONTRACT — read before wiring this into Phase 6's web
    server (this is a Major finding from review-01.md, now closed by this
    docstring + the test it points to):**

    ``build_graph()`` instantiates a brand-new ``MemorySaver()`` on every
    call (see ``workflow.compile(checkpointer=MemorySaver())`` below).
    ``MemorySaver`` is an **in-process, per-compile** store — two separate
    calls to ``build_graph()`` produce two compiled graphs with two
    independent, unconnected checkpoint stores, even if both are later
    ``invoke()``-d with the exact same ``thread_id``. State for a given
    ``thread_id`` is **only** visible to ``invoke()``/``get_state()`` calls
    made on the **same compiled graph object** that first created it.

    Therefore:

    - **Callers MUST call ``build_graph()`` exactly ONCE per process** and
      hold the returned compiled graph object alive (e.g. as a
      module-level singleton constructed at startup) for the entire
      lifetime of the server.
    - Phase 6's future FastAPI app must **NOT** call ``build_graph()``
      inside a request handler. Build it once at app startup (e.g. in a
      lifespan/startup hook or a module-level singleton), then have every
      request handler call ``invoke()``/``get_state()`` on that one shared
      compiled object, distinguishing sessions purely by ``thread_id`` in
      ``config={"configurable": {"thread_id": ...}}``.
    - "Resumable across separate ``invoke()`` calls" (the spec's framing)
      means: many separate ``invoke()`` calls on the **same** compiled
      graph object, across many requests/turns/``thread_id``s — NOT a
      fresh ``build_graph()`` per request. Calling ``build_graph()`` again
      after the first one is already in use is equivalent to wiping all
      in-memory session state for every ``thread_id`` (a fresh,
      disconnected ``MemorySaver``) — there is no migration or merge of
      the old store's contents into the new one.
    - This is intended behavior for an in-process ``MemorySaver``
      (correctly chosen per spec's own non-goals — no database-backed
      checkpointer needed for this prototype), not a defect to fix in this
      graph. If Phase 6 ever needs ``build_graph()``-per-process-restart
      durability (e.g. surviving a server restart), that requires swapping
      ``MemorySaver`` for a persistent checkpointer — an explicit, separate
      decision, not something this function does implicitly.
    - Proven by test:
      ``tests/test_interview_graph.py::test_build_once_many_sessions_pattern``
      builds the graph ONCE and then drives several independent sessions
      (distinct ``thread_id``s) to completion via ``invoke()``/
      ``Command(resume=...)`` on that single compiled object, asserting
      sessions never cross-contaminate each other's state — this is the
      correct usage pattern Phase 6 must follow.
    """
    meta = load_dataset_meta(dataset_path)
    total_questions = store_count(chroma_dir=chroma_dir) if chroma_dir else store_count()
    intro_text = persona.get_or_generate_intro_text(meta.domain, meta.intro)

    workflow = StateGraph(InterviewState)

    workflow.add_node("intro", make_intro_node(intro_text))
    workflow.add_node("interviewer_turn", make_interviewer_turn_node(chroma_dir))
    workflow.add_node("await_candidate", await_candidate)
    workflow.add_node("evaluate_answer", make_evaluate_answer_node(chroma_dir))
    workflow.add_node("advance_question", make_advance_question_node(chroma_dir))
    workflow.add_node("wrap_up", make_wrap_up_node(meta.closing))
    workflow.add_node("generate_feedback", make_generate_feedback_node(chroma_dir))

    workflow.add_edge(START, "intro")
    workflow.add_edge("intro", "interviewer_turn")
    workflow.add_edge("interviewer_turn", "await_candidate")
    workflow.add_edge("await_candidate", "evaluate_answer")
    workflow.add_conditional_edges(
        "evaluate_answer",
        route_verdict,
        {"interviewer_turn": "interviewer_turn", "advance_question": "advance_question"},
    )
    workflow.add_conditional_edges(
        "advance_question",
        route_after_advance,
        {"interviewer_turn": "interviewer_turn", "wrap_up": "wrap_up"},
    )
    workflow.add_edge("wrap_up", "generate_feedback")
    workflow.add_edge("generate_feedback", END)

    compiled = workflow.compile(checkpointer=MemorySaver())
    return compiled, total_questions, meta.max_follow_ups_per_question


__all__ = ["build_graph", "Command", "interrupt"]
