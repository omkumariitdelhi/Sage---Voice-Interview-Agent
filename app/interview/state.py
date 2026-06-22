"""Typed state schema for the interview graph.

See ``.claude/loop-state/phase-4-interview-graph/plan.md`` Section 1 for the
full reducer-choice rationale. Short version: ``transcript`` accumulates via
``operator.add`` (every node that produces a turn appends one record); every
other field is a point-in-time pointer/counter and uses LangGraph's default
replace-on-update semantics (no ``Annotated`` reducer needed).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

# The four possible grades the interviewer LLM can give a candidate's answer
# to the *current* question. "off_topic" is distinct from "wrong" in the
# transcript/feedback (see plan.md Section 4) even though both consume the
# same follow_up_count budget.
AnswerVerdict = Literal["strong", "weak", "wrong", "off_topic"]

# The high-level phase the graph is in — drives which prompt
# `interviewer_turn` builds and is informational for debugging/tests.
InterviewPhase = Literal[
    "intro",
    "asking",
    "follow_up",
    "redirect",
    "advancing",
    "wrap_up",
    "done",
]

Speaker = Literal["interviewer", "candidate"]


class TurnRecord(TypedDict):
    """One line of the session transcript (one interviewer turn OR one
    candidate reply)."""

    speaker: Speaker
    text: str
    qa_id: str | None
    verdict: AnswerVerdict | None


class QuestionFeedback(BaseModel):
    """Per-question slice of the final feedback report."""

    qa_id: str
    topic: str
    verdict: AnswerVerdict
    note: str = Field(description="A brief, specific note on this answer.")


class FeedbackReport(BaseModel):
    """Structured end-of-session feedback. Produced and validated by
    :func:`app.interview.feedback.generate_feedback` — this is the **public
    API boundary type**.

    NOT stored directly in checkpointed graph state (see
    ``InterviewState.feedback``'s comment below for why — msgpack
    serializer compatibility). ``graph.py``'s ``generate_feedback_node``
    converts an instance of this model to a plain dict via
    ``.model_dump()`` before returning it as the ``feedback`` state key; a
    caller that wants the typed object back can reconstruct it with
    ``FeedbackReport.model_validate(state["feedback"])``.
    """

    per_question: list[QuestionFeedback]
    overall_strengths: list[str]
    overall_improvements: list[str]


class InterviewState(TypedDict):
    """Full graph state. See plan.md Section 1 for the reducer rationale."""

    # --- Position / bookkeeping (scalars, default replace) ---
    qa_index: int
    follow_up_count: int
    total_turns: int
    total_questions: int
    max_follow_ups: int
    phase: InterviewPhase

    # --- Most recent evaluation result (scalars, default replace) ---
    last_verdict: AnswerVerdict | None
    last_missed_points: list[str]

    # --- Append-only per-question final-verdict ledger (reducer:
    # operator.add). One (qa_id, verdict) pair is appended by
    # `advance_question` each time a question is finished (whether by a
    # strong answer or by exhausting the follow-up budget) — this is the
    # ground truth `generate_feedback` uses to build each per-question
    # report line, without needing to mutate any historical `transcript`
    # entry (transcript stays strictly append-only). ---
    question_verdicts: Annotated[list[tuple[str, AnswerVerdict]], operator.add]

    # --- The candidate's most recent utterance, consumed then left in place
    # for transcript/debugging purposes (scalar, default replace) ---
    candidate_answer: str | None

    # --- Internal carrier: which qa_id the just-emitted interviewer turn
    # belongs to, read by the very next node (`await_candidate`) only.
    # Scalar, default replace. Not part of the public API/spec's state
    # vocabulary; an implementation seam between two adjacent nodes. ---
    _pending_qa_id: str | None

    # --- Append-only session log (reducer: operator.add / concatenation) ---
    transcript: Annotated[list[TurnRecord], operator.add]

    # --- Final artifact, set only once at the very end (scalar, default
    # replace). ---
    #
    # Stored as a PLAIN DICT (``FeedbackReport.model_dump()`` shape), not the
    # Pydantic ``FeedbackReport`` model itself — see review-01.md Finding 2
    # (Minor, revision iteration 1). MemorySaver's checkpoint serializer
    # round-trips ``InterviewState`` through msgpack; an unregistered
    # Pydantic ``BaseModel`` stored directly in checkpointed state hits an
    # unregistered-type fallback path on deserialize and emits:
    # "Deserializing unregistered type app.interview.state.FeedbackReport
    # from checkpoint. This will be blocked in a future version." A plain
    # dict of JSON-safe primitives (str/list/dict) is natively
    # msgpack-serializable with no fallback/warning, and is also the more
    # idiomatic TypedDict-state shape (the rest of InterviewState is already
    # plain dicts/lists/scalars — FeedbackReport was the only Pydantic model
    # living inside checkpointed state). ``FeedbackReport`` (the Pydantic
    # model) still exists and is still the *public API boundary* type: it is
    # constructed and validated inside
    # :func:`app.interview.feedback.generate_feedback`, and any caller that
    # wants the typed object back reconstructs it via
    # ``FeedbackReport.model_validate(state["feedback"])`` — graph.py's
    # ``generate_feedback_node`` is the one place that converts
    # model -> dict (via ``.model_dump()``) before it ever touches
    # checkpointed state. ---
    feedback: dict | None


def initial_state(*, total_questions: int, max_follow_ups: int) -> InterviewState:
    """Build the state dict the very first ``invoke()`` of a session should
    pass in. ``total_questions``/``max_follow_ups`` are snapshotted once at
    session start (from ``app.retrieval.store.count()`` and the dataset
    loader's meta accessor's ``max_follow_ups_per_question`` field
    respectively) so the bounded-termination ceiling is fixed for the
    lifetime of the session even if the underlying dataset changes mid-run.
    """
    return InterviewState(
        qa_index=0,
        follow_up_count=0,
        total_turns=0,
        total_questions=total_questions,
        max_follow_ups=max_follow_ups,
        phase="intro",
        last_verdict=None,
        last_missed_points=[],
        question_verdicts=[],
        candidate_answer=None,
        _pending_qa_id=None,
        transcript=[],
        feedback=None,
    )


def max_turns_ceiling(*, total_questions: int, max_follow_ups: int) -> int:
    """The provable hard ceiling on interviewer turns for a full session.

    ``MAX_TURNS = total_questions * (max_follow_ups + 1)``

    Per question, the worst case is ``max_follow_ups`` follow-up/redirect
    turns plus the 1 initial ask before ``route_verdict`` force-advances
    regardless of verdict. The one-time ``intro`` turn and the final
    ``wrap_up``/feedback step are outside this budget (each happens exactly
    once, unconditionally, never depends on candidate behavior) — see
    plan.md Section 3.
    """
    return total_questions * (max_follow_ups + 1)
