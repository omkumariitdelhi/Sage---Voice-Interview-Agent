"""Runnable example: drives the compiled interview graph end-to-end with a
scripted set of candidate answers and a fully mocked LLM gateway.

Run from the project root (module form, matching this project's other
examples — a direct ``python examples\\run_interview_example.py`` invocation
does not put the project root on ``sys.path``):

    .venv\\Scripts\\python.exe -m examples.run_interview_example

Prints the full transcript turn-by-turn as the session progresses, then the
final structured feedback report. Makes zero real network calls — both
gateway call sites (``app.interview.graph.complete`` for interviewer turns,
``app.interview.evaluation.complete`` for grading, and
``app.interview.feedback.complete`` for the final report) are monkeypatched
with a tiny scripted fake that returns canned, realistic-shaped responses.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from langgraph.types import Command

from app.interview.graph import build_graph
from app.interview.state import initial_state

# One scripted candidate answer per turn, in the order the graph will ask for
# them. Mix of strong / weak / off-topic / wrong to exercise every path, with
# enough strong answers to reach the end of the 10-question bank and the
# final feedback report within this demo's generous turn budget.
SCRIPTED_ANSWERS = [
    "I clarified scope with my PM, broke it into two sub-problems, "
    "compared a queue-based vs a polling approach, picked the queue for "
    "lower latency, added tests, and in hindsight would have load-tested "
    "earlier.",  # q01 strong
    "Arrays are contiguous so indexing is fast, linked lists use pointers.",  # q02 weak (incomplete)
    "Arrays give O(1) random access and are cache-friendly; linked lists "
    "give O(n) traversal but O(1) insert/delete at a known node, e.g. "
    "arrays for a lookup table, linked lists for a frequently-mutated queue.",  # q02 follow-up -> strong
    "I really like pizza, especially pepperoni.",  # q03 off-topic
    "Big-O describes worst-case growth with input size; linear search is "
    "O(n), binary search is O(log n) because it halves the search space "
    "each step, but it requires sorted input.",  # q03 redirect -> strong
    "An abstract class is just a regular class but cooler.",  # q04 wrong
    "Abstract classes share implementation/state and support single "
    "inheritance, fitting an is-a hierarchy; interfaces are pure contracts "
    "a class can implement many of, fitting a can-do capability.",  # q04 follow-up -> strong
    "SQL is for relational data with ACID guarantees; NoSQL trades some "
    "consistency for scale and flexible schema, e.g. document stores for "
    "rapidly evolving data.",  # q05 strong
    "A URL shortener needs a write path that generates a unique short code "
    "(base62 of an id, with collision handling) and a read path that looks "
    "it up and redirects; at scale, cache the hot set and partition the "
    "datastore.",  # q06 strong
    "A race condition is unsynchronized concurrent access to shared "
    "mutable state; prevent it with locks around the critical section, "
    "atomics for counters, or immutable/thread-local data.",  # q07 strong
    "I'd start from evidence — logs, error rates, recent deploys — form a "
    "hypothesis, narrow scope with tracing, and validate the fix against "
    "the same evidence before closing it.",  # q08 strong
    "Unit tests isolate one function with mocks and are fast/precise; "
    "integration tests exercise real components together and catch wiring "
    "issues; you need both for different failure modes.",  # q09 strong
    "git merge creates a merge commit preserving both histories; git "
    "rebase replays commits for a linear history but rewrites hashes, so "
    "use merge for shared branches and rebase to clean up a local branch.",  # q10 strong
]


def _fake_turn_complete(messages: list[dict], **kwargs) -> str:
    """Stands in for app.interview.graph.complete (interviewer turn text)."""
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    return f"[interviewer] {last_user.splitlines()[0][:80]}"


_EVAL_SCRIPT = iter(
    [
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q01
        {"verdict": "weak", "missed_key_points": ["cache-friendliness"], "reasoning": "partial"},  # q02 attempt 1
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it now"},  # q02 attempt 2
        {"verdict": "off_topic", "missed_key_points": [], "reasoning": "not related"},  # q03 attempt 1
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q03 attempt 2 (after redirect)
        {"verdict": "wrong", "missed_key_points": ["shared implementation"], "reasoning": "incorrect"},  # q04 attempt 1
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it now"},  # q04 attempt 2
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q05
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q06
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q07
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q08
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q09
        {"verdict": "strong", "missed_key_points": [], "reasoning": "covers it"},  # q10
    ]
)


def _fake_eval_complete(messages: list[dict], **kwargs) -> str:
    """Stands in for app.interview.evaluation.complete (grading calls)."""
    return json.dumps(next(_EVAL_SCRIPT))


def _fake_feedback_complete(messages: list[dict], **kwargs) -> str:
    """Stands in for app.interview.feedback.complete (final report)."""
    return json.dumps(
        {
            "per_question": [
                {
                    "qa_id": "q01",
                    "topic": "behavioral",
                    "verdict": "strong",
                    "note": "Clear structured walkthrough with trade-offs and reflection.",
                },
                {
                    "qa_id": "q02",
                    "topic": "data-structures",
                    "verdict": "weak",
                    "note": "Missed cache-friendliness and a concrete scenario.",
                },
            ],
            "overall_strengths": ["Clear communication", "Structured problem solving"],
            "overall_improvements": ["Go deeper on trade-offs", "Use concrete examples"],
        }
    )


def main() -> None:
    with patch("app.interview.graph.complete", side_effect=_fake_turn_complete), patch(
        "app.interview.evaluation.complete", side_effect=_fake_eval_complete
    ), patch("app.interview.feedback.complete", side_effect=_fake_feedback_complete):
        compiled, total_questions, max_follow_ups = build_graph()
        config = {"configurable": {"thread_id": "example-session"}}
        state = initial_state(total_questions=total_questions, max_follow_ups=max_follow_ups)

        result = compiled.invoke(state, config)
        _print_new_turns(result)

        for answer in SCRIPTED_ANSWERS:
            snapshot = compiled.get_state(config)
            if not snapshot.next:
                break
            print(f"\n>>> candidate: {answer}")
            result = compiled.invoke(Command(resume=answer), config)
            _print_new_turns(result)

            if result.get("phase") == "done":
                break

        print("\n=== FINAL FEEDBACK REPORT ===")
        # `feedback` is stored in checkpointed graph state as a plain dict
        # (FeedbackReport.model_dump() shape), not the Pydantic model itself
        # — see app/interview/state.py's InterviewState.feedback comment and
        # review-01.md Finding 2 (msgpack checkpoint-serializer
        # compatibility). Reconstruct the typed model here at this script's
        # own "API boundary" purely to demonstrate the round-trip; printing
        # the dict directly would work just as well.
        feedback = result.get("feedback")
        if feedback is not None:
            from app.interview.state import FeedbackReport

            print(FeedbackReport.model_validate(feedback).model_dump_json(indent=2))
        else:
            print("(no feedback produced — session did not reach 'done')")


_last_printed = 0


def _print_new_turns(state: dict) -> None:
    global _last_printed
    transcript = state.get("transcript", [])
    for turn in transcript[_last_printed:]:
        speaker = turn["speaker"]
        suffix = f" [verdict={turn['verdict']}]" if turn["verdict"] else ""
        print(f"{speaker}: {turn['text']}{suffix}")
    _last_printed = len(transcript)


if __name__ == "__main__":
    main()
