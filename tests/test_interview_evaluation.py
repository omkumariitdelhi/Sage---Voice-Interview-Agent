"""Tests for app/interview/evaluation.py.

The markdown-fence regression test below reproduces a real bug found via
live testing (not the mocked suite, which never exercised this until now):
Claude Haiku 4.5 wrapped its JSON-mode verdict in a ```json fence even after
the single re-ask-on-malformed-JSON retry explicitly asked for "no markdown
fences" — see app/json_mode.py's module docstring for the confirmed root
cause (response_format={"type": "json_object"} doesn't structurally enforce
JSON for the Anthropic provider without an explicit json_schema). Before the
fix, this silently degraded a correct "wrong" verdict (with real missed_key_
points and reasoning) to the generic "weak" fallback.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from app.interview.evaluation import evaluate_answer
from app.retrieval.schema import QAEntry

_ENTRY = QAEntry(
    id="q-test",
    topic="data-structures",
    difficulty="easy",
    question="Array vs linked list?",
    ideal_answer="Arrays give O(1) random access; linked lists give O(n).",
    key_points=["O(1) array access", "O(n) linked list traversal"],
)


def _fenced(verdict: str, missed: list[str] | None = None) -> str:
    payload = json.dumps(
        {"verdict": verdict, "missed_key_points": missed or [], "reasoning": "mock"}
    )
    return f"```json\n{payload}\n```"


def test_evaluate_answer_parses_markdown_fenced_verdict():
    """Regression test for the live-found bug: a fenced JSON verdict must
    still parse correctly on the FIRST attempt, not silently fall back to
    'weak' and not need the reask-retry at all."""
    with patch(
        "app.interview.evaluation.complete",
        return_value=_fenced("wrong", ["O(1) array access"]),
    ) as mock_complete:
        result = evaluate_answer(_ENTRY, "a wrong answer", [])

    mock_complete.assert_called_once()  # no retry needed
    assert result.verdict == "wrong"
    assert result.missed_key_points == ["O(1) array access"]
    assert result.reasoning == "mock"


def test_evaluate_answer_still_parses_unfenced_verdict():
    """Sanity check: plain (unfenced) JSON-mode output keeps working
    unchanged."""
    payload = json.dumps(
        {"verdict": "strong", "missed_key_points": [], "reasoning": "good"}
    )
    with patch("app.interview.evaluation.complete", return_value=payload):
        result = evaluate_answer(_ENTRY, "a strong answer", [])

    assert result.verdict == "strong"
