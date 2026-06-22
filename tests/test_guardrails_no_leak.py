"""Unit tests for app/guardrails/no_leak.py — the no-verbatim-leak custom
Guardrails validator. Loads the real q03 (Big-O) entry from
``data/qa_bank.yaml`` via the sanctioned loader (never hand-copies the
reference text), per spec.md's explicit instruction.
"""

from __future__ import annotations

import time

import pytest
from guardrails import OnFailAction

from app.guardrails.no_leak import (
    FUZZY_THRESHOLD,
    NGRAM_SIZE,
    NoLeakValidator,
    build_no_leak_guard,
    is_leaking,
)
from app.retrieval.ingest import DEFAULT_QA_BANK_PATH
from app.retrieval.loader import load_qa_bank


def _q03_ideal_answer() -> str:
    entries = load_qa_bank(DEFAULT_QA_BANK_PATH)
    q03 = next(e for e in entries if e.id == "q03")
    return q03.ideal_answer


# ---------------------------------------------------------------------------
# is_leaking() — pure detection function
# ---------------------------------------------------------------------------


def test_literal_verbatim_leak_is_detected():
    """A string containing the literal q03 ideal_answer (real dataset
    content, loaded via the loader) must be flagged."""
    ideal_answer = _q03_ideal_answer()
    candidate_text = f"Great question. {ideal_answer} Does that make sense?"
    assert is_leaking(candidate_text, ideal_answer) is True


def test_literal_verbatim_leak_exact_match_is_detected():
    ideal_answer = _q03_ideal_answer()
    assert is_leaking(ideal_answer, ideal_answer) is True


def test_partial_verbatim_sentence_leak_is_detected():
    """A single full sentence lifted verbatim from the ideal_answer, even
    embedded in otherwise-original text, must be flagged (the n-gram-overlap
    layer, not just whole-text containment)."""
    ideal_answer = _q03_ideal_answer()
    first_sentence = ideal_answer.split(".")[0] + "."
    candidate_text = (
        f"Let me put it this way: {first_sentence} That's the core idea."
    )
    assert is_leaking(candidate_text, ideal_answer) is True


def test_near_verbatim_synonym_substituted_paraphrase_is_detected():
    """A paraphrase that keeps the same sentence structure/length and only
    swaps a handful of words for synonyms must still be caught by the fuzzy
    layer, not just the exact-substring layer."""
    ideal_answer = _q03_ideal_answer()
    near_verbatim = (
        ideal_answer.replace("describes", "explains")
        .replace("worst case", "worst scenario")
        .replace("ignoring constant factors", "disregarding constant terms")
        .replace("check every element", "inspect every item")
    )
    assert near_verbatim != ideal_answer  # sanity: we actually changed it
    assert is_leaking(near_verbatim, ideal_answer) is True


def test_genuine_paraphrase_covering_same_key_points_is_not_flagged():
    """**False-positive guard (spec.md's explicit requirement).** A
    materially different rephrasing that still conveys the same key points
    must PASS — an over-aggressive guard that blocks every grounded,
    original interviewer phrasing is as wrong as a leaky one."""
    ideal_answer = _q03_ideal_answer()
    paraphrase = (
        "Big-O is a way to express how the cost of an algorithm scales as "
        "the input grows, focusing on the dominant term rather than exact "
        "constants. A simple scan through a list to find something takes "
        "time proportional to the list size. A search that exploits sorted "
        "order can instead cut the remaining candidates in half on every "
        "comparison, which is why it finishes in a logarithmic number of "
        "steps instead of a linear one."
    )
    assert is_leaking(paraphrase, ideal_answer) is False


def test_short_unrelated_reply_is_not_flagged():
    ideal_answer = _q03_ideal_answer()
    assert is_leaking("I'm not sure, can you give me a hint?", ideal_answer) is False


def test_empty_strings_do_not_crash_and_are_not_flagged():
    assert is_leaking("", "something") is False
    assert is_leaking("something", "") is False
    assert is_leaking("", "") is False


# ---------------------------------------------------------------------------
# Threshold sanity (documents WHY 10 / 0.72 were chosen, per plan.md)
# ---------------------------------------------------------------------------


def test_thresholds_are_the_documented_values():
    """Pins the constants the plan.md rationale is written against — if
    these ever change, the rationale doc must be revisited too."""
    assert NGRAM_SIZE == 10
    assert FUZZY_THRESHOLD == 0.72


# ---------------------------------------------------------------------------
# NoLeakValidator / build_no_leak_guard — the real Guardrails machinery
# ---------------------------------------------------------------------------


def test_validator_with_fix_on_fail_substitutes_fix_value():
    ideal_answer = _q03_ideal_answer()
    guard = build_no_leak_guard(
        ideal_answer, on_fail=OnFailAction.FIX, fix_value="[REDACTED]"
    )
    outcome = guard.validate(ideal_answer)
    assert outcome.validation_passed is True  # FIX still reports "passed" (fixed)
    assert outcome.validated_output == "[REDACTED]"


def test_validator_with_exception_on_fail_raises_on_leak():
    ideal_answer = _q03_ideal_answer()
    guard = build_no_leak_guard(ideal_answer, on_fail=OnFailAction.EXCEPTION)
    with pytest.raises(Exception):
        guard.validate(ideal_answer)


def test_validator_with_exception_on_fail_passes_clean_text():
    ideal_answer = _q03_ideal_answer()
    guard = build_no_leak_guard(ideal_answer, on_fail=OnFailAction.EXCEPTION)
    outcome = guard.validate("A totally unrelated reply with no leak at all.")
    assert outcome.validation_passed is True


def test_validator_non_string_value_passes_through(monkeypatch):
    """Defensive: _validate's isinstance guard means a non-string value
    (shouldn't normally reach this validator, but defensive nonetheless)
    does not crash and is treated as a pass."""
    validator = NoLeakValidator(reference="anything", on_fail=OnFailAction.EXCEPTION)
    result = validator._validate(12345, {})
    assert result.outcome == "pass"


# ---------------------------------------------------------------------------
# Latency (plan.md Section 8's latency note, backed by a real measurement)
# ---------------------------------------------------------------------------


def test_is_leaking_latency_is_sub_100ms():
    """Generous bound (not a tight perf assertion) confirming the
    deterministic substring+fuzzy check adds negligible latency, independent
    of any gateway/reask path — backs plan.md Section 8's latency claim."""
    ideal_answer = _q03_ideal_answer()
    candidate_text = (
        "A reasonably long but unrelated candidate answer about something "
        "else entirely, long enough to exercise the fuzzy ratio path fully."
    )
    start = time.perf_counter()
    for _ in range(50):
        is_leaking(candidate_text, ideal_answer)
    elapsed_ms = (time.perf_counter() - start) / 50 * 1000
    assert elapsed_ms < 100
