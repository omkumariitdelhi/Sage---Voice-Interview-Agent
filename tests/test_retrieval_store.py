"""Tests for app/retrieval/store.py — the runtime read API.

Covers the spec's exact-match-retrieval, determinism/field-integrity, and
empty-retrieval acceptance criteria.
"""

from __future__ import annotations

import shutil

import pytest

from app.retrieval.exceptions import QAEntryNotFoundError, StoreNotIngestedError
from app.retrieval.ingest import build_store
from app.retrieval.store import _open_store, count, get_by_id, get_by_sequence_index, retrieve

Q03_QUESTION_TEXT = (
    "What is Big-O notation, and what's the time complexity of binary search "
    "versus linear search? Why the difference?"
)


@pytest.fixture
def ingested_store(tmp_path):
    """A fresh Chroma store built from the real `data/qa_bank.yaml`, in an
    isolated temp directory."""
    chroma_dir = tmp_path / "chroma_store"
    build_store(chroma_dir=chroma_dir)
    yield chroma_dir
    shutil.rmtree(chroma_dir, ignore_errors=True)


def test_retrieve_exact_question_text_returns_top1_match(ingested_store):
    """The spec's core proof that real embedding similarity is happening,
    not a stub: searching q03's literal question text returns q03 top-1."""
    results = retrieve(Q03_QUESTION_TEXT, k=1, chroma_dir=ingested_store)

    assert len(results) == 1
    assert results[0].id == "q03"
    assert results[0].topic == "complexity-analysis"


def test_get_by_id_and_get_by_sequence_index_agree_with_full_fields(ingested_store):
    """get_by_id('q01') and get_by_sequence_index(0) must return the same
    entry deterministically, with every field intact through the vector
    store round trip."""
    by_id = get_by_id("q01", chroma_dir=ingested_store)
    by_seq = get_by_sequence_index(0, chroma_dir=ingested_store)

    assert by_id == by_seq
    assert by_id.id == "q01"
    assert by_id.topic == "behavioral"
    assert by_id.difficulty == "easy"
    assert "challenging technical problem" in by_id.question
    assert "clarifies the problem and constraints" in by_id.ideal_answer
    assert len(by_id.key_points) == 5
    assert by_id.key_points[0] == (
        "Clarified the problem/constraints before designing a solution"
    )


def test_get_by_id_unknown_raises(ingested_store):
    with pytest.raises(QAEntryNotFoundError):
        get_by_id("does-not-exist", chroma_dir=ingested_store)


def test_get_by_sequence_index_out_of_range_raises(ingested_store):
    with pytest.raises(QAEntryNotFoundError):
        get_by_sequence_index(999, chroma_dir=ingested_store)


def test_count_matches_qa_bank_size(ingested_store):
    assert count(chroma_dir=ingested_store) == 10


def test_store_not_ingested_raises_clear_error(tmp_path):
    """Reading from a chroma dir that was never built must raise a typed
    error, not a confusing low-level Chroma exception."""
    never_built_dir = tmp_path / "never_built"

    with pytest.raises(StoreNotIngestedError):
        count(chroma_dir=never_built_dir)
    with pytest.raises(StoreNotIngestedError):
        get_by_id("q01", chroma_dir=never_built_dir)
    with pytest.raises(StoreNotIngestedError):
        retrieve("anything", chroma_dir=never_built_dir)


def test_retrieve_semantic_followup_returns_relevant_entry(ingested_store):
    """A free-text query that doesn't literally match any question's text
    should still retrieve a semantically related entry — proves this serves
    the free-text grounding consumer, not just exact-text lookups."""
    results = retrieve(
        "How do I stop two threads from corrupting the same shared variable?",
        k=1,
        chroma_dir=ingested_store,
    )
    assert len(results) == 1
    assert results[0].id == "q07"  # race condition / concurrency


def test_retrieve_on_built_but_empty_store_returns_empty_list(ingested_store):
    """Distinct from `test_store_not_ingested_raises_clear_error`: this store
    *was* ingested (the dir/collection exists) but every entry has since been
    removed, leaving 0 live entries. `retrieve()` must return `[]` and must
    NOT raise — this is the empty-retrieval contract documented in
    `store.py::retrieve`'s docstring, previously verified only by manual
    review (review-01.md Minor finding #2), not by any test."""
    all_ids = [f"q{n:02d}" for n in range(1, 11)]

    store = _open_store(chroma_dir=ingested_store)
    store.delete(ids=all_ids)
    assert count(chroma_dir=ingested_store) == 0

    results = retrieve("anything at all", k=3, chroma_dir=ingested_store)
    assert results == []
