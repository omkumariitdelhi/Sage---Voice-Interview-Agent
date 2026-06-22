"""RETR-03 — the live-edit proof. The single most important test in this
phase (per spec): editing `data/qa_bank.yaml`-shaped content and re-running
ingest must make the new/changed content retrievable, with **zero** changes
to any file under `app/retrieval/`. This test only writes a temp YAML file
and calls the existing public `build_store`/`store` functions — it never
touches application source.
"""

from __future__ import annotations

import shutil

import yaml

from app.retrieval.ingest import build_store
from app.retrieval.store import count, get_by_id, retrieve

_BASE_QUESTIONS = [
    {
        "id": "q01",
        "topic": "behavioral",
        "difficulty": "easy",
        "question": "Tell me about a challenging technical problem you solved.",
        "ideal_answer": "A strong answer clarifies the problem first.",
        "key_points": ["Clarifies the problem first"],
    },
    {
        "id": "q02",
        "topic": "data-structures",
        "difficulty": "easy",
        "question": "What's the difference between an array and a linked list?",
        "ideal_answer": "Arrays give O(1) access; linked lists give O(1) insert.",
        "key_points": ["O(1) random access for arrays"],
    },
]


def _write_qa_bank(path, questions):
    payload = {
        "domain": "Test domain",
        "intro": "Test intro",
        "closing": "Test closing",
        "max_follow_ups_per_question": 2,
        "questions": questions,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_live_edit_new_question_is_retrievable_after_reingest(tmp_path):
    """Add an 11th (here, 3rd, to keep the fixture small) brand-new question
    to a temp dataset, re-run ingest against that temp file/temp store, and
    assert the new content is now retrievable — without touching
    app/retrieval/ at all."""
    qa_bank_path = tmp_path / "qa_bank_live_edit.yaml"
    chroma_dir = tmp_path / "chroma_live_edit"

    # --- Step 1: ingest the original (2-question) dataset. ---
    _write_qa_bank(qa_bank_path, _BASE_QUESTIONS)
    build_store(qa_bank_path=qa_bank_path, chroma_dir=chroma_dir)
    assert count(chroma_dir=chroma_dir) == 2

    # A query for content that doesn't exist yet must not return it.
    not_yet = retrieve(
        "How does a hash table resolve collisions between two keys?",
        k=1,
        chroma_dir=chroma_dir,
    )
    assert not_yet[0].id != "q11"  # nothing relevant ingested yet

    # --- Step 2: programmatically edit the dataset — add a new question. ---
    new_question = {
        "id": "q11",
        "topic": "data-structures",
        "difficulty": "medium",
        "question": "How does a hash table resolve collisions between two keys?",
        "ideal_answer": (
            "Common strategies are chaining (a linked list/bucket per slot) "
            "and open addressing (probing for the next free slot)."
        ),
        "key_points": [
            "Names chaining as one collision strategy",
            "Names open addressing/probing as another",
        ],
    }
    updated_questions = _BASE_QUESTIONS + [new_question]
    _write_qa_bank(qa_bank_path, updated_questions)

    # --- Step 3: re-run ingest against the SAME temp file/temp store path. ---
    ingested = build_store(qa_bank_path=qa_bank_path, chroma_dir=chroma_dir)
    assert ingested == 3

    # --- Step 4: assert the new content is now retrievable. ---
    assert count(chroma_dir=chroma_dir) == 3
    fetched = get_by_id("q11", chroma_dir=chroma_dir)
    assert fetched.question == new_question["question"]
    assert fetched.key_points == new_question["key_points"]

    results = retrieve(
        "How does a hash table resolve collisions between two keys?",
        k=1,
        chroma_dir=chroma_dir,
    )
    assert results[0].id == "q11"

    shutil.rmtree(chroma_dir, ignore_errors=True)


def test_live_edit_changed_wording_is_reflected_after_reingest(tmp_path):
    """Change an existing question's wording in a temp dataset, re-run
    ingest, and assert the *new* wording is what's now retrievable —
    proving edits (not just additions) propagate with zero app code
    changes."""
    qa_bank_path = tmp_path / "qa_bank_edit_wording.yaml"
    chroma_dir = tmp_path / "chroma_edit_wording"

    _write_qa_bank(qa_bank_path, _BASE_QUESTIONS)
    build_store(qa_bank_path=qa_bank_path, chroma_dir=chroma_dir)

    original = get_by_id("q01", chroma_dir=chroma_dir)
    assert "challenging technical problem" in original.question

    edited_questions = [dict(q) for q in _BASE_QUESTIONS]
    edited_questions[0] = {
        **edited_questions[0],
        "question": (
            "Describe the most complex distributed-systems bug you have "
            "ever debugged and how you tracked it down."
        ),
    }
    _write_qa_bank(qa_bank_path, edited_questions)

    build_store(qa_bank_path=qa_bank_path, chroma_dir=chroma_dir)

    updated = get_by_id("q01", chroma_dir=chroma_dir)
    assert "distributed-systems bug" in updated.question
    assert "challenging technical problem" not in updated.question

    results = retrieve(
        "Describe the most complex distributed-systems bug you have ever "
        "debugged and how you tracked it down.",
        k=1,
        chroma_dir=chroma_dir,
    )
    assert results[0].id == "q01"
    assert "distributed-systems bug" in results[0].question

    shutil.rmtree(chroma_dir, ignore_errors=True)
