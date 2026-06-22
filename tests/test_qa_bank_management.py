"""Tests for the Q&A bank management feature — manual CRUD + document-upload
extraction + bulk commit (see
``.claude/loop-state/qa-bank-management/spec.md`` for the full contract).

Every test drives the API through Starlette's ``TestClient`` (real HTTP
request/response objects), mirroring ``tests/test_web_session.py``'s own
convention. Each test gets its own isolated ``tmp_path`` copy of
``data/qa_bank.yaml`` + a freshly-ingested Chroma store, so **no test ever
touches the project's real dataset file or real ``.chroma/`` directory**.

Mocking chokepoints: ``app.retrieval.extractor.complete`` (the gateway call
site this feature's own module imports by name) — never the live LLM.
"""

from __future__ import annotations

import io
import json
import shutil
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.retrieval.ingest import build_store
from app.web.server import create_app

REAL_QA_BANK_PATH = "data/qa_bank.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qa_bank_copy(tmp_path):
    """A writable per-test copy of the REAL data/qa_bank.yaml (10 questions,
    real header comment block) — every CRUD/bulk test mutates this copy,
    never the project's real file."""
    dest = tmp_path / "qa_bank_under_test.yaml"
    shutil.copy(REAL_QA_BANK_PATH, dest)
    return dest


@pytest.fixture
def ingested_store(qa_bank_copy, tmp_path):
    chroma_dir = tmp_path / "qa_bank_chroma_store"
    build_store(qa_bank_path=qa_bank_copy, chroma_dir=chroma_dir)
    yield chroma_dir
    shutil.rmtree(chroma_dir, ignore_errors=True)


@pytest.fixture
def app_and_client(qa_bank_copy, ingested_store):
    app = create_app(dataset_path=qa_bank_copy, chroma_dir=ingested_store)
    with TestClient(app) as client:
        yield app, client


def _mock_turn_text(messages, **kwargs):
    return "MOCK_INTERVIEWER_TURN"


def _eval_json(verdict: str = "strong") -> str:
    return json.dumps({"verdict": verdict, "missed_key_points": [], "reasoning": "mock"})


def _feedback_json() -> str:
    return json.dumps(
        {
            "per_question": [
                {"qa_id": "q01", "topic": "behavioral", "verdict": "strong", "note": "ok"}
            ],
            "overall_strengths": ["Clear"],
            "overall_improvements": ["Deeper"],
        }
    )


def _valid_entry(suffix: str = "1") -> dict:
    return {
        "id": None,
        "topic": f"topic-{suffix}",
        "difficulty": "medium",
        "question": f"What is concept {suffix}?",
        "ideal_answer": f"Concept {suffix} is explained here.",
        "key_points": [f"key point {suffix}"],
    }


def _make_text_pdf(text: bytes) -> bytes:
    """Build a tiny, real, valid PDF in-memory using only pypdf's own
    PdfWriter + generic primitives (no extra PDF-writing dependency) —
    one page with a single text-showing content stream so
    ``PdfReader(...).pages[0].extract_text()`` returns ``text`` back.
    Mirrors the introspection done in plan.md Section 2 verifying pypdf's
    current API shape."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    content = b"BT /F1 16 Tf 20 100 Td (" + text + b") Tj ET"
    stream_obj = DecodedStreamObject()
    stream_obj.set_data(content)
    stream_ref = writer._add_object(stream_obj)

    font_dict = DictionaryObject()
    font_dict.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font_dict)
    resources = DictionaryObject()
    font_resources = DictionaryObject()
    font_resources[NameObject("/F1")] = font_ref
    resources[NameObject("/Font")] = font_resources
    page[NameObject("/Resources")] = resources
    page[NameObject("/Contents")] = stream_ref

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Manual CRUD
# ---------------------------------------------------------------------------


def test_get_qa_bank_returns_all_entries_and_meta(app_and_client):
    _app, client = app_and_client

    response = client.get("/api/qa-bank")

    assert response.status_code == 200
    body = response.json()
    assert len(body["questions"]) == 10
    assert body["max_follow_ups_per_question"] == 2


def test_post_creates_entry_with_explicit_id(app_and_client):
    _app, client = app_and_client

    payload = _valid_entry("explicit")
    payload["id"] = "q_custom"
    response = client.post("/api/qa-bank", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "q_custom"
    assert body["topic"] == "topic-explicit"


def test_post_creates_entry_without_explicit_id_auto_generates(app_and_client):
    _app, client = app_and_client

    response = client.post("/api/qa-bank", json=_valid_entry("auto"))

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "q11"  # highest existing q01..q10 + 1


def test_post_id_collision_returns_409(app_and_client):
    _app, client = app_and_client

    payload = _valid_entry("dup")
    payload["id"] = "q01"  # already exists
    response = client.post("/api/qa-bank", json=payload)

    assert response.status_code == 409


def test_post_empty_key_points_returns_422(app_and_client):
    _app, client = app_and_client

    payload = _valid_entry("nokp")
    payload["key_points"] = []
    response = client.post("/api/qa-bank", json=payload)

    assert response.status_code == 422


def test_post_blank_question_returns_422(app_and_client):
    _app, client = app_and_client

    payload = _valid_entry("blank")
    payload["question"] = "   "
    response = client.post("/api/qa-bank", json=payload)

    assert response.status_code == 422


def test_put_updates_existing_entry(app_and_client):
    _app, client = app_and_client

    payload = _valid_entry("updated")
    payload["id"] = "q01"  # PUT body id is ignored; URL path param is authoritative
    response = client.put("/api/qa-bank/q01", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "q01"
    assert body["topic"] == "topic-updated"

    # Confirm the change is reflected on a subsequent GET.
    get_response = client.get("/api/qa-bank")
    updated = next(q for q in get_response.json()["questions"] if q["id"] == "q01")
    assert updated["topic"] == "topic-updated"


def test_put_unknown_id_returns_404(app_and_client):
    _app, client = app_and_client

    response = client.put("/api/qa-bank/does-not-exist", json=_valid_entry("missing"))

    assert response.status_code == 404


def test_delete_removes_entry(app_and_client):
    _app, client = app_and_client

    response = client.delete("/api/qa-bank/q10")

    assert response.status_code == 204
    get_response = client.get("/api/qa-bank")
    ids = [q["id"] for q in get_response.json()["questions"]]
    assert "q10" not in ids
    assert len(ids) == 9


def test_delete_unknown_id_returns_404(app_and_client):
    _app, client = app_and_client

    response = client.delete("/api/qa-bank/does-not-exist")

    assert response.status_code == 404


def test_delete_last_entry_returns_409(app_and_client):
    _app, client = app_and_client

    # Delete 9 of the 10 entries, leaving exactly one.
    for qid in [f"q{i:02d}" for i in range(1, 10)]:
        assert client.delete(f"/api/qa-bank/{qid}").status_code == 204

    response = client.delete("/api/qa-bank/q10")
    assert response.status_code == 409

    # The last entry must still be present.
    get_response = client.get("/api/qa-bank")
    assert len(get_response.json()["questions"]) == 1


# ---------------------------------------------------------------------------
# POST /api/qa-bank/extract — document upload -> LLM draft generation.
# ---------------------------------------------------------------------------


def test_extract_with_markdown_fixture_returns_drafts_without_writing_file(
    app_and_client, qa_bank_copy
):
    _app, client = app_and_client

    before_mtime = qa_bank_copy.stat().st_mtime_ns
    before_text = qa_bank_copy.read_text(encoding="utf-8")

    draft_json = json.dumps(
        {
            "questions": [
                {
                    "topic": "race-conditions",
                    "difficulty": "medium",
                    "question": "What is a race condition?",
                    "ideal_answer": "Unsynchronized concurrent access to shared state.",
                    "key_points": ["shared mutable state", "synchronization"],
                }
            ]
        }
    )

    md_content = (
        b"# Concurrency Notes\n\nA race condition happens when multiple "
        b"threads access shared state without synchronization."
    )

    with patch("app.retrieval.extractor.complete", return_value=draft_json):
        response = client.post(
            "/api/qa-bank/extract",
            files={"file": ("notes.md", md_content, "text/markdown")},
            data={"count": "3"},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["questions"]) == 1
    assert body["questions"][0]["topic"] == "race-conditions"
    assert body["truncated"] is False

    # The dataset file must be byte-for-byte unchanged — extract never writes.
    after_mtime = qa_bank_copy.stat().st_mtime_ns
    after_text = qa_bank_copy.read_text(encoding="utf-8")
    assert after_mtime == before_mtime
    assert after_text == before_text


def test_extract_with_real_pdf_fixture_extracts_text_genuinely(app_and_client):
    """Proves PDF text extraction itself works (not just the markdown/text
    path) — builds a tiny real PDF whose only content is a literal phrase,
    uploads it, and asserts an LLM-call assertion: the prompt sent to the
    (mocked) gateway contains that literal phrase, proving extract_text_from_upload
    really pulled text out of the PDF bytes rather than e.g. silently
    returning an empty string."""
    _app, client = app_and_client

    pdf_bytes = _make_text_pdf(b"Distributed systems consistency models")

    draft_json = json.dumps(
        {
            "questions": [
                {
                    "topic": "distributed-systems",
                    "difficulty": "hard",
                    "question": "What is eventual consistency?",
                    "ideal_answer": "A consistency model where replicas converge.",
                    "key_points": ["convergence", "no strong ordering guarantee"],
                }
            ]
        }
    )

    captured_messages = []

    def _capture_and_return(messages, **kwargs):
        captured_messages.append(messages)
        return draft_json

    with patch("app.retrieval.extractor.complete", side_effect=_capture_and_return):
        response = client.post(
            "/api/qa-bank/extract",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["questions"]) == 1

    # The captured prompt sent to the LLM must contain the literal text
    # pulled out of the PDF bytes — proving real extraction happened.
    full_prompt_text = json.dumps(captured_messages)
    assert "Distributed systems consistency models" in full_prompt_text


def test_extract_with_unsupported_extension_returns_415(app_and_client):
    _app, client = app_and_client

    response = client.post(
        "/api/qa-bank/extract",
        files={"file": ("malware.exe", b"binary-content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_extract_never_writes_even_on_llm_failure(app_and_client, qa_bank_copy):
    """Extraction failing (gateway error / malformed JSON) must still never
    touch the dataset file — the write only ever happens via /bulk."""
    _app, client = app_and_client

    before_text = qa_bank_copy.read_text(encoding="utf-8")

    with patch("app.retrieval.extractor.complete", return_value="not valid json"):
        response = client.post(
            "/api/qa-bank/extract",
            files={"file": ("notes.txt", b"some plain text content", "text/plain")},
        )

    assert response.status_code == 502
    assert qa_bank_copy.read_text(encoding="utf-8") == before_text


def test_extract_truncates_long_documents_and_reports_truncated_flag(app_and_client):
    _app, client = app_and_client

    long_text = ("word " * 5000).encode("utf-8")  # well over the 8000-char cap
    draft_json = json.dumps({"questions": []})

    with patch("app.retrieval.extractor.complete", return_value=draft_json):
        response = client.post(
            "/api/qa-bank/extract",
            files={"file": ("long.txt", long_text, "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["truncated"] is True


# ---------------------------------------------------------------------------
# POST /api/qa-bank/bulk
# ---------------------------------------------------------------------------


def test_bulk_commit_writes_multiple_entries_in_one_call(app_and_client):
    _app, client = app_and_client

    payload = {
        "questions": [
            _valid_entry("bulk1"),
            _valid_entry("bulk2"),
            _valid_entry("bulk3"),
        ]
    }

    response = client.post("/api/qa-bank/bulk", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert len(body["questions"]) == 3

    get_response = client.get("/api/qa-bank")
    assert len(get_response.json()["questions"]) == 13


def test_bulk_commit_auto_resolves_id_collisions_instead_of_failing(app_and_client):
    _app, client = app_and_client

    payload = {
        "questions": [
            {**_valid_entry("collide"), "id": "q01"},  # collides with existing q01
        ]
    }

    response = client.post("/api/qa-bank/bulk", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["questions"][0]["id"] != "q01"  # auto-resolved, not rejected


def test_bulk_commit_increases_total_questions_state(app_and_client):
    app, client = app_and_client

    before = app.state.total_questions
    payload = {"questions": [_valid_entry("count1"), _valid_entry("count2")]}

    response = client.post("/api/qa-bank/bulk", json=payload)

    assert response.status_code == 201
    assert app.state.total_questions == before + 2


# ---------------------------------------------------------------------------
# The live-update proof — the single most important test in this feature.
# ---------------------------------------------------------------------------


def test_live_update_proof_bulk_add_reflected_in_new_session(app_and_client):
    """(a) Add a new question via POST /api/qa-bank/bulk (API only — no YAML
    editing, no manual ingest call). (b) Confirm app.state.total_questions
    increased. (c) Start a NEW session and drive it through enough turns
    (gateways mocked) to reach the newly-added question, confirming the
    interviewer's turn references the new question's content — proving the
    addition was picked up with zero core-logic code changes and zero
    server restart."""
    app, client = app_and_client

    before_total = app.state.total_questions
    assert before_total == 10

    unique_phrase = "ZZZ_UNIQUE_QUANTUM_COMPUTING_MARKER_ZZZ"
    new_question_payload = {
        "id": None,
        "topic": "quantum-computing",
        "difficulty": "hard",
        "question": f"Explain {unique_phrase} and how it applies here.",
        "ideal_answer": "A reference answer about quantum computing.",
        "key_points": ["qubit superposition", "entanglement"],
    }

    bulk_response = client.post(
        "/api/qa-bank/bulk", json={"questions": [new_question_payload]}
    )
    assert bulk_response.status_code == 201
    new_id = bulk_response.json()["questions"][0]["id"]
    assert new_id == "q11"

    # (b) app.state.total_questions increased.
    assert app.state.total_questions == before_total + 1

    # (c) Start a NEW session — this calls initial_state() fresh, which
    # reads the just-refreshed app.state.total_questions.
    def _turn_echoing_question(messages, **kwargs):
        # The interviewer_turn node's ask_question_message() embeds
        # entry.question verbatim in the user-role prompt content — echo it
        # back so the "interviewer's turn references the new question's
        # content" assertion below can check the *returned* interviewer
        # text, exactly like a real LLM asking the literal question would.
        for message in messages:
            if unique_phrase in message.get("content", ""):
                return f"Let's discuss this: {unique_phrase} in detail."
        return "MOCK_GENERIC_TURN"

    with patch(
        "app.interview.graph.complete", side_effect=_turn_echoing_question
    ), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ), patch(
        "app.interview.feedback.complete", return_value=_feedback_json()
    ), patch(
        "app.web.server.transcribe", return_value="a mocked candidate answer"
    ), patch(
        "app.web.server.synthesize", return_value=b"FAKE_AUDIO"
    ):
        start = client.post("/api/sessions")
        assert start.status_code == 200
        session_id = start.json()["session_id"]
        assert start.json()["total_questions"] == before_total + 1

        found_new_question = False
        last_body = None
        for _ in range(15):  # generous safety cap; only 11 questions to traverse
            response = client.post(
                f"/api/sessions/{session_id}/turn",
                files={"audio": ("a.webm", b"x", "audio/webm")},
            )
            assert response.status_code == 200
            last_body = response.json()
            if unique_phrase in last_body["interviewer_text"]:
                found_new_question = True
                break
            if last_body["done"]:
                break

        assert found_new_question, (
            "The new question's content never appeared in an interviewer "
            f"turn. Last response: {last_body}"
        )


# ---------------------------------------------------------------------------
# Header comment block survives every write.
# ---------------------------------------------------------------------------


def test_header_comment_block_survives_a_write(app_and_client, qa_bank_copy):
    _app, client = app_and_client

    original_text = qa_bank_copy.read_text(encoding="utf-8")
    first_header_line = original_text.splitlines()[0]
    assert first_header_line.startswith("#")

    response = client.post("/api/qa-bank", json=_valid_entry("header-check"))
    assert response.status_code == 201

    after_text = qa_bank_copy.read_text(encoding="utf-8")
    assert first_header_line in after_text
    # The full original header block (every leading comment/blank line)
    # must be present verbatim, not just one line of it.
    header_lines = []
    for line in original_text.splitlines():
        if line.strip() == "" or line.lstrip().startswith("#"):
            header_lines.append(line)
        else:
            break
    for line in header_lines:
        assert line in after_text

    # Confirm the file is still valid YAML with all entries intact.
    parsed = yaml.safe_load(after_text)
    assert len(parsed["questions"]) == 11


def test_header_survives_put_and_delete_too(app_and_client, qa_bank_copy):
    _app, client = app_and_client
    original_text = qa_bank_copy.read_text(encoding="utf-8")
    first_header_line = original_text.splitlines()[0]

    put_response = client.put("/api/qa-bank/q01", json=_valid_entry("put-check"))
    assert put_response.status_code == 200
    assert first_header_line in qa_bank_copy.read_text(encoding="utf-8")

    delete_response = client.delete("/api/qa-bank/q02")
    assert delete_response.status_code == 204
    assert first_header_line in qa_bank_copy.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Existing interview flow unaffected when this feature isn't used.
# ---------------------------------------------------------------------------


def test_existing_interview_flow_unaffected_by_new_feature_existing(app_and_client):
    """No regression: starting a session and taking a turn still works
    exactly as before, with zero Q&A-bank-management calls made."""
    _app, client = app_and_client

    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ), patch("app.interview.feedback.complete", return_value=_feedback_json()), patch(
        "app.web.server.transcribe", return_value="answer"
    ), patch("app.web.server.synthesize", return_value=b"FAKE_AUDIO"):
        start = client.post("/api/sessions")
        assert start.status_code == 200
        assert start.json()["total_questions"] == 10

        session_id = start.json()["session_id"]
        turn = client.post(
            f"/api/sessions/{session_id}/turn",
            files={"audio": ("a.webm", b"x", "audio/webm")},
        )
        assert turn.status_code == 200
