"""Tests for app/web/server.py — the FastAPI web integration.

Every test drives the API through Starlette's ``TestClient`` (real HTTP
request/response objects, real ASGI app, real FastAPI routing/validation) —
not in-process function calls — per spec.md's explicit instruction to prove
the chain "through the real HTTP layer, not just in-process function calls."

Mocking chokepoints (zero real API keys, zero network calls):
- ``app.web.server.transcribe`` — the STT gateway call site this module
  imports by name.
- ``app.web.server.stream_synthesize`` / ``app.web.server.synthesize`` — the
  two TTS gateway call sites this module imports by name. Per
  ``.claude/loop-state/streaming-tts/spec.md``, ``start_session()``/
  ``take_turn()`` no longer call either of these at all (they only stash
  ``pending_text``); both are exercised exclusively via
  ``GET /api/sessions/{id}/audio``.
- ``app.interview.graph.complete`` / ``app.interview.evaluation.complete`` /
  ``app.interview.feedback.complete`` — the graph's own existing chokepoints,
  reused exactly as ``tests/test_interview_graph.py`` already does.

A fresh, isolated Chroma store (built from the real ``data/qa_bank.yaml``
into a ``tmp_path``) backs every test, mirroring
``tests/test_interview_graph.py``'s ``ingested_store`` fixture — tests never
touch the project's real ``.chroma/`` directory.
"""

from __future__ import annotations

import json
import shutil
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.interview.graph import build_graph as real_build_graph
from app.retrieval.ingest import build_store
from app.web.server import create_app

FAKE_TTS_BYTES = b"FAKE_MP3_AUDIO_BYTES"

# Fake chunks for the new streaming TTS path (GET /api/sessions/{id}/audio).
# A real stream_synthesize() yields raw PCM chunks; these tests only assert
# on concatenation/ordering, so arbitrary byte strings suffice.
FAKE_STREAM_CHUNKS = [b"CHUNK-ONE-", b"CHUNK-TWO-", b"CHUNK-THREE"]


def _fake_stream_synthesize(text: str):
    """Stand-in for app.gateway.tts.stream_synthesize: a generator yielding
    FAKE_STREAM_CHUNKS, ignoring its ``text`` argument."""
    yield from FAKE_STREAM_CHUNKS


@pytest.fixture
def ingested_store(tmp_path):
    """Mirrors tests/test_interview_graph.py's own fixture: a fresh Chroma
    store built from the real 10-entry data/qa_bank.yaml, isolated per
    test."""
    chroma_dir = tmp_path / "web_chroma_store"
    build_store(chroma_dir=chroma_dir)
    yield chroma_dir
    shutil.rmtree(chroma_dir, ignore_errors=True)


def _mock_turn_text(messages, **kwargs):
    return "MOCK_INTERVIEWER_TURN"


def _eval_json(verdict: str, missed: list[str] | None = None) -> str:
    return json.dumps(
        {"verdict": verdict, "missed_key_points": missed or [], "reasoning": "mock"}
    )


def _feedback_json() -> str:
    return json.dumps(
        {
            "per_question": [
                {
                    "qa_id": "q01",
                    "topic": "behavioral",
                    "verdict": "strong",
                    "note": "Clear answer.",
                }
            ],
            "overall_strengths": ["Clear communication"],
            "overall_improvements": ["Go deeper on trade-offs"],
        }
    )


@pytest.fixture
def gateway_patches(ingested_store):
    """Patches every gateway call site the web layer + graph touch, for the
    full "strong answer every time" happy path. Yields nothing useful by
    itself — tests that need different verdict scripting patch
    app.interview.evaluation.complete themselves on top of/instead of this.

    ``stream_synthesize``/``synthesize`` are patched here too even though
    ``start_session()``/``take_turn()`` no longer call either (see
    .claude/loop-state/streaming-tts/spec.md) — they're exercised by the new
    ``GET /api/sessions/{id}/audio`` endpoint, and tests that hit that
    endpoint reuse this same fixture rather than re-declaring the patch."""
    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ), patch("app.interview.feedback.complete", return_value=_feedback_json()), patch(
        "app.web.server.transcribe", return_value="a mocked candidate answer"
    ), patch(
        "app.web.server.stream_synthesize", side_effect=_fake_stream_synthesize
    ), patch("app.web.server.synthesize", return_value=FAKE_TTS_BYTES):
        yield


@pytest.fixture
def app_and_client(ingested_store):
    """One isolated FastAPI app + TestClient per test, pointed at the
    isolated Chroma fixture store. The TestClient is used as a context
    manager so the lifespan hook (and therefore build_graph()) actually
    runs."""
    app = create_app(chroma_dir=ingested_store)
    with TestClient(app) as client:
        yield app, client


# ---------------------------------------------------------------------------
# POST /api/sessions — start a session
# ---------------------------------------------------------------------------


def test_start_session_returns_200_with_text_and_no_audio_field(app_and_client, gateway_patches):
    """Per .claude/loop-state/streaming-tts/spec.md, audio is no longer
    embedded in this response at all — the frontend fetches it separately
    via GET /api/sessions/{id}/audio (see the dedicated tests below)."""
    _app, client = app_and_client

    response = client.post("/api/sessions")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["interviewer_text"]
    assert body["done"] is False
    assert "audio_base64" not in body


# ---------------------------------------------------------------------------
# Full scripted session via repeated POST .../turn -> done + feedback.
# ---------------------------------------------------------------------------


def test_full_session_via_http_reaches_done_with_feedback(app_and_client, gateway_patches):
    _app, client = app_and_client

    start = client.post("/api/sessions")
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    done = False
    body = None
    for _ in range(40):  # generous, finite safety cap (mirrors project convention)
        response = client.post(
            f"/api/sessions/{session_id}/turn",
            files={"audio": ("recording.webm", b"fake-audio-bytes", "audio/webm")},
        )
        assert response.status_code == 200
        body = response.json()
        if body["done"]:
            done = True
            break

    assert done, "Session did not reach done within the safety cap."
    assert body["feedback"] is not None
    assert body["feedback"]["per_question"]
    assert body["feedback"]["overall_strengths"]
    assert body["feedback"]["overall_improvements"]
    assert "audio_base64" not in body


# ---------------------------------------------------------------------------
# build_graph() called exactly once — the highest-priority correctness check.
# ---------------------------------------------------------------------------


def test_build_graph_called_exactly_once_across_multiple_sessions(
    ingested_store, gateway_patches
):
    """Patches app.web.server's imported `build_graph` name with a
    MagicMock(wraps=...) so it still really builds the graph (call count is
    observable) but every call is counted. Drives THREE separate sessions
    (three POST /api/sessions calls) on the SAME app/TestClient, each with
    multiple turns, then asserts build_graph was invoked exactly once for
    the whole test — directly enforcing Phase 4's documented
    process-lifetime contract through the web layer."""
    build_graph_mock = MagicMock(wraps=real_build_graph)
    with patch("app.web.server.build_graph", build_graph_mock):
        app = create_app(chroma_dir=ingested_store)
        with TestClient(app) as client:
            session_ids = []
            for _ in range(3):
                start = client.post("/api/sessions")
                assert start.status_code == 200
                session_ids.append(start.json()["session_id"])

            for session_id in session_ids:
                response = client.post(
                    f"/api/sessions/{session_id}/turn",
                    files={"audio": ("a.webm", b"x", "audio/webm")},
                )
                assert response.status_code == 200

    assert build_graph_mock.call_count == 1


# ---------------------------------------------------------------------------
# Two concurrent/interleaved sessions don't cross-contaminate.
# ---------------------------------------------------------------------------


def test_two_interleaved_sessions_do_not_cross_contaminate(app_and_client):
    """Mirrors test_interview_graph.py's test_build_once_many_sessions_pattern
    but through the HTTP layer: session A always answers "strong"; session B
    answers "weak" once (provoking a follow-up, same qa_index) then "strong".
    Interleaves requests across the two sessions and asserts each session's
    own follow_up-vs-advance behavior is exactly what its own script implies,
    never the other session's."""
    _app, client = app_and_client

    eval_script = {"A": iter(["strong"] * 20), "B": iter(["weak", "strong"] * 20)}

    def eval_side_effect_for(label):
        def _side_effect(messages, **kwargs):
            return _eval_json(next(eval_script[label]))

        return _side_effect

    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.feedback.complete", return_value=_feedback_json()
    ), patch("app.web.server.transcribe", return_value="answer"), patch(
        "app.web.server.synthesize", return_value=FAKE_TTS_BYTES
    ):
        start_a = client.post("/api/sessions")
        start_b = client.post("/api/sessions")
        assert start_a.status_code == 200
        assert start_b.status_code == 200
        session_a = start_a.json()["session_id"]
        session_b = start_b.json()["session_id"]
        assert session_a != session_b

        # Turn 1 for A: "strong" -> advances to qa_index 1 immediately.
        with patch(
            "app.interview.evaluation.complete", side_effect=eval_side_effect_for("A")
        ):
            resp_a1 = client.post(
                f"/api/sessions/{session_a}/turn",
                files={"audio": ("a.webm", b"x", "audio/webm")},
            )
        assert resp_a1.status_code == 200

        # Turn 1 for B: "weak" -> follow-up, SAME question (qa_index stays 0).
        with patch(
            "app.interview.evaluation.complete", side_effect=eval_side_effect_for("B")
        ):
            resp_b1 = client.post(
                f"/api/sessions/{session_b}/turn",
                files={"audio": ("b.webm", b"x", "audio/webm")},
            )
        assert resp_b1.status_code == 200

        # Turn 2 for A: another "strong" -> advances again to qa_index 2.
        with patch(
            "app.interview.evaluation.complete", side_effect=eval_side_effect_for("A")
        ):
            resp_a2 = client.post(
                f"/api/sessions/{session_a}/turn",
                files={"audio": ("a.webm", b"x", "audio/webm")},
            )
        assert resp_a2.status_code == 200

        # Turn 2 for B: "strong" this time -> the follow-up resolves, B
        # finally advances past qa_index 0 to qa_index 1 (one question
        # behind A, since A never needed a follow-up).
        with patch(
            "app.interview.evaluation.complete", side_effect=eval_side_effect_for("B")
        ):
            resp_b2 = client.post(
                f"/api/sessions/{session_b}/turn",
                files={"audio": ("b.webm", b"x", "audio/webm")},
            )
        assert resp_b2.status_code == 200

        # The graph-internal state (not exposed over HTTP) is the ground
        # truth for "did these sessions cross-contaminate" — read it
        # straight from the shared compiled graph object via each
        # session's own thread_id, exactly mirroring how
        # test_build_once_many_sessions_pattern asserts non-contamination.
        compiled = _app.state.compiled
        snap_a = compiled.get_state({"configurable": {"thread_id": session_a}})
        snap_b = compiled.get_state({"configurable": {"thread_id": session_b}})

        assert snap_a.values["qa_index"] == 2
        assert snap_a.values["follow_up_count"] == 0

        assert snap_b.values["qa_index"] == 1
        assert snap_b.values["follow_up_count"] == 0


# ---------------------------------------------------------------------------
# Unknown session_id -> clean 404, not a 500/hang.
# ---------------------------------------------------------------------------


def test_unknown_session_id_returns_404(app_and_client):
    _app, client = app_and_client

    response = client.post(
        "/api/sessions/00000000-0000-0000-0000-000000000000/turn",
        files={"audio": ("a.webm", b"x", "audio/webm")},
    )

    assert response.status_code == 404
    assert "detail" in response.json()


def test_finished_session_id_returns_404_on_further_turns(app_and_client, gateway_patches):
    _app, client = app_and_client

    start = client.post("/api/sessions")
    session_id = start.json()["session_id"]

    done = False
    for _ in range(40):
        response = client.post(
            f"/api/sessions/{session_id}/turn",
            files={"audio": ("recording.webm", b"x", "audio/webm")},
        )
        if response.json()["done"]:
            done = True
            break
    assert done

    after_done = client.post(
        f"/api/sessions/{session_id}/turn",
        files={"audio": ("recording.webm", b"x", "audio/webm")},
    )
    assert after_done.status_code == 404


# ---------------------------------------------------------------------------
# Gateway failure -> clean HTTP error, not an unhandled 500 with a stack trace.
# ---------------------------------------------------------------------------


def test_gateway_transient_failure_on_turn_returns_clean_502(app_and_client, gateway_patches):
    """Was originally exercised by patching app.web.server.synthesize to
    fail on POST /api/sessions — but per
    .claude/loop-state/streaming-tts/spec.md, start_session() no longer
    calls any TTS function at all, so that trigger point no longer exists
    (the graph's own LLM-gateway failures are swallowed/degraded inside
    app.interview.graph._safe_complete, never escaping as a GatewayError —
    see that module). transcribe() failing on POST .../turn is the
    equivalent still-live trigger for this same _map_gateway_error path."""
    from app.gateway.exceptions import GatewayAllProvidersFailedError

    _app, client = app_and_client
    start = client.post("/api/sessions")
    session_id = start.json()["session_id"]

    with patch(
        "app.web.server.transcribe",
        side_effect=GatewayAllProvidersFailedError("provider down"),
    ):
        response = client.post(
            f"/api/sessions/{session_id}/turn",
            files={"audio": ("a.webm", b"x", "audio/webm")},
        )

    assert response.status_code == 502
    body = response.json()
    assert "detail" in body
    # Never leak the raw exception/stack trace text to the client.
    assert "Traceback" not in response.text


def test_gateway_auth_failure_on_turn_returns_clean_500_no_key_leak(
    app_and_client,
):
    _app, client = app_and_client

    from app.gateway.exceptions import GatewayAuthError

    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.interview.evaluation.complete", return_value=_eval_json("strong")
    ), patch("app.web.server.synthesize", return_value=FAKE_TTS_BYTES):
        start = client.post("/api/sessions")
    session_id = start.json()["session_id"]

    with patch(
        "app.web.server.transcribe",
        side_effect=GatewayAuthError("DEEPGRAM_API_KEY is not set"),
    ):
        response = client.post(
            f"/api/sessions/{session_id}/turn",
            files={"audio": ("a.webm", b"x", "audio/webm")},
        )

    assert response.status_code == 500
    assert "DEEPGRAM_API_KEY" not in response.text
    assert "Traceback" not in response.text


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}/audio — streaming TTS delivery.
# See .claude/loop-state/streaming-tts/spec.md.
# ---------------------------------------------------------------------------


def test_get_audio_after_start_returns_concatenated_stream_chunks(
    app_and_client, gateway_patches
):
    """After POST /api/sessions (which no longer embeds audio), the
    frontend's matching GET .../audio call should stream back exactly the
    mocked stream_synthesize() chunks, concatenated, with the internal
    octet-stream content type."""
    _app, client = app_and_client

    start = client.post("/api/sessions")
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    response = client.get(f"/api/sessions/{session_id}/audio")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.content == b"".join(FAKE_STREAM_CHUNKS)


def test_get_audio_after_turn_returns_concatenated_stream_chunks(
    app_and_client, gateway_patches
):
    """Same contract, after a real POST .../turn rather than session start —
    pending_text is overwritten by take_turn(), and /audio reflects it."""
    _app, client = app_and_client

    start = client.post("/api/sessions")
    session_id = start.json()["session_id"]
    turn = client.post(
        f"/api/sessions/{session_id}/turn",
        files={"audio": ("recording.webm", b"fake-audio-bytes", "audio/webm")},
    )
    assert turn.status_code == 200

    response = client.get(f"/api/sessions/{session_id}/audio")

    assert response.status_code == 200
    assert response.content == b"".join(FAKE_STREAM_CHUNKS)


def test_get_audio_unknown_session_id_returns_404(app_and_client):
    _app, client = app_and_client

    response = client.get(
        "/api/sessions/00000000-0000-0000-0000-000000000000/audio"
    )

    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_audio_before_any_turn_returns_404(ingested_store):
    """A session that exists in app.state.sessions but has never had
    start_session()/take_turn() populate pending_text (simulated directly
    here, since in practice start_session() always sets it immediately)
    must 404, per spec.md's explicit 'pending_text was never set' clause."""
    app = create_app(chroma_dir=ingested_store)
    with TestClient(app) as client:
        app.state.sessions["bare-session"] = {"done": False}

        response = client.get("/api/sessions/bare-session/audio")

    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_audio_falls_back_to_one_shot_synthesize_on_ws_failure_before_any_chunk(
    app_and_client,
):
    """If stream_synthesize's WebSocket connection fails before yielding
    ANY chunk, /audio must transparently fall back to the existing
    one-shot synthesize() and stream its full bytes as a single chunk —
    the documented spec.md fallback behavior."""
    from app.gateway.exceptions import GatewayAllProvidersFailedError

    _app, client = app_and_client

    with patch("app.interview.graph.complete", side_effect=_mock_turn_text), patch(
        "app.web.server.transcribe", return_value="answer"
    ), patch(
        "app.web.server.stream_synthesize",
        side_effect=GatewayAllProvidersFailedError("WS handshake failed"),
    ), patch("app.web.server.synthesize", return_value=FAKE_TTS_BYTES):
        start = client.post("/api/sessions")
        assert start.status_code == 200
        session_id = start.json()["session_id"]

        response = client.get(f"/api/sessions/{session_id}/audio")

    assert response.status_code == 200
    assert response.content == FAKE_TTS_BYTES


# ---------------------------------------------------------------------------
# Static frontend.
# ---------------------------------------------------------------------------


def test_index_page_loads_with_controls(app_and_client):
    _app, client = app_and_client

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="start-button"' in html
    assert 'id="record-button"' in html


def test_app_js_implements_four_named_ui_states(app_and_client):
    _app, client = app_and_client

    response = client.get("/static/app.js")

    assert response.status_code == 200
    js = response.text
    for state_name in ("idle", "recording", "processing", "speaking"):
        assert f'"{state_name}"' in js, f"app.js is missing explicit state: {state_name}"
