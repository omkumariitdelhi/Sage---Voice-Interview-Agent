"""FastAPI app wiring real STT -> the LangGraph interview (Guardrails already
applied inside it) -> real TTS, served to a browser frontend.

See ``.claude/loop-state/phase-6-web-integration/plan.md`` for the full
integration diagram and design rationale. Short version:

- ``build_graph()`` is called **exactly once per process**, inside this
  module's FastAPI ``lifespan`` hook (NOT inside a request handler) — see
  ``app.interview.graph.build_graph``'s docstring "PROCESS-LIFETIME CONTRACT"
  for why a second call would silently disconnect every existing session's
  checkpointed state. The compiled graph + its session metadata are stashed
  on ``app.state`` so every request handler reuses the one shared object.
- ``create_app()`` is a factory (rather than a single bare module-level
  ``FastAPI()``) so tests can construct independent app instances (each with
  its own isolated graph/session registry) without cross-test interference,
  while still proving the once-per-*app-instance* (== once-per-process in
  real deployment) property directly. A module-level ``app = create_app()``
  at the bottom is what ``run_server.py``/``uvicorn app.web.server:app``
  actually serve.
- A ``thread_id`` (the LangGraph session key) is generated as a UUID per new
  browser session and doubles as this module's own ``session_id``. A
  separate ``app.state.sessions`` registry (NOT LangGraph's own state) is
  what lets an unknown/expired ``session_id`` return a clean 404 instead of
  LangGraph silently treating it as a brand-new, never-before-seen thread.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.gateway.exceptions import GatewayAuthError, GatewayError
from app.gateway.stt import transcribe
from app.gateway.tts import stream_synthesize, synthesize
from app.interview.graph import Command, build_graph
from app.interview.state import FeedbackReport, initial_state
from app.retrieval.exceptions import (
    ExtractionFailedError,
    LastEntryDeletionError,
    QAEntryIdCollisionError,
    QAEntryNotFoundError,
    UnsupportedDocumentTypeError,
)
from app.retrieval.extractor import cap_text, draft_entries_from_text, extract_text_from_upload
from app.retrieval.ingest import DEFAULT_CHROMA_DIR, DEFAULT_DATASET_PATH, build_store
from app.retrieval.loader import load_dataset, load_dataset_meta
from app.retrieval.schema import QAEntry
from app.retrieval.store import count as store_count
from app.retrieval.writer import add_entries, delete_entry, replace_entry
from app.web.schemas import (
    BulkCreateRequest,
    BulkCreateResponse,
    ExtractResponse,
    QABankListResponse,
    QAEntryDraft,
    QAEntryIn,
    QAEntryOut,
    SessionStartResponse,
    TurnResponse,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _thread_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


def _latest_interviewer_text(result: dict[str, Any]) -> str:
    """The most recently emitted interviewer line. Every ``invoke()``/
    ``Command(resume=...)`` call AFTER the session has started passes through
    exactly one ``interviewer_turn``/``wrap_up`` node before either pausing
    again or finishing, appending exactly one new interviewer ``TurnRecord``
    — so ``transcript[-1]`` is always correct. NOT used for the very first
    turn of a session (see :func:`_session_start_interviewer_text`), since
    ``intro`` and the first ``interviewer_turn`` both land before the FIRST
    interrupt in one ``invoke()`` call, producing two new entries, not one.
    """
    transcript = result.get("transcript", [])
    if not transcript:
        raise RuntimeError("Graph produced no transcript turns; cannot extract interviewer text.")
    return transcript[-1]["text"]


def _session_start_interviewer_text(result: dict[str, Any]) -> str:
    """The combined text to speak for the very first turn of a session.

    ``intro`` and the first ``interviewer_turn`` (the first question) both
    run before the first ``interrupt()`` in the same ``invoke()`` call, so
    ``result["transcript"]`` has exactly two new entries at this point.
    Joining both (rather than ``_latest_interviewer_text``'s single
    ``transcript[-1]``) is what actually makes the persona's self-
    introduction (``app.interview.persona``) get spoken at all — previously
    only the first question was returned and the intro was silently dropped.
    """
    transcript = result.get("transcript", [])
    if not transcript:
        raise RuntimeError("Graph produced no transcript turns; cannot extract interviewer text.")
    return " ".join(turn["text"] for turn in transcript)


def _audio_byte_stream(text: str) -> Iterator[bytes]:
    """Yield audio bytes for ``text``, preferring the real streaming path
    and transparently falling back to one-shot synthesis if (and only if)
    the WebSocket connection fails before any chunk was produced.

    Per spec.md: a ``GatewayError`` raised by ``stream_synthesize`` BEFORE
    its generator has yielded anything means the WS handshake itself never
    succeeded — nothing has been sent to the browser yet, so it's safe to
    transparently retry via the existing REST ``synthesize()`` call instead
    and stream its full result as a single chunk. If a ``GatewayError``
    happens AFTER at least one chunk has already been yielded, it is
    deliberately let to propagate uncaught — the response has already
    started streaming to the browser at that point, and there is nothing
    sane to fall back to mid-stream (documented limitation, not a bug; see
    spec.md's "Non-goals / known risks").

    Format note: the fallback explicitly requests the SAME raw-PCM format
    ``stream_synthesize()`` produces (``encoding="linear16",
    sample_rate=24000`` — see ``synthesize()``'s own docstring for why these
    are dedicated named params, not folded into ``**kwargs``) so the browser
    player never needs to know or care which path actually served it.
    """
    started = False
    try:
        for chunk in stream_synthesize(text):
            started = True
            yield chunk
    except GatewayError:
        if started:
            raise
        yield synthesize(text, encoding="linear16", sample_rate=24000)


def _map_gateway_error(exc: GatewayError) -> HTTPException:
    """Maps the gateway's typed exceptions to clean, defined HTTP responses
    — never an unhandled 500 with a leaked stack trace (spec.md acceptance
    criterion). Auth errors are a server misconfiguration (missing API key),
    not the caller's fault, so they map to 500 but with a generic message
    that never echoes key names/values. Every other GatewayError (transient
    exhaustion, etc.) is an upstream-provider failure, mapped to 502."""
    if isinstance(exc, GatewayAuthError):
        logger.error("Gateway auth error (server misconfiguration): %s", exc)
        return HTTPException(status_code=500, detail="Server is missing a required API key.")
    logger.warning("Gateway call failed: %s", exc)
    return HTTPException(status_code=502, detail="An upstream voice/AI provider call failed. Please try again.")


def _entry_to_out(entry: QAEntry, *, draft: bool = False) -> QAEntryOut:
    model = QAEntryDraft if draft else QAEntryOut
    return model(
        id=entry.id,
        topic=entry.topic,
        difficulty=entry.difficulty,  # type: ignore[arg-type]
        question=entry.question,
        ideal_answer=entry.ideal_answer,
        key_points=list(entry.key_points),
    )


def _in_to_entry(payload: QAEntryIn) -> QAEntry:
    return QAEntry(
        id=payload.id or "",
        topic=payload.topic,
        difficulty=payload.difficulty,
        question=payload.question,
        ideal_answer=payload.ideal_answer,
        key_points=list(payload.key_points),
    )


def _refresh_total_questions(app: FastAPI) -> None:
    """Re-query the store's live count and update the server-level snapshot
    so the NEXT new session sees the right ``total_questions`` (see
    module docstring / spec.md's refresh requirement). A session already in
    progress keeps its own ``initial_state()`` snapshot, taken once at
    session start, and is intentionally unaffected — this project's existing
    design already treats a session's snapshot as fixed-for-stability; this
    helper only keeps the *server-level* snapshot in sync for the next one.
    """
    app.state.total_questions = store_count(chroma_dir=app.state.chroma_dir)


def create_app(
    *,
    dataset_path: str | Path | None = None,
    chroma_dir: str | Path | None = None,
) -> FastAPI:
    """Build a fresh FastAPI app with its own isolated graph + session
    registry. ``dataset_path``/``chroma_dir`` are forwarded to
    ``build_graph()`` (so tests can point at an isolated fixture dataset/
    store, mirroring ``tests/test_interview_graph.py``'s own pattern) — both
    default to ``build_graph()``'s own defaults when omitted.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The ONE and ONLY build_graph() call for this app's lifetime — see
        # module docstring / build_graph()'s own "PROCESS-LIFETIME CONTRACT".
        build_kwargs: dict[str, Any] = {}
        if dataset_path is not None:
            build_kwargs["dataset_path"] = dataset_path
        if chroma_dir is not None:
            build_kwargs["chroma_dir"] = chroma_dir
        compiled, total_questions, max_follow_ups = build_graph(**build_kwargs)
        app.state.compiled = compiled
        app.state.total_questions = total_questions
        app.state.max_follow_ups = max_follow_ups
        app.state.sessions = {}
        # Resolved (not None) dataset/store paths, kept on app.state so the
        # Q&A bank management endpoints below can read/write/rebuild against
        # exactly the same dataset+store this app instance's graph was built
        # from (critical for tests, which point both at isolated tmp_path
        # fixtures) and so _refresh_total_questions() can re-query the store
        # after a write without re-deriving these defaults itself.
        app.state.dataset_path = dataset_path if dataset_path is not None else DEFAULT_DATASET_PATH
        app.state.chroma_dir = chroma_dir if chroma_dir is not None else DEFAULT_CHROMA_DIR
        logger.info(
            "Interview graph built once for this app instance (%d questions, %d max follow-ups).",
            total_questions,
            max_follow_ups,
        )
        yield
        app.state.sessions.clear()

    app = FastAPI(title="Voice Interview Agent", lifespan=lifespan)

    @app.post("/api/sessions", response_model=SessionStartResponse)
    async def start_session() -> SessionStartResponse:
        session_id = str(uuid.uuid4())
        config = _thread_config(session_id)
        state = initial_state(
            total_questions=app.state.total_questions,
            max_follow_ups=app.state.max_follow_ups,
        )
        try:
            result = await run_in_threadpool(app.state.compiled.invoke, state, config)
            interviewer_text = _session_start_interviewer_text(result)
        except GatewayError as exc:
            raise _map_gateway_error(exc) from exc

        # No TTS call here — audio is no longer embedded in this response.
        # Stash the text server-side so GET /api/sessions/{id}/audio (below)
        # can stream its synthesis progressively; the frontend fetches that
        # endpoint immediately after receiving this response. See
        # .claude/loop-state/streaming-tts/spec.md.
        app.state.sessions[session_id] = {"done": False, "pending_text": interviewer_text}
        return SessionStartResponse(
            session_id=session_id,
            interviewer_text=interviewer_text,
            done=False,
            question_index=result.get("qa_index", 0),
            total_questions=app.state.total_questions,
        )

    @app.post("/api/sessions/{session_id}/turn", response_model=TurnResponse)
    async def take_turn(session_id: str, audio: UploadFile) -> TurnResponse:
        session = app.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id!r}")
        if session["done"]:
            raise HTTPException(status_code=404, detail=f"Session {session_id!r} has already finished.")

        config = _thread_config(session_id)
        audio_bytes = await audio.read()

        try:
            # Forward (filename, bytes) so Deepgram/litellm can infer the
            # codec from the browser's MediaRecorder mimeType-derived
            # filename (e.g. "recording.webm") — see plan.md Section 1.
            candidate_text = await run_in_threadpool(
                transcribe, (audio.filename or "recording.webm", audio_bytes)
            )
            result = await run_in_threadpool(
                app.state.compiled.invoke, Command(resume=candidate_text), config
            )
            interviewer_text = _latest_interviewer_text(result)
        except GatewayError as exc:
            raise _map_gateway_error(exc) from exc

        # No TTS call here — see start_session()'s matching comment above.
        session["pending_text"] = interviewer_text

        done = result.get("phase") == "done"
        feedback: FeedbackReport | None = None
        if done:
            session["done"] = True
            feedback_dict = result.get("feedback")
            if feedback_dict is not None:
                feedback = FeedbackReport.model_validate(feedback_dict)

        return TurnResponse(
            interviewer_text=interviewer_text,
            done=done,
            feedback=feedback,
            candidate_text=candidate_text,
            question_index=min(result.get("qa_index", 0), app.state.total_questions),
            total_questions=app.state.total_questions,
        )

    @app.get("/api/sessions/{session_id}/audio")
    async def get_session_audio(session_id: str) -> StreamingResponse:
        """Stream the most recently emitted interviewer turn's speech as raw
        PCM bytes (16-bit little-endian, mono, 24000 Hz — see
        ``app.gateway.tts.stream_synthesize``'s docstring), progressively as
        Deepgram generates them. The frontend calls this immediately after
        ``POST /api/sessions`` or ``.../turn`` returns its (now audio-less)
        JSON body.

        ``media_type="application/octet-stream"`` is deliberate: this is an
        internal contract between this backend and this project's own
        ``app.js`` (which hardcodes the sample rate/encoding), not a public,
        self-describing audio API — see spec.md.
        """
        session = app.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id!r}")
        pending_text = session.get("pending_text")
        if pending_text is None:
            raise HTTPException(
                status_code=404,
                detail=f"No turn has been taken yet for session {session_id!r}.",
            )

        # _audio_byte_stream returns a plain sync generator; Starlette's
        # StreamingResponse detects it's not an AsyncIterable and runs it
        # via iterate_in_threadpool internally, so the blocking
        # websockets.sync.client/httpx calls inside it never block the
        # event loop.
        return StreamingResponse(
            _audio_byte_stream(pending_text),
            media_type="application/octet-stream",
        )

    # -----------------------------------------------------------------
    # Q&A bank management — manual CRUD + document-upload extraction.
    # See .claude/loop-state/qa-bank-management/spec.md for the full
    # contract. None of these touch app.state.compiled/sessions; they only
    # read/write the dataset file + rebuild the retrieval store, and (for
    # any write) refresh app.state.total_questions for the NEXT new
    # session — an in-progress session's own snapshot is unaffected by
    # design (see _refresh_total_questions's docstring).
    # -----------------------------------------------------------------

    @app.get("/api/qa-bank", response_model=QABankListResponse)
    async def list_reference_questions() -> QABankListResponse:
        entries = load_dataset(app.state.dataset_path)
        meta = load_dataset_meta(app.state.dataset_path)
        return QABankListResponse(
            questions=[_entry_to_out(entry) for entry in entries],
            max_follow_ups_per_question=meta.max_follow_ups_per_question,
        )

    @app.post("/api/qa-bank", response_model=QAEntryOut, status_code=201)
    async def create_qa_entry(payload: QAEntryIn) -> QAEntryOut:
        try:
            [added] = add_entries(app.state.dataset_path, [_in_to_entry(payload)])
        except QAEntryIdCollisionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        build_store(app.state.dataset_path, app.state.chroma_dir)
        _refresh_total_questions(app)
        return _entry_to_out(added)

    @app.put("/api/qa-bank/{qa_id}", response_model=QAEntryOut)
    async def update_qa_entry(qa_id: str, payload: QAEntryIn) -> QAEntryOut:
        try:
            updated = replace_entry(app.state.dataset_path, qa_id, _in_to_entry(payload))
        except QAEntryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        build_store(app.state.dataset_path, app.state.chroma_dir)
        _refresh_total_questions(app)
        return _entry_to_out(updated)

    @app.delete("/api/qa-bank/{qa_id}", status_code=204)
    async def delete_qa_entry(qa_id: str) -> None:
        try:
            delete_entry(app.state.dataset_path, qa_id)
        except QAEntryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LastEntryDeletionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        build_store(app.state.dataset_path, app.state.chroma_dir)
        _refresh_total_questions(app)

    @app.post("/api/qa-bank/extract", response_model=ExtractResponse)
    async def extract_qa_drafts(
        file: UploadFile = File(...), count: int = Form(5)
    ) -> ExtractResponse:
        clamped_count = max(1, min(10, count))
        content = await file.read()

        try:
            raw_text = extract_text_from_upload(file.filename or "", content)
        except UnsupportedDocumentTypeError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        capped_text, truncated = cap_text(raw_text)
        existing_ids = {entry.id for entry in load_dataset(app.state.dataset_path)}

        try:
            drafts = await run_in_threadpool(
                draft_entries_from_text,
                capped_text,
                clamped_count,
                existing_ids=existing_ids,
            )
        except ExtractionFailedError as exc:
            logger.warning("Q&A extraction failed: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Document extraction failed (upstream LLM call). Please try again.",
            ) from exc

        return ExtractResponse(
            questions=[_entry_to_out(entry, draft=True) for entry in drafts],
            truncated=truncated,
        )

    @app.post("/api/qa-bank/bulk", response_model=BulkCreateResponse, status_code=201)
    async def bulk_create_qa_entries(payload: BulkCreateRequest) -> BulkCreateResponse:
        # Bulk commit auto-resolves any id collision (against existing
        # entries OR earlier items in this same batch) rather than failing
        # the whole batch over one collision — the documented policy
        # difference from single POST's 409 (see
        # app.retrieval.writer.add_entries's docstring and spec.md).
        entries: list[QAEntry] = []
        known_ids = {entry.id for entry in load_dataset(app.state.dataset_path)}
        for item in payload.questions:
            candidate = _in_to_entry(item)
            if candidate.id and candidate.id in known_ids:
                candidate = QAEntry(
                    id="",
                    topic=candidate.topic,
                    difficulty=candidate.difficulty,
                    question=candidate.question,
                    ideal_answer=candidate.ideal_answer,
                    key_points=candidate.key_points,
                )
            if candidate.id:
                known_ids.add(candidate.id)
            entries.append(candidate)

        added = add_entries(app.state.dataset_path, entries)
        build_store(app.state.dataset_path, app.state.chroma_dir)
        _refresh_total_questions(app)
        return BulkCreateResponse(questions=[_entry_to_out(entry) for entry in added])

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


# The module-level singleton served by `uvicorn app.web.server:app` /
# `run_server.py`. `create_app()` itself only constructs the FastAPI object
# and registers the `lifespan` hook above — `build_graph()` is NOT called at
# this module's import time. The one and only `build_graph()` call happens
# when the ASGI server actually starts the app (uvicorn's startup phase
# triggers `lifespan`), using build_graph()'s own default dataset_path/
# chroma_dir (the project's real dataset + real Chroma store — see
# app.retrieval.ingest's DEFAULT_QA_BANK_PATH/DEFAULT_DATASET_PATH for the
# single source of truth on those paths; this module never names the
# dataset's filename itself, mirroring app/interview/graph.py's own
# substring-free convention enforced by tests/test_retrieval_yaml_isolation.py).
# Tests use `create_app(...)` directly instead of importing this name, so
# they never trigger this module-level object or its lifespan.
app = create_app()
