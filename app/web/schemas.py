"""Pydantic request/response models for the web API.

See ``.claude/loop-state/phase-6-web-integration/spec.md`` "Suggested API
shape" for the contract these implement. Kept deliberately small — this is a
thin transport boundary, not a place for business logic (that lives in
``app.interview``/``app.gateway``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.interview.state import FeedbackReport


class SessionStartResponse(BaseModel):
    """Response to ``POST /api/sessions`` — the very first interviewer turn
    (intro folded together with the first question). Carries NO audio:
    the frontend fetches ``GET /api/sessions/{session_id}/audio``
    immediately after receiving this response to stream the synthesized
    speech progressively, rather than waiting for a full clip to be
    embedded here (see ``.claude/loop-state/streaming-tts/spec.md``).
    ``question_index``/``total_questions`` let the frontend render exact
    progress instead of a turn-count heuristic."""

    session_id: str
    interviewer_text: str
    done: bool = False
    question_index: int
    total_questions: int


class TurnResponse(BaseModel):
    """Response to ``POST /api/sessions/{session_id}/turn`` — the next
    interviewer turn (follow-up / next question / closing line). Carries NO
    audio (see ``SessionStartResponse``'s docstring — the same streaming
    ``/audio`` endpoint contract applies here). ``feedback`` is populated
    only when ``done`` is true. ``candidate_text`` is this turn's own STT
    transcription, returned so the frontend can show what the candidate
    actually said instead of a placeholder."""

    interviewer_text: str
    done: bool
    feedback: FeedbackReport | None = None
    candidate_text: str
    question_index: int
    total_questions: int


class ErrorResponse(BaseModel):
    """A clean, defined error body — never a raw stack trace (spec.md's
    "gateway failure -> clean HTTP error" acceptance criterion)."""

    detail: str


# ---------------------------------------------------------------------------
# Q&A bank management — see
# .claude/loop-state/qa-bank-management/spec.md for the full contract.
# ---------------------------------------------------------------------------

Difficulty = Literal["easy", "medium", "hard"]


class QAEntryIn(BaseModel):
    """Request body shape for creating/replacing a single reference Q&A
    entry (``POST``/``PUT``/each item of a bulk ``POST``). ``id`` is
    optional on create — omitted/``None`` means "auto-generate the next
    free id" (see ``app.retrieval.writer.generate_id``)."""

    id: str | None = None
    topic: str
    difficulty: Difficulty
    question: str
    ideal_answer: str
    key_points: list[str] = Field(default_factory=list)

    @field_validator("topic", "question", "ideal_answer")
    @classmethod
    def _non_empty_after_trim(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be non-empty after trimming whitespace")
        return value

    @field_validator("key_points")
    @classmethod
    def _at_least_one_key_point(cls, value: list[str]) -> list[str]:
        non_empty = [p for p in value if p and p.strip()]
        if not non_empty:
            raise ValueError(
                "key_points must contain at least one non-empty entry"
            )
        return non_empty


class QAEntryOut(BaseModel):
    """Response shape for a single, persisted reference Q&A entry — ``id``
    is always present (either the caller's own, or the auto-generated
    one)."""

    id: str
    topic: str
    difficulty: Difficulty
    question: str
    ideal_answer: str
    key_points: list[str]


class QAEntryDraft(QAEntryOut):
    """Identical shape to :class:`QAEntryOut`. The distinction is purely
    semantic, per spec.md: a draft's ``id`` is a provisional suggestion
    only, not yet reserved/persisted — a later bulk commit may need to
    re-resolve collisions if multiple drafts/edits happened in between."""


class QABankListResponse(BaseModel):
    """Response to ``GET /api/qa-bank``."""

    questions: list[QAEntryOut]
    max_follow_ups_per_question: int


class BulkCreateRequest(BaseModel):
    """Request body for ``POST /api/qa-bank/bulk``."""

    questions: list[QAEntryIn] = Field(min_length=1)


class BulkCreateResponse(BaseModel):
    """Response to ``POST /api/qa-bank/bulk`` — the full set of entries
    actually added, with their final (collision-resolved) ids."""

    questions: list[QAEntryOut]


class ExtractResponse(BaseModel):
    """Response to ``POST /api/qa-bank/extract`` — draft entries for the
    user to review/edit/discard before a separate ``/bulk`` commit.
    ``truncated`` is true when the uploaded document's extracted text
    exceeded the extraction text cap and was cut to the first N characters
    before being sent to the LLM."""

    questions: list[QAEntryDraft]
    truncated: bool
