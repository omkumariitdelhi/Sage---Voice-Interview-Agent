"""Document-upload -> LLM-extracted Q&A drafts.

Lives inside ``app/retrieval/`` for the same grep-checkable-isolation reason
as :mod:`app.retrieval.writer` (see that module's docstring) — this module's
own filename/content is exempt from
``tests/test_retrieval_yaml_isolation.py``'s "no ``qa_bank`` reference
outside ``app/retrieval/``" check, though it doesn't actually reference the
dataset filename at all; it never reads or writes ``data/qa_bank.yaml``.

This module is the *preview* half of the document-upload feature: it turns
uploaded document bytes into draft :class:`~app.retrieval.schema.QAEntry`
objects via the LLM gateway, and **never writes them anywhere**. Persisting
a reviewed/edited set of drafts is a separate, explicit step the caller
performs by calling :func:`app.retrieval.writer.add_entries` afterward (see
``app/web/server.py``'s ``POST /api/qa-bank/bulk`` endpoint) — this
separation is the entire point of the "drafts, not yet written" contract in
spec.md.

The LLM call here mirrors :mod:`app.interview.evaluation`'s exact
parse -> one-reask-on-malformed-JSON-retry pattern, with one deliberate,
spec-sanctioned divergence at the *final* failure step: ``evaluation.py``
must never raise (a live interview turn has to continue no matter what), so
it degrades to a safe fallback verdict. This module's caller is a one-shot
preview request with no "the show must go on" constraint, and spec.md
explicitly asks for a clean failure here ("If the LLM call fails or returns
persistently malformed JSON after one retry, return a clean 502, not a
crash") — so :func:`draft_entries_from_text` raises
:class:`~app.retrieval.exceptions.ExtractionFailedError` on that final
failure instead of fabricating a draft, and the web layer maps that to 502.
"""

from __future__ import annotations

import io
import json
import logging

from pydantic import BaseModel, Field, ValidationError, field_validator
from pypdf import PdfReader

from app.gateway.exceptions import GatewayError
from app.gateway.llm import complete
from app.json_mode import strip_json_fences
from app.retrieval.exceptions import ExtractionFailedError, UnsupportedDocumentTypeError
from app.retrieval.schema import QAEntry
from app.retrieval.writer import generate_id

logger = logging.getLogger(__name__)

# Cap extracted text BEFORE prompting the LLM — this prototype does not
# chunk/map-reduce arbitrarily long documents (see spec.md's non-goals).
# ~8000 characters is comfortably inside every model's context window for
# this app's prompts and keeps latency/cost bounded on large uploads.
TEXT_CAP_CHARS = 8000

_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

_REASK_INSTRUCTION = (
    "Your previous response was not valid JSON matching the required "
    "schema. Return ONLY the JSON object, with no markdown fences and no "
    "extra commentary."
)


class _DraftEntryModel(BaseModel):
    """Pydantic parse target for one LLM-generated draft entry."""

    topic: str
    difficulty: str
    question: str
    ideal_answer: str
    key_points: list[str] = Field(default_factory=list)

    @field_validator("topic", "question", "ideal_answer")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("difficulty")
    @classmethod
    def _valid_difficulty(cls, value: str) -> str:
        if value not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be one of: easy, medium, hard")
        return value

    @field_validator("key_points")
    @classmethod
    def _at_least_one_key_point(cls, value: list[str]) -> list[str]:
        non_empty = [p for p in value if p and p.strip()]
        if not non_empty:
            raise ValueError("key_points must contain at least one non-empty entry")
        return non_empty


class _DraftQuestionsModel(BaseModel):
    """Parse target for the LLM's whole JSON response."""

    questions: list[_DraftEntryModel] = Field(default_factory=list)


def _parse(raw_text: str) -> _DraftQuestionsModel | None:
    """Best-effort JSON parse + Pydantic validation. Returns ``None`` (never
    raises) on any failure, mirroring ``app.interview.evaluation``'s own
    ``_parse`` helper exactly."""
    try:
        data = json.loads(strip_json_fences(raw_text))
    except json.JSONDecodeError:
        return None
    try:
        return _DraftQuestionsModel.model_validate(data)
    except ValidationError:
        return None


def extract_text_from_upload(filename: str, content: bytes) -> str:
    """Extract plain text from an uploaded document's raw bytes, dispatching
    on the filename's extension.

    Raises:
        UnsupportedDocumentTypeError: the extension is not one of
            ``.pdf``/``.md``/``.txt``.
    """
    suffix = ""
    if "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()

    if suffix not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type '{suffix or filename}'. "
            f"Supported types: .pdf, .md, .txt."
        )

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # .md / .txt — read directly as UTF-8 text. Invalid byte sequences are
    # replaced rather than raising, since a slightly mangled character
    # should not hard-fail an otherwise-usable upload.
    return content.decode("utf-8", errors="replace")


def cap_text(text: str, *, limit: int = TEXT_CAP_CHARS) -> tuple[str, bool]:
    """Truncate ``text`` to the first ``limit`` characters.

    Returns ``(capped_text, truncated)`` — ``truncated`` is surfaced in the
    API response (``ExtractResponse.truncated``) per spec.md so the caller
    knows the LLM only ever saw a prefix of a long document.
    """
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _build_extraction_prompt(text: str, count: int) -> list[dict]:
    system = (
        "You generate interview screening questions grounded in supplied "
        "source material. Respond with STRICT JSON only, matching this "
        "schema exactly:\n"
        '{"questions": [{"topic": "...", '
        '"difficulty": "easy"|"medium"|"hard", "question": "...", '
        '"ideal_answer": "...", "key_points": ["...", ...]}]}\n'
        "Every question must be answerable from the supplied text. Every "
        "entry needs at least one key_points item. Do not include any "
        "commentary outside the JSON object."
    )
    user = (
        f"Generate up to {count} interview-style question/answer entries "
        f"grounded in the following source material:\n\n{text}\n\n"
        f"Return the JSON object now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def draft_entries_from_text(
    text: str, count: int, *, existing_ids: set[str] | None = None
) -> list[QAEntry]:
    """Generate up to ``count`` draft :class:`QAEntry` objects grounded in
    ``text`` via the gateway LLM.

    Each draft is assigned a provisional id via
    :func:`app.retrieval.writer.generate_id` against ``existing_ids`` (the
    current on-disk id set) — a best-effort suggestion only, per spec.md,
    since a later bulk commit may need to re-resolve collisions if multiple
    drafts/edits happened in between.

    Never writes to ``data/qa_bank.yaml`` and never calls
    :func:`app.retrieval.ingest.build_store` — this function has no import
    of either, by construction.

    Raises:
        ExtractionFailedError: the gateway call failed, or the response was
            persistently malformed JSON after the single sanctioned re-ask
            retry.
    """
    messages = _build_extraction_prompt(text, count)
    known_ids = set(existing_ids or set())

    try:
        raw_text = complete(messages, response_format={"type": "json_object"})
    except GatewayError as exc:
        logger.warning("draft_entries_from_text: gateway call failed (%s).", exc)
        raise ExtractionFailedError(f"LLM gateway call failed: {exc}") from exc

    parsed = _parse(raw_text)
    if parsed is None:
        retry_messages = messages + [
            {"role": "assistant", "content": raw_text},
            {"role": "user", "content": _REASK_INSTRUCTION},
        ]
        try:
            raw_text_retry = complete(
                retry_messages, response_format={"type": "json_object"}
            )
            parsed = _parse(raw_text_retry)
        except GatewayError as exc:
            logger.warning(
                "draft_entries_from_text: gateway call failed on retry (%s).", exc
            )
            raise ExtractionFailedError(
                f"LLM gateway call failed on retry: {exc}"
            ) from exc

        if parsed is None:
            logger.warning(
                "draft_entries_from_text: malformed LLM JSON after retry. "
                "Raw text: %r",
                raw_text_retry,
            )
            raise ExtractionFailedError(
                "LLM returned malformed JSON after one retry."
            )

    drafts: list[QAEntry] = []
    for item in parsed.questions[:count]:
        draft_id = generate_id(known_ids)
        known_ids.add(draft_id)
        drafts.append(
            QAEntry(
                id=draft_id,
                topic=item.topic,
                difficulty=item.difficulty,
                question=item.question,
                ideal_answer=item.ideal_answer,
                key_points=list(item.key_points),
            )
        )
    return drafts


__all__ = [
    "extract_text_from_upload",
    "cap_text",
    "draft_entries_from_text",
    "TEXT_CAP_CHARS",
]
