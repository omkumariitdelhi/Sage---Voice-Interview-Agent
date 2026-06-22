"""Typed exceptions raised by the retrieval/grounding store.

Mirrors ``app/gateway/exceptions.py``'s convention: callers elsewhere in the
app (the LangGraph interview flow in Phase 4, any future free-text grounding
lookup) should depend on *these* exceptions, not on Chroma's or PyYAML's
internal exception types directly. That indirection insulates the rest of the
app from a future vector-store or loader implementation change.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for all retrieval/store errors."""


class QABankLoadError(RetrievalError):
    """Raised when ``data/qa_bank.yaml`` (or a substitute path) cannot be
    read or fails schema validation. Never silently swallowed — a malformed
    dataset must fail loudly at ingest/load time, not produce a half-built
    store."""


class QAEntryNotFoundError(RetrievalError):
    """Raised by :func:`app.retrieval.store.get_by_id` /
    :func:`app.retrieval.store.get_by_sequence_index` when the requested
    entry does not exist in the current store."""


class StoreNotIngestedError(RetrievalError):
    """Raised when the runtime store API is used before
    ``python -m app.retrieval.ingest`` has ever been run (no persisted
    Chroma collection found at the expected path)."""


class QAEntryIdCollisionError(RetrievalError):
    """Raised by :func:`app.retrieval.writer.add_entries` when a caller
    supplies an explicit ``id`` that already exists in the dataset.
    Auto-generated ids (the caller passes ``id=None``) never raise this —
    :func:`app.retrieval.writer.generate_id` only ever returns a free id."""


class LastEntryDeletionError(RetrievalError):
    """Raised by :func:`app.retrieval.writer.delete_entry` when asked to
    remove the only remaining entry in the dataset — doing so would break
    ``max_turns_ceiling``'s formula and the whole interview flow (a session
    needs at least one question)."""


class UnsupportedDocumentTypeError(RetrievalError):
    """Raised by :func:`app.retrieval.extractor.extract_text_from_upload`
    when the uploaded file's extension/content-type is not one of the
    supported document types (``.pdf``, ``.md``, ``.txt``)."""


class ExtractionFailedError(RetrievalError):
    """Raised by :func:`app.retrieval.extractor.draft_entries_from_text`
    when the LLM gateway call fails, or returns persistently malformed JSON
    after the single sanctioned re-ask retry. Unlike
    ``app.interview.evaluation``'s evaluate-answer path (which must never
    raise mid-interview and instead degrades to a safe fallback verdict),
    this extraction-preview path is explicitly specified to surface a clean
    failure (mapped to HTTP 502 by the web layer) rather than fabricate a
    draft — see ``.claude/loop-state/qa-bank-management/spec.md``'s
    document-upload flow, step 4."""
