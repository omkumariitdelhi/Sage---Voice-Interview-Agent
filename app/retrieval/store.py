"""Runtime retrieval API over the persisted Chroma store.

This is the *only* module other phases should import to read Q&A bank
content — never ``app.retrieval.loader`` directly, and never
``data/qa_bank.yaml`` directly. It serves two future consumers per the spec:

- The LangGraph interview flow (Phase 4), which knows the next question
  deterministically (by id or by sequence position) — see :func:`get_by_id`
  and :func:`get_by_sequence_index`.
- A potential free-text similarity lookup (e.g. grounding a follow-up in the
  retrieved ``ideal_answer``/``key_points``) — see :func:`retrieve`.

This module never writes to the store; :mod:`app.retrieval.ingest` is the
sole writer.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.retrieval.exceptions import QAEntryNotFoundError, StoreNotIngestedError
from app.retrieval.ingest import (
    COLLECTION_NAME,
    DEFAULT_CHROMA_DIR,
    KEY_POINTS_DELIMITER,
    get_embeddings,
)
from app.retrieval.schema import QAEntry


def _document_to_entry(doc: Document) -> QAEntry:
    meta = doc.metadata
    key_points_raw = meta.get("key_points", "")
    key_points = key_points_raw.split(KEY_POINTS_DELIMITER) if key_points_raw else []
    return QAEntry(
        id=meta["id"],
        topic=meta["topic"],
        difficulty=meta["difficulty"],
        question=doc.page_content,
        ideal_answer=meta["ideal_answer"],
        key_points=key_points,
    )


def _open_store(chroma_dir: str | Path = DEFAULT_CHROMA_DIR) -> Chroma:
    """Open (read) the persisted collection at ``chroma_dir``.

    Raises:
        StoreNotIngestedError: ``chroma_dir`` doesn't exist yet, meaning
            ``python -m app.retrieval.ingest`` has never been run.
    """
    chroma_path = Path(chroma_dir)
    if not chroma_path.exists():
        raise StoreNotIngestedError(
            f"No Chroma store found at {chroma_path}. Run "
            f"`python -m app.retrieval.ingest` first."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(chroma_path),
    )


def get_by_id(qa_id: str, *, chroma_dir: str | Path = DEFAULT_CHROMA_DIR) -> QAEntry:
    """Deterministically fetch the entry whose YAML ``id`` is ``qa_id``.

    Raises:
        StoreNotIngestedError: the store hasn't been built yet.
        QAEntryNotFoundError: no entry with this id exists in the store.
    """
    store = _open_store(chroma_dir)
    result = store.get(ids=[qa_id], include=["metadatas", "documents"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    if not docs or not metas:
        raise QAEntryNotFoundError(f"No Q&A entry with id '{qa_id}' in the store.")
    return _document_to_entry(Document(page_content=docs[0], metadata=metas[0]))


def get_by_sequence_index(
    i: int, *, chroma_dir: str | Path = DEFAULT_CHROMA_DIR
) -> QAEntry:
    """Deterministically fetch the entry at 0-based declaration-order index
    ``i`` in ``data/qa_bank.yaml`` (recorded as ``seq_index`` metadata at
    ingest time — Chroma itself has no inherent ordering guarantee).

    Raises:
        StoreNotIngestedError: the store hasn't been built yet.
        QAEntryNotFoundError: ``i`` is out of range for the current store.
    """
    store = _open_store(chroma_dir)
    result = store.get(
        where={"seq_index": i}, include=["metadatas", "documents"]
    )
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    if not docs or not metas:
        raise QAEntryNotFoundError(f"No Q&A entry at sequence index {i} in the store.")
    return _document_to_entry(Document(page_content=docs[0], metadata=metas[0]))


def retrieve(
    query: str, k: int = 1, *, chroma_dir: str | Path = DEFAULT_CHROMA_DIR
) -> list[QAEntry]:
    """Semantic similarity search over question text.

    Returns an empty list (never raises) when the store has zero entries —
    callers (e.g. a future follow-up grounding step) must handle the
    empty-retrieval case explicitly rather than assume a result always
    exists.

    Raises:
        StoreNotIngestedError: the store hasn't been built yet.
    """
    store = _open_store(chroma_dir)
    if count(chroma_dir=chroma_dir) == 0:
        return []
    results = store.similarity_search(query, k=k)
    return [_document_to_entry(doc) for doc in results]


def count(*, chroma_dir: str | Path = DEFAULT_CHROMA_DIR) -> int:
    """Total number of entries currently in the store (so Phase 4 knows when
    the bank is exhausted).

    Uses the public ``get()`` API (ids only, no embeddings/documents) rather
    than any private Chroma client attribute, so this keeps working across
    langchain-chroma versions.

    Raises:
        StoreNotIngestedError: the store hasn't been built yet.
    """
    store = _open_store(chroma_dir)
    result = store.get(include=[])
    return len(result.get("ids") or [])
