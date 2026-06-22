"""Local retrieval/grounding store over ``data/qa_bank.yaml``.

Public surface for other phases (the LangGraph interview flow in Phase 4,
any future free-text grounding lookup):

    from app.retrieval import QAEntry, get_by_id, get_by_sequence_index, retrieve, count

Only :mod:`app.retrieval.loader` parses ``data/qa_bank.yaml`` directly.
Only :mod:`app.retrieval.ingest` writes to the persisted Chroma store.
Everything else should read through this package's :mod:`app.retrieval.store`
functions, re-exported below.
"""

from __future__ import annotations

from app.retrieval.exceptions import (
    QABankLoadError,
    QAEntryNotFoundError,
    RetrievalError,
    StoreNotIngestedError,
)
from app.retrieval.schema import QAEntry
from app.retrieval.store import count, get_by_id, get_by_sequence_index, retrieve

__all__ = [
    "QAEntry",
    "RetrievalError",
    "QABankLoadError",
    "QAEntryNotFoundError",
    "StoreNotIngestedError",
    "get_by_id",
    "get_by_sequence_index",
    "retrieve",
    "count",
]
