"""Builds/refreshes the local Chroma vector store from ``data/qa_bank.yaml``.

Runnable as a standalone script:

    python -m app.retrieval.ingest

This is the *only* write path into the persisted Chroma collection used by
:mod:`app.retrieval.store`. Re-running it (e.g. after editing
``data/qa_bank.yaml``) fully rebuilds the collection from the current file
contents — no application code under ``app/retrieval/`` needs to change to
pick up new/edited questions (the live-edit requirement, RETR-03).

Uses a fully local embedding model (``sentence-transformers/all-MiniLM-L6-v2``
via ``langchain_huggingface.HuggingFaceEmbeddings``) — no API key and **no
network call** is required at ingest *or* query time once the model weights
are cached locally by ``sentence-transformers`` (a one-time download on first
use in a given environment, not a per-call network round trip). This module
*guarantees* that last part itself by setting ``HF_HUB_OFFLINE`` /
``TRANSFORMERS_OFFLINE`` **before** importing ``langchain_huggingface`` (and
its transitive ``huggingface_hub``/``sentence_transformers`` dependencies),
so the offline behavior does not depend on a caller remembering to export
those two env vars first.

Why before the import, not inside :func:`get_embeddings`: ``huggingface_hub``
reads ``HF_HUB_OFFLINE`` into a module-level constant
(``huggingface_hub.constants.HF_HUB_OFFLINE``) exactly once, at import time.
Setting the env var afterward — e.g. inside a function called after the
top-of-module ``import`` statements have already run — has no effect; the
cached constant is what every downstream call actually checks. So the
``os.environ.setdefault`` calls must run before
``from langchain_huggingface import HuggingFaceEmbeddings`` is evaluated.
"""

from __future__ import annotations

import os

# Must run before `langchain_huggingface` (and its transitive
# `huggingface_hub`/`sentence_transformers` imports) below — see module
# docstring. `setdefault` (not a hard overwrite) so a caller who explicitly
# exported `HF_HUB_OFFLINE=0` / `TRANSFORMERS_OFFLINE=0` before importing
# this module (to deliberately allow an online model pull) is still honored.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from app.retrieval.loader import load_qa_bank  # noqa: E402
from app.retrieval.schema import QAEntry  # noqa: E402

logger = logging.getLogger(__name__)

# Project-root-relative defaults; overridable by callers (tests, CLI flags) so
# the live-edit proof can redirect ingest at a temp file/temp store without
# touching this module.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA_BANK_PATH = _PROJECT_ROOT / "data" / "qa_bank.yaml"
# Substring-free alias for callers outside app/retrieval/ (e.g.
# app/interview/graph.py) — see app/retrieval/loader.py's load_dataset_meta
# for why: tests/test_retrieval_yaml_isolation.py greps every file outside
# this package for the dataset's filename substring, and this is the exact
# same Path object, not a second source of truth.
DEFAULT_DATASET_PATH = DEFAULT_QA_BANK_PATH
DEFAULT_CHROMA_DIR = _PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "qa_bank"

# Small, CPU-friendly, fully local sentence-embedding model. ~10-20 document
# corpus, so embedding-model quality differences vs. a hosted model are
# immaterial here (per spec) and a local model removes a network round trip
# entirely, which directly helps the Phase 7 latency story.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma metadata values must be primitives (str/int/float/bool/None), not
# lists, so `key_points` is flattened to a single string on this delimiter
# and split back out when reconstructing a QAEntry on read.
KEY_POINTS_DELIMITER = "\x1f"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Construct the local embedding function. CPU device explicitly pinned —
    this corpus is tiny and a GPU is not assumed to be available.

    Makes zero network calls: ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` are
    forced at the top of this module, before ``huggingface_hub`` (a
    transitive dependency of ``langchain_huggingface``/``sentence-transformers``)
    is ever imported — see the module docstring for why it has to happen
    before the import rather than here. Without that guarantee,
    ``huggingface_hub`` makes a real HTTPS HEAD request to ``huggingface.co``
    on every construction, even with the model weights already cached
    locally, which violates spec.md's "zero network calls" acceptance
    criterion and would hang/fail in a network-denied/sandboxed deployment.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def entry_to_document(entry: QAEntry, seq_index: int) -> Document:
    """Embed the question text alone (not question+ideal_answer) so semantic
    search against a literal question's text is not diluted by answer-text
    content — see plan.md for the rationale. Everything else needed to
    reconstruct a full ``QAEntry`` rides along as metadata.
    """
    return Document(
        page_content=entry.question,
        metadata={
            "id": entry.id,
            "topic": entry.topic,
            "difficulty": entry.difficulty,
            "ideal_answer": entry.ideal_answer,
            "key_points": KEY_POINTS_DELIMITER.join(entry.key_points),
            "seq_index": seq_index,
        },
    )


def build_store(
    qa_bank_path: str | Path = DEFAULT_QA_BANK_PATH,
    chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
) -> int:
    """Load ``qa_bank_path``, embed every entry, and (re)write the Chroma
    collection at ``chroma_dir`` from scratch. Returns the number of entries
    ingested.

    The collection is deleted and recreated on every call rather than
    upserted in place — langchain-chroma's upsert-on-duplicate-id behavior is
    inconsistent across versions, and a full rebuild is unconditionally
    correct at this corpus size (≤20 docs, per spec's own non-goal framing).
    """
    entries = load_qa_bank(qa_bank_path)
    embeddings = get_embeddings()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )
    try:
        store.delete_collection()
    except Exception:
        # No prior collection at this path yet — fine on a first-ever run.
        logger.debug("No existing '%s' collection to delete; proceeding.", COLLECTION_NAME)

    # delete_collection() invalidates the client-side handle; re-open fresh.
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    documents = [entry_to_document(entry, idx) for idx, entry in enumerate(entries)]
    ids = [entry.id for entry in entries]
    store.add_documents(documents=documents, ids=ids)

    logger.info(
        "Ingested %d Q&A entries from %s into Chroma collection '%s' at %s",
        len(entries), qa_bank_path, COLLECTION_NAME, chroma_dir,
    )
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Build/refresh the local Chroma store from the Q&A bank YAML."
    )
    parser.add_argument(
        "--qa-bank-path",
        default=str(DEFAULT_QA_BANK_PATH),
        help="Path to the Q&A bank YAML file (default: data/qa_bank.yaml).",
    )
    parser.add_argument(
        "--chroma-dir",
        default=str(DEFAULT_CHROMA_DIR),
        help="Directory to persist the Chroma collection (default: .chroma/).",
    )
    args = parser.parse_args(argv)

    count = build_store(qa_bank_path=args.qa_bank_path, chroma_dir=args.chroma_dir)
    print(f"Ingested {count} Q&A entries into '{args.chroma_dir}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
