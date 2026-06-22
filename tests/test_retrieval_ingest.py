"""Tests for app/retrieval/ingest.py.

Covers the spec's first acceptance criterion: ingest succeeds with zero real
API keys and zero network calls at query time (local embedding model only).
"""

from __future__ import annotations

import shutil
import socket

import pytest

from app.retrieval.ingest import build_store
from app.retrieval.store import _open_store, count


@pytest.fixture
def temp_chroma_dir(tmp_path):
    chroma_dir = tmp_path / "chroma_store"
    yield chroma_dir
    shutil.rmtree(chroma_dir, ignore_errors=True)


def test_ingest_succeeds_with_no_api_keys_set(monkeypatch, temp_chroma_dir):
    """Proves ingest needs zero OPENROUTER_API_KEY /
    DEEPGRAM_API_KEY — local embeddings only, per spec's first acceptance
    criterion."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    ingested = build_store(chroma_dir=temp_chroma_dir)

    assert ingested == 10
    assert count(chroma_dir=temp_chroma_dir) == 10


def test_ingest_against_real_qa_bank_path_default(temp_chroma_dir):
    """Sanity check using the real default `data/qa_bank.yaml` path (the
    default argument), writing to a temp chroma dir so we don't touch the
    real `.chroma/`."""
    ingested = build_store(chroma_dir=temp_chroma_dir)
    assert ingested == 10


def test_rerunning_ingest_is_idempotent_in_count(temp_chroma_dir):
    """Running ingest twice against the same dir must not duplicate entries
    (full-rebuild-on-every-run strategy, not an upsert)."""
    build_store(chroma_dir=temp_chroma_dir)
    build_store(chroma_dir=temp_chroma_dir)

    assert count(chroma_dir=temp_chroma_dir) == 10


def test_ingest_and_store_open_make_zero_network_connections(monkeypatch, temp_chroma_dir):
    """Regression test for the Critical finding in review-01.md: ingest and
    store-open must never attempt an outbound network connection.

    `app/retrieval/ingest.py` forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`
    at *module import time*, before `huggingface_hub` is imported — not
    inside `get_embeddings()` — because `huggingface_hub` reads that env var
    into a module-level constant exactly once, at its own import time
    (confirmed against the installed package; see ingest.py's module
    docstring). That means deleting the env vars from `os.environ` inside a
    test (after `app.retrieval.ingest` has already been imported earlier in
    the pytest session) does NOT undo the fix — which is exactly what we want
    to assert here: the guarantee survives regardless of `os.environ`'s
    state at call time, only the import-time side effect matters.

    So this test asserts the real, durable signal of the fix —
    `huggingface_hub.constants.HF_HUB_OFFLINE` (the cached boolean the
    library actually branches on) — rather than `os.environ`, and proves the
    behavioral guarantee directly with the socket trap below.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    import huggingface_hub.constants as hf_constants

    assert hf_constants.HF_HUB_OFFLINE is True, (
        "huggingface_hub.constants.HF_HUB_OFFLINE was not forced True at "
        "import time — app/retrieval/ingest.py must set HF_HUB_OFFLINE=1 "
        "before `from langchain_huggingface import HuggingFaceEmbeddings` "
        "is evaluated."
    )

    connections_attempted: list[tuple] = []

    def _blocked_create_connection(address, *args, **kwargs):
        connections_attempted.append(address)
        raise RuntimeError(f"NETWORK CALL ATTEMPTED to {address!r}")

    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)

    # Both the write path (ingest) and the read path (store open) must be
    # network-free — exercise both with the socket trap active.
    ingested = build_store(chroma_dir=temp_chroma_dir)
    assert ingested == 10

    store = _open_store(chroma_dir=temp_chroma_dir)
    assert store is not None

    assert connections_attempted == [], (
        f"Expected zero network connections, but socket.create_connection "
        f"was called with: {connections_attempted}"
    )
