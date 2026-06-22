"""Runnable example: demonstrates the Phase 3 retrieval/grounding store API.

Usage (from the project root, inside the project's .venv):

    .venv\\Scripts\\python.exe -m app.retrieval.ingest   # build the store once
    .venv\\Scripts\\python.exe examples\\run_retrieval_example.py

Requires zero API keys and makes zero network calls — this script only
exercises local embeddings + the local Chroma store built from
`data/qa_bank.yaml`. `app.retrieval.ingest.get_embeddings()` forces
`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` before constructing the embedding
model, so once the `sentence-transformers` model weights have been downloaded
once (the only network call this stack ever makes), every subsequent ingest
or store open here is guaranteed offline by the code itself, not by the
caller remembering to export those env vars.
"""

from __future__ import annotations

from app.retrieval import StoreNotIngestedError, count, get_by_id, get_by_sequence_index, retrieve


def main() -> None:
    try:
        total = count()
    except StoreNotIngestedError:
        print("Store not built yet. Run: python -m app.retrieval.ingest")
        return

    print(f"Q&A bank has {total} entries.\n")

    by_id = get_by_id("q01")
    print(f"get_by_id('q01') -> [{by_id.id}] ({by_id.topic}/{by_id.difficulty})")
    print(f"  question: {by_id.question[:80]}...\n")

    by_seq = get_by_sequence_index(0)
    print(f"get_by_sequence_index(0) -> [{by_seq.id}] matches get_by_id('q01'): "
          f"{by_id == by_seq}\n")

    query = (
        "What is Big-O notation, and what's the time complexity of binary "
        "search versus linear search? Why the difference?"
    )
    top = retrieve(query, k=1)
    print(f"retrieve(<q03's literal question text>, k=1) -> top-1 id: {top[0].id}")
    print(f"  key_points: {top[0].key_points}\n")

    follow_up_query = "How do I prevent two threads from corrupting shared data?"
    semantic = retrieve(follow_up_query, k=2)
    print(f"retrieve(<free-text follow-up>, k=2) -> "
          f"{[entry.id for entry in semantic]}")


if __name__ == "__main__":
    main()
