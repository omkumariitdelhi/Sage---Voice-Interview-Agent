"""Typed representation of a single entry in ``data/qa_bank.yaml``.

This module defines the data shape only. Parsing the YAML lives exclusively
in :mod:`app.retrieval.loader` (the only place permitted to read
``data/qa_bank.yaml`` directly, per the project's grep-checkable convention —
see that module's docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QAEntry:
    """One reference question from the interview Q&A bank.

    Mirrors the schema documented in ``data/qa_bank.yaml``'s header comment:
    id / topic / difficulty / question / ideal_answer / key_points. Every
    field must survive a write-then-read round trip through the vector store
    intact (no truncation, no dropped ``key_points`` entries).
    """

    id: str
    topic: str
    difficulty: str
    question: str
    ideal_answer: str
    key_points: list[str] = field(default_factory=list)
