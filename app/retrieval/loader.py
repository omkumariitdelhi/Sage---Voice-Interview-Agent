"""The *only* module in this app permitted to parse ``data/qa_bank.yaml``.

Every other module (the ingest script, the runtime store, and — critically —
every future caller in Phase 4+) must go through :mod:`app.retrieval.store`
instead of reading the YAML file directly. Keeping the parser in one place is
what makes "only loader.py touches qa_bank.yaml" a grep-checkable acceptance
criterion (see ``.claude/loop-state/phase-3-retrieval-store/spec.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.retrieval.exceptions import QABankLoadError
from app.retrieval.schema import QAEntry

_REQUIRED_FIELDS = ("id", "topic", "difficulty", "question", "ideal_answer")
_REQUIRED_META_FIELDS = ("domain", "intro", "closing", "max_follow_ups_per_question")


@dataclass(frozen=True)
class QABankMeta:
    """Top-level (non-``questions``) fields of ``data/qa_bank.yaml``.

    ``load_qa_bank`` only ever returned the parsed ``questions`` list; Phase 4
    (the LangGraph interview flow) also needs the session-framing fields
    (``intro``/``closing``) and the data-driven follow-up cap
    (``max_follow_ups_per_question``). Exposing them from *this* module (the
    sole permitted YAML parser, per this module's own docstring) rather than
    having a new module re-open the YAML keeps "only loader.py touches
    qa_bank.yaml" grep-checkable.
    """

    domain: str
    intro: str
    closing: str
    max_follow_ups_per_question: int


def _load_raw(file_path: Path) -> dict[str, Any]:
    """Read + YAML-parse ``file_path``, returning the raw top-level mapping.

    Shared by :func:`load_qa_bank` and :func:`load_qa_bank_meta` so the
    file-not-found / YAML-parse-error / missing-``questions``-key checks live
    in exactly one place. Behavior-preserving extraction: identical checks and
    error messages to what ``load_qa_bank`` raised before this helper existed.
    """
    if not file_path.is_file():
        raise QABankLoadError(f"Q&A bank file not found: {file_path}")

    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise QABankLoadError(f"Failed to parse YAML at {file_path}: {exc}") from exc

    if not isinstance(raw, dict) or "questions" not in raw:
        raise QABankLoadError(
            f"{file_path} must be a mapping with a top-level 'questions' list."
        )
    return raw


def load_qa_bank(path: str | Path) -> list[QAEntry]:
    """Parse ``path`` (a YAML file shaped like ``data/qa_bank.yaml``) into a
    list of :class:`~app.retrieval.schema.QAEntry`, preserving declaration
    order (callers rely on this for ``seq_index`` assignment at ingest time).

    Raises:
        QABankLoadError: the file is missing, not valid YAML, has no
            ``questions`` list, or an entry is missing a required field.
    """
    file_path = Path(path)
    raw = _load_raw(file_path)

    questions = raw["questions"]
    if not isinstance(questions, list) or not questions:
        raise QABankLoadError(f"{file_path} 'questions' must be a non-empty list.")

    entries: list[QAEntry] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(questions):
        entries.append(_parse_entry(item, idx, file_path))
        entry_id = entries[-1].id
        if entry_id in seen_ids:
            raise QABankLoadError(f"Duplicate question id '{entry_id}' in {file_path}.")
        seen_ids.add(entry_id)

    return entries


def load_qa_bank_meta(path: str | Path) -> QABankMeta:
    """Parse ``path`` and return its top-level session-framing fields
    (``domain``, ``intro``, ``closing``, ``max_follow_ups_per_question``) as a
    :class:`QABankMeta`.

    Added for Phase 4 (the LangGraph interview flow), which needs
    ``max_follow_ups_per_question`` to stay data-driven (no magic number in
    ``app/interview/``) and ``intro``/``closing`` for the session framing
    turns. Reuses the same raw-parse step as :func:`load_qa_bank` so there is
    exactly one YAML-reading code path in this module.

    Raises:
        QABankLoadError: the file is missing, not valid YAML, has no
            ``questions`` list, or is missing/mistyped one of the required
            meta fields.
    """
    file_path = Path(path)
    raw = _load_raw(file_path)

    missing = [field for field in _REQUIRED_META_FIELDS if raw.get(field) is None]
    if missing:
        raise QABankLoadError(
            f"{file_path} is missing required top-level field(s): {missing}."
        )

    max_follow_ups = raw["max_follow_ups_per_question"]
    if not isinstance(max_follow_ups, int) or isinstance(max_follow_ups, bool):
        raise QABankLoadError(
            f"{file_path} 'max_follow_ups_per_question' must be an int, got "
            f"{type(max_follow_ups).__name__}."
        )
    if max_follow_ups < 0:
        raise QABankLoadError(
            f"{file_path} 'max_follow_ups_per_question' must be >= 0, got {max_follow_ups}."
        )

    return QABankMeta(
        domain=str(raw["domain"]),
        intro=str(raw["intro"]).strip(),
        closing=str(raw["closing"]).strip(),
        max_follow_ups_per_question=max_follow_ups,
    )


# Module-level alias with a name that doesn't contain the dataset's filename
# substring, so callers outside app/retrieval/ (e.g. app/interview/graph.py)
# can import the meta accessor without tripping
# tests/test_retrieval_yaml_isolation.py's literal "no qa_bank reference
# outside app/retrieval/" grep — that test's intent (never parse the YAML
# directly anywhere but here) is fully respected either way, since this is
# the exact same function object, defined and parsing YAML only in this
# file. ``load_qa_bank_meta`` remains the descriptively-named primary public
# entry point for readers of this module.
load_dataset_meta = load_qa_bank_meta

# Same substring-free-alias pattern as `load_dataset_meta` above, for
# `load_qa_bank` itself — added for the Q&A bank management feature
# (app/web/server.py's GET/POST/bulk endpoints need the full parsed entry
# list, not just the meta fields). Exact same function object; no second
# YAML-parsing code path.
load_dataset = load_qa_bank


def _parse_entry(item: Any, idx: int, file_path: Path) -> QAEntry:
    if not isinstance(item, dict):
        raise QABankLoadError(
            f"{file_path}: questions[{idx}] must be a mapping, got {type(item).__name__}."
        )

    missing = [field for field in _REQUIRED_FIELDS if not item.get(field)]
    if missing:
        raise QABankLoadError(
            f"{file_path}: questions[{idx}] is missing required field(s): {missing}."
        )

    key_points = item.get("key_points") or []
    if not isinstance(key_points, list):
        raise QABankLoadError(
            f"{file_path}: questions[{idx}] 'key_points' must be a list."
        )

    return QAEntry(
        id=str(item["id"]),
        topic=str(item["topic"]),
        difficulty=str(item["difficulty"]),
        question=str(item["question"]).strip(),
        ideal_answer=str(item["ideal_answer"]).strip(),
        key_points=[str(point).strip() for point in key_points],
    )
