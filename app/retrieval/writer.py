"""Write capability for ``data/qa_bank.yaml`` — the YAML-isolation-respecting
sibling of :mod:`app.retrieval.loader` (which only reads).

Lives inside ``app/retrieval/`` deliberately, per this project's
grep-checkable convention (see ``loader.py``'s own docstring and
``tests/test_retrieval_yaml_isolation.py``): this is the *only* other module
permitted to call ``yaml.safe_load``/``yaml.safe_dump`` on the dataset file.
Every write path (manual single-entry CRUD, bulk commit) funnels through the
three public entry points below — :func:`add_entries`, :func:`replace_entry`,
:func:`delete_entry` — each doing exactly one read-modify-write of the whole
file per call. None of these functions rebuild the Chroma index themselves;
callers (the FastAPI endpoints in ``app/web/server.py``) are responsible for
calling :func:`app.retrieval.ingest.build_store` exactly once after a
successful write, so a multi-entry bulk commit costs one rebuild, not one per
item.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any

import yaml

from app.retrieval.exceptions import (
    LastEntryDeletionError,
    QAEntryIdCollisionError,
    QABankLoadError,
    QAEntryNotFoundError,
)
from app.retrieval.schema import QAEntry

_ID_PATTERN = re.compile(r"^q(\d+)$")
_ENTRY_FIELD_ORDER = ("id", "topic", "difficulty", "question", "ideal_answer", "key_points")


def read_header_block(path: str | Path) -> str:
    """Return the file's leading comment block verbatim (every line that is
    blank or starts with ``#``, from the top, stopping at the first line
    that is neither). This is the schema-documentation block at the top of
    ``data/qa_bank.yaml`` that PyYAML's parser silently drops on a
    load+dump round trip — preserving it as a literal string (instead of
    trying to regenerate it) is the only way a rewrite doesn't lose it.

    Returns an empty string if the file has no such leading block (e.g. the
    first line is already a YAML key) — callers should still produce valid
    output in that case, just with no header to prepend.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    header_lines: list[str] = []
    for line in lines:
        if line.strip() == "" or line.lstrip().startswith("#"):
            header_lines.append(line)
        else:
            break
    return "".join(header_lines)


def _load_raw_dict(path: Path) -> dict[str, Any]:
    """Read + parse the full top-level mapping. Deliberately re-parses
    rather than reusing ``app.retrieval.loader``'s private ``_load_raw``
    (underscore-prefixed, not part of that module's public surface) — this
    module lives inside the same exempt package, so calling
    ``yaml.safe_load`` directly here does not violate the isolation
    invariant; it is a second, equally-sanctioned reader, not a third-party
    leak of YAML-parsing outside ``app/retrieval/``.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise QABankLoadError(f"Failed to parse YAML at {path}: {exc}") from exc
    if not isinstance(raw, dict) or "questions" not in raw:
        raise QABankLoadError(
            f"{path} must be a mapping with a top-level 'questions' list."
        )
    return raw


def generate_id(existing_ids: set[str]) -> str:
    """Highest existing ``q<digits>`` id + 1 (zero-padded to width 2 when
    the result still fits, e.g. ``q10`` -> ``q11``; unpadded once it
    overflows two digits, e.g. ``q99`` -> ``q100``). Falls back to a short
    random ``q_<8-hex>`` suffix if no existing id matches the ``q<digits>``
    shape at all (per spec.md's explicit fallback instruction).
    """
    max_num = -1
    for entry_id in existing_ids:
        match = _ID_PATTERN.match(entry_id)
        if match:
            max_num = max(max_num, int(match.group(1)))

    if max_num < 0:
        return f"q_{secrets.token_hex(4)}"

    next_num = max_num + 1
    return f"q{next_num:02d}" if next_num < 100 else f"q{next_num}"


def _entry_to_plain_dict(entry: QAEntry) -> dict[str, Any]:
    """Plain-dict shape matching ``data/qa_bank.yaml``'s existing per-entry
    field order, so ``yaml.safe_dump`` (with ``sort_keys=False``) emits new
    entries in the same field order as hand-authored ones."""
    return {field: getattr(entry, field) for field in _ENTRY_FIELD_ORDER}


def _dump(path: Path, header: str, raw: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    body = {
        "domain": raw["domain"],
        "intro": raw["intro"],
        "closing": raw["closing"],
        "max_follow_ups_per_question": raw["max_follow_ups_per_question"],
        "questions": questions,
    }
    dumped = yaml.safe_dump(body, sort_keys=False, allow_unicode=True)
    separator = "" if (not header or header.endswith("\n\n")) else "\n"
    path.write_text(header + separator + dumped, encoding="utf-8")


def add_entries(path: str | Path, new_entries: list[QAEntry]) -> list[QAEntry]:
    """Append ``new_entries`` to the dataset at ``path`` in one read-modify-
    write, resolving id collisions/auto-generation as it goes.

    Collision policy (per spec.md):
    - A new entry with an explicit ``id`` that already exists (on disk, or
      earlier in this same ``new_entries`` batch) raises
      :class:`QAEntryIdCollisionError` (single-POST's 409 path) — see the
      NOTE below for how bulk-commit gets a different effective policy
      without this function needing a second mode.
    - A new entry with ``id=None`` (or empty) gets the next free id via
      :func:`generate_id`, against the running union of existing + already-
      assigned-in-this-batch ids, so two id-less drafts in the same bulk
      call never collide with each other either.

    NOTE: bulk-commit's "auto-resolve any collision rather than fail the
    batch" policy (spec.md's documented choice for ``POST /api/qa-bank/bulk``)
    is implemented entirely by the caller, BEFORE calling this function: see
    ``app/web/server.py``'s bulk endpoint, which pre-checks each batch
    item's explicit id against the existing+running id set and clears it to
    ``None`` (forcing auto-generation) whenever it would collide, rather
    than passing the colliding id through. ``add_entries`` itself always
    raises :class:`QAEntryIdCollisionError` on an explicit-id collision it
    is given — keeping this module's contract simple and single-mode; the
    "auto-resolve instead of reject" behavior is a thin, explicit, testable
    policy layer one level up, not a hidden second mode in this function.

    Returns the final list of :class:`QAEntry` objects actually written
    (with resolved ids), in the same order as ``new_entries``.

    Raises:
        QABankLoadError: the file is missing or not valid YAML.
        QAEntryIdCollisionError: an explicit id in ``new_entries`` already
            exists on disk or earlier in this same batch.
    """
    file_path = Path(path)
    header = read_header_block(file_path)
    raw = _load_raw_dict(file_path)
    existing_questions: list[dict[str, Any]] = list(raw["questions"])
    existing_ids: set[str] = {str(q["id"]) for q in existing_questions}

    resolved: list[QAEntry] = []
    appended_dicts: list[dict[str, Any]] = []
    known_ids = set(existing_ids)

    for entry in new_entries:
        if entry.id:
            if entry.id in known_ids:
                raise QAEntryIdCollisionError(
                    f"Q&A entry id '{entry.id}' already exists."
                )
            final_id = entry.id
        else:
            final_id = generate_id(known_ids)

        final_entry = QAEntry(
            id=final_id,
            topic=entry.topic,
            difficulty=entry.difficulty,
            question=entry.question,
            ideal_answer=entry.ideal_answer,
            key_points=list(entry.key_points),
        )
        resolved.append(final_entry)
        appended_dicts.append(_entry_to_plain_dict(final_entry))
        known_ids.add(final_id)

    _dump(file_path, header, raw, existing_questions + appended_dicts)
    return resolved


def replace_entry(path: str | Path, qa_id: str, fields: QAEntry) -> QAEntry:
    """Full-replace the fields of the existing entry whose id is ``qa_id``
    (the id itself never changes via this function — position in the
    ``questions`` list is preserved, per the "no reordering" non-goal).

    Raises:
        QABankLoadError: the file is missing or not valid YAML.
        QAEntryNotFoundError: no entry with ``qa_id`` exists.
    """
    file_path = Path(path)
    header = read_header_block(file_path)
    raw = _load_raw_dict(file_path)
    questions: list[dict[str, Any]] = list(raw["questions"])

    index = next(
        (i for i, q in enumerate(questions) if str(q["id"]) == qa_id), None
    )
    if index is None:
        raise QAEntryNotFoundError(f"No Q&A entry with id '{qa_id}'.")

    updated_entry = QAEntry(
        id=qa_id,
        topic=fields.topic,
        difficulty=fields.difficulty,
        question=fields.question,
        ideal_answer=fields.ideal_answer,
        key_points=list(fields.key_points),
    )
    questions[index] = _entry_to_plain_dict(updated_entry)

    _dump(file_path, header, raw, questions)
    return updated_entry


def delete_entry(path: str | Path, qa_id: str) -> None:
    """Remove the entry whose id is ``qa_id``.

    Raises:
        QABankLoadError: the file is missing or not valid YAML.
        QAEntryNotFoundError: no entry with ``qa_id`` exists.
        LastEntryDeletionError: ``qa_id`` is the only remaining entry.
    """
    file_path = Path(path)
    header = read_header_block(file_path)
    raw = _load_raw_dict(file_path)
    questions: list[dict[str, Any]] = list(raw["questions"])

    index = next(
        (i for i, q in enumerate(questions) if str(q["id"]) == qa_id), None
    )
    if index is None:
        raise QAEntryNotFoundError(f"No Q&A entry with id '{qa_id}'.")
    if len(questions) <= 1:
        raise LastEntryDeletionError(
            "Refusing to delete the last remaining Q&A entry — the "
            "interview flow requires at least one question."
        )

    del questions[index]
    _dump(file_path, header, raw, questions)


__all__ = [
    "add_entries",
    "replace_entry",
    "delete_entry",
    "generate_id",
    "read_header_block",
]
