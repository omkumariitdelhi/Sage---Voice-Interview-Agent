"""Regression guard for the spec's acceptance criterion: "No application code
outside app/retrieval/ parses data/qa_bank.yaml directly (grep-checkable)."

This scans every .py file under app/ (excluding app/retrieval/ itself) and
asserts none of them references the qa_bank.yaml filename or path. Phase 4+
must go through app.retrieval.store, never read the YAML directly.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
RETRIEVAL_DIR = APP_DIR / "retrieval"


def test_only_loader_module_references_qa_bank_yaml():
    offending: list[str] = []

    for py_file in APP_DIR.rglob("*.py"):
        if RETRIEVAL_DIR in py_file.parents or py_file.parent == RETRIEVAL_DIR:
            continue
        text = py_file.read_text(encoding="utf-8")
        if "qa_bank.yaml" in text or "qa_bank" in text:
            offending.append(str(py_file.relative_to(PROJECT_ROOT)))

    assert offending == [], (
        f"Found references to the Q&A bank outside app/retrieval/: {offending}. "
        "Only app/retrieval/loader.py is allowed to know about qa_bank.yaml."
    )


def test_loader_module_is_the_one_that_parses_qa_bank_yaml():
    loader_text = (RETRIEVAL_DIR / "loader.py").read_text(encoding="utf-8")
    assert "yaml.safe_load" in loader_text
