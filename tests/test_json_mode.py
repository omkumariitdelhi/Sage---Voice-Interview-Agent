"""Tests for app/json_mode.py — the shared markdown-code-fence stripper.

Regression coverage for a real, live-observed bug: Claude Haiku 4.5 wraps
JSON-mode completions in ```json fences even when explicitly instructed not
to (response_format={"type": "json_object"} is a no-op for the Anthropic
provider in the installed litellm version when no explicit json_schema is
supplied — see app/json_mode.py's module docstring), and every JSON-mode
_parse() helper (evaluation.py, feedback.py, extractor.py) was calling
json.loads() directly, silently discarding a correctly-graded verdict.
"""

from __future__ import annotations

import json

from app.json_mode import strip_json_fences


def test_strips_json_tagged_fence():
    raw = '```json\n{"a": 1}\n```'
    assert strip_json_fences(raw) == '{"a": 1}'
    assert json.loads(strip_json_fences(raw)) == {"a": 1}


def test_strips_bare_fence_no_language_tag():
    raw = '```\n{"a": 1}\n```'
    assert strip_json_fences(raw) == '{"a": 1}'


def test_leaves_unfenced_json_unchanged():
    raw = '{"a": 1}'
    assert strip_json_fences(raw) == '{"a": 1}'


def test_strips_surrounding_whitespace_around_fence():
    raw = '  \n```json\n{"a": 1}\n```\n  '
    assert strip_json_fences(raw) == '{"a": 1}'


def test_handles_multiline_json_inside_fence():
    raw = '```json\n{\n  "a": 1,\n  "b": [1, 2, 3]\n}\n```'
    assert json.loads(strip_json_fences(raw)) == {"a": 1, "b": [1, 2, 3]}


def test_does_not_strip_a_fence_that_only_opens():
    """Malformed/truncated input (no closing fence) is left as-is — still
    correctly fails json.loads() downstream rather than being mangled into
    something that looks parseable but isn't."""
    raw = '```json\n{"a": 1}'
    assert strip_json_fences(raw) == raw
