"""Strips markdown formatting from LLM-generated interviewer text before it
reaches either the transcript display or TTS.

Real bug, not theoretical: the interviewer LLM sometimes emits markdown
(``**bold**``, headers, bullet lists) despite this being a spoken
conversation — the frontend renders the transcript as plain text (so a
literal ``**`` shows up in the bubble), and TTS synthesizes whatever string
it's given verbatim (so it's read aloud as "star star"). The root-cause fix
is the explicit "no markdown" instruction in
``app/interview/prompts.py``'s ``_SYSTEM_PERSONA``; this module is the
defensive backstop for whatever slips through anyway, applied once at the
single chokepoint where a turn's text is finalized
(``app/interview/graph.py``'s ``interviewer_turn`` node) rather than at each
of the five branches that can produce text.
"""

from __future__ import annotations

import re

# Deliberately asterisk-only for bold/italic — NOT underscore-based
# (no `__bold__`/`_italic_` handling). Underscores are extremely common in
# legitimate technical content this app's domain actually discusses
# (snake_case identifiers, dunder methods like `__init__`, ALL_CAPS
# constants) and a naive "any underscore pair is emphasis" rule mangles
# them — caught live by a test whose own fixture marker string
# (`ZZZ_UNIQUE_..._ZZZ`, several underscores) got its underscores silently
# stripped. Asterisks essentially never appear mid-word in normal spoken
# English/technical prose, so they're a safe signal for "this is markdown"
# in a way underscores are not.
#
# Order matters: strip bold/italic emphasis before the leading-character
# rules below, so a bolded list item ("**- item**") doesn't leave a stray
# leading "- " behind after the emphasis markers are removed.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    """Return ``text`` with common markdown markup stripped, leaving the
    underlying words intact. Never raises; returns ``text`` unchanged if it
    contains nothing recognizable as markdown."""
    if not text:
        return text

    cleaned = _BOLD_RE.sub(lambda m: m.group(1), text)
    cleaned = _ITALIC_RE.sub(lambda m: m.group(1), cleaned)
    cleaned = _INLINE_CODE_RE.sub(lambda m: m.group(1), cleaned)
    cleaned = _HEADING_RE.sub("", cleaned)
    cleaned = _LIST_MARKER_RE.sub("", cleaned)
    return cleaned


__all__ = ["clean_for_speech"]
