"""Tests for app/interview/speech_text.py — the markdown-for-speech stripper.

Regression coverage for a real bug: the interviewer LLM sometimes emits
markdown (e.g. ``**bold**``) despite this being a spoken conversation, and
TTS reads literal ``**`` characters aloud as "star star." See
speech_text.py's module docstring for the full root-cause writeup.
"""

from __future__ import annotations

from app.interview.speech_text import clean_for_speech


def test_strips_bold_double_asterisk():
    assert clean_for_speech("Let's talk about **arrays** now.") == "Let's talk about arrays now."


def test_strips_italic_single_asterisk():
    assert clean_for_speech("That's *really* clever.") == "That's really clever."


def test_does_not_touch_underscores():
    """Deliberate design choice, not an oversight: underscores are common
    in legitimate technical content this app's domain discusses (snake_case
    identifiers, dunder methods, ALL_CAPS constants) and a naive
    "any underscore pair is emphasis" rule mangles them — caught live by a
    real test fixture marker string full of underscores. Only asterisks are
    treated as emphasis."""
    text = "Explain __init__ and the user_id variable, plus MY_CONSTANT_NAME."
    assert clean_for_speech(text) == text


def test_strips_inline_code_backticks():
    assert clean_for_speech("Use the `git rebase` command.") == "Use the git rebase command."


def test_strips_heading_marker():
    assert clean_for_speech("## Next question\nWhat's an array?") == "Next question\nWhat's an array?"


def test_strips_bullet_list_markers():
    text = "Consider:\n- option one\n- option two\n* option three"
    assert clean_for_speech(text) == "Consider:\noption one\noption two\noption three"


def test_strips_numbered_list_markers():
    text = "Steps:\n1. clarify\n2) implement"
    assert clean_for_speech(text) == "Steps:\nclarify\nimplement"


def test_leaves_plain_text_unchanged():
    text = "What's the difference between an array and a linked list?"
    assert clean_for_speech(text) == text


def test_handles_multiple_markers_in_one_sentence():
    text = "**What's** the difference between an array and a linked list?"
    assert clean_for_speech(text) == "What's the difference between an array and a linked list?"


def test_does_not_strip_a_lone_asterisk_used_as_multiplication():
    """A single, unpaired `*` (e.g. in "2 * 3") isn't markdown emphasis —
    the regex requires non-empty content between two matching markers, so a
    lone multiplication sign should survive untouched."""
    assert clean_for_speech("The result is 2 * 3 = 6.") == "The result is 2 * 3 = 6."


def test_empty_string_returns_empty_string():
    assert clean_for_speech("") == ""
