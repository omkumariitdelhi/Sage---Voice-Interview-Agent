"""Shared helper for parsing JSON out of an LLM completion that may be
wrapped in a markdown code fence despite being asked for strict JSON-only
output.

Real, observed failure mode (not theoretical): with the interviewer LLM on
Claude Haiku 4.5, ``app.gateway.llm.complete(..., response_format={"type":
"json_object"})`` does NOT structurally enforce JSON output for the
``anthropic`` provider in the installed litellm version — confirmed via
direct introspection of
``litellm.llms.anthropic.chat.transformation.AnthropicConfig.map_response_format_to_anthropic_tool``,
which only engages Anthropic's tool-call-based JSON enforcement when an
explicit ``json_schema``/``response_schema`` key is present in
``response_format``; a bare ``{"type": "json_object"}`` (what every JSON-mode
call site in this app passes) makes that function return ``None`` — a
silent no-op. Without that structural enforcement, the model is free to
wrap its JSON in a ` ```json ... ``` ` fence even when explicitly told not
to (observed live, including on the single re-ask-on-malformed-JSON retry
every JSON-mode call site already has) — every ``_parse()`` helper
(``app/interview/evaluation.py``, ``app/interview/feedback.py``,
``app/retrieval/extractor.py``) was calling ``json.loads(raw_text)``
directly, so a fenced response silently failed to parse and fell back to a
generic safe verdict, discarding the model's actual (often correct)
judgement.

Stripping a wrapping fence before ``json.loads()`` is the minimal, low-risk
fix: it doesn't change the LLM call shape (no schema authoring/validation
risk), works regardless of *why* a fence appears, and slots into the
existing parse -> reask-once -> fallback pattern at every call site
unchanged otherwise.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def strip_json_fences(raw_text: str) -> str:
    """Strip a single leading/trailing markdown code fence (optionally
    tagged ```json) around ``raw_text``, if present. Returns ``raw_text``
    stripped of surrounding whitespace, unchanged otherwise. Never raises —
    callers still do their own ``json.loads()``/error handling."""
    stripped = raw_text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


__all__ = ["strip_json_fences"]
