"""The interviewer's named persona + a once-ever-cached self-introduction.

Caching rationale: the introduction is the one piece of interviewer-facing
text that should sound consistent across every session (it's the persona's
identity statement, not a varying creative response), so generating it fresh
via an LLM call on every single session start is pure waste — write it once,
read it from disk forever after. Only the TEXT is cached; TTS still
synthesizes audio for the combined intro+first-question turn exactly once per
session, same as every other turn (see ``app/web/server.py::start_session``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.gateway.exceptions import GatewayError
from app.gateway.llm import complete

logger = logging.getLogger(__name__)

INTERVIEWER_NAME = "Sage"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTRO_CACHE_PATH = _PROJECT_ROOT / ".cache" / "intro.json"


def _build_intro_prompt(domain: str, dataset_intro: str) -> list[dict]:
    system = (
        f"You are {INTERVIEWER_NAME}, a professional, friendly technical "
        "interviewer. Write your own opening self-introduction for a "
        "screening interview, in your own natural spoken voice — not a "
        "written announcement."
    )
    user = (
        f"Interview domain: {domain}\n\n"
        f"Session framing to weave in naturally (don't quote it verbatim, "
        f"capture the same intent in your own words): {dataset_intro}\n\n"
        f"Write a short (2-3 sentence) spoken self-introduction: say your "
        f"name, that you'll be conducting today's interview, set "
        f"expectations (a mix of questions, encourage thinking out loud), "
        f"and invite them to say when they're ready to start. Return ONLY "
        f"the introduction text, no preamble or formatting."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def get_or_generate_intro_text(domain: str, dataset_intro: str) -> str:
    """Return the cached introduction text, generating + persisting it via
    one LLM call if no cache exists yet.

    Fail-soft (mirrors ``app/interview/graph.py``'s ``_safe_complete``):
    if the gateway call fails (no key configured yet, transient failure),
    falls back to ``dataset_intro`` as-is and does NOT write a cache, so the
    next call (e.g. the next server startup) tries again rather than
    permanently freezing in a fallback. Never raises.
    """
    if INTRO_CACHE_PATH.is_file():
        try:
            cached = json.loads(INTRO_CACHE_PATH.read_text(encoding="utf-8"))
            text = cached.get("text")
            if text:
                return text
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "persona: failed to read intro cache at %s (%s); regenerating.",
                INTRO_CACHE_PATH,
                exc,
            )

    try:
        text = complete(_build_intro_prompt(domain, dataset_intro))
    except GatewayError as exc:
        logger.warning(
            "persona: gateway call failed generating cached intro (%s); "
            "falling back to the dataset's raw intro text for this run.",
            exc,
        )
        return dataset_intro

    try:
        INTRO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        INTRO_CACHE_PATH.write_text(json.dumps({"text": text}), encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "persona: failed to write intro cache at %s (%s); will regenerate "
            "next time.",
            INTRO_CACHE_PATH,
            exc,
        )

    return text


__all__ = ["INTERVIEWER_NAME", "INTRO_CACHE_PATH", "get_or_generate_intro_text"]
