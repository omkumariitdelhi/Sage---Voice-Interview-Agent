"""The no-verbatim-leak custom Guardrails validator + a small factory to
build a single-string ``Guard`` around it.

See ``.claude/loop-state/phase-5-guardrails/plan.md`` Section 2 for the full
threshold-justification writeup. Short version:

- **Exact check (always a violation):** the normalized reference text fully
  contained in the normalized candidate text, OR any contiguous run of
  ``NGRAM_SIZE`` (10) words from the reference appearing verbatim inside the
  candidate text. Catches full leaks and partial-but-still-verbatim sentence
  copying.
- **Fuzzy check (catches synonym-substituted near-verbatim paraphrase):**
  ``difflib.SequenceMatcher`` ratio of the normalized full texts >=
  ``FUZZY_THRESHOLD`` (0.72). Chosen empirically against this project's own
  reference Q&A dataset content (see ``app/retrieval/loader.py`` for the
  sanctioned loader — this module never reads that file directly):
  near-verbatim synonym-substituted copies of this dataset's answers land
  ~0.78-0.85; genuine original paraphrases covering the same key points land
  ~0.30-0.55. 0.72 sits in the gap with margin on both sides — see
  ``tests/test_guardrails_no_leak.py`` for the empirical cases that pin this
  down.
- Deterministic, stdlib-only (no second LLM call) — sub-millisecond per call,
  by design (spec.md's explicit non-goal: don't add latency/cost here).

This validator is intentionally "dumb": it only knows how to compare one
candidate string against one reference string and report pass/fail (plus a
caller-supplied ``fix_value`` for ``OnFailAction.FIX``). All retry/fallback
*orchestration* policy (reask wording, what a safe fallback looks like) lives
at the call site (``app/interview/graph.py``, ``app/interview/feedback.py``),
not here — see plan.md Section 3/4 for why that split was chosen.

**Performance note (iteration 2, self-check.md "Iteration 2"):** constructing
a ``Guard()`` is expensive — ``guardrails-ai==0.10.2``'s ``Guard.__init__``
unconditionally builds a ``GuardrailsApiClient`` (for the optional,
never-used-here Guardrails Cloud "Guard-as-a-Service" integration), which
constructs both an ``httpx.AsyncClient`` and ``httpx.Client``; each one's
init builds a fresh SSL context (``ssl.create_default_context()`` ->
``load_verify_locations()``), measured at ~1.3-1.6s per ``Guard()`` call on
this machine (confirmed via ``cProfile`` — see log.md). This has nothing to
do with telemetry (there is no supported env var/flag to disable it in
0.10.2; confirmed via live introspection of ``guardrails.settings`` and a
WebSearch hit on guardrails-ai GitHub issue #646, "No capability to disable
telemetry via env var") and nothing to do with validator registry/filesystem
scanning. It fires unconditionally, every time, regardless of
``use_server``.

Since there is no supported flag to suppress this at the root, the fix is
architectural: build each ``Guard`` **once** (memoized per ``on_fail`` action
below) and pass the per-call-varying ``reference``/``fix_value`` at
*validate*-time via Guardrails' ``metadata`` parameter
(``guard.validate(value, metadata={"reference": ..., "fix_value": ...})``)
rather than baking them into a freshly-constructed ``Guard``/``Validator`` on
every call. ``NoLeakValidator._validate`` below reads ``metadata`` first,
falling back to the constructor-time values only if ``metadata`` omits them
(keeps the direct-construction call shape, exercised by
``tests/test_guardrails_no_leak.py``, working unchanged).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

from guardrails import Guard, OnFailAction
from guardrails.validators import FailResult, PassResult, Validator, register_validator

# Tuned per plan.md Section 2b: large enough that a verbatim run is
# unambiguously "copied" (not a coincidental short phrase overlap), small
# enough to trip on a single stolen sentence fragment from a 40-80 word
# ideal_answer.
NGRAM_SIZE = 10

# Tuned per plan.md Section 2b against this project's real reference Q&A
# dataset content: near-verbatim synonym-substituted copies land ~0.78-0.85;
# genuine original paraphrases covering the same key points land ~0.30-0.55.
FUZZY_THRESHOLD = 0.72

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace — the only normalization applied
    before comparison. Deliberately simple/fast (no stemming/punctuation
    stripping) since both checks below are meant to be cheap, deterministic,
    and explainable."""
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _has_verbatim_ngram_overlap(reference_norm: str, value_norm: str, n: int) -> bool:
    """True if any contiguous run of ``n`` words from ``reference_norm``
    appears verbatim inside ``value_norm``. If the reference itself is
    shorter than ``n`` words, falls back to exact whole-reference
    containment (already covered by the caller's full-containment
    fast-path, but kept here too so this helper is correct standalone)."""
    ref_words = reference_norm.split(" ")
    if len(ref_words) <= n:
        return reference_norm in value_norm
    for i in range(len(ref_words) - n + 1):
        window = " ".join(ref_words[i : i + n])
        if window in value_norm:
            return True
    return False


def is_leaking(candidate_text: str, reference_text: str) -> bool:
    """Pure detection function backing :class:`NoLeakValidator`. Exposed at
    module level so a runnable example / quick script can call it directly
    without spinning up a full ``Guard`` (see plan.md Section 9 for the
    latency-measurement use of this)."""
    if not candidate_text or not reference_text:
        return False

    value_norm = _normalize(candidate_text)
    reference_norm = _normalize(reference_text)

    # Fast path: full containment either direction.
    if reference_norm in value_norm or (
        len(value_norm) > 20 and value_norm in reference_norm
    ):
        return True

    # Cheap exact-overlap check before the more expensive fuzzy ratio.
    if _has_verbatim_ngram_overlap(reference_norm, value_norm, NGRAM_SIZE):
        return True

    # Fuzzy near-verbatim check (only reached if the cheap checks passed).
    ratio = SequenceMatcher(None, reference_norm, value_norm).ratio()
    return ratio >= FUZZY_THRESHOLD


@register_validator(name="voice-interview-agent/no-leak", data_type="string")
class NoLeakValidator(Validator):
    """Fails when ``value`` contains the literal or near-verbatim
    ``reference`` text (see module docstring for the two-layer detection
    algorithm).

    ``reference``/``fix_value`` can be supplied two ways:

    1. **Per-instance** (constructor args) — the original shape, still fully
       supported for direct/standalone construction (e.g. in tests).
    2. **Per-call, via ``metadata``** (``guard.validate(value,
       metadata={"reference": ..., "fix_value": ...})``) — the path the real
       call sites use as of iteration 2, so a single long-lived
       ``Validator``/``Guard`` instance can be reused across many calls that
       each check against a *different* question's ``ideal_answer``, instead
       of constructing a brand-new (expensive — see module docstring) ``Guard``
       per call.

    When both are present, ``metadata`` wins — it represents the live,
    per-call-varying reference; the constructor value is only a fallback for
    callers that never pass ``metadata`` at all.
    """

    def __init__(
        self,
        reference: Optional[str] = None,
        fix_value: Optional[str] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, reference=reference, fix_value=fix_value)
        self.reference = reference
        self.fix_value = fix_value

    def _validate(self, value: Any, metadata: dict) -> Any:
        if not isinstance(value, str):
            return PassResult()
        metadata = metadata or {}
        reference = metadata.get("reference", self.reference)
        fix_value = metadata.get("fix_value", self.fix_value)
        if is_leaking(value, reference or ""):
            return FailResult(
                error_message=(
                    "Candidate-facing text contains the literal or "
                    "near-verbatim reference answer."
                ),
                fix_value=fix_value,
            )
        return PassResult()


# Per-call-construction of a ``Guard`` is expensive (~1.3-1.5s, see module
# docstring) and pointless: the only thing that ever varies between calls is
# the ``reference``/``fix_value`` text, both of which now flow through
# ``metadata`` at validate-time (see ``NoLeakValidator._validate`` above).
# What genuinely doesn't vary is the ``on_fail`` policy — both real call
# sites use exactly one fixed policy each (``EXCEPTION`` for the live-turn
# guard, ``FIX`` for the feedback-redaction guard) — so a ``Guard`` built
# with NO constructor-time ``reference``/``fix_value`` (i.e. one meant to be
# driven entirely via ``metadata`` at validate-time, the shape
# :func:`validate_no_leak` uses) is built at most once per distinct
# ``on_fail`` action and cached here, keyed by that action, for the lifetime
# of the process. A caller that supplies an explicit ``reference`` (the
# legacy/direct-construction shape some tests still use) always gets a fresh,
# uncached ``Guard`` bound to that exact reference/fix_value — correctness
# for that less-common shape matters more than caching it, and it was never
# the hot path the per-turn latency fix targets.
_guard_cache: dict[OnFailAction, Guard] = {}


def build_no_leak_guard(
    reference: Optional[str] = None,
    *,
    on_fail: OnFailAction,
    fix_value: Optional[str] = None,
) -> Guard:
    """Factory: returns a ``Guard`` wrapping a single :class:`NoLeakValidator`
    configured with ``on_fail``.

    Two shapes:

    - ``build_no_leak_guard(on_fail=...)`` (no ``reference``) — returns a
      memoized, reusable ``Guard`` with no fixed reference baked in; the
      expensive ``Guard()`` construction happens at most once per ``on_fail``
      action. Callers use ``metadata`` at validate-time to supply the
      per-call ``reference``/``fix_value`` (see :func:`validate_no_leak`).
      This is the shape both real call sites use as of iteration 2.
    - ``build_no_leak_guard(reference, on_fail=..., fix_value=...)`` — the
      original direct-construction shape: always builds a **fresh**,
      uncached ``Guard`` bound to that exact ``reference``/``fix_value``.
      Kept for standalone/test use where a single guard for one fixed
      reference is genuinely what's wanted.
    """
    if reference is None:
        cached = _guard_cache.get(on_fail)
        if cached is not None:
            return cached
        guard = Guard().use(NoLeakValidator(on_fail=on_fail))
        _guard_cache[on_fail] = guard
        return guard
    return Guard().use(
        NoLeakValidator(reference=reference, fix_value=fix_value, on_fail=on_fail)
    )


def validate_no_leak(
    value: str,
    reference: str,
    *,
    on_fail: OnFailAction,
    fix_value: Optional[str] = None,
):
    """Convenience wrapper for the common call-site shape: validate ``value``
    against ``reference`` using a cached, action-keyed ``Guard`` (see
    :func:`build_no_leak_guard`), passing the per-call-varying
    ``reference``/``fix_value`` via Guardrails' ``metadata`` parameter rather
    than constructing a fresh ``Guard``/``Validator``. Returns the same
    ``ValidationOutcome`` ``guard.validate(...)`` would."""
    guard = build_no_leak_guard(on_fail=on_fail)
    return guard.validate(
        value, metadata={"reference": reference, "fix_value": fix_value}
    )


__all__ = [
    "NGRAM_SIZE",
    "FUZZY_THRESHOLD",
    "NoLeakValidator",
    "build_no_leak_guard",
    "validate_no_leak",
    "is_leaking",
]
