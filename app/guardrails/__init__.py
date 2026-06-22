"""Phase 5 output-safety guardrails: structural, tested enforcement (via
Guardrails AI) of two properties of the interviewer that were previously only
prompt-level conventions (see ``app/interview/prompts.py``'s own
"best-effort, per spec.md" docstring note):

1. No candidate-facing interviewer text (live turns AND feedback notes) ever
   contains the literal/near-verbatim reference ``ideal_answer`` for the
   question it concerns (:mod:`app.guardrails.no_leak`).
2. The end-of-interview feedback is always returned in the exact
   ``FeedbackReport`` shape (:mod:`app.guardrails.feedback_guard`).

See ``.claude/loop-state/phase-5-guardrails/plan.md`` for the full design
rationale, threshold justification, and on-fail policy per call site.
"""

from __future__ import annotations
