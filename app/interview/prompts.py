"""Prompt builders for every LLM call the interview graph makes.

Each function returns a ``list[dict]`` chat-messages payload ready to pass to
``app.gateway.llm.complete``/``acomplete``. Nothing here calls the gateway
directly — that's :mod:`app.interview.evaluation` and
:mod:`app.interview.feedback`'s job.

**Leak-prevention rule (best-effort, per spec.md):** ``QAEntry.ideal_answer``
must never be interpolated into a prompt whose *output* becomes a
candidate-facing interviewer turn (``ask_question_message``,
``follow_up_message``, ``teach_wrong_answer_message``, ``redirect_message``).
It is only ever passed to ``evaluation_prompt``, whose output (a JSON
verdict) is never shown to the candidate. Full structural enforcement of
"the literal ideal_answer never appears in an interviewer turn" is Phase 5's
job (a guardrail wrapping this graph's output) — this module's job is to not
leak by design in the first place, per spec.md's explicit framing.

**One intentional, scoped exception:** ``recap_and_ask_message`` DOES
interpolate ``prev_entry.ideal_answer`` — by design. It is only ever built
once a question is fully *closed* (the interviewer is moving on to the next
one), so there is no remaining risk of handing the candidate the answer to a
question they could still be asked again this session. This is the one place
in the live interview flow where a full reveal is the intended behavior (the
human-interviewer "here's what I was looking for" recap) rather than a leak
to guard against — see ``app/interview/graph.py``'s ``make_interviewer_turn_node``
for why it is deliberately routed through plain ``_safe_complete``, not
``_safe_complete_no_leak``.
"""

from __future__ import annotations

import json

from app.interview.persona import INTERVIEWER_NAME
from app.interview.state import AnswerVerdict, TurnRecord
from app.retrieval.schema import QAEntry

_SYSTEM_PERSONA = (
    f"You are {INTERVIEWER_NAME}, a professional, friendly technical "
    "interviewer conducting a screening interview. Be concise and "
    "conversational, like a real interviewer speaking out loud — not a "
    "written report. This is a SPOKEN conversation: never use markdown "
    "formatting (no **bold**, no _italics_, no backticks, no bullet/"
    "numbered lists, no headers) — plain natural spoken sentences only."
)


def intro_message(intro_text: str) -> list[dict]:
    """No LLM call needed for the intro — it's read verbatim from the
    dataset's ``intro`` field. Kept as a builder for symmetry/testability and
    in case a future phase wants to LLM-paraphrase it; today it's a pass-through."""
    return [{"role": "assistant", "content": intro_text}]


def closing_message(closing_text: str) -> list[dict]:
    """Pass-through builder, mirrors :func:`intro_message`."""
    return [{"role": "assistant", "content": closing_text}]


def ask_question_message(entry: QAEntry) -> list[dict]:
    """Frame a brand-new question for ``entry``. Deliberately does not
    reference ``ideal_answer`` at all."""
    system = _SYSTEM_PERSONA
    user = (
        f"Ask the candidate the following interview question, in your own "
        f"natural interviewer voice. Do not add hints or extra context — "
        f"just ask it clearly.\n\nQuestion: {entry.question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def evaluation_prompt(
    entry: QAEntry, candidate_answer: str, history: list[TurnRecord]
) -> list[dict]:
    """JSON-mode grading prompt. The ONLY prompt in this module allowed to
    reference ``entry.ideal_answer`` — its output is a verdict object, never
    shown to the candidate verbatim."""
    system = (
        "You are grading a candidate's spoken interview answer against a "
        "reference answer and key points. Respond with STRICT JSON only, "
        "matching this schema exactly:\n"
        '{"verdict": "strong"|"weak"|"wrong"|"off_topic", '
        '"missed_key_points": ["..."], "reasoning": "..."}\n'
        "Use \"strong\" if the answer covers most key points correctly. "
        "Use \"weak\" if it's on-topic but incomplete or partially correct. "
        "Use \"wrong\" if it's on-topic but materially incorrect. "
        "Use \"off_topic\" if it does not address the question at all. "
        "Never include the reference answer text in your response."
    )
    history_lines = "\n".join(
        f"{turn['speaker']}: {turn['text']}" for turn in history[-6:]
    )
    user = (
        f"Question: {entry.question}\n\n"
        f"Reference answer (for grading only, never reveal to candidate): "
        f"{entry.ideal_answer}\n\n"
        f"Key points a strong answer should cover:\n"
        + "\n".join(f"- {kp}" for kp in entry.key_points)
        + f"\n\nRecent conversation:\n{history_lines}\n\n"
        f"Candidate's latest answer: {candidate_answer}\n\n"
        f"Return the JSON verdict now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def follow_up_message(
    entry: QAEntry, candidate_answer: str, missed_key_points: list[str], attempt_number: int
) -> list[dict]:
    """Scaffold/hint prompt for a **weak** (incomplete but on-track) answer.
    Grounded in ``missed_key_points`` plus the candidate's own
    ``candidate_answer`` text (so the follow-up can react to something they
    specifically said, instead of generic advice that could apply to any
    answer) — never interpolates ``ideal_answer``. A genuinely **wrong**
    answer gets the more directive :func:`teach_wrong_answer_message`
    instead — see ``app/interview/graph.py``'s ``make_interviewer_turn_node``
    for the verdict-based branch."""
    system = _SYSTEM_PERSONA + (
        " The candidate's last answer to the current question was "
        "incomplete. Ask a focused follow-up that reacts to something "
        "specific they said and nudges them toward the gap(s) below, as a "
        "hint — do NOT state or paraphrase the full reference answer, do "
        "NOT simply repeat the original question verbatim, and do NOT give "
        "generic advice that could apply to any answer."
    )
    points = "\n".join(f"- {kp}" for kp in missed_key_points) or "- (general gap)"
    user = (
        f"Original question: {entry.question}\n\n"
        f'The candidate said: "{candidate_answer}"\n\n'
        f"Gaps to nudge the candidate toward (do not reveal the answer, "
        f"just hint):\n{points}\n\n"
        f"This is follow-up attempt #{attempt_number} on this question. "
        f"Ask one concise follow-up question now, referencing something "
        f"specific from what they said."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def teach_wrong_answer_message(
    entry: QAEntry, candidate_answer: str, missed_key_points: list[str], attempt_number: int
) -> list[dict]:
    """Teaching follow-up for a genuinely **wrong** (materially incorrect,
    not just incomplete) answer — a real interviewer wouldn't just hint here,
    they'd clarify the misunderstanding **in response to what was actually
    said**. Clearly explains the relevant concept/gap (more directive than
    :func:`follow_up_message`'s gentle nudge), grounded in both
    ``missed_key_points`` and the literal ``candidate_answer``, and
    re-engages the candidate — but, like ``follow_up_message``, still does
    NOT paste the literal ``ideal_answer``: the question is still *open*
    (the candidate has follow-up budget left to apply the explanation), so
    the no-leak guardrail still wraps this prompt's output downstream. The
    full, unguarded reveal is reserved for :func:`recap_and_ask_message`,
    used only once the question is closed."""
    system = _SYSTEM_PERSONA + (
        " The candidate's last answer to the current question was "
        "materially incorrect, not just incomplete. Briefly and kindly "
        "point out specifically what was wrong in what they said, explain "
        "the relevant concept below in your own words, then re-engage them "
        "with a focused question so they can apply it — do NOT simply "
        "repeat the original question verbatim, do NOT state or paraphrase "
        "the full reference answer, and do NOT give a generic explanation "
        "that ignores what they actually said."
    )
    points = "\n".join(f"- {kp}" for kp in missed_key_points) or "- (general gap)"
    user = (
        f"Original question: {entry.question}\n\n"
        f'The candidate said: "{candidate_answer}"\n\n'
        f"The concept(s) the candidate got wrong (explain these clearly, "
        f"don't just hint):\n{points}\n\n"
        f"This is follow-up attempt #{attempt_number} on this question. "
        f"Point out what was specifically wrong in their answer, briefly "
        f"teach the relevant concept, then ask one focused follow-up "
        f"question now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def recap_and_ask_message(
    prev_entry: QAEntry,
    candidate_answer: str,
    missed_key_points: list[str],
    final_verdict: AnswerVerdict,
    next_entry: QAEntry,
) -> list[dict]:
    """Closes out a question that was NOT answered strongly, then asks the
    next one — the live, in-conversation "here's what I was looking for"
    moment a real interviewer gives before moving on. Grounded in the
    candidate's literal last answer on this question (not just the
    abstracted ``missed_key_points``) so the recap responds to what they
    specifically said rather than reading as generic, templated advice. The
    ONE prompt builder in this module allowed to interpolate
    ``prev_entry.ideal_answer`` (see the module docstring's "intentional,
    scoped exception" note) — by the time this is built, ``prev_entry`` is
    closed and cannot be asked again this session, so there is nothing left
    to leak."""
    system = _SYSTEM_PERSONA + (
        " You are wrapping up the previous question before moving to the "
        "next one. In 1-2 encouraging sentences, briefly react to what the "
        "candidate specifically said and summarize what a strong answer "
        "would have additionally covered — be constructive, not harsh, and "
        "avoid generic advice that doesn't reference their actual answer. "
        "Then smoothly transition and ask the next question, in your own "
        "natural interviewer voice."
    )
    points = "\n".join(f"- {kp}" for kp in missed_key_points) or "- (general gaps)"
    user = (
        f"Previous question: {prev_entry.question}\n\n"
        f'The candidate said: "{candidate_answer}"\n\n'
        f"The candidate's final verdict on it was: {final_verdict}\n\n"
        f"What a strong answer would have covered: {prev_entry.ideal_answer}\n\n"
        f"Specific gaps the candidate had:\n{points}\n\n"
        f"Next question to ask: {next_entry.question}\n\n"
        f"Give the brief recap (referencing what they actually said), then "
        f"ask the next question now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def redirect_message(entry: QAEntry, candidate_answer: str) -> list[dict]:
    """Off-topic redirect prompt. Acknowledges what the candidate actually
    said (briefly, naturally) before redirecting them back to the original
    question — does not reference key points (those are for genuine
    weak/wrong answers, not off-topic ones)."""
    system = _SYSTEM_PERSONA + (
        " The candidate's last reply did not address the question at all. "
        "Briefly and naturally acknowledge what they said, then gently "
        "redirect them back to the original question without being curt or "
        "generic."
    )
    user = (
        f'The candidate said: "{candidate_answer}" — which went off-topic. '
        f"Politely acknowledge it, then remind them of the original "
        f"question and ask them to address it:\n\n"
        f"Original question: {entry.question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def feedback_prompt(
    transcript: list[TurnRecord], entries_by_id: dict[str, QAEntry]
) -> list[dict]:
    """Final structured feedback report prompt. JSON-mode, parsed into
    :class:`app.interview.state.FeedbackReport` by
    :mod:`app.interview.feedback`."""
    system = (
        "You are summarizing a completed screening interview into structured "
        "feedback. Respond with STRICT JSON only, matching this schema "
        "exactly:\n"
        '{"per_question": [{"qa_id": "...", "topic": "...", '
        '"verdict": "strong"|"weak"|"wrong"|"off_topic", "note": "..."}], '
        '"overall_strengths": ["..."], "overall_improvements": ["..."]}\n'
        "Include one per_question entry for every question id mentioned in "
        "the transcript. Be specific and constructive."
    )
    def _format_turn(turn: TurnRecord) -> str:
        verdict_suffix = f" (verdict={turn['verdict']})" if turn["verdict"] else ""
        return f"[{turn['qa_id'] or '-'}] {turn['speaker']}{verdict_suffix}: {turn['text']}"

    transcript_text = "\n".join(_format_turn(turn) for turn in transcript)
    topics_text = "\n".join(
        f"- {qid}: topic={entry.topic}" for qid, entry in entries_by_id.items()
    )
    user = (
        f"Full interview transcript:\n{transcript_text}\n\n"
        f"Question topics:\n{topics_text}\n\n"
        f"Produce the JSON feedback report now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _dumps_example() -> str:  # pragma: no cover - debugging helper only
    """Tiny helper kept for local debugging of the expected JSON shape; not
    imported by any node."""
    return json.dumps(
        {"verdict": "weak", "missed_key_points": [], "reasoning": ""}, indent=2
    )
