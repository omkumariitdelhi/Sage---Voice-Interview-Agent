# Architecture Note — Voice Interview Agent

A candidate speaks into the browser; audio is transcribed (Deepgram Nova-2),
a LangGraph state machine grounds each turn in a reference Q&A store
(LangChain + local embeddings), an LLM (OpenRouter, via a LiteLLM gateway)
plays the interviewer, Guardrails AI enforces the two non-negotiable safety
properties, and Deepgram (Aura-2) speaks the reply back. This note covers
the three things the assignment asks for: retrieval design, how the
interviewer stays grounded *and* natural, and where the pipeline's latency
actually goes.

## Retrieval design

**Storage.** `data/qa_bank.yaml` is the single source of truth — 10
software-engineer-screening questions, each with an `ideal_answer`, a list of
`key_points`, a `topic`, and a `difficulty`. A standalone ingest script
(`python -m app.retrieval.ingest`) embeds each entry's **question text** (not
the answer) with a local `sentence-transformers/all-MiniLM-L6-v2` model and
writes it to a persisted Chroma collection, with the rest of the entry
(`ideal_answer`, `key_points`, `topic`, `difficulty`) riding along as document
metadata. Re-running the ingest script after editing the YAML rebuilds the
collection from scratch — no application code anywhere references a question
by name or index, which is exactly the assignment's "editable without code
changes" requirement; this is the one property we tested most aggressively
(Phase 3's `RETR-03`: edit a temp dataset, re-ingest, confirm the change is
retrievable, with zero touches to `app/retrieval/`).

**Chunking.** Deliberately none. Chunking exists to let retrieval return a
relevant *slice* of a long document; here, one Q&A entry — a question plus a
few sentences of reference material — already *is* the right retrieval unit.
Splitting it further would only fragment the metadata (which `ideal_answer`
goes with which `key_points`?) for no benefit at this corpus size.

**Matching.** The store supports two query modes against the same collection:
a **deterministic** lookup by id or by sequence position (`get_by_id`,
`get_by_sequence_index`), and a **semantic similarity** search over question
text (`retrieve(query, k=1)`, cosine similarity via Chroma). The live interview
graph uses only the deterministic path — the interview's question order is
fixed by the dataset's own list order, so the graph always knows exactly which
entry it needs next, with zero ambiguity and zero embedding cost at request
time. The semantic path is fully built and tested (it correctly returns the
right entry for both literal text and independent paraphrases of all 10
questions) but isn't invoked by today's strictly-sequential flow; it's the
mechanism that would back a future adaptive or candidate-driven question order,
or grounding a candidate's own free-form question back to a reference, without
changing the storage layer at all.

**Why local embeddings.** OpenRouter (this project's LLM provider) has no
embeddings endpoint, and a 10-entry corpus doesn't need a hosted model's
marginal quality gain. A local model removes a network round-trip from the one
place embeddings ever run (ingest time, and the unused-today semantic-query
path) and removes an API-key dependency for retrieval entirely.

**Beyond hand-editing the YAML.** The bank no longer requires a text editor
at all: a browser UI (the "Manage questions" panel) now supports direct
CRUD on `data/qa_bank.yaml` (`GET/POST/PUT/DELETE /api/qa-bank...`) and, more
interestingly, generating *new* candidate entries by uploading a document
(PDF/Markdown/text) and having the LLM draft grounded Q&A pairs from it
(`POST /api/qa-bank/extract`), which the user reviews and selectively commits
(`POST /api/qa-bank/bulk`) before the same ingest-and-reload path picks them
up. Both routes still go through the identical write-then-reingest mechanism
described above, so the "editable without code changes" property holds
exactly as tested — this just upgrades the assignment's bar from "edit a
YAML file" to "use the product itself."

## Keeping the LLM behaving like an interviewer while staying grounded

The interview is a **LangGraph state machine** (`greet → ask → listen →
evaluate → follow-up-or-advance → wrap-up → feedback`), not one long prompt
asking the model to track the whole conversation itself. Explicit state
(`qa_index`, `follow_up_count`, the running transcript) is what lets the
interviewer reliably know which question it's on and how many chances it's
already given — a single-prompt design would have to re-derive that from
chat history every turn, with no guarantee it gets it right.

Every LLM call that asks a question, asks a follow-up, or grades an answer is
given the retrieved entry's `ideal_answer` and `key_points` as context — the
model's job is phrasing and judgement, not recalling facts from its own
training data. The evaluator classifies each answer as `strong` / `weak` /
`wrong` / `off_topic` **against the key points**, never against the full ideal
answer text; a weak or wrong verdict drives a follow-up prompt that's told
*which* key points were missed and instructed to hint and scaffold toward
them, never to recite the reference. Off-topic answers are redirected back to
the current question instead of being graded at all.

**Staying bounded.** Every question carries a `max_follow_ups_per_question`
budget (read from the dataset, not hard-coded), shared across weak/wrong/
off-topic verdicts. Once it's exhausted, the graph force-advances regardless
of verdict — proven by a test that scripts every single answer as wrong
forever and asserts the session still reaches feedback in a provably bounded
number of turns.

**A named persona, and "weak" vs. "wrong" treated differently.** The
interviewer is named Sage and introduces itself at the start of every
session — that introduction is the one piece of interviewer text that should
stay consistent rather than vary creatively each time, so it's generated via
one LLM call ever and cached to disk (`app/interview/persona.py`); every
session after the first reads it back with zero API cost. The
fail-soft pattern matches the rest of the gateway layer: if the cache is
empty and the LLM call fails (no key configured yet), the server falls back
to the dataset's raw `intro` text rather than crashing startup, and simply
tries again next time. A `weak` (incomplete but on-track) answer still gets
the original gentle hint; a `wrong` (materially incorrect) answer now gets a
more direct *teaching* follow-up that explains the relevant concept before
re-engaging — closer to how a real interviewer corrects a misunderstanding
than a generic nudge. Both stay behind the no-leak guard described below,
since the question is still open.

**A fourth measured fix, found live while validating the teaching behavior
above: `response_format={"type": "json_object"}` is a silent no-op for
Claude on the installed LiteLLM.** Every JSON-mode call site
(`evaluation.py`, `feedback.py`, `extractor.py`) asks the model to "respond
with STRICT JSON only" and relies on `response_format` as a backstop — but
introspecting `litellm.llms.anthropic.chat.transformation.AnthropicConfig`
shows its tool-call-based JSON enforcement only engages when an explicit
`json_schema`/`response_schema` key is present; the bare `{"type":
"json_object"}` every call site actually sends resolves to `None` and
enforces nothing. Caught live: Claude graded a wrong answer correctly (right
verdict, right reasoning) but wrapped the JSON in a ` ```json ` fence —
*even on the existing single re-ask-on-malformed-JSON retry*, which
explicitly asks for "no markdown fences." `json.loads()` doesn't parse a
fenced string, so this silently degraded a correct "wrong" verdict to the
generic "weak" fallback, masking the new teaching behavior entirely. Fixed
once, in a shared `app/json_mode.py::strip_json_fences()` used by all three
`_parse()` call sites, rather than three inconsistent patches — the same
"one chokepoint, not N call sites" shape as the gateway fixes above.

**Not leaking the answer — except once, on purpose, when it's safe to.** The
hardest constraint, since "guide them toward a better answer" and "never read
out the reference" pull in opposite directions, is enforced **twice**: once at the prompt level (explicit
instructions to hint, not recite), and once **structurally**, via a Guardrails
AI validator that checks every candidate-facing line for exact-substring *and*
fuzzy (`difflib` ratio) overlap with that question's `ideal_answer`, with a
bounded one-time re-ask and a safe templated fallback if it still fails. The
two-layer design isn't theoretical caution: the adversarial review process for
this exact guard reproduced a real exploit — a forced internal error let one
code path return a literal leaked answer completely unredacted — that the
prompt layer alone would never have caught. The same check runs again on the
end-of-interview feedback notes, since "what you could improve" is a second,
easy-to-miss place an ungrounded model could paste the reference verbatim.

The one deliberate exception: when the graph moves on to the *next* question
after a question closes with a non-strong verdict, the turn that asks it
also gives a brief, encouraging recap of what a strong answer to the
*previous* question would have covered — the literal `ideal_answer`, by
design, routed through plain text generation rather than the no-leak guard
(`app/interview/prompts.py::recap_and_ask_message`). This is safe precisely
because the question is closed and cannot be asked again this session — the
guard's whole purpose (don't let the candidate get the answer to a question
they could still be asked) no longer applies once that's true.

## Latency: where the time goes, and how to reduce it

A full turn, today: browser records → upload → **STT** (Deepgram Nova-2, one
call) → graph resume (**two sequential LLM calls** — grade the answer, then
phrase the next turn or follow-up) → Guardrails check → **TTS** (Deepgram
Aura-2, one call) → audio returned → playback. The two LLM calls and the two
voice-provider calls are the entire latency budget; everything else is
effectively free, which is a deliberate design property, not an accident.
STT and TTS now both run on Deepgram (Nova-2 and Aura-2 respectively), sharing
one account/key — there *is* a shared-rate-limit surface to be aware of (the
same trade-off this project briefly had during an earlier ElevenLabs-for-both
arrangement, now reintroduced on Deepgram instead), even though the two calls
remain independent gateway calls:

- **Retrieval adds nothing at request time.** The live path never embeds at
  request time (deterministic lookup only); the unused semantic path, when
  invoked, runs a local model with no network call either.
- **Guardrails is a real, measured example of finding and fixing a hidden
  cost.** The first implementation built a fresh `Guard()` object on every
  call, which cost **~1.3–1.5 seconds per call** — traced to an unconditional,
  unused Guardrails Cloud REST client constructing a fresh SSL context inside
  `Guard.__init__`, unrelated to the actual validation logic. Caching one
  `Guard` per failure-policy and passing the per-call reference text through
  Guardrails' `metadata` parameter instead of rebuilding it dropped this to
  **~5–7ms per call** — roughly a 250x reduction, now negligible next to a
  model call. This is the single most concrete latency lesson from building
  this system: a correctness-and-safety layer can silently dominate the
  latency budget if you don't measure it, independent of how fast its actual
  logic is.
- **Gateway resilience (LiteLLM retries) costs nothing on the happy path** —
  retries only run on an actual transient failure — and the structured
  per-call logging (`call_type`/`model`/`latency_ms`/`outcome`) that produced
  every number in this section is itself sub-millisecond overhead.
- **A second real, measured resilience lesson, found the same way (live
  load, not mocks):** the free-tier `openrouter/openai/gpt-oss-120b:free`
  model occasionally returns `choices[0].message.content = None` instead of
  text — a provider-side hiccup, not a malformed response. Unchecked, that
  `None` reached `json.loads(None)` in the feedback/evaluation path and
  raised an uncaught `TypeError`, crashing the request instead of degrading
  gracefully. Fixed once, at the same gateway chokepoint every LLM call
  already passes through (`app/gateway/llm.py`'s new `_require_content()`),
  by raising the same retryable transient-failure type the retry loop already
  handles — so an empty completion now retries and falls back exactly like a
  rate limit or timeout, with zero changes to any caller. Re-running a full
  live interview against the same flaky free-tier model afterward confirmed
  it: the session completed successfully, surviving another empty-content
  hiccup mid-session. Same theme as the `Guard()` story above: real failure
  modes only show up under live load, and one well-placed chokepoint fix beats
  patching every call site.
- **A third measured fix: TTS was paying a fresh TLS handshake on every
  single turn.** `app/gateway/tts.py` (a hand-rolled `httpx` client, since
  the installed LiteLLM has no native Deepgram speech provider) originally
  opened a brand-new `httpx.Client()` inside every call and closed it
  immediately after — a full DNS+TCP+TLS handshake to `api.deepgram.com`
  every turn, even though STT's `litellm.transcription()` call already gets
  this for free (LiteLLM caches/reuses its own HTTP clients internally).
  Fixed by reusing one process-lifetime client with `keepalive_expiry=120s`
  (httpx's 5s default would already be dead by the time the candidate
  finishes recording the next answer — turns are naturally tens of seconds
  apart). Measured live: a cold call averaged **2.275s**; warm calls
  (including one taken after a deliberate 4s pause, to prove the longer
  keepalive actually survives realistic turn spacing) averaged **1.167s** —
  a **~49% reduction** per TTS call.
- **A fifth fix that changes *what* the app waits for, not just how it
  connects: TTS is now genuinely streamed, not buffered-then-sent.**
  Deepgram's real-time TTS is a WebSocket protocol
  (`wss://api.deepgram.com/v1/speak`), confirmed live (not docs-only) to use
  the same `Authorization: Token <key>` scheme as the REST endpoint, and to
  only support raw PCM-family encodings (`linear16`/`mulaw`/`alaw`) — never
  mp3. That turned out to be convenient rather than limiting: raw PCM needs
  no container/frame-boundary-aware decoding, so the browser can feed each
  chunk straight into the Web Audio API as it lands, no MediaSource
  Extensions complexity required. Implementation required splitting what
  was one HTTP response into two: the turn endpoint now returns text
  immediately (no TTS wait at all) and stashes it server-side, and a new
  `GET /api/sessions/{id}/audio` streams the corresponding audio
  progressively; the browser starts playing chunks via `AudioBufferSourceNode`s
  scheduled back-to-back the moment they arrive. Falls back transparently to
  the one-shot REST path (now requesting the *same* `linear16`/24kHz format,
  via a small additive `encoding`/`sample_rate` parameter on `synthesize()` —
  not a rewrite) if the WebSocket handshake itself fails before any chunk
  streams; a failure after streaming has already started to the browser is
  deliberately not retried, since there's nothing sane to fall back to
  mid-response. Measured live against a real ~60-word reply: first audio
  byte arrived at **2.06s**, with the full 714-chunk/1.37MB stream still
  trickling in continuously through **15.5s** — i.e. the candidate now hears
  Sage start speaking ~13s sooner than they would have waited for the whole
  clip under the old buffer-then-send REST path.
- **Streaming introduced its own bug: ~1-2s of audible crackling at the
  start of every turn, gone once playback "settled."** Root cause: the
  browser-side scheduler (`app/web/static/app.js`) anchored the first
  chunk's playback time at `audioContext.currentTime` with zero margin. The
  first several chunks of every turn arrive in a tight burst (observed live:
  multiple chunks within milliseconds of each other, before the connection
  settles into its steady ~19ms/chunk cadence), and converting each one
  (Int16 → Float32, `AudioBuffer` allocation, `copyToChannel`) costs real JS
  time — enough that, with zero scheduling margin, a later chunk in that
  same burst could end up scheduled for a `nextStartTime` already in the
  past by the time `.start()` actually ran. The Web Audio API responds to a
  past start time by playing immediately, overlapping whatever was still
  playing — exactly the crackling heard early in each turn, self-resolving
  once steady-state chunk spacing gave the scheduler enough headroom to keep
  up. Fixed with a small (150ms) initial scheduling lookahead plus a
  defensive catch-up guard (`scheduleAudioChunk` re-anchors to
  `audioContext.currentTime` if it ever detects `nextStartTime` has fallen
  behind, trading a brief silent gap for what would otherwise be another
  overlap) for any future jitter beyond the initial burst, not just at
  turn-start.

That leaves the real remaining cost: **two LLM round-trips and one STT call
per turn** to OpenRouter/Deepgram respectively, each commonly 300ms–2s
depending on model and text length (TTS's *first-byte* latency is now
addressed above; total TTS generation time itself is unchanged — streaming
moves the wait, it doesn't shrink it). Concrete reduction levers, in
expected order of impact:

1. **Collapse the two LLM calls into one.** A single structured-output call
   could return the verdict, missed key points, *and* the next turn's
   phrasing together, cutting one full model round-trip per turn. Kept
   separate in this build to keep grading and phrasing independently
   testable; the node interfaces don't need to change to merge them later.
2. **True streaming STT** (partial transcripts while the candidate is still
   speaking) would let the grading LLM call begin before they finish talking
   — the single biggest latency win available, and explicitly scoped out of
   this prototype (tracked as a v2 item) in favor of push-to-talk, which is
   simpler to get right end-to-end first.
3. **A smaller/faster OpenRouter model for the per-turn grading call**
   specifically — it needs far less reasoning depth than the final feedback
   report. The model string is an environment variable for exactly this
   reason: the quality/latency trade-off is tunable without a code change.
