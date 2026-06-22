// Voice Interview Agent — frontend state machine.
//
// Explicit states (named in the roadmap, checked by spec.md's acceptance
// criteria): "idle" -> "recording" -> "processing" -> "speaking" -> back to
// "recording" (or "idle"/done at the end of the session). No retry spinners
// or extra states beyond these four, per spec.md's non-goals.
//
// This file preserves the original MediaRecorder/codec logic exactly (see
// pickSupportedMimeType below) and adds presentation: a chat-bubble
// transcript (with the candidate's own transcribed text), an exact
// question-progress bar, richer error banners, and a feedback report
// renderer. Backend API contract:
//   POST /api/sessions                  -> {session_id, interviewer_text, done, question_index, total_questions}
//   POST /api/sessions/{id}/turn        -> {interviewer_text, done, feedback, candidate_text, question_index, total_questions}
//   GET  /api/sessions/{id}/audio       -> streamed raw PCM (16-bit little-
//                                          endian, mono, 24000 Hz) for the
//                                          most recently emitted interviewer
//                                          turn — see
//                                          .claude/loop-state/streaming-tts/spec.md.
//                                          Neither JSON response above
//                                          carries audio anymore; the
//                                          frontend fetches this endpoint
//                                          separately and plays it back
//                                          progressively via Web Audio as
//                                          chunks arrive.

const STATE = {
  IDLE: "idle",
  RECORDING: "recording",
  PROCESSING: "processing",
  SPEAKING: "speaking",
};

const STATE_CAPTIONS = {
  idle: "Ready when you are",
  recording: "Listening… speak your answer",
  processing: "Thinking through your answer…",
  speaking: "Interviewer is speaking…",
};

let currentState = STATE.IDLE;
let sessionId = null;
let mediaRecorder = null;
let recordedChunks = [];

let interviewerTurnCount = 0;

// Streaming TTS playback state (see showInterviewerTurn/playInterviewerAudio
// below). One AudioContext is shared for the whole page lifetime, created
// lazily on first use since browsers require it to be created/resumed after
// a user gesture (the existing start/mic button click provides that).
const AUDIO_SAMPLE_RATE = 24000;
// Scheduling lookahead: the first several chunks of a turn arrive in a tight
// burst (observed live: multiple chunks within a few ms of each other before
// the network settles into its steady ~19ms/chunk cadence). Converting each
// chunk (Int16 -> Float32, AudioBuffer allocation, copyToChannel) costs real
// JS time, so scheduling the very first chunk at `ctx.currentTime` with zero
// margin means later chunks in that same burst can end up being scheduled
// for a `nextStartTime` that's already in the past by the time `.start()`
// actually runs — the browser then starts them immediately, overlapping
// with whatever's still playing, which is exactly the ~1-2s of crackling
// heard at the start of every turn before things "catch up" and settle.
// A small fixed lookahead absorbs that initial burst's processing time.
const SCHEDULING_LOOKAHEAD_S = 0.15;
let audioContext = null;
let nextStartTime = 0;
let lastScheduledSource = null;

const stateVisual = document.getElementById("state-visual");
const stateCaption = document.getElementById("state-caption");
const stateIndicator = document.getElementById("state-indicator");
const stateIndicatorText = document.getElementById("state-indicator-text");
const micButton = document.getElementById("mic-button");
const transcriptEl = document.getElementById("transcript");
const emptyTranscript = document.getElementById("empty-transcript");
const startButton = document.getElementById("start-button");
const recordButton = document.getElementById("record-button");
const errorBanner = document.getElementById("error-banner");
const hintText = document.getElementById("hint-text");
const feedbackSection = document.getElementById("feedback-section");
const perQuestionList = document.getElementById("per-question-list");
const strengthsList = document.getElementById("strengths-list");
const improvementsList = document.getElementById("improvements-list");
const restartButton = document.getElementById("restart-button");
const progressWrap = document.getElementById("progress-wrap");
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");

function setState(next) {
  currentState = next;
  stateVisual.dataset.state = next;
  stateIndicator.dataset.state = next;
  stateIndicatorText.textContent = next.charAt(0).toUpperCase() + next.slice(1);
  stateCaption.textContent = STATE_CAPTIONS[next] || "";

  recordButton.disabled = next === STATE.PROCESSING || next === STATE.SPEAKING;
  startButton.disabled = next !== STATE.IDLE || sessionId !== null;
  micButton.disabled = next === STATE.PROCESSING || next === STATE.SPEAKING;
  micButton.setAttribute(
    "aria-label",
    next === STATE.RECORDING ? "Stop recording" : "Microphone status"
  );
}

function showError(message) {
  errorBanner.hidden = false;
  errorBanner.textContent = message;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function updateProgress(questionIndex, totalQuestions) {
  progressWrap.hidden = false;
  if (typeof totalQuestions === "number" && totalQuestions > 0) {
    const ratio = Math.min(Math.max(questionIndex, 0) / totalQuestions, 1);
    progressFill.style.width = `${Math.max(ratio * 100, 6)}%`;
    progressLabel.textContent = `Question ${Math.min(questionIndex + 1, totalQuestions)} of ${totalQuestions}`;
  } else {
    progressFill.style.width = "6%";
    progressLabel.textContent = "Getting started";
  }
}

function completeProgress() {
  progressFill.style.width = "100%";
  progressLabel.textContent = "Interview complete";
}

function appendBubble(speaker, text) {
  emptyTranscript.hidden = true;
  const row = document.createElement("div");
  row.className = `bubble-row ${speaker}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const label = document.createElement("span");
  label.className = "bubble-speaker";
  label.textContent = speaker === "interviewer" ? "Interviewer" : "You";

  const body = document.createElement("span");
  body.textContent = text;

  bubble.appendChild(label);
  bubble.appendChild(body);
  row.appendChild(bubble);
  transcriptEl.appendChild(row);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function getAudioContext() {
  // Created lazily — never at module load time — because browsers require
  // an AudioContext to be created/resumed after a user gesture. The
  // existing "Start interview"/mic-button click handlers are what
  // eventually call into showInterviewerTurn, so by the time this runs
  // there has always already been a click.
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioContext;
}

function returnToIdleAfterSpeaking() {
  if (currentState === STATE.SPEAKING) {
    setState(STATE.IDLE);
    if (sessionId && recordButton.hidden === false) {
      hintText.textContent = "Press the mic to answer.";
    }
  }
}

// Decodes one raw-PCM chunk (16-bit little-endian, mono, 24000 Hz) into an
// AudioBuffer and schedules it to start right after the previously
// scheduled chunk ends, so consecutive chunks play back-to-back with no
// gap/overlap. Returns the AudioBufferSourceNode it scheduled.
function scheduleAudioChunk(ctx, arrayBuffer) {
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) {
    float32[i] = int16[i] / 32768;
  }

  const audioBuffer = ctx.createBuffer(1, float32.length, AUDIO_SAMPLE_RATE);
  audioBuffer.copyToChannel(float32, 0);

  // Defensive catch-up: if processing/network jitter ever lets nextStartTime
  // fall behind real time (despite the initial lookahead below), scheduling
  // at a past `when` would make the browser start this chunk immediately —
  // overlapping with whatever's still playing and causing the same kind of
  // crackle the lookahead is meant to prevent. A short silent gap is far
  // less jarring than overlapping/distorted audio, so re-anchor to "now"
  // instead of letting the backlog compound.
  if (nextStartTime < ctx.currentTime) {
    nextStartTime = ctx.currentTime;
  }

  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  source.start(nextStartTime);
  nextStartTime += audioBuffer.duration;
  return source;
}

// Fetches the streamed PCM audio for the turn most recently stored
// server-side and plays it back progressively as chunks arrive. Transitions
// to SPEAKING as soon as the fetch begins, and back to IDLE only once BOTH
// the stream has fully finished AND the last scheduled chunk's playback has
// actually finished (not just "bytes stopped arriving") — chunks can still
// be queued/playing after the stream itself has closed.
async function playInterviewerAudio() {
  setState(STATE.SPEAKING);
  hintText.textContent = "";

  try {
    const response = await fetch(`/api/sessions/${sessionId}/audio`);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }

    const ctx = getAudioContext();
    // Start the first chunk slightly in the future rather than exactly "now"
    // — see SCHEDULING_LOOKAHEAD_S's comment above for why a zero-margin
    // start causes audible crackling for the first ~1-2s of every turn.
    nextStartTime = ctx.currentTime + SCHEDULING_LOOKAHEAD_S;
    lastScheduledSource = null;

    const reader = response.body.getReader();
    let chunkCount = 0;
    // HTTP-level chunk boundaries (from the network/fetch reader) don't
    // necessarily line up with the server's per-frame yield boundaries, so
    // a read can in principle end on an odd byte and split a 16-bit sample
    // across two reads. Carry any leftover single byte forward and
    // prepend it to the next read rather than assuming each `value` is
    // itself sample-aligned.
    let leftoverByte = null;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.length === 0) continue;

      let bytes = value;
      if (leftoverByte !== null) {
        const merged = new Uint8Array(leftoverByte.length + bytes.length);
        merged.set(leftoverByte, 0);
        merged.set(bytes, leftoverByte.length);
        bytes = merged;
        leftoverByte = null;
      }
      if (bytes.length % 2 !== 0) {
        leftoverByte = bytes.slice(bytes.length - 1);
        bytes = bytes.slice(0, bytes.length - 1);
      }
      if (bytes.length === 0) continue;

      chunkCount += 1;
      // Copy into a fresh, exactly-sized ArrayBuffer before reinterpreting
      // as Int16Array (the source Uint8Array may be a view over a larger,
      // possibly-pooled buffer with a non-zero byteOffset).
      const chunkBuffer = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength
      );
      const source = scheduleAudioChunk(ctx, chunkBuffer);
      lastScheduledSource = source;
      source.onended = () => {
        if (source === lastScheduledSource) {
          returnToIdleAfterSpeaking();
        }
      };
    }

    if (chunkCount === 0) {
      // Degenerate empty stream: nothing was ever scheduled, so no
      // onended will ever fire. Go straight back to IDLE instead of
      // hanging forever.
      returnToIdleAfterSpeaking();
    }
  } catch (err) {
    showError(describeError(err));
    setState(STATE.IDLE);
  }
}

function showInterviewerTurn(text, questionIndex, totalQuestions) {
  appendBubble("interviewer", text);
  interviewerTurnCount += 1;
  updateProgress(questionIndex, totalQuestions);
  playInterviewerAudio();
}

const VERDICT_LABELS = {
  strong: "Strong",
  weak: "Weak",
  wrong: "Wrong",
  off_topic: "Off topic",
};

function renderFeedback(feedback) {
  if (!feedback) return;

  document.getElementById("stage").hidden = true;
  feedbackSection.hidden = false;

  perQuestionList.innerHTML = "";
  feedback.per_question.forEach((q) => {
    const card = document.createElement("div");
    card.className = "qa-card";

    const head = document.createElement("div");
    head.className = "qa-card-head";

    const topic = document.createElement("span");
    topic.className = "qa-topic";
    topic.textContent = q.topic;

    const badge = document.createElement("span");
    badge.className = `verdict-badge verdict-${q.verdict}`;
    badge.textContent = VERDICT_LABELS[q.verdict] || q.verdict;

    head.appendChild(topic);
    head.appendChild(badge);

    const note = document.createElement("p");
    note.className = "qa-note";
    note.textContent = q.note;

    card.appendChild(head);
    card.appendChild(note);
    perQuestionList.appendChild(card);
  });

  strengthsList.innerHTML = "";
  feedback.overall_strengths.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    strengthsList.appendChild(li);
  });

  improvementsList.innerHTML = "";
  feedback.overall_improvements.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    improvementsList.appendChild(li);
  });

  completeProgress();
}

async function startSession() {
  clearError();
  setState(STATE.PROCESSING);
  hintText.textContent = "";
  try {
    const response = await fetch("/api/sessions", { method: "POST" });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }
    const data = await response.json();
    sessionId = data.session_id;
    startButton.hidden = true;
    recordButton.hidden = false;
    showInterviewerTurn(data.interviewer_text, data.question_index, data.total_questions);
  } catch (err) {
    showError(describeError(err));
    setState(STATE.IDLE);
  }
}

async function sendTurn(blob) {
  setState(STATE.PROCESSING);
  try {
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    const response = await fetch(`/api/sessions/${sessionId}/turn`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }
    const data = await response.json();
    appendBubble("candidate", data.candidate_text || "(no speech detected)");
    showInterviewerTurn(data.interviewer_text, data.question_index, data.total_questions);

    if (data.done) {
      recordButton.hidden = true;
      renderFeedback(data.feedback);
    }
  } catch (err) {
    showError(describeError(err));
    setState(STATE.IDLE);
  }
}

function describeError(err) {
  const message = err && err.message ? err.message : String(err);
  if (/Failed to fetch|NetworkError|network/i.test(message)) {
    return "Couldn't reach the interview server. Check your connection and try again.";
  }
  return message || "Something went wrong. Please try again.";
}

// Codec-pinning rationale (added in the Phase 6 revision after
// litellm-reviewer's review-litellm-01.md finding, written against the
// Deepgram STT provider — see app/gateway/stt.py; the same codec ambiguity
// applies to any STT provider relying on binary magic-byte sniffing as a
// fallback, so the mitigation below is kept regardless of provider):
// different browsers/OSes choose
// different container/codec pairs for MediaRecorder, some of which can be
// harder for a server-side transcription provider to disambiguate from
// binary content alone. So we explicitly pin to ONE specific,
// well-supported combination — audio/webm with the Opus codec — rather
// than the bare "audio/webm" container default (which lets the browser
// choose its own codec inside the container). Firefox does not support
// webm/opus recording, so it falls back to audio/ogg;codecs=opus
// (Firefox's own well-supported pairing). This narrows this app to
// producing exactly one of two well-known, unambiguous binary signatures
// in practice, instead of "whatever the browser/OS defaults to".
function pickSupportedMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus"];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "";
}

async function toggleRecording() {
  if (currentState !== STATE.RECORDING) {
    clearError();

    if (!window.MediaRecorder) {
      showError(
        "Your browser doesn't support audio recording (MediaRecorder API). Try the latest Chrome, Edge, or Firefox."
      );
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
        showError("Microphone access was denied. Please allow microphone access in your browser settings and try again.");
      } else if (err && err.name === "NotFoundError") {
        showError("No microphone was found. Please connect a microphone and try again.");
      } else {
        showError("Microphone access failed: " + (err && err.message ? err.message : err));
      }
      return;
    }

    try {
      const mimeType = pickSupportedMimeType();
      mediaRecorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      recordedChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunks.push(event.data);
      };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, {
          type: mediaRecorder.mimeType || "audio/webm",
        });
        stream.getTracks().forEach((track) => track.stop());
        sendTurn(blob);
      };

      mediaRecorder.start();
      setState(STATE.RECORDING);
      recordButton.textContent = "Stop answer";
      recordButton.classList.add("recording");
      hintText.textContent = "Recording… press again to submit your answer.";
    } catch (err) {
      stream.getTracks().forEach((track) => track.stop());
      showError("Could not start recording: " + (err && err.message ? err.message : err));
    }
  } else {
    mediaRecorder.stop();
    recordButton.textContent = "Start answer";
    recordButton.classList.remove("recording");
    hintText.textContent = "";
  }
}

function restartInterview() {
  sessionId = null;
  interviewerTurnCount = 0;
  recordedChunks = [];
  transcriptEl.innerHTML = "";
  transcriptEl.appendChild(emptyTranscript);
  emptyTranscript.hidden = false;
  progressWrap.hidden = true;
  progressFill.style.width = "6%";
  startButton.hidden = false;
  recordButton.hidden = true;
  feedbackSection.hidden = true;
  document.getElementById("stage").hidden = false;
  clearError();
  setState(STATE.IDLE);
  hintText.textContent = "Make sure your microphone is connected.";
}

startButton.addEventListener("click", startSession);
recordButton.addEventListener("click", toggleRecording);
micButton.addEventListener("click", () => {
  // The orb mirrors record-button behavior once a session is live; before
  // a session starts, clicking it is equivalent to "Start interview" for
  // discoverability (a visitor's natural instinct is to click the mic).
  if (!sessionId) {
    startSession();
  } else if (!recordButton.disabled) {
    toggleRecording();
  }
});
restartButton.addEventListener("click", restartInterview);

setState(STATE.IDLE);

// ---------------------------------------------------------------------------
// Q&A bank management — manual CRUD + document-upload extraction.
//
// Lives entirely in its own overlay/panel, independent of the interview
// state machine above: it does not read or write `currentState`/`sessionId`
// and none of its event listeners replace existing ones, so the interview
// flow (start/record/feedback) is unaffected whether this panel is ever
// opened or not. Errors reuse the same showError/clearError *pattern* as
// the interview UI (a plain text banner, no alert()/confirm()) but target
// the panel's own banner element so an error here never clobbers/clears an
// unrelated interview-side error and vice versa.
//
// Backend contract (see .claude/loop-state/qa-bank-management/spec.md):
//   GET    /api/qa-bank             -> {questions: QAEntryOut[], max_follow_ups_per_question}
//   POST   /api/qa-bank             -> 201 QAEntryOut   (409 id collision, 422 validation)
//   PUT    /api/qa-bank/{id}        -> 200 QAEntryOut   (404 unknown id, 422 validation)
//   DELETE /api/qa-bank/{id}        -> 204               (404 unknown id, 409 last entry)
//   POST   /api/qa-bank/extract     -> 200 {questions: QAEntryDraft[], truncated: bool} (415/502)
//   POST   /api/qa-bank/bulk        -> 201 {questions: QAEntryOut[]}
// ---------------------------------------------------------------------------

const DIFFICULTIES = ["easy", "medium", "hard"];

const manageBankButton = document.getElementById("manage-bank-button");
const bankOverlay = document.getElementById("bank-overlay");
const bankCloseButton = document.getElementById("bank-close-button");
const bankErrorBanner = document.getElementById("bank-error-banner");

const tabManual = document.getElementById("tab-manual");
const tabUpload = document.getElementById("tab-upload");
const panelManual = document.getElementById("panel-manual");
const panelUpload = document.getElementById("panel-upload");

const qaBankList = document.getElementById("qa-bank-list");
const qaBankLoading = document.getElementById("qa-bank-loading");

const manualForm = document.getElementById("manual-form");
const manualFormTitle = document.getElementById("manual-form-title");
const manualFormEditId = document.getElementById("manual-form-edit-id");
const manualTopic = document.getElementById("manual-topic");
const manualDifficulty = document.getElementById("manual-difficulty");
const manualQuestion = document.getElementById("manual-question");
const manualIdealAnswer = document.getElementById("manual-ideal-answer");
const manualKeyPoints = document.getElementById("manual-key-points");
const manualFormCancel = document.getElementById("manual-form-cancel");
const manualFormSubmit = document.getElementById("manual-form-submit");

const uploadForm = document.getElementById("upload-form");
const uploadFile = document.getElementById("upload-file");
const uploadCount = document.getElementById("upload-count");
const uploadSubmit = document.getElementById("upload-submit");
const extractProgress = document.getElementById("extract-progress");
const extractTruncatedNote = document.getElementById("extract-truncated-note");
const draftReviewSection = document.getElementById("draft-review-section");
const draftList = document.getElementById("draft-list");
const commitDraftsButton = document.getElementById("commit-drafts-button");

let qaBankCache = [];
let draftsCache = []; // [{ data: QAEntryDraft, discarded: bool }]
let pendingDeleteId = null;

function showBankError(message) {
  bankErrorBanner.hidden = false;
  bankErrorBanner.textContent = message;
}

function clearBankError() {
  bankErrorBanner.hidden = true;
  bankErrorBanner.textContent = "";
}

async function parseErrorDetail(response) {
  const body = await response.json().catch(() => ({}));
  if (body && typeof body.detail === "string") return body.detail;
  if (body && Array.isArray(body.detail) && body.detail.length > 0) {
    // FastAPI/Pydantic 422 validation errors: a list of {loc, msg, ...}.
    return body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `Request failed (${response.status})`;
}

function truncateText(text, maxLen) {
  if (!text) return "";
  return text.length > maxLen ? `${text.slice(0, maxLen - 1)}…` : text;
}

function openBankPanel() {
  clearBankError();
  bankOverlay.hidden = false;
  document.body.style.overflow = "hidden";
  loadQaBank();
}

function closeBankPanel() {
  bankOverlay.hidden = true;
  document.body.style.overflow = "";
  resetManualForm();
}

function switchBankTab(target) {
  const manualActive = target === "manual";
  tabManual.classList.toggle("active", manualActive);
  tabUpload.classList.toggle("active", !manualActive);
  tabManual.setAttribute("aria-selected", String(manualActive));
  tabUpload.setAttribute("aria-selected", String(!manualActive));
  panelManual.hidden = !manualActive;
  panelUpload.hidden = manualActive;
}

async function loadQaBank() {
  qaBankLoading.hidden = false;
  qaBankLoading.textContent = "Loading questions…";
  try {
    const response = await fetch("/api/qa-bank");
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    const data = await response.json();
    qaBankCache = data.questions || [];
    renderQaBankList();
  } catch (err) {
    qaBankLoading.hidden = false;
    qaBankLoading.textContent = "Could not load the question bank.";
    showBankError(describeError(err));
  }
}

function renderQaBankList() {
  qaBankList.innerHTML = "";
  if (qaBankCache.length === 0) {
    qaBankLoading.hidden = false;
    qaBankLoading.textContent = "No questions yet — add one below.";
    return;
  }
  qaBankLoading.hidden = true;

  qaBankCache.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "qa-bank-row";
    row.dataset.id = entry.id;

    const info = document.createElement("div");
    info.className = "qa-bank-row-info";

    const meta = document.createElement("div");
    meta.className = "qa-bank-row-meta";
    const idSpan = document.createElement("span");
    idSpan.className = "qa-bank-row-id";
    idSpan.textContent = entry.id;
    const topicSpan = document.createElement("span");
    topicSpan.textContent = entry.topic;
    const diffBadge = document.createElement("span");
    diffBadge.className = `difficulty-badge difficulty-${entry.difficulty}`;
    diffBadge.textContent = entry.difficulty;
    meta.appendChild(idSpan);
    meta.appendChild(topicSpan);
    meta.appendChild(diffBadge);

    const questionP = document.createElement("p");
    questionP.className = "qa-bank-row-question";
    questionP.textContent = truncateText(entry.question, 90);

    info.appendChild(meta);
    info.appendChild(questionP);

    const actions = document.createElement("div");
    actions.className = "qa-bank-row-actions";

    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "row-action-btn";
    editBtn.textContent = "Edit";
    editBtn.addEventListener("click", () => startEditEntry(entry));

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "row-action-btn";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", () => handleDeleteClick(entry.id, deleteBtn));

    actions.appendChild(editBtn);
    actions.appendChild(deleteBtn);

    row.appendChild(info);
    row.appendChild(actions);
    qaBankList.appendChild(row);
  });
}

// Delete uses a two-step click confirmation (not a bare browser confirm()):
// first click turns the button into an explicit "Confirm delete?" state;
// a second click within the same render actually deletes. Clicking any
// other delete button, or re-rendering the list, resets pending state.
function handleDeleteClick(id, buttonEl) {
  if (pendingDeleteId === id) {
    pendingDeleteId = null;
    deleteEntry(id);
    return;
  }
  pendingDeleteId = id;
  document.querySelectorAll(".row-action-btn.delete-confirm").forEach((btn) => {
    btn.classList.remove("delete-confirm");
    btn.textContent = "Delete";
  });
  buttonEl.classList.add("delete-confirm");
  buttonEl.textContent = "Confirm delete?";
}

async function deleteEntry(id) {
  clearBankError();
  try {
    const response = await fetch(`/api/qa-bank/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    qaBankCache = qaBankCache.filter((e) => e.id !== id);
    renderQaBankList();
    if (manualFormEditId.value === id) {
      resetManualForm();
    }
  } catch (err) {
    showBankError(describeError(err));
    renderQaBankList();
  }
}

function startEditEntry(entry) {
  manualFormEditId.value = entry.id;
  manualFormTitle.textContent = `Edit ${entry.id}`;
  manualTopic.value = entry.topic;
  manualDifficulty.value = entry.difficulty;
  manualQuestion.value = entry.question;
  manualIdealAnswer.value = entry.ideal_answer;
  manualKeyPoints.value = (entry.key_points || []).join("\n");
  manualFormSubmit.textContent = "Save changes";
  manualFormCancel.hidden = false;
  switchBankTab("manual");
  manualForm.scrollIntoView({ block: "nearest" });
}

function resetManualForm() {
  manualForm.reset();
  manualFormEditId.value = "";
  manualFormTitle.textContent = "Add a question";
  manualFormSubmit.textContent = "Add question";
  manualFormCancel.hidden = true;
  manualDifficulty.value = "medium";
}

function keyPointsFromTextarea(value) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function buildEntryPayload({ id, topic, difficulty, question, idealAnswer, keyPointsRaw }) {
  return {
    id: id || null,
    topic: topic.trim(),
    difficulty,
    question: question.trim(),
    ideal_answer: idealAnswer.trim(),
    key_points: keyPointsFromTextarea(keyPointsRaw),
  };
}

async function submitManualForm(event) {
  event.preventDefault();
  clearBankError();

  const editId = manualFormEditId.value;
  const payload = buildEntryPayload({
    id: editId || null,
    topic: manualTopic.value,
    difficulty: manualDifficulty.value,
    question: manualQuestion.value,
    idealAnswer: manualIdealAnswer.value,
    keyPointsRaw: manualKeyPoints.value,
  });

  manualFormSubmit.disabled = true;
  try {
    const url = editId ? `/api/qa-bank/${encodeURIComponent(editId)}` : "/api/qa-bank";
    const method = editId ? "PUT" : "POST";
    const response = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    const saved = await response.json();
    if (editId) {
      qaBankCache = qaBankCache.map((e) => (e.id === saved.id ? saved : e));
    } else {
      qaBankCache = [...qaBankCache, saved];
    }
    renderQaBankList();
    resetManualForm();
  } catch (err) {
    showBankError(describeError(err));
  } finally {
    manualFormSubmit.disabled = false;
  }
}

async function submitUploadForm(event) {
  event.preventDefault();
  clearBankError();
  extractTruncatedNote.hidden = true;
  draftReviewSection.hidden = true;

  const file = uploadFile.files && uploadFile.files[0];
  if (!file) {
    showBankError("Choose a document to upload first.");
    return;
  }
  const count = Math.max(1, Math.min(10, Number(uploadCount.value) || 5));
  uploadCount.value = String(count);

  const formData = new FormData();
  formData.append("file", file);
  formData.append("count", String(count));

  uploadSubmit.disabled = true;
  extractProgress.hidden = false;
  try {
    const response = await fetch("/api/qa-bank/extract", { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    const data = await response.json();
    draftsCache = (data.questions || []).map((q) => ({ data: { ...q }, discarded: false }));
    extractTruncatedNote.hidden = !data.truncated;
    renderDraftList();
    draftReviewSection.hidden = draftsCache.length === 0;
    if (draftsCache.length === 0) {
      showBankError("No draft questions were generated from this document.");
    }
  } catch (err) {
    showBankError(describeError(err));
  } finally {
    uploadSubmit.disabled = false;
    extractProgress.hidden = true;
  }
}

function renderDraftList() {
  draftList.innerHTML = "";
  draftsCache.forEach((draft, index) => {
    const card = document.createElement("div");
    card.className = "draft-card";
    card.classList.toggle("discarded", draft.discarded);

    const head = document.createElement("div");
    head.className = "draft-card-head";

    const label = document.createElement("label");
    label.className = "draft-card-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = !draft.discarded;
    const labelText = document.createElement("span");
    labelText.textContent = draft.discarded ? "Discarded" : `Keep (${draft.data.id})`;
    checkbox.addEventListener("change", () => {
      draft.discarded = !checkbox.checked;
      card.classList.toggle("discarded", draft.discarded);
      labelText.textContent = draft.discarded ? "Discarded" : `Keep (${draft.data.id})`;
    });
    label.appendChild(checkbox);
    label.appendChild(labelText);
    head.appendChild(label);
    card.appendChild(head);

    const topicRow = document.createElement("div");
    topicRow.className = "form-row";
    const topicLabel = document.createElement("label");
    topicLabel.textContent = "Topic";
    const topicInput = document.createElement("input");
    topicInput.type = "text";
    topicInput.value = draft.data.topic;
    topicInput.addEventListener("input", () => (draft.data.topic = topicInput.value));
    topicRow.appendChild(topicLabel);
    topicRow.appendChild(topicInput);

    const difficultyRow = document.createElement("div");
    difficultyRow.className = "form-row";
    const difficultyLabel = document.createElement("label");
    difficultyLabel.textContent = "Difficulty";
    const difficultySelect = document.createElement("select");
    DIFFICULTIES.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d;
      opt.textContent = d.charAt(0).toUpperCase() + d.slice(1);
      if (d === draft.data.difficulty) opt.selected = true;
      difficultySelect.appendChild(opt);
    });
    difficultySelect.addEventListener("change", () => (draft.data.difficulty = difficultySelect.value));
    difficultyRow.appendChild(difficultyLabel);
    difficultyRow.appendChild(difficultySelect);

    const questionRow = document.createElement("div");
    questionRow.className = "form-row";
    const questionLabel = document.createElement("label");
    questionLabel.textContent = "Question";
    const questionTextarea = document.createElement("textarea");
    questionTextarea.value = draft.data.question;
    questionTextarea.addEventListener("input", () => (draft.data.question = questionTextarea.value));
    questionRow.appendChild(questionLabel);
    questionRow.appendChild(questionTextarea);

    const answerRow = document.createElement("div");
    answerRow.className = "form-row";
    const answerLabel = document.createElement("label");
    answerLabel.textContent = "Ideal answer";
    const answerTextarea = document.createElement("textarea");
    answerTextarea.value = draft.data.ideal_answer;
    answerTextarea.addEventListener("input", () => (draft.data.ideal_answer = answerTextarea.value));
    answerRow.appendChild(answerLabel);
    answerRow.appendChild(answerTextarea);

    const keyPointsRow = document.createElement("div");
    keyPointsRow.className = "form-row";
    const keyPointsLabel = document.createElement("label");
    keyPointsLabel.textContent = "Key points (one per line)";
    const keyPointsTextarea = document.createElement("textarea");
    keyPointsTextarea.value = (draft.data.key_points || []).join("\n");
    keyPointsTextarea.addEventListener("input", () => {
      draft.data.key_points = keyPointsFromTextarea(keyPointsTextarea.value);
    });
    keyPointsRow.appendChild(keyPointsLabel);
    keyPointsRow.appendChild(keyPointsTextarea);

    card.appendChild(topicRow);
    card.appendChild(difficultyRow);
    card.appendChild(questionRow);
    card.appendChild(answerRow);
    card.appendChild(keyPointsRow);

    draftList.appendChild(card);
  });
}

async function commitDrafts() {
  clearBankError();
  const selected = draftsCache.filter((d) => !d.discarded);
  if (selected.length === 0) {
    showBankError("Select at least one draft to add.");
    return;
  }

  const questions = selected.map((d) =>
    buildEntryPayload({
      id: d.data.id || null,
      topic: d.data.topic,
      difficulty: d.data.difficulty,
      question: d.data.question,
      idealAnswer: d.data.ideal_answer,
      keyPointsRaw: (d.data.key_points || []).join("\n"),
    })
  );

  commitDraftsButton.disabled = true;
  try {
    const response = await fetch("/api/qa-bank/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ questions }),
    });
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    const data = await response.json();
    qaBankCache = [...qaBankCache, ...(data.questions || [])];
    draftsCache = [];
    renderDraftList();
    draftReviewSection.hidden = true;
    uploadForm.reset();
    uploadCount.value = "5";
    switchBankTab("manual");
    renderQaBankList();
  } catch (err) {
    showBankError(describeError(err));
  } finally {
    commitDraftsButton.disabled = false;
  }
}

manageBankButton.addEventListener("click", openBankPanel);
bankCloseButton.addEventListener("click", closeBankPanel);
bankOverlay.addEventListener("click", (event) => {
  if (event.target === bankOverlay) closeBankPanel();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !bankOverlay.hidden) closeBankPanel();
});
tabManual.addEventListener("click", () => switchBankTab("manual"));
tabUpload.addEventListener("click", () => switchBankTab("upload"));
manualForm.addEventListener("submit", submitManualForm);
manualFormCancel.addEventListener("click", resetManualForm);
uploadForm.addEventListener("submit", submitUploadForm);
commitDraftsButton.addEventListener("click", commitDrafts);
