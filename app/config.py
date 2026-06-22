"""Environment-driven configuration for the LiteLLM gateway.

Loads everything from environment variables only (via ``python-dotenv`` for
local development convenience). No secrets are ever hard-coded here. Importing
this module never raises, even with zero keys set — auth errors are surfaced
lazily, at call time, by the gateway modules (``app/gateway/*.py``), so test
suites and tooling can import the package safely with no API keys present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Loads `.env` if present; silently does nothing if the file is missing
# (e.g. in CI or test environments that inject env vars another way).
load_dotenv()

# Default STT model line: Deepgram Nova-2. Confirmed via the installed
# litellm==1.89.2 package (model_prices_and_context_window_backup.json lists
# "deepgram/nova-2" with mode="audio_transcription",
# litellm_provider="deepgram"; get_llm_provider("deepgram/nova-2") resolves to
# custom_llm_provider="deepgram"). Swappable via DEEPGRAM_MODEL with zero code
# changes.
DEFAULT_DEEPGRAM_MODEL = "deepgram/nova-2"

# Default TTS model line: Deepgram Aura-2 "Thalia" (clear, natural English
# voice; the canonical example voice in Deepgram's own current docs). Model
# id format is "[modelname]-[voicename]-[language]" per
# developers.deepgram.com/docs/tts-models (verified via WebFetch 2026-06-21).
# NOT routed through litellm.speech() -- confirmed via direct introspection
# that litellm==1.89.2 has no Deepgram audio_speech provider config (see
# app/gateway/tts.py module docstring) -- this is the bare model id passed as
# the Deepgram REST API's `model` query parameter. Swappable via
# DEEPGRAM_TTS_MODEL with zero code changes.
DEFAULT_DEEPGRAM_TTS_MODEL = "aura-2-thalia-en"

# Free-tier default (no cost) -- gpt-oss-120b chosen for reliable
# instruction-following/JSON-mode output, which this app's grading/feedback
# calls depend on. Verified current via web search 2026-06-21. Swappable via
# OPENROUTER_MODEL with zero code changes if rate-limited or for a paid model.
DEFAULT_OPENROUTER_MODEL = "openrouter/openai/gpt-oss-120b:free"


@dataclass(frozen=True)
class GatewayConfig:
    """Holds every env-derived setting the gateway needs.

    API keys may be ``None`` in this dataclass (e.g. in a dev/test
    environment with no real keys present) — the gateway modules check for
    a missing key at call time and raise ``GatewayAuthError`` rather than
    failing at import/construction time.
    """

    openrouter_api_key: str | None
    deepgram_api_key: str | None

    openrouter_model: str
    deepgram_model: str
    deepgram_tts_model: str

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Build a config snapshot from current environment variables."""
        return cls(
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
            deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY") or None,
            openrouter_model=os.environ.get(
                "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL
            ),
            deepgram_model=os.environ.get("DEEPGRAM_MODEL", DEFAULT_DEEPGRAM_MODEL),
            deepgram_tts_model=os.environ.get(
                "DEEPGRAM_TTS_MODEL", DEFAULT_DEEPGRAM_TTS_MODEL
            ),
        )


# Module-level singleton so callers can simply do `from app.config import config`.
# Tests may monkeypatch attributes on this instance (it's a frozen dataclass,
# so tests instead monkeypatch the *module-level name* `app.config.config`,
# see tests/conftest.py) to simulate missing/present keys without touching
# real environment variables.
config = GatewayConfig.from_env()
