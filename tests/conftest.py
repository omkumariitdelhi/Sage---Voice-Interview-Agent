"""Shared pytest fixtures for gateway tests.

All tests run with **zero real API keys** — every fixture either supplies a
dummy non-empty key string (so `GatewayAuthError` does not fire on the happy
path) or deliberately clears keys (to test that `GatewayAuthError` does
fire). No network call is ever made; `litellm.completion`/`transcription`
calls and the TTS gateway's `httpx` client calls are always monkeypatched.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.config import GatewayConfig


@pytest.fixture
def with_dummy_keys(monkeypatch: pytest.MonkeyPatch) -> GatewayConfig:
    """Replace app.config.config with a copy that has dummy (non-real) keys
    set, so happy-path tests aren't blocked by GatewayAuthError."""
    import app.config as config_module

    dummy = dataclasses.replace(
        config_module.config,
        openrouter_api_key="dummy-openrouter-key",
        deepgram_api_key="dummy-deepgram-key",
    )
    monkeypatch.setattr(config_module, "config", dummy)
    # Each gateway module imported `config` by reference at import time
    # (`from app.config import config`), so patch the name in every module
    # that holds its own reference too.
    monkeypatch.setattr("app.gateway.llm.config", dummy)
    monkeypatch.setattr("app.gateway.stt.config", dummy)
    monkeypatch.setattr("app.gateway.tts.config", dummy)
    return dummy


@pytest.fixture
def with_no_keys(monkeypatch: pytest.MonkeyPatch) -> GatewayConfig:
    """Replace app.config.config with a copy that has all keys cleared, to
    exercise the GatewayAuthError fail-fast path."""
    import app.config as config_module

    cleared = dataclasses.replace(
        config_module.config,
        openrouter_api_key=None,
        deepgram_api_key=None,
    )
    monkeypatch.setattr(config_module, "config", cleared)
    monkeypatch.setattr("app.gateway.llm.config", cleared)
    monkeypatch.setattr("app.gateway.stt.config", cleared)
    monkeypatch.setattr("app.gateway.tts.config", cleared)
    return cleared


@pytest.fixture(autouse=True)
def stub_intro_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-seed app.interview.persona's intro cache for every test, pointed
    at an isolated tmp_path file. Without this, the first test to call
    build_graph()/create_app() would hit persona.get_or_generate_intro_text's
    cache-miss branch and call the (mocked) LLM gateway an EXTRA, unplanned
    time — silently shifting other tests' mock call-count/`side_effect`
    sequence assumptions. Pre-seeding means that branch is never reached in
    any test."""
    import app.interview.persona as persona_module

    cache_path = tmp_path / "intro.json"
    cache_path.write_text(
        json.dumps({"text": "Hi, I'm a stubbed test interviewer."}),
        encoding="utf-8",
    )
    monkeypatch.setattr(persona_module, "INTRO_CACHE_PATH", cache_path)
