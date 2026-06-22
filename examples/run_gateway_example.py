"""Runnable example: how another module imports and calls the gateway.

Run with: python examples/run_gateway_example.py

This script makes **no real network call** — it stubs out
`litellm.completion` with a stand-in (clearly marked below) so it can run in
any environment with zero API keys, while still exercising the real
import path and call shape that Phase 4 (LangGraph nodes) and Phase 6
(FastAPI routes) will use:

    from app.gateway import complete

If you DO have a real OPENROUTER_API_KEY set in your environment (or `.env`),
delete the stub block below and this will make a live call instead.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Allow running as `python examples/run_gateway_example.py` directly (i.e.
# without the project root already on sys.path / without `-m`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.gateway.llm as llm_module
from app.gateway import complete


def _install_stub_for_demo_only() -> None:
    """Stand-in for a real provider call — NOT how production code should
    work. Replace/remove this when real API keys are available; see the
    module docstring above."""
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Sure — tell me about a time you debugged a "
                    "tricky production issue."
                )
            )
        ]
    )
    llm_module.litellm.completion = MagicMock(return_value=fake_response)

    # The gateway raises GatewayAuthError if no key is configured; set a
    # placeholder so this demo runs standalone with zero real secrets.
    import dataclasses

    import app.config as config_module

    config_module.config = dataclasses.replace(
        config_module.config, openrouter_api_key="demo-placeholder-key"
    )
    llm_module.config = config_module.config


def main() -> None:
    _install_stub_for_demo_only()

    messages = [
        {"role": "system", "content": "You are a calm technical interviewer."},
        {"role": "user", "content": "I'm ready for the next question."},
    ]
    answer = complete(messages)
    print("Interviewer LLM responded:")
    print(answer)


if __name__ == "__main__":
    main()
