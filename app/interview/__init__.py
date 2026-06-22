"""Phase 4: the LangGraph interview state machine.

Public entry point: :func:`app.interview.graph.build_graph`. Everything else
in this package (``state``, ``prompts``, ``evaluation``, ``feedback``) is
implementation detail consumed by ``graph.py``.

This package never imports ``litellm`` directly (all model calls go through
:mod:`app.gateway.llm`) and never parses the Q&A reference dataset's YAML
file directly (all dataset access goes through :mod:`app.retrieval.store` /
:mod:`app.retrieval.loader`) — both are grep-checkable invariants carried
over from Phase 2/Phase 3.
"""

from __future__ import annotations
