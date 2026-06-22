"""LiteLLM-backed model gateway: the only place this app calls LiteLLM.

Re-exports the public call functions for convenience:

    from app.gateway import complete, acomplete, transcribe, atranscribe, synthesize, asynthesize

Every other module in the app (LangGraph nodes, FastAPI routes, etc.) should
import from here rather than calling ``litellm`` directly — that keeps "no
inline litellm calls outside app/gateway/" true and grep-checkable.
"""

from app.gateway.llm import acomplete, complete
from app.gateway.stt import atranscribe, transcribe
from app.gateway.tts import asynthesize, synthesize

__all__ = [
    "complete",
    "acomplete",
    "transcribe",
    "atranscribe",
    "synthesize",
    "asynthesize",
]
