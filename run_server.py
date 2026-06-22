"""Run the Voice Interview Agent's FastAPI app locally.

    .venv\\Scripts\\python.exe run_server.py

Then open http://127.0.0.1:8000/ in a browser (Chrome/Edge recommended for
MediaRecorder's webm support) and grant microphone access.

Requires real API keys in `.env` (OPENROUTER_API_KEY, DEEPGRAM_API_KEY —
the latter drives both STT and TTS) — see `.env.example`. Without them,
`/api/sessions` will return a 500 ("Server is missing a required API key")
the first time it tries to call the gateway.
"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.web.server:app", host="127.0.0.1", port=8000, reload=False)
