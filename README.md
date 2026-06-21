# Life Log Search

Local-first multimodal personal search over journals, photos, audio, video, email, calendar exports, and browser history.

## v1.0 Defaults

- Single-user, local-only runtime.
- REST API, CLI, and minimal chat-style web UI.
- FastAPI for the API, APScheduler for lightweight local scheduling, Watchdog for file watching, Qdrant in Docker server mode, and SQLite for metadata.
- Local model suite: `intfloat/e5-large-v2`, OpenCLIP `ViT-L-14`, WhisperX `large-v3`, and `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Windows, macOS, and Linux are supported for development; macOS-only source integrations remain optional.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
docker compose up -d qdrant
.\.venv\Scripts\lifelog doctor
```

Install `ffmpeg` separately and ensure it is on PATH before ingesting audio or video.

See `docs/SETUP.md`, `docs/PRODUCT_SCOPE.md`, and `docs/ARCHITECTURE.md` for the locked decisions behind sections 1-3 of the implementation plan.

