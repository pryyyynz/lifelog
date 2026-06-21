# Setup

## Runtime Requirements

- Python 3.11 or newer. The current local workspace was initialized with Python 3.12.
- Windows, macOS, and Linux are supported for development. macOS-only source integrations are optional.
- Minimum RAM: 16 GB.
- Recommended VRAM: 8 GB for GPU inference.
- CPU-only mode is supported but audio/video transcription can be slow, especially with large Whisper models.
- Disk space depends on the raw corpus. Derived vector index storage is estimated at about 2 KB per chunk, and the full local model suite is expected to use about 10 GB.

## Local Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

## External Tools

Install `ffmpeg` system-wide and ensure it is available on PATH:

- Windows: use Chocolatey, Winget, or the official ffmpeg builds.
- macOS: `brew install ffmpeg`
- Linux: install through the distro package manager, such as `apt install ffmpeg`.

If system-wide installation is not available, place a portable `ffmpeg.exe` under the workspace and set `LIFELOG_FFMPEG_PATH` to its absolute path. This workspace uses a portable FFmpeg 7 shared build under `tools/ffmpeg` so command-line media extraction and Python audio libraries can use the same local binary path.

On Windows, `pyannote.audio` may warn that `torchcodec` native decoding is unavailable even when FFmpeg is present. The ingest pipeline should normalize audio through FFmpeg first and pass preloaded waveforms to pyannote when diarization is enabled.

Start Qdrant with Docker Compose:

```powershell
docker compose up -d qdrant
```

Run the startup check:

```powershell
.\.venv\Scripts\lifelog doctor
```

## Models

The default local model suite is:

- Text embeddings: `intfloat/e5-large-v2`
- Image and cross-modal embeddings: OpenCLIP `ViT-L-14` with `openai` weights
- Audio transcription: WhisperX using `large-v3`
- Cross-encoder re-rank: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Download model weights on first run:

```powershell
.\.venv\Scripts\python scripts\download_models.py
```

Preview the download plan without fetching weights:

```powershell
.\.venv\Scripts\python scripts\download_models.py --dry-run
```

Set `LIFELOG_OFFLINE_MODE=true` to disable features that would call network services after setup. Reverse geocoding uses `geopy` plus Nominatim by default and must be disabled for fully offline operation.
