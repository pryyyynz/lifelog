# Life Log Search Engine — Technical Specification

> A cross-modal personal search engine over journals, photos, voice memos, video, email, calendar, and browser history. Runs fully local. Queries return ranked, session-grouped results across all modalities.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Ingestion](#ingestion)
   - [Text — Journals, Notes, Email](#text--journals-notes-email)
   - [Photos](#photos)
   - [Voice Memos](#voice-memos)
   - [Video](#video)
   - [Calendar & Activity](#calendar--activity)
   - [Browser & Reading History](#browser--reading-history)
3. [Processing](#processing)
   - [Text Chunking & Embedding](#text-chunking--embedding)
   - [Image Processing Pipeline](#image-processing-pipeline)
   - [Video Processing Pipeline](#video-processing-pipeline)
   - [Audio / Voice Memo Processing](#audio--voice-memo-processing)
   - [Metadata Schema](#metadata-schema)
4. [Retrieval](#retrieval)
   - [Retrieval Strategies](#retrieval-strategies)
   - [Fusion — Combining Modality Results](#fusion--combining-modality-results)
   - [Session Grouping](#session-grouping)
5. [Full Stack](#full-stack)
   - [Vector Store](#vector-store)
   - [Embedding Models (Local)](#embedding-models-local)
   - [Transcription](#transcription)
   - [Orchestration & Serving](#orchestration--serving)
6. [Key Resources](#key-resources)

---

## Architecture Overview

```
Raw sources → Modality-specific processors → Embeddings → Unified vector index
                                           ↘ Metadata → SQLite
Query → Query analysis → Parallel retrieval paths → Fusion + re-rank → Ranked results
```

Two separate stores underpin the system:

- **Vector index** (Qdrant): one collection per modality (`text_chunks`, `image_frames`, `video_frames`, `audio_transcripts`). Each document stores its embedding + a payload containing `chunk_id`, `session_id`, `timestamp_utc`, `source_type`, and `lat/lon`.
- **Metadata store** (SQLite): full `chunks` table with all structured fields. The vector index payload is a subset — the SQLite record is the source of truth for display and filtering.

---

## Ingestion

### Text — Journals, Notes, Email

**Obsidian / Notion**
- Export vault to Markdown. Obsidian vaults are flat directories of `.md` files, watchable with `watchdog` for incremental ingest.
- Resource: [Obsidian export docs](https://help.obsidian.md/import/markdown)

**Gmail / Outlook**
- Google Takeout → MBOX. Parse with `mailbox` or `imaplib`. Filter to sent + received, strip HTML with `trafilatura`.
- Resource: [Google Takeout](https://takeout.google.com)

**Apple Notes / iCloud**
- Export via AppleScript on macOS. No official API — use iCloud Drive path if Notes sync is enabled.

**Day One / Journey**
- Both export to JSON with timestamps. Day One JSON includes location, weather, and tags natively.
- Resource: [Day One export guide](https://dayoneapp.com/guides/tips-and-tutorials/exporting-entries/)

---

### Photos

Highest spatial + temporal metadata density of any modality. CLIP enables caption-free semantic retrieval.

**Apple Photos**
- Export originals via Photos app or `osxphotos` CLI — preserves EXIF, albums, face names, GPS.
- Resource: [osxphotos](https://github.com/RhetTbull/osxphotos)

**Google Photos**
- Google Takeout → JSON sidecars + JPEG. Metadata lives in sidecar JSON files, not EXIF — join by filename.
- Resource: [Google Photos Takeout Helper](https://github.com/TheLastGimbus/GooglePhotosTakeoutHelper)

**Raw filesystem**
- Walk directories with `pathlib`, read EXIF with `exifread` or `Pillow`. GPS stored in rational format — convert to decimal degrees:

```python
def rational_to_decimal(rational_gps, ref):
    d, m, s = [float(x.num) / float(x.den) for x in rational_gps]
    decimal = d + m / 60 + s / 3600
    return -decimal if ref in ['S', 'W'] else decimal
```

---

### Voice Memos

High density of unfiltered thought — often more candid than written notes. Transcription is lossless for retrieval purposes.

**iOS Voice Memos**
- iCloud sync → accessible at `~/Library/Group Containers/.../Media/Recordings/` on macOS. Files in M4A format.

**Transcription options**
- Otter.ai exports JSON with word-level timestamps.
- Whisper locally produces SRT or JSON with `word_timestamps=True`.
- Resource: [OpenAI Whisper](https://github.com/openai/whisper)

**Pre-processing**
- Normalize to 16kHz mono WAV before Whisper — reduces processing time ~30%:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

---

### Video

Most expensive modality. Strategy: sample keyframes + transcribe audio. Full per-frame embedding is unnecessary for personal use.

**Frame sampling**
- Extract keyframes at scene boundaries with `PySceneDetect`, or 1 frame/sec with `ffmpeg` / `PyAV`.
- Apply perceptual hashing (`imagehash`) to skip near-duplicate frames within a scene — threshold delta > 8.
- Embed each unique frame with CLIP.
- Resource: [PySceneDetect](https://www.scenedetect.com), [PyAV](https://pyav.org/docs/stable/), [imagehash](https://github.com/JohannesBuchner/imagehash)

**Audio transcription**
- Extract audio track: `ffmpeg -i video.mp4 -vn audio.wav`
- Transcribe with Whisper, align transcript timestamps to video timeline.
- Chunk transcript by scene boundary — link transcript chunks and frame embeddings to the same `scene_id`.

**YouTube / screen recordings**
- `yt-dlp` downloads with auto-generated subtitles when available — skip transcription if captions exist.
- Resource: [yt-dlp](https://github.com/yt-dlp/yt-dlp)

**Storage schema per video**
- One row per scene: `video_id`, `scene_id`, `start_sec`, `end_sec`, `frame_path`, `transcript_chunk`, two vector IDs (frame embedding + transcript embedding).

---

### Calendar & Activity

Structured temporal anchor — doesn't need embedding. Used as metadata filter and session grouping signal.

**Google Calendar**
- Export ICS or use Google Calendar API. Parse with `icalendar` — events give start/end/location/title.
- Resource: [Google Calendar API](https://developers.google.com/calendar/api)

**Apple Health**
- Export XML from Health app. Contains steps, sleep, workouts, heart rate — timestamped to the minute.
- Resource: [Apple Health XML converter](https://github.com/tdambrin/apple_health_xml_convert)

**Google Location History**
- Google Takeout → `Records.json` GPS trace. Reverse-geocode with `geopy` + Nominatim (no API key required):

```python
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="lifelog")
location = geolocator.reverse(f"{lat}, {lon}")
place_name = location.address
```

- Resource: [geopy](https://geopy.readthedocs.io)

---

### Browser & Reading History

Often underestimated. What you read is a strong proxy for what you were thinking about.

**Chrome history**
- SQLite at `~/.config/google-chrome/Default/History` (Linux) or `~/Library/Application Support/Google/Chrome/Default/History` (macOS).
- Copy before querying — Chrome locks the file while running. Tables: `urls` + `visits`.

```python
import sqlite3, shutil
shutil.copy(chrome_history_path, "/tmp/history_copy")
conn = sqlite3.connect("/tmp/history_copy")
rows = conn.execute("SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC").fetchall()
```

**Readwise**
- API exports highlights + source URLs + timestamps. Best structured reading data available.
- Resource: [Readwise API](https://readwise.io/api_deets)

**Pocket / Instapaper**
- Both have export APIs. Fetch full article text with `trafilatura` using the saved URL.
- Resource: [Pocket API](https://getpocket.com/developer/)

---

## Processing

### Text Chunking & Embedding

**Chunking strategy**
- Do not chunk by token count alone. Use semantic boundaries.
- `paragraph` boundaries for journal entries (natural thought unit).
- One embedding per email — strip signatures, quoted reply chains, and HTML.
- Overlap of 64 tokens between chunks to preserve context at boundaries.
- Use `RecursiveCharacterTextSplitter` from `langchain-text-splitters` with separators `["\n\n", "\n", ". "]`.
- Target 256–512 tokens per chunk.
- Resource: [LangChain text splitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/)

**Embedding model selection**

| Model | Type | Dims | Context | Notes |
|---|---|---|---|---|
| `intfloat/e5-large-v2` | Local | 1024 | 512 tokens | Strong general retrieval, runs on CPU |
| `nomic-ai/nomic-embed-text-v1.5` | Local | 768 | 8192 tokens | Best for long journal entries |
| `text-embedding-3-large` | API | 3072 | 8191 tokens | Highest quality, ~$0.13/1M tokens |
| `voyage-3-large` | API | 1024 | 32000 tokens | Best retrieval benchmarks |

**Critical**: prefix queries with `"query: "` and documents with `"passage: "` for e5 models — omitting this degrades recall ~15%.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/e5-large-v2")

doc_embeddings = model.encode(["passage: " + chunk for chunk in chunks])
query_embedding = model.encode(["query: " + user_query])
```

Resources: [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2), [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

### Image Processing Pipeline

1. Decode EXIF — extract GPS coordinates, timestamp, camera model with `exifread`.
2. Resize to 224×224 (CLIP input size).
3. Embed with `open_clip`: use `ViT-L-14` for best quality, `ViT-B-32` for 4× faster inference.
4. Store 512-dim or 768-dim vector in a dedicated `image_frames` Qdrant collection.
5. Run face detection with `face_recognition` — store detected names as payload metadata.
6. Optionally generate a caption with BLIP-2 or LLaVA — enables hybrid text+vector search on images.

```python
import open_clip, torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-L-14")

img = preprocess(Image.open(path)).unsqueeze(0)
with torch.no_grad():
    image_features = model.encode_image(img)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # normalize
```

Resources: [open_clip](https://github.com/mlfoundations/open_clip), [BLIP-2](https://github.com/salesforce/LAVIS)

---

### Video Processing Pipeline

1. Run `PySceneDetect` to find scene cut boundaries.
2. Extract one representative frame per scene with `ffmpeg`.
3. Compute perceptual hash (`imagehash.phash`) — skip frames where hash delta < 8 (near-duplicate).
4. CLIP-embed each unique frame. Store with `video_id`, `timestamp_sec`, `scene_id`.
5. Extract full audio track: `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.wav`
6. Transcribe with WhisperX — word-level timestamps + optional speaker diarization.
7. Chunk transcript by scene boundary. Embed transcript chunks as text.
8. Both frame embedding and transcript embedding store the same `scene_id` — linked at query time.

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

video = open_video("video.mp4")
manager = SceneManager()
manager.add_detector(ContentDetector(threshold=27.0))
manager.detect_scenes(video)
scenes = manager.get_scene_list()  # list of (start_timecode, end_timecode)
```

Resources: [PySceneDetect](https://www.scenedetect.com/docs/), [WhisperX](https://github.com/m-bain/whisperX), [imagehash](https://github.com/JohannesBuchner/imagehash), [yt-dlp](https://github.com/yt-dlp/yt-dlp)

---

### Audio / Voice Memo Processing

**Transcription**
- `whisper-large-v3` for best quality; `whisper-turbo` for 8× speed with minor quality tradeoff.
- `word_timestamps=True` for word-level timing — enables timestamp-aligned chunking.
- For multi-speaker audio (calls, interviews): add `pyannote.audio` for speaker diarization, then align labels to Whisper output by timestamp.

```python
import whisper
model = whisper.load_model("large-v3")
result = model.transcribe("audio.wav", word_timestamps=True)
# result["segments"] → list of {start, end, text, words: [{word, start, end}]}
```

**Storage**
- Raw audio file path
- Full transcript text
- Transcript chunked by paragraph or 30-second windows
- Each chunk embedded and stored with `timestamp_start`, `timestamp_end`, `speaker_id` (if diarized)

Resources: [Whisper](https://github.com/openai/whisper), [WhisperX](https://github.com/m-bain/whisperX), [whisper.cpp](https://github.com/ggerganov/whisper.cpp), [pyannote.audio](https://github.com/pyannote/pyannote-audio)

---

### Metadata Schema

SQLite `chunks` table — source of truth for display, filtering, and session grouping:

```sql
CREATE TABLE chunks (
    id              TEXT PRIMARY KEY,
    vector_id       TEXT,           -- foreign key into Qdrant collection
    source_type     TEXT,           -- text|image|audio|video_frame|video_transcript
    file_path       TEXT,           -- absolute path to original file
    timestamp_utc   INTEGER,        -- unix epoch seconds
    duration_sec    REAL,           -- for audio/video chunks; NULL for text/image
    lat             REAL,           -- from EXIF or location history; NULL if unknown
    lon             REAL,
    place_name      TEXT,           -- reverse-geocoded; NULL if unknown
    session_id      TEXT,           -- events within a 4-hour window share a session_id
    raw_text        TEXT            -- original content for result display
);

CREATE INDEX idx_timestamp ON chunks(timestamp_utc);
CREATE INDEX idx_session   ON chunks(session_id);
CREATE INDEX idx_type      ON chunks(source_type);
```

**Session ID generation**: cluster events by a 4-hour sliding window. All chunks whose `timestamp_utc` falls within 4 hours of a seed event share a `session_id`. Computed at ingest time, not query time.

---

## Retrieval

### Retrieval Strategies

**Dense ANN — semantic text**
- Embed query with the same model used for documents (e5-large: prefix `"query: "`).
- Cosine similarity search in Qdrant against `text_chunks` collection.
- Best for: concept-based queries, emotional language, abstract topics.
- Resource: [Qdrant search docs](https://qdrant.tech/documentation/concepts/search/)

**CLIP cross-modal — text → image/video**
- Embed text query with CLIP text encoder. Search `image_frames` / `video_frames` collection.
- No captions required. Synthesize a visual-friendly query variant for better recall:
  - Raw query: `"felt isolated while working in Lisbon"`
  - Visual variant: `"quiet cafe afternoon Lisbon city"` → passed to CLIP
- Resource: [open_clip](https://github.com/mlfoundations/open_clip)

**Metadata pre-filter**
- Extract structured signals from the query with a fast LLM call or regex+NER: date range, place name, person name.
- Pass as Qdrant payload filter — runs before ANN, restricts the candidate set.
- Dramatically improves precision for temporal and spatial queries.
- Resource: [Qdrant filtering](https://qdrant.tech/documentation/concepts/filtering/)

**Hybrid BM25 + dense**
- BM25 sparse index (via `rank_bm25` or Qdrant sparse vectors) catches exact name/term matches that dense models miss.
- RRF combines sparse + dense scores in a single Qdrant query.
- Best for: person names, specific place names, technical terms, titles.
- Resource: [Qdrant hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/)

**Cross-encoder re-rank**
- After ANN retrieves top-40 candidates, run a cross-encoder that jointly attends to query + document.
- Produces much more accurate relevance scores than bi-encoder ANN alone.
- Run only on top-40 — too slow to apply to the full collection.
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Resource: [SBERT cross-encoder models](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)

**Temporal re-rank**
- Apply exponential decay boost to results whose timestamp falls near an extracted temporal hint.
- Formula: `final_score = base_score × (1 + α × exp(−d / τ))`
  - `d` = absolute distance in days from the target date
  - `τ` = decay radius in days (e.g., 7 for a weekly window)
  - `α` = 0.5 (max boost cap — a perfect temporal match boosts score by 50%)

```python
import math

def temporal_boost(score: float, result_ts: int, target_ts: int,
                   tau_days: float = 7.0, alpha: float = 0.5) -> float:
    d = abs(result_ts - target_ts) / 86400  # convert seconds to days
    decay = math.exp(-d / tau_days)
    return score * (1 + alpha * decay)
```

---

### Fusion — Combining Modality Results

**Reciprocal Rank Fusion (RRF)** — recommended

Rank-based combiner — immune to score scale differences between modalities. A document ranked #1 in images and #5 in text scores higher than one ranked #3 in both. No per-modality normalization required.

```
score(d) = Σ  1 / (k + rank(d, list))
```

where `k = 60` is a smoothing constant. Natively supported in Qdrant as a query fusion type.

```python
def rrf(result_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))
```

Resource: [RRF original paper](https://arxiv.org/abs/2009.10855), [RRF in Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/#reciprocal-rank-fusion)

**Weighted linear combination** — alternative

Normalize scores per modality (min-max), then combine with dynamic weights:

```
final = w_text × s_text + w_img × s_img + w_meta × s_meta
```

Weights are query-dependent:
- High `w_img` (up to 0.8) when query contains visual keywords: `"saw"`, `"photo"`, `"sunset"`, `"looked like"`.
- High `w_text` (up to 0.9) for emotional, abstract, or time-specific queries.
- `w_meta = 1.0` when a structured filter field matches exactly (e.g., GPS cluster hit).

```python
def normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    return [(s - lo) / (hi - lo + 1e-9) for s in scores]

VISUAL_KEYWORDS = {"photo", "picture", "sunset", "saw", "looked", "view", "scene", "face"}

def clip_weight(query: str) -> float:
    tokens = set(query.lower().split())
    overlap = len(VISUAL_KEYWORDS & tokens)
    return min(0.8, 0.2 + overlap * 0.15)
```

Resource: [Hybrid search intro (Pinecone)](https://www.pinecone.io/learn/hybrid-search-intro/)

---

### Session Grouping

After retrieval and re-ranking, collapse results from the same time window into a single result card.

- Group by `session_id` — computed at ingest time (4-hour sliding window).
- Each result card surfaces: primary modality result (highest-ranked) + N secondary hits from the same session.
- Example: a "July 14, Lisbon" card shows a journal entry (text), 3 afternoon photos (images), and a voice memo (audio) — one card, one moment in time.

```python
from itertools import groupby

def group_by_session(ranked_chunks: list[dict]) -> list[dict]:
    seen_sessions = {}
    grouped = []
    for chunk in ranked_chunks:
        sid = chunk["session_id"]
        if sid not in seen_sessions:
            seen_sessions[sid] = len(grouped)
            grouped.append({"primary": chunk, "secondary": [], "session_id": sid})
        else:
            grouped[seen_sessions[sid]]["secondary"].append(chunk)
    return grouped
```

---

## Full Stack

### Vector Store

| Option | Type | Hybrid search | Notes |
|---|---|---|---|
| **Qdrant** | Server / embedded | Native (sparse + dense + RRF) | Recommended. Rust core, Docker or embedded mode. |
| Chroma | Embedded | No native sparse | Good for prototyping. Simpler API. |
| LanceDB | Embedded | Partial | No Docker required. Native multimodal support. |
| pgvector | PostgreSQL ext. | Via separate BM25 | Good if you're already on Postgres. |

Resources: [Qdrant quickstart](https://qdrant.tech/documentation/quick-start/), [Chroma docs](https://docs.trychroma.com), [LanceDB docs](https://lancedb.github.io/lancedb/)

---

### Embedding Models (Local)

| Tool | Purpose | Notes |
|---|---|---|
| `sentence-transformers` | Text embeddings | Wraps HuggingFace models. Use for e5, nomic, BGE. |
| `open_clip` | Image + cross-modal | Best CLIP implementation. ViT-L-14, ViT-H-14. CPU/GPU. |
| `Ollama` | Local embed API | Run `nomic-embed-text` with zero Python setup via REST API. |

Resources: [sentence-transformers](https://www.sbert.net), [open_clip](https://github.com/mlfoundations/open_clip), [Ollama embed models](https://ollama.com/library/nomic-embed-text)

---

### Transcription

| Tool | Speed | Quality | Notes |
|---|---|---|---|
| `openai/whisper` | Baseline | Best | Use `large-v3`. Reference implementation. |
| `WhisperX` | ~4× faster | Near-identical | Batched inference, word timestamps, built-in diarization. Recommended. |
| `whisper.cpp` | Fast on Apple Silicon | Near-identical | C++ port, runs via Metal GPU. No Python required. |

Resources: [Whisper](https://github.com/openai/whisper), [WhisperX](https://github.com/m-bain/whisperX), [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

---

### Orchestration & Serving

| Tool | Purpose | Notes |
|---|---|---|
| **FastAPI** | Query endpoint | Async, typed. Run behind local Nginx for HTTPS. |
| **Prefect** / APScheduler | Incremental ingest scheduling | Prefect has a free local UI for pipeline monitoring. |
| **Haystack 2.0** | Full RAG framework | Pre-built components for chunking, embedding, retrieval, fusion. Native Qdrant support. |
| **Watchdog** | File system watcher | Trigger ingest on new files in watched directories. |

Resources: [FastAPI](https://fastapi.tiangolo.com), [Prefect](https://docs.prefect.io), [Haystack](https://docs.haystack.deepset.ai)

---

## Key Resources

| Resource | Link |
|---|---|
| MTEB leaderboard — compare all text embedding models | https://huggingface.co/spaces/mteb/leaderboard |
| CLIP paper (Radford et al.) | https://arxiv.org/abs/2212.09561 |
| ColPali — page-level multimodal retrieval | https://arxiv.org/abs/2205.00823 |
| FAISS tutorial — ANN search deep dive | https://www.pinecone.io/learn/series/faiss/ |
| RRF original paper | https://arxiv.org/abs/2009.10855 |
| Retrieve + re-rank patterns (SBERT) | https://sbert.net/examples/applications/retrieve_rerank/README.html |
| Qdrant documentation | https://qdrant.tech/documentation/ |
| open_clip GitHub | https://github.com/mlfoundations/open_clip |
| WhisperX GitHub | https://github.com/m-bain/whisperX |
| PySceneDetect docs | https://www.scenedetect.com/docs/ |
