# Life Log Search Engine - Detailed Implementation Todo List

This document converts the technical specification and SRS into an execution checklist for building v1.0 of the system. It combines objectives, use cases, functional requirements, non-functional requirements, constraints, dependencies, and out-of-scope boundaries into one working plan.

Use this as the build tracker for a local-first, privacy-first, multimodal personal search engine.

---

## 1. Product Scope, Success Criteria, and Boundaries

### 1.1 Lock v1.0 scope

- [x] Confirm the v1.0 target is a single-user, local-only system.
- [x] Confirm the core surfaces for v1.0 are REST API, CLI, and minimal chat-style web UI.
- [x] Confirm the supported modalities for v1.0: text, email, photos, audio, video, calendar, browser history.
- [x] Confirm that post-setup operation must not require third-party APIs.
- [x] Confirm that multi-device sync, cloud deployment, mobile apps, live streaming ingest, and social media ingestion remain out of scope.

### 1.2 Define measurable success criteria

- [x] Define acceptance criteria for OBJ-01 unified retrieval: one query can return grouped results spanning at least 3 modalities.
- [x] Define acceptance criteria for OBJ-02 privacy-first execution: all query-time inference and storage remain local.
- [x] Define acceptance criteria for OBJ-03 fusion: at least one end-to-end test returns a single session card containing multimodal results.
- [x] Define acceptance criteria for OBJ-04 temporal and spatial awareness: date/place hints affect filtering and ranking.
- [x] Define acceptance criteria for OBJ-05 incremental ingest: only changed or new data is processed on incremental runs.
- [x] Define acceptance criteria for OBJ-06 conversational refinement: follow-up queries can reuse prior result context.
- [x] Define acceptance criteria for OBJ-07 explainability: each result shows source, timestamp, path, and retrieval rationale.
- [x] Define acceptance criteria for OBJ-08 extensibility: a new modality can be added through one ingestor implementation without retrieval rewrites.

### 1.3 Write explicit non-goals into the plan

- [x] Add a visible "Not in v1.0" section to project docs so scope does not drift.
- [x] Exclude fine-tuning and personalization-by-training from the roadmap.
- [x] Exclude in-UI media playback and route users to native apps for opening source files.
- [x] Exclude authentication and shared multi-user indexing.

---

## 2. Architecture Decisions and Project Skeleton

### 2.1 Finalize high-level architecture

- [x] Document the final system flow: source connectors -> processors -> embeddings -> vector index plus SQLite metadata -> query analysis -> retrieval -> fusion -> re-rank -> session grouping -> result rendering.
- [x] Define clear boundaries between ingestion, indexing, retrieval, ranking, and presentation layers.
- [x] Decide whether Qdrant runs embedded, via local binary, or via Docker.
- [x] Define one internal data contract for a normalized chunk record shared across all modalities.
- [x] Define one internal result contract for retrieval results before and after grouping.

### 2.2 Create repository structure

- [x] Create folders for `app/api`, `app/cli`, `app/ingest`, `app/retrieval`, `app/ranking`, `app/storage`, `app/models`, `app/ui`, `tests`, `scripts`, and `docs`.
- [x] Add a config layer that supports local paths, enabled modalities, model selection, and scheduler settings.
- [x] Add environment-specific settings for development, CPU-only, and GPU-enabled execution.
- [x] Add a logging and observability package layout for ingest logs, query logs, and health checks.

### 2.3 Choose implementation framework options

Pick one option per major area:

- [x] API framework: FastAPI preferred, with Flask only if a lighter interface is required.
- [x] Task orchestration: Prefect for richer flows and visibility, APScheduler for lightweight scheduled jobs.
- [x] File watching: Watchdog preferred, polling fallback only if file watcher support is unreliable.
- [x] Vector store: Qdrant preferred, Chroma for quick prototyping, LanceDB for embedded workflows, pgvector only if Postgres is already required.

---

## 3. Environment, Dependencies, and Local Runtime Setup

### 3.1 Define baseline runtime requirements

- [x] Standardize on Python 3.11+.
- [x] Decide whether local development will support Windows in addition to the spec's primary macOS and supported Linux targets.
- [x] Document minimum RAM (16GB), recommended VRAM (8GB for GPU inference), and expected CPU-only limitations.
- [x] Document disk space expectations: raw corpus (user-dependent), derived index (~2KB per chunk), model weights (~10GB total).
- [x] Create `pyproject.toml` with project metadata and dependency groups: `core`, `dev`, `optional`.
- [x] Add a `.env.example` file documenting all configurable environment variables.

### 3.2 Install and verify external tools

- [x] Install `ffmpeg` system-wide or as a configured local portable binary and verify audio/video extraction:
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
  - Windows: download from ffmpeg.org or use chocolatey
- [x] Install and verify Qdrant in the chosen mode (Docker, embedded, or local binary).
- [x] Install core Python packages: `qdrant-client`, `sentence-transformers`, `open_clip_torch`, `torch`, `Pillow`, `exifread`, `langchain-text-splitters`, `rank_bm25`, `watchdog`, `geopy`, `icalendar`, `trafilatura`.
- [x] Install optional Python packages supported on this Windows setup: `pyannote.audio`, `pillow-heif`, `yt-dlp`, `PySceneDetect`, `PyAV`.
- [ ] Decide whether to install `face_recognition` with `dlib` native build support on Windows.
- [ ] Install `osxphotos` on macOS only.
- [x] Write a `scripts/download_models.py` script that downloads all configured models to a local `models/` directory on first run.
- [x] Verify total disk footprint ≤ ~10GB for the full model suite.
- [x] Define a reproducible setup path for fresh machines.

### 3.3 Choose model/tool options

Pick one option per capability, with optional fallbacks:

- [x] Text embeddings (choose one):
  - **Option A:** `intfloat/e5-large-v2` — strong general retrieval, runs on CPU, 1024-dim *(recommended)*
  - **Option B:** `nomic-ai/nomic-embed-text-v1.5` — 8192-token context, best for long journal entries, 768-dim
  - **Option C (API, non-local):** `text-embedding-3-large` (OpenAI) — highest quality, ~$0.13/1M tokens, 3072-dim
  - **Option D (API, non-local):** `voyage-3-large` — best retrieval benchmarks, 32K context, 1024-dim
- [x] Image and cross-modal embeddings (choose one):
  - **Option A:** `ViT-L-14` via `open_clip` — best quality, ~900MB, 768-dim *(recommended)*
  - **Option B:** `ViT-B-32` via `open_clip` — 4× faster inference, smaller, 512-dim
- [x] Audio transcription (choose one):
  - **Option A:** `openai-whisper` with `large-v3` — reference implementation, best quality, ~3GB
  - **Option B:** `WhisperX` — 4× faster batched inference, word timestamps, optional diarization *(recommended)*
  - **Option C:** `whisper.cpp` — best for Apple Silicon Metal GPU, C++ port, no Python overhead
  - **Option D:** `whisper-turbo` — 8× faster with minor quality tradeoff
- [x] Cross-encoder re-rank: `cross-encoder/ms-marco-MiniLM-L-6-v2` or another small local cross-encoder.
- [x] Reverse geocoding: `geopy` plus Nominatim (no API key required), with an option to disable for fully offline operation.

### 3.4 Build setup verification checklist

- [x] Run a startup check that confirms database path, vector store availability, model availability, and `ffmpeg` presence.
- [x] Add one CLI command to validate the environment before first ingest.
- [x] Add one smoke test that creates a small local index and executes a sample query.

---

## 4. Data Source Onboarding and Source Registry

### 4.1 Build source registry and config model

- [x] Define a source registry that stores source type, path, enabled status, last scan time, and ingest strategy.
- [x] Support path-based local sources and export-based sources separately.
- [x] Add validation rules for missing directories, locked files, unsupported formats, and permission failures.

### 4.2 Implement onboarding flow for UC-01

- [x] Add a `lifelog init` CLI flow that asks which sources to connect.
- [x] Validate user-provided paths and estimate item counts before ingest.
- [x] Show source-specific setup guidance for Takeout exports, Obsidian vaults, Apple Photos exports, and browser history copies.
- [x] Persist chosen configuration in a local config file.
- [x] Add a post-init checklist that prompts the user to run first full ingest.

### 4.3 Source-specific onboarding tasks

- [x] Text: support Markdown, plain text, and journal export directories.
- [x] Email: support MBOX import first, with IMAP sync optional later.
- [x] Photos: support filesystem folders, Apple Photos export, and Google Photos Takeout sidecars.
- [x] Audio: support M4A, MP3, and WAV source discovery.
- [x] Video: support MP4 and MOV discovery plus optional YouTube download workflows.
- [x] Calendar: support ICS import first, API sync optional later.
- [x] Browser history: support copied Chrome history SQLite databases and exported reading/highlight feeds.

---

## 5. Core Ingestion Framework and Incremental Processing

### 5.1 Build a shared ingestor contract

- [x] Define a `BaseIngestor` interface to satisfy extensibility requirement NFR-E-01.
- [x] Standardize methods for `discover`, `extract`, `normalize`, `chunk`, `embed`, `persist`, and `cleanup`.
- [x] Ensure each ingestor can emit chunk records plus source-level ingest metadata.
- [x] Ensure each ingestor can delete and replace all chunks for a modified file.

### 5.2 Implement ingest execution model

- [x] Add `lifelog ingest --full` for initial ingest.
- [x] Add `lifelog ingest --incremental` for nightly or manual incremental runs.
- [x] Add checkpointing so interrupted ingest can resume.
- [x] Log errors per item and continue processing to satisfy NFR-R-01.
- [x] Track run duration, processed item counts, skipped items, and failures.

### 5.3 Implement change detection

- [x] Track `last_ingest_timestamp` globally and per source.
- [x] Detect new or modified files by `mtime` where available.
- [x] For mutable sources like journals, re-embed all chunks belonging to a changed file and remove stale vectors.
- [x] Prevent duplicate embeddings by enforcing file-path plus chunk identity uniqueness.
- [x] Decide whether deletion detection is handled by explicit cleanup commands, watcher-based detection, or periodic reconciliation. Decision: v1 starts with explicit cleanup commands and later periodic reconciliation; watcher events only trigger incremental scans.

### 5.4 Scheduler and automation options

- [x] Scheduler choice: APScheduler for simple nightly jobs or Prefect for richer local flow control and monitoring. Decision: APScheduler for v1.
- [x] File watcher choice: Watchdog for near-real-time source detection or scheduled scans if watchers are unstable across platforms. Decision: keep Watchdog configured, with scheduled scans as the fallback.
- [x] Add reminder behavior carefully: clarify whether "remind users to upload data" is a CLI/UI notification or a scheduled local alert, because the SRS mentions it but the overall system is otherwise local and batch-based. Decision: CLI/UI status notification for v1, not OS-level alerts.

---

## 6. Text, Journal, Note, and Email Pipeline

### 6.1 Text source extraction

- [x] Implement `ObsidianIngestor` class inheriting from `BaseIngestor`.
- [x] Accept a configurable vault root directory path from `config.yaml`. Implemented through the source registry config file.
- [x] Recursively walk directories with `pathlib.Path.rglob("*.md")`.
- [x] Parse frontmatter (YAML between `---` delimiters) to extract `date`, `tags`, `aliases`.
- [x] Strip Obsidian wiki-link syntax (`[[link]]`) and embed syntax (`![[image]]`) before embedding.
- [x] Extract file `mtime` as the fallback timestamp if no frontmatter date is present.
- [x] Implement `NotionIngestor` — accepts Notion export directory (Markdown + CSV format).
- [x] Handle Notion's page hierarchy (nested folders = nested pages) — flatten to top-level chunks.
- [ ] `[OPTIONAL]` Implement `AppleNotesIngestor` using AppleScript export (macOS only).
- [ ] `[OPTIONAL]` Handle iCloud Notes path: `~/Library/Group Containers/.../NoteStore.sqlite`.
- [x] Implement `DayOneIngestor` — parses Day One JSON export format.
- [x] Extract per-entry fields: `creationDate`, `text`, `location` (lat/lon), `weather`, `tags`, `photos` (linked by UUID).
- [x] Implement `JourneyIngestor` — parses Journey JSON export format.

### 6.2 Email ingestion

- [x] Implement `EmailIngestor` supporting MBOX format (from Google Takeout).
- [x] Use `mailbox.mbox` (stdlib) to parse the MBOX file.
- [x] For each email, extract: `sender`, `recipient`, `date`, `subject`, `body`.
- [x] Strip HTML from email bodies (choose one):
  - **Option A:** `trafilatura.extract()` — best at main content extraction *(recommended)*
  - **Option B:** `BeautifulSoup` with `get_text()` — simpler, less noise filtering
- [x] Filter to sent + received only; skip auto-replies, newsletters (heuristic: sender domain ≠ known newsletter domains).
- [x] Strip quoted reply chains — detect `>` prefixed lines or `On [date], [person] wrote:` patterns.
- [x] Strip email signatures — detect 3+ line blocks after `--` or `Regards,` / `Best,` patterns.
- [x] Embed one chunk per email (not sub-chunked, per spec).

### 6.3 Text chunking and embedding

- [x] Implement semantic chunking using paragraph boundaries rather than raw token count only.
- [x] Use `RecursiveCharacterTextSplitter` or an equivalent splitter with paragraph-first separators.
- [x] Target 256 to 512 token chunks with 64-token overlap where applicable.
- [x] Implement mandatory e5 prefixes if e5 is chosen: `passage:` for documents and `query:` for queries.
- [x] Add tests that confirm prefixes are never omitted when using e5 models.

### 6.4 Text-specific retrieval readiness

- [x] Preserve raw text for display snippets in SQLite.
- [x] Store exact-term searchable fields for BM25 or equivalent sparse retrieval.
- [x] Add person-name-sensitive exact matching for use cases like Kofi/person-centric queries.

---

## 7. Photo and Image Processing Pipeline

### 7.1 Image discovery and normalization

- [x] Support JPEG, PNG, and HEIC ingestion.
- [x] Decide whether HEIC is handled with `pillow-heif`, ImageMagick conversion, or explicit pre-conversion instructions. Decision: register `pillow-heif` when installed; otherwise continue with recoverable metadata-only ingest.
- [x] Normalize image loading failures into recoverable ingest errors.

### 7.2 Metadata extraction

- [x] Implement `ExifExtractor` utility — shared by all photo ingestors.
- [x] Extract GPS coordinates using `exifread` or `Pillow`.
- [x] Convert GPS rational format to decimal degrees: `d + m/60 + s/3600`, negate for S/W.
- [x] Extract `DateTimeOriginal` as the canonical photo timestamp.
- [x] Extract `Model` (camera model) and store as metadata.
- [x] Handle missing EXIF gracefully — `lat`, `lon`, `timestamp` become `NULL` in SQLite.
- [ ] Implement `ApplePhotosIngestor` using `osxphotos` CLI.
- [ ] Run `osxphotos export --exiftool --directory <output>` to export originals with EXIF preserved.
- [ ] Preserve album names, face names, and GPS from `osxphotos` metadata.
- [ ] Warn user to close Photos.app before running (SQLite lock constraint).
- [x] Implement `GooglePhotosIngestor` — accepts Google Takeout export directory.
- [x] Match each `.jpg`/`.jpeg` to its `.json` sidecar file (same filename + `.json`).
- [x] Prefer sidecar JSON metadata over EXIF (Google strips EXIF on upload).
- [x] Implement `FilesystemPhotoIngestor` — walks any directory for image files.
- [x] Support extensions: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.webp`, `.tiff`.
- [x] For HEIC/HEIF: use `pillow-heif` plugin or convert via ImageMagick before processing.
- [x] Fall back to file `mtime` if EXIF timestamp is missing.

### 7.3 Image embedding and enrichment

- [x] Resize each photo to 224×224 before CLIP embedding.
- [x] Embed with `open_clip` — chosen model (ViT-L-14 or ViT-B-32). Implemented as disabled-by-default optional adapter.
- [x] Normalize embeddings: divide by L2 norm.
- [ ] Store embedding vector in `image_frames` Qdrant collection.
- [x] Store payload: `chunk_id`, `session_id`, `timestamp_utc`, `source_type=image`, `lat`, `lon`, `file_path`. Stored in SQLite chunk metadata until Qdrant wiring lands in section 11.
- [ ] `[OPTIONAL]` Run face detection with `face_recognition` — store detected face names in payload `detected_faces` field.
- [ ] `[OPTIONAL]` Generate image captions with BLIP-2 or LLaVA — enables hybrid text+vector search on images.

### 7.4 Tool options for image enrichment

- [x] Face recognition option: `face_recognition` or disabled by default for privacy and complexity reasons. Decision: disabled by default.
- [x] Captioning option: BLIP-2, LLaVA, or no captions in v1.0. Decision: no captions in v1.0.
- [x] Apple Photos export option: `osxphotos` or manual export workflow. Decision: manual export workflow first; `osxphotos` remains a later optional integration.

---

## 8. Audio and Voice Memo Pipeline

### 8.1 Audio normalization and preprocessing

- [x] Implement `VoiceMemoIngestor`.
- [ ] Locate iOS Voice Memos at: `~/Library/Group Containers/.../Media/Recordings/` (macOS iCloud sync path).
- [x] Support `.m4a` format (default iOS recording format).
- [x] Support M4A, MP3, and WAV discovery.
- [x] Pre-process audio before transcription: normalize to 16kHz mono WAV using ffmpeg.
- [ ] Document that this step reduces Whisper processing time by ~30%.
- [x] Store raw file path and conversion output mappings for reproducibility.

### 8.2 Transcription pipeline

- [x] Transcribe audio with chosen Whisper variant (matches model download choice from §3.3).
- [x] Enable `word_timestamps=True` — required for timestamp-aligned chunking.
- [ ] `[OPTIONAL]` Enable speaker diarization for multi-speaker recordings (calls, interviews) using `pyannote.audio`.
- [x] Align speaker labels to Whisper transcript by timestamp overlap.
- [x] Add handling for long audio batches, retries, and partial transcript recovery.
- [ ] `[OPTIONAL]` Implement `OtterIngestor` — parses Otter.ai JSON exports with word-level timestamps.

### 8.3 Audio chunking and storage

- [x] Chunk transcripts by paragraph, silence, or fixed windows.
- [x] Store transcript text, timestamp start, timestamp end, and optional speaker ID.
- [x] Embed transcript chunks into the text-capable retrieval path.
- [x] Ensure UI results can open audio at the correct timestamp.

### 8.4 Performance and quality tasks

- [ ] Benchmark transcription throughput on CPU-only and GPU-capable machines.
- [x] Define the default model by hardware tier.
- [ ] Add a user-facing warning when a chosen model will make ingest impractically slow.

---

## 9. Video Processing Pipeline

### 9.1 Video discovery and extraction

- [x] Implement `VideoIngestor`.
- [x] Support MP4 and MOV video ingestion.
- [x] Extract audio tracks via ffmpeg command: `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.wav`.
- [x] Define storage locations for extracted frames and audio artifacts.

### 9.2 Scene detection and frame sampling

- [x] Run `PySceneDetect` with `ContentDetector` (threshold 27.0) to find scene cut boundaries.
- [x] Extract one representative frame per scene with `ffmpeg` or `PyAV`.
- [x] Compute perceptual hash per frame with `imagehash.phash`.
- [x] Skip near-duplicate frames within a scene (hash delta < 8 threshold).
- [x] CLIP-embed each unique frame (same pipeline as photos).
- [x] Store frame embedding in `video_frames` Qdrant collection with payload: `video_id`, `timestamp_sec`, `scene_id`.
- [x] Store scene start and end timestamps.

### 9.3 Video transcript linkage

- [x] Transcribe video audio using the selected transcription engine.
- [x] Chunk transcript by scene boundaries.
- [x] Link scene frame embeddings and scene transcript embeddings with shared `scene_id` metadata.
- [x] Ensure the system can open the original video at scene start when the user clicks "Open original".

### 9.4 Optional video source choices

- [ ] `[OPTIONAL]` Implement `YouTubeIngestor` using `yt-dlp`.
- [ ] `[OPTIONAL]` Download auto-generated subtitles with `yt-dlp --write-auto-sub` — skip transcription if captions already exist.
- [ ] `[OPTIONAL]` Store video URL as `file_path` in metadata for YouTube sources.

---

## 10. Calendar, Activity, Location, and Browser History Ingestion

### 10.1 Calendar and activity data

- [x] Implement `GoogleCalendarIngestor`.
- [x] Accept ICS export file (from Google Takeout) or connect via Google Calendar API.
- [x] Parse with `icalendar` library: extract `DTSTART`, `DTEND`, `SUMMARY`, `LOCATION`, `DESCRIPTION`.
- [x] Store events as structured metadata only (no embedding needed — used as filter and session anchor).
- [x] Do NOT embed calendar events — they are structured temporal anchors, not semantic search targets.
- [ ] `[OPTIONAL]` Implement `AppleHealthIngestor`.
- [ ] `[OPTIONAL]` Parse Apple Health XML export (from Health app → Settings → Export Health Data).
- [ ] `[OPTIONAL]` Extract: steps, sleep intervals, workouts (type, start, end, distance), heart rate readings.
- [ ] `[OPTIONAL]` Use as session context enrichment — label sessions with dominant activity (e.g., "walked 8km", "slept 7h").

### 10.2 Location enrichment

- [ ] `[OPTIONAL]` Implement `GoogleLocationIngestor`.
- [ ] `[OPTIONAL]` Parse `Records.json` from Google Takeout (GPS trace, one record per minute).
- [x] Reverse-geocode sampled GPS points using `geopy` + Nominatim (no API key required).
- [x] Respect Nominatim rate limits (1 request/second max).
- [x] Store place names as `place_name` in chunks metadata — used for spatial pre-filtering.
- [x] Cache reverse-geocoding results locally to avoid repeated lookups.
- [x] Add an option to disable place-name resolution for fully offline operation after ingest.

### 10.3 Browser and reading history

- [x] Implement `ChromeHistoryIngestor`.
- [x] Locate Chrome SQLite history file:
  - macOS: `~/Library/Application Support/Google/Chrome/Default/History`
  - Linux: `~/.config/google-chrome/Default/History`
  - Windows: `%LOCALAPPDATA%\Google\Chrome\User Data\Default\History`
- [x] **Copy the file before querying** — Chrome locks the original while running.
- [x] Query `urls` and `visits` tables; extract `url`, `title`, `last_visit_time`.
- [x] Convert Chrome timestamp (microseconds since 1601-01-01) to Unix epoch.
- [x] Filter out internal URLs (`chrome://`, `chrome-extension://`, `about:`).
- [ ] `[OPTIONAL]` Fetch full article text from saved URLs using `trafilatura`.
- [ ] `[OPTIONAL]` Implement `ReadwiseIngestor` — highest-quality structured reading data.
- [ ] `[OPTIONAL]` Use Readwise API to export highlights: `GET /api/v2/highlights/`.
- [ ] `[OPTIONAL]` For each highlight: extract `text`, `source_url`, `book_title`, `author`, `highlighted_at`.
- [ ] `[OPTIONAL]` Implement `PocketIngestor` using Pocket API export.
- [ ] `[OPTIONAL]` Implement `InstapaperIngestor` using Instapaper CSV/JSON export.

### 10.4 Tool options for structured and reading sources

- [x] ICS parsing: `icalendar` preferred, alternative parser only if interoperability issues arise.
- [x] Reading extraction: `trafilatura` for article text extraction, raw metadata-only import as a fallback.
- [x] Geocoding: Nominatim via `geopy`, cached local results, or disabled mode.

---

## 11. Metadata Store, Vector Index, and Data Model

### 11.1 Finalize canonical schema

- [x] Implement the SQLite `chunks` table as the source of truth for display and filtering.
- [x] Include fields for chunk ID, vector ID, source type, file path, timestamp, duration, coordinates, place name, session ID, and raw text.
- [x] Extend the schema with modality-specific optional fields where necessary, such as `scene_id`, `speaker_id`, `detected_faces`, or thumbnail paths.
- [x] Add source registry and ingest run tables in addition to chunk storage.

### 11.2 Finalize indexing strategy

- [x] Create four Qdrant collections at startup (if not already existing):
  - `text_chunks` — vector dim matches chosen text model (e5-large: 1024, nomic: 768)
  - `image_frames` — 512 or 768 dim depending on CLIP model (ViT-B-32: 512, ViT-L-14: 768)
  - `video_frames` — same dim as `image_frames`
  - `audio_transcripts` — same dim as `text_chunks`
- [x] Configure cosine distance for all collections.
- [x] Enforce consistent payload fields for every vector point: `chunk_id`, `session_id`, `timestamp_utc`, `source_type`, `lat`, `lon`, `file_path`.
- [x] Create Qdrant payload indexes on: `session_id`, `timestamp_utc`, `source_type`, `lat`/`lon`.
- [ ] `[OPTIONAL]` Enable Qdrant sparse vector support for hybrid BM25+dense queries in `text_chunks`.
- [x] Decide whether sparse vectors are stored inside Qdrant or managed separately for BM25. Decision: external BM25 via `rank_bm25` in application code for v1.

### 11.3 Data consistency tasks

- [x] Guarantee that every vector entry has a matching SQLite record and vice versa to satisfy NFR-R-03.
- [x] Add reconciliation tooling to detect orphaned SQLite rows or orphaned vector entries.
- [x] Add transactional write sequencing or compensating cleanup for partial failures during ingest.
- [x] Implement a `lifelog consistency-check` CLI command that verifies no orphan records exist in either store.

### 11.4 Storage tool choices

- [x] Vector store: Qdrant preferred for dense plus sparse plus RRF support.
- [x] Prototype alternative: Chroma for simpler startup, with the explicit tradeoff that hybrid retrieval is weaker.
- [x] Embedded alternative: LanceDB if a file-based vector store is preferred over a server process.

---

## 12. Sessionization, Temporal Logic, and Query Signal Extraction

### 12.1 Session generation

- [x] Implement `session_id` assignment at ingest time using the default 4-hour sliding window.
- [x] Decide how to cluster sparse events across midnight or timezone changes. Decision: window is purely time-based; midnight and timezone differences are absorbed by the UTC timestamp comparison.
- [x] Add tests for adjacent events within and outside the threshold window.
- [x] Decide whether the time window is globally configurable or source-specific. Decision: globally configurable via `LIFELOG_SESSION_WINDOW_HOURS` (default 4.0).

### 12.2 Query signal extraction

- [x] Implement `QueryAnalyzer` that extracts structured signals from the raw user query before retrieval:
  - **Temporal hints:** date expressions (`"last summer"`, `"October 2023"`, `"the week before I left"`) → resolve to Unix timestamp range
  - **Spatial hints:** place names (`"Lisbon"`, `"at the office"`, `"Accra"`) → resolve to GPS bounding box or `place_name` string
  - **Person entities:** proper nouns likely to be people (`"Kofi"`, `"my grandmother"`) → used for BM25 name search
  - **Visual intent signals:** keywords suggesting image search (`"photo"`, `"picture"`, `"saw"`, `"looked like"`, `"sunset"`) → boost CLIP path weight
- [x] Use a fast regex + NER approach first (spaCy `en_core_web_sm` is fast enough).
- [ ] `[OPTIONAL]` Use a lightweight LLM call (local Ollama or API) for more nuanced temporal expression resolution.
- [x] Implement visual keyword set: `{"photo", "picture", "sunset", "saw", "looked", "view", "scene", "face", "rainy", "market"}`.

### 12.3 Temporal ranking logic

- [x] Implement temporal boosts around extracted target times.
- [x] Tune `tau` and `alpha` for sensible behavior on day-scale and week-scale queries.
- [x] Add tests for exact-date queries, week-level queries, and relative follow-up references.

---

## 13. Retrieval, Hybrid Search, Fusion, and Re-ranking ✅

### 13.1 Build parallel retrieval paths

- [x] Implement dense semantic retrieval for text-capable chunks.
- [x] Implement CLIP text-to-image and text-to-video-frame retrieval.
- [x] Implement metadata pre-filters for date, place, and source type.
- [x] Implement exact-match or sparse retrieval for names, titles, and technical terms.

### 13.2 Choose sparse and hybrid retrieval approach

- [ ] Option A: Qdrant native sparse vectors plus hybrid query support.
- [x] Option B: external BM25, such as `rank_bm25`, combined in application code. **Selected.**
- [x] Document tradeoffs in speed, simplicity, and ranking quality.

### 13.3 Build fusion layer

- [x] Implement `RRFFusion` — Reciprocal Rank Fusion as the default combiner.
- [x] Use `k=60` as the default smoothing constant.
- [x] RRF is immune to score scale differences between modalities — no normalization needed.
- [ ] `[OPTIONAL]` Use Qdrant native RRF fusion if using Qdrant sparse vectors.
- [ ] `[OPTIONAL]` Implement `WeightedFusion` as an alternative to RRF: normalize scores per modality (min-max), then combine with dynamic weights.
- [x] Add query-dependent modality weighting for visually dominant queries: `w_img` = 0.2 base + 0.15 per visual keyword, max 0.8.
- [x] Make fusion strategy configurable in `config.yaml`.

### 13.4 Build re-ranking layer

- [x] Implement `TemporalReranker` — apply exponential decay boost when a temporal hint was extracted.
- [x] Default `tau=7` days, `alpha=0.5` (max 50% boost for a perfect temporal match).
- [x] Make `tau` and `alpha` configurable in `config.yaml`.
- [x] Implement `CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- [x] Apply only to top-40 fused candidates (too slow for full collection).
- [x] For each candidate: score the (query, document_text) pair jointly.
- [x] Re-sort candidates by cross-encoder score.
- [x] For image/video results: use the caption or transcript chunk as the text input to the cross-encoder.
- [x] Preserve explainability metadata showing whether a result was boosted by time, exact match, or multimodal agreement.

### 13.5 Build grouping layer

- [x] Implement `SessionGrouper` that operates on the final ranked list after all re-ranking.
- [x] Group ranked chunks by `session_id`.
- [x] Each session card must expose:
  - `primary`: the highest-ranked chunk from this session
  - `secondary`: all other chunks from the same session (sorted by score)
  - `session_metadata`: date, place name (if available), list of source types present
- [x] Return top-5 session cards by default; make configurable.
- [x] For temporal exploration queries (UC-06): return a chronological list of session cards instead of relevance-sorted.

---

## 14. API, CLI, and Chat-Style Web UI ✅

### 14.1 REST API

- [x] Implement `POST /query` endpoint:
  - Request body: `{ "query": str, "filters": {}, "top_k": int, "conversation_id": str? }`
  - Response: `{ "sessions": [...], "query_debug": {...} }`
- [x] Implement `GET /status` — returns index stats (total chunks, by modality, last ingest).
- [x] Implement `POST /ingest/trigger` — manually trigger an incremental ingest run.
- [x] Bind to localhost (`127.0.0.1`) by default to satisfy NFR-S-03.
- [x] Add `--host` CLI flag to allow overriding (advanced users only, with warning).
- [x] Add structured request and response schemas.
- [x] No authentication required in v1.0 (localhost-only).

### 14.2 CLI

- [ ] Implement `lifelog init` — onboarding wizard (see §17 below).
- [ ] Implement `lifelog ingest --full` — full ingest of all configured sources.
- [ ] Implement `lifelog ingest --incremental` — ingest only new/changed items since last run.
- [ ] Implement `lifelog ingest --source <name>` — ingest a single source only.
- [x] Implement `lifelog query "<text>"` — run a query from the terminal, pretty-print results.
- [ ] Implement `lifelog status` — show index stats: total chunks, breakdown by modality, last ingest timestamp per source (FR-M-01).
- [x] Implement `lifelog delete --file <path>` — remove all chunks associated with a specific file from both Qdrant and SQLite (FR-M-02).
- [x] Implement `lifelog delete --source <name>` — remove all chunks from a given source directory.
- [x] Implement `lifelog consistency-check` — verify Qdrant and SQLite are in sync; report orphan records.
- [x] Implement `lifelog logs` — tail the ingest log with optional `--source` filter (FR-M-03).

### 14.3 Chat-style web UI (Option C: Next.js / React)

- [x] Choose UI framework: **Option C** — Next.js / React (`frontend/` at project root)
- [x] Build a minimal local UI for entering natural language queries and viewing grouped session cards.
- [x] Text input field for natural language queries.
- [x] Display results as session cards — grouped by `session_id`, not flat list.
- [x] Each session card shows:
  - Date + place name header
  - Primary result (text excerpt, image thumbnail, audio player link, or video frame)
  - Expandable secondary results (other modalities from the same session)
  - Source type icon/label for each result
- [x] "Open original" button per result — triggers `file://` link or calls `/open-file` API endpoint.
- [x] Maintain visible conversation history in the UI.
- [x] Send `conversation_id` with each follow-up query — enables server-side reference resolution.
- [x] Display clarification prompts from the server when references are ambiguous.
- [x] `[OPTIONAL]` Show query debug panel (extracted temporal hint, spatial hint, modality weights used).
- [x] Display image thumbnails inline in session cards (resize to max 400px width for display).
- [x] For audio results: show play button linking to original file (no in-UI playback — per spec, out of scope).
- [x] For video results: show keyframe thumbnail + timestamp + link to original file at scene start time.

### 14.4 Source linking (UC-08)

- [x] For text results: deep-link to editor at correct line:
  - **Option A:** Obsidian URI scheme: `obsidian://open?vault=...&file=...`
  - **Option B:** VS Code URI: `vscode://file/path/to/file:line`
- [x] For audio: open in default OS player seeked to `timestamp_start` (via `file://` + OS handler).
- [x] For video: open in default OS player at `scene_id` start timestamp.
- [x] For images: reveal in Finder (macOS) or Files (Linux) via `file://` link.

---

## 15. Conversational Memory and Follow-up Query Handling ✅

### 15.1 Context tracking for UC-03

- [x] Implement `ConversationManager`:
  - Store conversation history in memory (or SQLite) keyed by `conversation_id`
  - Resolve references to prior results: `"that same week"`, `"more from that session"`, `"what else happened that day"`
  - For ambiguous references: return clarification options (list of candidate sessions/people/places from context)
- [x] Multi-turn conversation must work without re-stating full context (FR-Q-09).
- [x] Store prior query context, top results, resolved dates, and selected session IDs.

### 15.2 Conversation state design

- [x] Decide whether conversation state lives in memory only, in SQLite session tables, or both. (In-memory with TTL cleanup; SQLite persistence deferred to v2)
- [x] Define TTL and cleanup behavior for conversation sessions. (1-hour idle TTL; `_cleanup_expired` called on each `resolve_context`)
- [x] Ensure follow-up retrieval can expand a date range or directly fetch all chunks by `session_id`.

### 15.3 Safety and UX tasks

- [x] Ask clarifying questions only when the reference cannot be resolved confidently.
- [x] Keep follow-up answers grounded in retrieved records rather than fabricated summaries.
- [x] Ensure the UI clearly shows which prior result or session the follow-up was anchored to.

---

## 16. Explainability, Result Linking, and Source Opening ✅

### 16.1 Explainability requirements

- [x] Each result chunk returned by API must include (FR-Q-10):
  - `source_type` (text / image / audio / video_frame / video_transcript)
  - `file_path` (absolute path on local machine)
  - `timestamp_utc` (and human-formatted `timestamp_display`)
  - `place_name` (if available)
  - `snippet` — 200-character excerpt for text, or thumbnail path for images/video
  - `score` and `rank` within its session
- [x] Show why a result matched: semantic similarity, exact keyword hit, temporal boost, location filter, or visual similarity. (via `match_reasons` field)
- [x] Decide whether to expose raw scores, normalized labels, or both. (Both: raw `score` + human `match_reasons` labels)

### 16.2 Open-original support for UC-08

- [x] For text, open the source file in the configured editor at the correct line or offset when possible.
- [x] For audio, open the file in the default player at the relevant timestamp if the platform allows it.
- [x] For video, open the file at scene start or nearest supported timestamp.
- [x] For photos, reveal the file in the OS file explorer or open it in the default viewer.

### 16.3 Platform handling tasks

- [x] Decide what level of source-opening support is cross-platform versus best-effort. (Best-effort: `os.startfile` on Windows, `open` on macOS, `xdg-open` on Linux)
- [x] Add fallback behavior when exact seek/open integration is not available. (HTTPException 500 with error detail)

---

## 17. Privacy, Security, and Local-Only Guarantees

### 17.1 Privacy-first enforcement

- [x] Audit the stack to ensure no query-time data leaves the machine during normal operation.
- [x] Document any optional components that may call external services during setup or data enrichment.
- [x] Add explicit offline-mode documentation.

### 17.2 Secure local defaults

- [x] Bind the API to localhost only by default.
- [x] Store SQLite and vector data in a user-configurable local directory.
- [x] Protect logs from leaking sensitive content unnecessarily.
- [x] Decide whether to redact or hash especially sensitive metadata in logs.

### 17.3 Dependency review

- [x] Review licenses and update frequencies for core ML and indexing dependencies.
- [x] Verify optional packages like face recognition and diarization meet privacy and maintenance expectations.

---

## 18. Reliability, Testing, and Quality Gates

### 18.1 Unit tests

- [x] Test `TextChunker` — verify chunk size bounds, overlap, separator selection.
- [x] Test `ExifExtractor` — GPS rational-to-decimal conversion (with edge cases: S/W negative, missing EXIF).
- [x] Test `SessionAssigner` — verify 4-hour window clustering with known timestamps.
- [x] Test `RRFFusion` — verify rank-based score computation with known ranked lists.
- [x] Test `TemporalReranker` — verify boost formula with known distance and decay values.
- [x] Test `QueryAnalyzer` — verify extraction of temporal, spatial, and person entities from sample queries.

### 18.2 Integration tests

- [x] Test full ingest pipeline for each modality with a small sample corpus (< 10 items per modality).
- [x] Test Qdrant ↔ SQLite consistency: after ingest, verify every Qdrant ID exists in SQLite and vice versa.
- [x] Test incremental ingest: add one new file per modality, run incremental, verify only new items ingested.
- [x] Test query end-to-end: submit a query, verify results are returned with correct fields.
- [x] Test cross-encoder re-ranking: verify top result changes after re-ranking on a known test case.

### 18.3 System-level tests (Use Cases)

- [x] UC-02: submit `"that afternoon I was stuck on the project in Lisbon"` against a seeded test corpus — verify result contains expected session.
- [x] UC-05: simulate interrupted ingest, restart, verify no duplicates in index.
- [x] UC-07: seed corpus with known person name in text + photo face tag — verify person search returns both modalities.
- [x] UC-08: verify "Open original" deep links resolve to correct file + position.

### 18.4 Failure handling tests

- [x] Test locked-file handling for Chrome history and Apple Photos workflows.
- [x] Test interrupted ingest resume behavior.
- [x] Test partial vector-store failures and consistency recovery.
- [x] Test corrupted media files and unsupported formats.

### 18.5 Performance benchmarks

- [x] Build a synthetic corpus of 100,000 chunks (mix of modalities).
- [x] Benchmark query latency at p50, p95, p99.
- [x] Benchmark ingest throughput: items/second per modality.
- [x] Profile memory usage under full ingest load.
- [x] Define minimum acceptable query quality benchmarks on a representative local fixture corpus.
- [x] Define latency budgets for dense retrieval, CLIP retrieval, fusion, and re-rank.
- [x] Add smoke tests for CPU-only mode.

---

## 19. Performance Tuning and Operational Monitoring

### 19.1 Query performance

- [ ] Keep end-to-end query latency under 3 seconds for the target corpus size.
- [ ] Profile candidate counts and re-rank costs.
- [ ] Optimize collection-level filters before ANN when date or place hints exist.
- [ ] Cache frequent metadata lookups and query analysis artifacts where safe.

### 19.2 Ingest performance

- [ ] Keep incremental ingest of 500 new items within 10 minutes on recommended hardware (NFR-P-02).
- [ ] Estimate full ingest time for a 100,000-item 10-year corpus — must complete within 24 hours (NFR-P-03).
- [ ] Benchmark full ingest on representative long-lived corpora.
- [ ] Implement batch embedding (not item-by-item) to maximize GPU/CPU utilization.
- [ ] Add batching for embeddings and transcription jobs.
- [ ] Add concurrency controls so the machine remains usable during long ingest runs.

### 19.3 Monitoring and diagnostics

- [ ] Add ingest progress reporting by modality.
- [ ] Add query tracing for stage timings.
- [ ] Add a local status page or CLI summary for index health and storage usage.

---

## 20. Documentation, Rollout, and Future Extensions

### 20.1 Core documentation

- [ ] Write `README.md` covering: project overview, installation, quickstart, configuration reference.
- [ ] Write `INSTALL.md` with OS-specific instructions for macOS, Linux, and Windows.
- [ ] Document all `config.yaml` keys with types, defaults, and descriptions.
- [ ] Write `ARCHITECTURE.md` summarizing the two-store design (Qdrant + SQLite), pipeline stages, and retrieval flow.
- [ ] Write per-modality ingestion guides: how to export from each source (Google Takeout, osxphotos, Apple Health, etc.).
- [ ] Write `EXTENDING.md`: how to add a new modality by implementing `BaseIngestor`.
- [ ] Document known constraints (HEIC support, Chrome lock, Apple Photos lock, CPU Whisper throughput).
- [ ] Document out-of-scope items from v1.0 (cloud deployment, multi-user, real-time ingest, mobile client, social media, fine-tuning).
- [ ] Write an FAQ covering common issues: model download failures, Chrome history lock error, HEIC not supported, Whisper too slow.

### 20.2 Release readiness

- [ ] Implement `lifelog init` interactive CLI wizard (UC-01):
  1. Check for required system dependencies (ffmpeg, Docker/Qdrant) — print status for each
  2. Run model download script if models not found
  3. Present list of supported sources with example paths for each
  4. Prompt user to select sources and provide paths/credentials
  5. Validate each path (exists? readable? contains expected file types?)
  6. Preview item counts per source before committing to full ingest
  7. Write validated config to `config.yaml`
  8. Confirm and trigger `lifelog ingest --full`
  9. Show live progress bar per modality during ingest
  10. On completion: report total chunks, breakdown by modality, index size on disk
  11. Run a test query: `lifelog query "hello"` to verify end-to-end
- [ ] Total time from installation to first query: target < 30 minutes for a new user (NFR-U-01).
- [ ] Support resumable full ingest — if interrupted, `lifelog init` detects prior partial ingest and resumes (NFR-R-02).
- [ ] Prepare a demo dataset and scripted walkthrough covering onboarding, query, refinement, and source opening.
- [ ] Verify the system works from a clean install on at least one primary target platform.
- [ ] Prepare a migration path for schema changes between early versions.

### 20.3 Planned post-v1 extensions

- [ ] Add Apple Health, Google Location History, Readwise, Pocket, and Instapaper when the core ingest pipeline is stable.
- [ ] Add additional modality ingestors through the shared `BaseIngestor` interface.
- [ ] Consider local personalization from user feedback without model fine-tuning.
- [ ] Consider better offline reverse-geocoding if online Nominatim access becomes undesirable.

---

## Appendix A. Requirement Traceability Checklist

Use this to confirm the implementation plan covers the major requirements.

- [ ] FR-I-01 to FR-I-09 mapped to ingestion sections 4 through 10.
- [ ] FR-Q-01 to FR-Q-10 mapped to retrieval, grouping, API, UI, and explainability sections 12 through 16.
- [ ] FR-M-01 to FR-M-03 mapped to CLI, status, delete, and logging work in sections 5 and 14.
- [ ] NFR-P-01 to NFR-P-03 mapped to sections 18 and 19.
- [ ] NFR-S-01 to NFR-S-04 mapped to section 17.
- [ ] NFR-R-01 to NFR-R-03 mapped to sections 5, 11, and 18.
- [ ] NFR-U-01 to NFR-U-03 mapped to sections 4, 14, and 16.
- [ ] NFR-E-01 to NFR-E-02 mapped to sections 5 and 3.

## Appendix B. Key Decision Points Before Coding Deeply

These choices should be finalized early because they affect large parts of the implementation.

- [ ] Choose the vector store: Qdrant (recommended), Chroma, LanceDB, or pgvector.
- [ ] Choose text embeddings: e5-large-v2 (recommended), nomic-embed-text, or Ollama-served model.
- [ ] Choose transcription path: WhisperX (recommended), openai Whisper, whisper.cpp, or whisper-turbo.
- [ ] Choose scheduler: Prefect (for richer flows) or APScheduler (lightweight).
- [ ] Choose UI stack: Streamlit (fastest), Gradio (better multimodal), or Next.js/React (most polished).
- [ ] Choose which optional enrichments are in v1.0: face recognition, captioning, diarization, reverse geocoding, reading-service connectors (Readwise, Pocket, Instapaper), Apple Health, Google Location History, YouTube ingestion, Otter.ai exports.

## Appendix C. System Constraints and Mitigations

- [ ] HEIC support: install `pillow-heif` at setup time; fall back to ImageMagick conversion if unavailable.
- [ ] Apple Photos lock: document clearly; `osxphotos` ingestor must warn if Photos.app is running.
- [ ] Chrome history lock: always copy the SQLite file to `/tmp` before querying.
- [ ] CPU-only Whisper: document expected throughput (1 hour audio ≈ 3–4 hours on CPU with large-v3); recommend `whisper-turbo` for CPU-only setups.
- [ ] Model storage: document total disk requirement (~10GB); check available disk space in `lifelog init`.
