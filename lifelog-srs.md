# Life Log Search Engine — Software Requirements Specification (SRS)

**Version:** 1.0  
**Status:** Draft  
**Last Updated:** 2026-05-05

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Project Objectives](#3-project-objectives)
4. [Stakeholders & User Personas](#4-stakeholders--user-personas)
5. [Use Cases](#5-use-cases)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [System Constraints](#8-system-constraints)
9. [Assumptions & Dependencies](#9-assumptions--dependencies)
10. [Out of Scope](#10-out-of-scope)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the software requirements for the **Life Log Search Engine** — a personal, privacy-first system that indexes a user's journals, photos, voice memos, videos, emails, calendar events, and browsing history, and makes them queryable through natural language.

### 1.2 Problem Statement

People generate enormous volumes of personal data across disconnected apps — journals in Obsidian, photos in Apple Photos, voice memos on iOS, videos on a hard drive, highlights in Readwise. None of these systems talk to each other. Searching your own past is fragmented, slow, and often impossible for cross-modal queries like:

> _"Find that afternoon in Lisbon where I was stuck on that project — I think I took photos and recorded a voice memo."_

No existing tool answers this query across all four modalities simultaneously.

### 1.3 Proposed Solution

A locally-running RAG system that ingests all personal data modalities into a unified vector index and metadata store. Users query the system in natural language via a chat-style interface. Results are ranked, fused across modalities, and grouped by time session — surfacing the right moment in a user's past, not just matching keywords.

### 1.4 Scope

The system covers ingestion, processing, indexing, and retrieval of personal multi-modal data. It includes a local query API and a minimal chat UI. It does not include cloud hosting or real-time data streaming.

### 1.5 Definitions

| Term | Definition |
|---|---|
| **Modality** | A type of data source: text, image, audio, video, structured metadata |
| **Chunk** | A unit of content (paragraph, image, audio segment) stored as a single embedding |
| **Session** | A cluster of events within a 4-hour window, treated as a single moment in time |
| **Session card** | A result card grouping all modality hits from the same session |
| **ANN** | Approximate Nearest Neighbor — fast vector similarity search |
| **RRF** | Reciprocal Rank Fusion — rank-based score combiner across modalities |
| **CLIP** | Contrastive Language-Image Pre-training — model that embeds text and images in a shared space |

---

## 2. Overall Description

### 2.1 System Context

The Life Log Search Engine runs entirely on the user's local machine. It has no mandatory external dependencies beyond model downloads during setup. It exposes:

- A **REST API** (FastAPI) for query and ingest operations
- A **chat-style web UI** for conversational memory retrieval
- A **CLI** for ingest management and index inspection

### 2.2 System Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        User interfaces                          │
│           Chat UI  ·  CLI  ·  REST API (FastAPI)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       Query engine                              │
│   Query analysis → parallel retrieval → fusion → re-rank        │
└───────┬────────────────────┬───────────────────────┬────────────┘
        │                    │                        │
┌───────▼──────┐   ┌─────────▼────────┐   ┌──────────▼──────────┐
│ Vector index │   │  Metadata store  │   │  Embedding models   │
│   (Qdrant)   │   │    (SQLite)      │   │  e5 · CLIP · Whisper│
└───────┬──────┘   └─────────┬────────┘   └──────────┬──────────┘
        │                    │                        │
┌───────▼────────────────────▼────────────────────────▼──────────┐
│                       Ingest pipeline                           │
│   Text · Photos · Audio · Video · Calendar · Browser history    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Operating Environment

- **OS:** macOS (primary), Linux (supported)
- **Hardware:** Minimum 16GB RAM, 8GB VRAM recommended for GPU inference. CPU-only mode supported with reduced throughput.
- **Storage:** Depends on corpus size. Vector index: ~2KB per chunk. Metadata: negligible. Model weights: ~10GB total (Whisper large-v3 + CLIP ViT-L-14 + e5-large).
- **Runtime:** Python 3.11+, Docker (optional, for Qdrant server mode)

---

## 3. Project Objectives

### 3.1 Primary Objectives

**OBJ-01: Unified personal memory retrieval**
Enable a user to query across all personal data modalities simultaneously using natural language, and receive ranked, grouped results that reflect actual moments in their life — not just keyword matches.

**OBJ-02: Privacy-first, local execution**
The system must run entirely on the user's local machine. No personal data leaves the device. All models run locally. No third-party API is required after initial setup.

**OBJ-03: Cross-modal result fusion**
A single query must be able to surface results from different modalities (e.g., a journal entry, two photos, and a voice memo) grouped into a single session card representing a coherent life moment.

**OBJ-04: Temporal and spatial awareness**
The system must correctly interpret temporal expressions ("last summer," "the week before I left") and spatial hints ("when I was in Lisbon," "at the office") and use them to filter and re-rank results.

**OBJ-05: Incremental ingest**
New data added to any source (new journal entries, new photos, new voice memos) must be automatically detected and ingested without re-processing the full corpus.

### 3.2 Secondary Objectives

**OBJ-06: Conversational memory interface**
Support multi-turn conversation where follow-up queries refine the previous result set ("show me more from that week," "what else happened that day").

**OBJ-07: Explainable results**
Each result must display its source (file path, timestamp, modality) and why it was retrieved (matched query terms, temporal relevance, visual similarity).

**OBJ-08: Extensible modality support**
The ingest pipeline must be designed so new modalities (e.g., Spotify listening history, fitness GPS traces) can be added without restructuring the core retrieval system.

**OBJ-09: Continuous data ingestion**
The system should access device to ingest new user data with appropriate permissions. It should also regularly remind users to upload data.


---

## 4. Stakeholders & User Personas

### 4.1 Primary User — The Personal Archivist

Someone who actively journals, takes photos, records voice memos, and cares about their personal history. Technically literate — comfortable running a local server and a CLI. Values privacy above convenience. Frustrated by fragmented apps that can't answer cross-modal questions about their own past.

**Representative query types:**
- "What was I thinking about the week before I quit my job?"
- "Find that photo from the rainy market in Accra."
- "What did I say in my voice memo after the meeting with the investors?"

### 4.2 Secondary User — The Knowledge Worker

Someone who uses the system primarily as a work memory tool — searching past meeting notes, emails, and reading highlights to reconstruct context for ongoing projects.

**Representative query types:**
- "What did I read about transformer attention mechanisms last year?"
- "Find the email thread where we discussed the API redesign."
- "What was I working on the week I started the recommendation system project?"

### 4.3 Secondary User — The Retrospective Thinker

Someone who uses the system episodically — pulling it up to explore a specific period of their life (a trip, a job change, a relationship) rather than querying it daily.

**Representative query types:**
- "Show me everything from my month in Lagos."
- "What was happening in my life in October 2023?"
- "Find any memories connected to my grandmother."

---

## 5. Use Cases

---

### UC-01: First-time Onboarding

**Actor:** New user  
**Preconditions:** System installed, models downloaded  
**Goal:** Connect data sources and build the initial index

**Main Flow:**

1. User launches the onboarding CLI wizard: `lifelog init`
2. System presents a list of supported data sources with connection instructions for each.
3. User selects sources to connect (e.g., Obsidian vault path, Google Takeout export folder, iOS Voice Memos path).
4. System validates each source path and previews the estimated number of items to ingest.
5. User confirms and triggers full ingest: `lifelog ingest --full`
6. System processes all sources in parallel, displaying a progress bar per modality.
7. System generates `session_id` clusters across all ingested items by timestamp.
8. System reports completion: total chunks indexed, breakdown by modality, estimated index size.
9. User runs a test query to verify: `lifelog query "hello"`

**Alternative Flow — interrupted ingest:**
- If ingest is interrupted, the system resumes from the last committed checkpoint on next run.

**Postconditions:** Vector index populated, metadata store populated, system ready for queries.

---

### UC-02: Natural Language Memory Query

**Actor:** User  
**Preconditions:** Index populated  
**Goal:** Retrieve memories matching a natural language description

**Main Flow:**

1. User submits a query via the chat UI or CLI: *"that afternoon I was stuck on the project in Lisbon"*
2. System runs query analysis — extracts:
   - Temporal hint: `afternoon`
   - Location hint: `Lisbon`
   - Mood/state: `stuck, frustrated`
   - Visual description: `Lisbon city afternoon`
3. System fans out to parallel retrieval paths:
   - Text ANN search with `"query: that afternoon I was stuck on the project in Lisbon"`
   - CLIP image search with synthesized visual query `"Lisbon city cafe afternoon light"`
   - Metadata pre-filter on GPS cluster matching Lisbon coordinates
4. System applies RRF fusion across all result pools.
5. System applies temporal re-rank boost for results timestamped in the afternoon window.
6. System applies cross-encoder re-rank on top-40 fused results.
7. System groups results by `session_id` — collapses related hits into session cards.
8. System returns top-5 session cards, each showing:
   - Primary result (highest-scored modality hit)
   - Secondary hits from the same session (other modalities, other chunks)
   - Session metadata: date, place name, sources present
9. User views results and selects a session card to expand.

**Postconditions:** User finds the memory or refines the query and system learns from the activity to provide better results in the future. 

---

### UC-03: Conversational Memory Refinement

**Actor:** User  
**Preconditions:** UC-02 completed, results returned  
**Goal:** Refine results through follow-up queries without re-stating full context

**Main Flow:**

1. System has returned 5 session cards from a prior query.
2. User sends a follow-up: *"show me more from that same week"*
3. System resolves the reference ("that same week") using the timestamp of the previously top-ranked result.
4. System re-queries with an expanded temporal window (7-day range centred on the resolved date).
5. System returns new results, still grouped by session.
6. User sends another follow-up: *"what else happened that day?"*
7. System queries by `session_id` of the previously selected card — returns all chunks from that session across all modalities.

**Alternative Flow — ambiguous reference:**
- If the reference is ambiguous (e.g., "that project" with no prior project mentioned), system asks for clarification: *"Which project are you referring to? I found mentions of: X, Y, Z in recent results."*

**Postconditions:** User has drilled into a specific period or moment with increasing specificity.

---

### UC-04: Visual Memory Search

**Actor:** User  
**Preconditions:** Photo and/or video modality ingested  
**Goal:** Find photos or video frames matching a visual description, with no text labels required

**Main Flow:**

1. User submits a query: *"find photos that feel like a rainy market"*
2. System identifies the query as primarily visual (keywords: `photos`, `rainy`, `market`).
3. System sets `w_img = 0.8` — heavily weights CLIP image path.
4. System embeds query with CLIP text encoder.
5. System searches `image_frames` and `video_frames` collections by cosine similarity.
6. System also searches `text_chunks` for any journal entries or voice memo transcripts mentioning markets or rain — as supporting context.
7. Results returned as a mixed-modality session card: photos + video frames (primary), text context (secondary).
8. Each image result displays: thumbnail, date, location (if EXIF available), similarity score.

**Postconditions:** User finds visually relevant photos/frames without needing to have tagged or captioned them.

---

### UC-05: Incremental Ingest

**Actor:** System (scheduled) / User (manual trigger)  
**Preconditions:** Initial ingest completed, source directories monitored  
**Goal:** Index new data added since the last ingest run without re-processing existing data

**Main Flow:**

1. Scheduler triggers `lifelog ingest --incremental` (default: nightly at 2am).
2. System reads the `last_ingest_timestamp` from the metadata store.
3. For each configured source:
   - Text sources: scan for files with `mtime > last_ingest_timestamp`
   - Photos/videos: check for new files in watched directories
   - Email: fetch messages received since `last_ingest_timestamp` via IMAP or from updated MBOX
4. System processes only new/modified items.
5. System updates `last_ingest_timestamp` on completion.
6. System logs: N new chunks indexed, modality breakdown.

**Alternative Flow — modified file:**
- If an existing file is modified (e.g., journal entry edited), system re-embeds all chunks from that file and updates the vector index. Old chunks are deleted by `file_path`.

**Postconditions:** Index is up to date. No duplicate embeddings exist.

---

### UC-06: Temporal Exploration — "What was happening in my life?"

**Actor:** User  
**Preconditions:** Index populated with at least 3 months of data  
**Goal:** Get a broad overview of a period in the user's life without a specific memory in mind

**Main Flow:**

1. User submits: *"what was I up to in October 2023?"*
2. System identifies this as a temporal exploration query (no semantic memory target — just a time range).
3. System applies a metadata pre-filter: `timestamp_utc BETWEEN oct_start AND oct_end`.
4. System retrieves a stratified sample across modalities: top journal entries by density, top photos by CLIP diversity, calendar events, voice memos.
5. System groups by session — returns a chronological list of session cards covering the month and a summary of the items in the time range.
6. Each card shows: date, dominant activity inferred from content, modalities present.
7. User can select any session card to explore in depth.

**Postconditions:** User has a navigable overview of a past period and a summary.

---

### UC-07: Person-Centric Search

**Actor:** User  
**Preconditions:** Face detection run on photos (optional), names appear in text  
**Goal:** Find all memories associated with a specific person

**Main Flow:**

1. User submits: *"show me everything involving my friend Kofi"*
2. System extracts entity: `person = "Kofi"`.
3. System runs BM25 keyword search across `text_chunks` for mentions of "Kofi".
4. System queries `image_frames` payload metadata for `detected_faces` containing "Kofi" (if face recognition was run during ingest).
5. System merges results, groups by session.
6. Results returned as a timeline of sessions involving Kofi: journal entries mentioning him, photos he appears in, voice memos that reference him.

**Postconditions:** User gets a person-centric view of a relationship across time and modality.

---

### UC-08: Source Export & Result Linking

**Actor:** User  
**Preconditions:** Result displayed in UI  
**Goal:** Open the original source file at the exact position of a retrieved chunk

**Main Flow:**

1. User views a session card result and clicks "Open original".
2. System resolves `file_path` and `timestamp_utc` (or character offset for text) from the metadata store.
3. For text: system opens the file in the configured editor (Obsidian, VS Code) at the correct line.
4. For audio: system opens the audio file in the default player, seeked to `timestamp_start`.
5. For video: system opens the video file seeked to `scene_id` start timestamp.
6. For photos: system opens the image in the default viewer or reveals it in Finder/Files.

**Postconditions:** User is looking at the original source artifact, not a copy.

---

## 6. Functional Requirements

### 6.1 Ingest

| ID | Requirement |
|---|---|
| FR-I-01 | System shall support ingestion of Markdown/plain text files from a configurable directory path. |
| FR-I-02 | System shall support ingestion of MBOX email exports, parsing sender, recipient, date, and body. |
| FR-I-03 | System shall ingest JPEG, PNG, and HEIC image files, extracting EXIF GPS and timestamp. |
| FR-I-04 | System shall ingest MP3, M4A, and WAV audio files and produce a timestamped transcript via Whisper. |
| FR-I-05 | System shall ingest MP4 and MOV video files, sampling keyframes via scene detection and transcribing the audio track. |
| FR-I-06 | System shall ingest ICS calendar exports and parse event titles, locations, start/end times. |
| FR-I-07 | System shall support incremental ingest based on file modification time, without re-processing unchanged files. |
| FR-I-08 | System shall assign a `session_id` to all chunks within a configurable time window (default: 4 hours). |
| FR-I-09 | System shall reverse-geocode GPS coordinates to a human-readable place name and store in metadata. |

### 6.2 Query & Retrieval

| ID | Requirement |
|---|---|
| FR-Q-01 | System shall accept natural language queries via REST API and chat UI. |
| FR-Q-02 | System shall extract temporal, spatial, and entity signals from the query before retrieval. |
| FR-Q-03 | System shall perform parallel ANN search across all relevant modality collections. |
| FR-Q-04 | System shall fuse multi-modality results using Reciprocal Rank Fusion. |
| FR-Q-05 | System shall apply temporal re-rank when a temporal hint is detected in the query. |
| FR-Q-06 | System shall apply cross-encoder re-ranking on the top-40 fused candidates. |
| FR-Q-07 | System shall group final results by `session_id` before returning to the user. |
| FR-Q-08 | System shall support metadata pre-filtering by date range, place name, and source type. |
| FR-Q-09 | System shall support multi-turn conversation, resolving references to prior query context. |
| FR-Q-10 | System shall return, with each result: source type, file path, timestamp, place name, and a snippet or thumbnail. |

### 6.3 Ingest Management

| ID | Requirement |
|---|---|
| FR-M-01 | System shall provide a CLI command to inspect index status: total chunks, breakdown by modality, last ingest timestamp. |
| FR-M-02 | System shall allow a user to delete all chunks associated with a specific file or source directory. |
| FR-M-03 | System shall log all ingest runs with item counts, errors, and duration. |

---

## 7. Non-Functional Requirements

### 7.1 Performance

| ID | Requirement |
|---|---|
| NFR-P-01 | Query latency (end-to-end, including fusion and re-rank) shall be under 3 seconds for a corpus of up to 500,000 chunks on recommended hardware. |
| NFR-P-02 | Incremental ingest of up to 500 new items shall complete within 10 minutes on recommended hardware. |
| NFR-P-03 | Full ingest of a 10-year personal corpus (estimated 100,000 items) shall complete within 24 hours. |

### 7.2 Privacy & Security

| ID | Requirement |
|---|---|
| NFR-S-01 | No personal data shall be transmitted to any external service during normal operation. |
| NFR-S-02 | All model inference shall run locally. |
| NFR-S-03 | The REST API shall bind to localhost only by default. |
| NFR-S-04 | The SQLite metadata store and Qdrant index shall be stored in a user-configurable, local directory. |

### 7.3 Reliability

| ID | Requirement |
|---|---|
| NFR-R-01 | Ingest failures for individual items shall be logged and skipped without halting the pipeline. |
| NFR-R-02 | The system shall be restartable — an interrupted ingest shall resume from the last checkpoint without data loss or duplication. |
| NFR-R-03 | The vector index and metadata store shall remain consistent — a chunk present in Qdrant shall always have a corresponding row in SQLite, and vice versa. |

### 7.4 Usability

| ID | Requirement |
|---|---|
| NFR-U-01 | Onboarding shall guide a non-expert user from installation to first query in under 30 minutes. |
| NFR-U-02 | Query results shall be displayed in a session-grouped format with clear modality indicators (icon, label) for each hit. |
| NFR-U-03 | Every result shall link directly to the original source file. |

### 7.5 Extensibility

| ID | Requirement |
|---|---|
| NFR-E-01 | New modality ingestors shall be implementable by creating a single class conforming to the `BaseIngestor` interface, without modifying retrieval logic. |
| NFR-E-02 | The embedding model used for text and images shall be configurable without code changes. |

---

## 8. System Constraints

- **CLIP embedding space mismatch**: Text queries and image embeddings live in the same CLIP latent space, but the alignment is imperfect for highly abstract or emotional text. The system must synthesize a visual-friendly query variant for the CLIP path.
- **Whisper transcription speed**: On CPU-only hardware, transcribing 1 hour of audio with Whisper large-v3 takes approximately 3–4 hours. Full audio corpus ingest may require GPU or use of `whisper-turbo`.
- **HEIC format support**: HEIC images (default iPhone format) require `pillow-heif` or conversion via `ImageMagick` before processing. Not natively supported by Pillow.
- **Apple Photos lock**: The Apple Photos library SQLite database is locked while Photos.app is running. `osxphotos` must be run with the app closed or via its export mode.
- **Chrome history lock**: The Chrome history SQLite file is locked while Chrome is running. Must be copied before querying.
- **Model storage**: Full model suite (Whisper large-v3 + CLIP ViT-L-14 + e5-large-v2 + cross-encoder) requires approximately 10GB of disk space.

---

## 9. Assumptions & Dependencies

**Assumptions:**

- The user's personal data files are stored locally or can be exported to local storage (Google Takeout, iCloud).
- The user has sufficient local storage for both the raw corpus and the derived index (~2–3× the raw corpus size).
- Python 3.11+ and either Docker or native Qdrant binary are available on the target machine.
- The user can tolerate a one-time full ingest run before the system is queryable.

**External Dependencies:**

| Dependency | Purpose | Required |
|---|---|---|
| Qdrant | Vector store | Yes (local) |
| `sentence-transformers` | Text embedding | Yes |
| `open_clip` | Image/cross-modal embedding | Yes |
| `whisper` / `whisperX` | Audio transcription | Yes |
| `ffmpeg` | Audio/video processing | Yes |
| `PySceneDetect` | Video scene detection | For video ingest |
| `pyannote.audio` | Speaker diarization | Optional |
| `osxphotos` | Apple Photos export | macOS only |
| `geopy` + Nominatim | Reverse geocoding | Optional |
| `face_recognition` | Face detection in photos | Optional |

---

## 10. Out of Scope

The following are explicitly excluded from version 1.0:

- **Cloud deployment or multi-device sync.** The system runs on one local machine. No cloud backend, no sync service.
- **Multi-user support.** The system is single-user. No authentication, no access control, no shared indexes.
- **Real-time ingest.** Ingest is batch-based (scheduled or manual). Streaming ingest of live data (e.g., live transcription of phone calls) is not supported.
- **Mobile client.** No iOS or Android app. The chat UI is a local web interface.
- **Social media ingestion.** Twitter/X, Instagram, Facebook exports are not supported in v1.0 due to format instability and extraction complexity.
- **Audio playback or video playback within the UI.** The UI links to the original file; playback happens in the native OS app.
- **Fine-tuning of embedding models on personal data.** All models are used off-the-shelf. Personalisation is achieved through retrieval, not model training.
