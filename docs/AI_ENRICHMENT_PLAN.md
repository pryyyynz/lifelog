# AI Enrichment & Awareness Plan

Plan for making Life Log Search "AI smart & aware." Captures the locked decisions
from the 2026-06-15 requirements session and the phased build.

## Locked decisions

| Dimension | Decision |
|---|---|
| Hardware | NVIDIA GPU, **8 GB VRAM** |
| Privacy | **Strictly local always** — no cloud, not even opt-in |
| Images | OCR + scene captioning + object/scene tags |
| Video | Per-scene captions + action recognition |
| Answers | Synthesize a cited answer **+** session cards on every query |
| Reasoning | Query decomposition (multi-search → merge → answer) |
| Faces | Auto-cluster, user names clusters later |
| Proactive | Auto titles/summaries, "on this day", daily/weekly digests, insights |
| Rollout | Enrich new items immediately; backfill existing in a low-priority background pass |

## Core architecture principle

**Describe non-text modalities as text, then reuse the existing retrieval stack.**

Audio already does this (Whisper transcripts). OCR, captions, and action labels are
the same move: each produces a *derived text chunk* (`vector_collection="text_chunks"`,
with `embedding_text` in metadata) that flows through the existing
e5 + BM25 + cross-encoder pipeline. CLIP stays for pure visual similarity. Faces become
a searchable people index. Every new capability is one `Enricher` subclass — honoring
OBJ-08 (add a modality without touching retrieval).

Derived chunks are tagged with `metadata.derived_from = <parent chunk_id>` so they are
never re-enriched, and use a stable `chunk_identity` of `"<enricher>:<suffix>"` so
re-runs upsert idempotently.

## Hardware-shaped execution model (8 GB)

- Enrichment runs as a **background pass that loads one model at a time**; the RAG LLM is
  served by **Ollama**, which swaps models in/out of VRAM.
- The enrichment runner accepts a `should_pause()` gate. While a user query is in flight,
  the API sets the gate so background enrichment yields the GPU — **answers stay fast,
  enrichment pauses**, then resumes.
- All heavy steps are **flag-gated and degrade gracefully** when a model/dependency is
  absent (mirrors the existing `OpenClipImageEmbedder` / `SentenceTransformerEmbedder`
  status pattern): a missing model records a status, never crashes ingest.

## Local model choices (strictly local, ≤8 GB)

| Capability | Library / model | VRAM | Notes |
|---|---|---|---|
| OCR | RapidOCR (onnxruntime) | ~0 (CPU) | No torch dependency; fast on short text |
| Captioning | moondream2 / BLIP via Ollama or transformers | ~2–4 GB | Smallest capable VLM |
| Object/scene tags | zero-shot CLIP (reuse `ViT-L-14`) | ~1.7 GB | Reuses model already in the suite |
| Action recognition | X-CLIP (ViT-B, zero-shot) | ~2–3 GB | Operates on short clips around scenes |
| Faces | InsightFace (buffalo_s, ONNX) + clustering | small | Fills the unused `face_names` field |
| RAG / reasoning / summaries | Ollama 7–8B quantized (llama3.1 / qwen2.5) | ~5–6 GB | One model, many jobs |

## Data model additions

- `enrichment_status(chunk_id, enricher, status, detail, updated_at)` — tracks per-source-chunk
  progress per enricher (`done` / `skipped` / `failed`). Drives incremental backfill and
  the `/enrich/status` surface.
- Derived text chunks are ordinary rows in `chunks` (carrying `derived_from`).
- **Phase 2** adds `faces(face_id, chunk_id, file_path, cluster_id, person_name, embedding…)`
  and a `people` table for named clusters.

## Phases

### Phase 0 — Enrichment framework  ✅ (this change)
- `app/enrich/`: `Enricher` ABC + `SourceChunk`/`EnrichmentOutput` (base.py), the batched
  `EnrichmentRunner` with GPU-yield gate (runner.py), the enricher `registry`, and the
  first enricher (`ocr.py`).
- `MetadataStore`: `enrichment_status` table, `source_chunks_needing_enrichment`,
  `mark_enrichment`, `enrichment_summary`.
- `EnrichmentConfig` + env flags; `lifelog enrich` CLI command; `enrich` optional extra.
- Tests with fake enrichers + a fake OCR backend (no model downloads required).

### Phase 1 — Content understanding (ingest-side, GPU)
Add enrichers, each a small subclass: `CaptionEnricher` (images + video frames),
`TagEnricher` (zero-shot CLIP labels), `ActionEnricher` (X-CLIP over scene clips).
Wire post-ingest auto-trigger and the background backfill schedule (APScheduler).
New API: `POST /enrich/trigger`, `GET /enrich/status`.

### Phase 2 — Faces & people
`FaceEnricher` (InsightFace detect+embed) → `faces` table; clustering job → clusters;
naming API + UI; names become searchable and populate `face_names`. Ties into the
existing `person_names` query signal.

### Phase 3 — RAG answers & reasoning
Extend `llm_client` beyond chit-chat with an `AnswerSynthesizer` (cited answers from
retrieved session cards) and a `QueryPlanner` (decompose → multi-retrieve → merge).
API `/query` returns an `answer` block; UI renders it above the cards. Auto
titles/summaries fill `SessionCard.title`/`summary` here (shared LLM plumbing).

### Phase 4 — Proactive & aware
Scheduled jobs: "on this day" resurfacing, daily/weekly digests, pattern insights.
New API surfaces + UI panels.

## Status

- Phase 0: implemented in this change (framework + OCR + tests).
- Phases 1–4: scoped above; each is additive on the Phase 0 framework.
