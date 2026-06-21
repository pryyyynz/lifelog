# Product Scope

## v1.0 Scope

The v1.0 target is a single-user, local-only multimodal personal search system. It stores and queries personal data on the user's machine and exposes three local surfaces:

- REST API for query, status, and ingest operations.
- CLI for setup validation, ingest, query, status, deletion, and consistency checks.
- Minimal chat-style web UI for natural-language queries and grouped session cards.

Supported v1.0 modalities are text, email, photos, audio, video, calendar exports, and browser history exports.

After setup, normal operation must not require third-party APIs. Initial setup may download model weights and packages from upstream registries. Optional enrichment that can call a network service, such as Nominatim reverse geocoding, must be configurable and must be disabled by offline mode.

## Success Criteria

- OBJ-01 unified retrieval: a seeded test query returns grouped results spanning at least three modalities.
- OBJ-02 privacy-first execution: query-time inference, SQLite metadata, vector storage, logs, and UI/API traffic remain local.
- OBJ-03 fusion: at least one end-to-end test returns a single session card containing multimodal results fused from independent retrieval paths.
- OBJ-04 temporal and spatial awareness: date and place hints affect filtering or ranking in repeatable tests.
- OBJ-05 incremental ingest: incremental runs process only new or changed inputs and leave unchanged inputs untouched.
- OBJ-06 conversational refinement: a follow-up query can reuse prior result context without restating the full original query.
- OBJ-07 explainability: each result exposes source type, timestamp, path, and retrieval rationale.
- OBJ-08 extensibility: a new modality can be added by implementing one ingestor class without rewriting retrieval logic.

## Not in v1.0

- Multi-device sync, cloud deployment, shared multi-user indexing, and authentication.
- Mobile apps.
- Live streaming ingest.
- Social media ingestion.
- Fine-tuning or personalization by training on the user's data.
- In-UI media playback; the app links users to native apps or source files.

