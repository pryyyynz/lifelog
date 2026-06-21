# Architecture

## System Flow

Raw sources flow through source connectors, modality processors, embeddings, and storage. Retrieval runs query analysis, parallel retrieval, fusion, re-ranking, session grouping, and result rendering.

```text
source connectors -> processors -> embeddings -> Qdrant vector index
                                      -> SQLite metadata
query -> query analysis -> retrieval -> fusion -> re-rank -> session grouping -> result rendering
```

## Layer Boundaries

- Ingestion discovers source items, extracts raw content, normalizes records, chunks content, calls embedding services, and persists data.
- Storage owns SQLite metadata, Qdrant vector collections, source registry state, ingest run records, and consistency checks.
- Retrieval owns query analysis, dense search, sparse or exact search, metadata filters, and modality-specific query variants.
- Ranking owns reciprocal-rank fusion, temporal boosts, cross-encoder re-ranking, and explainability metadata.
- Presentation owns REST schemas, CLI output, and chat-style UI session cards.

## Runtime Choices

- API framework: FastAPI.
- Task orchestration: APScheduler for v1.0's lightweight local scheduled jobs.
- File watching: Watchdog, with polling left as a fallback if a platform proves unreliable.
- Vector store: Qdrant server mode through Docker Compose.
- Metadata store: SQLite in a user-configurable local path.
- Development OS support: Windows, macOS, and Linux.

Docker Qdrant is the default because it matches the spec's dense, sparse, filtering, and RRF goals while keeping vector data local and reproducible. Tests may use in-memory or tiny local fakes where that keeps setup fast, but production runtime targets Qdrant.

## Data Contracts

`NormalizedChunkRecord` is the shared ingest-to-storage contract. It carries chunk identity, source type, file path, text or media reference metadata, timestamps, coordinates, session ID, vector collection, vector ID, and modality-specific metadata.

`RetrievalHit` is the ranking contract before grouping. It carries chunk identity, source type, file path, score, rationale, timestamp, session ID, snippet or thumbnail, place name, and extra metadata.

`SessionCard` is the presentation contract after grouping. It carries session ID, aggregate score, hit list, optional time bounds, title, and summary.

