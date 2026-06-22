"""One-time repair: embed text + transcript chunks that were ingested while
LIFELOG_ENABLE_TEXT_EMBEDDING was off, and upsert them into Qdrant so the dense
(e5) semantic search arm works. Idempotent: skips chunks that already have a
vector_id. Read the chunk's own `embedding_text` (already 'passage:'-prefixed)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ["LIFELOG_ENABLE_TEXT_EMBEDDING"] = "1"

from app.config import get_config
from app.ingest.embedders import SentenceTransformerEmbedder, embed_text_records
from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore
from app.storage.vector_store import VectorStore


def _collection_for(row) -> str | None:
    ident = (row["chunk_identity"] or "")
    if ident.startswith(("video_transcript", "audio")) or row["source_type"] == "audio":
        return "audio_transcripts"
    # Text docs AND derived text (OCR/caption/tags on photos/videos) → text_chunks.
    # The caller only passes rows that carry embedding_text, so this is safe.
    return "text_chunks"


def main() -> int:
    cfg = get_config()
    store = MetadataStore(cfg.paths.sqlite_path)
    vs = VectorStore.from_environment()
    if not vs.available:
        print("Qdrant unavailable — start it first (docker compose up -d qdrant).")
        return 1
    vs.ensure_collections()

    records: list[NormalizedChunkRecord] = []
    for row in store.fetch_chunks():
        if row["vector_id"]:
            continue
        meta = json.loads(row["metadata_json"] or "{}")
        if not meta.get("embedding_text"):
            continue
        collection = _collection_for(row)
        if collection is None:
            continue
        ts = None
        if row["timestamp_utc"]:
            try:
                ts = datetime.fromisoformat(str(row["timestamp_utc"]))
            except ValueError:
                ts = None
        records.append(
            NormalizedChunkRecord(
                chunk_id=str(row["chunk_id"]),
                source_type=str(row["source_type"]),
                file_path=Path(str(row["file_path"])),
                text=row["text"],
                timestamp_utc=ts,
                vector_collection=collection,
                session_id=row["session_id"],
                timestamp_start_sec=row["timestamp_start_sec"],
                timestamp_end_sec=row["timestamp_end_sec"],
                lat=row["lat"],
                lon=row["lon"],
                place_name=row["place_name"],
                metadata=meta,
            )
        )

    if not records:
        print("Nothing to backfill — all text/transcript chunks already embedded.")
        return 0

    by_col: dict[str, int] = {}
    for r in records:
        by_col[r.vector_collection] = by_col.get(r.vector_collection, 0) + 1
    print(f"Embedding {len(records)} chunks: {by_col}")

    embedder = SentenceTransformerEmbedder(enabled=True, model_name=cfg.models.text_embedding_model)
    if embedder.status != "ok":
        print(f"Embedder not ready: status={embedder.status}")
        return 1
    embedded = embed_text_records(records, embedder=embedder)
    upserted = vs.upsert_records(embedded)
    store.update_vector_ids(upserted)
    print(f"Upserted {len(upserted)} vectors and set vector_ids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
