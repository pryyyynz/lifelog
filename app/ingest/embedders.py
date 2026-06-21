"""Shared embedding utilities for text and image modalities.

These adapters are optional — each returns None or passes records through
unchanged when the underlying model or environment flag is absent.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.models.contracts import NormalizedChunkRecord

logger = logging.getLogger(__name__)

_TEXT_EMBED_ENV = "LIFELOG_ENABLE_TEXT_EMBEDDING"


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers to produce dense text vectors at ingest time.

    Disabled by default; set ``LIFELOG_ENABLE_TEXT_EMBEDDING=1`` to activate.
    """

    def __init__(self, enabled: bool, model_name: str) -> None:
        self._enabled = enabled
        self._model_name = model_name
        self._model: Any = None
        self.status = "disabled"
        if enabled:
            self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self._model_name)
            self.status = "ok"
            logger.debug("SentenceTransformer loaded: %s", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SentenceTransformer unavailable (%s): %s", self._model_name, exc)
            self.status = "unavailable"

    @classmethod
    def from_environment(cls) -> SentenceTransformerEmbedder:
        enabled = os.getenv(_TEXT_EMBED_ENV, "").lower() in {"1", "true", "yes"}
        model_name = os.getenv("LIFELOG_TEXT_EMBEDDING_MODEL", "intfloat/e5-large-v2")
        return cls(enabled=enabled, model_name=model_name)

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch of texts. Returns None when disabled or unavailable."""
        if not self._enabled or self._model is None:
            return None
        try:
            import numpy as np  # noqa: PLC0415

            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            # Convert numpy arrays to plain Python lists for JSON serialisation
            if hasattr(vecs, "tolist"):
                return vecs.tolist()
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Text embedding failed: %s", exc)
            return None


def embed_text_records(
    records: list[NormalizedChunkRecord],
    embedder: SentenceTransformerEmbedder | None = None,
) -> list[NormalizedChunkRecord]:
    """Apply text embeddings to records that carry ``embedding_text`` in metadata.

    Records without ``embedding_text`` are returned unchanged.
    When the embedder is disabled, records are returned unchanged but
    ``text_embedding_status`` is written so callers can observe the skip.
    """
    if embedder is None:
        embedder = SentenceTransformerEmbedder.from_environment()

    # Collect indices and texts to embed
    indices: list[int] = []
    texts: list[str] = []
    for i, record in enumerate(records):
        et = record.metadata.get("embedding_text")
        if et:
            indices.append(i)
            texts.append(str(et))

    if not indices:
        return records

    vectors = embedder.embed_batch(texts)

    result = list(records)
    for batch_pos, record_idx in enumerate(indices):
        record = records[record_idx]
        metadata = dict(record.metadata)
        if vectors is not None:
            metadata["text_embedding"] = vectors[batch_pos]
            metadata["text_embedding_status"] = "ok"
        else:
            metadata["text_embedding_status"] = embedder.status
        result[record_idx] = NormalizedChunkRecord(
            chunk_id=record.chunk_id,
            source_type=record.source_type,
            file_path=record.file_path,
            text=record.text,
            timestamp_utc=record.timestamp_utc,
            vector_collection=record.vector_collection,
            vector_id=record.vector_id,
            session_id=record.session_id,
            timestamp_start_sec=record.timestamp_start_sec,
            timestamp_end_sec=record.timestamp_end_sec,
            lat=record.lat,
            lon=record.lon,
            place_name=record.place_name,
            metadata=metadata,
        )
    return result
