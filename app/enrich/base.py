"""Core abstractions for the AI enrichment framework.

An :class:`Enricher` turns an already-ingested *source chunk* (a photo, a video
scene frame, an audio transcript, ...) into *derived* records — most commonly
derived text chunks that flow through the existing e5 + BM25 + cross-encoder
retrieval pipeline. See ``docs/AI_ENRICHMENT_PLAN.md`` for the architecture.

Enrichers degrade gracefully: when a model or dependency is missing,
``is_available()`` returns ``False`` and the runner skips the enricher without
crashing, mirroring the embedder/transcription status pattern elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.ingest.text import prepare_embedding_text
from app.models.contracts import FaceRecord, NormalizedChunkRecord

# Status values written to ``enrichment_status``.
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class SourceChunk:
    """A parsed row from the ``chunks`` table, the input to an enricher."""

    chunk_id: str
    source_id: str
    source_type: str
    file_path: Path
    chunk_identity: str
    timestamp_utc: datetime | None
    session_id: str | None
    lat: float | None
    lon: float | None
    metadata: dict[str, Any]
    timestamp_start_sec: float | None = None

    @classmethod
    def from_row(cls, row: Any) -> SourceChunk:
        keys = row.keys()
        raw_meta = row["metadata_json"] if "metadata_json" in keys else None
        metadata = json.loads(raw_meta) if raw_meta else {}
        ts = row["timestamp_utc"]
        return cls(
            chunk_id=str(row["chunk_id"]),
            source_id=str(row["source_id"]),
            source_type=str(row["source_type"]),
            file_path=Path(str(row["file_path"])),
            chunk_identity=str(row["chunk_identity"]),
            timestamp_utc=datetime.fromisoformat(ts) if ts else None,
            session_id=row["session_id"],
            lat=row["lat"],
            lon=row["lon"],
            metadata=metadata,
            timestamp_start_sec=row["timestamp_start_sec"] if "timestamp_start_sec" in keys else None,
        )


@dataclass(frozen=True)
class EnrichmentOutput:
    """Result of enriching one source chunk.

    ``records`` are derived text chunks to embed + persist; ``faces`` are detected
    face records persisted to the faces table. Both are empty for non-``done``.
    """

    status: str
    records: tuple[NormalizedChunkRecord, ...] = ()
    detail: str | None = None
    faces: tuple[FaceRecord, ...] = ()


class Enricher(ABC):
    """Interface implemented by every enrichment capability."""

    #: Stable identifier used as the ``enrichment_status.enricher`` key and the
    #: derived ``chunk_identity`` prefix. Lowercase, no colons.
    name: str = "enricher"
    #: Source types this enricher consumes (e.g. ``("photo",)``).
    source_types: tuple[str, ...] = ()

    def is_available(self) -> bool:
        """Whether the underlying model/dependency is present. Default: yes."""
        return True

    @abstractmethod
    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        """Produce derived records for ``chunk`` (or a non-``done`` status)."""
        raise NotImplementedError


def resolve_image_path(chunk: SourceChunk) -> Path | None:
    """Return the image to analyze for vision enrichers.

    Photos analyze their own file; video frame chunks analyze the extracted scene
    frame (``metadata.frame_path``). Video transcript chunks have no frame and
    return ``None`` so vision enrichers skip them.
    """
    if chunk.source_type == "video":
        frame_path = chunk.metadata.get("frame_path")
        return Path(str(frame_path)) if frame_path else None
    return chunk.file_path


def derived_suffix(chunk: SourceChunk) -> str:
    """Per-chunk identity suffix, unique across video frames that share a file path."""
    scene_id = chunk.metadata.get("scene_id")
    return str(scene_id) if scene_id else "0"


def derived_text_record(
    parent: SourceChunk,
    *,
    enricher_name: str,
    suffix: str,
    text: str,
    embedding_model: str = "intfloat/e5-large-v2",
    extra_metadata: dict[str, Any] | None = None,
) -> NormalizedChunkRecord:
    """Build a derived text chunk from a parent source chunk.

    The record targets the ``text_chunks`` collection and carries
    ``embedding_text`` so the existing text embedder picks it up, plus
    ``derived_from`` so it is never itself re-enriched.
    """
    identity = f"{enricher_name}:{suffix}"
    chunk_id = hashlib.sha256(f"{parent.file_path}::{identity}".encode()).hexdigest()[:24]
    metadata: dict[str, Any] = {
        "chunk_identity": identity,
        "derived_from": parent.chunk_id,
        "enricher": enricher_name,
        "raw_text": text,
        "embedding_text": prepare_embedding_text(text, model_name=embedding_model),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return NormalizedChunkRecord(
        chunk_id=chunk_id,
        source_type=parent.source_type,  # type: ignore[arg-type]
        file_path=parent.file_path,
        text=text,
        timestamp_utc=parent.timestamp_utc,
        vector_collection="text_chunks",
        session_id=parent.session_id,
        lat=parent.lat,
        lon=parent.lon,
        metadata=metadata,
    )
