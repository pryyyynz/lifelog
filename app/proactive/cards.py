"""Adapter from stored chunk rows to RetrievalHits for proactive grouping."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.contracts import RetrievalHit

_SNIPPET_CHARS = 300


def row_to_hit(row: Any, *, rationale: str) -> RetrievalHit:
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    ts = row["timestamp_utc"]
    text = row["text"] or row["search_text"] or ""
    return RetrievalHit(
        chunk_id=str(row["chunk_id"]),
        source_type=str(row["source_type"]),  # type: ignore[arg-type]
        file_path=Path(str(row["file_path"])),
        score=1.0,
        rationale=[rationale],
        timestamp_utc=datetime.fromisoformat(ts) if ts else None,
        session_id=row["session_id"],
        snippet=text[:_SNIPPET_CHARS] if text else None,
        place_name=row["place_name"],
        metadata=metadata,
    )
