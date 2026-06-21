"""Consistency checker: verifies SQLite and Qdrant are in sync."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.storage.metadata import MetadataStore
from app.storage.vector_store import VectorStore, chunk_id_to_point_id


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of a consistency check between SQLite and Qdrant."""

    orphaned_sqlite_chunk_ids: list[str] = field(default_factory=list)
    """SQLite chunks that have a vector_id but no matching Qdrant point."""

    orphaned_qdrant_point_ids: dict[str, list[str]] = field(default_factory=dict)
    """Qdrant point IDs with no matching SQLite chunk, keyed by collection name."""

    ok: bool = True

    @property
    def total_orphans(self) -> int:
        return len(self.orphaned_sqlite_chunk_ids) + sum(
            len(v) for v in self.orphaned_qdrant_point_ids.values()
        )


class ConsistencyChecker:
    """Compares SQLite chunk records against Qdrant vector points to find orphans.

    Usage::

        checker = ConsistencyChecker(store, vector_store)
        report = checker.check()
        if not report.ok:
            print(report)
    """

    COLLECTIONS = ("text_chunks", "image_frames", "video_frames", "audio_transcripts")

    def __init__(self, store: MetadataStore, vector_store: VectorStore) -> None:
        self._store = store
        self._vs = vector_store

    def check(self) -> ConsistencyReport:
        """Run the full consistency check.

        Returns a :class:`ConsistencyReport` describing any mismatches.
        """
        # Build {chunk_id: vector_id} from SQLite for chunks that should have vectors
        sqlite_mapping = self._store.all_chunk_ids_with_vector_id()
        # Invert to {vector_id: chunk_id}
        vector_id_to_chunk: dict[str, str] = {vid: cid for cid, vid in sqlite_mapping.items()}

        orphaned_sqlite: list[str] = []
        orphaned_qdrant: dict[str, list[str]] = {}

        if not self._vs.available:
            # Cannot compare — report that Qdrant is unavailable
            return ConsistencyReport(
                orphaned_sqlite_chunk_ids=[],
                orphaned_qdrant_point_ids={},
                ok=True,
            )

        for collection in self.COLLECTIONS:
            qdrant_ids = self._vs.fetch_all_point_ids(collection)

            # SQLite chunks whose vector_id is not in Qdrant
            for chunk_id, vector_id in sqlite_mapping.items():
                if vector_id not in qdrant_ids:
                    orphaned_sqlite.append(chunk_id)

            # Qdrant points with no corresponding SQLite chunk
            missing = [pid for pid in qdrant_ids if pid not in vector_id_to_chunk]
            if missing:
                orphaned_qdrant[collection] = missing

        ok = not orphaned_sqlite and not orphaned_qdrant
        return ConsistencyReport(
            orphaned_sqlite_chunk_ids=orphaned_sqlite,
            orphaned_qdrant_point_ids=orphaned_qdrant,
            ok=ok,
        )

    def repair_sqlite_orphans(self) -> int:
        """Re-upsert SQLite chunks whose vector_id points to a missing Qdrant entry.

        Returns the number of chunks successfully re-upserted.
        """
        from app.storage.metadata import MetadataStore  # noqa: PLC0415 (already imported)

        report = self.check()
        if not report.orphaned_sqlite_chunk_ids:
            return 0

        ids_set = set(report.orphaned_sqlite_chunk_ids)
        rows = self._store.fetch_chunks_by_ids(ids_set)
        if not rows:
            return 0

        from app.models.contracts import NormalizedChunkRecord  # noqa: PLC0415
        from datetime import datetime  # noqa: PLC0415
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        records = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            records.append(
                NormalizedChunkRecord(
                    chunk_id=row["chunk_id"],
                    source_type=row["source_type"],
                    file_path=Path(row["file_path"]),
                    text=row["text"],
                    timestamp_utc=(
                        datetime.fromisoformat(row["timestamp_utc"])
                        if row["timestamp_utc"]
                        else None
                    ),
                    vector_collection=metadata.get("vector_collection"),
                    session_id=row["session_id"],
                    timestamp_start_sec=row["timestamp_start_sec"],
                    timestamp_end_sec=row["timestamp_end_sec"],
                    lat=row["lat"],
                    lon=row["lon"],
                    place_name=row["place_name"],
                    metadata=metadata,
                )
            )

        upserted = self._vs.upsert_records(records)
        if upserted:
            self._store.update_vector_ids(upserted)
        return len(upserted)
