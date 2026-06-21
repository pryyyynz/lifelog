"""SQLite metadata store for source files, chunks, and ingest runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.contracts import FaceRecord, NormalizedChunkRecord


@dataclass(frozen=True)
class StoredFileState:
    source_id: str
    file_path: Path
    mtime_ns: int
    size_bytes: int
    last_ingested_at: datetime


class MetadataStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path.expanduser().resolve()
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ingest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_seconds REAL,
                    processed_items INTEGER NOT NULL DEFAULT 0,
                    skipped_items INTEGER NOT NULL DEFAULT 0,
                    failed_items INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS ingest_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    source_id TEXT NOT NULL,
                    file_path TEXT,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES ingest_runs(id)
                );

                CREATE TABLE IF NOT EXISTS ingest_checkpoints (
                    source_id TEXT PRIMARY KEY,
                    last_file_path TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_files (
                    source_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    last_ingested_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(source_id, file_path)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_identity TEXT NOT NULL,
                    text TEXT,
                    search_text TEXT,
                    timestamp_utc TEXT,
                    timestamp_start_sec REAL,
                    timestamp_end_sec REAL,
                    lat REAL,
                    lon REAL,
                    place_name TEXT,
                    session_id TEXT,
                    vector_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(file_path, chunk_identity)
                );

                CREATE TABLE IF NOT EXISTS enrichment_status (
                    chunk_id TEXT NOT NULL,
                    enricher TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, enricher)
                );

                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    timestamp_utc TEXT,
                    bbox_json TEXT,
                    det_score REAL,
                    embedding_json TEXT NOT NULL,
                    cluster_id TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS face_clusters (
                    cluster_id TEXT PRIMARY KEY,
                    person_name TEXT,
                    centroid_json TEXT NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS proactive_cache (
                    kind TEXT NOT NULL,
                    period_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(kind, period_key)
                );
                """
            )
            _ensure_column(connection, "chunks", "search_text", "TEXT")
            _ensure_column(connection, "chunks", "timestamp_utc", "TEXT")
            _ensure_column(connection, "chunks", "timestamp_start_sec", "REAL")
            _ensure_column(connection, "chunks", "timestamp_end_sec", "REAL")
            _ensure_column(connection, "chunks", "lat", "REAL")
            _ensure_column(connection, "chunks", "lon", "REAL")
            _ensure_column(connection, "chunks", "place_name", "TEXT")
            _ensure_column(connection, "chunks", "session_id", "TEXT")
            _ensure_column(connection, "chunks", "vector_id", "TEXT")

    def start_run(self, mode: str) -> int:
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO ingest_runs (mode, started_at) VALUES (?, ?)",
                (mode, now.isoformat()),
            )
            return int(cursor.lastrowid)

    def has_unfinished_run(self, mode: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM ingest_runs
                WHERE mode = ? AND finished_at IS NULL
                LIMIT 1
                """,
                (mode,),
            ).fetchone()
        return row is not None

    def finish_run(
        self,
        run_id: int,
        *,
        processed: int,
        skipped: int,
        failed: int,
        started_at: datetime,
    ) -> None:
        finished_at = _utc_now()
        duration = (finished_at - started_at).total_seconds()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_runs
                SET finished_at = ?, duration_seconds = ?, processed_items = ?,
                    skipped_items = ?, failed_items = ?
                WHERE id = ?
                """,
                (finished_at.isoformat(), duration, processed, skipped, failed, run_id),
            )

    def record_error(self, run_id: int, source_id: str, file_path: Path | None, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_errors (run_id, source_id, file_path, error, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, source_id, str(file_path) if file_path else None, error, _utc_now().isoformat()),
            )

    def checkpoint(self, source_id: str, file_path: Path) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_checkpoints (source_id, last_file_path, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    last_file_path = excluded.last_file_path,
                    updated_at = excluded.updated_at
                """,
                (source_id, str(file_path), _utc_now().isoformat()),
            )

    def get_file_state(self, source_id: str, file_path: Path) -> StoredFileState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_files WHERE source_id = ? AND file_path = ?",
                (source_id, str(file_path)),
            ).fetchone()
        if row is None:
            return None
        return StoredFileState(
            source_id=str(row["source_id"]),
            file_path=Path(row["file_path"]),
            mtime_ns=int(row["mtime_ns"]),
            size_bytes=int(row["size_bytes"]),
            last_ingested_at=datetime.fromisoformat(row["last_ingested_at"]),
        )

    def upsert_file_state(
        self,
        source_id: str,
        file_path: Path,
        *,
        mtime_ns: int,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_files
                    (source_id, file_path, mtime_ns, size_bytes, last_ingested_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, file_path) DO UPDATE SET
                    mtime_ns = excluded.mtime_ns,
                    size_bytes = excluded.size_bytes,
                    last_ingested_at = excluded.last_ingested_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    source_id,
                    str(file_path),
                    mtime_ns,
                    size_bytes,
                    _utc_now().isoformat(),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )

    def delete_chunks_for_file(self, file_path: Path) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM chunks WHERE file_path = ?", (str(file_path),))
            return int(cursor.rowcount)

    def delete_chunks_by_source_id(self, source_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            return int(cursor.rowcount)

    def delete_source_files_by_source_id(self, source_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM source_files WHERE source_id = ?", (source_id,))
            return int(cursor.rowcount)

    def file_paths_for_source(self, source_id: str) -> list[Path]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT file_path FROM chunks WHERE source_id = ?", (source_id,)
            ).fetchall()
        return [Path(str(row["file_path"])) for row in rows]

    def upsert_chunks(self, source_id: str, records: list[NormalizedChunkRecord]) -> None:
        if not records:
            return
        now = _utc_now().isoformat()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, source_id, source_type, file_path, chunk_identity, text,
                    search_text, timestamp_utc, timestamp_start_sec, timestamp_end_sec,
                    lat, lon, place_name, session_id, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path, chunk_identity) DO UPDATE SET
                    chunk_id = excluded.chunk_id,
                    source_id = excluded.source_id,
                    source_type = excluded.source_type,
                    text = excluded.text,
                    search_text = excluded.search_text,
                    timestamp_utc = excluded.timestamp_utc,
                    timestamp_start_sec = excluded.timestamp_start_sec,
                    timestamp_end_sec = excluded.timestamp_end_sec,
                    lat = excluded.lat,
                    lon = excluded.lon,
                    place_name = excluded.place_name,
                    session_id = excluded.session_id,
                    metadata_json = excluded.metadata_json
                """,
                [
                    (
                        record.chunk_id,
                        source_id,
                        record.source_type,
                        str(record.file_path),
                        str(record.metadata.get("chunk_identity", record.chunk_id)),
                        record.text,
                        record.text,
                        record.timestamp_utc.isoformat() if record.timestamp_utc else None,
                        record.timestamp_start_sec,
                        record.timestamp_end_sec,
                        record.lat,
                        record.lon,
                        record.place_name,
                        record.session_id,
                        json.dumps(_strip_embedding_vectors(record.metadata), sort_keys=True),
                        now,
                    )
                    for record in records
                ],
            )

    def source_chunks_needing_enrichment(
        self,
        enricher: str,
        source_types: list[str],
        *,
        limit: int = 100,
        include_failed: bool = False,
    ) -> list[sqlite3.Row]:
        """Return original (non-derived) chunks of given source types not yet enriched.

        A chunk is "needing" when it has no ``enrichment_status`` row for ``enricher``
        with a terminal status. Derived chunks (those carrying ``derived_from`` in
        metadata) are excluded so enrichment never feeds on its own output.
        """
        if not source_types:
            return []
        type_ph = ",".join("?" * len(source_types))
        skip_statuses = ["done", "skipped"] if include_failed else ["done", "skipped", "failed"]
        skip_ph = ",".join("?" * len(skip_statuses))
        sql = f"""
            SELECT * FROM chunks
            WHERE source_type IN ({type_ph})
              AND metadata_json NOT LIKE '%"derived_from"%'
              AND chunk_id NOT IN (
                  SELECT chunk_id FROM enrichment_status
                  WHERE enricher = ? AND status IN ({skip_ph})
              )
            ORDER BY created_at
            LIMIT ?
        """
        params = [*source_types, enricher, *skip_statuses, limit]
        with self._connect() as connection:
            return list(connection.execute(sql, params).fetchall())

    def mark_enrichment(
        self, chunk_id: str, enricher: str, status: str, detail: str | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO enrichment_status (chunk_id, enricher, status, detail, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, enricher) DO UPDATE SET
                    status = excluded.status,
                    detail = excluded.detail,
                    updated_at = excluded.updated_at
                """,
                (chunk_id, enricher, status, detail, _utc_now().isoformat()),
            )

    def enrichment_summary(self) -> dict[str, dict[str, int]]:
        """Return {enricher: {status: count}} across all tracked enrichment."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT enricher, status, COUNT(*) AS count"
                " FROM enrichment_status GROUP BY enricher, status"
            ).fetchall()
        summary: dict[str, dict[str, int]] = {}
        for row in rows:
            summary.setdefault(str(row["enricher"]), {})[str(row["status"])] = int(row["count"])
        return summary

    # ----- Faces (Phase 2) ---------------------------------------------------

    def upsert_faces(self, faces: list[FaceRecord]) -> None:
        if not faces:
            return
        now = _utc_now().isoformat()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO faces (
                    face_id, chunk_id, source_id, source_type, file_path,
                    timestamp_utc, bbox_json, det_score, embedding_json, cluster_id, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(face_id) DO UPDATE SET
                    chunk_id = excluded.chunk_id,
                    bbox_json = excluded.bbox_json,
                    det_score = excluded.det_score,
                    embedding_json = excluded.embedding_json,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        f.face_id,
                        f.chunk_id,
                        f.source_id,
                        f.source_type,
                        str(f.file_path),
                        f.timestamp_utc.isoformat() if f.timestamp_utc else None,
                        json.dumps(list(f.bbox)),
                        f.det_score,
                        json.dumps(f.embedding),
                        None,
                        now,
                    )
                    for f in faces
                ],
            )

    def faces_without_cluster(self, limit: int = 500) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM faces WHERE cluster_id IS NULL ORDER BY updated_at LIMIT ?",
                    (limit,),
                ).fetchall()
            )

    def assign_face_to_cluster(self, face_id: str, cluster_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE faces SET cluster_id = ? WHERE face_id = ?", (cluster_id, face_id)
            )

    def upsert_cluster(
        self, cluster_id: str, centroid: list[float], face_count: int
    ) -> None:
        """Insert/update a cluster centroid. Does not touch ``person_name``."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO face_clusters (cluster_id, person_name, centroid_json, face_count, updated_at)
                VALUES (?, NULL, ?, ?, ?)
                ON CONFLICT(cluster_id) DO UPDATE SET
                    centroid_json = excluded.centroid_json,
                    face_count = excluded.face_count,
                    updated_at = excluded.updated_at
                """,
                (cluster_id, json.dumps(centroid), face_count, _utc_now().isoformat()),
            )

    def name_cluster(self, cluster_id: str, person_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE face_clusters SET person_name = ?, updated_at = ? WHERE cluster_id = ?",
                (person_name, _utc_now().isoformat(), cluster_id),
            )

    def get_clusters(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM face_clusters ORDER BY face_count DESC, cluster_id"
                ).fetchall()
            )

    def get_cluster(self, cluster_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM face_clusters WHERE cluster_id = ?", (cluster_id,)
            ).fetchone()

    def faces_for_cluster(self, cluster_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM faces WHERE cluster_id = ? ORDER BY det_score DESC", (cluster_id,)
                ).fetchall()
            )

    def faces_summary(self) -> dict[str, int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS c FROM faces").fetchone()["c"]
            clustered = connection.execute(
                "SELECT COUNT(*) AS c FROM faces WHERE cluster_id IS NOT NULL"
            ).fetchone()["c"]
            clusters = connection.execute("SELECT COUNT(*) AS c FROM face_clusters").fetchone()["c"]
            named = connection.execute(
                "SELECT COUNT(*) AS c FROM face_clusters WHERE person_name IS NOT NULL"
            ).fetchone()["c"]
        return {
            "faces": int(total),
            "clustered": int(clustered),
            "clusters": int(clusters),
            "named_clusters": int(named),
        }

    # ----- Proactive (Phase 4) ----------------------------------------------

    def chunks_in_window(
        self, start_iso: str, end_iso: str, *, exclude_derived: bool = False, limit: int = 1000
    ) -> list[sqlite3.Row]:
        """Chunks with a timestamp in [start, end). ISO8601 strings sort correctly."""
        derived_clause = " AND metadata_json NOT LIKE '%\"derived_from\"%'" if exclude_derived else ""
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM chunks WHERE timestamp_utc IS NOT NULL"
                    " AND timestamp_utc >= ? AND timestamp_utc < ?" + derived_clause
                    + " ORDER BY timestamp_utc LIMIT ?",
                    (start_iso, end_iso, limit),
                ).fetchall()
            )

    def chunks_on_month_day(
        self, month: str, day: str, *, exclude_derived: bool = True, limit: int = 500
    ) -> list[sqlite3.Row]:
        """Chunks whose timestamp falls on the given month/day in any year.

        ``month``/``day`` are zero-padded strings; relies on ISO ``YYYY-MM-DD`` layout.
        """
        derived_clause = " AND metadata_json NOT LIKE '%\"derived_from\"%'" if exclude_derived else ""
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM chunks WHERE timestamp_utc IS NOT NULL"
                    " AND substr(timestamp_utc, 6, 2) = ? AND substr(timestamp_utc, 9, 2) = ?"
                    + derived_clause
                    + " ORDER BY timestamp_utc DESC LIMIT ?",
                    (month, day, limit),
                ).fetchall()
            )

    def get_proactive(self, kind: str, period_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM proactive_cache WHERE kind = ? AND period_key = ?",
                (kind, period_key),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def set_proactive(self, kind: str, period_key: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO proactive_cache (kind, period_key, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(kind, period_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (kind, period_key, json.dumps(payload), _utc_now().isoformat()),
            )

    def chunk_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])

    def fetch_chunks(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(connection.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall())

    def fetch_chunks_by_ids(self, chunk_ids: set[str]) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        with self._connect() as connection:
            return list(
                connection.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                    list(chunk_ids),
                ).fetchall()
            )

    def fetch_chunks_by_session(self, session_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM chunks WHERE session_id = ? ORDER BY timestamp_utc",
                    (session_id,),
                ).fetchall()
            )

    def update_vector_ids(self, mapping: dict[str, str]) -> None:
        """Update vector_id for multiple chunks after Qdrant upsert."""
        if not mapping:
            return
        with self._connect() as connection:
            connection.executemany(
                "UPDATE chunks SET vector_id = ? WHERE chunk_id = ?",
                [(vid, cid) for cid, vid in mapping.items()],
            )

    def chunks_missing_vector_id(self, collection: str | None = None) -> list[sqlite3.Row]:
        """Return chunks that have a vector_collection but no vector_id yet."""
        with self._connect() as connection:
            if collection:
                return list(
                    connection.execute(
                        "SELECT * FROM chunks WHERE vector_id IS NULL"
                        " AND metadata_json LIKE '%\"vector_collection\"%'"
                        " AND metadata_json LIKE ?",
                        (f'%"{collection}"%',),
                    ).fetchall()
                )
            return list(
                connection.execute(
                    "SELECT * FROM chunks WHERE vector_id IS NULL"
                    " AND metadata_json LIKE '%\"vector_collection\"%'"
                ).fetchall()
            )

    def all_chunk_ids_with_vector_id(self) -> dict[str, str]:
        """Return {chunk_id: vector_id} for all chunks that have been vectorised."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, vector_id FROM chunks WHERE vector_id IS NOT NULL"
            ).fetchall()
        return {str(row["chunk_id"]): str(row["vector_id"]) for row in rows}

    def chunk_counts_by_source_type(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_type, COUNT(*) AS count FROM chunks GROUP BY source_type"
            ).fetchall()
        return {str(row["source_type"]): int(row["count"]) for row in rows}

    def file_counts_by_source_type(self) -> dict[str, int]:
        """Count distinct files (not chunks) per source_type.

        A video is many chunks (keyframes + transcript segments) but one file, so
        chunk counts overstate it; the UI shows these item counts instead.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_type, COUNT(DISTINCT file_path) AS count"
                " FROM chunks GROUP BY source_type"
            ).fetchall()
        return {str(row["source_type"]): int(row["count"]) for row in rows}

    def latest_ingest_timestamp(self) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(finished_at) AS finished_at FROM ingest_runs WHERE finished_at IS NOT NULL"
            ).fetchone()
        if row is None or row["finished_at"] is None:
            return None
        return datetime.fromisoformat(row["finished_at"])

    def status_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_id, COUNT(*) AS count FROM source_files GROUP BY source_id"
            ).fetchall()
        return {str(row["source_id"]): int(row["count"]) for row in rows}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _strip_embedding_vectors(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove embedding vector lists from metadata before writing to SQLite."""
    return {k: v for k, v in metadata.items() if not (k.endswith("_embedding") and isinstance(v, list))}
