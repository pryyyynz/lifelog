"""Session assignment for ingest-time chunk grouping.

Events within ``window_hours`` of the previous event (by timestamp) are grouped
into the same session.  A new session starts when the gap exceeds the window or
when no timestamp is available.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import datetime

from app.models.contracts import NormalizedChunkRecord


class SessionAssigner:
    """Assigns deterministic ``session_id`` values to a list of chunk records.

    Algorithm
    ---------
    1. Separate records with and without timestamps.
    2. Sort timestamped records in ascending order.
    3. Walk the sorted list: start a new session whenever the gap to the
       previous event exceeds ``window_hours``, or on the very first event.
    4. Records without timestamps are returned unchanged (``session_id=None``
       unless already set).

    The session ID is a 16-hex-char SHA-1 digest of ``<start_iso>:<seed>``
    where ``seed`` is the ``chunk_id`` of the first event in the session.
    This makes it deterministic for the same input data, reproducible across
    incremental ingest runs.
    """

    DEFAULT_WINDOW_HOURS: float = 4.0

    def __init__(self, window_hours: float = DEFAULT_WINDOW_HOURS) -> None:
        self._window_seconds = window_hours * 3600

    @classmethod
    def from_environment(cls) -> SessionAssigner:
        """Build from ``LIFELOG_SESSION_WINDOW_HOURS`` env var (default 4.0)."""
        hours = float(os.getenv("LIFELOG_SESSION_WINDOW_HOURS", str(cls.DEFAULT_WINDOW_HOURS)))
        return cls(window_hours=hours)

    def assign(self, records: list[NormalizedChunkRecord]) -> list[NormalizedChunkRecord]:
        """Return a new list with ``session_id`` filled in for all records.

        Records that already have a ``session_id`` are left unchanged.
        Records without a ``timestamp_utc`` receive ``session_id=None``.
        """
        if not records:
            return records

        timestamped: list[tuple[int, NormalizedChunkRecord]] = []
        no_ts: list[tuple[int, NormalizedChunkRecord]] = []

        for i, record in enumerate(records):
            if record.timestamp_utc is not None:
                timestamped.append((i, record))
            else:
                no_ts.append((i, record))

        # Sort by timestamp ascending, preserving original index for output
        timestamped.sort(key=lambda pair: pair[1].timestamp_utc)  # type: ignore[arg-type]

        current_session_id: str | None = None
        last_ts: datetime | None = None

        result: list[NormalizedChunkRecord | None] = [None] * len(records)

        for orig_idx, record in timestamped:
            # Don't override an explicitly assigned session_id
            if record.session_id is not None:
                result[orig_idx] = record
                last_ts = record.timestamp_utc
                continue

            ts = record.timestamp_utc
            assert ts is not None  # already filtered above

            gap = (ts - last_ts).total_seconds() if last_ts is not None else None
            if gap is None or gap > self._window_seconds:
                # Start a new session
                current_session_id = _make_session_id(ts, record.chunk_id)

            last_ts = ts
            result[orig_idx] = _with_session_id(record, current_session_id)

        for orig_idx, record in no_ts:
            result[orig_idx] = record  # unchanged

        return [r for r in result if r is not None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_id(start: datetime, seed: str) -> str:
    """Deterministic 16-hex-char session ID."""
    key = f"{start.isoformat()}:{seed}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _with_session_id(record: NormalizedChunkRecord, session_id: str | None) -> NormalizedChunkRecord:
    """Return a copy of *record* with ``session_id`` set (frozen dataclass)."""
    return NormalizedChunkRecord(
        chunk_id=record.chunk_id,
        source_type=record.source_type,
        file_path=record.file_path,
        text=record.text,
        timestamp_utc=record.timestamp_utc,
        vector_collection=record.vector_collection,
        vector_id=record.vector_id,
        session_id=session_id,
        timestamp_start_sec=record.timestamp_start_sec,
        timestamp_end_sec=record.timestamp_end_sec,
        lat=record.lat,
        lon=record.lon,
        place_name=record.place_name,
        metadata=record.metadata,
    )
