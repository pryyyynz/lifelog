"""Calendar event ingestion from ICS exports (Google Takeout, Apple, etc.)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timezone
from pathlib import Path
from typing import Any

from app.ingest.base import DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.file_ingestor import LocalFileIngestor
from app.models.contracts import NormalizedChunkRecord


@dataclass(frozen=True)
class CalendarEvent:
    uid: str
    summary: str
    description: str | None
    location: str | None
    start: datetime | None
    end: datetime | None


class GoogleCalendarIngestor(LocalFileIngestor):
    """Ingests ICS export files (Google Takeout, Apple Calendar, etc.)."""

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        events = _parse_ics(item.path)
        return ExtractedItem(
            discovered=item,
            payload=events,
            metadata={"event_count": len(events)},
        )

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        events = item.payload
        if not isinstance(events, list):
            return []
        records: list[NormalizedChunkRecord] = []
        for event in events:
            if not isinstance(event, CalendarEvent):
                continue
            identity = f"calendar:{event.uid}"
            parts = [event.summary]
            if event.description:
                parts.append(event.description)
            if event.location:
                parts.append(event.location)
            text = "\n".join(parts)
            metadata: dict[str, Any] = {
                "chunk_identity": identity,
                "uid": event.uid,
                "summary": event.summary,
                "location": event.location,
                "start": event.start.isoformat() if event.start else None,
                "end": event.end.isoformat() if event.end else None,
                # Calendar events are structured temporal anchors — no embedding text.
                "embedding_disabled": True,
            }
            records.append(
                NormalizedChunkRecord(
                    chunk_id=_chunk_id(item.discovered.path, identity),
                    source_type="calendar",
                    file_path=item.discovered.path,
                    text=text,
                    timestamp_utc=event.start,
                    vector_collection=None,  # Not embedded — used as filter/session anchor only.
                    metadata=metadata,
                )
            )
        return records


def _parse_ics(path: Path) -> list[CalendarEvent]:
    """Parse an ICS file using the icalendar library if available."""
    try:
        import icalendar  # type: ignore[import-untyped]
    except ImportError:
        return []

    try:
        raw = path.read_bytes()
        cal = icalendar.Calendar.from_ical(raw)
    except Exception:  # noqa: BLE001
        return []

    events: list[CalendarEvent] = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        uid = str(component.get("uid", ""))
        summary = str(component.get("summary", ""))
        if not summary:
            continue
        description = _ics_text(component.get("description"))
        location = _ics_text(component.get("location"))
        start = _ics_datetime(component.get("dtstart"))
        end = _ics_datetime(component.get("dtend"))
        events.append(
            CalendarEvent(
                uid=uid or _hash_text(summary + str(start)),
                summary=summary,
                description=description,
                location=location,
                start=start,
                end=end,
            )
        )
    return events


def _ics_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ics_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    dt = getattr(value, "dt", None)
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    return None


def _chunk_id(path: Path, identity: str) -> str:
    raw = f"{path!s}::{identity}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
