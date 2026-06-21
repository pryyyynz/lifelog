"""Tests for calendar, geocoding, and browser history ingest pipelines (Section 10)."""

from __future__ import annotations

import json
import sqlite3
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.ingest.browser import (
    ChromeHistoryIngestor,
    _chrome_timestamp,
    _is_internal_url,
    default_chrome_history_path,
)
from app.ingest.calendar import GoogleCalendarIngestor, _parse_ics
from app.ingest.geocoding import ReverseGeocoder, _format_place
from app.ingest.registry import SourceKind, SourceRegistry, build_source_config
from app.ingest.runner import IngestRunner
from app.storage.metadata import MetadataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_ICS = """\
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:event-001@test
SUMMARY:Team lunch
DESCRIPTION:Post-sprint lunch with the team
LOCATION:The Canteen
DTSTART:20261005T120000Z
DTEND:20261005T130000Z
END:VEVENT
BEGIN:VEVENT
UID:event-002@test
SUMMARY:Doctor appointment
DTSTART:20261010T090000Z
DTEND:20261010T093000Z
END:VEVENT
END:VCALENDAR
"""

_SAMPLE_ICS_ALL_DAY = """\
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:allday-001@test
SUMMARY:Holiday
DTSTART;VALUE=DATE:20261225
DTEND;VALUE=DATE:20261226
END:VEVENT
END:VCALENDAR
"""


def _build_chrome_db(path: Path, visits: list[tuple[str, str, datetime | None]]) -> None:
    """Create a minimal Chrome History SQLite database at *path*."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT, title TEXT, visit_count INTEGER)"
    )
    con.execute(
        "CREATE TABLE visits (id INTEGER PRIMARY KEY, url INTEGER, visit_time INTEGER)"
    )
    for url_id, (url, title, dt) in enumerate(visits, start=1):
        con.execute("INSERT INTO urls VALUES (?, ?, ?, 1)", (url_id, url, title))
        if dt:
            chrome_ts = int((dt - datetime(1601, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000)
        else:
            chrome_ts = 0
        con.execute("INSERT INTO visits VALUES (?, ?, ?)", (url_id, url_id, chrome_ts))
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# 10.1  Calendar ingest
# ---------------------------------------------------------------------------


def test_parse_ics_returns_events() -> None:
    events = _parse_ics_from_string(_SAMPLE_ICS)
    assert len(events) == 2
    summaries = {e.summary for e in events}
    assert "Team lunch" in summaries
    assert "Doctor appointment" in summaries


def test_parse_ics_extracts_location_and_description() -> None:
    events = _parse_ics_from_string(_SAMPLE_ICS)
    lunch = next(e for e in events if e.summary == "Team lunch")
    assert lunch.location == "The Canteen"
    assert lunch.description == "Post-sprint lunch with the team"


def test_parse_ics_extracts_timestamps() -> None:
    events = _parse_ics_from_string(_SAMPLE_ICS)
    lunch = next(e for e in events if e.summary == "Team lunch")
    assert lunch.start is not None
    assert lunch.start.year == 2026
    assert lunch.start.month == 10
    assert lunch.start.tzinfo is not None


def test_parse_ics_handles_all_day_event() -> None:
    events = _parse_ics_from_string(_SAMPLE_ICS_ALL_DAY)
    assert len(events) == 1
    assert events[0].start is not None
    assert events[0].start.tzinfo is not None


def test_parse_ics_returns_empty_on_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.ics"
    bad.write_bytes(b"\x00\xff\xfe")
    assert _parse_ics(bad) == []


def test_calendar_ingest_no_embedding(tmp_path: Path) -> None:
    """Calendar events must not be embedded (vector_collection = None)."""
    ics_path = tmp_path / "cal.ics"
    ics_path.write_text(_SAMPLE_ICS, encoding="utf-8")

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.CALENDAR, ics_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert len(chunks) == 2
    for chunk in chunks:
        meta = json.loads(chunk["metadata_json"])
        assert meta.get("embedding_disabled") is True
        assert chunk["source_type"] == "calendar"


def test_calendar_ingest_stores_timestamps(tmp_path: Path) -> None:
    ics_path = tmp_path / "cal.ics"
    ics_path.write_text(_SAMPLE_ICS, encoding="utf-8")

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.CALENDAR, ics_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    lunch_chunk = next(c for c in chunks if "Team lunch" in c["text"])
    assert lunch_chunk["timestamp_utc"] is not None
    assert "2026-10-05" in lunch_chunk["timestamp_utc"]


def test_calendar_ingest_text_includes_location(tmp_path: Path) -> None:
    ics_path = tmp_path / "cal.ics"
    ics_path.write_text(_SAMPLE_ICS, encoding="utf-8")

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.CALENDAR, ics_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    lunch_chunk = next(c for c in chunks if "Team lunch" in c["text"])
    assert "The Canteen" in lunch_chunk["text"]


# ---------------------------------------------------------------------------
# 10.2  Geocoding
# ---------------------------------------------------------------------------


def test_format_place_city_and_country() -> None:
    address = {"city": "Accra", "country": "Ghana"}
    assert _format_place(address) == "Accra, Ghana"


def test_format_place_town_fallback() -> None:
    address = {"town": "Aburi", "country": "Ghana"}
    assert _format_place(address) == "Aburi, Ghana"


def test_format_place_empty_returns_none() -> None:
    assert _format_place({}) is None


def test_reverse_geocoder_disabled_returns_none() -> None:
    geocoder = ReverseGeocoder(cache_path=Path("/nonexistent/cache.json"), enabled=False)
    assert geocoder.lookup(5.55, -0.2) is None


def test_reverse_geocoder_caches_result(tmp_path: Path) -> None:
    cache_path = tmp_path / "geocache.json"
    geocoder = ReverseGeocoder(cache_path=cache_path, enabled=True)
    # Inject directly into the cache to avoid real network calls.
    key = geocoder._cache_key(5.55, -0.2)
    geocoder._cache[key] = "Accra, Ghana"

    result = geocoder.lookup(5.55, -0.2)
    assert result == "Accra, Ghana"


def test_reverse_geocoder_persists_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "geocache.json"
    geocoder = ReverseGeocoder(cache_path=cache_path, enabled=True)
    key = geocoder._cache_key(48.85, 2.35)
    geocoder._cache[key] = "Paris, France"
    geocoder._flush()

    # Reload from disk.
    geocoder2 = ReverseGeocoder(cache_path=cache_path, enabled=True)
    assert geocoder2._cache.get(key) == "Paris, France"


def test_reverse_geocoder_same_bucket_for_nearby_coords(tmp_path: Path) -> None:
    cache_path = tmp_path / "geocache.json"
    geocoder = ReverseGeocoder(cache_path=cache_path, enabled=True)
    key1 = geocoder._cache_key(48.85, 2.35)
    key2 = geocoder._cache_key(48.87, 2.37)
    # Both should snap to the same 2-degree bucket.
    assert key1 == key2


# ---------------------------------------------------------------------------
# 10.3  Chrome history ingest
# ---------------------------------------------------------------------------


def test_chrome_timestamp_conversion() -> None:
    # 2026-01-01 00:00:00 UTC
    dt_expected = datetime(2026, 1, 1, tzinfo=UTC)
    microseconds = int((dt_expected - datetime(1601, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000)
    assert _chrome_timestamp(microseconds) == dt_expected


def test_chrome_timestamp_zero_returns_none() -> None:
    assert _chrome_timestamp(0) is None


def test_chrome_timestamp_none_returns_none() -> None:
    assert _chrome_timestamp(None) is None


def test_is_internal_url_filters_chrome_urls() -> None:
    assert _is_internal_url("chrome://settings") is True
    assert _is_internal_url("chrome-extension://abc/popup.html") is True
    assert _is_internal_url("about:blank") is True
    assert _is_internal_url("https://example.com") is False


def test_chrome_history_ingest_reads_visits(tmp_path: Path) -> None:
    db_path = tmp_path / "History"
    dt = datetime(2026, 3, 15, 14, 0, 0, tzinfo=UTC)
    _build_chrome_db(db_path, [("https://example.com/article", "Example Article", dt)])

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.BROWSER_HISTORY, db_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert len(chunks) == 1
    assert chunks[0]["source_type"] == "browser_history"
    assert "Example Article" in chunks[0]["text"]
    assert "https://example.com/article" in chunks[0]["text"]


def test_chrome_history_ingest_filters_internal_urls(tmp_path: Path) -> None:
    db_path = tmp_path / "History"
    dt = datetime(2026, 3, 15, tzinfo=UTC)
    _build_chrome_db(db_path, [
        ("chrome://settings", "Settings", dt),
        ("https://github.com", "GitHub", dt),
        ("about:blank", "Blank", dt),
    ])

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.BROWSER_HISTORY, db_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert len(chunks) == 1
    assert "github.com" in chunks[0]["text"]


def test_chrome_history_ingest_embedding_prefix(tmp_path: Path) -> None:
    db_path = tmp_path / "History"
    dt = datetime(2026, 3, 15, tzinfo=UTC)
    _build_chrome_db(db_path, [("https://example.com", "Example", dt)])

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.BROWSER_HISTORY, db_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    meta = json.loads(chunks[0]["metadata_json"])
    assert meta["embedding_text"].startswith("passage:")


def test_chrome_history_ingest_stores_timestamp(tmp_path: Path) -> None:
    db_path = tmp_path / "History"
    dt = datetime(2026, 5, 10, 9, 30, 0, tzinfo=UTC)
    _build_chrome_db(db_path, [("https://example.com", "Example", dt)])

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.BROWSER_HISTORY, db_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert chunks[0]["timestamp_utc"] is not None
    assert "2026-05-10" in chunks[0]["timestamp_utc"]


def test_chrome_history_ingest_handles_corrupt_db(tmp_path: Path) -> None:
    """A corrupt database file must not crash the runner."""
    db_path = tmp_path / "History"
    db_path.write_bytes(b"\x00" * 128)

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.BROWSER_HISTORY, db_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    summary = IngestRunner(registry, store).run(full=True)
    assert summary.processed_items + summary.failed_items == 1


def test_chrome_history_incremental_skips_unchanged(tmp_path: Path) -> None:
    db_path = tmp_path / "History"
    dt = datetime(2026, 3, 15, tzinfo=UTC)
    _build_chrome_db(db_path, [("https://example.com", "Example", dt)])

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.BROWSER_HISTORY, db_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")
    runner = IngestRunner(registry, store)

    first = runner.run(full=True)
    second = runner.run(full=False)

    assert first.processed_items == 1
    assert second.processed_items == 0
    assert second.skipped_items == 1


# ---------------------------------------------------------------------------
# Helper: parse ICS from in-memory string
# ---------------------------------------------------------------------------


def _parse_ics_from_string(content: str):
    """Write content to a temp file and parse it."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ics", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    from app.ingest.calendar import _parse_ics

    try:
        return _parse_ics(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
