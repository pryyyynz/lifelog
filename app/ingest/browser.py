"""Chrome browsing history ingestor.

Chrome locks its History SQLite database while running, so this ingestor always
works on a *copy* of the file rather than the original.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.ingest.base import DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.file_ingestor import LocalFileIngestor
from app.ingest.text import prepare_embedding_text
from app.models.contracts import NormalizedChunkRecord

# Chrome timestamps are microseconds since 1601-01-01 00:00:00 UTC.
_CHROME_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)

# URL prefixes that belong to browser internals — never worth indexing.
_INTERNAL_PREFIXES = (
    "chrome://",
    "chrome-extension://",
    "about:",
    "edge://",
    "brave://",
)


@dataclass(frozen=True)
class BrowseVisit:
    url: str
    title: str
    visited_at: datetime | None


class ChromeHistoryIngestor(LocalFileIngestor):
    """Reads a *copy* of the Chrome History SQLite file and indexes page visits."""

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        visits = _read_history(item.path)
        return ExtractedItem(
            discovered=item,
            payload=visits,
            metadata={"visit_count": len(visits)},
        )

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        visits = item.payload
        if not isinstance(visits, list):
            return []
        records: list[NormalizedChunkRecord] = []
        for visit in visits:
            if not isinstance(visit, BrowseVisit):
                continue
            identity = f"browser:{_hash_url(visit.url)}"
            text = _visit_text(visit)
            metadata: dict[str, Any] = {
                "chunk_identity": identity,
                "url": visit.url,
                "title": visit.title,
                "raw_text": text,
                "embedding_text": prepare_embedding_text(text, model_name="intfloat/e5-large-v2"),
            }
            records.append(
                NormalizedChunkRecord(
                    chunk_id=_chunk_id(item.discovered.path, identity),
                    source_type="browser_history",
                    file_path=item.discovered.path,
                    text=text,
                    timestamp_utc=visit.visited_at,
                    vector_collection="text_chunks",
                    metadata=metadata,
                )
            )
        return records


def _read_history(db_path: Path) -> list[BrowseVisit]:
    """Copy the database to a temp file then query it.

    Copying is necessary because Chrome holds a write lock on the live file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        copy_path = Path(tmpdir) / "History.copy"
        try:
            shutil.copy2(db_path, copy_path)
        except Exception:  # noqa: BLE001
            return []

        try:
            connection = sqlite3.connect(f"file:{copy_path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    urls.url,
                    urls.title,
                    MAX(visits.visit_time) AS last_visit_time
                FROM urls
                JOIN visits ON urls.id = visits.url
                GROUP BY urls.id
                ORDER BY last_visit_time DESC
                """
            ).fetchall()
            connection.close()
        except Exception:  # noqa: BLE001
            return []

        visits: list[BrowseVisit] = []
        for row in rows:
            url = str(row["url"] or "")
            if _is_internal_url(url):
                continue
            title = str(row["title"] or "").strip()
            visited_at = _chrome_timestamp(row["last_visit_time"])
            visits.append(BrowseVisit(url=url, title=title, visited_at=visited_at))

        return visits


def _chrome_timestamp(microseconds: int | None) -> datetime | None:
    """Convert Chrome's microseconds-since-1601 format to a UTC datetime."""
    if not microseconds:
        return None
    try:
        return _CHROME_EPOCH + timedelta(microseconds=int(microseconds))
    except (OverflowError, ValueError):
        return None


def _is_internal_url(url: str) -> bool:
    return any(url.startswith(prefix) for prefix in _INTERNAL_PREFIXES)


def _visit_text(visit: BrowseVisit) -> str:
    if visit.title:
        return f"{visit.title}\n{visit.url}"
    return visit.url


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _chunk_id(path: Path, identity: str) -> str:
    raw = f"{path!s}::{identity}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def default_chrome_history_path() -> Path | None:
    """Return the platform-specific default Chrome History file path, if it exists."""
    candidates: list[Path] = []
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidates.append(
                Path(local_app_data) / "Google" / "Chrome" / "User Data" / "Default" / "History"
            )
    elif os.uname().sysname == "Darwin":
        candidates.append(
            Path.home()
            / "Library"
            / "Application Support"
            / "Google"
            / "Chrome"
            / "Default"
            / "History"
        )
    else:
        candidates.append(
            Path.home() / ".config" / "google-chrome" / "Default" / "History"
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
