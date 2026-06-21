"""Source registry and onboarding validation for local ingest sources."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class SourceKind(StrEnum):
    TEXT = "text"
    EMAIL = "email"
    PHOTOS = "photos"
    AUDIO = "audio"
    VIDEO = "video"
    CALENDAR = "calendar"
    BROWSER_HISTORY = "browser_history"


class SourceMode(StrEnum):
    PATH = "path"
    EXPORT = "export"


class IngestStrategy(StrEnum):
    FILE_MTIME = "file_mtime"
    EXPORT_SCAN = "export_scan"
    SQLITE_COPY = "sqlite_copy"


SUPPORTED_EXTENSIONS: dict[SourceKind, tuple[str, ...]] = {
    SourceKind.TEXT: (".md", ".markdown", ".txt", ".csv", ".json"),
    SourceKind.EMAIL: (".mbox",),
    SourceKind.PHOTOS: (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif"),
    SourceKind.AUDIO: (".m4a", ".mp3", ".wav"),
    SourceKind.VIDEO: (".mp4", ".mov"),
    SourceKind.CALENDAR: (".ics",),
    SourceKind.BROWSER_HISTORY: (".sqlite", ".sqlite3", ".db", ""),
}

DEFAULT_SOURCE_MODES: dict[SourceKind, SourceMode] = {
    SourceKind.TEXT: SourceMode.PATH,
    SourceKind.EMAIL: SourceMode.EXPORT,
    SourceKind.PHOTOS: SourceMode.PATH,
    SourceKind.AUDIO: SourceMode.PATH,
    SourceKind.VIDEO: SourceMode.PATH,
    SourceKind.CALENDAR: SourceMode.EXPORT,
    SourceKind.BROWSER_HISTORY: SourceMode.EXPORT,
}

DEFAULT_STRATEGIES: dict[SourceKind, IngestStrategy] = {
    SourceKind.BROWSER_HISTORY: IngestStrategy.SQLITE_COPY,
}

SETUP_GUIDANCE: dict[SourceKind, str] = {
    SourceKind.TEXT: "Choose an Obsidian vault or folder containing Markdown, plain text, or journal exports.",
    SourceKind.EMAIL: "Export mail as MBOX and choose the folder or .mbox file.",
    SourceKind.PHOTOS: "Choose a photo folder, Apple Photos export, or Google Photos Takeout folder.",
    SourceKind.AUDIO: "Choose a folder containing M4A, MP3, or WAV recordings.",
    SourceKind.VIDEO: "Choose a folder containing MP4 or MOV videos.",
    SourceKind.CALENDAR: "Export calendar data as .ics and choose the file or containing folder.",
    SourceKind.BROWSER_HISTORY: "Copy the Chrome History SQLite database first, then choose the copy.",
}


@dataclass(frozen=True)
class SourceValidation:
    ok: bool
    item_count: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceConfig:
    id: str
    source_type: SourceKind
    path: Path
    mode: SourceMode
    enabled: bool = True
    ingest_strategy: IngestStrategy = IngestStrategy.FILE_MTIME
    last_scan_time: datetime | None = None
    supported_extensions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = self.source_type.value
        data["path"] = str(self.path)
        data["mode"] = self.mode.value
        data["ingest_strategy"] = self.ingest_strategy.value
        data["last_scan_time"] = self.last_scan_time.isoformat() if self.last_scan_time else None
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SourceConfig:
        raw_last_scan = data.get("last_scan_time")
        last_scan = datetime.fromisoformat(raw_last_scan) if raw_last_scan else None
        return cls(
            id=str(data["id"]),
            source_type=SourceKind(data["source_type"]),
            path=Path(data["path"]).expanduser().resolve(),
            mode=SourceMode(data["mode"]),
            enabled=bool(data.get("enabled", True)),
            ingest_strategy=IngestStrategy(data.get("ingest_strategy", IngestStrategy.FILE_MTIME)),
            last_scan_time=last_scan,
            supported_extensions=tuple(data.get("supported_extensions", ())),
            metadata=dict(data.get("metadata", {})),
        )


class SourceRegistry:
    """JSON-backed registry for configured personal data sources."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.sources: list[SourceConfig] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.sources = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.sources = [SourceConfig.from_json(item) for item in payload.get("sources", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "sources": [source.to_json() for source in self.sources]}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]

    def upsert(self, source: SourceConfig) -> None:
        self.sources = [existing for existing in self.sources if existing.id != source.id]
        self.sources.append(source)
        self.sources.sort(key=lambda item: item.id)

    def get(self, source_id: str) -> SourceConfig | None:
        return next((source for source in self.sources if source.id == source_id), None)

    def mark_scanned(self, source_id: str, scanned_at: datetime | None = None) -> None:
        timestamp = scanned_at or datetime.now(UTC)
        self.sources = [
            source if source.id != source_id else _replace_source_last_scan(source, timestamp)
            for source in self.sources
        ]


def build_source_config(source_type: SourceKind, path: Path) -> SourceConfig:
    resolved = path.expanduser().resolve()
    mode = DEFAULT_SOURCE_MODES[source_type]
    strategy = DEFAULT_STRATEGIES.get(
        source_type,
        IngestStrategy.EXPORT_SCAN if mode == SourceMode.EXPORT else IngestStrategy.FILE_MTIME,
    )
    return SourceConfig(
        id=f"{source_type.value}:{_slug_path(resolved)}",
        source_type=source_type,
        path=resolved,
        mode=mode,
        ingest_strategy=strategy,
        supported_extensions=SUPPORTED_EXTENSIONS[source_type],
    )


def validate_source(source: SourceConfig) -> SourceValidation:
    errors: list[str] = []
    warnings: list[str] = []
    path = source.path
    if not path.exists():
        return SourceValidation(False, errors=(f"missing path: {path}",))

    if not os.access(path, os.R_OK):
        return SourceValidation(False, errors=(f"permission denied: {path}",))

    if path.is_file():
        if not _extension_supported(path, source.supported_extensions):
            errors.append(f"unsupported file format: {path.suffix or '<no extension>'}")
        if _looks_locked(path):
            errors.append(f"locked or unreadable file: {path}")
        return SourceValidation(not errors, item_count=0 if errors else 1, errors=tuple(errors))

    if not path.is_dir():
        return SourceValidation(False, errors=(f"not a file or directory: {path}",))

    count = 0
    unreadable = 0
    for item in path.rglob("*"):
        if item.is_dir():
            continue
        if not _extension_supported(item, source.supported_extensions):
            continue
        if not os.access(item, os.R_OK) or _looks_locked(item):
            unreadable += 1
            continue
        count += 1

    if count == 0:
        warnings.append(
            "no supported files found; expected "
            + ", ".join(ext or "<no extension>" for ext in source.supported_extensions)
        )
    if unreadable:
        warnings.append(f"{unreadable} supported files were unreadable or locked")
    return SourceValidation(True, item_count=count, warnings=tuple(warnings))


def _replace_source_last_scan(source: SourceConfig, timestamp: datetime) -> SourceConfig:
    return SourceConfig(
        id=source.id,
        source_type=source.source_type,
        path=source.path,
        mode=source.mode,
        enabled=source.enabled,
        ingest_strategy=source.ingest_strategy,
        last_scan_time=timestamp,
        supported_extensions=source.supported_extensions,
        metadata=source.metadata,
    )


def _slug_path(path: Path) -> str:
    slug = "_".join(part for part in path.parts if part not in {path.anchor, "\\", "/"})
    return "".join(char.lower() if char.isalnum() else "-" for char in slug).strip("-")


def _extension_supported(path: Path, extensions: tuple[str, ...]) -> bool:
    if "" in extensions and path.suffix == "":
        return True
    return path.suffix.lower() in extensions


def _looks_locked(path: Path) -> bool:
    try:
        with path.open("rb"):
            return False
    except OSError:
        return True
