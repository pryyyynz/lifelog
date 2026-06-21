"""Shared ingestion contracts for modality-specific processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.ingest.registry import SourceConfig
from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore

if TYPE_CHECKING:
    from app.storage.vector_store import VectorStore


@dataclass(frozen=True)
class DiscoveredItem:
    source: SourceConfig
    path: Path
    mtime_ns: int
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedItem:
    discovered: DiscoveredItem
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestContext:
    mode: str
    run_id: int
    started_at: datetime
    store: MetadataStore
    full: bool = False
    vector_store: VectorStore | None = None


class BaseIngestor(ABC):
    """Interface implemented by all modality-specific ingestors."""

    def __init__(self, source: SourceConfig) -> None:
        self.source = source

    @abstractmethod
    def discover(self, context: IngestContext) -> list[DiscoveredItem]:
        raise NotImplementedError

    @abstractmethod
    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        raise NotImplementedError

    @abstractmethod
    def chunk(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        raise NotImplementedError

    @abstractmethod
    def embed(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        raise NotImplementedError

    @abstractmethod
    def persist(
        self, item: DiscoveredItem, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, context: IngestContext) -> None:
        raise NotImplementedError
