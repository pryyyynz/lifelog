"""Generic local-file ingestor used until source-specific pipelines are added."""

from __future__ import annotations

from pathlib import Path

from app.ingest.base import BaseIngestor, DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.registry import SourceConfig
from app.models.contracts import NormalizedChunkRecord


class LocalFileIngestor(BaseIngestor):
    """Discovers supported files and records file-level ingest metadata."""

    def __init__(self, source: SourceConfig) -> None:
        super().__init__(source)
        self._seen_paths: set[Path] = set()

    def discover(self, context: IngestContext) -> list[DiscoveredItem]:
        paths = [self.source.path] if self.source.path.is_file() else list(self.source.path.rglob("*"))
        discovered: list[DiscoveredItem] = []
        for path in sorted(paths):
            suffix = path.suffix.lower()
            suffix_supported = suffix in self.source.supported_extensions or (
                suffix == "" and "" in self.source.supported_extensions
            )
            if not path.is_file() or not suffix_supported:
                continue
            stat = path.stat()
            self._seen_paths.add(path.resolve())
            discovered.append(
                DiscoveredItem(
                    source=self.source,
                    path=path.resolve(),
                    mtime_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                    metadata={"suffix": path.suffix.lower()},
                )
            )
        return discovered

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        return ExtractedItem(discovered=item, metadata=item.metadata)

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        return []

    def chunk(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        return records

    def embed(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        return records

    def persist(
        self, item: DiscoveredItem, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> None:
        if context.vector_store is not None:
            context.vector_store.delete_by_file_path(str(item.path))
        context.store.delete_chunks_for_file(item.path)
        context.store.upsert_chunks(self.source.id, records)
        if context.vector_store is not None:
            id_mapping = context.vector_store.upsert_records(records)
            context.store.update_vector_ids(id_mapping)
        context.store.upsert_file_state(
            self.source.id,
            item.path,
            mtime_ns=item.mtime_ns,
            size_bytes=item.size_bytes,
            metadata=item.metadata,
        )

    def cleanup(self, context: IngestContext) -> None:
        return None
