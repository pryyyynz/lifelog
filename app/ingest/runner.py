"""Ingest execution model with checkpointing and incremental change detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.ingest.audio import VoiceMemoIngestor
from app.ingest.base import BaseIngestor, IngestContext
from app.ingest.browser import ChromeHistoryIngestor
from app.ingest.calendar import GoogleCalendarIngestor
from app.ingest.file_ingestor import LocalFileIngestor
from app.ingest.images import FilesystemPhotoIngestor
from app.ingest.registry import SourceConfig, SourceKind, SourceRegistry
from app.ingest.session import SessionAssigner
from app.ingest.text import EmailIngestor, TextSourceIngestor
from app.ingest.video import VideoIngestor
from app.storage.metadata import MetadataStore

if TYPE_CHECKING:
    from app.storage.vector_store import VectorStore


@dataclass(frozen=True)
class IngestRunSummary:
    run_id: int
    mode: str
    processed_items: int
    skipped_items: int
    failed_items: int
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class IngestRunner:
    def __init__(
        self,
        registry: SourceRegistry,
        store: MetadataStore,
        vector_store: VectorStore | None = None,
        session_assigner: SessionAssigner | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.vector_store = vector_store
        self.session_assigner = session_assigner or SessionAssigner.from_environment()
        if vector_store is not None:
            vector_store.ensure_collections()

    def run(self, *, full: bool, source_id: str | None = None) -> IngestRunSummary:
        mode = "full" if full else "incremental"
        resume_previous = full and self.store.has_unfinished_run(mode)
        started_at = datetime.now(UTC)
        run_id = self.store.start_run(mode)
        context = IngestContext(
            mode=mode,
            run_id=run_id,
            started_at=started_at,
            store=self.store,
            full=full,
            vector_store=self.vector_store,
        )
        processed = 0
        skipped = 0
        failed = 0

        sources = self.registry.enabled_sources()
        if source_id:
            sources = [source for source in sources if source.id == source_id]

        for source in sources:
            ingestor = self._ingestor_for(source)
            try:
                items = ingestor.discover(context)
            except Exception as exc:  # noqa: BLE001 - errors are recorded and ingest continues.
                failed += 1
                self.store.record_error(run_id, source.id, None, str(exc))
                continue

            for item in items:
                try:
                    previous = self.store.get_file_state(source.id, item.path)
                    unchanged = (
                        previous is not None
                        and previous.mtime_ns == item.mtime_ns
                        and previous.size_bytes == item.size_bytes
                    )
                    if (not full or resume_previous) and unchanged:
                        skipped += 1
                        continue

                    extracted = ingestor.extract(item, context)
                    normalized = ingestor.normalize(extracted, context)
                    chunks = ingestor.chunk(normalized, context)
                    sessioned = self.session_assigner.assign(chunks)
                    embedded = ingestor.embed(sessioned, context)
                    ingestor.persist(item, embedded, context)
                    self.store.checkpoint(source.id, item.path)
                    processed += 1
                except Exception as exc:  # noqa: BLE001 - per-item failure isolation is intentional.
                    failed += 1
                    self.store.record_error(run_id, source.id, item.path, str(exc))

            try:
                ingestor.cleanup(context)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self.store.record_error(run_id, source.id, None, f"cleanup failed: {exc}")
            self.registry.mark_scanned(source.id)

        self.registry.save()
        self.store.finish_run(
            run_id,
            processed=processed,
            skipped=skipped,
            failed=failed,
            started_at=started_at,
        )
        finished_at = datetime.now(UTC)
        return IngestRunSummary(
            run_id=run_id,
            mode=mode,
            processed_items=processed,
            skipped_items=skipped,
            failed_items=failed,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _ingestor_for(self, source: SourceConfig) -> BaseIngestor:
        if source.source_type == SourceKind.TEXT:
            return TextSourceIngestor(source)
        if source.source_type == SourceKind.EMAIL:
            return EmailIngestor(source)
        if source.source_type == SourceKind.PHOTOS:
            return FilesystemPhotoIngestor(source)
        if source.source_type == SourceKind.AUDIO:
            return VoiceMemoIngestor(source)
        if source.source_type == SourceKind.VIDEO:
            return VideoIngestor(source)
        if source.source_type == SourceKind.CALENDAR:
            return GoogleCalendarIngestor(source)
        if source.source_type == SourceKind.BROWSER_HISTORY:
            return ChromeHistoryIngestor(source)
        return LocalFileIngestor(source)
