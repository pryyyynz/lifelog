"""Batched enrichment runner with a GPU-yield gate.

Walks un-enriched source chunks for each enricher, runs the enricher (one model
resident at a time), then embeds + persists the derived text chunks and records
per-chunk status. A ``should_pause`` callable lets the API pause background
enrichment while a user query is using the GPU.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from app.enrich.base import STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, Enricher, EnrichmentOutput, SourceChunk
from app.ingest.embedders import SentenceTransformerEmbedder, embed_text_records
from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentSummary:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    unavailable: list[str] = field(default_factory=list)
    paused: bool = False


class EnrichmentRunner:
    def __init__(
        self,
        store: MetadataStore,
        enrichers: list[Enricher],
        *,
        embedder: SentenceTransformerEmbedder | None = None,
        batch_size: int = 32,
        should_pause: Callable[[], bool] | None = None,
    ) -> None:
        self.store = store
        self.enrichers = enrichers
        self.embedder = embedder
        self.batch_size = max(1, batch_size)
        self.should_pause = should_pause or (lambda: False)

    def run(self, *, limit: int | None = None, include_failed: bool = False) -> EnrichmentSummary:
        summary = EnrichmentSummary()
        for enricher in self.enrichers:
            if not enricher.is_available():
                summary.unavailable.append(enricher.name)
                logger.info("Enricher '%s' unavailable (model/deps missing) — skipping", enricher.name)
                continue
            self._run_enricher(enricher, limit, include_failed, summary)
            if summary.paused:
                break
        return summary

    def _run_enricher(
        self,
        enricher: Enricher,
        limit: int | None,
        include_failed: bool,
        summary: EnrichmentSummary,
    ) -> None:
        remaining = limit
        while remaining is None or remaining > 0:
            batch_limit = self.batch_size if remaining is None else min(self.batch_size, remaining)
            rows = self.store.source_chunks_needing_enrichment(
                enricher.name,
                list(enricher.source_types),
                limit=batch_limit,
                include_failed=include_failed,
            )
            if not rows:
                break

            results: list[tuple[SourceChunk, EnrichmentOutput]] = []
            for row in rows:
                if self.should_pause():
                    summary.paused = True
                    break
                chunk = SourceChunk.from_row(row)
                try:
                    output = enricher.enrich(chunk)
                except Exception as exc:  # noqa: BLE001 - isolate per-item failures
                    logger.warning("Enricher '%s' failed on %s: %s", enricher.name, chunk.chunk_id, exc)
                    output = EnrichmentOutput(STATUS_FAILED, detail=str(exc))
                results.append((chunk, output))

            # Persist derived records before marking done, so a persist failure
            # leaves the chunk un-marked and it retries next run.
            derived = [
                (chunk.source_id, record)
                for chunk, output in results
                if output.status == STATUS_DONE
                for record in output.records
            ]
            if derived:
                self._persist(derived)

            faces = [
                face
                for _chunk, output in results
                if output.status == STATUS_DONE
                for face in output.faces
            ]
            if faces:
                self.store.upsert_faces(faces)

            for chunk, output in results:
                self.store.mark_enrichment(chunk.chunk_id, enricher.name, output.status, output.detail)
                if output.status == STATUS_DONE:
                    summary.done += 1
                elif output.status == STATUS_SKIPPED:
                    summary.skipped += 1
                else:
                    summary.failed += 1

            if remaining is not None:
                remaining -= len(results)
            if summary.paused or len(rows) < batch_limit:
                break

    def _persist(self, derived: list[tuple[str, NormalizedChunkRecord]]) -> None:
        by_source: dict[str, list[NormalizedChunkRecord]] = defaultdict(list)
        for source_id, record in derived:
            by_source[source_id].append(record)
        for source_id, records in by_source.items():
            to_write = embed_text_records(records, self.embedder) if self.embedder else records
            self.store.upsert_chunks(source_id, to_write)
