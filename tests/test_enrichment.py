"""Tests for the Phase 0 enrichment framework and the OCR enricher."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.enrich.base import (
    STATUS_DONE,
    STATUS_SKIPPED,
    Enricher,
    EnrichmentOutput,
    SourceChunk,
    derived_text_record,
)
from app.enrich.runner import EnrichmentRunner
from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(tmp_path / "test.db")


def _seed_photos(store: MetadataStore, n: int = 1, source_id: str = "src_photos") -> None:
    records = [
        NormalizedChunkRecord(
            chunk_id=f"photo{i}",
            source_type="photo",
            file_path=Path(f"/photos/img{i}.jpg"),
            text=None,
            timestamp_utc=_T0,
            vector_collection="image_frames",
            metadata={"chunk_identity": f"photo:{i}"},
        )
        for i in range(n)
    ]
    store.upsert_chunks(source_id, records)


class _FakeEnricher(Enricher):
    name = "fake"
    source_types = ("photo",)

    def __init__(self, *, text="hello world", available=True, raises=False, status=STATUS_DONE):
        self._text = text
        self._available = available
        self._raises = raises
        self._status = status
        self.calls = 0

    def is_available(self) -> bool:
        return self._available

    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        if self._status != STATUS_DONE:
            return EnrichmentOutput(self._status, detail="nope")
        record = derived_text_record(chunk, enricher_name=self.name, suffix="text", text=self._text)
        return EnrichmentOutput(STATUS_DONE, records=(record,))


# ---------------------------------------------------------------------------
# Runner behavior
# ---------------------------------------------------------------------------


class TestEnrichmentRunner:
    def test_creates_derived_text_chunks(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 2)
        summary = EnrichmentRunner(store, [_FakeEnricher(text="invoice total 42")]).run()
        assert summary.done == 2
        texts = [r["text"] for r in store.fetch_chunks()]
        assert texts.count("invoice total 42") == 2
        assert store.enrichment_summary()["fake"]["done"] == 2

    def test_idempotent_second_run_does_nothing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 1)
        first = _FakeEnricher()
        EnrichmentRunner(store, [first]).run()
        assert first.calls == 1
        second = _FakeEnricher()
        summary = EnrichmentRunner(store, [second]).run()
        assert second.calls == 0
        assert summary.done == 0

    def test_derived_chunks_are_not_re_enriched(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 1)
        EnrichmentRunner(store, [_FakeEnricher()]).run()
        # A different enricher should still only see the original photo, not the derived chunk.
        rows = store.source_chunks_needing_enrichment("other", ["photo"])
        identities = [r["chunk_identity"] for r in rows]
        assert "photo:0" in identities
        assert "fake:text" not in identities

    def test_failed_excluded_then_retried_with_flag(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 1)
        s1 = EnrichmentRunner(store, [_FakeEnricher(raises=True)]).run()
        assert s1.failed == 1
        assert store.enrichment_summary()["fake"]["failed"] == 1
        # Default run skips failed items.
        s2 = EnrichmentRunner(store, [_FakeEnricher(raises=True)]).run()
        assert s2.failed == 0
        # include_failed retries them.
        s3 = EnrichmentRunner(store, [_FakeEnricher(text="ok now")]).run(include_failed=True)
        assert s3.done == 1

    def test_skipped_status_recorded_and_excluded(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 1)
        summary = EnrichmentRunner(store, [_FakeEnricher(status=STATUS_SKIPPED)]).run()
        assert summary.skipped == 1
        assert store.source_chunks_needing_enrichment("fake", ["photo"]) == []

    def test_unavailable_enricher_skipped(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 1)
        enr = _FakeEnricher(available=False)
        summary = EnrichmentRunner(store, [enr]).run()
        assert "fake" in summary.unavailable
        assert enr.calls == 0
        assert store.enrichment_summary() == {}

    def test_should_pause_yields_gpu(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 3)
        runner = EnrichmentRunner(store, [_FakeEnricher()], should_pause=lambda: True)
        summary = runner.run()
        assert summary.paused is True
        assert summary.done == 0

    def test_limit_caps_processing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 5)
        summary = EnrichmentRunner(store, [_FakeEnricher()], batch_size=2).run(limit=3)
        assert summary.done == 3


# ---------------------------------------------------------------------------
# Store query
# ---------------------------------------------------------------------------


class TestNeedingEnrichmentQuery:
    def test_filters_by_source_type(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _seed_photos(store, 1)
        store.upsert_chunks(
            "src_text",
            [
                NormalizedChunkRecord(
                    chunk_id="t0",
                    source_type="text",
                    file_path=Path("/notes/a.md"),
                    text="a note",
                    timestamp_utc=_T0,
                    vector_collection="text_chunks",
                    metadata={"chunk_identity": "text:0"},
                )
            ],
        )
        rows = store.source_chunks_needing_enrichment("fake", ["photo"])
        assert len(rows) == 1
        assert rows[0]["source_type"] == "photo"


# ---------------------------------------------------------------------------
# OCR enricher
# ---------------------------------------------------------------------------


class TestOcrEnricher:
    def test_join_ocr_result_is_defensive(self) -> None:
        from app.enrich.ocr import _join_ocr_result

        assert _join_ocr_result(None) == ""
        assert _join_ocr_result([]) == ""
        assert _join_ocr_result([[[(0, 0)], "Hello", 0.9], [[(0, 0)], "World", 0.8]]) == "Hello World"

    def test_enrich_with_fake_engine(self, tmp_path: Path, monkeypatch) -> None:
        from app.enrich.ocr import OcrEnricher

        img = tmp_path / "shot.jpg"
        img.write_bytes(b"not-a-real-image")
        store = _store(tmp_path)
        store.upsert_chunks(
            "src",
            [
                NormalizedChunkRecord(
                    chunk_id="p0",
                    source_type="photo",
                    file_path=img,
                    text=None,
                    timestamp_utc=_T0,
                    vector_collection="image_frames",
                    metadata={"chunk_identity": "photo:0"},
                )
            ],
        )
        enr = OcrEnricher()
        monkeypatch.setattr(enr, "_available", True)
        enr._engine = lambda path: ([[[(0, 0)], "Hello", 0.99], [[(0, 0)], "World", 0.98]], 0.01)

        row = store.source_chunks_needing_enrichment("ocr", ["photo"])[0]
        out = enr.enrich(SourceChunk.from_row(row))
        assert out.status == STATUS_DONE
        assert out.records[0].text == "Hello World"
        assert out.records[0].metadata["derived_from"] == "p0"

    def test_missing_file_is_skipped(self, tmp_path: Path) -> None:
        from app.enrich.ocr import OcrEnricher

        enr = OcrEnricher()
        chunk = SourceChunk(
            chunk_id="p0",
            source_id="src",
            source_type="photo",
            file_path=tmp_path / "does_not_exist.jpg",
            chunk_identity="photo:0",
            timestamp_utc=_T0,
            session_id=None,
            lat=None,
            lon=None,
            metadata={},
        )
        assert enr.enrich(chunk).status == STATUS_SKIPPED


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestEnrichCli:
    def test_enrich_command_parsed(self) -> None:
        from app.cli.main import build_parser

        args = build_parser().parse_args(["enrich", "--limit", "10", "--retry-failed"])
        assert args.command == "enrich"
        assert args.limit == 10
        assert args.retry_failed is True
