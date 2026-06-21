from pathlib import Path

from app.ingest.registry import SourceKind, SourceRegistry, build_source_config
from app.ingest.runner import IngestRunner
from app.storage.metadata import MetadataStore


def test_incremental_ingest_skips_unchanged_files(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "entry.md").write_text("Project notes", encoding="utf-8")

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.TEXT, notes))
    registry.save()
    store = MetadataStore(tmp_path / "lifelog.sqlite3")
    runner = IngestRunner(registry, store)

    first = runner.run(full=True)
    second = runner.run(full=False)

    assert first.processed_items == 1
    assert second.processed_items == 0
    assert second.skipped_items == 1
    assert second.failed_items == 0


def test_incremental_ingest_processes_modified_files(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    note = notes / "entry.md"
    note.write_text("Project notes", encoding="utf-8")

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.TEXT, notes))
    registry.save()
    store = MetadataStore(tmp_path / "lifelog.sqlite3")
    runner = IngestRunner(registry, store)

    runner.run(full=True)
    note.write_text("Updated project notes", encoding="utf-8")
    second = runner.run(full=False)

    assert second.processed_items == 1
    assert second.skipped_items == 0
    assert second.failed_items == 0


def test_full_ingest_resumes_after_unfinished_run(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "entry.md").write_text("Project notes", encoding="utf-8")

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.TEXT, notes))
    registry.save()
    store = MetadataStore(tmp_path / "lifelog.sqlite3")
    runner = IngestRunner(registry, store)

    first = runner.run(full=True)
    store.start_run("full")
    resumed = runner.run(full=True)

    assert first.processed_items == 1
    assert resumed.processed_items == 0
    assert resumed.skipped_items == 1
    assert resumed.failed_items == 0
