from pathlib import Path

from app.ingest.registry import SourceKind, SourceRegistry, build_source_config, validate_source


def test_source_registry_persists_valid_text_source(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "entry.md").write_text("Project notes", encoding="utf-8")

    source = build_source_config(SourceKind.TEXT, notes)
    validation = validate_source(source)

    assert validation.ok
    assert validation.item_count == 1

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(source)
    registry.save()

    loaded = SourceRegistry(tmp_path / "sources.json")

    assert len(loaded.sources) == 1
    assert loaded.sources[0].source_type == SourceKind.TEXT
    assert loaded.sources[0].path == notes.resolve()


def test_source_validation_rejects_unsupported_file(tmp_path: Path) -> None:
    source_path = tmp_path / "notes.pdf"
    source_path.write_bytes(b"%PDF")
    source = build_source_config(SourceKind.TEXT, source_path)

    validation = validate_source(source)

    assert not validation.ok
    assert "unsupported file format" in validation.errors[0]
