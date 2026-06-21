import json
from pathlib import Path

from app.ingest.registry import SourceKind, SourceRegistry, build_source_config
from app.ingest.runner import IngestRunner
from app.storage.metadata import MetadataStore


def test_google_photo_sidecar_metadata_is_preferred(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    image_path = photos / "IMG_0001.jpg"
    image_path.write_bytes(b"not a real jpeg")
    sidecar = {
        "title": "IMG_0001.jpg",
        "description": "Accra beach",
        "photoTakenTime": {"timestamp": "1777636800"},
        "geoDataExif": {"latitude": 5.55, "longitude": -0.2},
        "cameraModel": "Pixel Test",
    }
    (photos / "IMG_0001.jpg.json").write_text(json.dumps(sidecar), encoding="utf-8")
    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.PHOTOS, photos))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    summary = IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert summary.processed_items == 1
    assert len(chunks) == 1
    assert chunks[0]["source_type"] == "photo"
    assert chunks[0]["text"] == "Accra beach"
    assert chunks[0]["lat"] == 5.55
    assert chunks[0]["lon"] == -0.2
    assert chunks[0]["metadata_json"].find("google_sidecar") != -1
