"""Tests for the video ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingest.audio import TranscriptSegment
from app.ingest.registry import SourceKind, SourceRegistry, build_source_config
from app.ingest.runner import IngestRunner
from app.ingest.video import (
    SceneFrame,
    VideoExtract,
    _compute_phash,
    _is_near_duplicate,
    _nearest_scene_id,
    chunk_transcript_segments,
)
from app.storage.metadata import MetadataStore


# ---------------------------------------------------------------------------
# Unit: perceptual hash deduplication
# ---------------------------------------------------------------------------


def test_is_near_duplicate_identical() -> None:
    assert _is_near_duplicate(0b11110000, [0b11110000]) is True


def test_is_near_duplicate_different() -> None:
    # Two hashes with large Hamming distance should not be duplicates.
    assert _is_near_duplicate(0b00000000, [0b11111111]) is False


def test_is_near_duplicate_empty_seen() -> None:
    assert _is_near_duplicate(42, []) is False


def test_is_near_duplicate_none_phash() -> None:
    assert _is_near_duplicate(None, [123, 456]) is False


# ---------------------------------------------------------------------------
# Unit: nearest scene assignment for transcript chunks
# ---------------------------------------------------------------------------


def _make_scene(scene_id: str, ts: float) -> SceneFrame:
    return SceneFrame(scene_id=scene_id, frame_path=Path("/fake"), timestamp_sec=ts, phash=None)


def test_nearest_scene_id_empty() -> None:
    assert _nearest_scene_id((), 5.0) is None


def test_nearest_scene_id_single() -> None:
    scenes = (_make_scene("scene_0000", 0.0),)
    assert _nearest_scene_id(scenes, 30.0) == "scene_0000"


def test_nearest_scene_id_picks_closest() -> None:
    scenes = (
        _make_scene("scene_0000", 0.0),
        _make_scene("scene_0001", 20.0),
        _make_scene("scene_0002", 50.0),
    )
    assert _nearest_scene_id(scenes, 18.0) == "scene_0001"
    assert _nearest_scene_id(scenes, 35.0) == "scene_0001"
    assert _nearest_scene_id(scenes, 40.0) == "scene_0002"


# ---------------------------------------------------------------------------
# Integration: VideoIngestor via IngestRunner
# ---------------------------------------------------------------------------


def _stub_extract(path: Path) -> VideoExtract:
    """Return a VideoExtract with no scenes and a single transcript segment."""
    from app.ingest.audio import AudioTranscript

    return VideoExtract(
        video_path=path,
        scenes=(),
        transcript=AudioTranscript(
            segments=(TranscriptSegment(text="A scene description.", start=0.0, end=5.0),),
            language="en",
            duration=5.0,
            file_path=path,
            metadata={"engine": "stub"},
        ),
        duration=5.0,
        metadata={"scene_count": 0, "transcription_engine": "stub"},
    )


def test_video_ingest_normalizes_stub_file_gracefully(tmp_path: Path) -> None:
    """A non-decodable video file must not crash the runner."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "clip.mp4").write_bytes(b"\x00" * 64)

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.VIDEO, video_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    summary = IngestRunner(registry, store).run(full=True)

    assert summary.processed_items + summary.failed_items == 1


def test_video_ingest_produces_transcript_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transcript segments must produce video-typed chunks in audio_transcripts collection."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "clip.mov").write_bytes(b"\x00" * 64)

    import app.ingest.video as video_module

    monkeypatch.setattr(video_module, "_extract_video", _stub_extract)

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.VIDEO, video_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    transcript_chunks = [c for c in chunks if json.loads(c["metadata_json"]).get("video_id")]
    assert len(transcript_chunks) >= 1
    assert transcript_chunks[0]["source_type"] == "video"
    assert transcript_chunks[0]["text"] == "A scene description."


def test_video_ingest_transcript_chunk_has_scene_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transcript chunks next to a scene must carry scene_id in metadata."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "clip.mp4").write_bytes(b"\x00" * 64)

    import app.ingest.video as video_module
    from app.ingest.audio import AudioTranscript

    def stub_with_scenes(path: Path) -> VideoExtract:
        return VideoExtract(
            video_path=path,
            scenes=(
                SceneFrame(scene_id="scene_0000", frame_path=Path("/fake"), timestamp_sec=0.0, phash=None),
            ),
            transcript=AudioTranscript(
                segments=(TranscriptSegment(text="Hello.", start=0.5, end=2.0),),
                language="en",
                duration=2.0,
                file_path=path,
                metadata={"engine": "stub"},
            ),
            duration=2.0,
            metadata={"scene_count": 1, "transcription_engine": "stub"},
        )

    monkeypatch.setattr(video_module, "_extract_video", stub_with_scenes)

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.VIDEO, video_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    transcript_chunk = next(
        c for c in chunks if json.loads(c["metadata_json"]).get("chunk_identity", "").startswith("video_transcript")
    )
    meta = json.loads(transcript_chunk["metadata_json"])
    assert meta["scene_id"] == "scene_0000"


def test_video_ingest_embedding_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transcript chunks in video must have the e5 'passage:' prefix on embedding_text."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "clip.mp4").write_bytes(b"\x00" * 64)

    import app.ingest.video as video_module

    monkeypatch.setattr(video_module, "_extract_video", _stub_extract)

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.VIDEO, video_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    transcript_chunk = next(
        c for c in chunks if json.loads(c["metadata_json"]).get("chunk_identity", "").startswith("video_transcript")
    )
    meta = json.loads(transcript_chunk["metadata_json"])
    assert meta["embedding_text"].startswith("passage:")


def test_video_ingest_incremental_skips_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second incremental ingest must skip the unchanged video file."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "clip.mp4").write_bytes(b"\x00" * 64)

    import app.ingest.video as video_module

    monkeypatch.setattr(video_module, "_extract_video", _stub_extract)

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.VIDEO, video_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")
    runner = IngestRunner(registry, store)

    first = runner.run(full=True)
    second = runner.run(full=False)

    assert first.processed_items == 1
    assert second.processed_items == 0
    assert second.skipped_items == 1
