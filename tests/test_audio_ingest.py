"""Tests for the audio and voice memo ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest.audio import (
    TranscriptSegment,
    VoiceMemoIngestor,
    chunk_transcript_segments,
    convert_audio_to_wav,
)
from app.ingest.registry import SourceKind, SourceRegistry, build_source_config
from app.ingest.runner import IngestRunner
from app.storage.metadata import MetadataStore


# ---------------------------------------------------------------------------
# Unit: transcript chunking logic
# ---------------------------------------------------------------------------


def test_chunk_transcript_empty_returns_empty() -> None:
    assert chunk_transcript_segments(()) == []


def test_chunk_transcript_single_segment() -> None:
    segs = (TranscriptSegment(text="Hello world.", start=0.0, end=3.0),)
    chunks = chunk_transcript_segments(segs)
    assert len(chunks) == 1
    text, start, end, speaker = chunks[0]
    assert text == "Hello world."
    assert start == 0.0
    assert end == 3.0
    assert speaker is None


def test_chunk_transcript_merges_adjacent_segments() -> None:
    segs = (
        TranscriptSegment(text="First.", start=0.0, end=2.0),
        TranscriptSegment(text="Second.", start=2.5, end=5.0),  # gap = 0.5s < threshold
    )
    chunks = chunk_transcript_segments(segs, gap_threshold=2.0)
    assert len(chunks) == 1
    assert "First." in chunks[0][0]
    assert "Second." in chunks[0][0]


def test_chunk_transcript_splits_on_silence() -> None:
    segs = (
        TranscriptSegment(text="Part A.", start=0.0, end=2.0),
        TranscriptSegment(text="Part B.", start=10.0, end=12.0),  # gap = 8s > threshold
    )
    chunks = chunk_transcript_segments(segs, gap_threshold=2.0)
    assert len(chunks) == 2
    assert chunks[0][0] == "Part A."
    assert chunks[1][0] == "Part B."


def test_chunk_transcript_splits_on_duration_limit() -> None:
    # Five 15-second segments with no silence; should be split when projected duration > max.
    segs = tuple(
        TranscriptSegment(text=f"Seg{i}.", start=float(i * 15), end=float(i * 15 + 14))
        for i in range(5)
    )
    chunks = chunk_transcript_segments(segs, max_duration=30.0, gap_threshold=100.0)
    # With max_duration=30s, segments of 15s each: 0+14=14s (ok), +15=29s (ok), +15=44s (split)
    # So we expect at least 2 chunks.
    assert len(chunks) >= 2


def test_chunk_transcript_splits_on_speaker_change() -> None:
    segs = (
        TranscriptSegment(text="Alice says.", start=0.0, end=3.0, speaker="SPEAKER_A"),
        TranscriptSegment(text="Bob responds.", start=3.1, end=6.0, speaker="SPEAKER_B"),
    )
    chunks = chunk_transcript_segments(segs, gap_threshold=5.0)
    assert len(chunks) == 2
    assert chunks[0][3] == "SPEAKER_A"
    assert chunks[1][3] == "SPEAKER_B"


def test_chunk_transcript_preserves_timestamps() -> None:
    segs = (
        TranscriptSegment(text="A.", start=1.0, end=2.0),
        TranscriptSegment(text="B.", start=8.0, end=9.0),
    )
    chunks = chunk_transcript_segments(segs, gap_threshold=2.0)
    assert chunks[0][1] == 1.0  # start of first chunk
    assert chunks[0][2] == 2.0  # end of first chunk
    assert chunks[1][1] == 8.0
    assert chunks[1][2] == 9.0


# ---------------------------------------------------------------------------
# Unit: convert_audio_to_wav skips gracefully without ffmpeg
# ---------------------------------------------------------------------------


def test_convert_audio_to_wav_returns_none_without_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """convert_audio_to_wav should return None when ffmpeg is not on PATH."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: None)
    result = convert_audio_to_wav(tmp_path / "fake.m4a", tmp_path / "out.wav")
    assert result is None


# ---------------------------------------------------------------------------
# Integration: VoiceMemoIngestor via IngestRunner
# ---------------------------------------------------------------------------


def test_audio_ingest_normalizes_stub_file_gracefully(tmp_path: Path) -> None:
    """A non-decodable audio file should not crash the runner; one item is attempted."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # Write a stub file that ffmpeg / whisper cannot decode — ingest should fail gracefully.
    (audio_dir / "memo.m4a").write_bytes(b"\x00" * 64)

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.AUDIO, audio_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    summary = IngestRunner(registry, store).run(full=True)

    # The runner must not raise. Either the item is processed (with empty transcript) or failed.
    assert summary.processed_items + summary.failed_items == 1


def test_audio_ingest_produces_audio_source_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When transcription returns segments the chunks must carry source_type='audio'."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "memo.wav").write_bytes(b"\x00" * 64)

    # Stub transcription so no real model is needed.
    import app.ingest.audio as audio_module

    monkeypatch.setattr(
        audio_module,
        "_transcribe",
        lambda path: audio_module.AudioTranscript(
            segments=(
                TranscriptSegment(text="This is a voice memo.", start=0.0, end=3.0),
            ),
            language="en",
            duration=3.0,
            file_path=path,
            metadata={"engine": "stub"},
        ),
    )

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.AUDIO, audio_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert len(chunks) == 1
    assert chunks[0]["source_type"] == "audio"
    assert chunks[0]["text"] == "This is a voice memo."
    assert '"transcription_engine": "stub"' in chunks[0]["metadata_json"]


def test_audio_ingest_stores_timestamp_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transcript chunks must carry timestamp_start_sec / timestamp_end_sec in metadata."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "memo.mp3").write_bytes(b"\x00" * 64)

    import app.ingest.audio as audio_module

    monkeypatch.setattr(
        audio_module,
        "_transcribe",
        lambda path: audio_module.AudioTranscript(
            segments=(TranscriptSegment(text="Hello.", start=1.5, end=4.0),),
            language="en",
            duration=4.0,
            file_path=path,
            metadata={"engine": "stub"},
        ),
    )

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.AUDIO, audio_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert len(chunks) == 1
    import json
    meta = json.loads(chunks[0]["metadata_json"])
    assert meta["timestamp_start_sec"] == pytest.approx(1.5)
    assert meta["timestamp_end_sec"] == pytest.approx(4.0)


def test_audio_ingest_embedding_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding text for audio chunks must carry the e5 'passage:' prefix."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "memo.wav").write_bytes(b"\x00" * 64)

    import app.ingest.audio as audio_module

    monkeypatch.setattr(
        audio_module,
        "_transcribe",
        lambda path: audio_module.AudioTranscript(
            segments=(TranscriptSegment(text="Test transcript.", start=0.0, end=2.0),),
            language="en",
            duration=2.0,
            file_path=path,
            metadata={"engine": "stub"},
        ),
    )

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.AUDIO, audio_dir))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    import json
    meta = json.loads(chunks[0]["metadata_json"])
    assert meta["embedding_text"].startswith("passage:")
