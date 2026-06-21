"""Video ingest pipeline: scene detection, frame extraction, and audio transcription."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingest.audio import (
    AudioTranscript,
    TranscriptionEngine,
    chunk_transcript_segments,
    convert_audio_to_wav,
)
from app.ingest.base import DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.file_ingestor import LocalFileIngestor
from app.ingest.images import OpenClipImageEmbedder
from app.ingest.text import prepare_embedding_text
from app.models.contracts import NormalizedChunkRecord

_SCENE_THRESHOLD = 27.0
# Perceptual hash Hamming distance below which a frame is considered a duplicate.
_PHASH_DEDUP_THRESHOLD = 8


@dataclass(frozen=True)
class SceneFrame:
    scene_id: str
    frame_path: Path
    timestamp_sec: float
    phash: int | None


@dataclass(frozen=True)
class VideoExtract:
    video_path: Path
    scenes: tuple[SceneFrame, ...]
    transcript: AudioTranscript
    duration: float | None
    metadata: dict[str, Any]


class VideoIngestor(LocalFileIngestor):
    """Ingests MP4 and MOV videos: scene frames with CLIP embeddings and transcript chunks."""

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        extract = _extract_video(item.path)
        return ExtractedItem(
            discovered=item,
            payload=extract,
            metadata=extract.metadata,
        )

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        extract = item.payload
        if not isinstance(extract, VideoExtract):
            return []

        base_ts = _file_timestamp(item.discovered.mtime_ns)
        records: list[NormalizedChunkRecord] = []

        # One record per unique scene frame.
        for frame in extract.scenes:
            identity = f"video_frame:{frame.scene_id}"
            metadata: dict[str, Any] = {
                "chunk_identity": identity,
                "scene_id": frame.scene_id,
                "frame_path": str(frame.frame_path),
                "video_id": str(extract.video_path),
                "phash": frame.phash,
            }
            records.append(
                NormalizedChunkRecord(
                    chunk_id=_chunk_id(item.discovered.path, identity),
                    source_type="video",
                    file_path=item.discovered.path,
                    text=None,
                    timestamp_utc=base_ts,
                    timestamp_start_sec=frame.timestamp_sec,
                    vector_collection="video_frames",
                    metadata=metadata,
                )
            )

        # One record per transcript chunk, linked to nearest scene.
        transcript_chunks = chunk_transcript_segments(extract.transcript.segments)
        for index, (text, start, end, speaker) in enumerate(transcript_chunks):
            scene_id = _nearest_scene_id(extract.scenes, start)
            identity = f"video_transcript:{index}"
            metadata = {
                "chunk_index": index,
                "chunk_identity": identity,
                "scene_id": scene_id,
                "video_id": str(extract.video_path),
                "language": extract.transcript.language,
                "transcription_engine": extract.transcript.metadata.get("engine", "unknown"),
                "raw_text": text,
                "embedding_text": prepare_embedding_text(text, model_name="intfloat/e5-large-v2"),
            }
            if speaker:
                metadata["speaker_id"] = speaker
            records.append(
                NormalizedChunkRecord(
                    chunk_id=_chunk_id(item.discovered.path, identity),
                    source_type="video",
                    file_path=item.discovered.path,
                    text=text,
                    timestamp_utc=base_ts,
                    timestamp_start_sec=start,
                    timestamp_end_sec=end,
                    vector_collection="audio_transcripts",
                    metadata=metadata,
                )
            )

        return records

    def embed(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        embedder = OpenClipImageEmbedder.from_environment()
        embedded: list[NormalizedChunkRecord] = []
        for record in records:
            metadata = dict(record.metadata)
            if record.vector_collection == "video_frames":
                frame_path_str = metadata.get("frame_path", "")
                if frame_path_str:
                    frame_path = Path(frame_path_str)
                    if frame_path.exists():
                        vector = embedder.embed(frame_path)
                        if vector is not None:
                            metadata["image_embedding"] = vector
                            metadata["image_embedding_status"] = "ok"
                        else:
                            metadata["image_embedding_status"] = embedder.status
                    else:
                        metadata["image_embedding_status"] = "frame_missing"
            embedded.append(_replace_metadata(record, metadata))
        return embedded


def _extract_video(path: Path) -> VideoExtract:
    """Extract scenes and transcript from a video file.

    Frames are copied to persistent storage before the temp directory is cleaned up.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Extract and transcribe audio.
        wav_path = convert_audio_to_wav(path, tmp / "audio.wav")
        engine = TranscriptionEngine.load()
        transcript = engine.transcribe(wav_path or path, original_path=path)

        # Detect scene cuts and extract one frame per scene.
        scenes_tmp = _detect_scenes_and_frames(path, tmp)

        # Copy frames out of the temp directory to permanent storage.
        frames_dir = _frames_dir(path)
        frames_dir.mkdir(parents=True, exist_ok=True)
        persisted: list[SceneFrame] = []
        for scene in scenes_tmp:
            if scene.frame_path.exists():
                dest = frames_dir / scene.frame_path.name
                shutil.copy2(scene.frame_path, dest)
                persisted.append(
                    SceneFrame(
                        scene_id=scene.scene_id,
                        frame_path=dest,
                        timestamp_sec=scene.timestamp_sec,
                        phash=scene.phash,
                    )
                )

    return VideoExtract(
        video_path=path,
        scenes=tuple(persisted),
        transcript=transcript,
        duration=None,
        metadata={
            "scene_count": len(persisted),
            "transcription_engine": transcript.metadata.get("engine"),
        },
    )


def _detect_scenes_and_frames(video_path: Path, tmp: Path) -> list[SceneFrame]:
    """Return per-scene representative frames with perceptual-hash deduplication."""
    timestamps = _detect_scene_timestamps(video_path)
    if not timestamps:
        timestamps = [0.0]

    frames: list[SceneFrame] = []
    seen_hashes: list[int] = []

    for idx, ts in enumerate(timestamps):
        scene_id = f"scene_{idx:04d}"
        frame_path = tmp / f"{scene_id}.jpg"
        if not _extract_frame(video_path, ts, frame_path):
            continue
        phash = _compute_phash(frame_path)
        if _is_near_duplicate(phash, seen_hashes):
            continue
        if phash is not None:
            seen_hashes.append(phash)
        frames.append(
            SceneFrame(scene_id=scene_id, frame_path=frame_path, timestamp_sec=ts, phash=phash)
        )

    return frames


def _detect_scene_timestamps(video_path: Path) -> list[float]:
    """Return scene-cut start timestamps using PySceneDetect. Returns [] on failure."""
    try:
        from scenedetect import ContentDetector, detect  # type: ignore[import-untyped]

        scene_list = detect(str(video_path), ContentDetector(threshold=_SCENE_THRESHOLD))
        return [float(start.get_seconds()) for start, _ in scene_list]
    except ImportError:
        return []
    except Exception:  # noqa: BLE001
        return []


def _extract_frame(video_path: Path, timestamp_sec: float, output_path: Path) -> bool:
    """Use ffmpeg to extract a single frame at the given timestamp."""
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        return False
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-ss", str(timestamp_sec),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ],
        capture_output=True,
        timeout=60,
    )
    return result.returncode == 0 and output_path.exists()


def _compute_phash(frame_path: Path) -> int | None:
    """Compute a perceptual hash for a frame image.

    Uses imagehash if available; falls back to a pure-PIL 8×8 grayscale hash.
    """
    try:
        import imagehash  # type: ignore[import-untyped]
        from PIL import Image

        with Image.open(frame_path) as img:
            return int(imagehash.phash(img))
    except ImportError:
        pass
    try:
        from PIL import Image

        with Image.open(frame_path).convert("L").resize((8, 8)) as img:
            pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = [1 if p > avg else 0 for p in pixels]
        return int("".join(str(b) for b in bits), 2)
    except Exception:  # noqa: BLE001
        return None


def _is_near_duplicate(phash: int | None, seen: list[int]) -> bool:
    if phash is None:
        return False
    for existing in seen:
        if bin(phash ^ existing).count("1") < _PHASH_DEDUP_THRESHOLD:
            return True
    return False


def _nearest_scene_id(scenes: tuple[SceneFrame, ...], timestamp: float) -> str | None:
    if not scenes:
        return None
    return min(scenes, key=lambda s: abs(s.timestamp_sec - timestamp)).scene_id


def _frames_dir(video_path: Path) -> Path:
    """Return a stable data-dir sub-path for storing extracted frames."""
    from app.config import get_config

    data_dir = get_config().paths.data_dir
    safe_name = hashlib.sha256(str(video_path).encode()).hexdigest()[:16]
    return data_dir / "frames" / safe_name


def _replace_metadata(record: NormalizedChunkRecord, metadata: dict[str, Any]) -> NormalizedChunkRecord:
    return NormalizedChunkRecord(
        chunk_id=record.chunk_id,
        source_type=record.source_type,
        file_path=record.file_path,
        text=record.text,
        timestamp_utc=record.timestamp_utc,
        vector_collection=record.vector_collection,
        vector_id=record.vector_id,
        session_id=record.session_id,
        timestamp_start_sec=record.timestamp_start_sec,
        timestamp_end_sec=record.timestamp_end_sec,
        lat=record.lat,
        lon=record.lon,
        place_name=record.place_name,
        metadata=metadata,
    )


def _chunk_id(path: Path, identity: str) -> str:
    raw = f"{path!s}::{identity}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _file_timestamp(mtime_ns: int) -> datetime:
    return datetime.fromtimestamp(mtime_ns / 1e9, tz=UTC)


def _ffmpeg_executable() -> str | None:
    from app.ingest.audio import _ffmpeg_executable as audio_ffmpeg

    return audio_ffmpeg()
