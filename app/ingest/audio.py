"""Audio and voice memo transcription pipeline."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingest.base import DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.file_ingestor import LocalFileIngestor
from app.ingest.text import prepare_embedding_text
from app.models.contracts import NormalizedChunkRecord

# Maximum silence gap (seconds) between segments to keep in the same chunk.
_SEGMENT_GAP_THRESHOLD = 2.0
# Target max duration (seconds) per transcript chunk.
_MAX_CHUNK_DURATION = 60.0


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass(frozen=True)
class AudioTranscript:
    segments: tuple[TranscriptSegment, ...]
    language: str | None
    duration: float | None
    file_path: Path
    metadata: dict[str, Any]


class VoiceMemoIngestor(LocalFileIngestor):
    """Ingests M4A, MP3, and WAV voice memos via audio transcription."""

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        transcript = _transcribe(item.path)
        return ExtractedItem(
            discovered=item,
            payload=transcript,
            metadata=transcript.metadata,
        )

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        transcript = item.payload
        if not isinstance(transcript, AudioTranscript):
            return []
        chunks = chunk_transcript_segments(transcript.segments)
        records: list[NormalizedChunkRecord] = []
        for index, (text, start, end, speaker) in enumerate(chunks):
            identity = f"audio:{index}"
            metadata: dict[str, Any] = {
                "chunk_index": index,
                "chunk_identity": identity,
                "timestamp_start_sec": start,
                "timestamp_end_sec": end,
                "language": transcript.language,
                "transcription_engine": transcript.metadata.get("engine", "unknown"),
                "raw_text": text,
                "embedding_text": prepare_embedding_text(text, model_name="intfloat/e5-large-v2"),
            }
            if speaker:
                metadata["speaker_id"] = speaker
            records.append(
                NormalizedChunkRecord(
                    chunk_id=_chunk_id(item.discovered.path, identity),
                    source_type="audio",
                    file_path=item.discovered.path,
                    text=text,
                    timestamp_utc=_file_timestamp(item.discovered.mtime_ns),
                    timestamp_start_sec=start,
                    timestamp_end_sec=end,
                    vector_collection="audio_transcripts",
                    metadata=metadata,
                )
            )
        return records


class TranscriptionEngine:
    """Unified transcription interface backed by WhisperX or openai-whisper.

    Loaded models are cached per model name on the instance so repeat calls
    (notably interactive voice-search queries) skip the multi-second load cost.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self._model_cache: dict[str, Any] = {}

    @classmethod
    def load(cls) -> TranscriptionEngine:
        preferred = os.getenv("LIFELOG_TRANSCRIPTION_ENGINE", "whisperx").lower()
        if preferred == "whisperx":
            try:
                import whisperx  # noqa: F401
                return cls("whisperx")
            except ImportError:
                pass
        try:
            import whisper  # noqa: F401
            return cls("openai_whisper")
        except ImportError:
            pass
        return cls("unavailable")

    def transcribe(
        self,
        audio_path: Path,
        *,
        original_path: Path,
        model_name: str | None = None,
    ) -> AudioTranscript:
        """Transcribe ``audio_path``.

        ``model_name`` overrides the default ingest model; callers handling short
        interactive queries pass a smaller model (e.g. ``base``) for low latency.
        """
        resolved_model = model_name or os.getenv("LIFELOG_TRANSCRIPTION_MODEL", "large-v3")
        if self.backend == "unavailable":
            return AudioTranscript(
                segments=(),
                language=None,
                duration=None,
                file_path=original_path,
                metadata={"engine": "unavailable", "error": "no transcription library installed"},
            )
        if self.backend == "whisperx":
            return self._run_whisperx(audio_path, original_path, resolved_model)
        return self._run_openai_whisper(audio_path, original_path, resolved_model)

    def _run_whisperx(self, audio_path: Path, original_path: Path, model_name: str) -> AudioTranscript:
        try:
            import torch
            import whisperx

            cache_key = f"whisperx:{model_name}"
            model = self._model_cache.get(cache_key)
            if model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                model_dir = os.getenv("LIFELOG_MODEL_DIR", "./models/transcription")
                model = whisperx.load_model(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=model_dir,
                )
                self._model_cache[cache_key] = model
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=16)
            language = result.get("language")
            segments = tuple(
                TranscriptSegment(
                    text=seg["text"].strip(),
                    start=float(seg.get("start", 0)),
                    end=float(seg.get("end", 0)),
                )
                for seg in result.get("segments", [])
                if seg.get("text", "").strip()
            )
            duration = float(audio.shape[0]) / 16000.0 if hasattr(audio, "shape") else None
            return AudioTranscript(
                segments=segments,
                language=language,
                duration=duration,
                file_path=original_path,
                metadata={"engine": "whisperx", "model": model_name},
            )
        except Exception as exc:  # noqa: BLE001
            return AudioTranscript(
                segments=(),
                language=None,
                duration=None,
                file_path=original_path,
                metadata={"engine": "whisperx", "error": str(exc)},
            )

    def _run_openai_whisper(self, audio_path: Path, original_path: Path, model_name: str) -> AudioTranscript:
        try:
            import whisper

            cache_key = f"openai_whisper:{model_name}"
            model = self._model_cache.get(cache_key)
            if model is None:
                model_dir = os.getenv("LIFELOG_MODEL_DIR", "./models/transcription")
                model = whisper.load_model(model_name, download_root=model_dir)
                self._model_cache[cache_key] = model
            result = model.transcribe(str(audio_path), verbose=False)
            language = result.get("language")
            segments = tuple(
                TranscriptSegment(
                    text=seg["text"].strip(),
                    start=float(seg.get("start", 0)),
                    end=float(seg.get("end", 0)),
                )
                for seg in result.get("segments", [])
                if seg.get("text", "").strip()
            )
            return AudioTranscript(
                segments=segments,
                language=language,
                duration=result.get("duration"),
                file_path=original_path,
                metadata={"engine": "openai_whisper", "model": model_name},
            )
        except Exception as exc:  # noqa: BLE001
            return AudioTranscript(
                segments=(),
                language=None,
                duration=None,
                file_path=original_path,
                metadata={"engine": "openai_whisper", "error": str(exc)},
            )


def chunk_transcript_segments(
    segments: tuple[TranscriptSegment, ...],
    max_duration: float = _MAX_CHUNK_DURATION,
    gap_threshold: float = _SEGMENT_GAP_THRESHOLD,
) -> list[tuple[str, float, float, str | None]]:
    """Group transcript segments into chunks separated by silence or speaker changes.

    Returns a list of (text, start_sec, end_sec, speaker_id | None) tuples.
    """
    if not segments:
        return []

    chunks: list[tuple[str, float, float, str | None]] = []
    current_texts: list[str] = []
    current_start = segments[0].start
    current_end = segments[0].start
    current_speaker = segments[0].speaker

    for seg in segments:
        silence = seg.start - current_end
        projected_duration = seg.end - current_start
        speaker_changed = seg.speaker is not None and seg.speaker != current_speaker

        if current_texts and (silence > gap_threshold or projected_duration > max_duration or speaker_changed):
            text = " ".join(current_texts).strip()
            if text:
                chunks.append((text, current_start, current_end, current_speaker))
            current_texts = []
            current_start = seg.start
            current_speaker = seg.speaker

        current_texts.append(seg.text)
        current_end = seg.end

    if current_texts:
        text = " ".join(current_texts).strip()
        if text:
            chunks.append((text, current_start, current_end, current_speaker))

    return chunks


def convert_audio_to_wav(input_path: Path, output_path: Path) -> Path | None:
    """Normalize audio to 16kHz mono WAV using ffmpeg. Returns output_path on success."""
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        return None
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-i", str(input_path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            str(output_path),
        ],
        capture_output=True,
        timeout=300,
    )
    if result.returncode == 0 and output_path.exists():
        return output_path
    return None


def _transcribe(path: Path) -> AudioTranscript:
    """Convert audio to WAV and transcribe. Handles missing ffmpeg gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = convert_audio_to_wav(path, Path(tmpdir) / "audio.wav")
        engine = TranscriptionEngine.load()
        return engine.transcribe(wav_path or path, original_path=path)


def _ffmpeg_executable() -> str | None:
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    from app.config import get_config

    configured = get_config().paths.ffmpeg_path
    if configured is not None and configured.exists():
        return str(configured)
    return None


def _chunk_id(path: Path, identity: str) -> str:
    raw = f"{path!s}::{identity}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _file_timestamp(mtime_ns: int) -> datetime:
    return datetime.fromtimestamp(mtime_ns / 1e9, tz=UTC)
