"""Action enricher: recognize actions in video scenes via zero-shot X-CLIP.

Operates on video frame chunks: samples a short clip of frames around the scene
timestamp and scores a candidate action vocabulary. Emits a derived text chunk with
the top actions. Degrades gracefully when transformers/av are absent; the backend is
injectable for testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.enrich.base import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    Enricher,
    EnrichmentOutput,
    SourceChunk,
    derived_suffix,
    derived_text_record,
)

DEFAULT_ACTION_MODEL = "microsoft/xclip-base-patch32"
DEFAULT_ACTION_LABELS: tuple[str, ...] = (
    "cooking", "eating", "drinking", "running", "walking", "dancing", "singing",
    "playing guitar", "playing piano", "riding a bike", "driving", "swimming",
    "reading", "writing", "typing", "laughing", "clapping", "waving",
    "opening a gift", "blowing out candles", "playing with a dog", "exercising",
    "hiking", "skiing", "playing soccer", "playing basketball", "giving a speech",
)


class ActionBackend(Protocol):
    def actions(self, video_path: Path, start_sec: float) -> list[str]: ...


class XClipActionBackend:
    """Lazy-loaded X-CLIP zero-shot action recognizer reading clips via PyAV."""

    def __init__(
        self,
        model_name: str,
        labels: tuple[str, ...],
        top_k: int,
        threshold: float,
        num_frames: int = 8,
        window_sec: float = 4.0,
    ) -> None:
        self._model_name = model_name
        self._labels = labels
        self._top_k = top_k
        self._threshold = threshold
        self._num_frames = num_frames
        self._window_sec = window_sec
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()

    def _read_frames(self, video_path: Path, start_sec: float) -> list:
        import av  # noqa: PLC0415  # type: ignore[import-untyped]
        import numpy as np  # noqa: PLC0415

        frames: list = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            begin = max(0.0, start_sec)
            container.seek(int(begin / float(stream.time_base)) if stream.time_base else 0, stream=stream)
            wanted = self._num_frames
            for frame in container.decode(video=0):
                if frame.time is None or frame.time < begin:
                    continue
                if frame.time > begin + self._window_sec:
                    break
                frames.append(np.array(frame.to_image()))
                if len(frames) >= wanted:
                    break
        return frames

    def actions(self, video_path: Path, start_sec: float) -> list[str]:
        import torch  # noqa: PLC0415

        self._load()
        frames = self._read_frames(video_path, start_sec)
        if not frames:
            return []
        inputs = self._processor(  # type: ignore[misc]
            text=list(self._labels), videos=[frames], return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[misc]
            probs = outputs.logits_per_video.softmax(dim=-1).squeeze(0)
        scored = sorted(
            ((float(probs[i]), self._labels[i]) for i in range(len(self._labels))),
            reverse=True,
        )
        return [label for score, label in scored[: self._top_k] if score >= self._threshold]


class ActionEnricher(Enricher):
    name = "action"
    source_types = ("video",)

    def __init__(
        self,
        model_name: str | None = None,
        labels: tuple[str, ...] | None = None,
        top_k: int = 3,
        threshold: float = 0.3,
        backend: ActionBackend | None = None,
    ) -> None:
        self._model_name = model_name or DEFAULT_ACTION_MODEL
        self._labels = labels or DEFAULT_ACTION_LABELS
        self._top_k = top_k
        self._threshold = threshold
        self._backend = backend
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._backend is not None:
            return True
        if self._available is None:
            try:
                import av  # noqa: F401,PLC0415  # type: ignore[import-untyped]
                import torch  # noqa: F401,PLC0415
                import transformers  # noqa: F401,PLC0415

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_backend(self) -> ActionBackend:
        if self._backend is None:
            self._backend = XClipActionBackend(
                self._model_name, self._labels, self._top_k, self._threshold
            )
        return self._backend

    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        # Only scene-frame chunks carry a scene_id; transcript chunks are skipped.
        if not chunk.metadata.get("scene_id"):
            return EnrichmentOutput(STATUS_SKIPPED, detail="not a scene chunk")
        if not chunk.file_path.exists():
            return EnrichmentOutput(STATUS_SKIPPED, detail="video missing")
        start_sec = chunk.timestamp_start_sec or 0.0
        try:
            actions = self._get_backend().actions(chunk.file_path, float(start_sec))
        except Exception as exc:  # noqa: BLE001 - per-item isolation
            return EnrichmentOutput(STATUS_FAILED, detail=str(exc))
        if not actions:
            return EnrichmentOutput(STATUS_SKIPPED, detail="no actions above threshold")
        record = derived_text_record(
            chunk,
            enricher_name=self.name,
            suffix=derived_suffix(chunk),
            text=", ".join(actions),
            extra_metadata={"actions": list(actions), "action_model": self._model_name},
        )
        return EnrichmentOutput(STATUS_DONE, records=(record,))
