"""Face enricher: detect and embed faces in photos and video scene frames.

Backed by InsightFace (buffalo_* ONNX packs) — fully local. Produces FaceRecords
(persisted to the faces table, then clustered by app.enrich.clustering), not text
chunks. Degrades gracefully when insightface/onnxruntime are absent; the backend is
injectable for testing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.enrich.base import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    Enricher,
    EnrichmentOutput,
    SourceChunk,
    resolve_image_path,
)
from app.models.contracts import FaceRecord


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]
    det_score: float
    embedding: list[float]


class FaceBackend(Protocol):
    def detect(self, image_path: Path) -> list[DetectedFace]: ...


class InsightFaceBackend:
    """Lazy-loaded InsightFace detector + ArcFace embedder."""

    def __init__(self, model_name: str, det_threshold: float) -> None:
        self._model_name = model_name
        self._det_threshold = det_threshold
        self._app = None

    def _load(self) -> None:
        if self._app is not None:
            return
        import onnxruntime  # noqa: PLC0415
        from insightface.app import FaceAnalysis  # noqa: PLC0415  # type: ignore[import-untyped]

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        app = FaceAnalysis(name=self._model_name, providers=providers)
        app.prepare(ctx_id=0, det_thresh=self._det_threshold)
        self._app = app

    def detect(self, image_path: Path) -> list[DetectedFace]:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._load()
        image = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        results = []
        for face in self._app.get(image):  # type: ignore[union-attr]
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            bbox = tuple(float(v) for v in face.bbox)
            results.append(
                DetectedFace(
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    det_score=float(getattr(face, "det_score", 0.0)),
                    embedding=[float(v) for v in embedding],
                )
            )
        return results


class FaceEnricher(Enricher):
    name = "faces"
    source_types = ("photo", "video")

    def __init__(
        self,
        model_name: str = "buffalo_s",
        det_threshold: float = 0.5,
        backend: FaceBackend | None = None,
    ) -> None:
        self._model_name = model_name
        self._det_threshold = det_threshold
        self._backend = backend
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._backend is not None:
            return True
        if self._available is None:
            try:
                import insightface  # noqa: F401,PLC0415
                import onnxruntime  # noqa: F401,PLC0415

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_backend(self) -> FaceBackend:
        if self._backend is None:
            self._backend = InsightFaceBackend(self._model_name, self._det_threshold)
        return self._backend

    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        image_path = resolve_image_path(chunk)
        if image_path is None:
            return EnrichmentOutput(STATUS_SKIPPED, detail="no image for chunk")
        if not image_path.exists():
            return EnrichmentOutput(STATUS_SKIPPED, detail="image missing")
        try:
            detections = self._get_backend().detect(image_path)
        except Exception as exc:  # noqa: BLE001 - per-item isolation
            return EnrichmentOutput(STATUS_FAILED, detail=str(exc))

        faces = tuple(
            FaceRecord(
                face_id=hashlib.sha256(f"{chunk.chunk_id}::face:{i}".encode()).hexdigest()[:24],
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_type=chunk.source_type,  # type: ignore[arg-type]
                file_path=image_path,
                timestamp_utc=chunk.timestamp_utc,
                bbox=det.bbox,
                det_score=det.det_score,
                embedding=det.embedding,
            )
            for i, det in enumerate(detections)
        )
        # "done" even with zero faces: the chunk has been scanned and won't be redone.
        return EnrichmentOutput(STATUS_DONE, faces=faces)
