"""Caption enricher: describe photos and video scene frames in natural language.

Backed by a local HuggingFace image-to-text model (BLIP by default; swap to a small
VLM like moondream2 via ``LIFELOG_CAPTION_MODEL``). Degrades gracefully when
transformers/torch are absent. The backend is injectable for testing.
"""

from __future__ import annotations

import os
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
    resolve_image_path,
)

DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"


class CaptionBackend(Protocol):
    def caption(self, image_path: Path) -> str: ...


class BlipCaptionBackend:
    """Lazy-loaded HuggingFace image-to-text pipeline."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is None:
            import torch  # noqa: PLC0415
            from transformers import pipeline  # noqa: PLC0415

            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline("image-to-text", model=self._model_name, device=device)

    def caption(self, image_path: Path) -> str:
        from PIL import Image  # noqa: PLC0415

        self._load()
        image = Image.open(image_path).convert("RGB")
        out = self._pipe(image)  # type: ignore[misc]
        if isinstance(out, list) and out:
            return str(out[0].get("generated_text", "")).strip()
        return ""


class CaptionEnricher(Enricher):
    name = "caption"
    source_types = ("photo", "video")

    def __init__(self, model_name: str | None = None, backend: CaptionBackend | None = None) -> None:
        self._model_name = model_name or os.getenv("LIFELOG_CAPTION_MODEL", DEFAULT_CAPTION_MODEL)
        self._backend = backend
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._backend is not None:
            return True
        if self._available is None:
            try:
                import torch  # noqa: F401,PLC0415
                import transformers  # noqa: F401,PLC0415
                from PIL import Image  # noqa: F401,PLC0415

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_backend(self) -> CaptionBackend:
        if self._backend is None:
            self._backend = BlipCaptionBackend(self._model_name)
        return self._backend

    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        image_path = resolve_image_path(chunk)
        if image_path is None:
            return EnrichmentOutput(STATUS_SKIPPED, detail="no image for chunk")
        if not image_path.exists():
            return EnrichmentOutput(STATUS_SKIPPED, detail="image missing")
        try:
            caption = self._get_backend().caption(image_path).strip()
        except Exception as exc:  # noqa: BLE001 - per-item isolation
            return EnrichmentOutput(STATUS_FAILED, detail=str(exc))
        if not caption:
            return EnrichmentOutput(STATUS_SKIPPED, detail="empty caption")
        record = derived_text_record(
            chunk,
            enricher_name=self.name,
            suffix=derived_suffix(chunk),
            text=caption,
            extra_metadata={"caption_model": self._model_name},
        )
        return EnrichmentOutput(STATUS_DONE, records=(record,))
