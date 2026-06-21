"""Tag enricher: zero-shot CLIP object/scene labels for photos and video frames.

Reuses the OpenCLIP model from the existing image suite. For each image it scores a
candidate label vocabulary and keeps the top-k above a similarity threshold, emitting
both a derived text chunk (searchable) and a ``tags`` metadata list (for faceting).
Degrades gracefully when open_clip/torch are absent. Backend is injectable.
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
    resolve_image_path,
)

DEFAULT_TAG_LABELS: tuple[str, ...] = (
    "person", "group of people", "selfie", "landscape", "beach", "mountains",
    "city street", "indoors", "food", "document", "screenshot", "chart",
    "sunset", "dog", "cat", "car", "nature", "party", "whiteboard",
    "handwriting", "building", "sky", "water", "forest", "snow", "night",
)
_PROMPT_TEMPLATE = "a photo of {}"


class TagBackend(Protocol):
    def tags(self, image_path: Path) -> list[str]: ...


class ClipTagBackend:
    """Zero-shot OpenCLIP tagger with precomputed label text features."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        labels: tuple[str, ...],
        top_k: int,
        threshold: float,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._labels = labels
        self._top_k = top_k
        self._threshold = threshold
        self._model = None
        self._preprocess = None
        self._text_features = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(self._model_name)
        prompts = tokenizer([_PROMPT_TEMPLATE.format(label) for label in self._labels])
        with torch.no_grad():
            feats = model.encode_text(prompts)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self._model = model
        self._preprocess = preprocess
        self._text_features = feats

    def tags(self, image_path: Path) -> list[str]:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._load()
        image = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0)  # type: ignore[misc]
        with torch.no_grad():
            feats = self._model.encode_image(tensor)  # type: ignore[union-attr]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sims = (feats @ self._text_features.T).squeeze(0)  # type: ignore[union-attr]
        scored = sorted(
            ((float(sims[i]), self._labels[i]) for i in range(len(self._labels))),
            reverse=True,
        )
        return [label for score, label in scored[: self._top_k] if score >= self._threshold]


class TagEnricher(Enricher):
    name = "tags"
    source_types = ("photo", "video")

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        labels: tuple[str, ...] | None = None,
        top_k: int = 5,
        threshold: float = 0.2,
        backend: TagBackend | None = None,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._labels = labels or DEFAULT_TAG_LABELS
        self._top_k = top_k
        self._threshold = threshold
        self._backend = backend
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._backend is not None:
            return True
        if self._available is None:
            try:
                import open_clip  # noqa: F401,PLC0415
                import torch  # noqa: F401,PLC0415
                from PIL import Image  # noqa: F401,PLC0415

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _get_backend(self) -> TagBackend:
        if self._backend is None:
            self._backend = ClipTagBackend(
                self._model_name, self._pretrained, self._labels, self._top_k, self._threshold
            )
        return self._backend

    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        image_path = resolve_image_path(chunk)
        if image_path is None:
            return EnrichmentOutput(STATUS_SKIPPED, detail="no image for chunk")
        if not image_path.exists():
            return EnrichmentOutput(STATUS_SKIPPED, detail="image missing")
        try:
            tags = self._get_backend().tags(image_path)
        except Exception as exc:  # noqa: BLE001 - per-item isolation
            return EnrichmentOutput(STATUS_FAILED, detail=str(exc))
        if not tags:
            return EnrichmentOutput(STATUS_SKIPPED, detail="no tags above threshold")
        record = derived_text_record(
            chunk,
            enricher_name=self.name,
            suffix=derived_suffix(chunk),
            text=", ".join(tags),
            extra_metadata={"tags": list(tags)},
        )
        return EnrichmentOutput(STATUS_DONE, records=(record,))
