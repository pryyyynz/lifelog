"""VLM enricher: describe photos and video frames with a small vision-language model.

Backed by Qwen2-VL-2B-Instruct (local, GPU when available). Unlike BLIP captions,
the VLM *reads* the image — naming the app/page/document and what it shows — which
makes non-OCR'able content searchable. Emits a derived text chunk that flows into
the existing e5 + BM25 + cross-encoder pipeline. Degrades gracefully when
transformers/torch are missing. Backend is injectable for testing.
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

DEFAULT_VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_PROMPT = (
    "In one sentence, describe what this image shows. If it is a screenshot or a "
    "document, name the app, site, or document type and its main content."
)


class VlmBackend(Protocol):
    def describe(self, image_path: Path) -> str: ...


class Qwen2VLBackend:
    """Lazy-loaded Qwen2-VL describe() — fp16 on GPU, image capped to bound cost."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 80,
        max_edge: int = 1024,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._max_edge = max_edge
        self._prompt = prompt
        self._model = None
        self._processor = None
        self._device = None

    def _load(self) -> None:
        if self._model is None:
            import torch  # noqa: PLC0415
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: PLC0415

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_name, torch_dtype=dtype
            ).to(self._device)
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(self._model_name)

    def describe(self, image_path: Path) -> str:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._load()
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((self._max_edge, self._max_edge))  # cap vision tokens
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self._prompt}]}
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class VlmEnricher(Enricher):
    name = "vlm"
    source_types = ("photo", "video")

    def __init__(self, model_name: str | None = None, backend: VlmBackend | None = None) -> None:
        self._model_name = model_name or os.getenv("LIFELOG_VLM_MODEL", DEFAULT_VLM_MODEL)
        self._backend = backend
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._backend is not None:
            return True
        if self._available is None:
            try:
                import torch  # noqa: F401,PLC0415
                import transformers  # noqa: PLC0415
                from PIL import Image  # noqa: F401,PLC0415

                self._available = hasattr(transformers, "Qwen2VLForConditionalGeneration")
            except ImportError:
                self._available = False
        return self._available

    def _get_backend(self) -> VlmBackend:
        if self._backend is None:
            self._backend = Qwen2VLBackend(self._model_name)
        return self._backend

    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        image_path = resolve_image_path(chunk)
        if image_path is None:
            return EnrichmentOutput(STATUS_SKIPPED, detail="no image for chunk")
        if not image_path.exists():
            return EnrichmentOutput(STATUS_SKIPPED, detail="image missing")
        try:
            description = self._get_backend().describe(image_path).strip()
        except Exception as exc:  # noqa: BLE001 - per-item isolation
            return EnrichmentOutput(STATUS_FAILED, detail=str(exc))
        if not description:
            return EnrichmentOutput(STATUS_SKIPPED, detail="empty description")
        record = derived_text_record(
            chunk,
            enricher_name=self.name,
            suffix=derived_suffix(chunk),
            text=description,
            extra_metadata={"vlm_model": self._model_name},
        )
        return EnrichmentOutput(STATUS_DONE, records=(record,))
