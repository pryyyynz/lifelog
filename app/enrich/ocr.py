"""OCR enricher: extract text from images so screenshots/documents are searchable.

Backed by RapidOCR (onnxruntime) — fully local, CPU-friendly, no torch dependency.
Degrades gracefully: if RapidOCR is not installed, ``is_available()`` is ``False``
and the runner skips this enricher.
"""

from __future__ import annotations

from typing import Any

from app.enrich.base import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    Enricher,
    EnrichmentOutput,
    SourceChunk,
    derived_text_record,
)


class OcrEnricher(Enricher):
    """Read text from photos via RapidOCR and emit a derived text chunk."""

    name = "ocr"
    source_types = ("photo",)

    def __init__(self, languages: tuple[str, ...] = ("en",), min_chars: int = 3) -> None:
        self._languages = tuple(languages)
        self._min_chars = min_chars
        self._engine: Any = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import rapidocr_onnxruntime  # noqa: F401  # type: ignore[import-untyped]

                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _load(self) -> None:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]

            self._engine = RapidOCR()

    def enrich(self, chunk: SourceChunk) -> EnrichmentOutput:
        if not chunk.file_path.exists():
            return EnrichmentOutput(STATUS_SKIPPED, detail="file missing")
        try:
            self._load()
            result, _elapsed = self._engine(str(chunk.file_path))
        except Exception as exc:  # noqa: BLE001 - per-item failure isolation
            return EnrichmentOutput(STATUS_FAILED, detail=str(exc))

        text = _join_ocr_result(result)
        if len(text) < self._min_chars:
            return EnrichmentOutput(STATUS_SKIPPED, detail="no text found")

        record = derived_text_record(
            chunk,
            enricher_name=self.name,
            suffix="text",
            text=text,
            extra_metadata={"ocr_languages": list(self._languages)},
        )
        return EnrichmentOutput(STATUS_DONE, records=(record,))


def _join_ocr_result(result: Any) -> str:
    """Flatten a RapidOCR result (list of ``[box, text, score]``) into a string."""
    if not result:
        return ""
    pieces: list[str] = []
    for entry in result:
        # RapidOCR rows are [box, text, score]; be defensive about shape.
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[1]:
            pieces.append(str(entry[1]).strip())
    return " ".join(p for p in pieces if p).strip()
