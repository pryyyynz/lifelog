"""Run the OCR enricher over photo chunks → derived searchable text chunks.

RapidOCR reads the text in screenshots/docs and emits derived text chunks
(vector_collection=text_chunks) into SQLite. Vectors are pushed to Qdrant
separately by scripts/backfill_text_embeddings.py. Pass an optional limit:
    python scripts/run_ocr_enrich.py 5
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("LIFELOG_LLM_ENABLED", "false")

from app.config import get_config
from app.enrich.ocr import OcrEnricher
from app.enrich.runner import EnrichmentRunner
from app.storage.metadata import MetadataStore


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = get_config()
    store = MetadataStore(cfg.paths.sqlite_path)
    ocr = OcrEnricher(languages=cfg.enrichment.ocr_languages)
    if not ocr.is_available():
        print("RapidOCR not available — pip install rapidocr-onnxruntime")
        return 1
    # embedder=None: derived chunks carry embedding_text; vectors go to Qdrant
    # via the backfill step (keeps the SQLite rows lean).
    runner = EnrichmentRunner(store, [ocr], embedder=None, batch_size=16)
    summary = runner.run(limit=limit)
    print(
        f"OCR: {summary.done} enriched, {summary.skipped} skipped (no text), "
        f"{summary.failed} failed, unavailable={summary.unavailable}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
