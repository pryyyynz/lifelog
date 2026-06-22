"""Run the VLM enricher (Qwen2-VL) over photo/video chunks → derived descriptions.

Emits derived text chunks (vector_collection=text_chunks). Push to Qdrant with
scripts/backfill_text_embeddings.py afterwards. Optional limit:
    python scripts/run_vlm_enrich.py 3
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("LIFELOG_LLM_ENABLED", "false")

from app.config import get_config
from app.enrich.runner import EnrichmentRunner
from app.enrich.vlm import VlmEnricher
from app.storage.metadata import MetadataStore


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = get_config()
    store = MetadataStore(cfg.paths.sqlite_path)
    vlm = VlmEnricher()
    if not vlm.is_available():
        print("Qwen2-VL not available (transformers/torch missing).")
        return 1
    runner = EnrichmentRunner(store, [vlm], embedder=None, batch_size=8)
    t = time.time()
    summary = runner.run(limit=limit)
    print(
        f"VLM: {summary.done} described, {summary.skipped} skipped, {summary.failed} failed "
        f"in {time.time() - t:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
