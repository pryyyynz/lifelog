"""Ad-hoc engine check: run representative queries through the real retriever
and report which retrieval paths fire and what comes back. Read-only."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("LIFELOG_LLM_ENABLED", "false")

from app.config import get_config
from app.ranking.reranker import CrossEncoderReranker, TemporalReranker
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.retriever import Retriever
from app.storage.metadata import MetadataStore
from app.storage.vector_store import VectorStore

cfg = get_config()
store = MetadataStore(cfg.paths.sqlite_path)
vs = VectorStore.from_environment()
print(f"Qdrant available: {vs.available}")
if vs.available:
    vs.ensure_collections()
retriever = Retriever(store, vector_store=vs if vs.available else None)
analyzer = QueryAnalyzer(use_spacy=False)
xenc = CrossEncoderReranker.from_environment()

QUERIES = [
    "deep learning roadmap",          # text content
    "PCA dimensionality reduction",   # text content (semantic)
    "grantify grant evaluation",      # video by filename
    "screenshot of code on screen",   # photo via CLIP (visual)
    "machine learning notes",         # text semantic
    "a chart or graph",               # photo via CLIP
]


def run(q: str) -> None:
    sig = analyzer.analyze(q)
    hits = retriever.retrieve(q, signals=sig, limit=20)
    paths: dict[str, int] = {}
    for h in hits:
        for r in h.rationale:
            paths[r] = paths.get(r, 0) + 1
    reranked = xenc.rerank(list(hits), q)
    print(f"\n{'='*70}\nQUERY: {q!r}")
    print(f"  signals: visual={sig.visual_intent} video={sig.video_intent} "
          f"modality_intents={sorted(sig.modality_intents)} temporal={sig.temporal_range}")
    print(f"  total hits: {len(hits)}   paths fired: {paths or 'NONE'}")
    print(f"  top 5 after rerank:")
    for h in reranked[:5]:
        from pathlib import Path
        snip = (h.snippet or "").replace("\n", " ")[:55]
        print(f"    {h.score:7.4f} [{h.source_type:5}] {Path(str(h.file_path)).name[:40]:40} "
              f"{('| '+snip) if snip else ''}  <{','.join(h.rationale)}>")


if __name__ == "__main__":
    for q in QUERIES:
        run(q)
    print(f"\n{'='*70}\nDONE")
