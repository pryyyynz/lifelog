"""Hybrid retriever: BM25 sparse + Qdrant dense paths fused with RRF."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.contracts import RetrievalHit
from app.ranking.fusion import RRFFusion
from app.retrieval.query_analyzer import QuerySignals
from app.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)

# Common English function words carry no retrieval signal; keeping them lets
# queries like "sunset on the beach" match documents purely on "on"/"the".
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
        "for", "with", "from", "by", "about", "as", "into", "is", "are", "was",
        "were", "be", "been", "being", "do", "did", "does", "done", "i", "me",
        "my", "we", "our", "us", "you", "your", "it", "its", "he", "she", "they",
        "them", "this", "that", "these", "those", "what", "which", "who", "whom",
        "when", "where", "how", "why", "show", "find", "get", "there", "here",
        "then", "than", "so", "up", "out", "over", "had", "has", "have", "will",
        "would", "can", "could", "should", "any", "some", "all", "no", "not",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and drop stopwords / single chars for BM25."""
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]

# Collections that carry text embeddings
_TEXT_COLLECTIONS = frozenset({"text_chunks", "audio_transcripts"})
# Collections that carry image embeddings
_IMAGE_COLLECTIONS = frozenset({"image_frames", "video_frames"})

# source_type values that are primarily non-text (BM25 gets little signal from them)
_VISUAL_SOURCE_TYPES = frozenset({"photo", "video"})
_AUDIO_SOURCE_TYPES = frozenset({"audio"})


def _torch_device() -> str:
    """Return 'cuda' when a GPU is available to torch, else 'cpu'."""
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class Retriever:
    """Hybrid retriever combining BM25 and Qdrant dense vector search.

    Usage::

        retriever = Retriever(store, vector_store)
        hits = retriever.retrieve("what happened last summer", limit=20)
        cards = SessionGrouper().group(hits)
    """

    def __init__(
        self,
        store: MetadataStore,
        vector_store: Any | None = None,  # VectorStore, optional
    ) -> None:
        self._store = store
        self._vs = vector_store
        # BM25 cache: (chunk_count, BM25Okapi_instance, [chunk_ids])
        self._bm25_cache: tuple[int, Any, list[str]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        signals: QuerySignals | None = None,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """Run hybrid retrieval and return RRF-fused results.

        Parameters
        ----------
        query:
            Raw user query string.
        signals:
            Pre-computed :class:`QuerySignals`; if omitted, no dense retrieval
            is attempted without a text embedder.
        limit:
            Maximum number of hits to return.
        filters:
            Optional metadata pre-filters (``source_type``, ``session_id``).
        """
        ranked_lists: dict[str, list[tuple[str, float]]] = {}

        # Determine whether the caller has locked to a specific source_type.
        # When locked, we respect it; when open, we retrieve across all modalities.
        locked_source_type: str | None = (filters or {}).get("source_type")

        # BM25 sparse path (always available, covers all modalities via enriched corpus)
        bm25 = self._bm25_retrieve(query, limit=limit, filters=filters)
        if bm25:
            ranked_lists["bm25"] = bm25

        # Dense paths (require vector_store)
        if self._vs is not None and getattr(self._vs, "available", False):
            text_vec = self._embed_query_text(query)
            if text_vec is not None:
                # Dense text/transcript search. Photos now carry OCR-derived text
                # chunks (source_type=photo), so run this even under a photo/video
                # lock — the source_type filter keeps results within the locked type.
                for col in _TEXT_COLLECTIONS:
                    results = self._vs.search(col, text_vec, limit=limit, filters=filters)
                    if results:
                        ranked_lists[f"dense_{col}"] = [
                            (r["payload"].get("chunk_id", r["id"]), r["score"])
                            for r in results
                        ]

            # CLIP text-to-image: fire when visual/video intent is detected OR when
            # no source_type filter is set (cross-modal: a text query may match photos)
            run_clip = (
                signals is not None
                and (
                    signals.visual_intent
                    or signals.video_intent
                    or (
                        # No explicit modality signal → try all modalities including visual
                        not signals.modality_intents
                        and locked_source_type is None
                    )
                )
            ) or locked_source_type in _VISUAL_SOURCE_TYPES

            if run_clip:
                clip_vec = self._embed_query_clip(query)
                if clip_vec is not None:
                    # Respect a visual-type lock so a "photo" filter doesn't return
                    # video frames (and vice versa); pass filters as a safety net.
                    if locked_source_type == "photo":
                        image_cols: tuple[str, ...] = ("image_frames",)
                    elif locked_source_type == "video":
                        image_cols = ("video_frames",)
                    else:
                        image_cols = tuple(_IMAGE_COLLECTIONS)
                    for col in image_cols:
                        results = self._vs.search(col, clip_vec, limit=limit, filters=filters)
                        if results:
                            ranked_lists[f"clip_{col}"] = [
                                (r["payload"].get("chunk_id", r["id"]), r["score"])
                                for r in results
                            ]

        if not ranked_lists:
            return []

        return self._fuse_and_hydrate(ranked_lists, limit)

    def retrieve_by_image(
        self,
        image_path: str,
        text: str | None = None,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        """Match an uploaded image (with optional accompanying text) against the index.

        The image is embedded with CLIP and searched against the image/video
        collections. When ``text`` is supplied it is blended into the CLIP query
        vector (multimodal search) and also drives keyword (BM25) and text-dense
        retrieval so the words steer the visual match.
        """
        img_vec = self.embed_image(image_path)
        if img_vec is None:
            return []

        query_vec = img_vec
        if text:
            txt_vec = self._embed_query_clip(text)
            if txt_vec is not None and len(txt_vec) == len(img_vec):
                combined = [a + b for a, b in zip(img_vec, txt_vec, strict=False)]
                norm = sum(v * v for v in combined) ** 0.5 or 1.0
                query_vec = [v / norm for v in combined]

        ranked_lists: dict[str, list[tuple[str, float]]] = {}
        if self._vs is not None and getattr(self._vs, "available", False):
            for col in _IMAGE_COLLECTIONS:
                results = self._vs.search(col, query_vec, limit=limit, filters=filters)
                if results:
                    ranked_lists[f"clip_{col}"] = [
                        (r["payload"].get("chunk_id", r["id"]), r["score"]) for r in results
                    ]
            if text:
                text_vec = self._embed_query_text(text)
                if text_vec is not None:
                    for col in _TEXT_COLLECTIONS:
                        results = self._vs.search(col, text_vec, limit=limit, filters=filters)
                        if results:
                            ranked_lists[f"dense_{col}"] = [
                                (r["payload"].get("chunk_id", r["id"]), r["score"]) for r in results
                            ]

        if text:
            bm25 = self._bm25_retrieve(text, limit=limit, filters=filters)
            if bm25:
                ranked_lists["bm25"] = bm25

        if not ranked_lists:
            return []

        return self._fuse_and_hydrate(ranked_lists, limit)

    def _fuse_and_hydrate(
        self, ranked_lists: dict[str, list[tuple[str, float]]], limit: int
    ) -> list[RetrievalHit]:
        """Fuse the per-retriever ranked lists with RRF and hydrate from SQLite."""
        # RRF fusion — immune to score-scale differences between modalities
        fusion = RRFFusion()
        fused = fusion.fuse(ranked_lists)[:limit]

        # Hydrate hits from SQLite
        chunk_id_order = [cid for cid, _ in fused]
        score_map = {cid: score for cid, score in fused}
        rows = self._store.fetch_chunks_by_ids(set(chunk_id_order))
        rows_by_id = {str(row["chunk_id"]): row for row in rows}

        hits: list[RetrievalHit] = []
        for chunk_id in chunk_id_order:
            if chunk_id not in rows_by_id:
                continue
            row = rows_by_id[chunk_id]
            meta = json.loads(row["metadata_json"] or "{}")
            ts = (
                datetime.fromisoformat(str(row["timestamp_utc"]))
                if row["timestamp_utc"]
                else None
            )
            # Determine which retrievers contributed (for rationale)
            rationale = [r for r in ranked_lists if any(c == chunk_id for c, _ in ranked_lists[r])]
            rationale = rationale or ["rrf"]

            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    source_type=str(row["source_type"]),  # type: ignore[arg-type]
                    file_path=Path(str(row["file_path"])),
                    score=score_map[chunk_id],
                    rationale=rationale,
                    timestamp_utc=ts,
                    session_id=row["session_id"],
                    snippet=(str(row["text"])[:250] if row["text"] else None),
                    place_name=row["place_name"],
                    metadata=meta,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # BM25 path
    # ------------------------------------------------------------------

    def _bm25_retrieve(
        self,
        query: str,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError:
            return []

        rows = self._store.fetch_chunks()
        if filters:
            rows = [
                row for row in rows
                if all(str(row[key]) == str(value) for key, value in filters.items() if key in row.keys())
            ]
        count = len(rows)
        if count == 0:
            return []

        # Rebuild index if stale (new chunks added since last query)
        use_cache = not filters
        if not use_cache or self._bm25_cache is None or self._bm25_cache[0] != count:
            corpus: list[list[str]] = []
            chunk_ids: list[str] = []
            for row in rows:
                file_path = Path(str(row["file_path"]))
                # Pull extra searchable fields from metadata_json for non-text modalities
                meta: dict[str, Any] = {}
                try:
                    meta = json.loads(row["metadata_json"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass

                # Collect all text signals for this chunk across all modalities
                parts = [
                    str(row["text"] or ""),
                    str(row["search_text"] or ""),
                    # File name and parent directory (e.g. "Paris_trip/IMG_0042.jpg")
                    file_path.stem.replace("_", " ").replace("-", " "),
                    str(file_path.parent.name).replace("_", " ").replace("-", " "),
                    str(row["source_type"] or ""),
                    str(row["place_name"] or ""),
                    # Metadata fields that ingestors populate for non-text modalities
                    str(meta.get("caption", "")),
                    str(meta.get("description", "")),
                    str(meta.get("title", "")),
                    str(meta.get("subject", "")),
                    str(meta.get("tags", "")),
                    str(meta.get("labels", "")),
                    str(meta.get("transcript", "")),
                    str(meta.get("ocr_text", "")),
                    str(meta.get("exif_description", "")),
                    str(meta.get("location", "")),
                    str(meta.get("city", "")),
                    str(meta.get("country", "")),
                    str(meta.get("album", "")),
                    str(meta.get("event", "")),
                    str(meta.get("people", "")),
                    str(meta.get("sender", "")),
                    str(meta.get("recipients", "")),
                    str(meta.get("summary", "")),
                ]
                text = " ".join(p for p in parts if p and p != "None")
                corpus.append(_tokenize(text))
                chunk_ids.append(str(row["chunk_id"]))
            index = BM25Okapi(corpus)
            if use_cache:
                self._bm25_cache = (count, index, chunk_ids)
        else:
            _, index, chunk_ids = self._bm25_cache

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = index.get_scores(q_tokens)
        pairs = sorted(zip(chunk_ids, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [(cid, float(score)) for cid, score in pairs[:limit] if score > 0.0]

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_query_text(self, query: str) -> list[float] | None:
        """Embed the query using the text embedding model."""
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            model_name = os.getenv("LIFELOG_TEXT_EMBEDDING_MODEL", "intfloat/e5-large-v2")
            # Lazily load and cache (on GPU when available)
            if not hasattr(self, "_text_model") or self._text_model_name != model_name:
                self._text_model = SentenceTransformer(model_name, device=_torch_device())
                self._text_model_name = model_name

            # e5 models need query prefix
            text = f"query: {query}" if "e5" in model_name.lower() else query
            vec = self._text_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            return vec[0].tolist() if hasattr(vec[0], "tolist") else list(vec[0])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Text embedding unavailable: %s", exc)
            return None

    def _ensure_clip(self) -> bool:
        """Lazily load and cache the CLIP model, preprocess, and tokenizer."""
        if getattr(self, "_clip_model", None) is not None:
            return True
        try:
            import open_clip  # noqa: PLC0415

            model_name = os.getenv("LIFELOG_IMAGE_MODEL", "ViT-L-14")
            pretrained = os.getenv("LIFELOG_IMAGE_PRETRAINED", "openai")
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._clip_tokenizer = open_clip.get_tokenizer(model_name)
            self._clip_device = _torch_device()
            self._clip_model.to(self._clip_device)
            self._clip_model.eval()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("CLIP unavailable: %s", exc)
            return False

    def _embed_query_clip(self, query: str) -> list[float] | None:
        """Embed query text using the CLIP text encoder."""
        if not self._ensure_clip():
            return None
        try:
            import torch  # noqa: PLC0415

            tokens = self._clip_tokenizer([query]).to(self._clip_device)
            with torch.no_grad():
                vec = self._clip_model.encode_text(tokens)
                vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec[0].cpu().tolist()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CLIP text embedding failed: %s", exc)
            return None

    def embed_image(self, path: str) -> list[float] | None:
        """Embed an image file with the CLIP image encoder (normalized vector)."""
        if not self._ensure_clip():
            return None
        try:
            import torch  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            image = Image.open(path).convert("RGB")
            tensor = self._clip_preprocess(image).unsqueeze(0).to(self._clip_device)
            with torch.no_grad():
                vec = self._clip_model.encode_image(tensor)
                vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec[0].cpu().tolist()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CLIP image embedding failed: %s", exc)
            return None
