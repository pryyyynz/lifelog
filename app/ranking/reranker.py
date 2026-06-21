"""Re-ranking layers: temporal boost and cross-encoder."""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.contracts import RetrievalHit

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic; maps a cross-encoder logit to [0, 1]."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _doc_text(hit: RetrievalHit) -> str:
    """The document text the cross-encoder can compare against the query."""
    text = hit.snippet or hit.metadata.get("raw_text") or hit.metadata.get("text") or ""
    return text.strip()


# ---------------------------------------------------------------------------
# Temporal reranker
# ---------------------------------------------------------------------------


class TemporalReranker:
    """Boosts hit scores when they are temporally close to a target time.

    Applies ``new_score = score * (1 + boost)`` where
    ``boost = alpha * exp(-|delta| / (tau_days * 86400))``.

    A perfect match (delta = 0 s) adds ``alpha`` fractional boost (up to 50%
    by default). Results without a timestamp pass through unchanged.
    """

    def __init__(self, tau_days: float = 7.0, alpha: float = 0.5) -> None:
        self._tau_secs = tau_days * 86400.0
        self._alpha = alpha

    @classmethod
    def from_environment(cls) -> TemporalReranker:
        tau = float(os.getenv("LIFELOG_TEMPORAL_TAU_DAYS", "7.0"))
        alpha = float(os.getenv("LIFELOG_TEMPORAL_ALPHA", "0.5"))
        return cls(tau_days=tau, alpha=alpha)

    def rerank(self, hits: list[RetrievalHit], target_dt: datetime) -> list[RetrievalHit]:
        """Return hits re-scored and re-sorted by temporal proximity.

        Parameters
        ----------
        hits:
            Input hits in any order.
        target_dt:
            The reference datetime extracted from the query.
        """
        boosted: list[RetrievalHit] = []
        for hit in hits:
            if hit.timestamp_utc is None:
                boosted.append(hit)
                continue
            delta = abs((hit.timestamp_utc - target_dt).total_seconds())
            boost = self._alpha * math.exp(-delta / self._tau_secs)
            new_score = hit.score * (1.0 + boost)
            rationale = list(hit.rationale) + [f"temporal_boost={boost:.4f}"]
            boosted.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    source_type=hit.source_type,
                    file_path=hit.file_path,
                    score=new_score,
                    rationale=rationale,
                    timestamp_utc=hit.timestamp_utc,
                    session_id=hit.session_id,
                    snippet=hit.snippet,
                    thumbnail_path=hit.thumbnail_path,
                    place_name=hit.place_name,
                    metadata=hit.metadata,
                )
            )
        return sorted(boosted, key=lambda h: h.score, reverse=True)


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Re-ranks the top-N hits using a cross-encoder model.

    Only applied to the first ``top_n`` hits (default 40) — cross-encoding
    every candidate would be too slow.

    The cross-encoder only judges hits that carry document text. Text-less hits
    (e.g. photos matched purely by CLIP) keep their fused score and position
    rather than being scored against an empty string. For the rest, the model's
    relevance is applied as a multiplicative boost on the fused score
    (``new_score = score * (1 + alpha * prob)``, mirroring :class:`TemporalReranker`)
    so RRF and temporal signal — the only scale shared with text-less hits — is
    preserved instead of discarded.

    If the model is unavailable, hits are returned unchanged.
    """

    def __init__(self, model_path: str, top_n: int = 40, alpha: float = 1.0) -> None:
        self._top_n = top_n
        self._alpha = alpha
        self._model_path = model_path
        self._model: Any = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._model = CrossEncoder(self._model_path)
            logger.debug("CrossEncoder loaded: %s", self._model_path)
        except Exception as exc:  # noqa: BLE001
            logger.info("CrossEncoder unavailable (%s): %s — pass-through mode", self._model_path, exc)

    @classmethod
    def from_environment(cls) -> CrossEncoderReranker:
        model = os.getenv(
            "LIFELOG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        top_n = int(os.getenv("LIFELOG_CROSS_ENCODER_TOP_N", "40"))
        alpha = float(os.getenv("LIFELOG_CROSS_ENCODER_ALPHA", "1.0"))
        return cls(model_path=model, top_n=top_n, alpha=alpha)

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, hits: list[RetrievalHit], query: str) -> list[RetrievalHit]:
        """Re-rank top-N hits using the cross-encoder.  Rest are appended unchanged."""
        if self._model is None or not hits:
            return hits

        top = hits[: self._top_n]
        rest = hits[self._top_n :]

        # Only cross-encode hits with usable document text. Text-less hits keep
        # their fused score and position (see class docstring).
        scorable = [(idx, hit) for idx, hit in enumerate(top) if _doc_text(hit)]
        if not scorable:
            return hits

        try:
            raw_scores = self._model.predict([(query, _doc_text(hit)) for _, hit in scorable])
            # numpy array or list
            scores: list[float] = (
                raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossEncoder prediction failed: %s", exc)
            return hits

        prob_by_idx = {
            idx: _sigmoid(float(score))
            for (idx, _), score in zip(scorable, scores, strict=False)
        }

        reranked: list[RetrievalHit] = []
        for idx, hit in enumerate(top):
            prob = prob_by_idx.get(idx)
            if prob is None:
                reranked.append(hit)  # no text to judge — leave untouched
                continue
            new_score = hit.score * (1.0 + self._alpha * prob)
            rationale = list(hit.rationale) + [f"cross_encoder={prob:.4f}"]
            reranked.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    source_type=hit.source_type,
                    file_path=hit.file_path,
                    score=new_score,
                    rationale=rationale,
                    timestamp_utc=hit.timestamp_utc,
                    session_id=hit.session_id,
                    snippet=hit.snippet,
                    thumbnail_path=hit.thumbnail_path,
                    place_name=hit.place_name,
                    metadata=hit.metadata,
                )
            )
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked + rest
