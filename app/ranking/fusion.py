"""Reciprocal Rank Fusion — combines multiple ranked lists into a single ranking."""

from __future__ import annotations


class RRFFusion:
    """Reciprocal Rank Fusion (RRF) combiner.

    RRF is immune to score-scale differences between modalities, so no
    normalisation is needed before calling :meth:`fuse`.

    Each ranked list contributes ``1 / (k + rank)`` to the aggregate score,
    where *rank* is 0-based (best result = 0).  Higher aggregate scores win.

    Parameters
    ----------
    k:
        Smoothing constant (default 60, as per the canonical RRF paper).
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def fuse(
        self,
        ranked_lists: dict[str, list[tuple[str, float]]],
    ) -> list[tuple[str, float]]:
        """Merge *ranked_lists* into one ranked list using RRF.

        Parameters
        ----------
        ranked_lists:
            Mapping of retriever name → list of ``(chunk_id, score)`` tuples
            sorted by descending score.

        Returns
        -------
        list[tuple[str, float]]
            ``(chunk_id, rrf_score)`` pairs sorted by descending RRF score.
        """
        scores: dict[str, float] = {}
        for _, hits in ranked_lists.items():
            for rank, (chunk_id, _) in enumerate(hits):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
