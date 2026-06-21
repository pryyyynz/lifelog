"""Tests for Section 13: Retrieval, Hybrid Search, Fusion, and Re-ranking."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.models.contracts import RetrievalHit, SessionCard
from app.ranking.fusion import RRFFusion
from app.ranking.grouper import SessionGrouper
from app.ranking.reranker import CrossEncoderReranker, TemporalReranker
from app.retrieval.retriever import Retriever
from app.storage.metadata import MetadataStore
from app.models.contracts import NormalizedChunkRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_T0 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)


def _hit(
    chunk_id: str,
    score: float,
    session_id: str | None = None,
    ts: datetime | None = None,
    snippet: str | None = "some text",
    source_type: str = "text",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_type=source_type,  # type: ignore[arg-type]
        file_path=Path(f"/fake/{chunk_id}.txt"),
        score=score,
        rationale=["bm25"],
        timestamp_utc=ts,
        session_id=session_id,
        snippet=snippet,
    )


def _chunk_record(chunk_id: str, ts: datetime | None, session_id: str | None = None) -> NormalizedChunkRecord:
    return NormalizedChunkRecord(
        chunk_id=chunk_id,
        source_type="text",
        file_path=Path("/fake/file.txt"),
        text=f"text for {chunk_id}",
        timestamp_utc=ts,
        session_id=session_id,
        metadata={"chunk_identity": f"{chunk_id}:0"},
    )


# ---------------------------------------------------------------------------
# RRFFusion
# ---------------------------------------------------------------------------


class TestRRFFusion:
    def test_single_list_preserves_order(self) -> None:
        fusion = RRFFusion(k=60)
        ranked = [("a", 1.0), ("b", 0.9), ("c", 0.5)]
        result = fusion.fuse({"bm25": ranked})
        ids = [cid for cid, _ in result]
        assert ids == ["a", "b", "c"]

    def test_higher_rank_beats_lower(self) -> None:
        fusion = RRFFusion(k=60)
        # "a" is rank 0 in bm25 (best) → higher score than "b" at rank 1
        result = fusion.fuse({"bm25": [("a", 1.0), ("b", 0.5)]})
        assert result[0][0] == "a"

    def test_rrf_score_formula(self) -> None:
        """Verify 1/(k+rank+1) formula."""
        k = 60
        fusion = RRFFusion(k=k)
        # Two lists: "a" is rank 0 in both; "b" rank 1 in bm25 only
        result_dict = dict(
            fusion.fuse({"bm25": [("a", 1.0), ("b", 0.9)], "dense": [("a", 0.8)]})
        )
        # a: 1/61 + 1/61 = 2/61
        # b: 1/62
        assert result_dict["a"] == pytest.approx(2 / 61, rel=1e-5)
        assert result_dict["b"] == pytest.approx(1 / 62, rel=1e-5)

    def test_item_appearing_in_multiple_lists_scores_higher(self) -> None:
        fusion = RRFFusion(k=60)
        result = dict(
            fusion.fuse(
                {
                    "bm25": [("a", 1.0), ("b", 0.8)],
                    "dense": [("a", 0.9), ("c", 0.7)],
                }
            )
        )
        # "a" appears in both lists, should beat "b" and "c" which appear in only one
        assert result["a"] > result["b"]
        assert result["a"] > result["c"]

    def test_empty_ranked_lists_returns_empty(self) -> None:
        fusion = RRFFusion()
        assert fusion.fuse({}) == []

    def test_empty_individual_list_is_ignored(self) -> None:
        fusion = RRFFusion()
        result = fusion.fuse({"bm25": [("a", 1.0)], "dense": []})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TemporalReranker
# ---------------------------------------------------------------------------


class TestTemporalReranker:
    def test_exact_match_increases_score(self) -> None:
        reranker = TemporalReranker(tau_days=7, alpha=0.5)
        hit = _hit("a", score=1.0, ts=_T0)
        result = reranker.rerank([hit], target_dt=_T0)
        assert result[0].score > 1.0  # boosted

    def test_perfect_match_boost_equals_alpha(self) -> None:
        reranker = TemporalReranker(tau_days=7, alpha=0.5)
        hit = _hit("a", score=1.0, ts=_T0)
        result = reranker.rerank([hit], target_dt=_T0)
        expected = 1.0 * (1.0 + 0.5)
        assert result[0].score == pytest.approx(expected)

    def test_distant_hit_barely_boosted(self) -> None:
        reranker = TemporalReranker(tau_days=7, alpha=0.5)
        far_ts = _T0 - timedelta(days=365)
        hit = _hit("a", score=1.0, ts=far_ts)
        result = reranker.rerank([hit], target_dt=_T0)
        # Score should be barely above 1.0
        assert result[0].score < 1.001

    def test_hit_without_timestamp_passes_through_unchanged(self) -> None:
        reranker = TemporalReranker()
        hit = _hit("a", score=0.8, ts=None)
        result = reranker.rerank([hit], target_dt=_T0)
        assert result[0].score == pytest.approx(0.8)

    def test_closer_hit_ranks_higher(self) -> None:
        # near: score=0.8, ts=1 hr from target → large boost
        # far:  score=1.0, ts=30 days from target → negligible boost
        # With tau=7d, alpha=0.5:
        #   near_boost ≈ 0.5 * exp(-3600/604800) ≈ 0.497  → new_score = 0.8 * 1.497 = 1.197
        #   far_boost  ≈ 0.5 * exp(-2592000/604800) ≈ 0.007 → new_score = 1.0 * 1.007 = 1.007
        reranker = TemporalReranker(tau_days=7, alpha=0.5)
        near = _hit("near", score=0.8, ts=_T0 + timedelta(hours=1))
        far = _hit("far", score=1.0, ts=_T0 + timedelta(days=30))
        result = reranker.rerank([near, far], target_dt=_T0)
        # "near" should overtake "far" despite lower original score
        assert result[0].chunk_id == "near"

    def test_rationale_updated(self) -> None:
        reranker = TemporalReranker()
        hit = _hit("a", score=1.0, ts=_T0)
        result = reranker.rerank([hit], target_dt=_T0)
        assert any("temporal_boost" in r for r in result[0].rationale)


# ---------------------------------------------------------------------------
# CrossEncoderReranker (unavailable model — passthrough)
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerUnavailable:
    def test_passthrough_when_model_missing(self) -> None:
        # Use a nonsense model path to force unavailability
        reranker = CrossEncoderReranker(model_path="/nonexistent/model", top_n=40)
        hits = [_hit("a", 1.0), _hit("b", 0.8)]
        result = reranker.rerank(hits, "my query")
        assert result == hits  # unchanged

    def test_available_false_when_model_missing(self) -> None:
        reranker = CrossEncoderReranker(model_path="/nonexistent/model")
        assert reranker.available is False

    def test_empty_hits_returns_empty(self) -> None:
        reranker = CrossEncoderReranker(model_path="/nonexistent/model")
        assert reranker.rerank([], "query") == []


class _FakeCrossEncoder:
    """Stand-in model: high logit when the query term appears in the doc text.

    Asserts it is never asked to score an empty document — text-less hits must be
    filtered out before prediction.
    """

    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            assert doc.strip(), "cross-encoder was handed an empty document"
            scores.append(5.0 if query.lower() in doc.lower() else -5.0)
        return scores


class TestCrossEncoderRerankerScoring:
    def _reranker(self) -> CrossEncoderReranker:
        reranker = CrossEncoderReranker(model_path="/nonexistent/model")
        reranker._model = _FakeCrossEncoder()  # noqa: SLF001
        return reranker

    def test_textless_hit_not_buried_by_text_hit(self) -> None:
        """A photo matched purely by CLIP (no text) must keep its standing."""
        reranker = self._reranker()
        photo = _hit("photo", score=0.05, snippet=None, source_type="photo")
        text = _hit("text", score=0.04, snippet="totally unrelated words")
        result = reranker.rerank([photo, text], "beach")
        ids = [h.chunk_id for h in result]
        assert ids.index("photo") < ids.index("text")

    def test_relevant_text_boosted_above_irrelevant(self) -> None:
        reranker = self._reranker()
        irrelevant = _hit("irr", score=0.049, snippet="quarterly tax documents")
        relevant = _hit("rel", score=0.030, snippet="a lovely day at the beach")
        result = reranker.rerank([irrelevant, relevant], "beach")
        assert result[0].chunk_id == "rel"

    def test_score_stays_on_fused_scale(self) -> None:
        """CE applies a bounded boost, not a raw logit, so scores stay comparable."""
        reranker = self._reranker()
        hit = _hit("a", score=0.04, snippet="a day at the beach")
        result = reranker.rerank([hit], "beach")
        # new_score = 0.04 * (1 + 1.0 * sigmoid(5)) ≈ 0.04 * 1.993
        assert 0.04 < result[0].score < 0.08

    def test_rationale_records_cross_encoder(self) -> None:
        reranker = self._reranker()
        hit = _hit("a", score=0.04, snippet="a day at the beach")
        result = reranker.rerank([hit], "beach")
        assert any("cross_encoder" in r for r in result[0].rationale)


# ---------------------------------------------------------------------------
# SessionGrouper
# ---------------------------------------------------------------------------


class TestSessionGrouper:
    def test_groups_by_session_id(self) -> None:
        grouper = SessionGrouper(top_n=10)
        hits = [
            _hit("a", 1.0, session_id="sess1"),
            _hit("b", 0.9, session_id="sess1"),
            _hit("c", 0.8, session_id="sess2"),
        ]
        cards = grouper.group(hits)
        assert len(cards) == 2
        # Best session is sess1 (score 1.0)
        assert cards[0].session_id == "sess1"
        assert len(cards[0].hits) == 2

    def test_none_session_gets_solo_card(self) -> None:
        grouper = SessionGrouper(top_n=10)
        hits = [_hit("a", 1.0, session_id=None)]
        cards = grouper.group(hits)
        assert len(cards) == 1

    def test_top_n_caps_output(self) -> None:
        grouper = SessionGrouper(top_n=2)
        hits = [_hit(f"x{i}", float(10 - i), session_id=f"sess{i}") for i in range(5)]
        cards = grouper.group(hits)
        assert len(cards) == 2

    def test_cards_sorted_by_score_descending(self) -> None:
        grouper = SessionGrouper(top_n=10)
        hits = [
            _hit("a", 0.5, session_id="sess1"),
            _hit("b", 0.9, session_id="sess2"),
        ]
        cards = grouper.group(hits)
        assert cards[0].score >= cards[1].score

    def test_start_end_utc_set(self) -> None:
        t1 = _T0
        t2 = _T0 + timedelta(hours=1)
        grouper = SessionGrouper(top_n=10)
        hits = [
            _hit("a", 1.0, session_id="s", ts=t1),
            _hit("b", 0.8, session_id="s", ts=t2),
        ]
        cards = grouper.group(hits)
        assert cards[0].start_utc == t1
        assert cards[0].end_utc == t2

    def test_chronological_order(self) -> None:
        grouper = SessionGrouper(top_n=10)
        hits = [
            _hit("a", 1.0, session_id="early", ts=_T0),
            _hit("b", 0.5, session_id="late", ts=_T0 + timedelta(days=5)),
        ]
        cards = grouper.group_chronological(hits)
        assert cards[0].session_id == "early"
        assert cards[1].session_id == "late"

    def test_empty_hits_returns_empty(self) -> None:
        grouper = SessionGrouper()
        assert grouper.group([]) == []


# ---------------------------------------------------------------------------
# Retriever — BM25 path (no Qdrant required)
# ---------------------------------------------------------------------------


class TestRetrieverBM25:
    def _make_store(self, tmp_path: Path) -> MetadataStore:
        """Populate store with diverse text so BM25 IDF values are non-zero."""
        store = MetadataStore(tmp_path / "test.db")
        # Use diverse texts so query words don't appear in every doc
        texts = [
            "hiking through mountain trails in autumn",
            "office meeting notes about project deadlines",
            "chocolate cake recipe with cream frosting",
        ]
        from app.models.contracts import NormalizedChunkRecord  # noqa: PLC0415

        records = [
            NormalizedChunkRecord(
                chunk_id=f"c{i}",
                source_type="text",
                file_path=Path("/fake/file.txt"),
                text=texts[i],
                timestamp_utc=_T0 + timedelta(hours=i),
                metadata={"chunk_identity": f"c{i}:0"},
            )
            for i in range(3)
        ]
        store.upsert_chunks("src1", records)
        return store

    def test_bm25_finds_matching_chunks(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        retriever = Retriever(store, vector_store=None)
        hits = retriever.retrieve("hiking mountains", limit=10)
        assert len(hits) >= 1
        # All chunks have "hiking" so all should appear
        assert all(isinstance(h, RetrievalHit) for h in hits)

    def test_no_results_for_unrelated_query(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        retriever = Retriever(store, vector_store=None)
        hits = retriever.retrieve("xyznonexistentword", limit=10)
        assert hits == []

    def test_bm25_respects_limit(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        retriever = Retriever(store, vector_store=None)
        hits = retriever.retrieve("hiking", limit=2)
        assert len(hits) <= 2

    def test_hit_contains_expected_fields(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        retriever = Retriever(store, vector_store=None)
        hits = retriever.retrieve("hiking", limit=5)
        hit = hits[0]
        assert hit.chunk_id is not None
        assert hit.score > 0
        assert hit.file_path is not None
        assert hit.source_type in ("text", "email", "photo", "audio", "video", "calendar", "browser_history")
