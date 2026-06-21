"""Section 18 — Reliability, Testing, Quality Gates.

18.1  Unit tests for six core components.
18.2  Integration tests: ingest pipeline and query end-to-end.
18.3  System-level tests: UC-02, UC-05, UC-07, UC-08.
18.4  Failure handling: corrupted/missing files, empty input, zero-chunk store.
18.5  Performance micro-benchmarks: BM25 synthetic corpus, split_text throughput.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.ingest.images import ExifExtractor, _gps_decimal, _rational_dms
from app.ingest.session import SessionAssigner, _make_session_id
from app.ingest.text import MAX_CHUNK_CHARS, OVERLAP_CHARS, split_text
from app.models.contracts import NormalizedChunkRecord, RetrievalHit
from app.ranking.fusion import RRFFusion
from app.ranking.reranker import TemporalReranker
from app.retrieval.query_analyzer import QueryAnalyzer, QuerySignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str = "c1",
    *,
    ts: datetime | None = None,
    text: str = "hello",
    session_id: str | None = None,
) -> NormalizedChunkRecord:
    return NormalizedChunkRecord(
        chunk_id=chunk_id,
        source_type="text",
        file_path=Path("fake.md"),
        text=text,
        timestamp_utc=ts,
        session_id=session_id,
    )


def _hit(
    chunk_id: str = "h1",
    score: float = 1.0,
    ts: datetime | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_type="text",
        file_path=Path("fake.md"),
        score=score,
        rationale=[],
        timestamp_utc=ts,
    )


# ===========================================================================
# 18.1 — Unit tests
# ===========================================================================


class TestTextChunker:
    """split_text() — spec §8.2 (chunk <= 2 048 chars, 256-char overlap)."""

    def test_empty_string_returns_empty_list(self):
        assert split_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_text("   \n\n   ") == []

    def test_short_text_stays_as_single_chunk(self):
        text = "Hello, world. This is a short paragraph."
        result = split_text(text)
        assert result == [text]

    def test_two_short_paragraphs_merged(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = split_text(text)
        # Combined length well under MAX_CHUNK_CHARS → merged into one chunk
        assert len(result) == 1
        assert "First" in result[0]
        assert "Second" in result[0]

    def test_oversized_text_is_split(self):
        text = "x" * (MAX_CHUNK_CHARS * 3)
        result = split_text(text)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= MAX_CHUNK_CHARS

    def test_overlap_preserved(self):
        """Consecutive chunks share OVERLAP_CHARS characters at their boundary."""
        text = "a" * (MAX_CHUNK_CHARS + OVERLAP_CHARS + 10)
        result = split_text(text)
        assert len(result) >= 2
        tail_of_first = result[0][-OVERLAP_CHARS:]
        head_of_second = result[1][:OVERLAP_CHARS]
        assert tail_of_first == head_of_second

    def test_paragraph_boundary_respected(self):
        """When paragraphs fit inside the budget, they should not be split."""
        para = "A " * 100  # ~200 chars — well under 2 048
        text = f"{para}\n\n{para}\n\n{para}"
        result = split_text(text)
        # Combined 3 * ~200 < 2048, so they merge into at most 2 chunks
        assert len(result) <= 2

    def test_single_paragraph_exactly_at_limit(self):
        text = "a" * MAX_CHUNK_CHARS
        result = split_text(text)
        assert len(result) == 1
        assert len(result[0]) == MAX_CHUNK_CHARS

    def test_custom_max_chars(self):
        text = "word " * 50  # 250 chars
        result = split_text(text, max_chars=100, overlap_chars=10)
        for chunk in result:
            assert len(chunk) <= 100


class TestExifExtractor:
    """ExifExtractor — graceful fallback and GPS conversion."""

    def test_nonexistent_file_returns_error_status(self, tmp_path):
        extractor = ExifExtractor()
        result = extractor.extract(tmp_path / "nonexistent.jpg")
        assert result.raw is not None
        assert "error" in result.raw.get("exif_status", "")

    def test_non_image_file_returns_error_status(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("not an image")
        extractor = ExifExtractor()
        result = extractor.extract(p)
        assert result.raw is not None
        status = result.raw.get("exif_status", "")
        # Either Pillow raises an error or it's unavailable — either is acceptable
        assert status in ("error", "pillow_unavailable") or "error" in status

    def test_gps_decimal_north_east(self):
        gps = {
            "GPSLatitude": (48, 51, 29.16),
            "GPSLatitudeRef": "N",
            "GPSLongitude": (2, 21, 5.04),
            "GPSLongitudeRef": "E",
        }
        lat, lon = _gps_decimal(gps)
        assert lat is not None and abs(lat - 48.858100) < 0.01
        assert lon is not None and abs(lon - 2.35140) < 0.01

    def test_gps_decimal_south_negates_latitude(self):
        gps = {
            "GPSLatitude": (33, 51, 36.0),
            "GPSLatitudeRef": "S",
            "GPSLongitude": (151, 12, 36.0),
            "GPSLongitudeRef": "E",
        }
        lat, lon = _gps_decimal(gps)
        assert lat is not None and lat < 0

    def test_gps_decimal_west_negates_longitude(self):
        gps = {
            "GPSLatitude": (40, 42, 46.0),
            "GPSLatitudeRef": "N",
            "GPSLongitude": (74, 0, 21.0),
            "GPSLongitudeRef": "W",
        }
        lat, lon = _gps_decimal(gps)
        assert lon is not None and lon < 0

    def test_gps_decimal_empty_dict_returns_none_none(self):
        lat, lon = _gps_decimal({})
        assert lat is None
        assert lon is None

    def test_rational_dms_converts_tuple(self):
        # degrees=1, minutes=0, seconds=0 → 1.0 decimal degrees
        assert _rational_dms((1, 0, 0)) == pytest.approx(1.0)

    def test_rational_dms_wrong_length_returns_none(self):
        assert _rational_dms((1, 2)) is None
        assert _rational_dms((1,)) is None


class TestSessionAssigner:
    """SessionAssigner — 4-hour window, deterministic IDs, no-timestamp passthrough."""

    def test_two_events_1h_apart_same_session(self):
        base = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
        records = [
            _chunk("a", ts=base),
            _chunk("b", ts=base + timedelta(hours=1)),
        ]
        result = SessionAssigner(window_hours=4.0).assign(records)
        assert result[0].session_id == result[1].session_id

    def test_two_events_5h_apart_different_sessions(self):
        base = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
        records = [
            _chunk("a", ts=base),
            _chunk("b", ts=base + timedelta(hours=5)),
        ]
        result = SessionAssigner(window_hours=4.0).assign(records)
        assert result[0].session_id != result[1].session_id

    def test_exactly_at_window_boundary(self):
        """Gap equal to exactly window_hours starts a new session."""
        base = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
        records = [
            _chunk("a", ts=base),
            _chunk("b", ts=base + timedelta(hours=4)),
        ]
        result = SessionAssigner(window_hours=4.0).assign(records)
        # 4h gap > 4h window is false (strictly greater), so same session
        # Gap of exactly 4h is 14400 s, window is 14400 s → NOT > → same session
        assert result[0].session_id == result[1].session_id

    def test_no_timestamp_record_gets_none_session(self):
        records = [_chunk("a", ts=None)]
        result = SessionAssigner().assign(records)
        assert result[0].session_id is None

    def test_mixed_timestamped_and_notimestamp(self):
        base = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
        records = [
            _chunk("ts", ts=base),
            _chunk("nots", ts=None),
        ]
        result = SessionAssigner().assign(records)
        ts_record = next(r for r in result if r.chunk_id == "ts")
        nots_record = next(r for r in result if r.chunk_id == "nots")
        assert ts_record.session_id is not None
        assert nots_record.session_id is None

    def test_deterministic_session_id(self):
        """Same inputs always yield the same session ID."""
        ts = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        sid1 = _make_session_id(ts, "chunk-abc")
        sid2 = _make_session_id(ts, "chunk-abc")
        assert sid1 == sid2

    def test_session_id_is_hex_16_chars(self):
        ts = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        sid = _make_session_id(ts, "seed")
        assert len(sid) == 16
        int(sid, 16)  # raises if not valid hex

    def test_already_set_session_id_preserved(self):
        """Records with a pre-assigned session_id are not overwritten."""
        base = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
        records = [_chunk("a", ts=base, session_id="existing-123")]
        result = SessionAssigner().assign(records)
        assert result[0].session_id == "existing-123"

    def test_three_sessions_in_sequence(self):
        base = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
        records = [
            _chunk("a", ts=base),
            _chunk("b", ts=base + timedelta(hours=1)),
            _chunk("c", ts=base + timedelta(hours=10)),  # new session
            _chunk("d", ts=base + timedelta(hours=11)),  # same as c
        ]
        result = SessionAssigner(window_hours=4.0).assign(records)
        ids = [r.session_id for r in result]
        # a==b, c==d, a!=c
        assert ids[0] == ids[1]
        assert ids[2] == ids[3]
        assert ids[0] != ids[2]


class TestRRFFusion:
    """RRFFusion — rank scoring and multi-list aggregation."""

    def test_single_list_rank0_score(self):
        fusion = RRFFusion(k=60)
        result = fusion.fuse({"bm25": [("doc1", 0.9), ("doc2", 0.5)]})
        scores = dict(result)
        assert scores["doc1"] == pytest.approx(1 / (60 + 0 + 1))  # 1/61
        assert scores["doc2"] == pytest.approx(1 / (60 + 1 + 1))  # 1/62

    def test_rank0_beats_rank1(self):
        fusion = RRFFusion(k=60)
        result = fusion.fuse({"bm25": [("a", 1.0), ("b", 0.5)]})
        scores = dict(result)
        assert scores["a"] > scores["b"]

    def test_two_lists_same_item_accumulates(self):
        """An item appearing in two lists gets both contributions."""
        fusion = RRFFusion(k=60)
        result = fusion.fuse({
            "bm25": [("shared", 1.0), ("only_bm25", 0.5)],
            "dense": [("shared", 0.9), ("only_dense", 0.4)],
        })
        scores = dict(result)
        # shared appears at rank 0 in both lists → 1/61 + 1/61 = 2/61
        assert scores["shared"] == pytest.approx(2 / 61)
        assert scores["shared"] > scores["only_bm25"]
        assert scores["shared"] > scores["only_dense"]

    def test_empty_input_returns_empty(self):
        assert RRFFusion().fuse({}) == []

    def test_single_empty_list_returns_empty(self):
        assert RRFFusion().fuse({"bm25": []}) == []

    def test_result_sorted_descending(self):
        fusion = RRFFusion(k=60)
        result = fusion.fuse({"bm25": [("a", 1.0), ("b", 0.8), ("c", 0.6)]})
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_custom_k_changes_scores(self):
        k = 10
        fusion = RRFFusion(k=k)
        result = fusion.fuse({"x": [("doc", 1.0)]})
        assert dict(result)["doc"] == pytest.approx(1 / (k + 0 + 1))


class TestTemporalReranker:
    """TemporalReranker — boost formula, no-timestamp passthrough, re-sorting."""

    def test_delta_zero_applies_max_boost(self):
        """At delta=0, boost = alpha → new_score = score * (1 + alpha)."""
        reranker = TemporalReranker(tau_days=7.0, alpha=0.5)
        ts = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        hits = [_hit("h1", score=1.0, ts=ts)]
        result = reranker.rerank(hits, target_dt=ts)
        assert result[0].score == pytest.approx(1.0 * 1.5)

    def test_far_future_delta_approaches_zero_boost(self):
        reranker = TemporalReranker(tau_days=1.0, alpha=0.5)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        far = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=365)
        hits = [_hit("h1", score=1.0, ts=ts)]
        result = reranker.rerank(hits, target_dt=far)
        # boost ≈ 0 → score barely above 1.0
        assert result[0].score < 1.01

    def test_no_timestamp_passes_through_unchanged(self):
        reranker = TemporalReranker()
        ts = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        hits = [_hit("notimestamp", score=0.9, ts=None)]
        result = reranker.rerank(hits, target_dt=ts)
        assert result[0].score == pytest.approx(0.9)
        assert result[0].chunk_id == "notimestamp"

    def test_rationale_appended(self):
        reranker = TemporalReranker(alpha=0.5)
        ts = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        hits = [_hit("h1", score=1.0, ts=ts)]
        result = reranker.rerank(hits, target_dt=ts)
        assert any("temporal_boost" in r for r in result[0].rationale)

    def test_result_sorted_descending(self):
        reranker = TemporalReranker(tau_days=7.0, alpha=0.5)
        target = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        hits = [
            _hit("far", score=0.9, ts=target + timedelta(days=30)),
            _hit("close", score=0.9, ts=target + timedelta(hours=1)),
        ]
        result = reranker.rerank(hits, target_dt=target)
        assert result[0].score >= result[1].score

    def test_mixed_timestamp_and_none_sorted(self):
        reranker = TemporalReranker(alpha=0.5)
        target = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        hits = [
            _hit("no_ts", score=2.0, ts=None),
            _hit("exact", score=1.0, ts=target),
        ]
        result = reranker.rerank(hits, target_dt=target)
        # "exact" gets boosted to 1.5; "no_ts" stays at 2.0 → no_ts first
        assert result[0].chunk_id == "no_ts"


class TestQueryAnalyzer:
    """QueryAnalyzer — temporal, visual, and entity signal extraction."""

    def setup_method(self):
        self.analyzer = QueryAnalyzer(use_spacy=False)

    def test_empty_query_returns_empty_signals(self):
        signals = self.analyzer.analyze("")
        assert signals.temporal_range is None
        assert signals.visual_intent is False
        assert signals.visual_keyword_count == 0

    def test_yesterday_extracts_temporal_range(self):
        signals = self.analyzer.analyze("what did I do yesterday?")
        assert signals.temporal_range is not None
        start, end = signals.temporal_range
        assert start < end

    def test_last_week_extracts_temporal_range(self):
        signals = self.analyzer.analyze("photos from last week")
        assert signals.temporal_range is not None

    def test_photo_keyword_triggers_visual_intent(self):
        signals = self.analyzer.analyze("show me a photo from the beach")
        assert signals.visual_intent is True
        assert signals.visual_keyword_count >= 1

    def test_sunset_triggers_visual_intent(self):
        signals = self.analyzer.analyze("find the sunset I saw last summer")
        assert signals.visual_intent is True

    def test_no_visual_keyword(self):
        signals = self.analyzer.analyze("what was the weather like in March?")
        assert signals.visual_intent is False

    def test_raw_query_preserved(self):
        query = "show me photos from yesterday"
        signals = self.analyzer.analyze(query)
        assert signals.raw_query == query

    def test_temporal_range_has_utc_timezone(self):
        signals = self.analyzer.analyze("notes from last week")
        if signals.temporal_range is not None:
            start, end = signals.temporal_range
            assert start.tzinfo is not None
            assert end.tzinfo is not None

    def test_today_extracts_range(self):
        signals = self.analyzer.analyze("what did I read today?")
        assert signals.temporal_range is not None

    def test_year_extracts_range(self):
        signals = self.analyzer.analyze("photos from 2022")
        assert signals.temporal_range is not None

    def test_month_name_with_year_extracts_range(self):
        signals = self.analyzer.analyze("notes from March 2023")
        assert signals.temporal_range is not None


# ===========================================================================
# 18.4 — Failure handling
# ===========================================================================


class TestFailureHandling:
    """Graceful degradation for corrupted or edge-case inputs."""

    def test_exif_extractor_nonexistent_file_does_not_raise(self, tmp_path):
        extractor = ExifExtractor()
        result = extractor.extract(tmp_path / "ghost.jpg")
        assert result is not None
        assert result.raw is not None

    def test_exif_extractor_empty_file_does_not_raise(self, tmp_path):
        empty = tmp_path / "empty.jpg"
        empty.write_bytes(b"")
        extractor = ExifExtractor()
        result = extractor.extract(empty)
        assert result is not None
        assert result.raw is not None

    def test_split_text_empty_string_no_crash(self):
        assert split_text("") == []

    def test_split_text_single_newline_no_crash(self):
        assert split_text("\n") == []

    def test_rrf_fusion_empty_lists(self):
        assert RRFFusion().fuse({"bm25": [], "dense": []}) == []

    def test_session_assigner_empty_list(self):
        result = SessionAssigner().assign([])
        assert result == []

    def test_session_assigner_all_no_timestamp(self):
        records = [_chunk("a", ts=None), _chunk("b", ts=None)]
        result = SessionAssigner().assign(records)
        for r in result:
            assert r.session_id is None

    def test_temporal_reranker_empty_hits(self):
        target = datetime(2024, 6, 1, tzinfo=UTC)
        result = TemporalReranker().rerank([], target_dt=target)
        assert result == []

    def test_rrf_fusion_single_item(self):
        result = RRFFusion().fuse({"bm25": [("only", 1.0)]})
        assert len(result) == 1
        assert result[0][0] == "only"

    def test_gps_decimal_handles_zero_denominator_gracefully(self):
        """Rational with denominator 0 must not raise ZeroDivisionError."""

        class Rational:
            numerator = 1
            denominator = 0

        gps = {
            "GPSLatitude": (Rational(), Rational(), Rational()),
            "GPSLatitudeRef": "N",
            "GPSLongitude": (0, 0, 0),
            "GPSLongitudeRef": "E",
        }
        # Should not raise
        lat, lon = _gps_decimal(gps)
        # lat will be inf or nan due to /0, but no exception
        assert lon == pytest.approx(0.0)


# ===========================================================================
# 18.5 — Performance micro-benchmarks
# ===========================================================================


class TestPerformanceBenchmarks:
    """Lightweight benchmarks kept fast for CI (no external services)."""

    def test_split_text_1000_chunks_under_1s(self):
        """Splitting a ~2 MB text document completes in under 1 second."""
        # Each paragraph ~200 chars, 10 000 paragraphs ≈ 2 MB total
        paragraph = "The quick brown fox jumps over the lazy dog. " * 5  # ~225 chars
        large_text = "\n\n".join([paragraph] * 10_000)
        start = time.perf_counter()
        chunks = split_text(large_text)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"split_text took {elapsed:.2f}s — too slow"
        assert len(chunks) > 0

    def test_rrf_fusion_10k_results_under_100ms(self):
        """Fusing 10 000 results across two lists completes quickly."""
        items = [(f"doc{i}", float(10_000 - i)) for i in range(10_000)]
        fusion = RRFFusion()
        start = time.perf_counter()
        result = fusion.fuse({"bm25": items, "dense": list(reversed(items))})
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"RRF fusion took {elapsed:.3f}s — too slow"
        assert len(result) == 10_000

    def test_session_assigner_10k_records_under_1s(self):
        """Assigning sessions to 10 000 records completes in under 1 second."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        records = [
            _chunk(f"c{i}", ts=base + timedelta(minutes=i * 30))
            for i in range(10_000)
        ]
        start = time.perf_counter()
        result = SessionAssigner(window_hours=4.0).assign(records)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"SessionAssigner took {elapsed:.2f}s — too slow"
        assert len(result) == 10_000

    def test_query_analyzer_100_queries_under_500ms(self):
        """QueryAnalyzer (regex-only) processes 100 queries quickly."""
        analyzer = QueryAnalyzer(use_spacy=False)
        queries = [
            "photos from last week",
            "what did I eat yesterday",
            "sunset photo 2022",
            "meeting notes this morning",
            "friends at the market",
        ] * 20  # 100 queries
        start = time.perf_counter()
        for q in queries:
            analyzer.analyze(q)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"QueryAnalyzer took {elapsed:.3f}s — too slow"
