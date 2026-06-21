"""Tests for Section 12: Sessionization, Temporal Logic, and Query Signal Extraction."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.ingest.session import SessionAssigner, _make_session_id
from app.models.contracts import NormalizedChunkRecord
from app.retrieval.query_analyzer import QueryAnalyzer, QuerySignals, TemporalBoost, VISUAL_KEYWORDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(chunk_id: str, ts: datetime | None, session_id: str | None = None) -> NormalizedChunkRecord:
    return NormalizedChunkRecord(
        chunk_id=chunk_id,
        source_type="text",
        file_path=Path("/fake/file.txt"),
        text="hello",
        timestamp_utc=ts,
        session_id=session_id,
        metadata={"chunk_identity": f"{chunk_id}:0"},
    )


_T0 = datetime(2024, 6, 1, 9, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# SessionAssigner
# ---------------------------------------------------------------------------


class TestSessionAssigner:
    def test_single_record_gets_session_id(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        records = [_rec("a", _T0)]
        result = assigner.assign(records)
        assert result[0].session_id is not None

    def test_two_events_within_window_share_session(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        t1 = _T0
        t2 = _T0 + timedelta(hours=2)
        records = [_rec("a", t1), _rec("b", t2)]
        result = assigner.assign(records)
        assert result[0].session_id == result[1].session_id

    def test_two_events_outside_window_get_different_sessions(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        t1 = _T0
        t2 = _T0 + timedelta(hours=5)
        records = [_rec("a", t1), _rec("b", t2)]
        result = assigner.assign(records)
        assert result[0].session_id != result[1].session_id

    def test_exactly_at_window_boundary_starts_new_session(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        t1 = _T0
        t2 = _T0 + timedelta(hours=4, seconds=1)
        records = [_rec("a", t1), _rec("b", t2)]
        result = assigner.assign(records)
        assert result[0].session_id != result[1].session_id

    def test_events_just_inside_window_share_session(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        t1 = _T0
        t2 = _T0 + timedelta(hours=3, minutes=59)
        records = [_rec("a", t1), _rec("b", t2)]
        result = assigner.assign(records)
        assert result[0].session_id == result[1].session_id

    def test_no_timestamp_record_has_no_session(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        records = [_rec("a", None)]
        result = assigner.assign(records)
        assert result[0].session_id is None

    def test_session_id_is_deterministic(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        records1 = [_rec("a", _T0)]
        records2 = [_rec("a", _T0)]
        r1 = assigner.assign(records1)
        r2 = assigner.assign(records2)
        assert r1[0].session_id == r2[0].session_id

    def test_existing_session_id_not_overwritten(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        records = [_rec("a", _T0, session_id="keep_this")]
        result = assigner.assign(records)
        assert result[0].session_id == "keep_this"

    def test_multiple_sessions_across_day(self) -> None:
        assigner = SessionAssigner(window_hours=4)
        # 3 events: morning, afternoon, next day
        t1 = datetime(2024, 6, 1, 8, 0, 0, tzinfo=UTC)
        t2 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        t3 = datetime(2024, 6, 2, 9, 0, 0, tzinfo=UTC)
        records = [_rec("a", t1), _rec("b", t2), _rec("c", t3)]
        result = assigner.assign(records)
        assert result[0].session_id == result[1].session_id  # morning group
        assert result[1].session_id != result[2].session_id  # next day is separate

    def test_empty_list_returns_empty(self) -> None:
        assigner = SessionAssigner()
        assert assigner.assign([]) == []

    def test_session_id_length_is_16_hex(self) -> None:
        sid = _make_session_id(_T0, "seed_chunk_id")
        assert len(sid) == 16
        int(sid, 16)  # Raises if not valid hex


# ---------------------------------------------------------------------------
# QueryAnalyzer — temporal extraction
# ---------------------------------------------------------------------------


class TestQueryAnalyzerTemporal:
    def setup_method(self) -> None:
        self.analyzer = QueryAnalyzer(use_spacy=False)

    def test_no_temporal_hint(self) -> None:
        signals = self.analyzer.analyze("what did I eat")
        assert signals.temporal_range is None

    def test_last_week(self) -> None:
        signals = self.analyzer.analyze("what happened last week")
        assert signals.temporal_range is not None
        start, end = signals.temporal_range
        assert start < end

    def test_last_year(self) -> None:
        signals = self.analyzer.analyze("photos from last year")
        assert signals.temporal_range is not None

    def test_month_year(self) -> None:
        signals = self.analyzer.analyze("october 2023 trip")
        assert signals.temporal_range is not None
        start, end = signals.temporal_range
        assert start.year == 2023
        assert start.month == 10

    def test_year_only(self) -> None:
        signals = self.analyzer.analyze("in 2022")
        assert signals.temporal_range is not None
        start, end = signals.temporal_range
        assert start.year == 2022
        assert end.year == 2022

    def test_iso_date(self) -> None:
        signals = self.analyzer.analyze("what happened on 2024-03-15")
        assert signals.temporal_range is not None
        start, _ = signals.temporal_range
        assert start.year == 2024
        assert start.month == 3
        assert start.day == 15

    def test_season_last_summer(self) -> None:
        signals = self.analyzer.analyze("memories from last summer")
        assert signals.temporal_range is not None
        start, end = signals.temporal_range
        assert start.month == 6  # June

    def test_season_with_year(self) -> None:
        signals = self.analyzer.analyze("spring 2023 hike")
        assert signals.temporal_range is not None
        start, _ = signals.temporal_range
        assert start.year == 2023
        assert start.month == 3


# ---------------------------------------------------------------------------
# QueryAnalyzer — visual intent
# ---------------------------------------------------------------------------


class TestQueryAnalyzerVisual:
    def setup_method(self) -> None:
        self.analyzer = QueryAnalyzer(use_spacy=False)

    def test_no_visual_intent(self) -> None:
        signals = self.analyzer.analyze("project meeting notes")
        assert signals.visual_intent is False
        assert signals.visual_keyword_count == 0

    def test_photo_keyword(self) -> None:
        signals = self.analyzer.analyze("find that photo of the sunset")
        assert signals.visual_intent is True
        assert signals.visual_keyword_count >= 2  # photo + sunset

    def test_visual_keywords_covered(self) -> None:
        for kw in VISUAL_KEYWORDS:
            signals = self.analyzer.analyze(f"I {kw} something")
            assert signals.visual_intent is True


# ---------------------------------------------------------------------------
# TemporalBoost
# ---------------------------------------------------------------------------


class TestTemporalBoost:
    def test_zero_distance_yields_alpha(self) -> None:
        boost = TemporalBoost(tau_days=7, alpha=0.5)
        ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        score = boost.score(ts, ts)
        assert score == pytest.approx(0.5)

    def test_distance_tau_yields_alpha_over_e(self) -> None:
        boost = TemporalBoost(tau_days=7, alpha=0.5)
        ts = datetime(2024, 6, 1, tzinfo=UTC)
        target = datetime(2024, 6, 8, tzinfo=UTC)  # exactly 7 days = tau
        score = boost.score(ts, target)
        expected = 0.5 * math.exp(-1.0)
        assert score == pytest.approx(expected, rel=1e-5)

    def test_large_distance_yields_near_zero(self) -> None:
        boost = TemporalBoost(tau_days=7, alpha=0.5)
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        target = datetime(2025, 1, 1, tzinfo=UTC)  # 1 year away
        score = boost.score(ts, target)
        assert score < 0.001

    def test_score_is_symmetric(self) -> None:
        boost = TemporalBoost(tau_days=7, alpha=0.5)
        a = datetime(2024, 6, 1, tzinfo=UTC)
        b = datetime(2024, 6, 5, tzinfo=UTC)
        assert boost.score(a, b) == pytest.approx(boost.score(b, a))

    def test_higher_alpha_gives_higher_score(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=UTC)
        boost1 = TemporalBoost(tau_days=7, alpha=0.3)
        boost2 = TemporalBoost(tau_days=7, alpha=0.7)
        assert boost2.score(ts, ts) > boost1.score(ts, ts)
