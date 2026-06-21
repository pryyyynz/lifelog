"""Unit tests for cross-modality session card merging and diverse sorting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.models.contracts import RetrievalHit
from app.ranking.grouper import SessionGrouper

_T0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _hit(
    chunk_id: str,
    source_type: str,
    score: float,
    ts: datetime,
    session_id: str,
    place_name: str | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        source_type=source_type,  # type: ignore
        file_path=Path(f"/fake/{chunk_id}.ext"),
        score=score,
        rationale=["test"],
        timestamp_utc=ts,
        session_id=session_id,
        snippet=f"Snippet for {chunk_id}",
        place_name=place_name,
    )


class TestMultiModalityMerging:
    def test_merges_close_overlapping_sessions(self) -> None:
        # Paris session hits - all close in time, compatible place name
        h1 = _hit("photo1", "photo", 0.9, _T0, "session_photo", "Paris")
        h2 = _hit("audio1", "audio", 0.85, _T0 + timedelta(minutes=10), "session_audio", "Paris")
        h3 = _hit("text1", "text", 0.75, _T0 + timedelta(minutes=20), "session_text", "Paris")

        # London hit - far away in time and different place
        h4 = _hit("photo_london", "photo", 0.8, _T0 + timedelta(hours=5), "session_london", "London")

        grouper = SessionGrouper(top_n=5, merge_radius_secs=1800.0)
        cards = grouper.group([h1, h2, h3, h4])

        # We expect 2 session cards: one merged for Paris, one for London
        assert len(cards) == 2

        # Card 1 should be the merged Paris card since its max score is 0.9, and London's is 0.8
        paris_card = cards[0]
        assert "merged_" in paris_card.session_id
        assert "session_photo" in paris_card.session_id
        assert "session_audio" in paris_card.session_id
        assert "session_text" in paris_card.session_id
        assert paris_card.score == 0.9
        assert len(paris_card.hits) == 3
        assert set(paris_card.modalities) == {"photo", "audio", "text"}

        # Verification of diverse modality sorting:
        # Hits should lead with diverse modalities: photo (0.9), audio (0.85), text (0.75)
        assert paris_card.hits[0].source_type == "photo"
        assert paris_card.hits[1].source_type == "audio"
        assert paris_card.hits[2].source_type == "text"

        # Card 2 should be the London card
        london_card = cards[1]
        assert london_card.session_id == "session_london"
        assert london_card.score == 0.8
        assert len(london_card.hits) == 1
        assert london_card.modalities == ["photo"]

    def test_does_not_merge_different_places(self) -> None:
        # Close in time but different place names -> should NOT merge
        h1 = _hit("photo1", "photo", 0.9, _T0, "session_paris", "Paris")
        h2 = _hit("audio1", "audio", 0.85, _T0 + timedelta(minutes=5), "session_london", "London")

        grouper = SessionGrouper(top_n=5, merge_radius_secs=1800.0)
        cards = grouper.group([h1, h2])

        assert len(cards) == 2
        assert cards[0].session_id == "session_paris"
        assert cards[1].session_id == "session_london"

    def test_merges_place_with_no_place_context(self) -> None:
        # Close in time, one has place name, the other doesn't -> should merge!
        h1 = _hit("photo1", "photo", 0.9, _T0, "session_paris", "Paris")
        h2 = _hit("audio1", "audio", 0.85, _T0 + timedelta(minutes=5), "session_none", None)

        grouper = SessionGrouper(top_n=5, merge_radius_secs=1800.0)
        cards = grouper.group([h1, h2])

        assert len(cards) == 1
        assert "merged_" in cards[0].session_id

    def test_sorting_diverse_modality(self) -> None:
        # Group with multiple hits of same modality to test diverse list sorting
        # photo: 0.9, 0.6
        # audio: 0.8
        # text: 0.7
        h1 = _hit("p1", "photo", 0.9, _T0, "s1")
        h2 = _hit("p2", "photo", 0.6, _T0 + timedelta(seconds=1), "s1")
        h3 = _hit("a1", "audio", 0.8, _T0 + timedelta(seconds=2), "s1")
        h4 = _hit("t1", "text", 0.7, _T0 + timedelta(seconds=3), "s1")

        grouper = SessionGrouper()
        cards = grouper.group([h1, h2, h3, h4])

        assert len(cards) == 1
        hits = cards[0].hits
        # Expected: lead hits first (best photo 0.9, best audio 0.8, best text 0.7), then remaining (photo 0.6)
        assert [h.chunk_id for h in hits] == ["p1", "a1", "t1", "p2"]
