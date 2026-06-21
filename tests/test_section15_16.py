"""Tests for Section 15 (ConversationManager) and Section 16 (Explainability)."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.models.contracts import NormalizedChunkRecord
from app.retrieval.conversation import ConversationManager
from app.storage.metadata import MetadataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
_T1 = datetime(2024, 6, 2, 10, 0, 0, tzinfo=UTC)
_RANGE = (_T0, _T1)


def _populate_store(store: MetadataStore) -> None:
    # Deliberately varied text so BM25 IDF is positive for each distinct term
    texts = [
        "walked along the river in Paris on a sunny afternoon",
        "morning coffee at the local cafe before commute",
        "visited the museum with friends on the weekend",
    ]
    records = [
        NormalizedChunkRecord(
            chunk_id=f"s{i}",
            source_type="text",
            file_path=Path("/notes/diary.md"),
            text=texts[i],
            timestamp_utc=_T0 + timedelta(days=i),
            metadata={"chunk_identity": f"s{i}:0"},
        )
        for i in range(3)
    ]
    store.upsert_chunks("source_text", records)


@pytest.fixture()
def api_client(tmp_path: Path):
    """FastAPI TestClient with pre-populated store."""
    import importlib
    import os

    os.environ["LIFELOG_SQLITE_PATH"] = str(tmp_path / "test.db")
    os.environ["LIFELOG_DATA_DIR"] = str(tmp_path)
    os.environ["LIFELOG_LOG_DIR"] = str(tmp_path / "logs")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    import app.api.main as api_mod
    importlib.reload(api_mod)

    store = MetadataStore(tmp_path / "test.db")
    _populate_store(store)

    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as client:
        yield client


# ---------------------------------------------------------------------------
# Section 15 — ConversationManager unit tests
# ---------------------------------------------------------------------------


class TestConversationManagerBasics:
    def test_new_id_returns_uuid(self) -> None:
        mgr = ConversationManager()
        cid = mgr.new_id()
        assert len(cid) == 36  # UUID4 string length
        assert cid != mgr.new_id()  # each call unique

    def test_resolve_context_no_history_returns_original(self) -> None:
        mgr = ConversationManager()
        ctx = mgr.resolve_context("what did I do last weekend", None)
        assert ctx.effective_query == "what did I do last weekend"
        assert ctx.session_id_filter is None
        assert ctx.temporal_range_override is None
        assert not ctx.clarification_needed

    def test_resolve_context_unknown_conv_returns_original(self) -> None:
        mgr = ConversationManager()
        ctx = mgr.resolve_context("that day", "nonexistent-id")
        assert ctx.effective_query == "that day"
        assert ctx.temporal_range_override is None

    def test_store_turn_and_get_history(self) -> None:
        mgr = ConversationManager()
        cid = mgr.new_id()
        mgr.store_turn(cid, "hiking trip", _RANGE, ["sess1"], ["Alps"], 5)
        history = mgr.get_history(cid)
        assert len(history) == 1
        assert history[0]["query"] == "hiking trip"
        assert history[0]["session_ids"] == ["sess1"]
        assert history[0]["result_count"] == 5

    def test_history_is_capped_at_20_turns(self) -> None:
        mgr = ConversationManager()
        cid = mgr.new_id()
        for i in range(25):
            mgr.store_turn(cid, f"query {i}", None, [], [], 0)
        assert len(mgr.get_history(cid)) == 20
        # Oldest turn should be gone; most recent retained
        history = mgr.get_history(cid)
        assert history[-1]["query"] == "query 24"
        assert history[0]["query"] == "query 5"

    def test_empty_history_returns_empty_list(self) -> None:
        mgr = ConversationManager()
        assert mgr.get_history("nope") == []

    def test_history_persists_to_disk(self, tmp_path: Path) -> None:
        storage_path = tmp_path / "conversations.json"
        mgr = ConversationManager(storage_path=storage_path)
        cid = mgr.new_id()
        mgr.store_turn(cid, "hiking trip", _RANGE, ["sess1"], ["Alps"], 5)

        reloaded = ConversationManager(storage_path=storage_path)
        history = reloaded.get_history(cid)

        assert len(history) == 1
        assert history[0]["query"] == "hiking trip"
        assert history[0]["session_ids"] == ["sess1"]
        assert reloaded.resolve_context("more from that session", cid).session_id_filter == "sess1"


class TestConversationManagerTTL:
    def test_expired_conversation_not_returned(self) -> None:
        mgr = ConversationManager(ttl_seconds=0.01)
        cid = mgr.new_id()
        mgr.store_turn(cid, "test", None, [], [], 0)
        time.sleep(0.05)
        # resolve_context triggers _cleanup_expired internally
        ctx = mgr.resolve_context("that same week", cid)
        # After expiry, treated as unknown conversation
        assert ctx.temporal_range_override is None


class TestConversationManagerReferenceResolution:
    def _mgr_with_prior(
        self,
        session_ids: list[str],
        temporal_range: tuple[datetime, datetime] | None = None,
    ) -> tuple[ConversationManager, str]:
        mgr = ConversationManager()
        cid = mgr.new_id()
        mgr.store_turn(cid, "original query", temporal_range, session_ids, ["Paris"], 3)
        return mgr, cid

    def test_session_ref_single_session(self) -> None:
        mgr, cid = self._mgr_with_prior(["sess-abc"])
        ctx = mgr.resolve_context("more from that session", cid)
        assert ctx.session_id_filter == "sess-abc"
        assert ctx.resolved_from == "prior_session"
        assert not ctx.clarification_needed

    def test_session_ref_multiple_sessions_triggers_clarification(self) -> None:
        mgr, cid = self._mgr_with_prior(["sess-abc", "sess-def", "sess-ghi"])
        ctx = mgr.resolve_context("that session", cid)
        assert ctx.clarification_needed
        assert len(ctx.clarification_options) == 3
        assert ctx.resolved_from == "prior_session"

    def test_that_same_week_uses_prior_temporal_range(self) -> None:
        mgr, cid = self._mgr_with_prior([], temporal_range=_RANGE)
        ctx = mgr.resolve_context("what else happened that same week", cid)
        assert ctx.temporal_range_override == _RANGE
        assert ctx.resolved_from == "prior_temporal"
        assert not ctx.clarification_needed

    def test_that_day_uses_prior_temporal_range(self) -> None:
        mgr, cid = self._mgr_with_prior([], temporal_range=_RANGE)
        ctx = mgr.resolve_context("what happened that day", cid)
        assert ctx.temporal_range_override == _RANGE

    def test_what_else_happened_uses_prior_temporal_range(self) -> None:
        mgr, cid = self._mgr_with_prior([], temporal_range=_RANGE)
        ctx = mgr.resolve_context("what else happened", cid)
        assert ctx.temporal_range_override == _RANGE

    def test_no_prior_temporal_no_override(self) -> None:
        mgr, cid = self._mgr_with_prior([], temporal_range=None)
        ctx = mgr.resolve_context("that same week", cid)
        assert ctx.temporal_range_override is None

    def test_plain_query_no_override(self) -> None:
        mgr, cid = self._mgr_with_prior(["s1"], temporal_range=_RANGE)
        ctx = mgr.resolve_context("show me photos from Lisbon", cid)
        assert ctx.session_id_filter is None
        assert ctx.temporal_range_override is None
        assert not ctx.clarification_needed

    def test_more_from_then_uses_prior_temporal(self) -> None:
        mgr, cid = self._mgr_with_prior([], temporal_range=_RANGE)
        ctx = mgr.resolve_context("more from then", cid)
        assert ctx.temporal_range_override == _RANGE

    def test_around_that_time_uses_prior_temporal(self) -> None:
        mgr, cid = self._mgr_with_prior([], temporal_range=_RANGE)
        ctx = mgr.resolve_context("around that time", cid)
        assert ctx.temporal_range_override == _RANGE

    def test_same_session_reference(self) -> None:
        mgr, cid = self._mgr_with_prior(["sess-xyz"])
        ctx = mgr.resolve_context("same session", cid)
        assert ctx.session_id_filter == "sess-xyz"

    def test_multiple_turns_uses_most_recent(self) -> None:
        mgr = ConversationManager()
        cid = mgr.new_id()
        mgr.store_turn(cid, "first query", _RANGE, ["old-session"], [], 1)
        new_range = (_T1, _T1 + timedelta(days=1))
        mgr.store_turn(cid, "second query", new_range, ["new-session"], [], 2)
        ctx = mgr.resolve_context("that same week", cid)
        assert ctx.temporal_range_override == new_range


# ---------------------------------------------------------------------------
# Section 15 — API integration (conversation_id threading + reference resolution)
# ---------------------------------------------------------------------------


class TestConversationAPIIntegration:
    def test_fresh_query_returns_conversation_id(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "river walk Paris"})
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert len(data["conversation_id"]) == 36

    def test_follow_up_preserves_conversation_id(self, api_client) -> None:
        resp1 = api_client.post("/query", json={"query": "river walk Paris"})
        cid = resp1.json()["conversation_id"]
        resp2 = api_client.post("/query", json={"query": "more details", "conversation_id": cid})
        assert resp2.json()["conversation_id"] == cid

    def test_clarification_returned_for_ambiguous_session_ref(self, api_client) -> None:
        """Seed the conversation with multiple sessions then ask 'that session'."""
        import app.api.main as api_mod  # noqa: PLC0415

        # Manually plant multi-session prior turn
        cid = api_mod._conv_manager.new_id()  # noqa: SLF001
        api_mod._conv_manager.store_turn(cid, "photos", None, ["s1", "s2", "s3"], [], 9)  # noqa: SLF001

        resp = api_client.post("/query", json={"query": "that session", "conversation_id": cid})
        assert resp.status_code == 200
        data = resp.json()
        assert data["clarification_prompt"] is not None
        assert "Session" in data["clarification_prompt"]
        assert data["sessions"] == []

    def test_query_debug_includes_resolved_from(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "river walk Paris"})
        assert "resolved_from" in resp.json()["query_debug"]

    def test_clarification_prompt_absent_for_normal_query(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "Paris afternoon"})
        data = resp.json()
        assert data.get("clarification_prompt") is None


# ---------------------------------------------------------------------------
# Section 16 — Explainability fields in API response
# ---------------------------------------------------------------------------


class TestExplainabilityFields:
    def _first_primary(self, api_client) -> dict:
        # Use exact tokens from seeded corpus so BM25 (exact-match) returns hits
        resp = api_client.post("/query", json={"query": "Paris afternoon"})
        data = resp.json()
        assert data["sessions"], "expected at least one session"
        return data["sessions"][0]["primary"]

    def test_hit_has_rank_field(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert "rank" in primary
        assert primary["rank"] == 1

    def test_hit_has_timestamp_display(self, api_client) -> None:
        primary = self._first_primary(api_client)
        # timestamp_display may be None if no timestamp, but field must exist
        assert "timestamp_display" in primary

    def test_hit_has_match_reasons(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert "match_reasons" in primary
        assert isinstance(primary["match_reasons"], list)
        assert len(primary["match_reasons"]) >= 1

    def test_secondary_hits_have_incremental_rank(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "Paris afternoon", "top_k": 10})
        data = resp.json()
        for card in data["sessions"]:
            # primary is rank 1
            assert card["primary"]["rank"] == 1
            for i, sec in enumerate(card["secondary"], start=2):
                assert sec["rank"] == i

    def test_match_reasons_not_empty(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert primary["match_reasons"] != []

    def test_hit_source_type_present(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert primary["source_type"] == "text"

    def test_hit_file_path_present(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert "file_path" in primary
        assert primary["file_path"]

    def test_hit_score_present(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert "score" in primary
        assert isinstance(primary["score"], float)

    def test_rationale_list_present(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert "rationale" in primary
        assert isinstance(primary["rationale"], list)

    def test_timestamp_utc_present(self, api_client) -> None:
        primary = self._first_primary(api_client)
        # May be None but field must exist
        assert "timestamp_utc" in primary

    def test_snippet_present(self, api_client) -> None:
        primary = self._first_primary(api_client)
        assert "snippet" in primary
        # Should contain part of our seeded text
        if primary["snippet"]:
            assert any(w in primary["snippet"] for w in ("river", "Paris", "afternoon", "sunny"))

    def test_timestamp_display_format_when_set(self, api_client) -> None:
        """timestamp_display should look like 'Sat 01 Jun 2024 at 10:00'."""
        primary = self._first_primary(api_client)
        if primary["timestamp_display"]:
            # Rough check: contains 4-digit year and colon for time
            assert "2024" in primary["timestamp_display"]
            assert ":" in primary["timestamp_display"]


class TestComputeMatchReasons:
    """Unit test _compute_match_reasons directly."""

    def test_bm25_rationale(self) -> None:
        import app.api.main as api_mod  # noqa: PLC0415

        class _FakeSignals:
            temporal_range = None
            place_names: list = []
            visual_intent = False

        reasons = api_mod._compute_match_reasons(["bm25"], _FakeSignals())  # noqa: SLF001
        assert "keyword match (BM25)" in reasons

    def test_dense_text_rationale(self) -> None:
        import app.api.main as api_mod  # noqa: PLC0415

        class _FakeSignals:
            temporal_range = None
            place_names: list = []
            visual_intent = False

        reasons = api_mod._compute_match_reasons(["dense_text_chunks"], _FakeSignals())  # noqa: SLF001
        assert "semantic text similarity" in reasons

    def test_temporal_boost_added_when_temporal_range_set(self) -> None:
        import app.api.main as api_mod  # noqa: PLC0415

        class _FakeSignals:
            temporal_range = (_T0, _T1)
            place_names: list = []
            visual_intent = False

        reasons = api_mod._compute_match_reasons(["bm25"], _FakeSignals())  # noqa: SLF001
        assert "temporal boost applied" in reasons

    def test_visual_intent_added(self) -> None:
        import app.api.main as api_mod  # noqa: PLC0415

        class _FakeSignals:
            temporal_range = None
            place_names: list = []
            visual_intent = True

        reasons = api_mod._compute_match_reasons([], _FakeSignals())  # noqa: SLF001
        assert "visual intent detected" in reasons

    def test_place_names_added(self) -> None:
        import app.api.main as api_mod  # noqa: PLC0415

        class _FakeSignals:
            temporal_range = None
            place_names = ["Lisbon", "Porto"]
            visual_intent = False

        reasons = api_mod._compute_match_reasons([], _FakeSignals())  # noqa: SLF001
        assert any("Lisbon" in r for r in reasons)

    def test_fallback_relevance_score(self) -> None:
        import app.api.main as api_mod  # noqa: PLC0415

        class _FakeSignals:
            temporal_range = None
            place_names: list = []
            visual_intent = False

        reasons = api_mod._compute_match_reasons([], _FakeSignals())  # noqa: SLF001
        assert reasons == ["relevance score"]
