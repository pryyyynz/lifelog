"""Bugfix tests for query intent classifier.

This file follows the exploratory bugfix workflow:
  - Task 1: Bug condition exploration tests (EXPECTED TO FAIL on unfixed code)
  - Task 2: Preservation property tests (EXPECTED TO PASS on unfixed code)
  - Task 3.3: Unit tests for IntentClassifier (after fix)
  - Task 3.4: Integration tests for POST /query (after fix)
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# HARDCODED_REPLY_SET — the finite set of strings that the old
# conversational_reply() could return. Used in tests to assert the fix
# eliminates them. Defined here (test file only), NOT in production code.
# ---------------------------------------------------------------------------

HARDCODED_REPLY_SET: frozenset[str] = frozenset(
    {
        (
            "You're welcome! Ask me anytime you want to search your journals, photos, "
            "videos, or other indexed memories."
        ),
        (
            "Hi! I'm your Life Log assistant. I search your personal history — try "
            '"what did I do last summer?" or "photos from the market."'
        ),
        (
            "I'm Life Log Search — a local assistant that finds moments across your "
            "text, photos, audio, video, email, calendar, and browser history. "
            "Ask a question about your past and I'll surface matching sessions with "
            "snippets and links to the original files."
        ),
        (
            "I can search everything you've ingested: notes, screenshots, voice memos, "
            "videos, emails, calendar events, and more. Use natural language with times "
            'or places, e.g. "emails from Alex in March" or "rainy day photos."'
        ),
        (
            "Point the app at folders on your machine, run ingest, then ask questions here. "
            "I retrieve relevant chunks, group them by session, and show the best matches — "
            "without sending your data to the cloud."
        ),
        (
            "Use Ingest Data to add sources, then ask questions in this chat. "
            "Filters narrow by type or date. For follow-ups like \"more from that session,\" "
            "stay in the same conversation so I keep context."
        ),
        "I'm doing well, thanks! Ready when you want to search your life log.",
        (
            "I'm here to help you search your personal life log. Ask about a time, place, "
            'or topic — for example "journal entries from June" or "that hiking trip."'
        ),
    }
)


# ===========================================================================
# Task 1 — Bug Condition Exploration Tests
# These tests MUST FAIL on unfixed code (failure confirms the bug exists).
# After the fix (Task 3.1), these tests MUST PASS.
# ===========================================================================


class TestBugConditionExploration:
    """Property 1: Bug Condition — Hardcoded Chit-Chat Replies and Follow-Up Misclassification.

    These tests encode the EXPECTED behavior after the fix.
    They were written to FAIL on unfixed code (confirming the bug existed),
    and now PASS on fixed code (confirming the bug is resolved).

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """

    def test_greeting_reply_is_not_hardcoded(self) -> None:
        """Test 1 — Hardcoded greeting.

        chit_chat_reply("Hi") must NOT return a hardcoded per-pattern string.
        On unfixed code: conversational_reply("Hi") returned the exact hardcoded greeting.
        On fixed code: IntentClassifier.chit_chat_reply("Hi") returns GENERIC_FALLBACK (not hardcoded).
        """
        from app.retrieval.chat_intent import IntentClassifier

        classifier = IntentClassifier(llm_client=None)
        result = classifier.chit_chat_reply("Hi")
        assert result not in HARDCODED_REPLY_SET, (
            f"Bug not fixed: chit_chat_reply('Hi') returned a hardcoded string: {result!r}"
        )

    def test_thanks_reply_is_not_hardcoded(self) -> None:
        """Test 2 — Hardcoded thanks.

        chit_chat_reply("thanks") must NOT return a hardcoded per-pattern string.
        On unfixed code: conversational_reply("thanks") returned the exact hardcoded thanks string.
        On fixed code: IntentClassifier.chit_chat_reply("thanks") returns GENERIC_FALLBACK (not hardcoded).
        """
        from app.retrieval.chat_intent import IntentClassifier

        classifier = IntentClassifier(llm_client=None)
        result = classifier.chit_chat_reply("thanks")
        assert result not in HARDCODED_REPLY_SET, (
            f"Bug not fixed: chit_chat_reply('thanks') returned a hardcoded string: {result!r}"
        )

    def test_follow_up_session_not_misclassified(self) -> None:
        """Test 3 — Follow-up misclassification.

        classify("more from that session") must return follow_up (not chit_chat).
        On unfixed code: is_conversational_query("more from that session") could return True.
        On fixed code: IntentClassifier.classify() returns follow_up for session references.
        """
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        result = classifier.classify("more from that session")
        assert result == QueryIntent.follow_up, (
            f"Bug not fixed: classify('more from that session') returned {result!r} "
            "(should be follow_up — this is a follow-up query, not chit-chat)"
        )

    def test_temporal_follow_up_not_misclassified(self) -> None:
        """Test 4 — Temporal follow-up misclassification.

        classify("what else happened then") must return follow_up (not chit_chat).
        On unfixed code: is_conversational_query("what else happened then") could return True.
        On fixed code: IntentClassifier.classify() returns follow_up for temporal references.
        """
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        result = classifier.classify("what else happened then")
        assert result == QueryIntent.follow_up, (
            f"Bug not fixed: classify('what else happened then') returned {result!r} "
            "(should be follow_up — this is a temporal follow-up query, not chit-chat)"
        )


# ===========================================================================
# Task 2 — Preservation Property Tests
# These tests MUST PASS on BOTH unfixed and fixed code.
# They verify that non-buggy inputs are unaffected by the fix.
# ===========================================================================


class TestPreservationProperties:
    """Property 2: Preservation — Retrieval Queries and Filter-Bearing Queries Always Route to Retrieval.

    **Validates: Requirements 3.1, 3.2**
    """

    def test_has_filters_true_returns_false(self) -> None:
        """Baseline: classify with has_filters=True returns retrieve."""
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        assert classifier.classify("what do you do?", has_filters=True) == QueryIntent.retrieve
        assert classifier.classify("hello", has_filters=True) == QueryIntent.retrieve
        assert classifier.classify("hi there", has_filters=True) == QueryIntent.retrieve
        assert classifier.classify("thanks", has_filters=True) == QueryIntent.retrieve

    def test_search_override_patterns_return_false(self) -> None:
        """Baseline: queries matching _SEARCH_OVERRIDE_PATTERNS return retrieve."""
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        assert classifier.classify("what did I do last summer?") == QueryIntent.retrieve
        assert classifier.classify("photos from the market") == QueryIntent.retrieve
        assert classifier.classify("show me videos from June") == QueryIntent.retrieve

    def test_temporal_signal_returns_false(self) -> None:
        """Baseline: queries with temporal signals return retrieve."""
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        assert classifier.classify("what did I do yesterday") == QueryIntent.retrieve
        assert classifier.classify("show me photos from last summer") == QueryIntent.retrieve
        assert classifier.classify("emails from 2023") == QueryIntent.retrieve


# Property-based preservation tests using hypothesis

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

# Prefixes that match _SEARCH_OVERRIDE_PATTERNS
_SEARCH_PREFIXES = [
    "what did I ",
    "where did I ",
    "when did I ",
    "who did I ",
    "show me ",
    "search my ",
    "search for ",
    "photos from ",
    "videos from ",
    "notes from ",
]

# Temporal signals that trigger _extract_temporal
_TEMPORAL_SIGNALS = [
    "yesterday",
    "today",
    "last week",
    "last month",
    "last year",
    "last summer",
    "last winter",
    "in 2023",
    "in 2022",
    "this year",
    "this week",
]

# Chit-chat / meta queries that should be conversational without filters
_CHIT_CHAT_QUERIES = [
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "what do you do",
    "who are you",
    "help",
    "how are you",
    "good morning",
]


@pytest.mark.skipif(not _HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPreservationPropertyBased:
    """Property-based preservation tests using hypothesis.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        chit_chat=st.sampled_from(_CHIT_CHAT_QUERIES),
        suffix=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")),
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_property_2a_has_filters_always_returns_false(
        self, chit_chat: str, suffix: str
    ) -> None:
        """Property 2a: For any query with has_filters=True, classify returns retrieve.

        **Validates: Requirements 3.1**
        """
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        query = (chit_chat + " " + suffix).strip()
        result = classifier.classify(query, has_filters=True)
        assert result == QueryIntent.retrieve, (
            f"Preservation violated: classify({query!r}, has_filters=True) "
            f"returned {result!r} (should always be retrieve when has_filters=True)"
        )

    @given(
        prefix=st.sampled_from(_SEARCH_PREFIXES),
        suffix=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")),
            min_size=1,
            max_size=30,
        ),
    )
    @settings(max_examples=50)
    def test_property_2b_search_override_patterns_return_false(
        self, prefix: str, suffix: str
    ) -> None:
        """Property 2b: For any query matching _SEARCH_OVERRIDE_PATTERNS, classify returns retrieve.

        **Validates: Requirements 3.2**
        """
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        query = prefix + suffix
        result = classifier.classify(query)
        assert result == QueryIntent.retrieve, (
            f"Preservation violated: classify({query!r}) returned {result!r} "
            "(should be retrieve for search override patterns)"
        )

    @given(
        temporal=st.sampled_from(_TEMPORAL_SIGNALS),
        prefix=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs")),
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_property_2c_temporal_signals_return_false(
        self, temporal: str, prefix: str
    ) -> None:
        """Property 2c: For any query containing a temporal signal, classify returns retrieve.

        **Validates: Requirements 3.2**
        """
        from app.retrieval.chat_intent import IntentClassifier, QueryIntent

        classifier = IntentClassifier(llm_client=None)
        query = (prefix + " " + temporal).strip()
        result = classifier.classify(query)
        assert result == QueryIntent.retrieve, (
            f"Preservation violated: classify({query!r}) returned {result!r} "
            "(should be retrieve for queries with temporal signals)"
        )


# ===========================================================================
# Task 3.3 — Unit tests for IntentClassifier (after fix)
# ===========================================================================


class TestIntentClassifier:
    """Unit tests for IntentClassifier.classify() and IntentClassifier.chit_chat_reply().

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    """

    @pytest.fixture()
    def classifier(self):
        from app.retrieval.chat_intent import IntentClassifier  # type: ignore[attr-defined]

        return IntentClassifier(llm_client=None)

    @pytest.fixture()
    def QueryIntent(self):
        from app.retrieval.chat_intent import QueryIntent as QI  # type: ignore[attr-defined]

        return QI

    @pytest.fixture()
    def GENERIC_FALLBACK(self):
        from app.retrieval.chat_intent import GENERIC_FALLBACK as GF  # type: ignore[attr-defined]

        return GF

    # --- chit_chat intent ---

    @pytest.mark.parametrize("query", ["Hi", "hello", "good morning"])
    def test_classify_greeting_returns_chit_chat(self, classifier, QueryIntent, query: str) -> None:
        assert classifier.classify(query) == QueryIntent.chit_chat

    @pytest.mark.parametrize("query", ["thanks", "thank you"])
    def test_classify_thanks_returns_chit_chat(self, classifier, QueryIntent, query: str) -> None:
        assert classifier.classify(query) == QueryIntent.chit_chat

    @pytest.mark.parametrize("query", ["what do you do?", "who are you", "help"])
    def test_classify_meta_returns_chit_chat(self, classifier, QueryIntent, query: str) -> None:
        assert classifier.classify(query) == QueryIntent.chit_chat

    # --- follow_up intent ---

    @pytest.mark.parametrize(
        "query",
        ["more from that session", "that session", "same session"],
    )
    def test_classify_session_ref_returns_follow_up(self, classifier, QueryIntent, query: str) -> None:
        assert classifier.classify(query) == QueryIntent.follow_up

    @pytest.mark.parametrize(
        "query",
        ["what else happened then", "more from then", "around that time"],
    )
    def test_classify_temporal_ref_returns_follow_up(self, classifier, QueryIntent, query: str) -> None:
        assert classifier.classify(query) == QueryIntent.follow_up

    # --- retrieve intent ---

    @pytest.mark.parametrize(
        "query",
        [
            "what did I do last summer?",
            "photos from the market",
            "show me videos from June",
        ],
    )
    def test_classify_retrieval_queries_returns_retrieve(
        self, classifier, QueryIntent, query: str
    ) -> None:
        assert classifier.classify(query) == QueryIntent.retrieve

    # --- has_filters override ---

    @pytest.mark.parametrize(
        "query",
        ["Hi", "hello", "thanks", "what do you do?", "more from that session"],
    )
    def test_classify_has_filters_true_always_returns_retrieve(
        self, classifier, QueryIntent, query: str
    ) -> None:
        assert classifier.classify(query, has_filters=True) == QueryIntent.retrieve

    # --- chit_chat_reply ---

    def test_chit_chat_reply_no_llm_returns_generic_fallback(
        self, classifier, GENERIC_FALLBACK
    ) -> None:
        reply = classifier.chit_chat_reply("Hi")
        assert reply == GENERIC_FALLBACK

    def test_chit_chat_reply_no_llm_not_in_hardcoded_set(self, classifier) -> None:
        reply = classifier.chit_chat_reply("Hi")
        assert reply not in HARDCODED_REPLY_SET

    def test_chit_chat_reply_with_mock_llm_returns_mock_output(self, QueryIntent) -> None:
        from app.retrieval.chat_intent import IntentClassifier  # type: ignore[attr-defined]

        class MockLLMClient:
            def generate(self, prompt: str) -> str:
                return "Mock LLM response"

        classifier = IntentClassifier(llm_client=MockLLMClient())
        reply = classifier.chit_chat_reply("Hi")
        assert reply == "Mock LLM response"

    def test_chit_chat_reply_with_mock_llm_not_generic_fallback(
        self, GENERIC_FALLBACK
    ) -> None:
        from app.retrieval.chat_intent import IntentClassifier  # type: ignore[attr-defined]

        class MockLLMClient:
            def generate(self, prompt: str) -> str:
                return "Dynamic LLM reply"

        classifier = IntentClassifier(llm_client=MockLLMClient())
        reply = classifier.chit_chat_reply("Hi")
        assert reply != GENERIC_FALLBACK

    # --- edge cases ---

    def test_classify_empty_query_returns_retrieve(self, classifier, QueryIntent) -> None:
        """Edge case: empty query string → classify() returns retrieve (default)."""
        assert classifier.classify("") == QueryIntent.retrieve

    def test_classify_follow_up_takes_priority_over_chit_chat(
        self, classifier, QueryIntent
    ) -> None:
        """Edge case: query matching both follow-up and chit-chat patterns → follow_up wins."""
        # "more from that session" matches _SESSION_PATTERNS (follow_up)
        # but could superficially look like small talk
        result = classifier.classify("more from that session")
        assert result == QueryIntent.follow_up

    def test_classify_temporal_query_returns_retrieve(self, classifier, QueryIntent) -> None:
        """Temporal signal in query → retrieve."""
        assert classifier.classify("what did I do yesterday") == QueryIntent.retrieve
        assert classifier.classify("photos from last summer") == QueryIntent.retrieve


# ===========================================================================
# Task 3.4 — Integration tests for POST /query endpoint
# ===========================================================================


@pytest.fixture()
def api_client_bugfix(tmp_path: Path):
    """FastAPI TestClient with pre-populated store for bugfix integration tests."""
    from app.models.contracts import NormalizedChunkRecord
    from app.storage.metadata import MetadataStore
    from datetime import UTC, datetime, timedelta

    os.environ["LIFELOG_SQLITE_PATH"] = str(tmp_path / "test_bugfix.db")
    os.environ["LIFELOG_DATA_DIR"] = str(tmp_path)
    os.environ["LIFELOG_LOG_DIR"] = str(tmp_path / "logs")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    import app.api.main as api_mod

    importlib.reload(api_mod)

    # Seed the store with some data
    store = MetadataStore(tmp_path / "test_bugfix.db")
    _T0 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
    records = [
        NormalizedChunkRecord(
            chunk_id=f"bugfix_s{i}",
            source_type="text",
            file_path=Path("/notes/diary.md"),
            text=f"walked along the river in Paris on a sunny afternoon day {i}",
            timestamp_utc=_T0 + timedelta(days=i),
            metadata={"chunk_identity": f"bugfix_s{i}:0"},
        )
        for i in range(3)
    ]
    store.upsert_chunks("source_text", records)

    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as client:
        yield client


class TestIntegrationPostQuery:
    """Integration tests for POST /query endpoint with IntentClassifier.

    **Validates: Requirements 2.1, 2.2, 2.3, 3.1, 3.6**
    """

    def test_chit_chat_query_returns_chat_message(self, api_client_bugfix) -> None:
        """Chit-chat query: chat_message is set, sessions is empty, intent is chit_chat."""
        resp = api_client_bugfix.post("/query", json={"query": "Hi", "conversation_id": None})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_message"] is not None
        assert data["sessions"] == []
        assert data["query_debug"]["intent"] == "chit_chat"

    def test_chit_chat_reply_not_in_hardcoded_set(self, api_client_bugfix) -> None:
        """Chit-chat reply must NOT be a member of HARDCODED_REPLY_SET."""
        resp = api_client_bugfix.post("/query", json={"query": "Hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_message"] not in HARDCODED_REPLY_SET, (
            f"Bug not fixed: chat_message is still a hardcoded string: {data['chat_message']!r}"
        )

    def test_retrieval_query_returns_retrieve_intent(self, api_client_bugfix) -> None:
        """Retrieval query: query_debug.intent == 'retrieve'."""
        resp = api_client_bugfix.post(
            "/query", json={"query": "what did I do last summer?"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_debug"]["intent"] == "retrieve"

    def test_filter_override_bypasses_chit_chat(self, api_client_bugfix) -> None:
        """Filter override: POST with filters → intent is 'retrieve' even for chit-chat query."""
        resp = api_client_bugfix.post(
            "/query",
            json={"query": "Hi", "filters": {"source_type": "text"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_debug"]["intent"] == "retrieve"

    def test_chit_chat_stores_turn_for_context(self, api_client_bugfix) -> None:
        """Chit-chat query stores a turn so subsequent follow-up queries have context."""
        import app.api.main as api_mod

        # Send chit-chat query
        resp1 = api_client_bugfix.post("/query", json={"query": "Hi"})
        assert resp1.status_code == 200
        cid = resp1.json()["conversation_id"]

        # Verify turn was stored
        history = api_mod._conv_manager.get_history(cid)  # noqa: SLF001
        assert len(history) >= 1
        assert history[-1]["query"] == "Hi"

    def test_follow_up_query_routes_to_retrieval(self, api_client_bugfix) -> None:
        """Follow-up query: routes to retrieval pipeline, intent is 'follow_up'."""
        import app.api.main as api_mod

        # Seed a prior turn with session data
        cid = api_mod._conv_manager.new_id()  # noqa: SLF001
        from datetime import UTC, datetime, timedelta

        _T0 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        api_mod._conv_manager.store_turn(  # noqa: SLF001
            cid,
            "river walk Paris",
            (_T0, _T0 + timedelta(days=1)),
            ["sess-abc"],
            ["Paris"],
            3,
        )

        resp = api_client_bugfix.post(
            "/query",
            json={"query": "more from that session", "conversation_id": cid},
        )
        assert resp.status_code == 200
        data = resp.json()
        # follow_up routes to retrieval pipeline
        assert data["query_debug"]["intent"] == "follow_up"
