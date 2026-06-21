"""Tests for conversational vs search query routing.

Updated to use IntentClassifier and QueryIntent after the bugfix that replaced
is_conversational_query() / conversational_reply() with the three-way classifier.
"""

from __future__ import annotations

import pytest

from app.retrieval.chat_intent import GENERIC_FALLBACK, IntentClassifier, QueryIntent


@pytest.fixture()
def classifier() -> IntentClassifier:
    return IntentClassifier(llm_client=None)


@pytest.mark.parametrize(
    "query",
    [
        "what do you do?",
        "Who are you",
        "hello",
        "hi there",
        "thanks!",
        "how does this work",
        "what can you search",
        "help",
    ],
)
def test_conversational_queries(classifier: IntentClassifier, query: str) -> None:
    """Chit-chat queries classify as chit_chat and produce a non-empty reply."""
    assert classifier.classify(query) == QueryIntent.chit_chat
    reply = classifier.chit_chat_reply(query)
    assert reply  # non-empty
    assert reply == GENERIC_FALLBACK  # no LLM configured → generic fallback


@pytest.mark.parametrize(
    "query",
    [
        "what did I do last summer",
        "photos from the market",
        "show me videos from June",
        "find my journal entries about hiking",
        "emails from Alex yesterday",
    ],
)
def test_search_queries(classifier: IntentClassifier, query: str) -> None:
    """Retrieval queries classify as retrieve."""
    assert classifier.classify(query) == QueryIntent.retrieve


def test_filters_force_retrieve(classifier: IntentClassifier) -> None:
    """has_filters=True always returns retrieve regardless of query text."""
    assert classifier.classify("what do you do?", has_filters=True) == QueryIntent.retrieve
