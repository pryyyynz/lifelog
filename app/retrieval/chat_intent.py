"""Three-way query intent classifier for lifelog queries.

Replaces the old binary is_conversational_query() / conversational_reply()
with a proper IntentClassifier that classifies every query as:
  - follow_up: references prior conversation context (session/temporal refs)
  - retrieve: should execute the retrieval pipeline
  - chit_chat: small talk / meta queries that skip retrieval
"""

from __future__ import annotations

import enum
import re
from typing import Any

from app.retrieval.conversation import _SESSION_PATTERNS, _TEMPORAL_PATTERNS
from app.retrieval.query_analyzer import _extract_temporal

# ---------------------------------------------------------------------------
# QueryIntent enum
# ---------------------------------------------------------------------------


class QueryIntent(enum.Enum):
    """Three-way intent classification for user queries."""

    follow_up = "follow_up"
    retrieve = "retrieve"
    chit_chat = "chit_chat"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENERIC_FALLBACK: str = (
    "I'm here to help you search your life log. What would you like to find?"
)
"""Single non-prescriptive fallback reply when no LLM is configured."""

# ---------------------------------------------------------------------------
# Regex patterns (kept for classification — not for reply dispatch)
# ---------------------------------------------------------------------------

_META_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"^\s*(hi|hello|hey|howdy)\b",
        r"\b(thanks?|thank you)\b",
        r"\bwhat do you do\b",
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bwhat can you (do|help|search)\b",
        r"\bhow (do you|does this|can i) work\b",
        r"\bwhat is (this|lifelog|life\s*log)\b",
        r"^\s*help\s*$",
        r"\bhelp me use\b",
        r"\bhow are you\b",
        r"\bgood (morning|afternoon|evening)\b",
        r"\bwhat(?:'s| is) your (?:name|purpose)\b",
        r"\btell me about yourself\b",
        r"\bwhat(?:'s| is) lifelog\b",
    )
)

_SEARCH_OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bwhat did i\b",
        r"\bwhere did i\b",
        r"\bwhen did i\b",
        r"\bwho did i\b",
        r"\bshow me\b",
        r"\bfind\b.+\b(photo|picture|video|email|note|journal|memo|calendar)\b",
        r"\bremember\b.+\b(when|where|about)\b",
        r"\bmy\s+(photo|picture|video|email|journal|calendar|note|memo)s?\b",
        r"\bsearch (?:my|for)\b",
        r"\bphotos? from\b",
        r"\bvideos? (?:from|of)\b",
        r"\bnotes? (?:from|about)\b",
    )
)

_PLEASANTRY = re.compile(
    r"^\s*(hi|hello|hey|thanks?|thank you|ok|okay|cool|great|bye|goodbye)\s*[!?.]*\s*$",
    re.I,
)


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """Three-way query intent classifier.

    Classifies every query as ``follow_up``, ``retrieve``, or ``chit_chat``.

    Args:
        llm_client: Optional LLM client with a ``generate(prompt: str) -> str``
            method. When provided, ``chit_chat_reply()`` calls the LLM for a
            dynamic response. When None, returns ``GENERIC_FALLBACK``.
    """

    def __init__(self, llm_client: Any = None) -> None:
        self._llm_client = llm_client

    def classify(self, query: str, *, has_filters: bool = False) -> QueryIntent:
        """Classify *query* into one of three intents.

        Classification order (first match wins):
        1. ``has_filters=True`` → ``retrieve``
        2. Matches ``_SESSION_PATTERNS`` or ``_TEMPORAL_PATTERNS`` → ``follow_up``
        3. ``_extract_temporal(query.lower())`` is not None → ``retrieve``
        4. Matches ``_SEARCH_OVERRIDE_PATTERNS`` → ``retrieve``
        5. Matches ``_PLEASANTRY`` or ``_META_PATTERNS`` → ``chit_chat``
        6. Default → ``retrieve``
        """
        if has_filters:
            return QueryIntent.retrieve

        text = query.strip()
        if not text:
            return QueryIntent.retrieve

        # Follow-up: session or temporal reference patterns (from conversation.py)
        if any(p.search(text) for p in _SESSION_PATTERNS):
            return QueryIntent.follow_up
        if any(p.search(text) for p in _TEMPORAL_PATTERNS):
            return QueryIntent.follow_up

        lower = text.lower()

        # Retrieve: explicit temporal signal
        if _extract_temporal(lower) is not None:
            return QueryIntent.retrieve

        # Retrieve: search override patterns
        for pattern in _SEARCH_OVERRIDE_PATTERNS:
            if pattern.search(lower):
                return QueryIntent.retrieve

        # Chit-chat: pleasantry or meta patterns
        if _PLEASANTRY.match(lower):
            return QueryIntent.chit_chat
        if any(pattern.search(lower) for pattern in _META_PATTERNS):
            return QueryIntent.chit_chat

        # Default: retrieve
        return QueryIntent.retrieve

    def chit_chat_reply(self, query: str) -> str:
        """Return a reply for a chit-chat query.

        If an LLM client is configured, calls ``llm_client.generate(prompt)``
        and returns the generated text. Falls back to ``GENERIC_FALLBACK`` when
        no LLM is configured or the LLM returns an empty response.
        """
        if self._llm_client is not None:
            reply = self._llm_client.generate(query)
            if reply:
                return reply
        return GENERIC_FALLBACK
