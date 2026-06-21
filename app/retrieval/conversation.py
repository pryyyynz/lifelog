"""Conversational memory manager for multi-turn lifelog queries (Section 15)."""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

_DEFAULT_TTL_SECONDS: float = 3600.0  # 1 hour idle timeout
_MAX_TURNS: int = 20


@dataclass
class ConversationTurn:
    """A single completed turn in a conversation."""

    query: str
    temporal_range: tuple[datetime, datetime] | None
    session_ids: list[str]
    place_names: list[str]
    result_count: int
    monotonic_ts: float = field(default_factory=time.monotonic)
    updated_at_epoch: float = field(default_factory=time.time)


@dataclass
class ResolvedContext:
    """Enriched context derived from prior conversation turns."""

    effective_query: str
    """The query to execute (may equal original when no references resolved)."""

    session_id_filter: str | None = None
    """If set, restrict retrieval to this session_id."""

    temporal_range_override: tuple[datetime, datetime] | None = None
    """If set, override temporal signals with this range from prior turn."""

    clarification_needed: bool = False
    """True when the reference is ambiguous and needs user input."""

    clarification_options: list[str] = field(default_factory=list)
    """Human-readable options to present when clarification is needed."""

    resolved_from: str | None = None
    """Which prior context was used: 'prior_session', 'prior_temporal', or None."""


# ---------------------------------------------------------------------------
# Reference pattern matching
# ---------------------------------------------------------------------------

_SESSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bmore\s+from\s+that\s+session\b", re.I),
    re.compile(r"\bthat\s+session\b", re.I),
    re.compile(r"\bsame\s+session\b", re.I),
]

_TEMPORAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bthat\s+same\s+(day|week|month)\b", re.I),
    re.compile(r"\bthat\s+(day|week|month)\b", re.I),
    re.compile(r"\bwhat\s+else\s+happened\b", re.I),
    re.compile(r"\banything\s+else\s+from\s+then\b", re.I),
    re.compile(r"\bmore\s+from\s+then\b", re.I),
    re.compile(r"\bmore\s+from\s+that\s+(day|week)\b", re.I),
    re.compile(r"\baround\s+that\s+time\b", re.I),
]


def _matches_session_ref(query: str) -> bool:
    return any(p.search(query) for p in _SESSION_PATTERNS)


def _matches_temporal_ref(query: str) -> bool:
    return any(p.search(query) for p in _TEMPORAL_PATTERNS)


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------


class ConversationManager:
    """Thread-safe in-memory conversation state manager.

    Stores conversation history keyed by ``conversation_id`` and resolves
    forward references ("that same week", "more from that session") by
    anchoring them to prior retrieval results.

    TTL: conversations idle longer than *ttl_seconds* are silently purged.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS, storage_path: Path | None = None) -> None:
        self._ttl = ttl_seconds
        self._storage_path = storage_path
        self._store: dict[str, list[ConversationTurn]] = {}
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_id(self) -> str:
        """Return a fresh, unique conversation ID."""
        return str(uuid.uuid4())

    def store_turn(
        self,
        conv_id: str,
        query: str,
        temporal_range: tuple[datetime, datetime] | None,
        session_ids: list[str],
        place_names: list[str],
        result_count: int,
    ) -> None:
        """Record a completed turn so future queries can reference it."""
        turn = ConversationTurn(
            query=query,
            temporal_range=temporal_range,
            session_ids=session_ids,
            place_names=place_names,
            result_count=result_count,
        )
        with self._lock:
            turns = self._store.setdefault(conv_id, [])
            turns.append(turn)
            if len(turns) > _MAX_TURNS:
                turns[:] = turns[-_MAX_TURNS:]
            self._persist_locked()

    def resolve_context(self, query: str, conv_id: str | None) -> ResolvedContext:
        """Resolve forward references in *query* using prior conversation context.

        Rules (applied in order):
        1. Session references ("that session", "more from that session"):
           - Single prior session → apply as ``session_id_filter``.
           - Multiple prior sessions → request clarification.
        2. Temporal references ("that same week", "what else happened"):
           - Prior temporal range available → apply as ``temporal_range_override``.
        3. No match → return unchanged ``effective_query``.
        """
        self._cleanup_expired()

        if not conv_id:
            return ResolvedContext(effective_query=query)

        with self._lock:
            turns = list(self._store.get(conv_id, []))

        if not turns:
            return ResolvedContext(effective_query=query)

        prior = turns[-1]

        # Session-reference resolution
        if _matches_session_ref(query):
            if len(prior.session_ids) == 1:
                return ResolvedContext(
                    effective_query=query,
                    session_id_filter=prior.session_ids[0],
                    resolved_from="prior_session",
                )
            if len(prior.session_ids) > 1:
                options = [f"Session {sid[:8]}…" for sid in prior.session_ids[:5]]
                return ResolvedContext(
                    effective_query=query,
                    clarification_needed=True,
                    clarification_options=options,
                    resolved_from="prior_session",
                )

        # Temporal-reference resolution
        if _matches_temporal_ref(query) and prior.temporal_range is not None:
            return ResolvedContext(
                effective_query=query,
                temporal_range_override=prior.temporal_range,
                resolved_from="prior_temporal",
            )

        return ResolvedContext(effective_query=query)

    def get_history(self, conv_id: str) -> list[dict[str, Any]]:
        """Return a JSON-serialisable history for a conversation."""
        self._cleanup_expired()
        with self._lock:
            turns = self._store.get(conv_id, [])
            return [
                {
                    "query": t.query,
                    "temporal_range": (
                        [t.temporal_range[0].isoformat(), t.temporal_range[1].isoformat()]
                        if t.temporal_range
                        else None
                    ),
                    "session_ids": t.session_ids,
                    "place_names": t.place_names,
                    "result_count": t.result_count,
                    "updated_at": datetime.fromtimestamp(t.updated_at_epoch).isoformat(),
                }
                for t in turns
            ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cleanup_expired(self) -> None:
        """Remove conversations that have been idle longer than TTL."""
        now = time.monotonic()
        wall_now = time.time()
        changed = False
        with self._lock:
            expired = [
                cid
                for cid, turns in self._store.items()
                if turns
                and (
                    now - turns[-1].monotonic_ts > self._ttl
                    or wall_now - turns[-1].updated_at_epoch > self._ttl
                )
            ]
            for cid in expired:
                del self._store[cid]
                changed = True
            if changed:
                self._persist_locked()

    def _load(self) -> None:
        """Load persisted conversation turns, ignoring malformed storage."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        loaded: dict[str, list[ConversationTurn]] = {}
        conversations = data.get("conversations", {}) if isinstance(data, dict) else {}
        if not isinstance(conversations, dict):
            return

        for conv_id, raw_turns in conversations.items():
            if not isinstance(conv_id, str) or not isinstance(raw_turns, list):
                continue
            turns: list[ConversationTurn] = []
            for raw in raw_turns[-_MAX_TURNS:]:
                if not isinstance(raw, dict) or not isinstance(raw.get("query"), str):
                    continue
                temporal_range = None
                raw_range = raw.get("temporal_range")
                if isinstance(raw_range, list) and len(raw_range) == 2:
                    try:
                        temporal_range = (
                            datetime.fromisoformat(raw_range[0]),
                            datetime.fromisoformat(raw_range[1]),
                        )
                    except (TypeError, ValueError):
                        temporal_range = None
                turns.append(
                    ConversationTurn(
                        query=raw["query"],
                        temporal_range=temporal_range,
                        session_ids=[str(s) for s in raw.get("session_ids", []) if s is not None],
                        place_names=[str(p) for p in raw.get("place_names", []) if p is not None],
                        result_count=int(raw.get("result_count", 0)),
                        updated_at_epoch=float(raw.get("updated_at_epoch", time.time())),
                    )
                )
            if turns:
                loaded[conv_id] = turns
        self._store = loaded

    def _persist_locked(self) -> None:
        """Persist the current store. Caller must hold ``self._lock``."""
        if self._storage_path is None:
            return

        payload: dict[str, Any] = {
            "version": 1,
            "conversations": {
                conv_id: [
                    {
                        "query": turn.query,
                        "temporal_range": (
                            [turn.temporal_range[0].isoformat(), turn.temporal_range[1].isoformat()]
                            if turn.temporal_range
                            else None
                        ),
                        "session_ids": turn.session_ids,
                        "place_names": turn.place_names,
                        "result_count": turn.result_count,
                        "updated_at_epoch": turn.updated_at_epoch,
                    }
                    for turn in turns[-_MAX_TURNS:]
                ]
                for conv_id, turns in self._store.items()
            },
        }

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self._storage_path)
