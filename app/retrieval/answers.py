"""Local RAG: grounded answer synthesis and query decomposition over Ollama.

Both degrade gracefully — when no LLM client is configured, ``synthesize`` returns
``None`` (UI falls back to plain cards) and ``decompose`` returns the original query.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str, *, system: str | None = ..., num_predict: int = ...) -> str: ...


@dataclass(frozen=True)
class AnswerResult:
    text: str
    cited_session_ids: list[str] = field(default_factory=list)


_ANSWER_SYSTEM = (
    "You are the user's personal life-log assistant. Answer their question from the "
    "numbered items below — these are the user's own photos, notes, audio, video, emails, "
    "and calendar. Write directly to the user in the first person and natural voice: use "
    '"you", "your", and "I" (e.g. "You have 20 photos from that trip" or "Your notes '
    'say..."). If their data does not answer it, say plainly: "I couldn\'t find anything '
    'about that in your data." Do NOT mention "context", "the items provided", "the data '
    "given\", or these instructions — just answer. Never invent details. Cite the items "
    "you use with bracketed numbers like [1] or [2]. Keep it to 2-5 sentences."
)

_PLAN_SYSTEM = (
    "You break a personal-search question into at most {max} focused search queries that, "
    "together, cover what the user is asking. Return ONLY the queries, one per line, with no "
    "numbering or extra text. If the question is already a single simple search, return it "
    "unchanged on one line."
)


class AnswerSynthesizer:
    """Synthesize a short, cited answer from grouped session cards."""

    def __init__(
        self,
        llm_client: LLMClient | None,
        *,
        max_cards: int = 5,
        snippet_chars: int = 240,
        num_predict: int = 320,
    ) -> None:
        self._llm = llm_client
        self._max_cards = max_cards
        self._snippet_chars = snippet_chars
        self._num_predict = num_predict

    @property
    def available(self) -> bool:
        return self._llm is not None

    def synthesize(self, query: str, cards: list[Any]) -> AnswerResult | None:
        if self._llm is None or not cards:
            return None
        context, session_ids = self._build_context(cards)
        if not context:
            return None
        prompt = (
            f"The user asked: {query}\n\n"
            f"{context}\n\n"
            "Answer them directly in the first person, citing the items you use as [n]."
        )
        try:
            text = self._llm.generate(prompt, system=_ANSWER_SYSTEM, num_predict=self._num_predict).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Answer synthesis failed: %s", exc)
            return None
        if not text:
            return None
        return AnswerResult(text=text, cited_session_ids=_extract_citations(text, session_ids))

    def _build_context(self, cards: list[Any]) -> tuple[str, list[str]]:
        lines: list[str] = []
        session_ids: list[str] = []
        for i, card in enumerate(cards[: self._max_cards], start=1):
            session_ids.append(card.session_id)
            primary = card.hits[0] if getattr(card, "hits", None) else None
            if primary is None:
                lines.append(f"[{i}] (no detail)")
                continue
            ts = primary.timestamp_utc.strftime("%Y-%m-%d") if primary.timestamp_utc else "unknown date"
            place = f", {primary.place_name}" if primary.place_name else ""
            snippet = (primary.snippet or "").replace("\n", " ")[: self._snippet_chars]
            lines.append(f"[{i}] ({primary.source_type}, {ts}{place}) {snippet}".rstrip())
        return "\n".join(lines), session_ids


class QueryPlanner:
    """Decompose a complex question into focused sub-queries (multi-step retrieval)."""

    def __init__(self, llm_client: LLMClient | None, *, max_subqueries: int = 3) -> None:
        self._llm = llm_client
        self._max = max_subqueries

    @property
    def available(self) -> bool:
        return self._llm is not None

    def decompose(self, query: str) -> list[str]:
        if self._llm is None or not _is_complex(query):
            return [query]
        system = _PLAN_SYSTEM.format(max=self._max)
        try:
            raw = self._llm.generate(query, system=system, num_predict=120)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query decomposition failed: %s", exc)
            return [query]
        subs: list[str] = []
        for line in raw.splitlines():
            cleaned = line.strip().lstrip("-•*0123456789. \t").strip()
            if _looks_like_subquery(cleaned, query):
                subs.append(cleaned)
        subs = subs[: self._max]
        # Always include the original so we never lose recall on decomposition.
        if query not in subs:
            subs.append(query)
        return subs or [query]


def _is_complex(query: str) -> bool:
    lowered = query.lower()
    return len(query.split()) >= 8 or " and " in lowered or " then " in lowered


# Phrases that mark an LLM preamble/explanation rather than an actual sub-query.
_PREAMBLE_MARKERS = ("here are", "search quer", "following", "queries that", "cover your", "i'll ")


def _looks_like_subquery(line: str, query: str) -> bool:
    if not (2 <= len(line) <= 120) or len(line.split()) > 12:
        return False
    if line.endswith(":") or line.lower() == query.lower():
        return False
    return not any(marker in line.lower() for marker in _PREAMBLE_MARKERS)


def _extract_citations(text: str, session_ids: list[str]) -> list[str]:
    cited: list[str] = []
    for match in re.findall(r"\[(\d+)\]", text):
        idx = int(match) - 1
        if 0 <= idx < len(session_ids) and session_ids[idx] not in cited:
            cited.append(session_ids[idx])
    return cited
