"""Auto session-card titles via one batched LLM call.

Best-effort and order-based: one short title per card. Returns an empty mapping on
any parse/availability failure so callers simply fall back to untitled cards.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.retrieval.answers import LLMClient

logger = logging.getLogger(__name__)

_TITLE_SYSTEM = (
    "You give each numbered item a short, specific title of 3-6 words. Return one line per "
    "item in the exact format '<n>: <title>', in order, nothing else."
)
_LINE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(.+?)\s*$")


class CardTitler:
    def __init__(self, llm_client: LLMClient | None, *, max_cards: int = 8) -> None:
        self._llm = llm_client
        self._max_cards = max_cards

    @property
    def available(self) -> bool:
        return self._llm is not None

    def title_cards(self, cards: list[Any]) -> dict[str, str]:
        if self._llm is None or not cards:
            return {}
        subset = cards[: self._max_cards]
        lines = []
        for i, card in enumerate(subset, start=1):
            primary = card.hits[0] if getattr(card, "hits", None) else None
            snippet = (getattr(primary, "snippet", "") or "").replace("\n", " ")[:160]
            stype = getattr(primary, "source_type", "?")
            lines.append(f"{i}. ({stype}) {snippet}".rstrip())
        prompt = "Items:\n" + "\n".join(lines) + "\n\nTitles:"
        try:
            raw = self._llm.generate(prompt, system=_TITLE_SYSTEM, num_predict=200)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Card titling failed: %s", exc)
            return {}

        titles: dict[str, str] = {}
        for line in raw.splitlines():
            match = _LINE.match(line)
            if not match:
                continue
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(subset):
                titles[subset[idx].session_id] = match.group(2).strip().strip('"')
        return titles
