"""Daily / weekly digests: an LLM-written recap of a time window.

Degrades to a deterministic summary (counts by modality + date range) when no LLM
client is configured, so digests always return something useful.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from app.retrieval.answers import LLMClient
from app.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)

_DIGEST_SYSTEM = (
    "You write a brief, warm recap of the user's own recent activity from their personal "
    "life log. Use ONLY the provided items. 3-5 sentences, plain and specific. Do not invent "
    "anything not present in the items."
)
_PERIODS = {"day": 1, "week": 7}


class DigestGenerator:
    def __init__(self, store: MetadataStore, llm_client: LLMClient | None) -> None:
        self.store = store
        self._llm = llm_client

    def generate(
        self, period: str = "day", *, end: datetime | None = None, use_cache: bool = True
    ) -> dict[str, Any]:
        days = _PERIODS.get(period, 1)
        end = end or datetime.now()
        start = end - timedelta(days=days)
        period_key = f"{period}:{end.date().isoformat()}"

        if use_cache:
            cached = self.store.get_proactive("digest", period_key)
            if cached is not None:
                return cached

        rows = self.store.chunks_in_window(start.isoformat(), end.isoformat(), limit=400)
        if not rows:
            result = {
                "period": period,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "empty": True,
                "body": f"Nothing recorded in the last {period}.",
                "item_count": 0,
            }
            self.store.set_proactive("digest", period_key, result)
            return result

        counts = Counter(str(r["source_type"]) for r in rows)
        body = self._summarize(rows) if self._llm is not None else _fallback_summary(counts, days)
        result = {
            "period": period,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "empty": False,
            "body": body,
            "item_count": len(rows),
            "by_modality": dict(counts),
        }
        self.store.set_proactive("digest", period_key, result)
        return result

    def _summarize(self, rows: list) -> str:
        lines: list[str] = []
        for row in rows[:60]:
            ts = str(row["timestamp_utc"])[:10] if row["timestamp_utc"] else "?"
            text = (row["text"] or "").replace("\n", " ")[:160]
            place = f" @ {row['place_name']}" if row["place_name"] else ""
            lines.append(f"- ({row['source_type']}, {ts}{place}) {text}".rstrip())
        prompt = "Recent items:\n" + "\n".join(lines) + "\n\nWrite the recap."
        try:
            text = self._llm.generate(prompt, system=_DIGEST_SYSTEM, num_predict=350).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Digest summarization failed: %s", exc)
            text = ""
        return text or _fallback_summary(Counter(str(r["source_type"]) for r in rows), len(rows))


def _fallback_summary(counts: "Counter[str]", days: int) -> str:
    parts = ", ".join(f"{n} {kind}" for kind, n in counts.most_common())
    span = "day" if days == 1 else f"{days} days"
    return f"In the last {span}: {parts}." if parts else f"Nothing recorded in the last {span}."
