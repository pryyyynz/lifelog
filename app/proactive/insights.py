"""Patterns & insights: deterministic stats over the log, optionally narrated by the LLM."""

from __future__ import annotations

import calendar
import logging
from collections import Counter, defaultdict
from typing import Any

from app.retrieval.answers import LLMClient
from app.storage.metadata import MetadataStore

logger = logging.getLogger(__name__)

_INSIGHT_SYSTEM = (
    "You point out a few friendly, factual observations about patterns in the user's own "
    "life-log statistics. Use ONLY the numbers given. 2-4 short sentences. No speculation."
)
_WEEKDAYS = list(calendar.day_name)


class InsightGenerator:
    def __init__(self, store: MetadataStore, llm_client: LLMClient | None) -> None:
        self.store = store
        self._llm = llm_client

    def generate(self, *, use_cache: bool = True) -> dict[str, Any]:
        if use_cache:
            cached = self.store.get_proactive("insights", "global")
            if cached is not None:
                return cached
        stats = self._compute_stats()
        narrative = self._narrate(stats) if self._llm is not None and stats["total"] else None
        result = {"stats": stats, "narrative": narrative}
        self.store.set_proactive("insights", "global", result)
        return result

    def _compute_stats(self) -> dict[str, Any]:
        rows = self.store.fetch_chunks()
        by_modality: Counter[str] = Counter()
        by_weekday: Counter[str] = Counter()
        by_month: Counter[str] = Counter()
        place_counts: Counter[str] = Counter()
        place_years: dict[str, set[str]] = defaultdict(set)

        total = 0
        for row in rows:
            if row["metadata_json"] and '"derived_from"' in row["metadata_json"]:
                continue  # skip derived chunks in stats
            total += 1
            by_modality[str(row["source_type"])] += 1
            ts = row["timestamp_utc"]
            if ts:
                iso = str(ts)
                year, month = iso[:4], iso[5:7]
                by_month[month] += 1
                try:
                    from datetime import datetime  # noqa: PLC0415

                    by_weekday[_WEEKDAYS[datetime.fromisoformat(iso).weekday()]] += 1
                except ValueError:
                    pass
                place = row["place_name"]
                if place:
                    place_counts[str(place)] += 1
                    place_years[str(place)].add(year)

        recurring = sorted(
            (p for p, years in place_years.items() if len(years) >= 2),
            key=lambda p: len(place_years[p]),
            reverse=True,
        )
        return {
            "total": total,
            "by_modality": dict(by_modality.most_common()),
            "top_places": dict(place_counts.most_common(5)),
            "busiest_weekday": by_weekday.most_common(1)[0][0] if by_weekday else None,
            "busiest_month": (
                calendar.month_name[int(by_month.most_common(1)[0][0])] if by_month else None
            ),
            "recurring_places": recurring[:5],
        }

    def _narrate(self, stats: dict[str, Any]) -> str | None:
        prompt = (
            "Statistics:\n"
            f"- total items: {stats['total']}\n"
            f"- by type: {stats['by_modality']}\n"
            f"- top places: {stats['top_places']}\n"
            f"- busiest weekday: {stats['busiest_weekday']}\n"
            f"- busiest month: {stats['busiest_month']}\n"
            f"- places seen across multiple years: {stats['recurring_places']}\n\n"
            "Write the observations."
        )
        try:
            text = self._llm.generate(prompt, system=_INSIGHT_SYSTEM, num_predict=200).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Insight narration failed: %s", exc)
            return None
        return text or None
