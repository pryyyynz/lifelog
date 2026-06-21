"""'On this day' resurfacing: memories from the same calendar day in past years."""

from __future__ import annotations

from datetime import date, datetime

from app.proactive.cards import row_to_hit
from app.ranking.grouper import SessionGrouper
from app.storage.metadata import MetadataStore


class OnThisDay:
    def __init__(self, store: MetadataStore, grouper: SessionGrouper | None = None) -> None:
        self.store = store
        self.grouper = grouper or SessionGrouper.from_environment()

    def for_date(self, on: date | None = None, *, exclude_current_year: bool = True) -> list:
        """Return session cards for items recorded on this month/day in past years."""
        on = on or datetime.now().date()
        rows = self.store.chunks_on_month_day(f"{on.month:02d}", f"{on.day:02d}")
        hits = []
        for row in rows:
            ts = row["timestamp_utc"]
            if exclude_current_year and ts and str(ts).startswith(f"{on.year:04d}-"):
                continue
            hits.append(row_to_hit(row, rationale="on_this_day"))
        if not hits:
            return []
        return self.grouper.group_chronological(hits)
