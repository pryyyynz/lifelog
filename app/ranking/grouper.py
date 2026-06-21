"""Session grouping: cluster ranked hits into SessionCard objects."""

from __future__ import annotations

import os
from datetime import datetime

from app.models.contracts import RetrievalHit, SessionCard


class SessionGrouper:
    """Groups a ranked list of :class:`RetrievalHit` objects into :class:`SessionCard` objects.

    Hits that share the same ``session_id`` are placed in the same card.
    Hits with no ``session_id`` each get their own card (using ``chunk_id``
    as a synthetic session identifier).

    The card's ``score`` is the score of its highest-ranking hit.
    Cards are returned sorted by descending score and capped at ``top_n``.

    Parameters
    ----------
    top_n:
        Maximum number of session cards to return (default 5).
    merge_radius_secs:
        Maximum gap in seconds between sessions to merge them (default 1800s/30min).
    """

    def __init__(self, top_n: int = 5, merge_radius_secs: float = 1800.0) -> None:
        self._top_n = top_n
        self._merge_radius_secs = merge_radius_secs

    @classmethod
    def from_environment(cls) -> SessionGrouper:
        top_n = int(os.getenv("LIFELOG_SESSION_TOP_N", "5"))
        merge_radius = float(os.getenv("LIFELOG_SESSION_MERGE_RADIUS_SECS", "1800.0"))
        return cls(top_n=top_n, merge_radius_secs=merge_radius)

    def _sort_hits_diverse(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
            
        # Group hits by modality (source_type)
        by_modality: dict[str, list[RetrievalHit]] = {}
        for hit in hits:
            by_modality.setdefault(hit.source_type, []).append(hit)
            
        # Sort each modality's hits by score descending
        for mod in by_modality:
            by_modality[mod].sort(key=lambda h: h.score, reverse=True)
            
        # Get the lead hit for each modality
        lead_hits: list[RetrievalHit] = []
        for mod in by_modality:
            if by_modality[mod]:
                lead_hits.append(by_modality[mod][0])
                
        # Sort lead hits by score descending
        lead_hits.sort(key=lambda h: h.score, reverse=True)
        
        # Collect the remaining hits and sort them by score descending
        lead_ids = {h.chunk_id for h in lead_hits}
        remaining_hits = [h for h in hits if h.chunk_id not in lead_ids]
        remaining_hits.sort(key=lambda h: h.score, reverse=True)
        
        return lead_hits + remaining_hits

    def _should_merge(self, c1: SessionCard, c2: SessionCard) -> bool:
        if c1.start_utc is None or c1.end_utc is None or c2.start_utc is None or c2.end_utc is None:
            return False

        # Place name checks: if both have place name list, and they don't overlap, do not merge
        places1 = {h.place_name.strip().lower() for h in c1.hits if h.place_name}
        places2 = {h.place_name.strip().lower() for h in c2.hits if h.place_name}
        if places1 and places2 and not (places1 & places2):
            return False

        s1, e1 = c1.start_utc, c1.end_utc
        s2, e2 = c2.start_utc, c2.end_utc
        
        overlap = max(s1, s2) <= min(e1, e2)
        if overlap:
            return True
            
        if s2 > e1:
            gap = (s2 - e1).total_seconds()
        else:
            gap = (s1 - e2).total_seconds()
            
        return gap <= self._merge_radius_secs

    def _merge_cards(self, c1: SessionCard, c2: SessionCard) -> SessionCard:
        # Combine all hits
        all_hits = c1.hits + c2.hits
        
        # Deduplicate by file_path
        unique_hits: dict[str, RetrievalHit] = {}
        for hit in all_hits:
            fp_str = str(hit.file_path.resolve()) if hasattr(hit.file_path, 'resolve') else str(hit.file_path)
            existing = unique_hits.get(fp_str)
            if existing is None or hit.score > existing.score:
                unique_hits[fp_str] = hit
                
        merged_hits = list(unique_hits.values())
        
        # Sort merged hits with modality diversity
        sorted_hits = self._sort_hits_diverse(merged_hits)
        
        score = max(c1.score, c2.score)
        
        timestamps = [h.timestamp_utc for h in sorted_hits if h.timestamp_utc is not None]
        start_utc = min(timestamps) if timestamps else None
        end_utc = max(timestamps) if timestamps else None
        
        # Combine session IDs
        sids = []
        for c in (c1, c2):
            if c.session_id.startswith("merged_"):
                parts = c.session_id[len("merged_"):].split("__")
                sids.extend(parts)
            else:
                sids.append(c.session_id)
        unique_sids = sorted(list(set(sids)))
        merged_session_id = f"merged_{'__'.join(unique_sids)}"
        
        modalities = sorted(list(set(h.source_type for h in sorted_hits)))
        
        return SessionCard(
            session_id=merged_session_id,
            score=score,
            hits=sorted_hits,
            start_utc=start_utc,
            end_utc=end_utc,
            modalities=modalities,
        )

    def group(self, hits: list[RetrievalHit]) -> list[SessionCard]:
        """Convert a flat ranked list into grouped session cards with cross-modality merges."""
        sessions: dict[str, list[RetrievalHit]] = {}

        for hit in hits:
            key = hit.session_id if hit.session_id else f"_solo_{hit.chunk_id}"
            sessions.setdefault(key, []).append(hit)

        cards: list[SessionCard] = []
        for session_key, session_hits in sessions.items():
            # Deduplicate by file_path
            unique_hits: dict[str, RetrievalHit] = {}
            for hit in session_hits:
                fp_str = str(hit.file_path.resolve()) if hasattr(hit.file_path, 'resolve') else str(hit.file_path)
                existing = unique_hits.get(fp_str)
                if existing is None or hit.score > existing.score:
                    unique_hits[fp_str] = hit
            
            session_hits = list(unique_hits.values())
            
            # Diverse sorting
            sorted_hits = self._sort_hits_diverse(session_hits)

            timestamps = [h.timestamp_utc for h in sorted_hits if h.timestamp_utc is not None]
            start_utc = min(timestamps) if timestamps else None
            end_utc = max(timestamps) if timestamps else None

            # Use the real session_id (or None for solo hits)
            card_session_id = (
                sorted_hits[0].session_id
                if not session_key.startswith("_solo_")
                else None
            )

            modalities = sorted(list(set(h.source_type for h in sorted_hits)))

            cards.append(
                SessionCard(
                    session_id=card_session_id or session_key,
                    score=sorted_hits[0].score,
                    hits=sorted_hits,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    modalities=modalities,
                )
            )

        # Cross-modality enrichment pass
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(cards):
                j = i + 1
                while j < len(cards):
                    c1 = cards[i]
                    c2 = cards[j]
                    if self._should_merge(c1, c2):
                        merged_card = self._merge_cards(c1, c2)
                        cards[i] = merged_card
                        cards.pop(j)
                        changed = True
                        break
                    j += 1
                if changed:
                    break
                i += 1

        cards.sort(key=lambda c: c.score, reverse=True)
        return cards[: self._top_n]

    def group_chronological(self, hits: list[RetrievalHit]) -> list[SessionCard]:
        """Like :meth:`group` but sort session cards by start time (UC-06)."""
        cards = self.group(hits)
        # Re-sort by start_utc ascending; cards without timestamps go last
        return sorted(
            cards,
            key=lambda c: (c.start_utc is None, c.start_utc or datetime.min),
        )
