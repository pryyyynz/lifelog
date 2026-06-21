"""Extract structured signals from natural language queries before retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Visual intent keyword set (spec §12.2)
# ---------------------------------------------------------------------------

VISUAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "photo", "photos", "picture", "pictures", "pic", "pics",
        "sunset", "saw", "looked", "view", "scene", "face", "faces",
        "rainy", "market", "image", "images", "screenshot", "screenshots",
        "selfie", "selfies", "landscape", "portrait", "snapshot", "snapshots",
        "album", "albums", "gallery", "camera", "shot", "shots", "footage",
    }
)

# Audio intent keywords
AUDIO_KEYWORDS: frozenset[str] = frozenset(
    {
        "audio", "recording", "recordings", "voice", "podcast", "podcasts",
        "call", "calls", "transcript", "transcripts", "dictated", "whisper",
        "spoken", "narrated", "memo", "memos",
    }
)

# Document / text intent keywords
TEXT_KEYWORDS: frozenset[str] = frozenset(
    {
        "note", "notes", "journal", "journals", "diary", "wrote", "writing",
        "document", "documents", "file", "files", "text", "article", "articles",
        "blog", "entry", "entries", "log", "logs", "markdown", "obsidian",
        "draft", "drafts", "essay", "essays", "letter", "letters",
    }
)

# Email intent keywords
EMAIL_KEYWORDS: frozenset[str] = frozenset(
    {
        "email", "emails", "mail", "inbox", "sent", "newsletter", "newsletters",
        "reply", "replies", "forwarded", "unread", "attachment", "attachments",
        "cc", "bcc", "mbox",
    }
)

# Calendar / event intent keywords
CALENDAR_KEYWORDS: frozenset[str] = frozenset(
    {
        "calendar", "event", "events", "meeting", "meetings", "appointment",
        "appointments", "schedule", "reminder", "reminders", "birthday",
        "birthdays", "anniversary", "deadline", "deadlines", "booked", "planned",
        "invited", "invite",
    }
)

# Video intent keywords
VIDEO_KEYWORDS: frozenset[str] = frozenset(
    {
        "video", "videos", "clip", "clips", "footage", "film", "films",
        "movie", "movies", "recording", "recordings", "watched", "timelapse",
        "reel", "reels",
    }
)

# ---------------------------------------------------------------------------
# Temporal pattern helpers
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

_SEASONS: dict[str, tuple[int, int]] = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2),
}

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySignals:
    """Structured signals extracted from a raw user query."""

    temporal_range: tuple[datetime, datetime] | None = None
    """Resolved UTC timestamp range, or None if no temporal hint detected."""

    place_names: list[str] = field(default_factory=list)
    """Geographic place names extracted by NER (GPE/LOC entities)."""

    person_names: list[str] = field(default_factory=list)
    """Person names extracted by NER (PERSON entities)."""

    visual_intent: bool = False
    """True when the query likely refers to images or visual memories."""

    visual_keyword_count: int = 0
    """Number of visual intent keywords found in the query."""

    audio_intent: bool = False
    """True when the query likely refers to audio recordings or transcripts."""

    text_intent: bool = False
    """True when the query likely refers to notes, journals, or documents."""

    email_intent: bool = False
    """True when the query likely refers to emails or messages."""

    calendar_intent: bool = False
    """True when the query likely refers to calendar events or meetings."""

    video_intent: bool = False
    """True when the query likely refers to video recordings."""

    modality_intents: frozenset[str] = field(default_factory=frozenset)
    """Set of source_type strings that are explicitly signalled by the query.
    Empty means no explicit modality — retrieve across all modalities."""

    raw_query: str = ""


# ---------------------------------------------------------------------------
# QueryAnalyzer
# ---------------------------------------------------------------------------


class QueryAnalyzer:
    """Extracts temporal, spatial, person, and visual signals from user queries.

    Uses regex-based temporal parsing plus optional spaCy NER for entities.
    spaCy is a soft dependency — if unavailable, entity extraction is skipped.
    """

    def __init__(self, use_spacy: bool = True) -> None:
        self._nlp: Any = None
        if use_spacy:
            try:
                import spacy  # noqa: PLC0415

                self._nlp = spacy.load("en_core_web_sm")
            except Exception:  # noqa: BLE001
                pass  # spaCy unavailable — regex-only mode

    def analyze(self, query: str) -> QuerySignals:
        """Parse *query* and return extracted signals."""
        query_lower = query.lower()
        words = set(re.split(r"\W+", query_lower))

        # Per-modality intent detection
        visual_kws = [w for w in VISUAL_KEYWORDS if w in words]
        audio_kws = [w for w in AUDIO_KEYWORDS if w in words]
        text_kws = [w for w in TEXT_KEYWORDS if w in words]
        email_kws = [w for w in EMAIL_KEYWORDS if w in words]
        calendar_kws = [w for w in CALENDAR_KEYWORDS if w in words]
        video_kws = [w for w in VIDEO_KEYWORDS if w in words]

        # Build the set of explicitly signalled modalities
        modality_intents: set[str] = set()
        if visual_kws:
            modality_intents.add("photo")
        if audio_kws:
            modality_intents.add("audio")
        if text_kws:
            modality_intents.add("text")
        if email_kws:
            modality_intents.add("email")
        if calendar_kws:
            modality_intents.add("calendar")
        if video_kws:
            modality_intents.update({"video", "photo"})  # video frames are also visual

        # Temporal hint
        temporal = _extract_temporal(query_lower)

        # NER-based entity extraction (optional)
        place_names: list[str] = []
        person_names: list[str] = []
        if self._nlp is not None:
            doc = self._nlp(query)
            place_names = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
            person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        return QuerySignals(
            temporal_range=temporal,
            place_names=place_names,
            person_names=person_names,
            visual_intent=bool(visual_kws),
            visual_keyword_count=len(visual_kws),
            audio_intent=bool(audio_kws),
            text_intent=bool(text_kws),
            email_intent=bool(email_kws),
            calendar_intent=bool(calendar_kws),
            video_intent=bool(video_kws),
            modality_intents=frozenset(modality_intents),
            raw_query=query,
        )


# ---------------------------------------------------------------------------
# Temporal boost (spec §12.3)
# ---------------------------------------------------------------------------


class TemporalBoost:
    """Computes an exponential decay boost score given a time distance.

    ``score = alpha * exp(-|delta_seconds| / (tau_days * 86400))``

    A perfect temporal match (delta = 0) yields ``alpha``; a result one ``tau``
    away yields ``alpha / e ≈ alpha * 0.37``.
    """

    def __init__(self, tau_days: float = 7.0, alpha: float = 0.5) -> None:
        self._tau_secs = tau_days * 86400.0
        self._alpha = alpha

    def score(self, ts: datetime, target: datetime) -> float:
        """Return a boost in [0, alpha] based on temporal distance."""
        delta = abs((ts - target).total_seconds())
        return self._alpha * math.exp(-delta / self._tau_secs)


# ---------------------------------------------------------------------------
# Private temporal extraction helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _extract_temporal(query: str) -> tuple[datetime, datetime] | None:
    """Return (start, end) UTC range or None if no temporal hint is found."""
    now = _now_utc()

    # ---------- relative shorthands ----------

    if re.search(r"\byesterday\b", query):
        d = (now - timedelta(days=1)).date()
        return _day_range(d)

    if re.search(r"\btoday\b", query):
        return _day_range(now.date())

    if re.search(r"\bthis\s+week\b", query):
        start = now - timedelta(days=now.weekday())
        return _day_range(start.date()), _day_range(now.date())[1]

    if re.search(r"\blast\s+week\b", query):
        start = now - timedelta(days=now.weekday() + 7)
        end = start + timedelta(days=6)
        return _day_range(start.date())[0], _day_range(end.date())[1]

    if re.search(r"\blast\s+month\b", query):
        first_this = now.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return _day_range(last_month_start.date())[0], _day_range(last_month_end.date())[1]

    if re.search(r"\blast\s+year\b", query):
        y = now.year - 1
        return _year_range(y)

    if re.search(r"\bthis\s+year\b", query):
        return _year_range(now.year)

    # ---------- season + optional year ----------
    season_match = re.search(
        r"\b(last\s+)?(spring|summer|autumn|fall|winter)(?:\s+(\d{4}))?\b", query
    )
    if season_match:
        is_last = bool(season_match.group(1))
        season = season_match.group(2)
        year_str = season_match.group(3)
        start_m, end_m = _SEASONS[season]
        year = int(year_str) if year_str else (now.year - 1 if is_last else now.year)
        # Handle winter wrap-around
        if start_m > end_m:  # e.g. Dec–Feb
            s = datetime(year, start_m, 1, tzinfo=UTC)
            e = datetime(year + 1, end_m, 28, 23, 59, 59, tzinfo=UTC)
        else:
            s = datetime(year, start_m, 1, tzinfo=UTC)
            e = datetime(year, end_m, 28, 23, 59, 59, tzinfo=UTC)
        return s, e

    # ---------- "Month YYYY" or "Month, YYYY" ----------
    month_year = re.search(
        r"\b("
        + "|".join(_MONTHS)
        + r")[\s,]+(\d{4})\b",
        query,
    )
    if month_year:
        m = _MONTHS[month_year.group(1)]
        y = int(month_year.group(2))
        return _month_range(y, m)

    # ---------- ISO date (must come before year-only check) ----------
    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", query)
    if iso_match:
        y, m, d_int = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        try:
            from datetime import date as _date  # noqa: PLC0415

            return _day_range(_date(y, m, d_int))
        except ValueError:
            pass

    # ---------- "in YYYY" or standalone year ----------
    year_match = re.search(r"\bin\s+(\d{4})\b|\b(20\d{2}|19\d{2})\b", query)
    if year_match:
        y = int(year_match.group(1) or year_match.group(2))
        return _year_range(y)

    return None


def _day_range(d: Any) -> tuple[datetime, datetime]:
    """Full-day UTC range for a ``date`` object."""
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=UTC)
    end = datetime(d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=UTC)
    return start, end


def _month_range(year: int, month: int) -> tuple[datetime, datetime]:
    import calendar  # noqa: PLC0415

    _, last_day = calendar.monthrange(year, month)
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year, month, last_day, 23, 59, 59, 999999, tzinfo=UTC)
    return start, end


def _year_range(year: int) -> tuple[datetime, datetime]:
    return (
        datetime(year, 1, 1, tzinfo=UTC),
        datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
    )
