"""Tests for Phase 4 proactive features: on-this-day, digests, insights, titles."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from app.proactive.digests import DigestGenerator
from app.proactive.insights import InsightGenerator
from app.proactive.on_this_day import OnThisDay
from app.proactive.titles import CardTitler
from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore


class _FakeLLM:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None, num_predict: int = 150) -> str:
        self.calls += 1
        return self._reply


def _store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(tmp_path / "t.db")


def _add(store, chunk_id, ts, *, source_type="text", place=None, text="an entry"):
    store.upsert_chunks(
        "s",
        [
            NormalizedChunkRecord(
                chunk_id=chunk_id, source_type=source_type, file_path=Path(f"/d/{chunk_id}.md"),
                text=text, timestamp_utc=ts, place_name=place, vector_collection="text_chunks",
                metadata={"chunk_identity": f"{source_type}:{chunk_id}"},
            )
        ],
    )


# ---------------------------------------------------------------------------
# On this day
# ---------------------------------------------------------------------------


class TestOnThisDay:
    def test_returns_past_year_memory(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, "c1", datetime(2022, 6, 16, 10, tzinfo=UTC))
        cards = OnThisDay(store).for_date(on=date(2024, 6, 16))
        assert len(cards) >= 1

    def test_excludes_current_year(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, "c1", datetime(2024, 6, 16, 10, tzinfo=UTC))
        assert OnThisDay(store).for_date(on=date(2024, 6, 16)) == []


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


class TestDigests:
    def test_llm_recap_and_cache(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, "d1", datetime(2026, 6, 16, 9, tzinfo=UTC))
        end = datetime(2026, 6, 16, 12)
        r1 = DigestGenerator(store, _FakeLLM("Recap A")).generate(period="day", end=end)
        assert r1["body"] == "Recap A"
        assert r1["item_count"] == 1 and r1["empty"] is False
        # Cached: a different LLM is not consulted.
        r2 = DigestGenerator(store, _FakeLLM("Recap B")).generate(period="day", end=end)
        assert r2["body"] == "Recap A"
        # refresh bypasses cache.
        r3 = DigestGenerator(store, _FakeLLM("Recap B")).generate(period="day", end=end, use_cache=False)
        assert r3["body"] == "Recap B"

    def test_fallback_without_llm(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, "d1", datetime(2026, 6, 16, 9, tzinfo=UTC), source_type="photo")
        result = DigestGenerator(store, None).generate(
            period="day", end=datetime(2026, 6, 16, 12), use_cache=False
        )
        assert "1 photo" in result["body"]

    def test_empty_window(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = DigestGenerator(store, None).generate(period="day", end=datetime(2026, 6, 16, 12))
        assert result["empty"] is True


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


class TestInsights:
    def test_stats_and_recurring_places(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, "p1", datetime(2022, 8, 1, tzinfo=UTC), source_type="photo", place="Lisbon")
        _add(store, "p2", datetime(2023, 8, 1, tzinfo=UTC), source_type="photo", place="Lisbon")
        _add(store, "n1", datetime(2023, 8, 2, tzinfo=UTC), source_type="text")
        res = InsightGenerator(store, None).generate(use_cache=False)
        stats = res["stats"]
        assert stats["total"] == 3
        assert stats["by_modality"]["photo"] == 2
        assert "Lisbon" in stats["top_places"]
        assert "Lisbon" in stats["recurring_places"]
        assert res["narrative"] is None

    def test_narrative_with_llm(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, "p1", datetime(2022, 8, 1, tzinfo=UTC), source_type="photo", place="Lisbon")
        res = InsightGenerator(store, _FakeLLM("You love Lisbon.")).generate(use_cache=False)
        assert res["narrative"] == "You love Lisbon."

    def test_ignores_derived_chunks(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, "p1", datetime(2022, 8, 1, tzinfo=UTC), source_type="photo")
        store.upsert_chunks(
            "s",
            [
                NormalizedChunkRecord(
                    chunk_id="ocr1", source_type="photo", file_path=Path("/d/p1.md"),
                    text="some ocr text", timestamp_utc=datetime(2022, 8, 1, tzinfo=UTC),
                    vector_collection="text_chunks",
                    metadata={"chunk_identity": "ocr:text", "derived_from": "p1"},
                )
            ],
        )
        assert InsightGenerator(store, None).generate(use_cache=False)["stats"]["total"] == 1


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    source_type: str
    snippet: str


@dataclass
class _Card:
    session_id: str
    hits: list


class TestCardTitler:
    def test_titles_mapped_by_order(self) -> None:
        cards = [_Card("s1", [_Hit("photo", "beach")]), _Card("s2", [_Hit("text", "notes")])]
        titles = CardTitler(_FakeLLM("1: Beach Day\n2: Work Notes")).title_cards(cards)
        assert titles == {"s1": "Beach Day", "s2": "Work Notes"}

    def test_no_llm_returns_empty(self) -> None:
        assert CardTitler(None).title_cards([_Card("s1", [_Hit("text", "x")])]) == {}

    def test_unparseable_reply_returns_empty(self) -> None:
        assert CardTitler(_FakeLLM("sorry, no idea")).title_cards([_Card("s1", [_Hit("text", "x")])]) == {}


# ---------------------------------------------------------------------------
# Proactive API (LLM disabled for determinism)
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LIFELOG_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LIFELOG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LIFELOG_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LIFELOG_LLM_ENABLED", "false")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    store = MetadataStore(tmp_path / "test.db")
    _add(store, "recent", datetime.now(UTC) - timedelta(hours=2), source_type="photo")

    import app.api.main as api_mod
    importlib.reload(api_mod)
    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as client:
        yield client


class TestProactiveApi:
    def test_on_this_day_ok(self, api_client) -> None:
        resp = api_client.get("/proactive/on-this-day")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_digest_day(self, api_client) -> None:
        resp = api_client.get("/proactive/digest?period=day")
        assert resp.status_code == 200
        assert "body" in resp.json()

    def test_digest_bad_period(self, api_client) -> None:
        assert api_client.get("/proactive/digest?period=year").status_code == 400

    def test_insights(self, api_client) -> None:
        resp = api_client.get("/proactive/insights")
        assert resp.status_code == 200
        assert "stats" in resp.json()
