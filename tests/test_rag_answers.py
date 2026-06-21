"""Tests for Phase 3 RAG: answer synthesis and query decomposition (fake LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.retrieval.answers import AnswerSynthesizer, QueryPlanner

_T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


class _FakeLLM:
    """Records calls and returns a scripted response."""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None, num_predict: int = 150) -> str:
        self.calls.append({"prompt": prompt, "system": system, "num_predict": num_predict})
        return self._reply


@dataclass
class _Hit:
    source_type: str
    timestamp_utc: datetime | None
    place_name: str | None
    snippet: str | None


@dataclass
class _Card:
    session_id: str
    hits: list


def _cards() -> list[_Card]:
    return [
        _Card("s1", [_Hit("photo", _T0, "Lisbon", "sunset over the bridge")]),
        _Card("s2", [_Hit("text", _T0, None, "journal entry about the trip")]),
    ]


# ---------------------------------------------------------------------------
# Answer synthesis
# ---------------------------------------------------------------------------


class TestAnswerSynthesizer:
    def test_returns_none_without_llm(self) -> None:
        assert AnswerSynthesizer(None).available is False
        assert AnswerSynthesizer(None).synthesize("q", _cards()) is None

    def test_returns_none_for_no_cards(self) -> None:
        assert AnswerSynthesizer(_FakeLLM("x")).synthesize("q", []) is None

    def test_synthesizes_and_extracts_citations(self) -> None:
        llm = _FakeLLM("You were in Lisbon at sunset [1], and wrote about it [2].")
        result = AnswerSynthesizer(llm).synthesize("where was I at sunset?", _cards())
        assert result is not None
        assert "Lisbon" in result.text
        assert result.cited_session_ids == ["s1", "s2"]
        # Context should include both items and the place.
        assert "Lisbon" in llm.calls[0]["prompt"]
        assert "[2]" in llm.calls[0]["prompt"]

    def test_citation_out_of_range_ignored(self) -> None:
        llm = _FakeLLM("See [1] and [9].")
        result = AnswerSynthesizer(llm).synthesize("q", _cards())
        assert result is not None
        assert result.cited_session_ids == ["s1"]

    def test_empty_reply_returns_none(self) -> None:
        assert AnswerSynthesizer(_FakeLLM("   ")).synthesize("q", _cards()) is None

    def test_respects_max_cards(self) -> None:
        llm = _FakeLLM("answer")
        many = [_Card(f"s{i}", [_Hit("text", _T0, None, f"item {i}")]) for i in range(10)]
        AnswerSynthesizer(llm, max_cards=3).synthesize("q", many)
        prompt = llm.calls[0]["prompt"]
        assert "[3]" in prompt
        assert "[4]" not in prompt


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------


class TestQueryPlanner:
    def test_no_llm_returns_original(self) -> None:
        assert QueryPlanner(None).decompose("anything at all here") == ["anything at all here"]

    def test_simple_query_not_decomposed(self) -> None:
        llm = _FakeLLM("a\nb")
        # Short, no conjunction -> treated as simple, LLM not called.
        assert QueryPlanner(llm).decompose("beach photos") == ["beach photos"]
        assert llm.calls == []

    def test_complex_query_decomposed(self) -> None:
        llm = _FakeLLM("photos from the beach\njournal about the beach trip")
        q = "show me beach photos and what I wrote about the trip"
        subs = QueryPlanner(llm).decompose(q)
        assert "photos from the beach" in subs
        assert "journal about the beach trip" in subs
        # Original is always appended to preserve recall.
        assert q in subs

    def test_strips_numbering(self) -> None:
        llm = _FakeLLM("1. first sub query\n2. second sub query")
        subs = QueryPlanner(llm).decompose("a long enough question with and inside it")
        assert "first sub query" in subs
        assert "second sub query" in subs

    def test_drops_preamble_lines(self) -> None:
        # Observed live: llama3 prepends an explanatory line + numbered queries.
        llm = _FakeLLM(
            "Here are three focused search queries that cover your request:\n"
            "beach photos\nLisbon trip review"
        )
        subs = QueryPlanner(llm).decompose("show me beach photos and what I wrote about Lisbon")
        assert "beach photos" in subs
        assert "Lisbon trip review" in subs
        assert not any("here are" in s.lower() for s in subs)

    def test_caps_subqueries(self) -> None:
        llm = _FakeLLM("one one\ntwo two\nthree three\nfour four\nfive five")
        subs = QueryPlanner(llm, max_subqueries=2).decompose("question with and that is quite long indeed")
        # 2 sub-queries + the appended original.
        assert len(subs) == 3
