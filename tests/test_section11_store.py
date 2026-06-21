"""Tests for Section 11: Metadata Store, Vector Index, and Data Model."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.ingest.embedders import SentenceTransformerEmbedder, embed_text_records
from app.models.contracts import NormalizedChunkRecord
from app.storage.consistency import ConsistencyChecker, ConsistencyReport
from app.storage.metadata import MetadataStore
from app.storage.vector_store import VectorStore, chunk_id_to_point_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    chunk_id: str = "abc123",
    source_type: str = "text",
    vector_collection: str | None = "text_chunks",
    session_id: str | None = None,
    timestamp_start_sec: float | None = None,
    timestamp_end_sec: float | None = None,
    metadata: dict | None = None,
) -> NormalizedChunkRecord:
    return NormalizedChunkRecord(
        chunk_id=chunk_id,
        source_type=source_type,
        file_path=Path("/fake/file.txt"),
        text="hello world",
        timestamp_utc=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        vector_collection=vector_collection,
        session_id=session_id,
        timestamp_start_sec=timestamp_start_sec,
        timestamp_end_sec=timestamp_end_sec,
        metadata=metadata or {"chunk_identity": f"{chunk_id}:0"},
    )


# ---------------------------------------------------------------------------
# MetadataStore schema tests
# ---------------------------------------------------------------------------


class TestMetadataStoreSchema:
    def test_new_columns_exist(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        import sqlite3

        conn = sqlite3.connect(store.sqlite_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        conn.close()
        assert "session_id" in cols
        assert "vector_id" in cols
        assert "timestamp_start_sec" in cols
        assert "timestamp_end_sec" in cols

    def test_upsert_writes_new_columns(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        record = _make_record(
            chunk_id="aaa111",
            session_id="sess_01",
            timestamp_start_sec=1.5,
            timestamp_end_sec=45.0,
            metadata={"chunk_identity": "aaa111:0"},
        )
        store.upsert_chunks("src1", [record])
        rows = store.fetch_chunks()
        assert len(rows) == 1
        row = rows[0]
        assert row["session_id"] == "sess_01"
        assert row["timestamp_start_sec"] == pytest.approx(1.5)
        assert row["timestamp_end_sec"] == pytest.approx(45.0)
        assert row["vector_id"] is None  # not yet vectorised

    def test_embedding_vectors_stripped_from_json(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        record = _make_record(
            chunk_id="bbb222",
            metadata={
                "chunk_identity": "bbb222:0",
                "text_embedding": [0.1, 0.2, 0.3],  # should be stripped
                "other_key": "keep_me",
            },
        )
        store.upsert_chunks("src1", [record])
        row = store.fetch_chunks()[0]
        meta = json.loads(row["metadata_json"])
        assert "text_embedding" not in meta
        assert meta["other_key"] == "keep_me"

    def test_update_vector_ids(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        record = _make_record(chunk_id="ccc333", metadata={"chunk_identity": "ccc333:0"})
        store.upsert_chunks("src1", [record])
        store.update_vector_ids({"ccc333": "uuid-test-1234"})
        row = store.fetch_chunks()[0]
        assert row["vector_id"] == "uuid-test-1234"

    def test_all_chunk_ids_with_vector_id(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        r1 = _make_record(chunk_id="d1", metadata={"chunk_identity": "d1:0"})
        r2 = _make_record(chunk_id="d2", metadata={"chunk_identity": "d2:0"})
        store.upsert_chunks("src1", [r1, r2])
        store.update_vector_ids({"d1": "uuid-d1"})
        mapping = store.all_chunk_ids_with_vector_id()
        assert mapping == {"d1": "uuid-d1"}

    def test_fetch_chunks_by_ids(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        r1 = _make_record(chunk_id="e1", metadata={"chunk_identity": "e1:0"})
        r2 = _make_record(chunk_id="e2", metadata={"chunk_identity": "e2:0"})
        store.upsert_chunks("src1", [r1, r2])
        rows = store.fetch_chunks_by_ids({"e1"})
        assert len(rows) == 1
        assert rows[0]["chunk_id"] == "e1"

    def test_chunk_counts_by_source_type(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        r1 = _make_record(chunk_id="f1", source_type="text", metadata={"chunk_identity": "f1:0"})
        r2 = _make_record(chunk_id="f2", source_type="photo", metadata={"chunk_identity": "f2:0"})
        r3 = _make_record(chunk_id="f3", source_type="text", metadata={"chunk_identity": "f3:0"})
        store.upsert_chunks("src1", [r1, r2, r3])
        counts = store.chunk_counts_by_source_type()
        assert counts["text"] == 2
        assert counts["photo"] == 1


# ---------------------------------------------------------------------------
# VectorStore UUID helper
# ---------------------------------------------------------------------------


class TestChunkIdToPointId:
    def test_returns_valid_uuid_string(self) -> None:
        import uuid

        result = chunk_id_to_point_id("abc123def456789012345678")
        uuid.UUID(result)  # raises if invalid

    def test_deterministic(self) -> None:
        a = chunk_id_to_point_id("mychunk")
        b = chunk_id_to_point_id("mychunk")
        assert a == b

    def test_different_inputs_different_outputs(self) -> None:
        a = chunk_id_to_point_id("chunk_a")
        b = chunk_id_to_point_id("chunk_b")
        assert a != b


# ---------------------------------------------------------------------------
# VectorStore graceful degradation (Qdrant not running)
# ---------------------------------------------------------------------------


class TestVectorStoreOffline:
    def test_available_false_when_qdrant_down(self) -> None:
        vs = VectorStore(url="http://127.0.0.1:19999")  # nothing running here
        assert vs.available is False

    def test_ensure_collections_empty_when_offline(self) -> None:
        vs = VectorStore(url="http://127.0.0.1:19999")
        result = vs.ensure_collections()
        assert result == {}

    def test_upsert_records_returns_empty_when_offline(self) -> None:
        vs = VectorStore(url="http://127.0.0.1:19999")
        record = _make_record(chunk_id="xyz", metadata={"chunk_identity": "xyz:0", "text_embedding": [0.1] * 1024})
        result = vs.upsert_records([record])
        assert result == {}

    def test_search_returns_empty_when_offline(self) -> None:
        vs = VectorStore(url="http://127.0.0.1:19999")
        result = vs.search("text_chunks", [0.1] * 1024)
        assert result == []

    def test_fetch_all_point_ids_returns_empty_when_offline(self) -> None:
        vs = VectorStore(url="http://127.0.0.1:19999")
        result = vs.fetch_all_point_ids("text_chunks")
        assert result == set()


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder (disabled mode)
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbedderDisabled:
    def test_embed_batch_returns_none_when_disabled(self) -> None:
        embedder = SentenceTransformerEmbedder(enabled=False, model_name="intfloat/e5-large-v2")
        result = embedder.embed_batch(["hello"])
        assert result is None

    def test_status_is_disabled(self) -> None:
        embedder = SentenceTransformerEmbedder(enabled=False, model_name="intfloat/e5-large-v2")
        assert embedder.status == "disabled"


class TestEmbedTextRecords:
    def test_passthrough_when_no_embedding_text(self) -> None:
        records = [_make_record(chunk_id="t1", metadata={"chunk_identity": "t1:0"})]
        result = embed_text_records(records)
        assert result == records

    def test_status_written_when_disabled(self) -> None:
        embedder = SentenceTransformerEmbedder(enabled=False, model_name="intfloat/e5-large-v2")
        record = _make_record(
            chunk_id="t2",
            metadata={"chunk_identity": "t2:0", "embedding_text": "passage: hello world"},
        )
        result = embed_text_records([record], embedder=embedder)
        assert result[0].metadata["text_embedding_status"] == "disabled"
        assert "text_embedding" not in result[0].metadata

    def test_embedding_written_when_model_available(self) -> None:
        """Test that embed_text_records writes vectors when the model is functional."""

        class _StubEmbedder(SentenceTransformerEmbedder):
            def __init__(self) -> None:
                self._enabled = True
                self._model = True  # truthy sentinel
                self.status = "ok"
                self._model_name = "stub"

            def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
                return [[float(i) for i in range(4)] for _ in texts]

        record = _make_record(
            chunk_id="t3",
            metadata={"chunk_identity": "t3:0", "embedding_text": "passage: hello"},
        )
        result = embed_text_records([record], embedder=_StubEmbedder())
        assert result[0].metadata["text_embedding"] == [0.0, 1.0, 2.0, 3.0]
        assert result[0].metadata["text_embedding_status"] == "ok"


# ---------------------------------------------------------------------------
# ConsistencyChecker (offline Qdrant — report should be ok=True)
# ---------------------------------------------------------------------------


class TestConsistencyCheckerOffline:
    def test_check_ok_when_qdrant_offline(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "test.db")
        vs = VectorStore(url="http://127.0.0.1:19999")
        checker = ConsistencyChecker(store, vs)
        report = checker.check()
        assert report.ok is True
        assert report.total_orphans == 0
