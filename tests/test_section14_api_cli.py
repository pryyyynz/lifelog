"""Tests for Section 14: API, CLI, and frontend wiring."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)


def _populate_store(store: MetadataStore) -> None:
    records = [
        NormalizedChunkRecord(
            chunk_id=f"c{i}",
            source_type="text",
            file_path=Path("/notes/journal.md"),
            text=f"went hiking through mountain trails on day {i}",
            timestamp_utc=_T0 + timedelta(days=i),
            metadata={"chunk_identity": f"c{i}:0"},
        )
        for i in range(3)
    ]
    store.upsert_chunks("source_text", records)


# ---------------------------------------------------------------------------
# API tests (FastAPI TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path):
    """Return a FastAPI TestClient with an in-memory store."""
    import importlib
    import os

    os.environ["LIFELOG_SQLITE_PATH"] = str(tmp_path / "test.db")
    os.environ["LIFELOG_DATA_DIR"] = str(tmp_path)
    os.environ["LIFELOG_LOG_DIR"] = str(tmp_path / "logs")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    # Re-import the app to pick up fresh env
    import app.api.main as api_mod
    importlib.reload(api_mod)

    # Prime the store before lifespan (both point to same SQLite file)
    store = MetadataStore(tmp_path / "test.db")
    _populate_store(store)

    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as client:
        yield client


class TestStatusEndpoint:
    def test_status_returns_ok(self, api_client) -> None:
        resp = api_client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_status_has_chunk_counts(self, api_client) -> None:
        resp = api_client.get("/status")
        data = resp.json()
        assert "total_chunks" in data
        assert "chunks_by_modality" in data

    def test_status_has_sqlite_path(self, api_client) -> None:
        resp = api_client.get("/status")
        data = resp.json()
        assert "sqlite_path" in data


class TestQueryEndpoint:
    def test_query_returns_200(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "hiking mountains"})
        assert resp.status_code == 200

    def test_query_response_has_sessions(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "hiking"})
        data = resp.json()
        assert "sessions" in data
        assert "conversation_id" in data
        assert "query_debug" in data

    def test_query_sessions_have_primary(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "hiking trails"})
        data = resp.json()
        for session in data["sessions"]:
            assert "primary" in session
            assert "secondary" in session
            assert "score" in session

    def test_query_conversation_id_preserved(self, api_client) -> None:
        r1 = api_client.post("/query", json={"query": "hiking"})
        cid = r1.json()["conversation_id"]
        r2 = api_client.post("/query", json={"query": "more hiking", "conversation_id": cid})
        assert r2.json()["conversation_id"] == cid

    def test_query_empty_returns_422(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_query_no_results_for_nonsense(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "zzzznonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []

    def test_query_debug_fields(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "hiking yesterday"})
        debug = resp.json()["query_debug"]
        assert "visual_intent" in debug
        assert "total_hits_before_grouping" in debug

    def test_conversational_query_skips_retrieval(self, api_client) -> None:
        resp = api_client.post("/query", json={"query": "what do you do?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_message"]
        assert data["sessions"] == []
        assert data["query_debug"]["intent"] == "chit_chat"


class TestIngestTriggerEndpoint:
    def test_trigger_returns_started(self, api_client) -> None:
        resp = api_client.post("/ingest/trigger", json={"full": False})
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


class TestEditorLinkEndpoint:
    def test_vscode_link(self, api_client) -> None:
        resp = api_client.get("/editor-link", params={"file_path": "/notes/file.md", "editor": "vscode"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scheme"] == "vscode"
        assert "vscode://file/" in data["uri"]

    def test_obsidian_link(self, api_client) -> None:
        resp = api_client.get("/editor-link", params={"file_path": "/vault/note.md", "editor": "obsidian"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scheme"] == "obsidian"
        assert "obsidian://" in data["uri"]

    def test_default_file_link(self, api_client) -> None:
        resp = api_client.get("/editor-link", params={"file_path": "/some/file.txt", "editor": "default"})
        assert resp.status_code == 200
        assert resp.json()["scheme"] == "file"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_query_subcommand_parsed(self) -> None:
        from app.cli.main import build_parser  # noqa: PLC0415
        parser = build_parser()
        args = parser.parse_args(["query", "what did I do last summer"])
        assert args.command == "query"
        assert args.query == "what did I do last summer"

    def test_query_limit_default(self) -> None:
        from app.cli.main import build_parser  # noqa: PLC0415
        parser = build_parser()
        args = parser.parse_args(["query", "test"])
        assert args.limit == 5

    def test_delete_file_parsed(self) -> None:
        from app.cli.main import build_parser  # noqa: PLC0415
        parser = build_parser()
        args = parser.parse_args(["delete", "--file", "/path/to/file.md"])
        assert args.file == "/path/to/file.md"
        assert args.source is None

    def test_delete_source_parsed(self) -> None:
        from app.cli.main import build_parser  # noqa: PLC0415
        parser = build_parser()
        args = parser.parse_args(["delete", "--source", "my_texts"])
        assert args.source == "my_texts"

    def test_consistency_check_parsed(self) -> None:
        from app.cli.main import build_parser  # noqa: PLC0415
        parser = build_parser()
        args = parser.parse_args(["consistency-check"])
        assert args.command == "consistency-check"

    def test_logs_parsed_with_filter(self) -> None:
        from app.cli.main import build_parser  # noqa: PLC0415
        parser = build_parser()
        args = parser.parse_args(["logs", "--source", "texts", "--lines", "100"])
        assert args.source == "texts"
        assert args.lines == 100


class TestDeleteCommand:
    def test_delete_file_removes_chunks(self, tmp_path: Path) -> None:
        import os  # noqa: PLC0415
        os.environ["LIFELOG_SQLITE_PATH"] = str(tmp_path / "test.db")
        file_path = tmp_path / "journal.md"
        file_path.touch()
        store = MetadataStore(tmp_path / "test.db")

        from app.models.contracts import NormalizedChunkRecord  # noqa: PLC0415
        records = [
            NormalizedChunkRecord(
                chunk_id=f"d{i}",
                source_type="text",
                file_path=file_path,
                text=f"entry {i}",
                timestamp_utc=_T0,
                metadata={"chunk_identity": f"d{i}:0"},
            )
            for i in range(3)
        ]
        store.upsert_chunks("source_text", records)
        assert store.chunk_count() == 3

        from app.cli.main import build_parser, delete_cmd  # noqa: PLC0415
        args = build_parser().parse_args(["delete", "--file", str(file_path)])
        rc = delete_cmd(args)
        assert rc == 0
        assert store.chunk_count() == 0

    def test_delete_source_removes_chunks(self, tmp_path: Path) -> None:
        import os  # noqa: PLC0415
        os.environ["LIFELOG_SQLITE_PATH"] = str(tmp_path / "test.db")
        store = MetadataStore(tmp_path / "test.db")
        _populate_store(store)

        from app.cli.main import build_parser, delete_cmd  # noqa: PLC0415
        args = build_parser().parse_args(["delete", "--source", "source_text"])
        rc = delete_cmd(args)
        assert rc == 0
        assert store.chunk_count() == 0


class TestLogsCommand:
    def test_logs_no_log_dir(self, tmp_path: Path) -> None:
        import os  # noqa: PLC0415
        os.environ["LIFELOG_LOG_DIR"] = str(tmp_path / "nonexistent_logs")

        from app.cli.main import build_parser, logs_cmd  # noqa: PLC0415
        args = build_parser().parse_args(["logs"])
        rc = logs_cmd(args)
        assert rc == 1

    def test_logs_reads_log_file(self, tmp_path: Path, capsys) -> None:
        import os  # noqa: PLC0415
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "ingest.log"
        log_file.write_text("2024-01-01 source_text processed 3 items\n2024-01-01 source_text done\n")
        os.environ["LIFELOG_LOG_DIR"] = str(log_dir)

        from app.cli.main import build_parser, logs_cmd  # noqa: PLC0415
        args = build_parser().parse_args(["logs"])
        rc = logs_cmd(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "source_text" in captured.out


class TestFilePreviewEndpoint:
    def test_file_preview_image(self, api_client, tmp_path) -> None:
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake image data")
        resp = api_client.get(f"/file-preview?path={str(img_file)}")
        assert resp.status_code == 200
        assert resp.content == b"fake image data"

    def test_file_preview_video(self, api_client, tmp_path) -> None:
        vid_file = tmp_path / "test.mp4"
        vid_file.write_bytes(b"fake video data")
        resp = api_client.get(f"/file-preview?path={str(vid_file)}")
        assert resp.status_code == 200
        assert resp.content == b"fake video data"

    def test_file_preview_audio(self, api_client, tmp_path) -> None:
        aud_file = tmp_path / "test.mp3"
        aud_file.write_bytes(b"fake audio data")
        resp = api_client.get(f"/file-preview?path={str(aud_file)}")
        assert resp.status_code == 200
        assert resp.content == b"fake audio data"

    def test_file_preview_invalid_extension(self, api_client, tmp_path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_bytes(b"fake text data")
        resp = api_client.get(f"/file-preview?path={str(txt_file)}")
        assert resp.status_code == 400

    def test_file_preview_thumbnail(self, api_client, tmp_path) -> None:
        import io

        from PIL import Image

        img_file = tmp_path / "big.jpg"
        Image.new("RGB", (1000, 800), (10, 120, 200)).save(img_file, format="JPEG")

        resp = api_client.get(f"/file-preview?path={str(img_file)}&thumb=1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

        thumb = Image.open(io.BytesIO(resp.content))
        assert max(thumb.size) <= 512  # downscaled
        assert len(resp.content) < img_file.stat().st_size  # smaller than the original

    def test_file_preview_thumbnail_falls_back_on_corrupt_image(self, api_client, tmp_path) -> None:
        # A file with an image extension but non-decodable bytes should still
        # serve the original rather than error.
        img_file = tmp_path / "broken.jpg"
        img_file.write_bytes(b"not really an image")
        resp = api_client.get(f"/file-preview?path={str(img_file)}&thumb=1")
        assert resp.status_code == 200
        assert resp.content == b"not really an image"


class TestImageQueryAnswer:
    """Photo + text queries synthesize a grounded answer like the text path."""

    @staticmethod
    def _wire(api_mod, monkeypatch):
        """Stub CLIP retrieval and the LLM synthesizer so the endpoint is testable.

        CLIP and Ollama aren't available in the test env, so we feed the endpoint
        one real photo hit and a fake synthesizer; the real grouper/serialization
        still run.
        """
        from app.models.contracts import RetrievalHit
        from app.retrieval.answers import AnswerResult

        hit = RetrievalHit(
            chunk_id="c1",
            source_type="photo",
            file_path=Path("/tmp/lisbon.jpg"),
            score=0.9,
            rationale=["clip_image_frames"],
            timestamp_utc=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
            session_id="s1",
            snippet="Beach trip in Lisbon",
            place_name="Lisbon",
        )
        monkeypatch.setattr(api_mod._retriever, "retrieve_by_image", lambda *a, **k: [hit])
        monkeypatch.setattr(api_mod._cross_encoder, "rerank", lambda hits, query: hits)

        class _FakeSynth:
            available = True

            def synthesize(self, query, cards):
                return AnswerResult(text="You were at the beach in Lisbon. [1]", cited_session_ids=[cards[0].session_id])

        monkeypatch.setattr(api_mod, "_answer_synthesizer", _FakeSynth())

    def test_image_query_with_text_synthesizes_answer(self, api_client, monkeypatch) -> None:
        import app.api.main as api_mod

        self._wire(api_mod, monkeypatch)
        resp = api_client.post(
            "/query/image",
            files={"image": ("q.jpg", b"\xff\xd8\xff-fake-jpeg", "image/jpeg")},
            data={"query": "where was this beach", "top_k": "5"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Lisbon" in body["answer"]
        assert body["answer_citations"] == ["s1"]

    def test_image_query_without_text_has_no_answer(self, api_client, monkeypatch) -> None:
        import app.api.main as api_mod

        # Synthesizer is available, but with no question there's nothing to answer.
        self._wire(api_mod, monkeypatch)
        resp = api_client.post(
            "/query/image",
            files={"image": ("q.jpg", b"\xff\xd8\xff-fake-jpeg", "image/jpeg")},
            data={"top_k": "5"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] is None
        assert body["answer_citations"] == []

