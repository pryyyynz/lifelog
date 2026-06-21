"""Tests for Phase 1 enrichers (caption, tags, action) and the enrich API."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.enrich.action import ActionEnricher
from app.enrich.base import STATUS_DONE, STATUS_SKIPPED, SourceChunk
from app.enrich.caption import CaptionEnricher
from app.enrich.runner import EnrichmentRunner
from app.enrich.tags import TagEnricher
from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fake backends + chunk builders
# ---------------------------------------------------------------------------


class _FakeCaption:
    def __init__(self, text="a cat on a sofa"):
        self.text = text

    def caption(self, image_path: Path) -> str:
        return self.text


class _FakeTags:
    def __init__(self, tags=("cat", "sofa")):
        self._tags = list(tags)

    def tags(self, image_path: Path) -> list[str]:
        return self._tags


class _FakeActions:
    def __init__(self, actions=("playing with a dog",)):
        self._actions = list(actions)

    def actions(self, video_path: Path, start_sec: float) -> list[str]:
        return self._actions


def _photo_chunk(img: Path) -> SourceChunk:
    return SourceChunk(
        chunk_id="p0", source_id="s", source_type="photo", file_path=img,
        chunk_identity="photo:0", timestamp_utc=_T0, session_id=None,
        lat=None, lon=None, metadata={},
    )


def _video_frame_chunk(video: Path, frame: Path) -> SourceChunk:
    return SourceChunk(
        chunk_id="vf0", source_id="s", source_type="video", file_path=video,
        chunk_identity="video_frame:scene_0001", timestamp_utc=_T0, session_id=None,
        lat=None, lon=None,
        metadata={"scene_id": "scene_0001", "frame_path": str(frame)},
        timestamp_start_sec=12.0,
    )


def _video_transcript_chunk(video: Path) -> SourceChunk:
    return SourceChunk(
        chunk_id="vt0", source_id="s", source_type="video", file_path=video,
        chunk_identity="video_transcript:0", timestamp_utc=_T0, session_id=None,
        lat=None, lon=None, metadata={},
    )


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------


class TestCaptionEnricher:
    def test_caption_photo(self, tmp_path: Path) -> None:
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        out = CaptionEnricher(backend=_FakeCaption("a cat on a sofa")).enrich(_photo_chunk(img))
        assert out.status == STATUS_DONE
        rec = out.records[0]
        assert rec.text == "a cat on a sofa"
        assert rec.metadata["chunk_identity"] == "caption:0"
        assert rec.metadata["derived_from"] == "p0"
        assert rec.vector_collection == "text_chunks"

    def test_caption_video_frame_uses_frame_path(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"x")
        out = CaptionEnricher(backend=_FakeCaption("people dancing")).enrich(
            _video_frame_chunk(video, frame)
        )
        assert out.status == STATUS_DONE
        assert out.records[0].metadata["chunk_identity"] == "caption:scene_0001"

    def test_caption_skips_transcript_chunk(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        out = CaptionEnricher(backend=_FakeCaption()).enrich(_video_transcript_chunk(video))
        assert out.status == STATUS_SKIPPED

    def test_runner_persists_caption(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "t.db")
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        store.upsert_chunks(
            "s",
            [
                NormalizedChunkRecord(
                    chunk_id="p0", source_type="photo", file_path=img, text=None,
                    timestamp_utc=_T0, vector_collection="image_frames",
                    metadata={"chunk_identity": "photo:0"},
                )
            ],
        )
        summary = EnrichmentRunner(store, [CaptionEnricher(backend=_FakeCaption("a sunny beach"))]).run()
        assert summary.done == 1
        assert "a sunny beach" in [r["text"] for r in store.fetch_chunks()]


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTagEnricher:
    def test_tags_photo(self, tmp_path: Path) -> None:
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        out = TagEnricher(backend=_FakeTags(("beach", "sunset"))).enrich(_photo_chunk(img))
        assert out.status == STATUS_DONE
        rec = out.records[0]
        assert rec.text == "beach, sunset"
        assert rec.metadata["tags"] == ["beach", "sunset"]
        assert rec.metadata["chunk_identity"] == "tags:0"

    def test_no_tags_is_skipped(self, tmp_path: Path) -> None:
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        out = TagEnricher(backend=_FakeTags(())).enrich(_photo_chunk(img))
        assert out.status == STATUS_SKIPPED


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class TestActionEnricher:
    def test_action_video_scene(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        out = ActionEnricher(backend=_FakeActions(("cooking", "eating"))).enrich(
            _video_frame_chunk(video, tmp_path / "f.jpg")
        )
        assert out.status == STATUS_DONE
        rec = out.records[0]
        assert rec.text == "cooking, eating"
        assert rec.metadata["actions"] == ["cooking", "eating"]
        assert rec.metadata["chunk_identity"] == "action:scene_0001"

    def test_action_skips_non_scene_chunk(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        out = ActionEnricher(backend=_FakeActions()).enrich(_video_transcript_chunk(video))
        assert out.status == STATUS_SKIPPED


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builds_phase1_enrichers_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("LIFELOG_ENRICH_OCR", "0")
        monkeypatch.setenv("LIFELOG_ENRICH_CAPTION", "1")
        monkeypatch.setenv("LIFELOG_ENRICH_TAGS", "1")
        monkeypatch.setenv("LIFELOG_ENRICH_ACTION", "1")
        from app.config import get_config
        from app.enrich.registry import build_enrichers

        names = [e.name for e in build_enrichers(get_config())]
        assert names == ["caption", "tags", "action"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LIFELOG_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LIFELOG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LIFELOG_LOG_DIR", str(tmp_path / "logs"))
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    import app.api.main as api_mod
    importlib.reload(api_mod)
    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as client:
        yield client, api_mod


class TestEnrichApi:
    def test_trigger_and_status(self, api_client) -> None:
        client, _ = api_client
        # OCR is enabled by default; trigger should start a run.
        resp = client.post("/enrich/trigger", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        status = client.get("/enrich/status")
        assert status.status_code == 200
        assert status.json()["state"] in ("idle", "running", "paused", "done", "error")

    def test_trigger_400_when_all_disabled(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("LIFELOG_SQLITE_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("LIFELOG_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("LIFELOG_LOG_DIR", str(tmp_path / "logs"))
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        for flag in ("OCR", "CAPTION", "TAGS", "ACTION"):
            monkeypatch.setenv(f"LIFELOG_ENRICH_{flag}", "0")

        import app.api.main as api_mod
        importlib.reload(api_mod)
        from fastapi.testclient import TestClient

        with TestClient(api_mod.app) as client:
            resp = client.post("/enrich/trigger", json={})
            assert resp.status_code == 400

    def test_query_gate_starts_idle(self, api_client) -> None:
        _, api_mod = api_client
        assert api_mod._query_in_progress() is False
