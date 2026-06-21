"""Tests for the voice-search /transcribe endpoint and model caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest.audio import AudioTranscript, TranscriptSegment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path):
    """FastAPI TestClient with a temp store, mirroring test_section14_api_cli."""
    import importlib
    import os

    os.environ["LIFELOG_SQLITE_PATH"] = str(tmp_path / "test.db")
    os.environ["LIFELOG_DATA_DIR"] = str(tmp_path)
    os.environ["LIFELOG_LOG_DIR"] = str(tmp_path / "logs")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    import app.api.main as api_mod
    importlib.reload(api_mod)

    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as client:
        yield client, api_mod


class _FakeEngine:
    """Stand-in transcriber that records calls and returns a canned transcript."""

    def __init__(self, backend: str = "fake", text: str = "find my hiking photos") -> None:
        self.backend = backend
        self._text = text
        self.calls: list[dict] = []

    def transcribe(self, audio_path, *, original_path, model_name=None):  # noqa: ANN001
        self.calls.append({"model_name": model_name})
        if self.backend == "unavailable":
            return AudioTranscript((), None, None, original_path, {"engine": "unavailable", "error": "x"})
        return AudioTranscript(
            segments=(TranscriptSegment(text=self._text, start=0.0, end=1.0),),
            language="en",
            duration=1.0,
            file_path=original_path,
            metadata={"engine": "fake"},
        )


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestTranscribeEndpoint:
    def test_transcribes_audio_to_text(self, api_client, monkeypatch) -> None:
        client, api_mod = api_client
        fake = _FakeEngine(text="find my hiking photos")
        api_mod._query_transcriber = fake
        # Skip real ffmpeg; pretend conversion succeeded.
        monkeypatch.setattr(api_mod, "_get_query_transcriber", lambda: fake)
        monkeypatch.setattr("app.ingest.audio.convert_audio_to_wav", lambda src, dst: dst)

        resp = client.post("/transcribe", files={"audio": ("q.webm", b"fake-bytes", "audio/webm")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "find my hiking photos"
        assert data["language"] == "en"

    def test_uses_query_model(self, api_client, monkeypatch) -> None:
        client, api_mod = api_client
        fake = _FakeEngine()
        monkeypatch.setattr(api_mod, "_get_query_transcriber", lambda: fake)
        monkeypatch.setattr("app.ingest.audio.convert_audio_to_wav", lambda src, dst: dst)

        client.post("/transcribe", files={"audio": ("q.webm", b"fake-bytes", "audio/webm")})
        # Endpoint should pass the configured (smaller) query model, not the ingest default.
        assert fake.calls[0]["model_name"] == api_mod._config.models.query_transcription_model

    def test_empty_upload_rejected(self, api_client, monkeypatch) -> None:
        client, api_mod = api_client
        monkeypatch.setattr(api_mod, "_get_query_transcriber", lambda: _FakeEngine())
        resp = client.post("/transcribe", files={"audio": ("q.webm", b"", "audio/webm")})
        assert resp.status_code == 400

    def test_engine_unavailable_returns_503(self, api_client, monkeypatch) -> None:
        client, api_mod = api_client
        monkeypatch.setattr(api_mod, "_get_query_transcriber", lambda: _FakeEngine(backend="unavailable"))
        resp = client.post("/transcribe", files={"audio": ("q.webm", b"fake-bytes", "audio/webm")})
        assert resp.status_code == 503

    def test_ffmpeg_failure_returns_500(self, api_client, monkeypatch) -> None:
        client, api_mod = api_client
        monkeypatch.setattr(api_mod, "_get_query_transcriber", lambda: _FakeEngine())
        monkeypatch.setattr("app.ingest.audio.convert_audio_to_wav", lambda src, dst: None)
        resp = client.post("/transcribe", files={"audio": ("q.webm", b"fake-bytes", "audio/webm")})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Engine caching / override behavior
# ---------------------------------------------------------------------------


class TestTranscriptionEngineCaching:
    def test_unavailable_backend_is_safe(self) -> None:
        from app.ingest.audio import TranscriptionEngine

        engine = TranscriptionEngine("unavailable")
        result = engine.transcribe(Path("x.wav"), original_path=Path("x.wav"), model_name="base")
        assert result.segments == ()
        assert result.metadata["engine"] == "unavailable"

    def test_model_cache_initialized(self) -> None:
        from app.ingest.audio import TranscriptionEngine

        engine = TranscriptionEngine("whisperx")
        assert engine._model_cache == {}
