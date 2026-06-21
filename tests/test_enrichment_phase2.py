"""Tests for Phase 2: face detection, clustering, naming, and the people API."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.enrich.base import STATUS_DONE, STATUS_SKIPPED, SourceChunk
from app.enrich.clustering import FaceClusterer, name_cluster
from app.enrich.faces import DetectedFace, FaceEnricher
from app.enrich.runner import EnrichmentRunner
from app.models.contracts import FaceRecord, NormalizedChunkRecord
from app.storage.metadata import MetadataStore

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


class _FakeFaceBackend:
    def __init__(self, faces):
        self._faces = faces  # list of (bbox, score, embedding)

    def detect(self, image_path: Path):
        return [DetectedFace(bbox=b, det_score=s, embedding=e) for (b, s, e) in self._faces]


def _photo_chunk(img: Path, chunk_id: str = "p0") -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id, source_id="s", source_type="photo", file_path=img,
        chunk_identity="photo:0", timestamp_utc=_T0, session_id=None,
        lat=None, lon=None, metadata={},
    )


def _face(face_id: str, chunk_id: str, emb: list[float], path: str = "/p/a.jpg") -> FaceRecord:
    return FaceRecord(
        face_id=face_id, chunk_id=chunk_id, source_id="s", source_type="photo",
        file_path=Path(path), timestamp_utc=_T0, bbox=(0.0, 0.0, 10.0, 10.0),
        det_score=0.9, embedding=emb,
    )


# ---------------------------------------------------------------------------
# FaceEnricher
# ---------------------------------------------------------------------------


class TestFaceEnricher:
    def test_detects_faces(self, tmp_path: Path) -> None:
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        enr = FaceEnricher(backend=_FakeFaceBackend([((0, 0, 10, 10), 0.9, [1.0, 0.0, 0.0, 0.0])]))
        out = enr.enrich(_photo_chunk(img))
        assert out.status == STATUS_DONE
        assert len(out.faces) == 1
        assert out.faces[0].chunk_id == "p0"
        assert out.faces[0].embedding == [1.0, 0.0, 0.0, 0.0]
        assert out.faces[0].face_id

    def test_no_faces_is_done_with_empty(self, tmp_path: Path) -> None:
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        out = FaceEnricher(backend=_FakeFaceBackend([])).enrich(_photo_chunk(img))
        assert out.status == STATUS_DONE
        assert out.faces == ()

    def test_missing_image_skipped(self, tmp_path: Path) -> None:
        out = FaceEnricher(backend=_FakeFaceBackend([])).enrich(_photo_chunk(tmp_path / "nope.jpg"))
        assert out.status == STATUS_SKIPPED

    def test_runner_persists_faces(self, tmp_path: Path) -> None:
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
        backend = _FakeFaceBackend(
            [((0, 0, 10, 10), 0.9, [1.0, 0.0, 0.0, 0.0]), ((0, 0, 5, 5), 0.8, [0.0, 1.0, 0.0, 0.0])]
        )
        summary = EnrichmentRunner(store, [FaceEnricher(backend=backend)]).run()
        assert summary.done == 1
        assert len(store.faces_without_cluster()) == 2


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestFaceClusterer:
    def _seed(self, store: MetadataStore) -> None:
        store.upsert_faces(
            [
                _face("f1", "p0", [1.0, 0.0, 0.0, 0.0], "/p/a.jpg"),
                _face("f2", "p1", [0.99, 0.01, 0.0, 0.0], "/p/b.jpg"),
                _face("f3", "p2", [0.0, 1.0, 0.0, 0.0], "/p/c.jpg"),
            ]
        )

    def _face_to_cluster(self, store: MetadataStore) -> dict[str, str]:
        return {
            str(f["face_id"]): str(cluster["cluster_id"])
            for cluster in store.get_clusters()
            for f in store.faces_for_cluster(cluster["cluster_id"])
        }

    def test_groups_similar_separates_distinct(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "t.db")
        self._seed(store)
        summary = FaceClusterer(store, threshold=0.5).cluster_new()
        assert summary.processed == 3
        assert len(store.get_clusters()) == 2
        mapping = self._face_to_cluster(store)
        assert mapping["f1"] == mapping["f2"]
        assert mapping["f1"] != mapping["f3"]

    def test_idempotent(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "t.db")
        self._seed(store)
        FaceClusterer(store, threshold=0.5).cluster_new()
        again = FaceClusterer(store, threshold=0.5).cluster_new()
        assert again.processed == 0
        assert again.new_clusters == 0

    def test_name_cluster_makes_person_searchable(self, tmp_path: Path) -> None:
        store = MetadataStore(tmp_path / "t.db")
        self._seed(store)
        FaceClusterer(store, threshold=0.5).cluster_new()
        mapping = self._face_to_cluster(store)
        cid = mapping["f1"]  # cluster with f1 + f2

        updated = name_cluster(store, cid, "Sarah")
        assert updated == 2  # two distinct source chunks (p0, p1)
        assert store.get_cluster(cid)["person_name"] == "Sarah"
        assert "Sarah" in [r["text"] for r in store.fetch_chunks()]


# ---------------------------------------------------------------------------
# People API
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LIFELOG_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LIFELOG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LIFELOG_LOG_DIR", str(tmp_path / "logs"))
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    # Seed faces + clusters before lifespan (same sqlite path).
    store = MetadataStore(tmp_path / "test.db")
    store.upsert_faces(
        [
            _face("f1", "p0", [1.0, 0.0, 0.0, 0.0], "/p/a.jpg"),
            _face("f2", "p1", [0.98, 0.02, 0.0, 0.0], "/p/b.jpg"),
            _face("f3", "p2", [0.0, 1.0, 0.0, 0.0], "/p/c.jpg"),
        ]
    )
    FaceClusterer(store, threshold=0.5).cluster_new()

    import app.api.main as api_mod
    importlib.reload(api_mod)
    from fastapi.testclient import TestClient

    with TestClient(api_mod.app) as client:
        yield client


class TestPeopleApi:
    def test_list_people(self, api_client) -> None:
        resp = api_client.get("/people")
        assert resp.status_code == 200
        people = resp.json()
        assert len(people) == 2
        assert sum(p["face_count"] for p in people) == 3

    def test_name_then_reflected(self, api_client) -> None:
        cid = api_client.get("/people").json()[0]["cluster_id"]
        named = api_client.post(f"/people/{cid}/name", json={"name": "Alex"})
        assert named.status_code == 200
        assert named.json()["status"] == "named"
        people = {p["cluster_id"]: p["person_name"] for p in api_client.get("/people").json()}
        assert people[cid] == "Alex"

    def test_cluster_faces_listed(self, api_client) -> None:
        cid = api_client.get("/people").json()[0]["cluster_id"]
        faces = api_client.get(f"/people/{cid}/faces")
        assert faces.status_code == 200
        assert len(faces.json()) >= 1
        assert "preview_url" in faces.json()[0]

    def test_unknown_cluster_404(self, api_client) -> None:
        assert api_client.get("/people/nope/faces").status_code == 404
        assert api_client.post("/people/nope/name", json={"name": "X"}).status_code == 404
