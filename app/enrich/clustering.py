"""Incremental face clustering and people naming.

``FaceClusterer`` greedily assigns each un-clustered face to the nearest existing
cluster centroid (cosine), creating a new cluster when none is close enough. Running
centroids make it online-friendly and deterministic — no heavy clustering deps.

``name_cluster`` records a person name for a cluster and writes derived ``person:*``
text chunks so the named person becomes searchable (and populates ``person_names``
query matches).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from app.ingest.text import prepare_embedding_text
from app.models.contracts import NormalizedChunkRecord
from app.storage.metadata import MetadataStore


@dataclass
class ClusterSummary:
    processed: int = 0
    new_clusters: int = 0


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


class FaceClusterer:
    def __init__(self, store: MetadataStore, threshold: float = 0.5) -> None:
        self.store = store
        self.threshold = threshold

    def cluster_new(self, batch_limit: int = 500) -> ClusterSummary:
        summary = ClusterSummary()
        # cluster_id -> [centroid (normalized), count]
        clusters: dict[str, list] = {}
        for row in self.store.get_clusters():
            clusters[str(row["cluster_id"])] = [
                _normalize(np.array(json.loads(row["centroid_json"]), dtype=float)),
                int(row["face_count"]),
            ]

        faces = self.store.faces_without_cluster(limit=batch_limit)
        while faces:
            for face in faces:
                emb = _normalize(np.array(json.loads(face["embedding_json"]), dtype=float))
                best_cid, best_sim = None, -1.0
                for cid, (centroid, _count) in clusters.items():
                    sim = float(np.dot(emb, centroid))
                    if sim > best_sim:
                        best_sim, best_cid = sim, cid

                if best_cid is not None and best_sim >= self.threshold:
                    centroid, count = clusters[best_cid]
                    new_count = count + 1
                    merged = _normalize((centroid * count + emb) / new_count)
                    clusters[best_cid] = [merged, new_count]
                    cid = best_cid
                else:
                    cid = "cl_" + hashlib.sha256(str(face["face_id"]).encode()).hexdigest()[:12]
                    clusters[cid] = [emb, 1]
                    summary.new_clusters += 1

                self.store.assign_face_to_cluster(str(face["face_id"]), cid)
                self.store.upsert_cluster(cid, clusters[cid][0].tolist(), clusters[cid][1])
                summary.processed += 1

            faces = self.store.faces_without_cluster(limit=batch_limit)
        return summary


def name_cluster(store: MetadataStore, cluster_id: str, name: str) -> int:
    """Name a cluster and (re)generate searchable ``person:*`` text chunks.

    Returns the number of source chunks updated. Idempotent: the derived chunk
    identity ``person:<cluster_id>`` is stable, so renaming overwrites in place.
    """
    store.name_cluster(cluster_id, name)
    identity = f"person:{cluster_id}"

    # One derived text chunk per distinct parent chunk containing this person.
    seen: dict[tuple, dict] = {}
    for face in store.faces_for_cluster(cluster_id):
        key = (str(face["source_id"]), str(face["file_path"]), str(face["chunk_id"]))
        if key not in seen:
            seen[key] = {
                "source_type": str(face["source_type"]),
                "timestamp_utc": face["timestamp_utc"],
            }

    for (source_id, file_path, chunk_id), extra in seen.items():
        ts = extra["timestamp_utc"]
        record = NormalizedChunkRecord(
            chunk_id=hashlib.sha256(f"{file_path}::{identity}".encode()).hexdigest()[:24],
            source_type=extra["source_type"],  # type: ignore[arg-type]
            file_path=Path(file_path),
            text=name,
            timestamp_utc=datetime.fromisoformat(ts) if ts else None,
            vector_collection="text_chunks",
            metadata={
                "chunk_identity": identity,
                "derived_from": chunk_id,
                "enricher": "faces",
                "person_name": name,
                "cluster_id": cluster_id,
                "embedding_text": prepare_embedding_text(name, model_name="intfloat/e5-large-v2"),
            },
        )
        store.upsert_chunks(source_id, [record])

    return len(seen)
