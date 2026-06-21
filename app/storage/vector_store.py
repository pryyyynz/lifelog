"""Qdrant vector store wrapper with graceful degradation when server is unavailable."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any

from app.models.contracts import NormalizedChunkRecord

logger = logging.getLogger(__name__)

# Payload embedding key per collection
_EMBEDDING_KEY: dict[str, str] = {
    "text_chunks": "text_embedding",
    "image_frames": "image_embedding",
    "video_frames": "image_embedding",
    "audio_transcripts": "text_embedding",
}


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert an arbitrary chunk_id string to a deterministic Qdrant-compatible UUID."""
    digest = hashlib.sha256(chunk_id.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex=digest))


class VectorStore:
    """Wraps qdrant-client with connection resilience and a clean ingest interface."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        text_dim: int = 1024,
        image_dim: int = 768,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._text_dim = text_dim
        self._image_dim = image_dim
        self._client: Any = None
        self._available = False
        self._connect()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(cls) -> VectorStore:
        url = os.getenv("LIFELOG_QDRANT_URL", "http://127.0.0.1:6333")
        api_key = os.getenv("LIFELOG_QDRANT_API_KEY") or None
        text_model = os.getenv("LIFELOG_TEXT_EMBEDDING_MODEL", "intfloat/e5-large-v2")
        image_model = os.getenv("LIFELOG_IMAGE_MODEL", "ViT-L-14")
        text_dim = 768 if "nomic" in text_model or "base" in text_model else 1024
        image_dim = 512 if "B-32" in image_model else 768
        return cls(url=url, api_key=api_key, text_dim=text_dim, image_dim=image_dim)

    def _connect(self) -> None:
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415

            self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=5.0)
            self._client.get_collections()  # lightweight connectivity probe
            self._available = True
            logger.debug("Qdrant connected at %s", self._url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant unavailable at %s — vectors will be skipped: %s", self._url, exc)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collections(self) -> dict[str, bool]:
        """Create the four standard collections if they don't already exist.

        Returns a mapping of {collection_name: was_created}.
        """
        if not self._available or self._client is None:
            return {}
        try:
            from qdrant_client.models import (  # noqa: PLC0415
                Distance,
                PayloadSchemaType,
                VectorParams,
            )
        except ImportError:
            return {}

        dims = {
            "text_chunks": self._text_dim,
            "image_frames": self._image_dim,
            "video_frames": self._image_dim,
            "audio_transcripts": self._text_dim,
        }
        payload_indexes: list[tuple[str, Any]] = [
            ("session_id", PayloadSchemaType.KEYWORD),
            ("timestamp_utc", PayloadSchemaType.KEYWORD),
            ("source_type", PayloadSchemaType.KEYWORD),
            ("file_path", PayloadSchemaType.KEYWORD),
        ]

        existing = {col.name for col in self._client.get_collections().collections}
        result: dict[str, bool] = {}
        for name, dim in dims.items():
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                for field, schema_type in payload_indexes:
                    try:
                        self._client.create_payload_index(
                            collection_name=name,
                            field_name=field,
                            field_schema=schema_type,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                result[name] = True
            else:
                result[name] = False
        return result

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert_records(self, records: list[NormalizedChunkRecord]) -> dict[str, str]:
        """Upsert vectors for all records that carry an embedding.

        Returns a mapping of {chunk_id: point_uuid} for successfully upserted vectors.
        """
        if not self._available or self._client is None:
            return {}
        try:
            from qdrant_client.models import PointStruct  # noqa: PLC0415
        except ImportError:
            return {}

        points_by_collection: dict[str, list[Any]] = {}
        id_mapping: dict[str, str] = {}

        for record in records:
            collection = record.vector_collection
            if collection is None:
                continue
            emb_key = _EMBEDDING_KEY.get(collection)
            if emb_key is None:
                continue
            vector = record.metadata.get(emb_key)
            if not isinstance(vector, list) or not vector:
                continue

            point_id = chunk_id_to_point_id(record.chunk_id)
            payload: dict[str, Any] = {
                "chunk_id": record.chunk_id,
                "session_id": record.session_id,
                "timestamp_utc": record.timestamp_utc.isoformat() if record.timestamp_utc else None,
                "source_type": record.source_type,
                "lat": record.lat,
                "lon": record.lon,
                "file_path": str(record.file_path),
            }
            points_by_collection.setdefault(collection, []).append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )
            id_mapping[record.chunk_id] = point_id

        upserted: dict[str, str] = {}
        for collection, points in points_by_collection.items():
            try:
                self._client.upsert(collection_name=collection, points=points, wait=True)
                for point in points:
                    chunk_id = point.payload["chunk_id"]
                    upserted[chunk_id] = str(point.id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qdrant upsert failed for %s: %s", collection, exc)
        return upserted

    # ------------------------------------------------------------------
    # Delete path
    # ------------------------------------------------------------------

    def delete_by_file_path(self, file_path: str) -> None:
        """Delete all vectors for a given file path across all collections."""
        if not self._available or self._client is None:
            return
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415
        except ImportError:
            return

        for collection in _EMBEDDING_KEY:
            try:
                self._client.delete(
                    collection_name=collection,
                    points_selector=Filter(
                        must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
                    ),
                )
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Search path
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense vector search. Returns list of {id, score, payload} dicts."""
        if not self._available or self._client is None:
            return []
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415
        except ImportError:
            return []

        qdrant_filter = None
        if filters:
            must = []
            for key in ("source_type", "session_id"):
                if key in filters:
                    must.append(FieldCondition(key=key, match=MatchValue(value=filters[key])))
            if must:
                qdrant_filter = Filter(must=must)

        try:
            # qdrant-client >= 1.10 replaced `.search()` with `.query_points()`;
            # fall back to the legacy call for older clients.
            if hasattr(self._client, "query_points"):
                results = self._client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                ).points
            else:
                results = self._client.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
            return [
                {"id": str(r.id), "score": float(r.score), "payload": r.payload or {}}
                for r in results
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant search failed in %s: %s", collection, exc)
            return []

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def fetch_all_point_ids(self, collection: str) -> set[str]:
        """Fetch all point IDs from a collection for consistency checking."""
        if not self._available or self._client is None:
            return set()
        try:
            all_ids: set[str] = set()
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for point in points:
                    all_ids.add(str(point.id))
                if offset is None:
                    break
            return all_ids
        except Exception:  # noqa: BLE001
            return set()
