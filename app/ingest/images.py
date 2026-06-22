"""Photo discovery, metadata extraction, and optional image embedding."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingest.base import DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.file_ingestor import LocalFileIngestor
from app.models.contracts import NormalizedChunkRecord


@dataclass(frozen=True)
class PhotoMetadata:
    timestamp_utc: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    camera_model: str | None = None
    width: int | None = None
    height: int | None = None
    description: str | None = None
    album_names: tuple[str, ...] = ()
    face_names: tuple[str, ...] = ()
    raw: dict[str, Any] | None = None


class ExifExtractor:
    """Best-effort EXIF reader with graceful fallback when Pillow is unavailable."""

    def extract(self, path: Path) -> PhotoMetadata:
        try:
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            from PIL import ExifTags, Image
        except ImportError:
            return PhotoMetadata(raw={"exif_status": "pillow_unavailable"})

        try:
            with Image.open(path) as image:
                width, height = image.size
                exif = image.getexif()
                if not exif:
                    return PhotoMetadata(width=width, height=height, raw={"exif_status": "missing"})
                tags = {
                    ExifTags.TAGS.get(tag_id, str(tag_id)): value for tag_id, value in exif.items()
                }
                gps = _extract_gps_ifd(exif, ExifTags.TAGS, ExifTags.GPSTAGS)
                lat, lon = _gps_decimal(gps)
                return PhotoMetadata(
                    timestamp_utc=_parse_exif_datetime(tags.get("DateTimeOriginal")),
                    lat=lat,
                    lon=lon,
                    camera_model=str(tags["Model"]) if tags.get("Model") else None,
                    width=width,
                    height=height,
                    raw={"exif_status": "ok"},
                )
        except Exception as exc:  # noqa: BLE001 - ingest runner records item-level failures.
            return PhotoMetadata(raw={"exif_status": "error", "error": str(exc)})


class FilesystemPhotoIngestor(LocalFileIngestor):
    """Ingests photos from ordinary filesystem folders."""

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        sidecar = _sidecar_metadata(item.path)
        exif = ExifExtractor().extract(item.path)
        metadata = _merge_photo_metadata(exif, sidecar, item.mtime_ns)
        return ExtractedItem(discovered=item, payload=metadata, metadata=metadata.raw or {})

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        photo = item.payload
        if not isinstance(photo, PhotoMetadata):
            return []
        identity = "photo:0"
        metadata = {
            "chunk_identity": identity,
            "camera_model": photo.camera_model,
            "width": photo.width,
            "height": photo.height,
            "description": photo.description,
            "album_names": list(photo.album_names),
            "face_names": list(photo.face_names),
            "raw": photo.raw or {},
        }
        return [
            NormalizedChunkRecord(
                chunk_id=_chunk_id(item.discovered.path, identity),
                source_type="photo",
                file_path=item.discovered.path,
                text=photo.description,
                timestamp_utc=photo.timestamp_utc,
                vector_collection="image_frames",
                lat=photo.lat,
                lon=photo.lon,
                metadata=metadata,
            )
        ]

    def embed(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        embedder = OpenClipImageEmbedder.from_environment()
        embedded: list[NormalizedChunkRecord] = []
        for record in records:
            metadata = dict(record.metadata)
            vector = embedder.embed(record.file_path)
            if vector is None:
                metadata["image_embedding_status"] = embedder.status
                embedded.append(_replace_metadata(record, metadata))
                continue
            metadata["image_embedding"] = vector
            metadata["image_embedding_status"] = "ok"
            embedded.append(_replace_metadata(record, metadata))
        return embedded


class GooglePhotosIngestor(FilesystemPhotoIngestor):
    """Google Photos Takeout ingestor that prefers JSON sidecars over EXIF."""


class ApplePhotosIngestor(FilesystemPhotoIngestor):
    """Manual Apple Photos export ingestor for originals with preserved EXIF."""


class OpenClipImageEmbedder:
    def __init__(self, enabled: bool, model_name: str = "ViT-L-14", pretrained: str = "openai") -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.pretrained = pretrained
        self.status = "disabled"

    @classmethod
    def from_environment(cls) -> OpenClipImageEmbedder:
        enabled = os.getenv("LIFELOG_ENABLE_IMAGE_EMBEDDING", "").lower() in {"1", "true", "yes"}
        return cls(
            enabled=enabled,
            model_name=os.getenv("LIFELOG_IMAGE_MODEL", "ViT-L-14"),
            pretrained=os.getenv("LIFELOG_IMAGE_PRETRAINED", "openai"),
        )

    def embed(self, path: Path) -> list[float] | None:
        if not self.enabled:
            return None
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError:
            self.status = "dependencies_unavailable"
            return None

        try:
            # Load the model once (on GPU when available) and cache it — recreating
            # it per image was both slow and CPU-only.
            if getattr(self, "_model", None) is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                )
                self._model.to(self._device)
                self._model.eval()
            image = Image.open(path).convert("RGB").resize((224, 224))
            tensor = self._preprocess(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            self.status = "ok"
            return [float(value) for value in features.squeeze(0).cpu().tolist()]
        except Exception as exc:  # noqa: BLE001
            self.status = f"embedding_error: {exc}"
            return None


def _sidecar_metadata(path: Path) -> PhotoMetadata | None:
    candidates = (path.with_name(path.name + ".json"), path.with_suffix(path.suffix + ".json"))
    for candidate in candidates:
        if candidate.exists():
            return _parse_google_sidecar(candidate)
    return None


def _parse_google_sidecar(path: Path) -> PhotoMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    geo = payload.get("geoDataExif") or payload.get("geoData") or {}
    timestamp = None
    photo_time = payload.get("photoTakenTime") or payload.get("creationTime") or {}
    if isinstance(photo_time, dict) and photo_time.get("timestamp"):
        timestamp = datetime.fromtimestamp(int(photo_time["timestamp"]), tz=UTC)
    return PhotoMetadata(
        timestamp_utc=timestamp,
        lat=_valid_coordinate(geo.get("latitude")),
        lon=_valid_coordinate(geo.get("longitude")),
        camera_model=payload.get("cameraModel"),
        description=payload.get("description") or payload.get("title"),
        raw={"sidecar": payload, "metadata_source": "google_sidecar"},
    )


def _merge_photo_metadata(
    exif: PhotoMetadata,
    sidecar: PhotoMetadata | None,
    mtime_ns: int,
) -> PhotoMetadata:
    preferred = sidecar or exif
    fallback_time = datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=UTC)
    raw = {**(exif.raw or {})}
    if sidecar and sidecar.raw:
        raw.update(sidecar.raw)
    return PhotoMetadata(
        timestamp_utc=preferred.timestamp_utc or exif.timestamp_utc or fallback_time,
        lat=preferred.lat if preferred.lat is not None else exif.lat,
        lon=preferred.lon if preferred.lon is not None else exif.lon,
        camera_model=preferred.camera_model or exif.camera_model,
        width=exif.width,
        height=exif.height,
        description=preferred.description,
        album_names=preferred.album_names,
        face_names=preferred.face_names,
        raw=raw,
    )


def _gps_tags(value: Any, gps_tag_names: dict[int, str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {gps_tag_names.get(key, str(key)): item for key, item in value.items()}


def _extract_gps_ifd(
    exif: Any,
    tag_names: dict[int, str],
    gps_tag_names: dict[int, str],
) -> dict[str, Any]:
    gps_tag_id = next((tag_id for tag_id, name in tag_names.items() if name == "GPSInfo"), None)
    if gps_tag_id is None or gps_tag_id not in exif:
        return {}
    try:
        return _gps_tags(exif.get_ifd(gps_tag_id), gps_tag_names)
    except Exception:  # noqa: BLE001
        return _gps_tags(exif.get(gps_tag_id), gps_tag_names)


def _gps_decimal(gps: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = _rational_dms(gps.get("GPSLatitude"))
    lon = _rational_dms(gps.get("GPSLongitude"))
    if lat is not None and str(gps.get("GPSLatitudeRef", "")).upper() == "S":
        lat = -lat
    if lon is not None and str(gps.get("GPSLongitudeRef", "")).upper() == "W":
        lon = -lon
    return lat, lon


def _rational_dms(value: Any) -> float | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    degrees, minutes, seconds = (_rational_float(part) for part in value)
    return degrees + minutes / 60 + seconds / 3600


def _rational_float(value: Any) -> float:
    try:
        return float(value)
    except TypeError:
        numerator = getattr(value, "numerator", 0)
        denominator = getattr(value, "denominator", 1)
        return float(numerator) / float(denominator or 1)


def _parse_exif_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _valid_coordinate(value: Any) -> float | None:
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None
    return coordinate if coordinate != 0 else None


def _replace_metadata(record: NormalizedChunkRecord, metadata: dict[str, Any]) -> NormalizedChunkRecord:
    return NormalizedChunkRecord(
        chunk_id=record.chunk_id,
        source_type=record.source_type,
        file_path=record.file_path,
        text=record.text,
        timestamp_utc=record.timestamp_utc,
        vector_collection=record.vector_collection,
        vector_id=record.vector_id,
        session_id=record.session_id,
        timestamp_start_sec=record.timestamp_start_sec,
        timestamp_end_sec=record.timestamp_end_sec,
        lat=record.lat,
        lon=record.lon,
        place_name=record.place_name,
        metadata=metadata,
    )


def _chunk_id(path: Path, identity: str) -> str:
    return hashlib.sha256(f"{path.resolve()}::{identity}".encode("utf-8")).hexdigest()
