"""Ingestion layer boundary."""

from app.ingest.base import BaseIngestor, DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.images import ExifExtractor, FilesystemPhotoIngestor
from app.ingest.registry import SourceConfig, SourceKind, SourceRegistry
from app.ingest.runner import IngestRunner, IngestRunSummary
from app.ingest.text import EmailIngestor, ObsidianIngestor, TextSourceIngestor

__all__ = [
    "BaseIngestor",
    "DiscoveredItem",
    "ExtractedItem",
    "ExifExtractor",
    "FilesystemPhotoIngestor",
    "EmailIngestor",
    "IngestContext",
    "IngestRunner",
    "IngestRunSummary",
    "ObsidianIngestor",
    "SourceConfig",
    "SourceKind",
    "SourceRegistry",
    "TextSourceIngestor",
]
