"""AI enrichment framework: derive searchable understanding from ingested media."""

from app.enrich.base import (
    Enricher,
    EnrichmentOutput,
    SourceChunk,
    derived_text_record,
)
from app.enrich.registry import build_enrichers
from app.enrich.runner import EnrichmentRunner, EnrichmentSummary

__all__ = [
    "Enricher",
    "EnrichmentOutput",
    "EnrichmentRunner",
    "EnrichmentSummary",
    "SourceChunk",
    "build_enrichers",
    "derived_text_record",
]
