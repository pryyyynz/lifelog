"""Build the list of active enrichers from configuration.

Only enrichers whose per-feature flag is on are constructed. Later phases register
caption/tags/action/faces enrichers here; each is one line.
"""

from __future__ import annotations

from app.config import AppConfig, get_config
from app.enrich.base import Enricher


def build_enrichers(config: AppConfig | None = None) -> list[Enricher]:
    config = config or get_config()
    enr = config.enrichment
    enrichers: list[Enricher] = []

    if enr.ocr:
        from app.enrich.ocr import OcrEnricher  # noqa: PLC0415

        enrichers.append(OcrEnricher(languages=enr.ocr_languages))

    if enr.caption:
        from app.enrich.caption import CaptionEnricher  # noqa: PLC0415

        enrichers.append(CaptionEnricher(model_name=enr.caption_model))

    if enr.vlm:
        from app.enrich.vlm import VlmEnricher  # noqa: PLC0415

        enrichers.append(VlmEnricher(model_name=enr.vlm_model))

    if enr.tags:
        from app.enrich.tags import TagEnricher  # noqa: PLC0415

        enrichers.append(
            TagEnricher(
                model_name=enr.tag_model,
                pretrained=enr.tag_pretrained,
                labels=enr.tag_labels or None,
                top_k=enr.tag_top_k,
                threshold=enr.tag_threshold,
            )
        )

    if enr.action:
        from app.enrich.action import ActionEnricher  # noqa: PLC0415

        enrichers.append(
            ActionEnricher(
                model_name=enr.action_model,
                labels=enr.action_labels or None,
                top_k=enr.action_top_k,
                threshold=enr.action_threshold,
            )
        )

    if enr.faces:
        from app.enrich.faces import FaceEnricher  # noqa: PLC0415

        enrichers.append(
            FaceEnricher(model_name=enr.face_model, det_threshold=enr.face_det_threshold)
        )

    return enrichers
