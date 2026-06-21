"""Tiny local retrieval smoke path used before the real vector index exists."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeDocument:
    doc_id: str
    text: str


class SmokeIndex:
    def __init__(self) -> None:
        self._docs: list[SmokeDocument] = []

    def add(self, document: SmokeDocument) -> None:
        self._docs.append(document)

    def query(self, text: str, limit: int = 5) -> list[SmokeDocument]:
        terms = {term.lower() for term in text.split()}

        def score(document: SmokeDocument) -> int:
            document_terms = {term.lower().strip(".,!?") for term in document.text.split()}
            return len(terms & document_terms)

        ranked = sorted(self._docs, key=score, reverse=True)
        return [document for document in ranked[:limit] if score(document) > 0]

