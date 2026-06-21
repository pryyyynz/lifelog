from app.retrieval.smoke import SmokeDocument, SmokeIndex


def test_smoke_index_returns_matching_document() -> None:
    index = SmokeIndex()
    index.add(SmokeDocument(doc_id="journal-1", text="Lisbon cafe project notes"))
    index.add(SmokeDocument(doc_id="photo-1", text="Accra beach photo"))

    results = index.query("project in Lisbon")

    assert [result.doc_id for result in results] == ["journal-1"]

