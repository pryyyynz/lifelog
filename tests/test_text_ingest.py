from email.message import EmailMessage
from pathlib import Path
import mailbox

from app.ingest.registry import SourceKind, SourceRegistry, build_source_config
from app.ingest.runner import IngestRunner
from app.ingest.text import prepare_embedding_text, prepare_query_text
from app.storage.metadata import MetadataStore


def test_markdown_ingest_parses_frontmatter_and_cleans_wikilinks(tmp_path: Path) -> None:
    notes = tmp_path / "vault"
    notes.mkdir()
    (notes / "entry.md").write_text(
        "---\n"
        "date: 2026-05-01\n"
        "tags: [project, kofi]\n"
        "aliases:\n"
        "  - Lisbon note\n"
        "---\n\n"
        "Met [[Kofi Mensah|Kofi]] at the cafe.\n\n"
        "![[photo.jpg]]\n",
        encoding="utf-8",
    )
    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.TEXT, notes))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    summary = IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert summary.processed_items == 1
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Met Kofi at the cafe."
    assert chunks[0]["timestamp_utc"].startswith("2026-05-01T00:00:00")
    assert "Kofi" in chunks[0]["metadata_json"]
    assert "passage:" in chunks[0]["metadata_json"]


def test_e5_embedding_prefix_is_added() -> None:
    assert prepare_embedding_text("hello", "intfloat/e5-large-v2") == "passage: hello"
    assert prepare_query_text("hello", "intfloat/e5-large-v2") == "query: hello"


def test_mbox_ingest_strips_reply_chain_and_signature(tmp_path: Path) -> None:
    mail_path = tmp_path / "mail.mbox"
    box = mailbox.mbox(mail_path)
    message = EmailMessage()
    message["From"] = "kofi@example.com"
    message["To"] = "me@example.com"
    message["Subject"] = "Cafe plan"
    message["Date"] = "Fri, 1 May 2026 12:00:00 +0000"
    message["Message-Id"] = "<cafe@example.com>"
    message.set_content("Let's meet at 4.\n\nBest,\nKofi\n\n> quoted old note")
    box.add(message)
    box.flush()
    box.close()

    registry = SourceRegistry(tmp_path / "sources.json")
    registry.upsert(build_source_config(SourceKind.EMAIL, mail_path))
    store = MetadataStore(tmp_path / "lifelog.sqlite3")

    summary = IngestRunner(registry, store).run(full=True)
    chunks = store.fetch_chunks()

    assert summary.processed_items == 1
    assert len(chunks) == 1
    assert chunks[0]["source_type"] == "email"
    assert chunks[0]["text"] == "Let's meet at 4."
    assert "Cafe plan" in chunks[0]["metadata_json"]
