"""Text, notes, journal export, and email ingestion."""

from __future__ import annotations

import csv
import hashlib
import json
import mailbox
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.message import Message
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any

from app.ingest.base import DiscoveredItem, ExtractedItem, IngestContext
from app.ingest.embedders import embed_text_records
from app.ingest.file_ingestor import LocalFileIngestor
from app.models.contracts import NormalizedChunkRecord

MAX_CHUNK_CHARS = 2048
OVERLAP_CHARS = 256
AUTO_REPLY_HEADERS = ("auto-submitted", "x-autoreply", "x-autorespond")
NEWSLETTER_HINTS = ("list-unsubscribe", "x-mailchimp", "x-campaign", "precedence")


@dataclass(frozen=True)
class TextDocument:
    text: str
    timestamp_utc: datetime | None
    metadata: dict[str, Any]


class TextSourceIngestor(LocalFileIngestor):
    """Ingests local Markdown, text, Notion CSV, Day One JSON, and Journey JSON exports."""

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        suffix = item.path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            document = _extract_markdown(item.path, item.mtime_ns)
        elif suffix == ".txt":
            document = TextDocument(
                text=item.path.read_text(encoding="utf-8", errors="replace"),
                timestamp_utc=_datetime_from_mtime_ns(item.mtime_ns),
                metadata={"format": "plain_text"},
            )
        elif suffix == ".csv":
            document = _extract_notion_csv(item.path, item.mtime_ns)
        elif suffix == ".json":
            document = _extract_json_journal(item.path, item.mtime_ns)
        else:
            document = TextDocument(
                text="",
                timestamp_utc=_datetime_from_mtime_ns(item.mtime_ns),
                metadata={"format": "unsupported"},
            )
        return ExtractedItem(discovered=item, payload=document, metadata=document.metadata)

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        document = item.payload
        if not isinstance(document, TextDocument):
            return []
        chunks = split_text(document.text)
        records: list[NormalizedChunkRecord] = []
        for index, text in enumerate(chunks):
            chunk_identity = f"text:{index}"
            metadata = {
                **document.metadata,
                "chunk_index": index,
                "chunk_identity": chunk_identity,
                "embedding_text": prepare_embedding_text(text, model_name="intfloat/e5-large-v2"),
                "raw_text": text,
                "exact_terms": _extract_exact_terms(text),
            }
            records.append(
                NormalizedChunkRecord(
                    chunk_id=_chunk_id(item.discovered.path, chunk_identity),
                    source_type="text",
                    file_path=item.discovered.path,
                    text=text,
                    timestamp_utc=document.timestamp_utc,
                    vector_collection="text_chunks",
                    metadata=metadata,
                )
            )
        return records

    def embed(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        return embed_text_records(records)


class ObsidianIngestor(TextSourceIngestor):
    """Obsidian-compatible Markdown ingestor."""


class NotionIngestor(TextSourceIngestor):
    """Notion export ingestor for Markdown plus CSV exports."""


class DayOneIngestor(TextSourceIngestor):
    """Day One JSON export ingestor."""


class JourneyIngestor(TextSourceIngestor):
    """Journey JSON export ingestor."""


class EmailIngestor(LocalFileIngestor):
    """Ingests MBOX exports, including Google Takeout mail archives."""

    def extract(self, item: DiscoveredItem, context: IngestContext) -> ExtractedItem:
        messages: list[TextDocument] = []
        for message in mailbox.mbox(item.path):
            if _skip_email(message):
                continue
            text = _strip_reply_chain(_strip_signature(_message_body(message)))
            if not text.strip():
                continue
            timestamp = _message_date(message)
            subject = _header(message, "subject")
            sender = _header(message, "from")
            recipient = _header(message, "to")
            messages.append(
                TextDocument(
                    text=text.strip(),
                    timestamp_utc=timestamp,
                    metadata={
                        "format": "mbox",
                        "message_id": _header(message, "message-id") or _hash_text(subject + sender),
                        "sender": sender,
                        "recipient": recipient,
                        "subject": subject,
                    },
                )
            )
        return ExtractedItem(discovered=item, payload=messages, metadata={"email_count": len(messages)})

    def normalize(self, item: ExtractedItem, context: IngestContext) -> list[NormalizedChunkRecord]:
        messages = item.payload if isinstance(item.payload, list) else []
        records: list[NormalizedChunkRecord] = []
        for index, document in enumerate(messages):
            if not isinstance(document, TextDocument):
                continue
            identity = f"email:{document.metadata['message_id']}:{index}"
            text = document.text
            metadata = {
                **document.metadata,
                "chunk_identity": identity,
                "embedding_text": prepare_embedding_text(text, model_name="intfloat/e5-large-v2"),
                "raw_text": text,
                "exact_terms": _extract_exact_terms(text + " " + document.metadata.get("subject", "")),
            }
            records.append(
                NormalizedChunkRecord(
                    chunk_id=_chunk_id(item.discovered.path, identity),
                    source_type="email",
                    file_path=item.discovered.path,
                    text=text,
                    timestamp_utc=document.timestamp_utc,
                    vector_collection="text_chunks",
                    metadata=metadata,
                )
            )
        return records

    def embed(
        self, records: list[NormalizedChunkRecord], context: IngestContext
    ) -> list[NormalizedChunkRecord]:
        return embed_text_records(records)


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
            continue
        chunks.extend(_split_oversized(current, max_chars, overlap_chars))
        current = paragraph
    if current:
        chunks.extend(_split_oversized(current, max_chars, overlap_chars))
    return chunks


def prepare_embedding_text(text: str, model_name: str) -> str:
    if "e5" in model_name.lower() and not text.startswith("passage:"):
        return f"passage: {text}"
    return text


def prepare_query_text(text: str, model_name: str) -> str:
    if "e5" in model_name.lower() and not text.startswith("query:"):
        return f"query: {text}"
    return text


def _extract_markdown(path: Path, mtime_ns: int) -> TextDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    frontmatter, body = _parse_frontmatter(raw)
    timestamp = _parse_datetime(frontmatter.get("date")) or _datetime_from_mtime_ns(mtime_ns)
    return TextDocument(
        text=_clean_obsidian_text(body),
        timestamp_utc=timestamp,
        metadata={
            "format": "markdown",
            "frontmatter": frontmatter,
            "tags": _as_list(frontmatter.get("tags")),
            "aliases": _as_list(frontmatter.get("aliases")),
        },
    )


def _extract_notion_csv(path: Path, mtime_ns: int) -> TextDocument:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            values = [str(value).strip() for value in row.values() if str(value).strip()]
            if values:
                rows.append("\n".join(values))
    return TextDocument(
        text="\n\n".join(rows),
        timestamp_utc=_datetime_from_mtime_ns(mtime_ns),
        metadata={"format": "notion_csv"},
    )


def _extract_json_journal(path: Path, mtime_ns: int) -> TextDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = _json_entries(payload)
    documents: list[str] = []
    metadata: dict[str, Any] = {"format": "journal_json"}
    timestamp = _datetime_from_mtime_ns(mtime_ns)
    for entry in entries:
        text = str(entry.get("text") or entry.get("content") or entry.get("body") or "").strip()
        if not text:
            continue
        documents.append(text)
        timestamp = _parse_datetime(
            entry.get("creationDate")
            or entry.get("creation_date")
            or entry.get("date")
            or entry.get("date_journal")
        ) or timestamp
        if "location" in entry:
            metadata["location"] = entry["location"]
        for key in ("weather", "tags", "photos"):
            if key in entry:
                metadata[key] = entry[key]
    return TextDocument(text="\n\n".join(documents), timestamp_utc=timestamp, metadata=metadata)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter = _parse_simple_yaml(lines[1:index])
            return frontmatter, "\n".join(lines[index + 1 :])
    return {}, raw


def _parse_simple_yaml(lines: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    current_key: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-") and current_key:
            if not isinstance(parsed.get(current_key), list):
                parsed[current_key] = []
            parsed[current_key].append(stripped[1:].strip())
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current_key = key.strip()
        parsed[current_key] = _parse_yaml_value(value.strip())
    return parsed


def _parse_yaml_value(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        return [part.strip().strip("\"'") for part in value[1:-1].split(",") if part.strip()]
    return value.strip("\"'")


def _clean_obsidian_text(text: str) -> str:
    text = re.sub(r"!\[\[[^\]]+\]\]", "", text)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text.strip()


def _message_body(message: Message) -> str:
    plain = _message_part(message, "text/plain")
    if plain:
        return plain
    html = _message_part(message, "text/html")
    return _html_to_text(html) if html else ""


def _message_part(message: Message, content_type: str) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == content_type:
                return _decode_payload(part)
        return ""
    if message.get_content_type() == content_type:
        return _decode_payload(message)
    return ""


def _decode_payload(message: Message) -> str:
    payload = message.get_payload(decode=True)
    if payload is None:
        raw = message.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = message.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(re.sub(r"[ \t]+", " ", text)).strip()


def _skip_email(message: Message) -> bool:
    header_names = {key.lower() for key in message.keys()}
    if any(header in header_names for header in AUTO_REPLY_HEADERS):
        return True
    if any(header in header_names for header in NEWSLETTER_HINTS):
        return True
    sender = _header(message, "from").lower()
    return any(hint in sender for hint in ("newsletter", "noreply", "no-reply"))


def _strip_reply_chain(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">") or re.match(r"On .+ wrote:", stripped):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _strip_signature(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "--" or line.strip().lower() in {"regards,", "best,"}:
            return "\n".join(lines[:index]).strip()
    return text


def _message_date(message: Message) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(_header(message, "date"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _header(message: Message, name: str) -> str:
    value = message.get(name, "")
    return str(value)


def _split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return [chunk for chunk in chunks if chunk]


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        raw = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(raw, pattern)
                    break
                except ValueError:
                    parsed = None
            if parsed is None:
                return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _datetime_from_mtime_ns(mtime_ns: int) -> datetime:
    return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=UTC)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _json_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("entries", "journal_entries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _extract_exact_terms(text: str) -> list[str]:
    return sorted({match.group(0) for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)})


def _chunk_id(path: Path, identity: str) -> str:
    return _hash_text(f"{path.resolve()}::{identity}")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
