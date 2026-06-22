"""FastAPI application for local query, status, ingest, and file-open endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import get_config
from app.ingest.registry import (
    SUPPORTED_EXTENSIONS,
    SourceKind,
    SourceRegistry,
    build_source_config,
    validate_source,
)
from app.ranking.grouper import SessionGrouper
from app.ranking.reranker import CrossEncoderReranker, TemporalReranker
from app.retrieval.conversation import ConversationManager
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.retriever import Retriever
from app.storage.metadata import MetadataStore

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install API dependencies with `pip install -e .` before running the API.") from exc

if TYPE_CHECKING:
    from app.proactive.titles import CardTitler
    from app.retrieval.answers import AnswerSynthesizer, QueryPlanner
    from app.retrieval.chat_intent import IntentClassifier

# ---------------------------------------------------------------------------
# Application state (single-process, single-user)
# ---------------------------------------------------------------------------

_config = get_config()

# Lazy singletons populated in lifespan
_store: MetadataStore | None = None
_retriever: Retriever | None = None
_vector_store: Any = None  # VectorStore, attached in lifespan when Qdrant is reachable

# Progress for the one-time CLIP image-embedding backfill (POST /index/images).
_image_index_status: dict[str, Any] = {
    "state": "idle",  # idle | running | done | error
    "processed": 0,
    "total": 0,
    "skipped": 0,
    "failed": 0,
    "message": "",
}
_image_index_lock = threading.Lock()
_query_analyzer: QueryAnalyzer | None = None
_temporal_reranker: TemporalReranker | None = None
_cross_encoder: CrossEncoderReranker | None = None
_session_grouper: SessionGrouper | None = None
_intent_classifier: "IntentClassifier | None" = None
_answer_synthesizer: "AnswerSynthesizer | None" = None
_query_planner: "QueryPlanner | None" = None
_card_titler: "CardTitler | None" = None
_llm_client: Any = None
_proactive_scheduler: Any = None

# Voice-search transcriber. Loaded lazily on first /transcribe call and cached,
# so users who never use voice pay no startup cost. Separate from the ingest path.
_query_transcriber: Any = None
_query_transcriber_lock = threading.Lock()
# Reject oversized uploads (~25 MB comfortably exceeds 60s of opus voice).
_MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Conversation memory manager (Section 15)
_conversation_ttl = float(os.getenv("LIFELOG_CONVERSATION_TTL_SECONDS", str(60 * 60 * 24 * 365)))
_conv_manager = ConversationManager(
    ttl_seconds=_conversation_ttl,
    storage_path=_config.paths.data_dir / "conversations.json",
)
_ingest_status_lock = threading.Lock()
_ingest_status: dict[str, Any] = {
    "state": "idle",
    "message": "No ingest running",
    "mode": None,
    "source_id": None,
    "started_at": None,
    "finished_at": None,
    "processed_items": 0,
    "skipped_items": 0,
    "failed_items": 0,
}

# Enrichment run state (Phase 1) and the GPU-yield gate. ``_active_query_count``
# tracks in-flight /query requests so background enrichment pauses while the GPU
# is serving a query.
_enrich_status_lock = threading.Lock()
_enrich_status: dict[str, Any] = {
    "state": "idle",
    "message": "No enrichment running",
    "done": 0,
    "skipped": 0,
    "failed": 0,
    "unavailable": [],
    "paused": False,
    "started_at": None,
    "finished_at": None,
}
_active_query_lock = threading.Lock()
_active_query_count = 0


def _query_in_progress() -> bool:
    with _active_query_lock:
        return _active_query_count > 0


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _store, _retriever, _query_analyzer, _temporal_reranker, _cross_encoder, _session_grouper, _intent_classifier
    global _answer_synthesizer, _query_planner, _card_titler, _llm_client, _proactive_scheduler, _vector_store
    from app.proactive.titles import CardTitler  # noqa: PLC0415
    from app.retrieval.answers import AnswerSynthesizer, QueryPlanner  # noqa: PLC0415
    from app.retrieval.chat_intent import IntentClassifier  # noqa: PLC0415
    from app.retrieval.llm_client import build_ollama_client  # noqa: PLC0415
    from app.storage.vector_store import VectorStore  # noqa: PLC0415
    _config.ensure_directories()
    _config.activate_tool_paths()
    _store = MetadataStore(_config.paths.sqlite_path)
    # Attach Qdrant when reachable so dense + CLIP retrieval (incl. photo search) work.
    vector_store = VectorStore.from_environment()
    if vector_store.available:
        vector_store.ensure_collections()
    else:
        vector_store = None
    _vector_store = vector_store
    _retriever = Retriever(_store, vector_store=vector_store)
    _query_analyzer = QueryAnalyzer(use_spacy=False)
    _temporal_reranker = TemporalReranker.from_environment()
    _cross_encoder = CrossEncoderReranker.from_environment()
    _session_grouper = SessionGrouper.from_environment()
    llm_client = None
    if _config.llm.enabled:
        llm_client = build_ollama_client(
            model=_config.llm.ollama_model,
            host=_config.llm.ollama_url,
        )
    _llm_client = llm_client
    _intent_classifier = IntentClassifier(llm_client=llm_client)
    _answer_synthesizer = AnswerSynthesizer(llm_client, max_cards=_config.llm.answer_max_cards)
    _query_planner = QueryPlanner(llm_client)
    _card_titler = CardTitler(llm_client)
    _proactive_scheduler = _start_proactive_scheduler() if _config.proactive.schedule_enabled else None
    try:
        yield
    finally:
        if _proactive_scheduler is not None:
            _proactive_scheduler.shutdown(wait=False)


app = FastAPI(title="Life Log Search", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _track_query_activity(request, call_next):
    """Count in-flight /query requests so enrichment can yield the GPU."""
    global _active_query_count
    is_query = request.url.path.rstrip("/").endswith("/query") or request.url.path == "/query"
    if is_query:
        with _active_query_lock:
            _active_query_count += 1
    try:
        return await call_next(request)
    finally:
        if is_query:
            with _active_query_lock:
                _active_query_count -= 1


# Endpoints reachable without a token. Everything else requires auth when a
# password is configured. ``/auth/status`` lets the UI decide whether to show
# the login screen; ``/auth/login`` mints the token.
_AUTH_ALLOWLIST = {"/auth/login", "/auth/status"}


@app.middleware("http")
async def _require_auth(request, call_next):
    """Reject unauthenticated requests when a login password is configured.

    The token may arrive as ``Authorization: Bearer <t>`` (fetch calls) or as a
    ``?token=`` query param (``<img>``/``<video>`` media tags can't set headers).
    """
    if not _config.auth.enabled:
        return await call_next(request)
    if request.method == "OPTIONS" or request.url.path.rstrip("/") in _AUTH_ALLOWLIST:
        return await call_next(request)

    from app.api.auth import verify_token  # noqa: PLC0415

    token = None
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
    if not token:
        token = request.query_params.get("token")
    if not token or not verify_token(_config.auth.secret, token):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return await call_next(request)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class QueryFilters(BaseModel):
    source_type: str | None = None
    session_id: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    top_k: int = Field(default=10, ge=1, le=100)
    conversation_id: str | None = None
    chronological: bool = False


class HitOut(BaseModel):
    chunk_id: str
    source_type: str
    file_path: str
    score: float
    rank: int
    rationale: list[str]
    match_reasons: list[str]
    timestamp_utc: str | None
    timestamp_display: str | None
    session_id: str | None
    snippet: str | None
    place_name: str | None
    thumbnail_path: str | None
    preview_url: str | None = None
    preview_type: str | None = None



class SessionCardOut(BaseModel):
    session_id: str
    score: float
    start_utc: str | None
    end_utc: str | None
    modalities: list[str]
    primary: HitOut
    secondary: list[HitOut]
    title: str | None = None
    summary: str | None = None


class QueryResponse(BaseModel):
    sessions: list[SessionCardOut]
    conversation_id: str
    clarification_prompt: str | None = None
    chat_message: str | None = None
    answer: str | None = None
    answer_citations: list[str] = Field(default_factory=list)
    query_debug: dict[str, Any]


class StatusResponse(BaseModel):
    status: str
    environment: str
    api_host: str
    api_port: int
    total_chunks: int
    chunks_by_modality: dict[str, int]
    files_by_modality: dict[str, int]
    last_ingest_timestamp: str | None
    sqlite_path: str
    vector_store_mode: str
    qdrant_url: str


class IngestTriggerRequest(BaseModel):
    full: bool = False
    source_id: str | None = None
    source_type: str | None = None
    path: str | None = None


class IngestTriggerResponse(BaseModel):
    status: str
    message: str
    source_id: str | None = None


class IngestStatusResponse(BaseModel):
    state: str
    message: str
    mode: str | None = None
    source_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    processed_items: int = 0
    skipped_items: int = 0
    failed_items: int = 0


class EnrichTriggerRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)
    retry_failed: bool = False


class EnrichTriggerResponse(BaseModel):
    status: str
    message: str


class EnrichStatusResponse(BaseModel):
    state: str
    message: str
    done: int = 0
    skipped: int = 0
    failed: int = 0
    unavailable: list[str] = Field(default_factory=list)
    paused: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    summary: dict[str, dict[str, int]] = Field(default_factory=dict)


class PersonOut(BaseModel):
    cluster_id: str
    person_name: str | None = None
    face_count: int
    sample_file_path: str | None = None
    sample_bbox: list[float] | None = None
    sample_preview_url: str | None = None


class NamePersonRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class FaceOut(BaseModel):
    face_id: str
    file_path: str
    bbox: list[float]
    det_score: float
    preview_url: str | None = None


class SourcePreviewRequest(BaseModel):
    source_type: str
    path: str
    limit: int = Field(default=24, ge=1, le=100)


class LocalPathSelectRequest(BaseModel):
    source_type: str
    target: str = Field(default="folder", pattern="^(file|folder)$")


class LocalPathSelectResponse(BaseModel):
    path: str | None = None


class PreviewFileOut(BaseModel):
    path: str
    name: str
    extension: str
    preview_url: str | None = None


class SourcePreviewResponse(BaseModel):
    source_id: str
    source_type: str
    path: str
    ok: bool
    item_count: int
    errors: list[str]
    warnings: list[str]
    files: list[PreviewFileOut]


class SourceOut(BaseModel):
    id: str
    source_type: str
    path: str
    enabled: bool
    last_scan_time: str | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None


class OpenFileRequest(BaseModel):
    file_path: str
    timestamp_sec: float | None = None


class EditorLinkResponse(BaseModel):
    uri: str
    scheme: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_match_reasons(rationale: list[str], signals: Any) -> list[str]:
    """Translate retriever-level rationale codes into human-readable match reasons."""
    reasons: list[str] = []
    for r in rationale:
        if r == "bm25":
            reasons.append("keyword match (BM25)")
        elif r == "dense_text_chunks":
            reasons.append("semantic text similarity")
        elif r == "dense_audio_transcripts":
            reasons.append("semantic audio transcript match")
        elif "clip_image" in r:
            reasons.append("visual similarity — image (CLIP)")
        elif "clip_video" in r:
            reasons.append("visual similarity — video frame (CLIP)")
        elif "clip" in r:
            reasons.append("visual similarity (CLIP)")
        elif "dense" in r:
            reasons.append("semantic similarity")
    if getattr(signals, "temporal_range", None) is not None:
        reasons.append("temporal boost applied")
    if getattr(signals, "place_names", None):
        reasons.append(f"location: {', '.join(signals.place_names)}")
    if getattr(signals, "visual_intent", False):
        reasons.append("visual intent detected")
    if getattr(signals, "audio_intent", False):
        reasons.append("audio intent detected")
    if getattr(signals, "email_intent", False):
        reasons.append("email intent detected")
    if getattr(signals, "calendar_intent", False):
        reasons.append("calendar intent detected")
    return reasons or ["relevance score"]


def _hit_to_out(hit: Any, rank: int, signals: Any) -> HitOut:
    td = hit.timestamp_utc.strftime("%a %d %b %Y at %H:%M") if hit.timestamp_utc else None
    
    preview_url = None
    preview_type = None
    
    from urllib.parse import quote
    ext = Path(str(hit.file_path)).suffix.lower()
    source_type_str = str(hit.source_type)
    
    if source_type_str == "photo" or ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        preview_type = "image"
        preview_url = f"/api/file-preview?path={quote(str(hit.file_path))}"
    elif source_type_str == "video" or ext in {".mp4", ".webm", ".ogg"}:
        preview_type = "video"
        preview_url = f"/api/file-preview?path={quote(str(hit.file_path))}"
    elif source_type_str == "audio" or ext in {".mp3", ".wav", ".m4a"}:
        preview_type = "audio"
        preview_url = f"/api/file-preview?path={quote(str(hit.file_path))}"

    thumb = preview_url if preview_type == "image" else (str(hit.thumbnail_path) if getattr(hit, "thumbnail_path", None) else None)

    return HitOut(
        chunk_id=hit.chunk_id,
        source_type=str(hit.source_type),
        file_path=str(hit.file_path),
        score=hit.score,
        rank=rank,
        rationale=hit.rationale,
        match_reasons=_compute_match_reasons(hit.rationale, signals),
        timestamp_utc=hit.timestamp_utc.isoformat() if hit.timestamp_utc else None,
        timestamp_display=td,
        session_id=hit.session_id,
        snippet=hit.snippet,
        place_name=hit.place_name,
        thumbnail_path=thumb,
        preview_url=preview_url,
        preview_type=preview_type,
    )


def _card_to_out(card: Any, signals: Any, *, title: str | None = None) -> SessionCardOut:
    hits = [_hit_to_out(h, rank=i + 1, signals=signals) for i, h in enumerate(card.hits)]
    return SessionCardOut(
        session_id=card.session_id,
        score=card.score,
        start_utc=card.start_utc.isoformat() if card.start_utc else None,
        end_utc=card.end_utc.isoformat() if card.end_utc else None,
        modalities=getattr(card, "modalities", []),
        primary=hits[0],
        secondary=hits[1:],
        title=title or getattr(card, "title", None),
        summary=getattr(card, "summary", None),
    )


def _run_ingest_bg(full: bool, source_id: str | None) -> None:
    """Runs ingest in a background thread. Errors are logged but not raised."""
    started_at = datetime.now().isoformat()
    with _ingest_status_lock:
        _ingest_status.update(
            {
                "state": "running",
                "message": "Ingest running",
                "mode": "full" if full else "incremental",
                "source_id": source_id,
                "started_at": started_at,
                "finished_at": None,
                "processed_items": 0,
                "skipped_items": 0,
                "failed_items": 0,
            }
        )
    try:
        registry = SourceRegistry(_config.paths.source_registry_path)
        store = MetadataStore(_config.paths.sqlite_path)
        from app.ingest.runner import IngestRunner  # noqa: PLC0415
        runner = IngestRunner(registry, store)
        summary = runner.run(full=full, source_id=source_id)
        with _ingest_status_lock:
            _ingest_status.update(
                {
                    "state": "done" if summary.failed_items == 0 else "done_with_errors",
                    "message": (
                        f"Done: {summary.processed_items} processed, "
                        f"{summary.skipped_items} skipped, {summary.failed_items} failed"
                    ),
                    "finished_at": summary.finished_at.isoformat(),
                    "processed_items": summary.processed_items,
                    "skipped_items": summary.skipped_items,
                    "failed_items": summary.failed_items,
                }
            )
        # Enrich newly ingested items immediately (and backfill) when enabled.
        if _config.enrichment.enabled:
            try:
                _run_enrich_bg(limit=None, retry_failed=False)
            except Exception as exc:  # noqa: BLE001
                import logging  # noqa: PLC0415
                logging.getLogger(__name__).warning("Post-ingest enrichment failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).error("Background ingest failed: %s", exc)
        with _ingest_status_lock:
            _ingest_status.update(
                {
                    "state": "error",
                    "message": str(exc),
                    "finished_at": datetime.now().isoformat(),
                }
            )


def _get_query_transcriber():
    """Lazily load and cache the voice-query transcription engine (thread-safe)."""
    global _query_transcriber
    if _query_transcriber is None:
        with _query_transcriber_lock:
            if _query_transcriber is None:
                from app.ingest.audio import TranscriptionEngine  # noqa: PLC0415
                _query_transcriber = TranscriptionEngine.load()
    return _query_transcriber


def _run_enrich_bg(limit: int | None, retry_failed: bool) -> None:
    """Run AI enrichment in a background thread, yielding the GPU during queries."""
    from app.enrich.registry import build_enrichers  # noqa: PLC0415
    from app.enrich.runner import EnrichmentRunner  # noqa: PLC0415
    from app.ingest.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

    with _enrich_status_lock:
        _enrich_status.update(
            {
                "state": "running",
                "message": "Enrichment running",
                "done": 0,
                "skipped": 0,
                "failed": 0,
                "unavailable": [],
                "paused": False,
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
            }
        )
    try:
        store = MetadataStore(_config.paths.sqlite_path)
        enrichers = build_enrichers(_config)
        embedder = SentenceTransformerEmbedder.from_environment()
        runner = EnrichmentRunner(
            store,
            enrichers,
            embedder=embedder,
            batch_size=_config.enrichment.batch_size,
            should_pause=_query_in_progress,
        )
        summary = runner.run(limit=limit, include_failed=retry_failed)

        # Cluster newly detected faces (Phase 2).
        if any(e.name == "faces" for e in enrichers) and "faces" not in summary.unavailable:
            from app.enrich.clustering import FaceClusterer  # noqa: PLC0415

            FaceClusterer(store, threshold=_config.enrichment.face_cluster_threshold).cluster_new()

        with _enrich_status_lock:
            _enrich_status.update(
                {
                    "state": "paused" if summary.paused else "done",
                    "message": (
                        f"Done: {summary.done} enriched, {summary.skipped} skipped, "
                        f"{summary.failed} failed"
                        + (" (paused for queries)" if summary.paused else "")
                    ),
                    "done": summary.done,
                    "skipped": summary.skipped,
                    "failed": summary.failed,
                    "unavailable": summary.unavailable,
                    "paused": summary.paused,
                    "finished_at": datetime.now().isoformat(),
                }
            )
    except Exception as exc:  # noqa: BLE001
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).error("Background enrichment failed: %s", exc)
        with _enrich_status_lock:
            _enrich_status.update(
                {"state": "error", "message": str(exc), "finished_at": datetime.now().isoformat()}
            )


def _source_from_request(source_type: str, path: str):
    try:
        kind = SourceKind(source_type)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SourceKind)
        raise HTTPException(status_code=400, detail=f"Unsupported source type. Use one of: {allowed}") from exc
    return build_source_config(kind, Path(path))


def _preview_files(source_type: SourceKind, path: Path, limit: int) -> list[PreviewFileOut]:
    source = build_source_config(source_type, path)
    candidates = [source.path] if source.path.is_file() else sorted(source.path.rglob("*"))
    files: list[PreviewFileOut] = []
    for item in candidates:
        if not item.is_file() or item.suffix.lower() not in source.supported_extensions:
            continue
        preview_url = None
        if source_type == SourceKind.PHOTOS:
            from urllib.parse import quote  # noqa: PLC0415
            preview_url = f"/api/file-preview?path={quote(str(item))}"
        files.append(
            PreviewFileOut(
                path=str(item),
                name=item.name,
                extension=item.suffix.lower(),
                preview_url=preview_url,
            )
        )
        if len(files) >= limit:
            break
    return files


def _select_local_path(source_type: SourceKind, target: str) -> str | None:
    """Open a native local file/folder picker and return the selected path."""
    try:
        import tkinter as tk  # noqa: PLC0415
        from tkinter import filedialog  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Local file picker unavailable: {exc}") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if target == "folder":
            selected = filedialog.askdirectory(
                parent=root,
                title=f"Select {source_type.value.replace('_', ' ')} folder",
                mustexist=True,
            )
        else:
            source = build_source_config(source_type, Path.cwd())
            patterns = " ".join(f"*{ext}" for ext in source.supported_extensions if ext)
            filetypes = [(f"{source_type.value.replace('_', ' ').title()} files", patterns)]
            filetypes.append(("All files", "*.*"))
            selected = filedialog.askopenfilename(
                parent=root,
                title=f"Select {source_type.value.replace('_', ' ')} file",
                filetypes=filetypes,
            )
    finally:
        root.destroy()

    return str(Path(selected).expanduser().resolve()) if selected else None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=500)


class LoginResponse(BaseModel):
    token: str


@app.get("/auth/status")
def auth_status() -> dict[str, bool]:
    """Tell the UI whether a login is required (no token needed to ask)."""
    return {"auth_required": _config.auth.enabled}


@app.post("/auth/login", response_model=LoginResponse)
def auth_login(request: LoginRequest) -> LoginResponse:
    import hmac as _hmac  # noqa: PLC0415

    from app.api.auth import create_token  # noqa: PLC0415

    if not _config.auth.enabled:
        raise HTTPException(status_code=400, detail="Login is not configured on the server.")
    if not _hmac.compare_digest(request.password, _config.auth.password or ""):
        raise HTTPException(status_code=401, detail="Incorrect password")
    return LoginResponse(token=create_token(_config.auth.secret, _config.auth.ttl_seconds))


@app.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    assert _store is not None
    counts = _store.chunk_counts_by_source_type()
    file_counts = _store.file_counts_by_source_type()
    latest = _store.latest_ingest_timestamp()
    return StatusResponse(
        status="ok",
        environment=_config.env,
        api_host=_config.api_host,
        api_port=_config.api_port,
        total_chunks=sum(counts.values()),
        chunks_by_modality=counts,
        files_by_modality=file_counts,
        last_ingest_timestamp=latest.isoformat() if latest else None,
        sqlite_path=str(_config.paths.sqlite_path),
        vector_store_mode=_config.vector_store.mode,
        qdrant_url=_config.vector_store.url,
    )


@app.get("/sources", response_model=list[SourceOut])
def list_sources() -> list[SourceOut]:
    registry = SourceRegistry(_config.paths.source_registry_path)
    return [
        SourceOut(
            id=source.id,
            source_type=source.source_type.value,
            path=str(source.path),
            enabled=source.enabled,
            last_scan_time=source.last_scan_time.isoformat() if source.last_scan_time else None,
        )
        for source in registry.sources
    ]


@app.post("/sources/preview", response_model=SourcePreviewResponse)
def preview_source(request: SourcePreviewRequest) -> SourcePreviewResponse:
    source = _source_from_request(request.source_type, request.path)
    validation = validate_source(source)
    files = (
        _preview_files(source.source_type, source.path, request.limit)
        if source.path.exists() and validation.ok
        else []
    )
    return SourcePreviewResponse(
        source_id=source.id,
        source_type=source.source_type.value,
        path=str(source.path),
        ok=validation.ok,
        item_count=validation.item_count,
        errors=list(validation.errors),
        warnings=list(validation.warnings),
        files=files,
    )


@app.post("/local-path/select", response_model=LocalPathSelectResponse)
def select_local_path(request: LocalPathSelectRequest) -> LocalPathSelectResponse:
    try:
        source_type = SourceKind(request.source_type)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SourceKind)
        raise HTTPException(status_code=400, detail=f"Unsupported source type. Use one of: {allowed}") from exc

    return LocalPathSelectResponse(path=_select_local_path(source_type, request.target))


_COUNT_RE = re.compile(r"\b(how many|how much|number of|count of|total number)\b", re.IGNORECASE)
_COUNT_MODALITY_WORDS: dict[str, tuple[str, ...]] = {
    "photo": ("photo", "photos", "picture", "pictures", "image", "images", "screenshot", "screenshots"),
    "video": ("video", "videos", "clip", "clips", "recording", "recordings"),
    "text": ("document", "documents", "note", "notes", "doc", "docs", "markdown", "file", "files"),
    "audio": ("audio", "voice", "podcast", "song", "songs"),
}
_COUNT_LABELS = {"photo": "photos", "video": "videos", "text": "documents", "audio": "audio files"}


def _maybe_count_answer(query: str, store: MetadataStore) -> str | None:
    """Answer a 'how many X' question from DB item counts instead of retrieving.

    Retrieval can't aggregate, so meta questions used to return noisy CLIP hits.
    Counts are by distinct file, so "how many videos" returns 3, not 30 chunks.
    """
    if not _COUNT_RE.search(query):
        return None
    counts = store.file_counts_by_source_type()
    ql = query.lower()
    for canonical, words in _COUNT_MODALITY_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", ql) for word in words):
            return f"You have {counts.get(canonical, 0):,} {_COUNT_LABELS[canonical]} indexed."
    parts = [f"{counts.get(k, 0):,} {lbl}" for k, lbl in _COUNT_LABELS.items() if counts.get(k, 0)]
    if not parts:
        return None
    return f"You have {sum(counts.values()):,} items indexed — " + ", ".join(parts) + "."


@app.post("/query", response_model=QueryResponse)
def post_query(request: QueryRequest) -> QueryResponse:
    assert _store is not None
    assert _retriever is not None
    assert _query_analyzer is not None
    assert _temporal_reranker is not None
    assert _cross_encoder is not None
    assert _session_grouper is not None

    # Resolve forward references from prior conversation turns (Section 15)
    conv_id = request.conversation_id or _conv_manager.new_id()
    ctx = _conv_manager.resolve_context(request.query, conv_id)

    # If reference is ambiguous, short-circuit with a clarification prompt
    if ctx.clarification_needed:
        return QueryResponse(
            sessions=[],
            conversation_id=conv_id,
            clarification_prompt="Which session did you mean?\n" + "\n".join(
                f"  {i + 1}. {o}" for i, o in enumerate(ctx.clarification_options)
            ),
            query_debug={"clarification_needed": True, "options": ctx.clarification_options},
        )

    # Aggregate questions ("how many photos?") are answered from counts, not retrieval.
    count_message = _maybe_count_answer(ctx.effective_query, _store)
    if count_message is not None:
        _conv_manager.store_turn(
            conv_id=conv_id,
            query=request.query,
            temporal_range=None,
            session_ids=[],
            place_names=[],
            result_count=0,
        )
        return QueryResponse(
            sessions=[],
            conversation_id=conv_id,
            chat_message=count_message,
            query_debug={"intent": "count"},
        )

    has_filters = bool(
        request.filters.source_type
        or request.filters.session_id
        or request.filters.date_from
        or request.filters.date_to
    )
    from app.retrieval.chat_intent import QueryIntent  # noqa: PLC0415

    assert _intent_classifier is not None
    intent = _intent_classifier.classify(request.query, has_filters=has_filters)
    if intent == QueryIntent.chit_chat:
        reply = _intent_classifier.chit_chat_reply(request.query)
        _conv_manager.store_turn(
            conv_id=conv_id,
            query=request.query,
            temporal_range=None,
            session_ids=[],
            place_names=[],
            result_count=0,
        )
        return QueryResponse(
            sessions=[],
            conversation_id=conv_id,
            chat_message=reply,
            query_debug={"intent": "chit_chat"},
        )

    # Analyze query
    signals = _query_analyzer.analyze(ctx.effective_query)

    # Apply temporal override from prior context if no fresh temporal signal
    if ctx.temporal_range_override is not None and signals.temporal_range is None:
        from dataclasses import replace  # noqa: PLC0415
        signals = replace(signals, temporal_range=ctx.temporal_range_override)

    # Build filters dict for retriever
    filters: dict[str, Any] = {}
    if request.filters.source_type:
        filters["source_type"] = request.filters.source_type
    # Prefer explicit session_id filter from request; fall back to ctx resolution
    effective_session = request.filters.session_id or ctx.session_id_filter
    if effective_session:
        filters["session_id"] = effective_session

    # Retrieve. With query decomposition (Phase 3) we run each sub-query and merge,
    # keeping the best score per chunk. Falls back to a single query when the planner
    # is unavailable or the question is simple.
    subqueries = [ctx.effective_query]
    if _query_planner is not None and _query_planner.available and _config.llm.decomposition_enabled:
        subqueries = _query_planner.decompose(ctx.effective_query)

    retrieve_limit = max(request.top_k * 4, 50)
    if len(subqueries) == 1:
        hits = _retriever.retrieve(
            subqueries[0], signals=signals, limit=retrieve_limit, filters=filters or None
        )
    else:
        merged: dict[str, Any] = {}
        for subquery in subqueries:
            for hit in _retriever.retrieve(
                subquery, signals=signals, limit=retrieve_limit, filters=filters or None
            ):
                existing = merged.get(hit.chunk_id)
                if existing is None or hit.score > existing.score:
                    merged[hit.chunk_id] = hit
        hits = list(merged.values())

    # Apply date range post-filter
    if request.filters.date_from or request.filters.date_to:
        date_from = datetime.fromisoformat(request.filters.date_from) if request.filters.date_from else None
        date_to = datetime.fromisoformat(request.filters.date_to) if request.filters.date_to else None
        hits = [
            h for h in hits
            if h.timestamp_utc is not None
            and (date_from is None or h.timestamp_utc >= date_from)
            and (date_to is None or h.timestamp_utc <= date_to)
        ]

    # Temporal handling: an explicit range from the query ("in 2019", "last
    # summer") is a hard filter so out-of-range memories don't crowd out the
    # right ones. Fall back to keeping all hits when nothing lands in range (the
    # regex parse may be spurious, or there's simply no data there), then apply
    # the proximity boost so the closest-in-time results rank first.
    if signals.temporal_range is not None:
        start, end = signals.temporal_range
        in_range = [
            h for h in hits
            if h.timestamp_utc is not None and start <= h.timestamp_utc <= end
        ]
        if in_range:
            hits = in_range
        target_dt = start + (end - start) / 2
        hits = _temporal_reranker.rerank(hits, target_dt)

    # Cross-encoder reranking
    hits = _cross_encoder.rerank(hits, ctx.effective_query)

    # Group into session cards
    if request.chronological:
        cards = _session_grouper.group_chronological(hits)
    else:
        top_n_cards = _session_grouper._top_n  # noqa: SLF001
        _session_grouper._top_n = request.top_k  # type: ignore[misc]
        cards = _session_grouper.group(hits)
        _session_grouper._top_n = top_n_cards  # type: ignore[misc]

    # Synthesize a grounded, cited answer from the top cards (Phase 3).
    answer_text: str | None = None
    answer_citations: list[str] = []
    if (
        _answer_synthesizer is not None
        and _answer_synthesizer.available
        and _config.llm.answers_enabled
        and cards
    ):
        result = _answer_synthesizer.synthesize(ctx.effective_query, cards)
        if result is not None:
            answer_text = result.text
            answer_citations = result.cited_session_ids

    # Auto card titles (Phase 4), opt-in to protect latency.
    titles_map: dict[str, str] = {}
    if (
        _card_titler is not None
        and _card_titler.available
        and _config.proactive.titles_enabled
        and cards
    ):
        titles_map = _card_titler.title_cards(cards)

    # Store turn in conversation manager (Section 15)
    _conv_manager.store_turn(
        conv_id=conv_id,
        query=request.query,
        temporal_range=signals.temporal_range,
        session_ids=[c.session_id for c in cards],
        place_names=signals.place_names,
        result_count=sum(len(c.hits) for c in cards),
    )

    query_debug: dict[str, Any] = {
        "intent": intent.value,
        "subqueries": subqueries if len(subqueries) > 1 else None,
        "temporal_range": [t.isoformat() for t in signals.temporal_range] if signals.temporal_range else None,
        "place_names": signals.place_names,
        "person_names": signals.person_names,
        "visual_intent": signals.visual_intent,
        "visual_keyword_count": signals.visual_keyword_count,
        "audio_intent": signals.audio_intent,
        "text_intent": signals.text_intent,
        "email_intent": signals.email_intent,
        "calendar_intent": signals.calendar_intent,
        "video_intent": signals.video_intent,
        "modality_intents": sorted(signals.modality_intents),
        "total_hits_before_grouping": len(hits),
        "resolved_from": ctx.resolved_from,
    }

    return QueryResponse(
        sessions=[_card_to_out(c, signals, title=titles_map.get(c.session_id)) for c in cards],
        conversation_id=conv_id,
        answer=answer_text,
        answer_citations=answer_citations,
        query_debug=query_debug,
    )


# ---------------------------------------------------------------------------
# Photo search: match an uploaded image (with optional text) against the index
# ---------------------------------------------------------------------------

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _run_image_embedding_backfill() -> None:
    """Compute CLIP image embeddings for photo/video chunks and upsert to Qdrant.

    Idempotent: chunks already present in the vector store are skipped, so the
    job can be safely re-run after new photos are ingested.
    """
    from app.models.contracts import NormalizedChunkRecord  # noqa: PLC0415
    from app.storage.vector_store import chunk_id_to_point_id  # noqa: PLC0415

    if _store is None or _retriever is None or _vector_store is None:
        with _image_index_lock:
            _image_index_status.update(state="error", message="vector store unavailable")
        return

    rows = [r for r in _store.fetch_chunks() if str(r["source_type"]) in ("photo", "video")]
    existing = _vector_store.fetch_all_point_ids("image_frames") | _vector_store.fetch_all_point_ids("video_frames")

    with _image_index_lock:
        _image_index_status.update(
            state="running", processed=0, total=len(rows), skipped=0, failed=0, message="embedding images"
        )

    batch: list[Any] = []
    processed = skipped = failed = 0

    def _flush() -> None:
        if not batch:
            return
        upserted = _vector_store.upsert_records(batch)
        if upserted:
            _store.update_vector_ids(upserted)
        batch.clear()

    for row in rows:
        chunk_id = str(row["chunk_id"])
        if chunk_id_to_point_id(chunk_id) in existing:
            skipped += 1
            with _image_index_lock:
                _image_index_status.update(processed=processed, skipped=skipped, failed=failed)
            continue

        source_type = str(row["source_type"])
        meta = json.loads(row["metadata_json"] or "{}")
        # Photos embed the file itself; videos embed their extracted keyframe.
        image_path = meta.get("frame_path") if source_type == "video" else row["file_path"]
        collection = "video_frames" if source_type == "video" else "image_frames"

        vec = _retriever.embed_image(str(image_path)) if image_path else None
        if vec is None:
            failed += 1
        else:
            ts = None
            if row["timestamp_utc"]:
                try:
                    ts = datetime.fromisoformat(str(row["timestamp_utc"]))
                except ValueError:
                    ts = None
            batch.append(
                NormalizedChunkRecord(
                    chunk_id=chunk_id,
                    source_type=source_type,  # type: ignore[arg-type]
                    file_path=Path(str(row["file_path"])),
                    text=row["text"],
                    timestamp_utc=ts,
                    vector_collection=collection,
                    session_id=row["session_id"],
                    lat=row["lat"],
                    lon=row["lon"],
                    place_name=row["place_name"],
                    metadata={"image_embedding": vec},
                )
            )
            processed += 1
            if len(batch) >= 32:
                _flush()
        with _image_index_lock:
            _image_index_status.update(processed=processed, skipped=skipped, failed=failed)

    _flush()
    with _image_index_lock:
        _image_index_status.update(
            state="done", message=f"embedded {processed}, skipped {skipped}, failed {failed}"
        )


@app.post("/index/images")
def index_images(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Kick off the one-time CLIP image-embedding backfill in the background."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Qdrant is not reachable; cannot build the photo index.")
    with _image_index_lock:
        if _image_index_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Image indexing already running")
        _image_index_status.update(state="running", processed=0, total=0, skipped=0, failed=0, message="starting")
    background_tasks.add_task(_run_image_embedding_backfill)
    return {"status": "started"}


@app.get("/index/images/status")
def index_images_status() -> dict[str, Any]:
    with _image_index_lock:
        return dict(_image_index_status)


@app.post("/query/image", response_model=QueryResponse)
def post_query_image(
    image: UploadFile = File(...),
    query: str | None = Form(None),
    top_k: int = Form(5),
    conversation_id: str | None = Form(None),
) -> QueryResponse:
    """Match an uploaded photo (with optional accompanying text) against the index."""
    assert _retriever is not None
    assert _session_grouper is not None
    assert _cross_encoder is not None

    conv_id = conversation_id or _conv_manager.new_id()
    text = (query or "").strip() or None

    ext = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    if ext not in _IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported image type")
    raw = image.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"query{ext}"
        tmp_path.write_bytes(raw)
        hits = _retriever.retrieve_by_image(str(tmp_path), text=text, limit=max(top_k * 4, 50))

    # The accompanying text steers ranking; rerank by it when present.
    if text and hits:
        hits = _cross_encoder.rerank(hits, text)

    top_n_cards = _session_grouper._top_n  # noqa: SLF001
    _session_grouper._top_n = top_k  # type: ignore[misc]
    cards = _session_grouper.group(hits)
    _session_grouper._top_n = top_n_cards  # type: ignore[misc]

    # Synthesize a grounded, cited answer when the user paired the photo with a
    # text question — mirrors the text /query path. Pure image-to-image search has
    # no question to answer, so it stays cards-only.
    answer_text: str | None = None
    answer_citations: list[str] = []
    if (
        text
        and _answer_synthesizer is not None
        and _answer_synthesizer.available
        and _config.llm.answers_enabled
        and cards
    ):
        result = _answer_synthesizer.synthesize(text, cards)
        if result is not None:
            answer_text = result.text
            answer_citations = result.cited_session_ids

    _conv_manager.store_turn(
        conv_id=conv_id,
        query=text or "[photo search]",
        temporal_range=None,
        session_ids=[c.session_id for c in cards],
        place_names=[],
        result_count=sum(len(c.hits) for c in cards),
    )

    return QueryResponse(
        sessions=[_card_to_out(c, None) for c in cards],
        conversation_id=conv_id,
        answer=answer_text,
        answer_citations=answer_citations,
        query_debug={"intent": "image_search", "total_hits_before_grouping": len(hits)},
    )


@app.post("/ingest/trigger", response_model=IngestTriggerResponse)
def trigger_ingest(request: IngestTriggerRequest, background_tasks: BackgroundTasks) -> IngestTriggerResponse:
    source_id = request.source_id
    if request.source_type and request.path:
        source = _source_from_request(request.source_type, request.path)
        validation = validate_source(source)
        if not validation.ok:
            raise HTTPException(status_code=400, detail="; ".join(validation.errors))
        registry = SourceRegistry(_config.paths.source_registry_path)
        registry.upsert(source)
        registry.save()
        source_id = source.id

    with _ingest_status_lock:
        if _ingest_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="An ingest is already running")

    background_tasks.add_task(_run_ingest_bg, request.full, source_id)
    mode = "full" if request.full else "incremental"
    return IngestTriggerResponse(status="started", message=f"{mode} ingest triggered", source_id=source_id)


@app.get("/ingest/status", response_model=IngestStatusResponse)
def ingest_status() -> IngestStatusResponse:
    with _ingest_status_lock:
        return IngestStatusResponse(**_ingest_status)


_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB per file

# Reverse map: file extension → the SourceKind that ingests it. Lets uploads
# (and the share sheet) auto-route each file by its extension. "" (browser
# history's extensionless DB) is excluded — it can't be inferred from a name.
_EXT_TO_KIND: dict[str, SourceKind] = {
    ext: kind
    for kind, exts in SUPPORTED_EXTENSIONS.items()
    for ext in exts
    if ext
}


def _unique_path(path: Path) -> Path:
    """Return a non-colliding path, suffixing _1, _2, … so uploads never overwrite."""
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _run_ingest_sources_bg(source_ids: list[str]) -> None:
    """Incrementally ingest each given source in turn (used after an upload)."""
    for source_id in source_ids:
        _run_ingest_bg(full=False, source_id=source_id)


@app.post("/ingest/upload")
async def ingest_upload(
    background_tasks: BackgroundTasks,
    source_type: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Ingest files uploaded from a device (e.g. a phone).

    ``source_type="auto"`` routes each file to the right ingestor by its
    extension (so the OS share sheet can dump mixed files); otherwise every file
    is treated as the given type. Files land under ``data/uploads/<kind>/``,
    each dir is registered as a source, then ingested in the background — the
    remote counterpart to the PC's folder-path ingest.
    """
    auto = source_type == "auto"
    forced_kind: SourceKind | None = None
    if not auto:
        try:
            forced_kind = SourceKind(source_type)
        except ValueError as exc:
            allowed = "auto, " + ", ".join(item.value for item in SourceKind)
            raise HTTPException(
                status_code=400, detail=f"Unsupported source type. Use one of: {allowed}"
            ) from exc

    saved_by_kind: dict[SourceKind, int] = {}
    skipped: list[str] = []
    for upload in files:
        name = Path(upload.filename or "").name
        if not name or name.startswith("."):
            continue
        ext = Path(name).suffix.lower()
        kind = _EXT_TO_KIND.get(ext) if auto else forced_kind
        if kind is None or ext not in SUPPORTED_EXTENSIONS[kind]:
            skipped.append(name)
            continue
        data = await upload.read()
        if not data or len(data) > _MAX_UPLOAD_BYTES:
            skipped.append(name)
            continue
        dest_dir = _config.paths.data_dir / "uploads" / kind.value
        dest_dir.mkdir(parents=True, exist_ok=True)
        _unique_path(dest_dir / name).write_bytes(data)
        saved_by_kind[kind] = saved_by_kind.get(kind, 0) + 1

    if not saved_by_kind:
        hint = "recognized file types" if auto else f"{source_type} files"
        raise HTTPException(status_code=400, detail=f"No {hint} in upload.")

    registry = SourceRegistry(_config.paths.source_registry_path)
    source_ids: list[str] = []
    for kind in saved_by_kind:
        source = build_source_config(kind, _config.paths.data_dir / "uploads" / kind.value)
        registry.upsert(source)
        source_ids.append(source.id)
    registry.save()
    background_tasks.add_task(_run_ingest_sources_bg, source_ids)
    return {
        "status": "started",
        "saved": sum(saved_by_kind.values()),
        "skipped": skipped,
        "by_type": {k.value: v for k, v in saved_by_kind.items()},
    }


@app.post("/enrich/trigger", response_model=EnrichTriggerResponse)
def trigger_enrich(request: EnrichTriggerRequest, background_tasks: BackgroundTasks) -> EnrichTriggerResponse:
    from app.enrich.registry import build_enrichers  # noqa: PLC0415

    if not build_enrichers(_config):
        raise HTTPException(
            status_code=400,
            detail="No enrichers enabled. Enable at least one, e.g. LIFELOG_ENRICH_OCR=1.",
        )
    with _enrich_status_lock:
        if _enrich_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="An enrichment run is already in progress")

    background_tasks.add_task(_run_enrich_bg, request.limit, request.retry_failed)
    return EnrichTriggerResponse(status="started", message="enrichment triggered")


@app.get("/enrich/status", response_model=EnrichStatusResponse)
def enrich_status() -> EnrichStatusResponse:
    with _enrich_status_lock:
        data = dict(_enrich_status)
    data["summary"] = _store.enrichment_summary() if _store is not None else {}
    return EnrichStatusResponse(**data)


@app.get("/proactive/on-this-day", response_model=list[SessionCardOut])
def proactive_on_this_day() -> list[SessionCardOut]:
    assert _store is not None
    if not _config.proactive.on_this_day_enabled:
        return []
    from app.proactive.on_this_day import OnThisDay  # noqa: PLC0415

    cards = OnThisDay(_store, _session_grouper).for_date()
    return [_card_to_out(card, None) for card in cards]


@app.get("/proactive/digest")
def proactive_digest(period: str = "day", refresh: bool = False) -> dict[str, Any]:
    assert _store is not None
    if not _config.proactive.digests_enabled:
        raise HTTPException(status_code=404, detail="Digests are disabled")
    if period not in ("day", "week"):
        raise HTTPException(status_code=400, detail="period must be 'day' or 'week'")
    from app.proactive.digests import DigestGenerator  # noqa: PLC0415

    return DigestGenerator(_store, _llm_client).generate(period=period, use_cache=not refresh)


@app.get("/proactive/insights")
def proactive_insights(refresh: bool = False) -> dict[str, Any]:
    assert _store is not None
    if not _config.proactive.insights_enabled:
        raise HTTPException(status_code=404, detail="Insights are disabled")
    from app.proactive.insights import InsightGenerator  # noqa: PLC0415

    return InsightGenerator(_store, _llm_client).generate(use_cache=not refresh)


def _start_proactive_scheduler():
    """Start a background daily refresh of digests/insights. Best-effort."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # noqa: PLC0415

        def _refresh() -> None:
            from app.proactive.digests import DigestGenerator  # noqa: PLC0415
            from app.proactive.insights import InsightGenerator  # noqa: PLC0415

            if _store is None:
                return
            DigestGenerator(_store, _llm_client).generate(period="day", use_cache=False)
            DigestGenerator(_store, _llm_client).generate(period="week", use_cache=False)
            InsightGenerator(_store, _llm_client).generate(use_cache=False)

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(_refresh, "interval", hours=24, id="proactive_refresh")
        scheduler.start()
        return scheduler
    except Exception as exc:  # noqa: BLE001
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).warning("Proactive scheduler not started: %s", exc)
        return None


def _face_preview_url(file_path: str) -> str:
    from urllib.parse import quote  # noqa: PLC0415

    return f"/api/file-preview?path={quote(file_path)}"


@app.get("/people", response_model=list[PersonOut])
def list_people() -> list[PersonOut]:
    assert _store is not None
    people: list[PersonOut] = []
    for cluster in _store.get_clusters():
        cid = str(cluster["cluster_id"])
        faces = _store.faces_for_cluster(cid)
        sample = faces[0] if faces else None
        sample_path = str(sample["file_path"]) if sample else None
        people.append(
            PersonOut(
                cluster_id=cid,
                person_name=cluster["person_name"],
                face_count=int(cluster["face_count"]),
                sample_file_path=sample_path,
                sample_bbox=json.loads(sample["bbox_json"]) if sample and sample["bbox_json"] else None,
                sample_preview_url=_face_preview_url(sample_path) if sample_path else None,
            )
        )
    return people


@app.get("/people/{cluster_id}/faces", response_model=list[FaceOut])
def list_cluster_faces(cluster_id: str) -> list[FaceOut]:
    assert _store is not None
    if _store.get_cluster(cluster_id) is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    out: list[FaceOut] = []
    for face in _store.faces_for_cluster(cluster_id):
        path = str(face["file_path"])
        out.append(
            FaceOut(
                face_id=str(face["face_id"]),
                file_path=path,
                bbox=json.loads(face["bbox_json"]) if face["bbox_json"] else [],
                det_score=float(face["det_score"] or 0.0),
                preview_url=_face_preview_url(path),
            )
        )
    return out


@app.post("/people/{cluster_id}/name")
def name_person(cluster_id: str, request: NamePersonRequest) -> dict[str, Any]:
    assert _store is not None
    if _store.get_cluster(cluster_id) is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    from app.enrich.clustering import name_cluster  # noqa: PLC0415

    updated = name_cluster(_store, cluster_id, request.name.strip())
    return {"status": "named", "cluster_id": cluster_id, "updated_chunks": updated}


# Inline image previews in chat can pull in dozens of multi-MB originals at once,
# which is slow and memory-heavy. The chat uses ?thumb=1 to get a downscaled JPEG
# instead, cached on disk and keyed by (path, mtime) so source edits invalidate it.
_THUMBNAIL_DIR = _config.paths.data_dir / "thumbnails"
_THUMBNAIL_MAX_EDGE = 512
_THUMBNAIL_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _build_thumbnail(file_path: Path) -> Path:
    """Return a cached <=512px JPEG for an image, generating it on first use."""
    from PIL import Image, ImageOps  # noqa: PLC0415

    mtime = int(file_path.stat().st_mtime)
    key = hashlib.sha1(f"{file_path}:{mtime}".encode()).hexdigest()
    thumb_path = _THUMBNAIL_DIR / f"{key}.jpg"
    if thumb_path.exists():
        return thumb_path

    _THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((_THUMBNAIL_MAX_EDGE, _THUMBNAIL_MAX_EDGE))
        tmp_path = thumb_path.with_suffix(".jpg.tmp")
        image.save(tmp_path, format="JPEG", quality=80)
    os.replace(tmp_path, thumb_path)  # atomic so concurrent reads never see a partial file
    return thumb_path


@app.get("/file-preview")
def file_preview(path: str, thumb: bool = False):
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    supported_exts = {
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
        ".mp4", ".webm", ".ogg",
        ".mp3", ".wav", ".m4a"
    }
    suffix = file_path.suffix.lower()
    if suffix not in supported_exts:
        raise HTTPException(status_code=400, detail="Preview is only available for common image, video, and audio files")
    if thumb and suffix in _THUMBNAIL_EXTS:
        try:
            return FileResponse(_build_thumbnail(file_path), media_type="image/jpeg")
        except Exception:  # noqa: BLE001 - fall back to the original if thumbnailing fails
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning("Thumbnail generation failed for %s", file_path, exc_info=True)
    return FileResponse(file_path)


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_query(audio: UploadFile = File(...)) -> TranscribeResponse:
    """Transcribe a short spoken query to text using the local Whisper stack.

    Backs the web UI's voice-search mic. Fully local — audio is written to a temp
    file, normalized via ffmpeg, transcribed, and discarded; nothing leaves the
    machine. Uses the smaller ``query_transcription_model`` for low latency.
    """
    import tempfile  # noqa: PLC0415

    from app.ingest.audio import convert_audio_to_wav  # noqa: PLC0415

    engine = _get_query_transcriber()
    if engine.backend == "unavailable":
        raise HTTPException(
            status_code=503,
            detail='Voice search needs the audio extras. Run: pip install -e ".[audio]"',
        )

    raw_bytes = audio.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if len(raw_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large (max 25 MB)")

    suffix = Path(audio.filename or "query.webm").suffix or ".webm"
    model_name = _config.models.query_transcription_model

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / f"query{suffix}"
        raw_path.write_bytes(raw_bytes)
        wav_path = convert_audio_to_wav(raw_path, Path(tmpdir) / "query.wav")
        if wav_path is None:
            raise HTTPException(
                status_code=500,
                detail="Audio conversion failed (is ffmpeg installed and on PATH?)",
            )
        transcript = engine.transcribe(wav_path, original_path=raw_path, model_name=model_name)

    error = transcript.metadata.get("error")
    if error:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {error}")

    text = " ".join(seg.text for seg in transcript.segments).strip()
    return TranscribeResponse(text=text, language=transcript.language)


@app.post("/open-file")
def open_file(request: OpenFileRequest) -> dict[str, str]:
    """Open a file in the default OS application. Localhost-only — no auth needed."""
    path = Path(request.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])  # noqa: S603, S607
        else:
            subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "opened", "file": str(path)}


@app.get("/editor-link", response_model=EditorLinkResponse)
def editor_link(file_path: str, line: int = 1, editor: str = "vscode") -> EditorLinkResponse:
    """Return a deep-link URI to open a file in a local editor.

    ``editor`` options: ``vscode``, ``obsidian``, ``default``.
    For ``obsidian`` the caller must also supply ``vault`` as a query param.
    """
    path = Path(file_path)
    if editor == "obsidian":
        # obsidian://open?path=<abs-path>
        from urllib.parse import quote  # noqa: PLC0415
        uri = f"obsidian://open?path={quote(str(path))}"
        return EditorLinkResponse(uri=uri, scheme="obsidian")
    if editor == "vscode":
        from urllib.parse import quote  # noqa: PLC0415
        uri = f"vscode://file/{quote(str(path))}:{line}"
        return EditorLinkResponse(uri=uri, scheme="vscode")
    # Default: plain file:// URI
    from urllib.parse import quote  # noqa: PLC0415
    uri = "file://" + quote(str(path).replace("\\", "/"))
    return EditorLinkResponse(uri=uri, scheme="file")
