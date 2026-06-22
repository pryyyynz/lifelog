"""Runtime configuration for the local Life Log Search service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODALITIES = (
    "text",
    "email",
    "photos",
    "audio",
    "video",
    "calendar",
    "browser_history",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load ./.env once at import so os.getenv below sees it. Real environment
# variables take precedence (override=False) so launchers can still override.
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    raw = Path(os.getenv(name, default)).expanduser()
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    return raw.resolve()


def _optional_tool_path(env_name: str, pattern: str) -> Path | None:
    value = os.getenv(env_name)
    if value:
        return Path(value).expanduser().resolve()
    matches = sorted(PROJECT_ROOT.glob(pattern))
    return matches[0].resolve() if matches else None


def _ffmpeg_path() -> Path | None:
    value = os.getenv("LIFELOG_FFMPEG_PATH")
    if value:
        return Path(value).expanduser().resolve()
    for pattern in (
        "tools/ffmpeg/*shared*/bin/ffmpeg.exe",
        "tools/ffmpeg/*7*/bin/ffmpeg.exe",
        "tools/ffmpeg/**/bin/ffmpeg.exe",
    ):
        matches = sorted(PROJECT_ROOT.glob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_from_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = field(default_factory=lambda: _path_from_env("LIFELOG_DATA_DIR", "./data"))
    model_dir: Path = field(default_factory=lambda: _path_from_env("LIFELOG_MODEL_DIR", "./models"))
    log_dir: Path = field(default_factory=lambda: _path_from_env("LIFELOG_LOG_DIR", "./logs"))
    sqlite_path: Path = field(
        default_factory=lambda: _path_from_env("LIFELOG_SQLITE_PATH", "./data/lifelog.sqlite3")
    )
    source_registry_path: Path = field(
        default_factory=lambda: _path_from_env(
            "LIFELOG_SOURCE_REGISTRY_PATH", "./data/sources.json"
        )
    )
    ffmpeg_path: Path | None = field(default_factory=_ffmpeg_path)


@dataclass(frozen=True)
class ModelsConfig:
    text_embedding_model: str = os.getenv("LIFELOG_TEXT_EMBEDDING_MODEL", "intfloat/e5-large-v2")
    image_model: str = os.getenv("LIFELOG_IMAGE_MODEL", "ViT-L-14")
    image_pretrained: str = os.getenv("LIFELOG_IMAGE_PRETRAINED", "openai")
    transcription_engine: str = os.getenv("LIFELOG_TRANSCRIPTION_ENGINE", "whisperx")
    transcription_model: str = os.getenv("LIFELOG_TRANSCRIPTION_MODEL", "large-v3")
    # Smaller model for short interactive voice queries (vs. large-v3 for batch ingest).
    query_transcription_model: str = os.getenv("LIFELOG_QUERY_TRANSCRIPTION_MODEL", "base")
    cross_encoder_model: str = os.getenv(
        "LIFELOG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@dataclass(frozen=True)
class VectorStoreConfig:
    mode: str = os.getenv("LIFELOG_VECTOR_STORE_MODE", "qdrant_server")
    url: str = os.getenv("LIFELOG_QDRANT_URL", "http://127.0.0.1:6333")
    api_key: str | None = os.getenv("LIFELOG_QDRANT_API_KEY") or None


@dataclass(frozen=True)
class LLMConfig:
    ollama_url: str = os.getenv("LIFELOG_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("LIFELOG_OLLAMA_MODEL", "llama3")
    enabled: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_LLM_ENABLED", True))
    # RAG answers + query decomposition (Phase 3). Both also require the LLM to be reachable.
    answers_enabled: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_ANSWERS", True))
    decomposition_enabled: bool = field(
        default_factory=lambda: _bool_from_env("LIFELOG_QUERY_DECOMPOSITION", True)
    )
    answer_max_cards: int = int(os.getenv("LIFELOG_ANSWER_MAX_CARDS", "5"))


@dataclass(frozen=True)
class EnrichmentConfig:
    """AI content-enrichment toggles (OCR, captions, tags, actions, faces).

    ``enabled`` gates the automatic/scheduled pass; individual enrichers can still be
    run explicitly via the CLI when their per-feature flag is on. Heavy enrichers
    degrade gracefully when their model/dependency is missing.
    """

    enabled: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_ENRICHMENT", False))
    ocr: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENRICH_OCR", True))
    caption: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENRICH_CAPTION", False))
    vlm: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENRICH_VLM", False))
    tags: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENRICH_TAGS", False))
    action: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENRICH_ACTION", False))
    faces: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENRICH_FACES", False))
    batch_size: int = int(os.getenv("LIFELOG_ENRICH_BATCH_SIZE", "32"))
    ocr_languages: tuple[str, ...] = field(
        default_factory=lambda: _csv_from_env("LIFELOG_OCR_LANGUAGES", ("en",))
    )
    # Captioning (Phase 1)
    caption_model: str = os.getenv("LIFELOG_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    # VLM captioner (reads + describes; richer than BLIP). GPU recommended.
    vlm_model: str = os.getenv("LIFELOG_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    # Zero-shot tags (Phase 1) — reuses the OpenCLIP image model by default.
    tag_model: str = os.getenv("LIFELOG_TAG_MODEL", "ViT-L-14")
    tag_pretrained: str = os.getenv("LIFELOG_TAG_PRETRAINED", "openai")
    tag_top_k: int = int(os.getenv("LIFELOG_TAG_TOP_K", "5"))
    tag_threshold: float = float(os.getenv("LIFELOG_TAG_THRESHOLD", "0.2"))
    tag_labels: tuple[str, ...] = field(
        default_factory=lambda: _csv_from_env("LIFELOG_TAG_LABELS", ())
    )
    # Video action recognition (Phase 1)
    action_model: str = os.getenv("LIFELOG_ACTION_MODEL", "microsoft/xclip-base-patch32")
    action_top_k: int = int(os.getenv("LIFELOG_ACTION_TOP_K", "3"))
    action_threshold: float = float(os.getenv("LIFELOG_ACTION_THRESHOLD", "0.3"))
    action_labels: tuple[str, ...] = field(
        default_factory=lambda: _csv_from_env("LIFELOG_ACTION_LABELS", ())
    )
    # Faces (Phase 2)
    face_model: str = os.getenv("LIFELOG_FACE_MODEL", "buffalo_s")
    face_det_threshold: float = float(os.getenv("LIFELOG_FACE_DET_THRESHOLD", "0.5"))
    face_cluster_threshold: float = float(os.getenv("LIFELOG_FACE_CLUSTER_THRESHOLD", "0.5"))


@dataclass(frozen=True)
class ProactiveConfig:
    """Proactive features (Phase 4): on-this-day, digests, insights, auto titles."""

    on_this_day_enabled: bool = field(
        default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_ON_THIS_DAY", True)
    )
    digests_enabled: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_DIGESTS", True))
    insights_enabled: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_INSIGHTS", True))
    # Auto card titles add an LLM call per query — opt-in to protect latency.
    titles_enabled: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_TITLES", False))
    # Background daily refresh of digests/on-this-day. Off by default.
    schedule_enabled: bool = field(
        default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_PROACTIVE_SCHEDULE", False)
    )


@dataclass(frozen=True)
class AuthConfig:
    """Single-user web login. Auth is OFF unless a password is set."""

    password: str | None = field(
        default_factory=lambda: os.getenv("LIFELOG_AUTH_PASSWORD") or None
    )
    # Token signing key; falls back to the password so a single setting suffices.
    secret: str = field(
        default_factory=lambda: os.getenv("LIFELOG_AUTH_SECRET")
        or os.getenv("LIFELOG_AUTH_PASSWORD")
        or ""
    )
    ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("LIFELOG_AUTH_TTL_SECONDS", str(60 * 60 * 24 * 30)))
    )

    @property
    def enabled(self) -> bool:
        return bool(self.password)


@dataclass(frozen=True)
class SchedulerConfig:
    engine: str = os.getenv("LIFELOG_SCHEDULER", "apscheduler")
    file_watcher: str = os.getenv("LIFELOG_FILE_WATCHER", "watchdog")


@dataclass(frozen=True)
class AppConfig:
    env: str = os.getenv("LIFELOG_ENV", "development")
    api_host: str = os.getenv("LIFELOG_API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("LIFELOG_API_PORT", "8000"))
    enabled_modalities: tuple[str, ...] = field(
        default_factory=lambda: _csv_from_env("LIFELOG_ENABLED_MODALITIES", DEFAULT_MODALITIES)
    )
    offline_mode: bool = field(default_factory=lambda: _bool_from_env("LIFELOG_OFFLINE_MODE", False))
    enable_reverse_geocoding: bool = field(
        default_factory=lambda: _bool_from_env("LIFELOG_ENABLE_REVERSE_GEOCODING", True)
    )
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    def ensure_directories(self) -> None:
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.source_registry_path.parent.mkdir(parents=True, exist_ok=True)

    def activate_tool_paths(self) -> None:
        if not self.paths.ffmpeg_path:
            return
        ffmpeg_bin = str(self.paths.ffmpeg_path.parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if ffmpeg_bin not in path_parts:
            os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")


def get_config() -> AppConfig:
    return AppConfig()
