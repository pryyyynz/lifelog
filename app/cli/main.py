"""Command-line interface for setup validation and early local runtime checks."""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.config import get_config
from app.ingest.registry import (
    SETUP_GUIDANCE,
    SourceKind,
    SourceRegistry,
    build_source_config,
    validate_source,
)
from app.ingest.runner import IngestRunner
from app.storage.metadata import MetadataStore


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_python() -> CheckResult:
    version = sys.version_info
    ok = version >= (3, 11)
    return CheckResult("python", ok, f"{version.major}.{version.minor}.{version.micro}")


def _check_executable(name: str) -> CheckResult:
    path = shutil.which(name)
    return CheckResult(name, path is not None, path or "not found on PATH")


def _check_ffmpeg() -> CheckResult:
    config = get_config()
    if config.paths.ffmpeg_path:
        path = config.paths.ffmpeg_path
        return CheckResult("ffmpeg", path.exists(), str(path) if path.exists() else f"missing: {path}")
    return _check_executable("ffmpeg")


def _check_qdrant(url: str) -> CheckResult:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=2) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
        return CheckResult("qdrant", response.status == 200, body or f"HTTP {response.status}")
    except urllib.error.URLError as exc:
        return CheckResult("qdrant", False, str(exc.reason))
    except TimeoutError:
        return CheckResult("qdrant", False, "timed out")


def _check_localhost_bind(host: str, port: int) -> CheckResult:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            return CheckResult("api_port", False, f"{host}:{port} unavailable: {exc}")
    return CheckResult("api_port", True, f"{host}:{port} available")


def _check_model_cache() -> CheckResult:
    config = get_config()
    expected_roots = (
        config.paths.model_dir / "sentence-transformers",
        config.paths.model_dir / "open_clip",
        config.paths.model_dir / "transcription",
        config.paths.model_dir / "cross-encoders",
    )
    missing = [str(path) for path in expected_roots if not path.exists() or not any(path.rglob("*"))]
    if missing:
        return CheckResult("model_cache", False, "missing or empty: " + ", ".join(missing))
    return CheckResult("model_cache", True, str(config.paths.model_dir))


def doctor(_: argparse.Namespace) -> int:
    config = get_config()
    config.ensure_directories()
    config.activate_tool_paths()
    checks = [
        _check_python(),
        _check_ffmpeg(),
        _check_executable("docker"),
        _check_qdrant(config.vector_store.url),
        _check_localhost_bind(config.api_host, config.api_port),
        _check_model_cache(),
        CheckResult("data_dir", config.paths.data_dir.exists(), str(config.paths.data_dir)),
        CheckResult("model_dir", config.paths.model_dir.exists(), str(config.paths.model_dir)),
        CheckResult("sqlite_parent", config.paths.sqlite_path.parent.exists(), str(config.paths.sqlite_path)),
    ]
    for check in checks:
        marker = "OK" if check.ok else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


def status(_: argparse.Namespace) -> int:
    config = get_config()
    registry = SourceRegistry(config.paths.source_registry_path)
    store = MetadataStore(config.paths.sqlite_path)
    print(f"environment: {config.env}")
    print(f"api: http://{config.api_host}:{config.api_port}")
    print(f"sqlite: {config.paths.sqlite_path}")
    print(f"source_registry: {config.paths.source_registry_path}")
    print(f"qdrant: {config.vector_store.url}")
    print(f"modalities: {', '.join(config.enabled_modalities)}")
    latest = store.latest_ingest_timestamp()
    print(f"last_ingest_timestamp: {latest.isoformat() if latest else 'never'}")
    counts = store.status_counts()
    if registry.sources:
        print("sources:")
        for source in registry.sources:
            count = counts.get(source.id, 0)
            state = "enabled" if source.enabled else "disabled"
            print(
                f"  - {source.id} [{state}] {source.source_type.value} "
                f"{source.mode.value} {source.path} ({count} tracked items)"
            )
    else:
        print("sources: none configured")
    return 0


def init_sources(args: argparse.Namespace) -> int:
    config = get_config()
    config.ensure_directories()
    registry_path = Path(args.config).expanduser().resolve() if args.config else config.paths.source_registry_path
    registry = SourceRegistry(registry_path)

    source_specs = list(args.source or [])
    if not source_specs:
        if not sys.stdin.isatty():
            print("No sources provided. Use --source type=path, for example --source text=C:\\Notes.")
            return 2
        source_specs = _prompt_for_sources()

    added = 0
    for spec in source_specs:
        try:
            source_type, path = _parse_source_spec(spec)
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 2

        source = build_source_config(source_type, path)
        validation = validate_source(source)
        marker = "OK" if validation.ok else "FAIL"
        print(f"[{marker}] {source.source_type.value}: {source.path} ({validation.item_count} items)")
        for warning in validation.warnings:
            print(f"  warning: {warning}")
        for error in validation.errors:
            print(f"  error: {error}")
        if not validation.ok:
            continue

        registry.upsert(source)
        added += 1

    registry.save()
    print(f"saved source registry: {registry.path}")
    if added:
        print("next: run lifelog ingest --full")
    return 0 if added or registry.sources else 1


def ingest(args: argparse.Namespace) -> int:
    config = get_config()
    config.ensure_directories()
    config.activate_tool_paths()
    registry = SourceRegistry(config.paths.source_registry_path)
    if not registry.enabled_sources():
        print("No enabled sources configured. Run lifelog init first.")
        return 2

    store = MetadataStore(config.paths.sqlite_path)
    runner = IngestRunner(registry, store)
    summary = runner.run(full=args.full, source_id=args.source_id)
    print(f"run_id: {summary.run_id}")
    print(f"mode: {summary.mode}")
    print(f"processed_items: {summary.processed_items}")
    print(f"skipped_items: {summary.skipped_items}")
    print(f"failed_items: {summary.failed_items}")
    print(f"duration_seconds: {summary.duration_seconds:.3f}")
    return 0 if summary.failed_items == 0 else 1


def _prompt_for_sources() -> list[str]:
    specs: list[str] = []
    print("Configure local data sources. Leave a path blank to skip that source.")
    for source_type in SourceKind:
        print(f"\n{source_type.value}: {SETUP_GUIDANCE[source_type]}")
        value = input("path: ").strip()
        if value:
            specs.append(f"{source_type.value}={value}")
    return specs


def _parse_source_spec(spec: str) -> tuple[SourceKind, Path]:
    if "=" not in spec:
        raise ValueError(f"source must use type=path format: {spec}")
    raw_type, raw_path = spec.split("=", 1)
    try:
        source_type = SourceKind(raw_type.strip())
    except ValueError as exc:
        supported = ", ".join(item.value for item in SourceKind)
        raise ValueError(f"unsupported source type '{raw_type}'. Supported: {supported}") from exc
    path = Path(raw_path.strip()).expanduser()
    if not str(path):
        raise ValueError(f"empty path for source type '{source_type.value}'")
    return source_type, path


def query_cmd(args: argparse.Namespace) -> int:
    """Run a natural-language query and pretty-print session cards."""
    config = get_config()
    from app.ranking.grouper import SessionGrouper  # noqa: PLC0415
    from app.ranking.reranker import CrossEncoderReranker, TemporalReranker  # noqa: PLC0415
    from app.retrieval.chat_intent import conversational_reply, is_conversational_query  # noqa: PLC0415
    from app.retrieval.query_analyzer import QueryAnalyzer  # noqa: PLC0415
    from app.retrieval.retriever import Retriever  # noqa: PLC0415

    if is_conversational_query(args.query):
        print(conversational_reply(args.query))
        return 0

    store = MetadataStore(config.paths.sqlite_path)
    retriever = Retriever(store)
    analyzer = QueryAnalyzer(use_spacy=False)
    signals = analyzer.analyze(args.query)

    limit = getattr(args, "limit", 10)
    hits = retriever.retrieve(args.query, signals=signals, limit=max(limit * 4, 50))

    if signals.temporal_range:
        hits = TemporalReranker.from_environment().rerank(hits, signals.temporal_range[0])

    hits = CrossEncoderReranker.from_environment().rerank(hits, args.query)
    cards = SessionGrouper(top_n=limit).group(hits)

    if not cards:
        print("No results.")
        return 0

    for i, card in enumerate(cards, 1):
        start = card.start_utc.strftime("%Y-%m-%d %H:%M") if card.start_utc else "unknown time"
        place = f"  {card.hits[0].place_name}" if card.hits[0].place_name else ""
        print(f"\n{'─' * 60}")
        print(f"[{i}] Session  {start}{place}  (score={card.score:.3f})")
        print(f"{'─' * 60}")
        for j, hit in enumerate(card.hits):
            label = "▶" if j == 0 else " "
            ts = hit.timestamp_utc.strftime("%H:%M") if hit.timestamp_utc else ""
            snippet = (hit.snippet or "")[:120].replace("\n", " ")
            rationale = ",".join(hit.rationale[:2])
            print(f"  {label} [{hit.source_type}] {ts}  {snippet}")
            print(f"      {hit.file_path}  [{rationale}]")
    return 0


def enrich_cmd(args: argparse.Namespace) -> int:
    """Run AI enrichment (OCR, captions, ...) over already-ingested chunks."""
    config = get_config()
    config.ensure_directories()
    config.activate_tool_paths()

    from app.enrich.registry import build_enrichers  # noqa: PLC0415
    from app.enrich.runner import EnrichmentRunner  # noqa: PLC0415
    from app.ingest.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

    enrichers = build_enrichers(config)
    if not enrichers:
        print("No enrichers enabled. Enable at least one, e.g. LIFELOG_ENRICH_OCR=1.")
        return 2

    store = MetadataStore(config.paths.sqlite_path)
    embedder = SentenceTransformerEmbedder.from_environment()
    runner = EnrichmentRunner(
        store,
        enrichers,
        embedder=embedder,
        batch_size=config.enrichment.batch_size,
    )
    summary = runner.run(limit=args.limit, include_failed=args.retry_failed)

    print(f"enrichers: {', '.join(e.name for e in enrichers)}")
    print(f"done: {summary.done}")
    print(f"skipped: {summary.skipped}")
    print(f"failed: {summary.failed}")
    if summary.unavailable:
        print(f"unavailable (model/deps missing): {', '.join(summary.unavailable)}")

    # Cluster any newly detected faces.
    if any(e.name == "faces" for e in enrichers) and "faces" not in summary.unavailable:
        from app.enrich.clustering import FaceClusterer  # noqa: PLC0415

        cluster_summary = FaceClusterer(
            store, threshold=config.enrichment.face_cluster_threshold
        ).cluster_new()
        print(
            f"faces clustered: {cluster_summary.processed} "
            f"({cluster_summary.new_clusters} new clusters)"
        )

    return 0 if summary.failed == 0 else 1


def delete_cmd(args: argparse.Namespace) -> int:
    config = get_config()
    store = MetadataStore(config.paths.sqlite_path)

    # Try to load vector store for Qdrant cleanup
    try:
        from app.storage.vector_store import VectorStore  # noqa: PLC0415
        vs = VectorStore.from_environment()
    except Exception:  # noqa: BLE001
        vs = None

    if args.file:
        path = Path(args.file).expanduser().resolve()
        n = store.delete_chunks_for_file(path)
        if vs is not None:
            try:
                vs.delete_by_file_path(str(path))
            except Exception:  # noqa: BLE001
                pass
        print(f"deleted {n} chunks for file: {path}")
        return 0

    if args.source:
        file_paths = store.file_paths_for_source(args.source)
        n = store.delete_chunks_by_source_id(args.source)
        store.delete_source_files_by_source_id(args.source)
        if vs is not None:
            for fp in file_paths:
                try:
                    vs.delete_by_file_path(str(fp))
                except Exception:  # noqa: BLE001
                    pass
        print(f"deleted {n} chunks for source: {args.source}")
        return 0

    print("Provide --file <path> or --source <id>.")
    return 2


def consistency_check_cmd(_: argparse.Namespace) -> int:
    config = get_config()
    try:
        from app.storage.consistency import ConsistencyChecker  # noqa: PLC0415
        from app.storage.vector_store import VectorStore  # noqa: PLC0415
    except ImportError as exc:
        print(f"[FAIL] Cannot import storage layer: {exc}")
        return 1

    store = MetadataStore(config.paths.sqlite_path)
    try:
        vs = VectorStore.from_environment()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Cannot connect to Qdrant: {exc}")
        return 1

    checker = ConsistencyChecker(store, vs)
    report = checker.check()

    if report.ok:
        print("[OK] SQLite and Qdrant are consistent.")
        return 0

    print(f"[FAIL] {report.total_orphans} orphan record(s) found.")
    if report.orphaned_sqlite_chunk_ids:
        print(f"  SQLite chunks with no Qdrant vector ({len(report.orphaned_sqlite_chunk_ids)}):")
        for cid in list(report.orphaned_sqlite_chunk_ids)[:10]:
            print(f"    {cid}")
    if report.orphaned_qdrant_point_ids:
        print(f"  Qdrant points with no SQLite chunk ({len(report.orphaned_qdrant_point_ids)}):")
        for pid in list(report.orphaned_qdrant_point_ids)[:10]:
            print(f"    {pid}")
    return 1


def logs_cmd(args: argparse.Namespace) -> int:
    config = get_config()
    log_dir = config.paths.log_dir
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return 1

    pattern = "*.log"
    log_files = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        print(f"No log files in: {log_dir}")
        return 0

    target = log_files[0]
    source_filter = getattr(args, "source", None)
    lines_shown = 0
    max_lines = getattr(args, "lines", 50)

    try:
        with target.open(encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()

        filtered = [
            line for line in all_lines
            if source_filter is None or source_filter in line
        ]
        for line in filtered[-max_lines:]:
            print(line, end="")
            lines_shown += 1
    except OSError as exc:
        print(f"[FAIL] Cannot read log: {exc}")
        return 1

    if lines_shown == 0:
        print(f"No matching log entries (source filter: {source_filter}).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lifelog")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subcommands.add_parser("doctor", help="Validate local runtime prerequisites.")
    doctor_parser.set_defaults(func=doctor)

    status_parser = subcommands.add_parser("status", help="Print configured local runtime paths.")
    status_parser.set_defaults(func=status)

    init_parser = subcommands.add_parser("init", help="Configure local data sources for ingest.")
    init_parser.add_argument(
        "--source",
        action="append",
        metavar="TYPE=PATH",
        help="Add a source non-interactively. Can be repeated.",
    )
    init_parser.add_argument(
        "--config",
        help="Optional source registry path. Defaults to LIFELOG_SOURCE_REGISTRY_PATH.",
    )
    init_parser.set_defaults(func=init_sources)

    ingest_parser = subcommands.add_parser("ingest", help="Run full or incremental ingest.")
    ingest_mode = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_mode.add_argument("--full", action="store_true", help="Process every discovered item.")
    ingest_mode.add_argument(
        "--incremental",
        action="store_true",
        help="Process only new or modified items.",
    )
    ingest_parser.add_argument("--source-id", help="Optional source id to ingest.")
    ingest_parser.set_defaults(func=ingest)

    query_parser = subcommands.add_parser("query", help="Run a natural-language query.")
    query_parser.add_argument("query", help="Query string.")
    query_parser.add_argument("--limit", type=int, default=5, help="Number of session cards to return.")
    query_parser.set_defaults(func=query_cmd)

    enrich_parser = subcommands.add_parser(
        "enrich", help="Run AI enrichment (OCR, captions, ...) over ingested data."
    )
    enrich_parser.add_argument(
        "--limit", type=int, default=None, help="Max source chunks to process per enricher."
    )
    enrich_parser.add_argument(
        "--retry-failed", action="store_true", help="Also retry previously failed items."
    )
    enrich_parser.set_defaults(func=enrich_cmd)

    delete_parser = subcommands.add_parser("delete", help="Remove chunks from the index.")
    delete_group = delete_parser.add_mutually_exclusive_group(required=True)
    delete_group.add_argument("--file", metavar="PATH", help="Remove all chunks for a specific file.")
    delete_group.add_argument("--source", metavar="ID", help="Remove all chunks for a source id.")
    delete_parser.set_defaults(func=delete_cmd)

    cc_parser = subcommands.add_parser(
        "consistency-check", help="Verify SQLite and Qdrant are in sync."
    )
    cc_parser.set_defaults(func=consistency_check_cmd)

    logs_parser = subcommands.add_parser("logs", help="Tail the ingest log.")
    logs_parser.add_argument("--source", help="Filter log lines to a specific source id.")
    logs_parser.add_argument("--lines", type=int, default=50, help="Number of lines to show.")
    logs_parser.set_defaults(func=logs_cmd)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
