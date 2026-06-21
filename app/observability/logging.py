"""Logging setup for ingest, query, and health-check flows."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler

from app.config import AppConfig

# ---------------------------------------------------------------------------
# Sensitive-content redaction (Section 17.2 — NFR-S-03)
# ---------------------------------------------------------------------------

# Patterns are applied to the formatted log message before emission.
# Each tuple is (compiled_pattern, replacement_string).
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    # File paths (must contain a directory separator) whose filename contains a sensitive keyword
    (re.compile(r"[^\s]*[/\\][^\s]*(?:password|secret|token|credential)[^\s]*", re.I), "[REDACTED_PATH]"),
    # Bearer / Authorization header values
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*"), r"\1[TOKEN]"),
    # API keys / secrets (key=<value> or key: <value> patterns)
    (re.compile(r"(?i)((?:api[_-]?key|secret|password|passwd|token)\s*[=:]\s*)\S+"), r"\1[REDACTED]"),
]


class RedactingFilter(logging.Filter):
    """Strips sensitive patterns from log records before they are written.

    Applied to every handler so the redaction happens regardless of sink
    (stream, file, etc.).  The filter mutates the formatted message *after*
    ``logging.Formatter`` has run, so it is safe to attach at the handler
    level rather than the logger level.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.msg = _redact(str(record.msg))
        if record.args:
            # Redact each arg individually so sensitive values in args are scrubbed
            # even if the template was also mutated by redaction.
            if isinstance(record.args, tuple):
                clean_args: object = tuple(_redact(str(a)) for a in record.args)
            elif isinstance(record.args, dict):
                clean_args = {k: _redact(str(v)) for k, v in record.args.items()}
            else:
                clean_args = record.args
            try:
                # Render template with clean args, then redact the combined message.
                record.msg = _redact(record.msg % clean_args)
            except Exception:  # noqa: BLE001
                # Placeholder count may have changed after redaction — leave template as-is.
                pass
            record.args = None
        return True


def _redact(text: str) -> str:
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Configuration entry point
# ---------------------------------------------------------------------------


def configure_logging(config: AppConfig) -> None:
    config.ensure_directories()
    log_path = config.paths.log_dir / "lifelog.log"
    redacting_filter = RedactingFilter()
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5),
    ]
    for handler in handlers:
        handler.addFilter(redacting_filter)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )

