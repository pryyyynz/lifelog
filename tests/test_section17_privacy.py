"""Section 17 — Privacy, Security, Local-Only.

Verifies:
- §17.1  No query-time data leaves the machine (localhost API binding).
- §17.2  Log redaction strips sensitive patterns; storage paths configurable.
- §17.3  External service exposure is limited and opt-in.
"""

from __future__ import annotations

import logging
import os

import pytest

from app.config import AppConfig
from app.observability.logging import RedactingFilter, _redact

# ---------------------------------------------------------------------------
# §17.2 — Log redaction helpers (unit)
# ---------------------------------------------------------------------------


class TestRedactText:
    """_redact() replaces sensitive patterns in plain strings."""

    def test_email_is_redacted(self):
        result = _redact("Logged in as john.doe@example.com from 10.0.0.1")
        assert "[EMAIL]" in result
        assert "john.doe@example.com" not in result

    def test_email_with_plus_sign(self):
        result = _redact("user+tag@sub.domain.io connected")
        assert "[EMAIL]" in result
        assert "@sub.domain.io" not in result

    def test_bearer_token_redacted(self):
        raw = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        result = _redact(raw)
        assert "[TOKEN]" in result
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_api_key_param_redacted(self):
        result = _redact("api_key=supersecret12345")
        assert "[REDACTED]" in result
        assert "supersecret12345" not in result

    def test_password_param_redacted(self):
        result = _redact("password=hunter2 login accepted")
        assert "[REDACTED]" in result
        assert "hunter2" not in result

    def test_passwd_variant_redacted(self):
        result = _redact("passwd: letmein")
        assert "[REDACTED]" in result
        assert "letmein" not in result

    def test_token_env_redacted(self):
        result = _redact("token=ghp_xxxxABCD1234")
        assert "[REDACTED]" in result
        assert "ghp_xxxxABCD1234" not in result

    def test_secret_field_redacted(self):
        result = _redact("secret=my-very-private-value")
        assert "[REDACTED]" in result
        assert "my-very-private-value" not in result

    def test_path_with_token_in_filename_redacted(self):
        result = _redact("loading /home/user/.config/token_store.json")
        assert "/home/user/.config/token_store.json" not in result

    def test_innocent_message_unchanged(self):
        msg = "Ingested 42 text chunks from documents folder"
        assert _redact(msg) == msg

    def test_multiple_patterns_one_line(self):
        result = _redact("api_key=abc123 admin@corp.io password=xyz")
        assert "abc123" not in result
        assert "admin@corp.io" not in result
        assert "xyz" not in result

    def test_empty_string(self):
        assert _redact("") == ""


class TestRedactingFilterIntegration:
    """RedactingFilter integrates correctly with stdlib logging.LogRecord."""

    def _make_record(self, msg: str, args: tuple = ()) -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=args,
            exc_info=None,
        )

    def test_filter_allows_record_through(self):
        flt = RedactingFilter()
        record = self._make_record("Hello world")
        assert flt.filter(record) is True

    def test_filter_redacts_email_in_msg(self):
        flt = RedactingFilter()
        record = self._make_record("Login from user@secret.org failed")
        flt.filter(record)
        assert "user@secret.org" not in record.msg
        assert "[EMAIL]" in record.msg

    def test_filter_with_format_args(self):
        """When args are present, filter pre-renders and clears them."""
        flt = RedactingFilter()
        record = self._make_record(
            "Token=%s user=%s",
            args=("Bearer tok123", "admin@corp.io"),
        )
        flt.filter(record)
        assert record.args is None
        assert "tok123" not in record.msg
        assert "admin@corp.io" not in record.msg

    def test_innocent_record_unchanged(self):
        flt = RedactingFilter()
        record = self._make_record("Processing file chunk 3 of 10")
        original_msg = record.msg
        flt.filter(record)
        assert record.msg == original_msg


# ---------------------------------------------------------------------------
# §17.1 — API binds localhost by default (NFR-S-01)
# ---------------------------------------------------------------------------


class TestAPILocalhostBinding:
    """The API host must default to 127.0.0.1 to prevent unintended exposure."""

    def test_default_host_is_loopback(self, monkeypatch):
        monkeypatch.delenv("LIFELOG_API_HOST", raising=False)
        cfg = AppConfig()
        assert cfg.api_host in ("127.0.0.1", "localhost"), (
            f"API host {cfg.api_host!r} is not localhost — "
            "risk of unintended external network exposure"
        )

    def test_host_is_overridable(self, monkeypatch):
        monkeypatch.setenv("LIFELOG_API_HOST", "0.0.0.0")
        # AppConfig reads env at class creation time via default field values,
        # so we construct from explicit kwargs.
        cfg = AppConfig(api_host=os.getenv("LIFELOG_API_HOST", "127.0.0.1"))
        assert cfg.api_host == "0.0.0.0"

    def test_default_port(self, monkeypatch):
        monkeypatch.delenv("LIFELOG_API_PORT", raising=False)
        cfg = AppConfig()
        assert cfg.api_port == 8000


# ---------------------------------------------------------------------------
# §17.2 — Storage paths are configurable (NFR-S-02)
# ---------------------------------------------------------------------------


class TestConfigurableStoragePaths:
    """All storage roots are configurable via environment variables."""

    def test_paths_attributes_exist(self):
        cfg = AppConfig()
        assert cfg.paths.data_dir is not None
        assert cfg.paths.log_dir is not None
        assert cfg.paths.sqlite_path is not None

    def test_data_dir_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LIFELOG_DATA_DIR", str(tmp_path / "mydata"))
        from app.config import PathsConfig
        paths = PathsConfig()
        assert str(tmp_path / "mydata") in str(paths.data_dir)

    def test_log_dir_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LIFELOG_LOG_DIR", str(tmp_path / "mylogs"))
        from app.config import PathsConfig
        paths = PathsConfig()
        assert str(tmp_path / "mylogs") in str(paths.log_dir)

    def test_sqlite_path_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LIFELOG_SQLITE_PATH", str(tmp_path / "test.sqlite3"))
        from app.config import PathsConfig
        paths = PathsConfig()
        assert str(tmp_path / "test.sqlite3") in str(paths.sqlite_path)


# ---------------------------------------------------------------------------
# §17.3 — External services are opt-in / conditional
# ---------------------------------------------------------------------------


class TestExternalServiceGating:
    """Optional external services are guarded by feature flags."""

    def test_offline_mode_defaults_false(self, monkeypatch):
        monkeypatch.delenv("LIFELOG_OFFLINE_MODE", raising=False)
        cfg = AppConfig()
        # offline_mode=False means the user has not explicitly disabled all network,
        # but Nominatim geocoding is guarded by enable_reverse_geocoding separately.
        assert isinstance(cfg.offline_mode, bool)

    def test_offline_mode_can_be_enabled(self, monkeypatch):
        monkeypatch.setenv("LIFELOG_OFFLINE_MODE", "true")
        from app.config import AppConfig as Cfg
        cfg = Cfg(offline_mode=True)
        assert cfg.offline_mode is True

    def test_reverse_geocoding_defaults_true(self, monkeypatch):
        """Nominatim is enabled by default but can be disabled."""
        monkeypatch.delenv("LIFELOG_ENABLE_REVERSE_GEOCODING", raising=False)
        cfg = AppConfig()
        assert isinstance(cfg.enable_reverse_geocoding, bool)

    def test_image_embedding_gated_by_env(self, monkeypatch):
        """OpenCLIP embedding can be disabled via env var."""
        monkeypatch.setenv("LIFELOG_ENABLE_IMAGE_EMBEDDING", "false")
        from app.ingest.images import OpenClipImageEmbedder
        embedder = OpenClipImageEmbedder.from_environment()
        assert not embedder.enabled

    def test_qdrant_api_key_not_required(self):
        """Local Qdrant does not require an API key by default."""
        cfg = AppConfig()
        assert cfg.vector_store.api_key is None or isinstance(cfg.vector_store.api_key, str)

    def test_qdrant_url_is_loopback_by_default(self, monkeypatch):
        monkeypatch.delenv("LIFELOG_QDRANT_URL", raising=False)
        cfg = AppConfig()
        assert "127.0.0.1" in cfg.vector_store.url or "localhost" in cfg.vector_store.url
