"""Reverse geocoding utility shared across ingestors that carry GPS coordinates."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class ReverseGeocoder:
    """Resolves (lat, lon) to a human-readable place name via Nominatim.

    Results are cached in a local JSON file so repeated lookups for the same
    coordinate bucket are free.  Pass ``enabled=False`` (or set
    ``LIFELOG_ENABLE_REVERSE_GEOCODING=false``) to skip all network calls for
    fully offline operation.
    """

    # Snap to a ~1 km grid to maximise cache reuse.
    _BUCKET = 2

    def __init__(self, cache_path: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.cache_path = cache_path
        self._cache: dict[str, str | None] = {}
        self._last_request: float = 0.0
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._cache = {}

    @classmethod
    def from_environment(cls) -> ReverseGeocoder:
        enabled = os.getenv("LIFELOG_ENABLE_REVERSE_GEOCODING", "true").lower() not in {
            "0", "false", "no", "off"
        }
        data_dir = Path(os.getenv("LIFELOG_DATA_DIR", "./data"))
        cache_path = data_dir / "geocode_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(cache_path=cache_path, enabled=enabled)

    def lookup(self, lat: float, lon: float) -> str | None:
        """Return a place name string or None.  Respects Nominatim 1 req/s rate limit."""
        if not self.enabled:
            return None
        key = self._cache_key(lat, lon)
        if key in self._cache:
            return self._cache[key]
        result = self._nominatim(lat, lon)
        self._cache[key] = result
        self._flush()
        return result

    # ------------------------------------------------------------------

    def _cache_key(self, lat: float, lon: float) -> str:
        b = self._BUCKET
        snapped_lat = round(round(lat / b) * b, b)
        snapped_lon = round(round(lon / b) * b, b)
        return f"{snapped_lat},{snapped_lon}"

    def _nominatim(self, lat: float, lon: float) -> str | None:
        try:
            from geopy.geocoders import Nominatim  # type: ignore[import-untyped]
            from geopy.exc import GeocoderTimedOut, GeocoderUnavailable  # type: ignore[import-untyped]
        except ImportError:
            return None

        # Respect rate limit: max 1 request per second.
        elapsed = time.monotonic() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        try:
            geocoder = Nominatim(user_agent="lifelog-search/1.0")
            location = geocoder.reverse((lat, lon), language="en", timeout=10)
            self._last_request = time.monotonic()
            if location is None:
                return None
            raw: dict[str, Any] = location.raw.get("address", {})
            return _format_place(raw)
        except (GeocoderTimedOut, GeocoderUnavailable):
            self._last_request = time.monotonic()
            return None
        except Exception:  # noqa: BLE001
            self._last_request = time.monotonic()
            return None

    def _flush(self) -> None:
        try:
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass


def _format_place(address: dict[str, Any]) -> str | None:
    """Build a short human-readable place label from a Nominatim address dict."""
    parts: list[str] = []
    for key in ("city", "town", "village", "suburb", "county", "state", "country"):
        value = address.get(key)
        if value and value not in parts:
            parts.append(value)
        if len(parts) == 2:
            break
    return ", ".join(parts) if parts else None
