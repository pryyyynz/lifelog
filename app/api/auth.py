"""Minimal session tokens for the single-user web login.

A token is ``base64url(payload).base64url(hmac_sha256(secret, payload))`` where
the payload carries only an expiry. Stateless (no server-side session store) and
dependency-free — verification just recomputes the HMAC and checks the clock.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(secret: str, payload_b64: str) -> str:
    digest = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    return _b64e(digest)


def create_token(secret: str, ttl_seconds: int) -> str:
    """Mint a signed token that expires ``ttl_seconds`` from now."""
    payload = {"exp": int(time.time()) + int(ttl_seconds)}
    payload_b64 = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    return f"{payload_b64}.{_sign(secret, payload_b64)}"


def verify_token(secret: str, token: str) -> bool:
    """Return True iff the token's signature is valid and it hasn't expired."""
    if not secret or not token or "." not in token:
        return False
    payload_b64, sig = token.split(".", 1)
    if not hmac.compare_digest(sig, _sign(secret, payload_b64)):
        return False
    try:
        payload = json.loads(_b64d(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return False
    return int(payload.get("exp", 0)) > int(time.time())
