"""HMAC signature verification and cryptographic utilities."""

from __future__ import annotations

import hashlib
import hmac
import secrets


def verify_hmac_sha256(secret_key: str, message: str | bytes, expected_signature: str) -> bool:
    """Verify an incoming payload against an expected HMAC SHA256 hex signature."""
    if not secret_key or not message or not expected_signature:
        return False

    msg_bytes = message.encode("utf-8") if isinstance(message, str) else message
    key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key

    computed = hmac.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()
    return secrets.compare_digest(computed.lower(), expected_signature.strip().lower())


def generate_hmac_sha256(secret_key: str, message: str | bytes) -> str:
    """Generate a hex-encoded HMAC SHA256 signature for a message."""
    msg_bytes = message.encode("utf-8") if isinstance(message, str) else message
    key_bytes = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    return hmac.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()


__all__ = ["generate_hmac_sha256", "verify_hmac_sha256"]
