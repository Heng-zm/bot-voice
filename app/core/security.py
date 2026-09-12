"""Centralized authentication and API key verification helpers."""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)


def get_allowed_api_keys() -> set[str]:
    """Retrieve authorized API keys from environment variables."""
    keys: set[str] = set()
    for var in ("BOT_API_KEY", "API_KEY", "BOT_API_KEYS", "AI_API_KEY"):
        val = (os.environ.get(var) or "").strip()
        if val:
            keys.update(k.strip() for k in val.split(",") if k.strip())
    return keys


def validate_api_key(x_api_key: str | None, authorization: str | None) -> bool:
    """Validate incoming API key against configured environment keys or dynamic DB keys.

    Accepts key from either 'X-Api-Key' header or 'Authorization: Bearer <key>'.
    Uses constant-time comparison to prevent timing attacks.
    """
    api_key = (x_api_key or "").strip()
    if not api_key and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            api_key = parts[1].strip()
        else:
            api_key = authorization.strip()

    if not api_key:
        return False

    allowed = get_allowed_api_keys()
    if allowed and any(secrets.compare_digest(api_key, valid_key) for valid_key in allowed):
        return True

    # Check dynamic API keys generated via Telegram /api create
    try:
        from app import legacy

        validate_dynamic = getattr(legacy, "_validate_dynamic_ai_api_key", None)
        if callable(validate_dynamic) and validate_dynamic(api_key):
            return True
    except Exception as exc:
        logger.debug("Dynamic API key check failed: %s", exc)

    return False


def verify_api_key_dependency(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> bool:
    """FastAPI dependency to enforce API key authentication on protected endpoints."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")
    return True


__all__ = [
    "get_allowed_api_keys",
    "validate_api_key",
    "verify_api_key_dependency",
]
