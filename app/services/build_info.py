"""Build and deploy metadata helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any


def _clean_text(value: Any, *, fallback: str = "", max_length: int = 128) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _clean_timestamp(value: Any) -> str | None:
    text = _clean_text(value, max_length=64)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return text


def _runtime_started_iso(started_at: Any) -> str | None:
    try:
        timestamp = float(started_at or 0.0)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0.0:
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def get_build_info(*, role: str | None = None, started_at: Any = None) -> dict[str, Any]:
    """Return safe deployment metadata for status endpoints and bot commands."""

    version = _clean_text(
        os.getenv("BOT_BUILD_VERSION")
        or os.getenv("APP_VERSION")
        or os.getenv("RELEASE_VERSION"),
        fallback="dev",
        max_length=64,
    )
    commit = _clean_text(
        os.getenv("RELEASE_SHA")
        or os.getenv("COMMIT_SHA")
        or os.getenv("GIT_SHA")
        or os.getenv("RENDER_GIT_COMMIT"),
        max_length=64,
    )
    process_role = _clean_text(
        role or os.getenv("PROCESS_ROLE"),
        fallback="combined",
        max_length=32,
    ).lower()
    deployed_at = _clean_timestamp(
        os.getenv("RELEASE_CREATED_AT")
        or os.getenv("BUILD_CREATED_AT")
        or os.getenv("BUILD_DATE")
    )
    return {
        "version": version,
        "commit": commit or None,
        "commit_short": commit[:12] if commit else None,
        "deployed_at": deployed_at,
        "process_role": process_role,
        "runtime_started_at": _runtime_started_iso(started_at),
    }


__all__ = ["get_build_info"]
