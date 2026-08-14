"""Central resource budgets for small bot deployments.

The efficient profile is intentionally the default: it keeps enough concurrency
for interactive Telegram/TTS traffic without reserving large pools and caches on
a small server.  Operators can opt into ``balanced`` or ``performance`` through
``BOT_RESOURCE_PROFILE`` when the host has more CPU and memory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

_PROFILE_ALIASES = {
    "": "efficient",
    "auto": "efficient",
    "small": "efficient",
    "safe": "efficient",
    "efficient": "efficient",
    "balanced": "balanced",
    "normal": "balanced",
    "high": "performance",
    "fast": "performance",
    "performance": "performance",
}

_EFFICIENT_DEFAULTS: dict[str, int | float] = {
    "TELEGRAM_CONCURRENT_UPDATES": 4,
    "TELEGRAM_CONNECTION_POOL_SIZE": 12,
    "HTTP_MAX_CONNECTIONS": 50,
    "HTTP_MAX_KEEPALIVE_CONNECTIONS": 12,
    "DB_EXECUTOR_MAX_WORKERS": 3,
    "MAX_CONCURRENT_TTS_USERS": 2,
    "MAX_CONCURRENT_AI": 2,
    "MAX_CONCURRENT_GEMINI": 2,
    "MAX_CONCURRENT_BROADCAST": 2,
    "WEB_BROADCAST_WORKERS": 2,
    "TTS_AUDIO_CACHE_MAX_MB": 32,
    "TTS_AUDIO_CACHE_ITEM_MAX_MB": 8,
    "EDGE_TTS_PARALLEL_CHUNKS": 1,
    "GRADIO_CLIENT_MAX_WORKERS": 2,
    "PROVIDER_SYNC_MAX_WORKERS": 2,
    "PROVIDER_SYNC_MAX_INFLIGHT": 4,
    "BOT_JOB_WORKERS": 2,
    "BOT_ARTIFACT_CLEANUP_SECONDS": 900,
    "BOT_ARTIFACT_CLEANUP_LIMIT": 100,
    "PREFS_CACHE_MAX_SIZE": 3_000,
    "PREFS_LOAD_LOCKS_MAX": 1_000,
    "USER_SYNC_MAX": 5_000,
    "TEXT_CACHE_MEMORY_MAX": 5_000,
    "BLOCKED_USER_CACHE_MAX": 5_000,
    "HISTORY_CACHE_MAX_USERS": 1_500,
    "AI_API_KEY_CACHE_MAX": 2_000,
}

# Hot settings restored from Supabase/Redis are capped only in efficient mode.
# This makes the resource choice effective even on deployments that persisted
# the older, larger defaults through the admin panel.
_EFFICIENT_CAPS = {
    key: _EFFICIENT_DEFAULTS[key]
    for key in (
        "TELEGRAM_CONCURRENT_UPDATES",
        "TELEGRAM_CONNECTION_POOL_SIZE",
        "HTTP_MAX_CONNECTIONS",
        "HTTP_MAX_KEEPALIVE_CONNECTIONS",
        "DB_EXECUTOR_MAX_WORKERS",
        "MAX_CONCURRENT_TTS_USERS",
        "MAX_CONCURRENT_AI",
        "MAX_CONCURRENT_GEMINI",
        "MAX_CONCURRENT_BROADCAST",
        "WEB_BROADCAST_WORKERS",
        "TTS_AUDIO_CACHE_MAX_MB",
        "TTS_AUDIO_CACHE_ITEM_MAX_MB",
        "EDGE_TTS_PARALLEL_CHUNKS",
        "GRADIO_CLIENT_MAX_WORKERS",
        "PROVIDER_SYNC_MAX_WORKERS",
        "PROVIDER_SYNC_MAX_INFLIGHT",
        "BOT_JOB_WORKERS",
        "BOT_ARTIFACT_CLEANUP_LIMIT",
        "PREFS_CACHE_MAX_SIZE",
        "PREFS_LOAD_LOCKS_MAX",
        "USER_SYNC_MAX",
        "TEXT_CACHE_MEMORY_MAX",
        "BLOCKED_USER_CACHE_MAX",
        "HISTORY_CACHE_MAX_USERS",
        "AI_API_KEY_CACHE_MAX",
    )
}


def resource_profile(environ: Mapping[str, str] | None = None) -> str:
    """Return the normalized server resource profile."""

    source = os.environ if environ is None else environ
    raw = str(source.get("BOT_RESOURCE_PROFILE", "efficient") or "efficient")
    return _PROFILE_ALIASES.get(raw.strip().lower(), "efficient")


def resource_default(
    name: str,
    fallback: Any,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Return a resource-conscious default for the active profile."""

    if resource_profile(environ) == "efficient":
        return _EFFICIENT_DEFAULTS.get(name, fallback)
    return fallback


def resource_value(
    name: str,
    value: Any,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Apply an efficient-profile upper bound while preserving value type."""

    if resource_profile(environ) != "efficient" or name not in _EFFICIENT_CAPS:
        return value
    cap = _EFFICIENT_CAPS[name]
    try:
        limited = min(value, cap)
    except TypeError:
        limited = min(float(value), float(cap))
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(limited)
    if isinstance(value, float):
        return float(limited)
    return limited
