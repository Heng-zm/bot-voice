"""User preferences normalization, in-memory caching, and persistence facade."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from typing import Any

from app.services.tts.voices import (
    DEFAULT_SPEED,
    DEFAULT_TTS_MODEL,
    get_default_tts_model,
    normalize_tts_model,
)

logger = logging.getLogger(__name__)

DEFAULT_GENDER = "female"
DEFAULT_BOT_MODE = "auto"
VALID_BOT_MODES = ("auto", "tts", "ai_chat")
SPEED_MIN = 0.5
SPEED_MAX = 2.0

DEFAULT_USER_PREFS: dict[str, Any] = {
    "gender": DEFAULT_GENDER,
    "speed": DEFAULT_SPEED,
    "tts_model": DEFAULT_TTS_MODEL,
    "bot_mode": DEFAULT_BOT_MODE,
}


def _clean_user_id(user_id: Any) -> int | None:
    """Safely coerce user ID to a positive integer."""
    try:
        val = int(str(user_id).strip())
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


def normalize_user_prefs(row: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize raw user preferences dictionary with safe bounds, defaults, and identity preservation."""
    default_model = get_default_tts_model()
    prefs: dict[str, Any] = {
        "gender": DEFAULT_GENDER,
        "speed": DEFAULT_SPEED,
        "tts_model": default_model,
        "bot_mode": DEFAULT_BOT_MODE,
    }

    if not isinstance(row, dict):
        return prefs

    # Preserve user identity metadata if present
    for meta_key in ("first_name", "last_name", "username", "language_code", "user_id"):
        if meta_key in row and row[meta_key] is not None:
            prefs[meta_key] = row[meta_key]

    # Normalize gender (bilingual English + Khmer support)
    raw_gender = str(row.get("gender") or "").strip().lower()
    if raw_gender in ("female", "girl", "ស្រី"):
        prefs["gender"] = "female"
    elif raw_gender in ("male", "boy", "ប្រុស"):
        prefs["gender"] = "male"

    # Normalize speech speed
    raw_speed = row.get("speed", prefs["speed"])
    try:
        prefs["speed"] = max(SPEED_MIN, min(SPEED_MAX, float(raw_speed)))
    except (TypeError, ValueError):
        prefs["speed"] = DEFAULT_SPEED

    # Normalize TTS model
    prefs["tts_model"] = normalize_tts_model(row.get("tts_model", prefs.get("tts_model")))

    # Normalize interaction mode
    mode = str(row.get("bot_mode") or "").strip().lower()
    if mode in VALID_BOT_MODES:
        prefs["bot_mode"] = mode
    elif mode in ("ai", "chat"):
        prefs["bot_mode"] = "ai_chat"

    return prefs


class UserPrefsCache:
    """Thread-safe bounded in-memory LRU cache for user preferences."""

    def __init__(
        self,
        *,
        max_size: int = 10_000,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.max_size = max(100, int(max_size))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._cache: OrderedDict[int, tuple[dict[str, Any], float]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, user_id: Any) -> dict[str, Any] | None:
        clean_id = _clean_user_id(user_id)
        if clean_id is None:
            return None

        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(clean_id)
            if entry is None:
                return None
            prefs, cached_at = entry
            if now - cached_at >= self.ttl_seconds:
                self._cache.pop(clean_id, None)
                return None
            self._cache.move_to_end(clean_id)
            return dict(prefs)

    def set(self, user_id: Any, prefs: dict[str, Any]) -> None:
        clean_id = _clean_user_id(user_id)
        if clean_id is None:
            return

        norm = normalize_user_prefs(prefs)
        now = time.monotonic()
        with self._lock:
            self._cache.pop(clean_id, None)
            self._cache[clean_id] = (norm, now)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def update(self, user_id: Any, updates: dict[str, Any]) -> dict[str, Any]:
        """Atomically update a subset of cached preferences for a user."""
        clean_id = _clean_user_id(user_id)
        if clean_id is None:
            return normalize_user_prefs(updates)

        with self._lock:
            current = self.get(clean_id) or normalize_user_prefs(None)
            current.update(updates)
            self.set(clean_id, current)
            return current

    def invalidate(self, user_id: Any) -> None:
        clean_id = _clean_user_id(user_id)
        if clean_id is None:
            return
        with self._lock:
            self._cache.pop(clean_id, None)

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def __contains__(self, user_id: Any) -> bool:
        return self.get(user_id) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_GLOBAL_PREFS_CACHE = UserPrefsCache()


def get_global_user_prefs_cache() -> UserPrefsCache:
    """Retrieve global user preferences in-memory cache instance."""
    return _GLOBAL_PREFS_CACHE


# ============================================================================
# ASYNC PERSISTENCE AND APPLICATION INTERACTION FACADE
# ============================================================================

async def get_user_prefs_async(user_id: int | str | None) -> dict[str, Any]:
    """Retrieve normalized user preferences with fast memory-cache hit and DB fallback."""
    uid = _clean_user_id(user_id)
    if uid is None:
        return normalize_user_prefs(None)

    cache = get_global_user_prefs_cache()
    cached = cache.get(uid)
    if cached is not None:
        return cached

    # 1. Try legacy provider if present
    with suppress(Exception):
        from app import legacy

        leg_fn = getattr(legacy, "get_user_prefs_async", None)
        if callable(leg_fn) and leg_fn is not get_user_prefs_async:
            raw = await leg_fn(uid)
            if isinstance(raw, dict):
                norm = normalize_user_prefs(raw)
                cache.set(uid, norm)
                return norm

    # 2. Try SettingsStore JSON fallback
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        raw = await store.get_json(f"user_prefs:{uid}", None)
        if isinstance(raw, dict):
            norm = normalize_user_prefs(raw)
            cache.set(uid, norm)
            return norm

    # Default fallback
    default_prefs = normalize_user_prefs(None)
    cache.set(uid, default_prefs)
    return default_prefs


def get_user_prefs_sync(user_id: int | str | None) -> dict[str, Any]:
    """Synchronous memory cache read with default fallback."""
    uid = _clean_user_id(user_id)
    if uid is None:
        return normalize_user_prefs(None)

    cache = get_global_user_prefs_cache()
    cached = cache.get(uid)
    if cached is not None:
        return cached

    # Check SettingsStore sync cache
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        raw = store.get_text_sync(f"user_prefs:{uid}", "")
        if raw:
            import json

            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                norm = normalize_user_prefs(loaded)
                cache.set(uid, norm)
                return norm

    return normalize_user_prefs(None)


async def set_user_pref_async(
    user_id: int | str | None,
    key_or_dict: str | dict[str, Any],
    value: Any = None,
) -> bool:
    """Update and persist one or more preference fields for a user."""
    uid = _clean_user_id(user_id)
    if uid is None:
        return False

    current_prefs = await get_user_prefs_async(uid)
    if isinstance(key_or_dict, dict):
        current_prefs.update(key_or_dict)
    elif isinstance(key_or_dict, str):
        current_prefs[key_or_dict] = value

    norm = normalize_user_prefs(current_prefs)
    cache = get_global_user_prefs_cache()
    cache.set(uid, norm)

    # Persist via legacy if active
    persisted = False
    with suppress(Exception):
        from app import legacy

        leg_set = getattr(legacy, "set_user_pref_async", None)
        if callable(leg_set) and leg_set is not set_user_pref_async:
            await leg_set(uid, key_or_dict, value)
            persisted = True

    # Persist to SettingsStore
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        await store.set_json(f"user_prefs:{uid}", norm)
        persisted = True

    return persisted


def invalidate_user_prefs(user_id: int | str | None) -> None:
    """Invalidate memory cache entry for a user."""
    get_global_user_prefs_cache().invalidate(user_id)


__all__ = [
    "DEFAULT_BOT_MODE",
    "DEFAULT_GENDER",
    "DEFAULT_SPEED",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_USER_PREFS",
    "SPEED_MAX",
    "SPEED_MIN",
    "UserPrefsCache",
    "VALID_BOT_MODES",
    "get_global_user_prefs_cache",
    "get_user_prefs_async",
    "get_user_prefs_sync",
    "invalidate_user_prefs",
    "normalize_user_prefs",
    "set_user_pref_async",
]