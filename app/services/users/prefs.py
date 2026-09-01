"""User preferences normalization and in-memory caching."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from app.services.tts.voices import (
    DEFAULT_SPEED,
    DEFAULT_TTS_MODEL,
    get_default_tts_model,
    normalize_tts_model,
)

DEFAULT_GENDER = "female"
SPEED_MIN = 0.5
SPEED_MAX = 2.0

DEFAULT_USER_PREFS: dict[str, Any] = {
    "gender": DEFAULT_GENDER,
    "speed": DEFAULT_SPEED,
    "tts_model": DEFAULT_TTS_MODEL,
}


def normalize_user_prefs(row: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize raw user preferences dictionary with safe bounds and defaults."""
    default_model = get_default_tts_model()
    prefs = {
        "gender": DEFAULT_GENDER,
        "speed": DEFAULT_SPEED,
        "tts_model": default_model,
    }
    if not isinstance(row, dict):
        return prefs

    gender = str(row.get("gender") or "").strip().lower()
    if gender in ("female", "male"):
        prefs["gender"] = gender

    raw_speed = row.get("speed", prefs["speed"])
    try:
        prefs["speed"] = max(SPEED_MIN, min(SPEED_MAX, float(raw_speed)))
    except (TypeError, ValueError):
        prefs["speed"] = DEFAULT_SPEED

    prefs["tts_model"] = normalize_tts_model(row.get("tts_model", prefs.get("tts_model")))
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

    def get(self, user_id: int) -> dict[str, Any] | None:
        clean_id = int(user_id)
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

    def set(self, user_id: int, prefs: dict[str, Any]) -> None:
        clean_id = int(user_id)
        norm = normalize_user_prefs(prefs)
        now = time.monotonic()
        with self._lock:
            self._cache.pop(clean_id, None)
            self._cache[clean_id] = (norm, now)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, user_id: int) -> None:
        clean_id = int(user_id)
        with self._lock:
            self._cache.pop(clean_id, None)

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count


_GLOBAL_PREFS_CACHE = UserPrefsCache()


def get_global_user_prefs_cache() -> UserPrefsCache:
    return _GLOBAL_PREFS_CACHE


__all__ = [
    "DEFAULT_GENDER",
    "DEFAULT_USER_PREFS",
    "SPEED_MAX",
    "SPEED_MIN",
    "UserPrefsCache",
    "get_global_user_prefs_cache",
    "normalize_user_prefs",
]
