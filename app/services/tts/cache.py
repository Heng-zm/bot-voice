"""In-memory LRU audio cache and user TTS history tracking."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from app.services.tts.voices import clean_tts_text, normalize_tts_model


def make_tts_audio_cache_key(
    text: str,
    gender: str,
    speed: float,
    tts_model: str,
    *,
    provider_context: str = "",
    lang: str = "",
) -> str:
    """Generate deterministic SHA-256 cache key for synthesized audio."""
    cleaned = clean_tts_text(text)
    norm_gender = "male" if str(gender).strip().lower() == "male" else "female"
    rounded_speed = round(float(speed), 2)
    norm_model = normalize_tts_model(tts_model)

    payload: dict[str, Any] = {
        "v": 5,
        "lang": lang,
        "gender": norm_gender,
        "speed": rounded_speed,
        "model": norm_model,
        "provider_context": str(provider_context or ""),
        "text_hash": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TTSAudioCache:
    """Thread-safe bounded in-memory LRU cache for audio bytes."""

    def __init__(
        self,
        *,
        max_bytes: int = 64 * 1024 * 1024,  # 64 MB default
        item_max_bytes: int = 4 * 1024 * 1024,  # 4 MB max single audio
        ttl_seconds: float = 3600.0,  # 1 hour TTL
    ) -> None:
        self.max_bytes = max(1024 * 1024, int(max_bytes))
        self.item_max_bytes = max(64 * 1024, int(item_max_bytes))
        self.ttl_seconds = max(10.0, float(ttl_seconds))
        self._cache: OrderedDict[str, tuple[bytes, float, int]] = OrderedDict()
        self._lock = threading.RLock()
        self._current_bytes = 0

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def get(self, key: str) -> bytes | None:
        now = time.monotonic()
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            data, created_at, size = item
            if now - created_at > self.ttl_seconds:
                self._cache.pop(key, None)
                self._current_bytes = max(0, self._current_bytes - size)
                return None
            self._cache.move_to_end(key)
            return bytes(data)

    def set(self, key: str, data: bytes) -> None:
        if not data:
            return
        size = len(data)
        if size > self.item_max_bytes:
            return
        now = time.monotonic()
        with self._lock:
            old = self._cache.pop(key, None)
            if old is not None:
                self._current_bytes = max(0, self._current_bytes - old[2])
            self._cache[key] = (bytes(data), now, size)
            self._current_bytes += size

            # Evict oldest entries until within max_bytes
            while self._current_bytes > self.max_bytes and self._cache:
                _old_key, (_old_data, _old_created, old_size) = self._cache.popitem(last=False)
                self._current_bytes = max(0, self._current_bytes - old_size)

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_bytes = 0
            return count

    def trim_expired(self) -> int:
        now = time.monotonic()
        removed = 0
        with self._lock:
            for key, (_data, created_at, size) in list(self._cache.items()):
                if now - created_at > self.ttl_seconds:
                    self._cache.pop(key, None)
                    self._current_bytes = max(0, self._current_bytes - size)
                    removed += 1
        return removed


class TTSUserHistoryTracker:
    """Thread-safe bounded tracker for user last TTS timestamps and text history."""

    def __init__(self, *, max_users: int = 10_000) -> None:
        self.max_users = max(100, int(max_users))
        self._last_tts: OrderedDict[int, float] = OrderedDict()
        self._last_text: OrderedDict[int, tuple[str, float]] = OrderedDict()
        self._lock = threading.RLock()

    def set_last_tts(self, user_id: int) -> None:
        clean_id = int(user_id)
        now = time.monotonic()
        with self._lock:
            self._last_tts.pop(clean_id, None)
            self._last_tts[clean_id] = now
            while len(self._last_tts) > self.max_users:
                self._last_tts.popitem(last=False)

    def get_last_tts(self, user_id: int) -> float:
        clean_id = int(user_id)
        with self._lock:
            return self._last_tts.get(clean_id, 0.0)

    def set_last_tts_text(self, user_id: int, text: str) -> None:
        clean_text = (text or "").strip()
        if not clean_text:
            return
        clean_id = int(user_id)
        now = time.monotonic()
        with self._lock:
            self._last_text.pop(clean_id, None)
            self._last_text[clean_id] = (clean_text, now)
            while len(self._last_text) > self.max_users:
                self._last_text.popitem(last=False)

    def get_last_tts_text(self, user_id: int) -> str | None:
        clean_id = int(user_id)
        with self._lock:
            item = self._last_text.get(clean_id)
            if not item:
                return None
            self._last_text.move_to_end(clean_id)
            return item[0]

    def clear_user(self, user_id: int) -> None:
        clean_id = int(user_id)
        with self._lock:
            self._last_tts.pop(clean_id, None)
            self._last_text.pop(clean_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._last_tts.clear()
            self._last_text.clear()


_GLOBAL_TTS_CACHE = TTSAudioCache()
_GLOBAL_TTS_HISTORY = TTSUserHistoryTracker()


def get_global_tts_cache() -> TTSAudioCache:
    return _GLOBAL_TTS_CACHE


def get_global_tts_history() -> TTSUserHistoryTracker:
    return _GLOBAL_TTS_HISTORY


# Module-level convenience functions
def set_last_tts(user_id: int) -> None:
    _GLOBAL_TTS_HISTORY.set_last_tts(user_id)


def get_last_tts(user_id: int) -> float:
    return _GLOBAL_TTS_HISTORY.get_last_tts(user_id)


def set_last_tts_text(user_id: int, text: str) -> None:
    _GLOBAL_TTS_HISTORY.set_last_tts_text(user_id, text)


def get_last_tts_text(user_id: int) -> str | None:
    return _GLOBAL_TTS_HISTORY.get_last_tts_text(user_id)


def clear_user_tts_history(user_id: int | None = None) -> None:
    """Clear TTS history for a specific user or all users if user_id is None."""
    if user_id is None:
        _GLOBAL_TTS_HISTORY.clear_all()
    else:
        _GLOBAL_TTS_HISTORY.clear_user(int(user_id))



__all__ = [
    "TTSAudioCache",
    "TTSUserHistoryTracker",
    "clear_user_tts_history",
    "get_global_tts_cache",
    "get_global_tts_history",
    "get_last_tts",
    "get_last_tts_text",
    "make_tts_audio_cache_key",
    "set_last_tts",
    "set_last_tts_text",
]
