"""High-performance Audio Response Caching & Deduplication.

Provides:
- Deterministic SHA-256 audio deduplication key generation (Unicode NFC normalized).
- Multi-tier Telegram voice file_id caching (L1 In-Memory LRU + L2 Redis) for 0ms voice delivery.
- In-memory bounded LRU raw audio bytes caching with metrics.
- SingleFlight request coalescing to eliminate duplicate in-flight TTS syntheses.
- User TTS history and activity tracking.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import Any, Callable, Coroutine

from app.services.tts.voices import clean_tts_text, normalize_tts_model

logger = logging.getLogger(__name__)


def normalize_tts_text_for_hash(text: str) -> str:
    """Normalize text into canonical Unicode NFC form and clean non-speech artifacts."""
    if not text:
        return ""
    # Unicode NFC normalization for consistent Khmer & multilingual combining mark ordering
    nfc = unicodedata.normalize("NFC", text)
    return clean_tts_text(nfc)


def make_tts_audio_cache_key(
    text: str,
    gender: str,
    speed: float,
    tts_model: str,
    *,
    provider_context: str = "",
    lang: str = "",
    tts_provider: str = "",
    khmer_provider: str = "",
) -> str:
    """Generate deterministic, canonical SHA-256 cache key for synthesized audio.

    Binds:
    - Canonical Unicode NFC cleaned text hash
    - Language tag
    - Normalized gender ('male' | 'female')
    - Speed rounded to 2 decimal places
    - Normalized model name
    - Active TTS engine providers
    - Optional provider context (e.g. voice clone reference signature)
    """
    cleaned = normalize_tts_text_for_hash(text)
    norm_gender = "male" if str(gender).strip().lower() == "male" else "female"
    rounded_speed = round(float(speed), 2)
    norm_model = normalize_tts_model(tts_model)

    payload: dict[str, Any] = {
        "v": 6,
        "gender": norm_gender,
        "khmer_provider": str(khmer_provider or "").strip().lower(),
        "lang": str(lang or "").strip().lower(),
        "model": norm_model,
        "provider_context": str(provider_context or "").strip(),
        "speed": rounded_speed,
        "text_hash": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
        "tts_provider": str(tts_provider or "").strip().lower(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_redis_client() -> Any | None:
    """Safely obtain the shared Redis client without tight circular import."""
    try:
        import sys
        legacy = sys.modules.get("app.legacy")
        if legacy is None:
            try:
                from app import legacy
            except Exception:
                legacy = None
        if legacy is not None:
            client = getattr(legacy, "redis_client", None)
            if client is not None:
                return client
    except Exception:
        pass
    return None


def _async_submit(fn: Callable[[], Any]) -> None:
    """Dispatch a background task to legacy db threadpool or daemon thread."""
    try:
        import sys
        legacy = sys.modules.get("app.legacy")
        submit_func = getattr(legacy, "_submit_db", None) if legacy else None
        if callable(submit_func):
            submit_func(fn)
            return
    except Exception:
        pass
    t = threading.Thread(target=fn, daemon=True)
    t.start()


class TTSFileIdCache:
    """Thread-safe multi-tier cache for Telegram voice file_ids.

    Allows instant (0ms CPU, 0 byte VPS upload) delivery of voice messages
    directly from Telegram's internal CDN when responding to matching requests.
    """

    def __init__(
        self,
        *,
        max_items: int = 10_000,
        ttl_seconds: float = 7 * 86400.0,  # 7 days
    ) -> None:
        self.max_items = max(100, int(max_items))
        self.ttl_seconds = max(60.0, float(ttl_seconds))
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    def get(self, key: str) -> str | None:
        """Lookup cached Telegram file_id from L1 Memory LRU or L2 Redis."""
        now = time.monotonic()

        # 1. L1 Memory LRU
        with self._lock:
            item = self._cache.get(key)
            if item is not None:
                file_id, created_at = item
                if now - created_at <= self.ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return file_id
                # Expired
                self._cache.pop(key, None)
                self._evictions += 1

        # 2. L2 Redis Cache
        rclient = _get_redis_client()
        if rclient is not None:
            try:
                rkey = f"audio:tgfile:{key}"
                file_id = rclient.get(rkey)
                if file_id and isinstance(file_id, str):
                    # Re-populate L1 Memory LRU
                    with self._lock:
                        self._cache[key] = (file_id, now)
                        self._cache.move_to_end(key)
                        self._hits += 1
                    return file_id
            except Exception as exc:
                logger.debug("Redis TTS file_id get error: %s", exc)

        with self._lock:
            self._misses += 1
        return None

    def set(self, key: str, file_id: str) -> None:
        """Store Telegram file_id in L1 Memory LRU and L2 Redis."""
        clean_fid = str(file_id or "").strip()
        if not key or not clean_fid:
            return

        now = time.monotonic()
        with self._lock:
            self._cache.pop(key, None)
            self._cache[key] = (clean_fid, now)
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
                self._evictions += 1

        # Asynchronously store in L2 Redis
        rclient = _get_redis_client()
        if rclient is not None:
            def _write_redis() -> None:
                try:
                    rkey = f"audio:tgfile:{key}"
                    rclient.set(rkey, clean_fid, ex=int(self.ttl_seconds))
                except Exception as exc:
                    logger.debug("Redis TTS file_id set error: %s", exc)

            _async_submit(_write_redis)

    def invalidate(self, key: str) -> bool:
        """Evict an invalid or expired Telegram file_id from all cache tiers."""
        removed = False
        with self._lock:
            if self._cache.pop(key, None) is not None:
                removed = True

        rclient = _get_redis_client()
        if rclient is not None:
            def _del_redis() -> None:
                try:
                    rclient.delete(f"audio:tgfile:{key}")
                except Exception as exc:
                    logger.debug("Redis TTS file_id del error: %s", exc)

            _async_submit(_del_redis)
        return removed

    def clear(self) -> int:
        """Clear all in-memory file_id entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def trim_expired(self) -> int:
        """Prune expired in-memory file_id entries."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            for key, (_fid, created_at) in list(self._cache.items()):
                if now - created_at > self.ttl_seconds:
                    self._cache.pop(key, None)
                    removed += 1
                    self._evictions += 1
        return removed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_reqs = self._hits + self._misses
            hit_rate = round((self._hits / total_reqs * 100), 1) if total_reqs > 0 else 0.0
            return {
                "items": len(self._cache),
                "max_items": self.max_items,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate_pct": hit_rate,
            }


class TTSSingleFlight:
    """Coalesce duplicate in-flight TTS generation requests for identical keys.

    If 5 concurrent requests arrive for the exact same text and settings:
    - 1 request performs the actual voice synthesis (the Leader).
    - 4 requests wait on the same future and receive the result directly (Followers).
    """

    def __init__(self) -> None:
        self._flights: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()

    @property
    def active_flights(self) -> int:
        return len(self._flights)

    async def execute_or_wait(
        self,
        key: str,
        coro_fn: Callable[[], Coroutine[Any, Any, Any]],
    ) -> tuple[Any, bool]:
        """Execute coro_fn if this is the first caller for key, otherwise wait for result.

        Returns (result, was_leader).
        """
        async with self._lock:
            existing = self._flights.get(key)
            if existing is not None and not existing.done():
                fut = existing
                leader = False
            else:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                self._flights[key] = fut
                leader = True

        if not leader:
            # Wait for leader to finish
            try:
                result = await asyncio.shield(fut)
                return result, False
            except Exception:
                return None, False

        # Leader executes the synthesis coroutine
        try:
            res = await coro_fn()
            if not fut.done():
                fut.set_result(res)
            return res, True
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._flights.pop(key, None)


class TTSAudioCache:
    """Thread-safe bounded in-memory LRU cache for raw audio bytes with L2 Redis support."""

    def __init__(
        self,
        *,
        max_bytes: int = 64 * 1024 * 1024,  # 64 MB default
        item_max_bytes: int = 8 * 1024 * 1024,  # 8 MB max single audio
        ttl_seconds: float = 3600.0,  # 1 hour TTL
    ) -> None:
        self.max_bytes = max(1024 * 1024, int(max_bytes))
        self.item_max_bytes = max(64 * 1024, int(item_max_bytes))
        self.ttl_seconds = max(10.0, float(ttl_seconds))
        self._cache: OrderedDict[str, tuple[bytes, float, int]] = OrderedDict()
        self._lock = threading.RLock()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    def get(self, key: str) -> bytes | None:
        now = time.monotonic()
        with self._lock:
            item = self._cache.get(key)
            if item is not None:
                data, created_at, size = item
                if now - created_at <= self.ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return bytes(data)
                self._cache.pop(key, None)
                self._current_bytes = max(0, self._current_bytes - size)

        # L2 Redis Check
        rclient = _get_redis_client()
        if rclient is not None:
            try:
                rkey = f"audio:tts:{key}"
                raw = rclient.get(rkey)
                if raw:
                    audio_data = base64.b64decode(raw.encode("ascii")) if isinstance(raw, str) else bytes(raw)
                    if audio_data:
                        self.set(key, audio_data)
                        with self._lock:
                            self._hits += 1
                        return audio_data
            except Exception as exc:
                logger.debug("Redis audio cache get error: %s", exc)

        with self._lock:
            self._misses += 1
        return None

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

        # Write to L2 Redis
        rclient = _get_redis_client()
        if rclient is not None:
            def _write_redis() -> None:
                try:
                    rkey = f"audio:tts:{key}"
                    b64_str = base64.b64encode(data).decode("ascii")
                    rclient.set(rkey, b64_str, ex=int(self.ttl_seconds))
                except Exception as exc:
                    logger.debug("Redis audio cache set error: %s", exc)

            _async_submit(_write_redis)

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

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_reqs = self._hits + self._misses
            hit_rate = round((self._hits / total_reqs * 100), 1) if total_reqs > 0 else 0.0
            return {
                "items": len(self._cache),
                "current_bytes": self._current_bytes,
                "current_mb": round(self._current_bytes / (1024 * 1024), 2),
                "max_bytes": self.max_bytes,
                "max_mb": round(self.max_bytes / (1024 * 1024), 1),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": hit_rate,
            }


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


# Global singletons
_GLOBAL_TTS_CACHE = TTSAudioCache()
_GLOBAL_TTS_FILE_ID_CACHE = TTSFileIdCache()
_GLOBAL_TTS_SINGLE_FLIGHT = TTSSingleFlight()
_GLOBAL_TTS_HISTORY = TTSUserHistoryTracker()


def get_global_tts_cache() -> TTSAudioCache:
    return _GLOBAL_TTS_CACHE


def get_global_tts_file_id_cache() -> TTSFileIdCache:
    return _GLOBAL_TTS_FILE_ID_CACHE


def get_global_tts_single_flight() -> TTSSingleFlight:
    return _GLOBAL_TTS_SINGLE_FLIGHT


def get_global_tts_history() -> TTSUserHistoryTracker:
    return _GLOBAL_TTS_HISTORY


# Module-level convenience functions for Telegram file_id cache
def get_cached_telegram_file_id(key: str) -> str | None:
    return _GLOBAL_TTS_FILE_ID_CACHE.get(key)


def set_cached_telegram_file_id(key: str, file_id: str) -> None:
    _GLOBAL_TTS_FILE_ID_CACHE.set(key, file_id)


def invalidate_cached_telegram_file_id(key: str) -> bool:
    return _GLOBAL_TTS_FILE_ID_CACHE.invalidate(key)


# Module-level convenience functions for user history
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


def clear_all_tts_caches() -> dict[str, int]:
    """Purge both in-memory TTS audio cache and Telegram file_id CDN cache."""
    cleared_fids = _GLOBAL_TTS_FILE_ID_CACHE.clear()
    cleared_audios = _GLOBAL_TTS_CACHE.clear()
    return {"file_ids_cleared": cleared_fids, "audio_items_cleared": cleared_audios}


def get_tts_cache_summary() -> dict[str, Any]:
    """Return an aggregated snapshot of all TTS cache tiers for admin dashboard."""
    fid_stats = _GLOBAL_TTS_FILE_ID_CACHE.stats()
    audio_stats = _GLOBAL_TTS_CACHE.stats()
    return {
        "file_id": fid_stats,
        "audio": audio_stats,
        "single_flight_pending": _GLOBAL_TTS_SINGLE_FLIGHT.active_flights,
    }


__all__ = [
    "TTSAudioCache",
    "TTSFileIdCache",
    "TTSSingleFlight",
    "TTSUserHistoryTracker",
    "clear_all_tts_caches",
    "clear_user_tts_history",
    "get_cached_telegram_file_id",
    "get_global_tts_cache",
    "get_global_tts_file_id_cache",
    "get_global_tts_history",
    "get_global_tts_single_flight",
    "get_last_tts",
    "get_last_tts_text",
    "get_tts_cache_summary",
    "invalidate_cached_telegram_file_id",
    "make_tts_audio_cache_key",
    "normalize_tts_text_for_hash",
    "set_cached_telegram_file_id",
    "set_last_tts",
    "set_last_tts_text",
]
