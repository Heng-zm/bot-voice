"""Generic in-memory thread-safe LRU cache with TTL expiration."""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class MemoryCache(Generic[T]):
    """Thread-safe in-memory LRU cache with TTL."""

    def __init__(self, max_size: int = 1000, default_ttl_s: float = 300.0) -> None:
        self.max_size = max(10, int(max_size))
        self.default_ttl_s = float(default_ttl_s)
        self._cache: collections.OrderedDict[str, tuple[T, float]] = collections.OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str, default: T | None = None) -> T | None:
        now = time.monotonic()
        with self._lock:
            if key not in self._cache:
                return default
            val, expiry = self._cache[key]
            if now > expiry:
                self._cache.pop(key, None)
                return default
            self._cache.move_to_end(key)
            return val

    def set(self, key: str, value: T, ttl_s: float | None = None) -> None:
        ttl = self.default_ttl_s if ttl_s is None else float(ttl_s)
        expiry = time.monotonic() + ttl
        with self._lock:
            if key in self._cache:
                self._cache.pop(key, None)
            self._cache[key] = (value, expiry)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


__all__ = ["MemoryCache"]
