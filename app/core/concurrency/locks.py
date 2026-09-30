"""Asynchronous and threading locking utilities for per-user and per-resource concurrency control."""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncGenerator

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_MASTER_LOCK = threading.Lock()


def get_keyed_lock(key: str) -> threading.Lock:
    """Return a thread-safe Lock for a specific resource key."""
    with _THREAD_MASTER_LOCK:
        if key not in _THREAD_LOCKS:
            _THREAD_LOCKS[key] = threading.Lock()
        return _THREAD_LOCKS[key]


class ResourceLockManager:
    """Coroutine-safe keyed locking mechanism."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._master_lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncGenerator[None, None]:
        """Acquire a named lock by key."""
        async with self._master_lock:
            lock = self._locks[key]

        async with lock:
            try:
                yield
            finally:
                async with self._master_lock:
                    if not lock.locked():
                        self._locks.pop(key, None)


_USER_LOCKS = ResourceLockManager()


def get_user_lock_manager() -> ResourceLockManager:
    return _USER_LOCKS


__all__ = [
    "ResourceLockManager",
    "get_keyed_lock",
    "get_user_lock_manager",
]
