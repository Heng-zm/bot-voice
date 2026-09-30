"""Telegram session state and user interaction caching."""

from __future__ import annotations

import time
from typing import Any

from app.cache.memory import MemoryCache

_USER_STATE_CACHE = MemoryCache[dict[str, Any]](max_size=5000, default_ttl_s=1800.0)


def get_user_state(user_id: int) -> dict[str, Any]:
    return _USER_STATE_CACHE.get(str(user_id), {}) or {}


def set_user_state(user_id: int, state: dict[str, Any], ttl_s: float = 1800.0) -> None:
    _USER_STATE_CACHE.set(str(user_id), state, ttl_s=ttl_s)


def clear_user_state(user_id: int) -> None:
    _USER_STATE_CACHE.delete(str(user_id))


__all__ = ["clear_user_state", "get_user_state", "set_user_state"]
