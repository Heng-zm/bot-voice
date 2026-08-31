"""Fast asynchronous access to Telegram user security state."""

from __future__ import annotations

import asyncio

from app._legacy_bridge import legacy_module


async def is_user_blocked(user_id: int) -> bool:
    """Return block state without scheduling a worker for cache hits.

    The synchronous database helper owns the authoritative bounded TTL cache.
    Checking that cache on the event loop is cheap and thread-safe; only a true
    cache miss needs the bounded database executor.
    """

    clean_user_id = int(user_id)
    legacy = legacy_module()
    cached = legacy._blocked_cache_get(clean_user_id)
    if cached is not None:
        return bool(cached)
    return bool(
        await asyncio.get_running_loop().run_in_executor(
            legacy._DB_EXECUTOR,
            legacy.db_user_is_blocked,
            clean_user_id,
        )
    )


__all__ = ["is_user_blocked"]
