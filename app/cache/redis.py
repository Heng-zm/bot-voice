"""Redis connection client and distributed cache layer with graceful memory fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config.settings import SETTINGS

logger = logging.getLogger("app.cache.redis")

_REDIS_CLIENT: Any | None = None


def get_redis_client() -> Any | None:
    """Retrieve global Redis client instance if configured."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    redis_url = (os.environ.get("REDIS_URL") or getattr(SETTINGS, "REDIS_URL", "")).strip()
    if not redis_url:
        return None

    try:
        import redis

        _REDIS_CLIENT = redis.from_url(redis_url, decode_responses=True, socket_timeout=2.0)
        _REDIS_CLIENT.ping()
        logger.info("Connected to Redis successfully.")
        return _REDIS_CLIENT
    except Exception as exc:
        logger.warning("Redis connection failed, falling back to memory cache: %s", exc)
        return None


__all__ = ["get_redis_client"]
