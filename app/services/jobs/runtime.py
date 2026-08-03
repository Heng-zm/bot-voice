"""Process-wide durable queue configuration."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from app.services.jobs.queue import JobQueueError, RedisJobQueue

BOT_JOB_TYPES = frozenset(
    {
        "tts",
        "voxcpm2",
        "ocr",
        "transcription",
        "broadcast",
    }
)

_QUEUE: RedisJobQueue | None = None
_QUEUE_REDIS: Any | None = None
_LOCK = threading.RLock()


def configure_job_queue(
    redis_client: Any | None,
    *,
    redis_prefix: str = "tgbot",
) -> RedisJobQueue | None:
    global _QUEUE, _QUEUE_REDIS
    with _LOCK:
        if redis_client is None:
            _QUEUE = None
            _QUEUE_REDIS = None
            return None
        if _QUEUE is None or redis_client is not _QUEUE_REDIS:
            _QUEUE = RedisJobQueue(
                redis_client,
                redis_prefix=redis_prefix,
            )
            _QUEUE_REDIS = redis_client
        return _QUEUE


def get_job_queue() -> RedisJobQueue:
    with _LOCK:
        if _QUEUE is None:
            raise JobQueueError("The durable Redis job queue is not configured.")
        return _QUEUE


async def enqueue_bot_job(
    job_type: str,
    payload: Mapping[str, Any],
    **options: Any,
):
    """Enqueue one supported bot workload using durable references in payload."""

    clean_type = str(job_type or "").strip().lower()
    if clean_type not in BOT_JOB_TYPES:
        raise ValueError(f"Unsupported bot job type: {clean_type or '<empty>'}")
    return await get_job_queue().enqueue(clean_type, payload, **options)


__all__ = [
    "BOT_JOB_TYPES",
    "configure_job_queue",
    "enqueue_bot_job",
    "get_job_queue",
]
