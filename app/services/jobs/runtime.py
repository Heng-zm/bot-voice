"""Process-wide durable queue and worker lifecycle configuration."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from app.services.jobs.memory import MemoryJobQueue
from app.services.jobs.queue import (
    JobHandler,
    JobQueueError,
    RedisJobQueue,
    RedisJobWorker,
)

logger = logging.getLogger(__name__)

BOT_JOB_TYPES = frozenset(
    {
        "tts",
        "ocr",
        "transcription",
        "broadcast",
    }
)

_QUEUE: RedisJobQueue | MemoryJobQueue | None = None
_QUEUE_REDIS: Any | None = None
_LOCK = threading.RLock()
_WORKER_LOCK = asyncio.Lock()
_WORKER_STOP: asyncio.Event | None = None
_WORKER_TASKS: dict[str, asyncio.Task[None]] = {}
_WORKER_HEARTBEAT_TASKS: dict[str, asyncio.Task[None]] = {}
_WORKER_STATUS: dict[str, dict[str, Any]] = {}
_WORKER_RESTART_HISTORY: deque[dict[str, Any]] = deque(maxlen=200)
_WORKERS_ACCEPTING = True
_WORKER_RESTART_BASE_SECONDS = 1.0
_WORKER_RESTART_MAX_SECONDS = 30.0
_WORKER_STABLE_RUN_SECONDS = 60.0
_monotonic = time.monotonic


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    worker_id: str
    alive: bool
    started_at: float
    last_heartbeat_at: float
    last_error: str
    restart_count: int
    last_restart_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "alive": self.alive,
            "started_at": self.started_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_error": self.last_error,
            "restart_count": self.restart_count,
            "last_restart_at": self.last_restart_at,
        }


def configure_job_queue(
    redis_client: Any | None,
    *,
    redis_prefix: str = "tgbot",
    max_queued_jobs: int | None = None,
    memory_fallback: bool = False,
) -> RedisJobQueue | MemoryJobQueue | None:
    global _QUEUE, _QUEUE_REDIS
    with _LOCK:
        if redis_client is None:
            _QUEUE_REDIS = None
            if not memory_fallback:
                _QUEUE = None
                return None
            if not isinstance(_QUEUE, MemoryJobQueue):
                queue_limit = max_queued_jobs
                if queue_limit is None:
                    queue_limit = int(os.getenv("BOT_JOB_QUEUE_MAX", "1000") or 1000)
                _QUEUE = MemoryJobQueue(max_queued_jobs=queue_limit)
            return _QUEUE
        if _QUEUE is None or redis_client is not _QUEUE_REDIS:
            queue_limit = max_queued_jobs
            if queue_limit is None:
                queue_limit = int(os.getenv("BOT_JOB_QUEUE_MAX", "1000") or 1000)
            _QUEUE = RedisJobQueue(
                redis_client,
                redis_prefix=redis_prefix,
                max_queued_jobs=queue_limit,
            )
            _QUEUE_REDIS = redis_client
        return _QUEUE


def get_job_queue() -> RedisJobQueue | MemoryJobQueue:
    with _LOCK:
        if _QUEUE is None:
            raise JobQueueError("The durable Redis job queue is not configured.")
        return _QUEUE


async def enqueue_bot_job(
    job_type: str,
    payload: Mapping[str, Any],
    **options: Any,
):
    """Enqueue one supported workload using durable references in the payload."""

    clean_type = str(job_type or "").strip().lower()
    if clean_type not in BOT_JOB_TYPES:
        raise ValueError(f"Unsupported bot job type: {clean_type or '<empty>'}")
    return await get_job_queue().enqueue(clean_type, payload, **options)


def _worker_id(index: int) -> str:
    instance = str(os.getenv("INSTANCE_ID") or os.getenv("RENDER_INSTANCE_ID") or "").strip()
    if not instance:
        instance = f"{socket.gethostname()}-{os.getpid()}"
    return f"{instance[:80]}-jobs-{index}-{uuid.uuid4().hex[:6]}"


async def _heartbeat_loop(
    worker_id: str,
    worker_task: asyncio.Task[None],
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set() and not worker_task.done():
        status = _WORKER_STATUS.get(worker_id)
        if status is not None:
            status["last_heartbeat_at"] = time.time()
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=5.0)


def _worker_error_recorder(worker_id: str) -> Any:
    """Return a callback that stores a worker loop error for health reporting."""

    def _record(message: str) -> None:
        status = _WORKER_STATUS.get(worker_id)
        if status is not None:
            status["last_error"] = message

    return _record


async def _worker_runner(
    worker: RedisJobWorker,
    stop_event: asyncio.Event,
) -> None:
    worker_id = worker.worker_id
    started_at = time.time()
    _WORKER_STATUS[worker_id] = {
        "started_at": started_at,
        "last_heartbeat_at": started_at,
        "last_error": "",
        "restart_count": 0,
        "restart_streak": 0,
        "last_restart_at": 0.0,
    }
    while not stop_event.is_set():
        run_started = _monotonic()
        try:
            await worker.run(stop_event)
            if stop_event.is_set():
                return
            error = RuntimeError("Worker loop exited unexpectedly.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - worker process boundary
            error = exc

        status = _WORKER_STATUS[worker_id]
        run_duration = max(0.0, _monotonic() - run_started)
        if run_duration >= _WORKER_STABLE_RUN_SECONDS:
            status["restart_streak"] = 0
        status["last_error"] = f"{type(error).__name__}: {error}"[:500]
        status["restart_count"] = int(status.get("restart_count") or 0) + 1
        status["restart_streak"] = int(status.get("restart_streak") or 0) + 1
        status["last_restart_at"] = time.time()
        restart_count = int(status["restart_count"])
        restart_streak = int(status["restart_streak"])
        delay = min(
            _WORKER_RESTART_MAX_SECONDS,
            _WORKER_RESTART_BASE_SECONDS * (2 ** min(restart_streak - 1, 5)),
        )
        _WORKER_RESTART_HISTORY.appendleft(
            {
                "timestamp": time.time(),
                "worker_id": worker_id,
                "error": f"{type(error).__name__}: {error}"[:500],
                "restart_count": restart_count,
                "restart_streak": restart_streak,
                "delay_seconds": delay,
            }
        )
        logger.error(
            "Durable worker stopped unexpectedly; restarting worker=%s count=%s: %s",
            worker_id,
            restart_count,
            error,
        )
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=delay)


def set_job_workers_accepting(accepting: bool) -> bool:
    """Enable or pause claims while allowing in-flight jobs to finish."""

    global _WORKERS_ACCEPTING
    _WORKERS_ACCEPTING = bool(accepting)
    return _WORKERS_ACCEPTING


def job_workers_accepting() -> bool:
    return _WORKERS_ACCEPTING


async def start_job_workers(
    handlers: Mapping[str, JobHandler],
    *,
    worker_count: int | None = None,
) -> tuple[str, ...]:
    """Start process-local workers exactly once and return their IDs."""

    global _WORKER_STOP, _WORKERS_ACCEPTING
    async with _WORKER_LOCK:
        alive = {name: task for name, task in _WORKER_TASKS.items() if not task.done()}
        if alive:
            return tuple(alive)

        queue = get_job_queue()
        clean_handlers = {
            str(name).strip().lower(): handler
            for name, handler in handlers.items()
            if str(name).strip().lower() in BOT_JOB_TYPES
        }
        missing = sorted(BOT_JOB_TYPES - set(clean_handlers))
        if missing:
            raise ValueError(f"Missing durable job handler(s): {', '.join(missing)}")

        start_drained = str(
            os.getenv("BOT_JOB_START_DRAINED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        _WORKERS_ACCEPTING = not start_drained

        count = worker_count
        if count is None:
            count = int(os.getenv("BOT_JOB_WORKERS", "2") or 2)
        count = max(1, min(32, int(count)))
        _WORKER_STOP = asyncio.Event()
        _WORKER_TASKS.clear()
        _WORKER_HEARTBEAT_TASKS.clear()
        _WORKER_STATUS.clear()

        for index in range(1, count + 1):
            worker_id = _worker_id(index)
            worker = RedisJobWorker(
                queue,
                clean_handlers,
                worker_id=worker_id,
                poll_interval_seconds=float(
                    os.getenv("BOT_JOB_POLL_SECONDS", "0.5") or 0.5
                ),
                can_claim=job_workers_accepting,
                on_error=_worker_error_recorder(worker_id),
            )
            task = asyncio.create_task(
                _worker_runner(worker, _WORKER_STOP),
                name=f"durable-{worker.worker_id}",
            )
            heartbeat = asyncio.create_task(
                _heartbeat_loop(worker.worker_id, task, _WORKER_STOP),
                name=f"heartbeat-{worker.worker_id}",
            )
            _WORKER_TASKS[worker.worker_id] = task
            _WORKER_HEARTBEAT_TASKS[worker.worker_id] = heartbeat
        return tuple(_WORKER_TASKS)


async def stop_job_workers() -> None:
    """Request graceful shutdown, then cancel workers that do not finish."""

    global _WORKER_STOP
    async with _WORKER_LOCK:
        stop_event = _WORKER_STOP
        tasks = list(_WORKER_TASKS.values())
        heartbeats = list(_WORKER_HEARTBEAT_TASKS.values())
        if stop_event is not None:
            stop_event.set()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=10.0)
            del done
            for task in pending:
                task.cancel()
        for task in tasks + heartbeats:
            if not task.done():
                task.cancel()
        if tasks or heartbeats:
            await asyncio.gather(*tasks, *heartbeats, return_exceptions=True)
        _WORKER_TASKS.clear()
        _WORKER_HEARTBEAT_TASKS.clear()
        # Without this the snapshot keeps reporting the stopped workers as
        # dead (count=N, alive=0) after an intentional shutdown, so health
        # endpoints show a permanently unhealthy fleet.
        _WORKER_STATUS.clear()
        _WORKER_STOP = None


def job_worker_snapshot(*, stale_after_seconds: float = 20.0) -> dict[str, Any]:
    now = time.time()
    workers: list[WorkerSnapshot] = []
    for worker_id, status in sorted(_WORKER_STATUS.items()):
        task = _WORKER_TASKS.get(worker_id)
        heartbeat = float(status.get("last_heartbeat_at") or 0.0)
        alive = bool(
            task is not None
            and not task.done()
            and now - heartbeat <= max(5.0, float(stale_after_seconds))
        )
        workers.append(
            WorkerSnapshot(
                worker_id=worker_id,
                alive=alive,
                started_at=float(status.get("started_at") or 0.0),
                last_heartbeat_at=heartbeat,
                last_error=str(status.get("last_error") or ""),
                restart_count=int(status.get("restart_count") or 0),
                last_restart_at=float(status.get("last_restart_at") or 0.0),
            )
        )
    return {
        "configured": _QUEUE is not None,
        "accepting": _WORKERS_ACCEPTING,
        "count": len(workers),
        "alive": sum(1 for worker in workers if worker.alive),
        "healthy": bool(workers) and all(worker.alive for worker in workers),
        "workers": [worker.as_dict() for worker in workers],
        "restart_total": len(_WORKER_RESTART_HISTORY),
        "restart_history": list(_WORKER_RESTART_HISTORY)[:50],
    }


__all__ = [
    "BOT_JOB_TYPES",
    "WorkerSnapshot",
    "configure_job_queue",
    "enqueue_bot_job",
    "get_job_queue",
    "job_worker_snapshot",
    "job_workers_accepting",
    "set_job_workers_accepting",
    "start_job_workers",
    "stop_job_workers",
]
