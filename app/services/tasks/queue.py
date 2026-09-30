"""High-throughput background task manager and concurrency coordinator.

Provides bounded asynchronous worker pools for CPU- or I/O-intensive jobs
(media processing, batch operations, webhooks, notifications) isolating them
from Telegram event handling.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, TypeVar

logger = logging.getLogger("app.tasks.queue")

_T = TypeVar("_T")


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 1024) -> int:
    try:
        val = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        val = default
    return max(minimum, min(maximum, val))


class BackgroundTaskManager:
    """Manages asynchronous background worker tasks with bounded concurrency and drain lifecycle."""

    def __init__(self) -> None:
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._submitted_count: int = 0
        self._completed_count: int = 0
        self._failed_count: int = 0
        self._max_concurrency: int = _env_int("TASK_MANAGER_MAX_CONCURRENCY", 64, minimum=4, maximum=512)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._category_semaphores: dict[str, asyncio.Semaphore] = {
            "default": asyncio.Semaphore(_env_int("TASK_CONCURRENCY_DEFAULT", 32, minimum=2, maximum=256)),
            "media": asyncio.Semaphore(_env_int("TASK_CONCURRENCY_MEDIA", 16, minimum=1, maximum=64)),
            "batcher": asyncio.Semaphore(_env_int("TASK_CONCURRENCY_BATCHER", 8, minimum=1, maximum=32)),
            "broadcast": asyncio.Semaphore(_env_int("TASK_CONCURRENCY_BROADCAST", 12, minimum=1, maximum=64)),
        }

    def _get_category_semaphore(self, category: str) -> asyncio.Semaphore:
        sem = self._category_semaphores.get(category)
        if sem is None:
            sem = asyncio.Semaphore(16)
            self._category_semaphores[category] = sem
        return sem

    def submit(
        self,
        func_or_coro: Callable[..., Awaitable[_T]] | Awaitable[_T],
        *args: Any,
        name: str | None = None,
        category: str = "default",
        timeout: float | None = None,
        on_success: Callable[[_T], Any] | None = None,
        on_error: Callable[[Exception], Any] | None = None,
        **kwargs: Any,
    ) -> asyncio.Task[Any]:
        """Submit an async callable or coroutine for background execution."""
        task_name = name or f"bg-task-{self._submitted_count + 1}"
        self._submitted_count += 1
        cat_sem = self._get_category_semaphore(category)

        async def _wrapped_runner() -> Any:
            t0 = time.monotonic()
            async with self._semaphore:
                async with cat_sem:
                    try:
                        if inspect.iscoroutine(func_or_coro):
                            coro = func_or_coro
                        elif callable(func_or_coro):
                            coro = func_or_coro(*args, **kwargs)
                        else:
                            raise TypeError(f"Expected callable or coroutine, got {type(func_or_coro).__name__}")

                        if timeout is not None and timeout > 0:
                            result = await asyncio.wait_for(coro, timeout=timeout)
                        else:
                            result = await coro

                        self._completed_count += 1
                        if on_success:
                            with suppress(Exception):
                                if inspect.iscoroutinefunction(on_success):
                                    await on_success(result)
                                else:
                                    on_success(result)
                        return result
                    except asyncio.CancelledError:
                        logger.debug("Task %s was cancelled after %.2fs", task_name, time.monotonic() - t0)
                        raise
                    except Exception as exc:
                        self._failed_count += 1
                        logger.error("Background task %s failed: %s", task_name, exc, exc_info=True)
                        if on_error:
                            with suppress(Exception):
                                if inspect.iscoroutinefunction(on_error):
                                    await on_error(exc)
                                else:
                                    on_error(exc)
                        raise

        task = asyncio.create_task(_wrapped_runner(), name=task_name)
        self._active_tasks.add(task)
        
        # Add callback safely by ensuring it modifies the set safely.
        # asyncio callbacks are run on the loop, but wrapping prevents any cross-contamination.
        loop = asyncio.get_running_loop()
        def _on_done(t: asyncio.Task[Any]) -> None:
            loop.call_soon_threadsafe(self._active_tasks.discard, t)
            
        task.add_done_callback(_on_done)
        return task

    async def drain(self, timeout: float = 10.0) -> None:
        """Gracefully wait for active background tasks to complete before shutdown."""
        if not self._active_tasks:
            return

        logger.info("Draining %s active background tasks (timeout=%.1fs)...", len(self._active_tasks), timeout)
        t0 = time.monotonic()
        pending = set(self._active_tasks)
        try:
            done, pending = await asyncio.wait(pending, timeout=timeout)
            logger.info("Drained %s background tasks in %.2fs (%s remaining).", len(done), time.monotonic() - t0, len(pending))
        except Exception as exc:
            logger.warning("Error during background task drain wait: %s", exc)

        for task in pending:
            if not task.done():
                task.cancel()

        if pending:
            with suppress(Exception):
                await asyncio.gather(*pending, return_exceptions=True)

    def get_metrics(self) -> dict[str, Any]:
        """Return real-time task manager telemetry and health metrics."""
        return {
            "submitted": self._submitted_count,
            "completed": self._completed_count,
            "failed": self._failed_count,
            "active": len(self._active_tasks),
            "max_concurrency": self._max_concurrency,
        }


_GLOBAL_TASK_MANAGER: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Return the global BackgroundTaskManager singleton."""
    global _GLOBAL_TASK_MANAGER
    if _GLOBAL_TASK_MANAGER is None:
        _GLOBAL_TASK_MANAGER = BackgroundTaskManager()
    return _GLOBAL_TASK_MANAGER


__all__ = [
    "BackgroundTaskManager",
    "get_task_manager",
]
