"""Backward compatibility shim for app.services.tasks.queue forwarding to app.core.concurrency.queue."""

from __future__ import annotations

from app.core.concurrency.queue import (
    BackgroundTaskManager,
    get_task_manager,
)

__all__ = [
    "BackgroundTaskManager",
    "get_task_manager",
]
