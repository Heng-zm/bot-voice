"""Core concurrency dispatcher and bounded async execution engine."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class ConcurrencyDispatcher:
    """Dispatches tasks with bounded concurrency."""

    def __init__(self, max_concurrency: int = 32) -> None:
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, coro_fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        async with self._semaphore:
            return await coro_fn(*args, **kwargs)


__all__ = ["ConcurrencyDispatcher"]
