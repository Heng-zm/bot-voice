"""Async retry utilities with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger("app.utils.retry")

T = TypeVar("T")


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to retry an async function with exponential backoff and jitter."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.warning(
                            "Function %s failed after %d retries: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
                        raise
                    jitter = random.uniform(0.8, 1.2)  # noqa: S311
                    sleep_time = min(max_delay, delay * jitter)
                    logger.debug(
                        "Retry attempt %d/%d for %s after %.2fs due to %s",
                        attempt,
                        max_retries,
                        func.__name__,
                        sleep_time,
                        exc,
                    )
                    await asyncio.sleep(sleep_time)
                    delay *= exponential
            if last_exc:
                raise last_exc
            return None

        return wrapper

    return decorator


__all__ = ["async_retry"]
