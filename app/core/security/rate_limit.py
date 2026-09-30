"""In-memory and Redis rate limiting utilities for APIs and Telegram interactions."""

from __future__ import annotations

import collections
import time


class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._history: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        history = self._history[key]

        cutoff = now - self.window_seconds
        while history and history[0] < cutoff:
            history.popleft()

        if len(history) < self.limit:
            history.append(now)
            return True

        return False

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        history = self._history[key]
        cutoff = now - self.window_seconds
        while history and history[0] < cutoff:
            history.popleft()
        return max(0, self.limit - len(history))

    def reset(self, key: str) -> None:
        self._history.pop(key, None)


_GLOBAL_LIMITERS: dict[tuple[int, float], SlidingWindowRateLimiter] = {}


def _get_limiter(limit: int, window_seconds: float) -> SlidingWindowRateLimiter:
    key = (limit, window_seconds)
    if key not in _GLOBAL_LIMITERS:
        _GLOBAL_LIMITERS[key] = SlidingWindowRateLimiter(limit=limit, window_seconds=window_seconds)
    return _GLOBAL_LIMITERS[key]


def check_rate_limit(key: str, max_requests: int = 60, window_seconds: float = 60.0) -> bool:
    """Return True if allowed, False if rate limit exceeded."""
    limiter = _get_limiter(max_requests, window_seconds)
    return limiter.is_allowed(key)


def is_rate_limited(key: str, max_requests: int = 60, window_seconds: float = 60.0) -> bool:
    """Return True if rate limit exceeded, False if allowed."""
    return not check_rate_limit(key, max_requests=max_requests, window_seconds=window_seconds)


__all__ = [
    "SlidingWindowRateLimiter",
    "check_rate_limit",
    "is_rate_limited",
]
