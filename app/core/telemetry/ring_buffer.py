"""Thread-safe generic ring buffer for telemetry, log streaming, and event buffering."""

from __future__ import annotations

import collections
import threading
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Thread-safe FIFO circular buffer with fixed maximum capacity."""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self._buffer: collections.deque[T] = collections.deque(maxlen=capacity)
        self._lock = threading.RLock()

    def append(self, item: T) -> None:
        """Add an item to the ring buffer. Evicts oldest item if at capacity."""
        with self._lock:
            self._buffer.append(item)

    def extend(self, items: Iterator[T] | list[T]) -> None:
        """Add multiple items to the ring buffer."""
        with self._lock:
            self._buffer.extend(items)

    def get_all(self) -> list[T]:
        """Return a snapshot list of all items from oldest to newest."""
        with self._lock:
            return list(self._buffer)

    def get_latest(self, n: int) -> list[T]:
        """Return up to `n` most recent items in chronological order."""
        with self._lock:
            if n <= 0:
                return []
            items = list(self._buffer)
            return items[-n:]

    def clear(self) -> None:
        """Clear all contents from the ring buffer."""
        with self._lock:
            self._buffer.clear()

    def get_entries(self) -> list[T]:
        """Alias for get_all."""
        return self.get_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __iter__(self) -> Iterator[T]:
        with self._lock:
            return iter(list(self._buffer))


TelemetryRingBuffer = RingBuffer

__all__ = ["RingBuffer", "TelemetryRingBuffer"]
