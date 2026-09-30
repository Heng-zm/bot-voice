"""Admission control for expensive Telegram workloads.

Protects the single-process runtime from bursts of OCR, transcription, AI vision,
and media processing requests. Tracks real-time telemetry, queue wait times,
and slot saturation.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

WorkloadKind = Literal["ocr", "transcribe", "audio", "vision", "media", "download"]
_T = TypeVar("_T")


def _resolve_config_int(name: str, default: int, *, minimum: int = 1, maximum: int = 64) -> int:
    """Retrieve bounded integer setting from environment or SETTINGS."""
    val = os.getenv(name)
    if val is None:
        with suppress(Exception):
            from app.core.config import SETTINGS

            val = getattr(SETTINGS, name, None)
    try:
        if val is not None:
            parsed = int(str(val).strip())
            return max(minimum, min(maximum, parsed))
    except (TypeError, ValueError):
        pass
    return default


def _resolve_config_float(name: str, default: float, *, minimum: float = 0.1, maximum: float = 300.0) -> float:
    """Retrieve bounded float setting from environment or SETTINGS."""
    val = os.getenv(name)
    if val is None:
        with suppress(Exception):
            from app.core.config import SETTINGS

            val = getattr(SETTINGS, name, None)
    try:
        if val is not None:
            parsed = float(str(val).strip())
            return max(minimum, min(maximum, parsed))
    except (TypeError, ValueError):
        pass
    return default


class WorkloadBusy(RuntimeError):
    """Raised when an edge workload exceeds queue capacity and times out."""

    def __init__(self, kind: str, timeout_s: float) -> None:
        self.kind = str(kind).lower()
        self.timeout_s = float(timeout_s)
        super().__init__(f"{self.kind} workload is busy after waiting {self.timeout_s:g}s")

    @property
    def friendly_message(self) -> str:
        """User-friendly Khmer error message for Telegram notifications."""
        return f"⏳ សេវា {self.kind.upper()} កំពុងរវល់ខ្លាំង។ សូមសាកល្បងម្ដងទៀតបន្តិចក្រោយ។"


@dataclass(slots=True)
class _Bucket:
    capacity: int
    semaphore: asyncio.Semaphore
    in_use: int = 0
    waiting: int = 0
    accepted: int = 0
    rejected: int = 0
    completed: int = 0
    failed: int = 0
    total_wait_s: float = 0.0


class TelegramWorkloadLimiter:
    """Multi-bucket adaptive concurrency controller for expensive operations."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _capacity(kind: str) -> int:
        defaults = {
            "ocr": 8,
            "transcribe": 8,
            "audio": 8,
            "vision": 8,
            "media": 8,
            "download": 6,
        }
        kind_clean = str(kind).lower().strip()
        default_cap = defaults.get(kind_clean, 8)
        env_var = f"TELEGRAM_{kind_clean.upper()}_MAX_CONCURRENT"
        return _resolve_config_int(env_var, default_cap, minimum=1, maximum=64)

    @staticmethod
    def queue_timeout_s() -> float:
        """Maximum wait time before rejecting queued requests with WorkloadBusy."""
        return _resolve_config_float(
            "TELEGRAM_WORKLOAD_QUEUE_TIMEOUT_S",
            25.0,
            minimum=0.1,
            maximum=180.0,
        )

    def _bucket(self, kind: str) -> _Bucket:
        loop = asyncio.get_running_loop()
        kind_clean = str(kind).lower().strip()

        with self._lock:
            # Rebind buckets if event loop restarted
            if self._loop is not loop:
                self._loop = loop
                self._buckets.clear()

            capacity = self._capacity(kind_clean)
            bucket = self._buckets.get(kind_clean)

            if bucket is None or (
                bucket.capacity != capacity and bucket.in_use == 0 and bucket.waiting == 0
            ):
                bucket = _Bucket(capacity=capacity, semaphore=asyncio.Semaphore(capacity))
                self._buckets[kind_clean] = bucket

            return bucket

    def available_slots(self, kind: str) -> int:
        """Return the number of slots currently free for the specified workload."""
        bucket = self._bucket(kind)
        return max(0, bucket.capacity - bucket.in_use)

    def is_busy(self, kind: str) -> bool:
        """Return True if all concurrency slots are currently in use."""
        bucket = self._bucket(kind)
        return bucket.in_use >= bucket.capacity

    @asynccontextmanager
    async def slot(self, kind: str, timeout: float | None = None) -> AsyncIterator[None]:
        """Acquire a workload slot with timeout protection and usage tracking."""
        bucket = self._bucket(kind)
        timeout_s = max(0.1, float(timeout)) if timeout is not None else self.queue_timeout_s()
        started = time.monotonic()
        acquired = False

        bucket.waiting += 1
        try:
            try:
                await asyncio.wait_for(bucket.semaphore.acquire(), timeout=timeout_s)
                acquired = True
            except TimeoutError as exc:
                bucket.rejected += 1
                raise WorkloadBusy(kind, timeout_s) from exc
        finally:
            bucket.waiting = max(0, bucket.waiting - 1)
            bucket.total_wait_s += max(0.0, time.monotonic() - started)

        bucket.in_use += 1
        bucket.accepted += 1
        try:
            yield
            bucket.completed += 1
        except Exception:
            bucket.failed += 1
            raise
        finally:
            bucket.in_use = max(0, bucket.in_use - 1)
            if acquired:
                bucket.semaphore.release()

    def snapshot(self) -> dict[str, Any]:
        """Return a structured telemetry snapshot of all workload buckets."""
        result: dict[str, Any] = {"queue_timeout_s": self.queue_timeout_s()}
        kinds = sorted(set(self._buckets.keys()) | {"ocr", "transcribe", "audio"})

        with self._lock:
            for kind in kinds:
                bucket = self._buckets.get(kind)
                capacity = self._capacity(kind)
                if bucket is None:
                    result[kind] = {
                        "capacity": capacity,
                        "in_use": 0,
                        "waiting": 0,
                        "accepted": 0,
                        "rejected": 0,
                        "completed": 0,
                        "failed": 0,
                        "avg_wait_ms": 0.0,
                    }
                    continue

                attempts = bucket.accepted + bucket.rejected
                result[kind] = {
                    "capacity": bucket.capacity,
                    "in_use": bucket.in_use,
                    "waiting": bucket.waiting,
                    "accepted": bucket.accepted,
                    "rejected": bucket.rejected,
                    "completed": bucket.completed,
                    "failed": bucket.failed,
                    "avg_wait_ms": round((bucket.total_wait_s / attempts * 1000.0), 2) if attempts else 0.0,
                }

        return result


# Global singleton instance
_LIMITER = TelegramWorkloadLimiter()


def get_telegram_workload_limiter() -> TelegramWorkloadLimiter:
    """Retrieve global workload limiter singleton."""
    return _LIMITER


async def run_telegram_workload(
    kind: WorkloadKind | str,
    factory: Callable[[], Awaitable[_T] | _T] | Awaitable[_T],
    *,
    timeout: float | None = None,
) -> _T:
    """Execute an expensive workload under slot admission control.

    Seamlessly supports coroutines, async callables, and synchronous callables.
    """
    limiter = get_telegram_workload_limiter()
    async with limiter.slot(str(kind), timeout=timeout):
        loop = asyncio.get_running_loop()
        if asyncio.iscoroutinefunction(factory):
            return await factory()  # type: ignore[return-value]

        if callable(factory):
            res = await loop.run_in_executor(None, factory)
        else:
            res = factory

        if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
            return await res

        return res  # type: ignore[return-value]


__all__ = [
    "TelegramWorkloadLimiter",
    "WorkloadBusy",
    "WorkloadKind",
    "get_telegram_workload_limiter",
    "run_telegram_workload",
]