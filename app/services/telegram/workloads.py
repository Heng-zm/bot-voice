"""Admission control for expensive Telegram workloads.

This protects the single-process runtime from bursts of OCR/transcription/media
requests.  TTS already has its own legacy semaphore and per-user reservation;
these slots cover the remaining expensive handler classes at the Telegram edge.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, TypeVar

WorkloadKind = Literal["ocr", "transcribe", "audio"]
_T = TypeVar("_T")


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 32) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


class WorkloadBusy(RuntimeError):
    def __init__(self, kind: WorkloadKind, timeout_s: float) -> None:
        self.kind = kind
        self.timeout_s = timeout_s
        super().__init__(f"{kind} workload is busy after waiting {timeout_s:g}s")


@dataclass(slots=True)
class _Bucket:
    capacity: int
    semaphore: asyncio.Semaphore
    in_use: int = 0
    waiting: int = 0
    accepted: int = 0
    rejected: int = 0
    total_wait_s: float = 0.0


class TelegramWorkloadLimiter:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._buckets: dict[WorkloadKind, _Bucket] = {}

    @staticmethod
    def _capacity(kind: WorkloadKind) -> int:
        defaults = {"ocr": 2, "transcribe": 2, "audio": 2}
        names = {
            "ocr": "TELEGRAM_OCR_MAX_CONCURRENT",
            "transcribe": "TELEGRAM_TRANSCRIBE_MAX_CONCURRENT",
            "audio": "TELEGRAM_AUDIO_MAX_CONCURRENT",
        }
        return _env_int(names[kind], defaults[kind], minimum=1, maximum=16)

    @staticmethod
    def queue_timeout_s() -> float:
        return _env_float(
            "TELEGRAM_WORKLOAD_QUEUE_TIMEOUT_S",
            6.0,
            minimum=0.1,
            maximum=60.0,
        )

    def _bucket(self, kind: WorkloadKind) -> _Bucket:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._buckets.clear()
        capacity = self._capacity(kind)
        bucket = self._buckets.get(kind)
        if bucket is None or (bucket.capacity != capacity and bucket.in_use == 0 and bucket.waiting == 0):
            bucket = _Bucket(capacity=capacity, semaphore=asyncio.Semaphore(capacity))
            self._buckets[kind] = bucket
        return bucket

    @asynccontextmanager
    async def slot(self, kind: WorkloadKind) -> AsyncIterator[None]:
        bucket = self._bucket(kind)
        timeout_s = self.queue_timeout_s()
        started = time.monotonic()
        bucket.waiting += 1
        try:
            try:
                await asyncio.wait_for(bucket.semaphore.acquire(), timeout=timeout_s)
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
        finally:
            bucket.in_use = max(0, bucket.in_use - 1)
            bucket.semaphore.release()

    def snapshot(self) -> dict[str, object]:
        result: dict[str, object] = {"queue_timeout_s": self.queue_timeout_s()}
        for kind in ("ocr", "transcribe", "audio"):
            bucket = self._buckets.get(kind)
            capacity = self._capacity(kind)
            if bucket is None:
                result[kind] = {
                    "capacity": capacity,
                    "in_use": 0,
                    "waiting": 0,
                    "accepted": 0,
                    "rejected": 0,
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
                "avg_wait_ms": round((bucket.total_wait_s / attempts * 1000.0), 2) if attempts else 0.0,
            }
        return result


_LIMITER = TelegramWorkloadLimiter()


def get_telegram_workload_limiter() -> TelegramWorkloadLimiter:
    return _LIMITER


async def run_telegram_workload(
    kind: WorkloadKind,
    factory: Callable[[], Awaitable[_T]],
) -> _T:
    limiter = get_telegram_workload_limiter()
    async with limiter.slot(kind):
        return await factory()


__all__ = [
    "TelegramWorkloadLimiter",
    "WorkloadBusy",
    "get_telegram_workload_limiter",
    "run_telegram_workload",
]
