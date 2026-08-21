"""Health-aware routing for AI and speech providers.

Provider health is intentionally process-local.  The administrative API exposes
``scope=process`` and an instance identifier so a multi-instance deployment does
not accidentally present one worker's circuit-breaker state as cluster-wide.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import socket
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")
ProviderOperation = Callable[[str], T | Awaitable[T]]


class ProviderManagerError(RuntimeError):
    """Base provider routing error."""


class ProviderTimeout(ProviderManagerError):
    """Raised when a synchronous provider exceeds its configured timeout."""


class ProviderBusy(ProviderManagerError):
    """Raised when the bounded synchronous provider executor is saturated."""


class NoProviderAvailable(ProviderManagerError):
    """Raised after every eligible provider is unavailable or has failed."""

    def __init__(self, capability: str, errors: dict[str, str]) -> None:
        self.capability = capability
        self.errors = dict(errors)
        detail = "; ".join(f"{name}: {error}" for name, error in errors.items())
        super().__init__(
            f"No provider completed capability {capability!r}"
            + (f": {detail}" if detail else ".")
        )


@dataclass(frozen=True, slots=True)
class ProviderPolicy:
    name: str
    capabilities: frozenset[str]
    priority: int = 100
    failure_threshold: int = 3
    cooldown_seconds: float = 300.0
    timeout_seconds: float = 60.0


@dataclass(slots=True)
class ProviderState:
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    disabled_until: float = 0.0
    latency_ewma_ms: float | None = None
    last_latency_ms: float | None = None
    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_error: str = ""


@dataclass(slots=True)
class _ProviderEntry:
    policy: ProviderPolicy
    state: ProviderState = field(default_factory=ProviderState)


def _default_instance_id() -> str:
    configured = str(os.getenv("INSTANCE_ID") or os.getenv("RENDER_INSTANCE_ID") or "").strip()
    if configured:
        return configured[:128]
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class ProviderManager:
    """Route capabilities using priority, timeouts, circuit breakers, and health.

    Async calls are bounded with :func:`asyncio.wait_for`.  Synchronous calls
    run in a bounded executor and use ``Future.result(timeout=...)``.  Python
    cannot forcibly terminate a thread already executing foreign SDK code, so a
    timed-out call may finish in the background; the bounded in-flight semaphore
    prevents those calls from growing without limit.
    """

    def __init__(
        self,
        *,
        sync_max_workers: int = 4,
        sync_max_inflight: int | None = None,
        instance_id: str = "",
    ) -> None:
        workers = max(1, min(32, int(sync_max_workers)))
        inflight = (
            max(workers, min(128, int(sync_max_inflight)))
            if sync_max_inflight is not None
            else workers * 2
        )
        self._providers: dict[str, _ProviderEntry] = {}
        self._lock = threading.RLock()
        self._sync_executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="provider-sync",
        )
        self._sync_slots = threading.BoundedSemaphore(inflight)
        self._sync_max_workers = workers
        self._sync_max_inflight = inflight
        self._closed = False
        self.instance_id = str(instance_id or _default_instance_id())[:128]

    @staticmethod
    def _name(value: str) -> str:
        name = str(value or "").strip().lower()
        if not name or len(name) > 64:
            raise ValueError("Provider name is missing or too long.")
        return name

    @staticmethod
    def _capability(value: str) -> str:
        capability = str(value or "").strip().lower()
        if not capability or len(capability) > 64:
            raise ValueError("Provider capability is missing or too long.")
        return capability

    def register(
        self,
        name: str,
        *,
        capabilities: Iterable[str],
        priority: int = 100,
        failure_threshold: int = 3,
        cooldown_seconds: float = 300.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        provider_name = self._name(name)
        clean_capabilities = frozenset(
            self._capability(capability) for capability in capabilities
        )
        if not clean_capabilities:
            raise ValueError("A provider must expose at least one capability.")
        policy = ProviderPolicy(
            provider_name,
            clean_capabilities,
            max(0, int(priority)),
            max(1, int(failure_threshold)),
            max(1.0, float(cooldown_seconds)),
            max(0.01, float(timeout_seconds)),
        )
        with self._lock:
            existing = self._providers.get(provider_name)
            state = existing.state if existing is not None else ProviderState()
            self._providers[provider_name] = _ProviderEntry(policy, state)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._providers.pop(self._name(name), None)

    @staticmethod
    def _health_score(entry: _ProviderEntry, now: float) -> float:
        state = entry.state
        if now < state.disabled_until:
            return 0.0
        total = state.successes + state.failures
        success_ratio = state.successes / total if total else 1.0
        failure_penalty = min(40.0, state.consecutive_failures * 12.0)
        latency_penalty = min(20.0, (state.latency_ewma_ms or 0.0) / 500.0)
        return round(
            max(1.0, 100.0 * success_ratio - failure_penalty - latency_penalty),
            2,
        )

    def ordered(
        self,
        capability: str,
        *,
        preferred: Iterable[str] = (),
    ) -> list[str]:
        requested = self._capability(capability)
        preferred_order = [
            self._name(name)
            for name in preferred
            if str(name or "").strip()
        ]
        preferred_rank = {
            name: index for index, name in enumerate(dict.fromkeys(preferred_order))
        }
        now = time.monotonic()
        with self._lock:
            candidates = [
                entry
                for entry in self._providers.values()
                if requested in entry.policy.capabilities
                and now >= entry.state.disabled_until
            ]
            candidates.sort(
                key=lambda entry: (
                    preferred_rank.get(
                        entry.policy.name,
                        len(preferred_rank) + entry.policy.priority,
                    ),
                    entry.policy.priority,
                    -self._health_score(entry, now),
                    entry.policy.name,
                )
            )
            return [entry.policy.name for entry in candidates]

    def record_success(self, name: str, latency_ms: float) -> None:
        provider_name = self._name(name)
        latency = max(0.0, float(latency_ms))
        with self._lock:
            entry = self._providers.get(provider_name)
            if entry is None:
                raise KeyError(f"Unknown provider: {provider_name}")
            state = entry.state
            state.successes += 1
            state.consecutive_failures = 0
            state.disabled_until = 0.0
            state.last_latency_ms = latency
            state.latency_ewma_ms = (
                latency
                if state.latency_ewma_ms is None
                else (0.25 * latency) + (0.75 * state.latency_ewma_ms)
            )
            state.last_success_at = time.time()
            state.last_error = ""

    def record_failure(self, name: str, error: BaseException | str) -> None:
        provider_name = self._name(name)
        with self._lock:
            entry = self._providers.get(provider_name)
            if entry is None:
                raise KeyError(f"Unknown provider: {provider_name}")
            state = entry.state
            state.failures += 1
            state.consecutive_failures += 1
            state.last_failure_at = time.time()
            state.last_error = str(error)[:500]
            if state.consecutive_failures >= entry.policy.failure_threshold:
                state.disabled_until = max(
                    state.disabled_until,
                    time.monotonic() + entry.policy.cooldown_seconds,
                )

    def reset(self, name: str) -> None:
        provider_name = self._name(name)
        with self._lock:
            entry = self._providers.get(provider_name)
            if entry is None:
                raise KeyError(f"Unknown provider: {provider_name}")
            entry.state = ProviderState()

    async def execute(
        self,
        capability: str,
        operation: ProviderOperation[T],
        *,
        preferred: Iterable[str] = (),
    ) -> tuple[T, str]:
        errors: dict[str, str] = {}
        providers = self.ordered(capability, preferred=preferred)
        for provider_name in providers:
            with self._lock:
                entry = self._providers[provider_name]
                timeout = entry.policy.timeout_seconds
            started = time.perf_counter()
            try:

                async def invoke(selected: str = provider_name) -> T:
                    if inspect.iscoroutinefunction(operation):
                        return await operation(selected)
                    value = await asyncio.to_thread(operation, selected)
                    if inspect.isawaitable(value):
                        return await value
                    return value

                result = await asyncio.wait_for(invoke(), timeout=timeout)
                latency_ms = (time.perf_counter() - started) * 1_000
                self.record_success(provider_name, latency_ms)
                return result, provider_name
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - provider boundary
                self.record_failure(provider_name, exc)
                errors[provider_name] = f"{type(exc).__name__}: {exc}"[:500]
        raise NoProviderAvailable(self._capability(capability), errors)

    def _submit_sync(
        self,
        operation: Callable[[str], T],
        provider_name: str,
        timeout: float,
    ) -> Future[T]:
        with self._lock:
            if self._closed:
                raise ProviderManagerError("Provider manager is closed.")
        acquired = self._sync_slots.acquire(timeout=min(1.0, timeout))
        if not acquired:
            raise ProviderBusy(
                "The bounded synchronous provider executor is saturated."
            )
        try:
            future = self._sync_executor.submit(operation, provider_name)
        except BaseException:
            self._sync_slots.release()
            raise
        future.add_done_callback(lambda _future: self._sync_slots.release())
        return future

    def execute_sync(
        self,
        capability: str,
        operation: Callable[[str], T],
        *,
        preferred: Iterable[str] = (),
    ) -> tuple[T, str]:
        """Run synchronous providers with the same policy timeout as async calls."""

        errors: dict[str, str] = {}
        for provider_name in self.ordered(capability, preferred=preferred):
            with self._lock:
                timeout = self._providers[provider_name].policy.timeout_seconds
            started = time.perf_counter()
            future: Future[T] | None = None
            try:
                future = self._submit_sync(operation, provider_name, timeout)
                result = future.result(timeout=timeout)
                if inspect.isawaitable(result):
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
                    raise TypeError(
                        "execute_sync provider operations must not return awaitables."
                    )
                latency_ms = (time.perf_counter() - started) * 1_000
                self.record_success(provider_name, latency_ms)
                return result, provider_name
            except FutureTimeoutError:
                if future is not None:
                    future.cancel()
                error = ProviderTimeout(
                    f"Provider {provider_name!r} timed out after {timeout:g} seconds."
                )
                self.record_failure(provider_name, error)
                errors[provider_name] = f"ProviderTimeout: {error}"[:500]
            except Exception as exc:  # noqa: BLE001 - provider boundary
                self.record_failure(provider_name, exc)
                errors[provider_name] = f"{type(exc).__name__}: {exc}"[:500]
        raise NoProviderAvailable(self._capability(capability), errors)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now_monotonic = time.monotonic()
        with self._lock:
            return {
                name: {
                    "scope": "process",
                    "instance_id": self.instance_id,
                    "capabilities": sorted(entry.policy.capabilities),
                    "priority": entry.policy.priority,
                    "failure_threshold": entry.policy.failure_threshold,
                    "cooldown_seconds": entry.policy.cooldown_seconds,
                    "timeout_seconds": entry.policy.timeout_seconds,
                    "available": now_monotonic >= entry.state.disabled_until,
                    "cooldown_remaining_seconds": round(
                        max(0.0, entry.state.disabled_until - now_monotonic),
                        3,
                    ),
                    "health_score": self._health_score(entry, now_monotonic),
                    "successes": entry.state.successes,
                    "failures": entry.state.failures,
                    "consecutive_failures": entry.state.consecutive_failures,
                    "latency_ewma_ms": (
                        round(entry.state.latency_ewma_ms, 3)
                        if entry.state.latency_ewma_ms is not None
                        else None
                    ),
                    "last_latency_ms": (
                        round(entry.state.last_latency_ms, 3)
                        if entry.state.last_latency_ms is not None
                        else None
                    ),
                    "last_success_at": entry.state.last_success_at,
                    "last_failure_at": entry.state.last_failure_at,
                    "last_error": entry.state.last_error,
                }
                for name, entry in sorted(self._providers.items())
            }

    def metadata(self) -> dict[str, Any]:
        return {
            "scope": "process",
            "instance_id": self.instance_id,
            "sync_max_workers": self._sync_max_workers,
            "sync_max_inflight": self._sync_max_inflight,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._sync_executor.shutdown(wait=False, cancel_futures=True)


_PROVIDER_MANAGER = ProviderManager(
    sync_max_workers=int(os.getenv("PROVIDER_SYNC_MAX_WORKERS", "4") or 4),
    sync_max_inflight=int(os.getenv("PROVIDER_SYNC_MAX_INFLIGHT", "8") or 8),
)


def configure_default_providers() -> ProviderManager:
    defaults = (
        ("gemini", {"ai", "ocr", "transcription"}, 10, 90.0),
        ("huggingface", {"ai", "ocr", "tts"}, 20, 120.0),
        ("edge_tts", {"tts"}, 30, 60.0),
        ("voxcpm2", {"tts", "voice_clone"}, 10, 300.0),
    )
    for name, capabilities, priority, timeout in defaults:
        _PROVIDER_MANAGER.register(
            name,
            capabilities=capabilities,
            priority=priority,
            failure_threshold=3,
            cooldown_seconds=300.0,
            timeout_seconds=timeout,
        )
    return _PROVIDER_MANAGER


def get_provider_manager() -> ProviderManager:
    return _PROVIDER_MANAGER


configure_default_providers()


__all__ = [
    "NoProviderAvailable",
    "ProviderBusy",
    "ProviderManager",
    "ProviderManagerError",
    "ProviderPolicy",
    "ProviderState",
    "ProviderTimeout",
    "configure_default_providers",
    "get_provider_manager",
]
