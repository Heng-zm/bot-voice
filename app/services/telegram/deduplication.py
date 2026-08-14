"""Telegram webhook replay protection with Redis and memory fallback.

This module owns the complete update-claim lifecycle.  The legacy runtime only
configures its Redis client provider and key namespace, which keeps existing
call sites compatible while removing the implementation from ``app.legacy``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

ClaimResult = str | tuple[str, str | None]
MemoryRecord = tuple[str, float, str | None]
RedisProvider = Callable[[], Any | None]
ReplayKeyBuilder = Callable[[int], str]


def _bounded_env_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(str(environ.get(name, default) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


class WebhookUpdateDeduplicator:
    """Atomically claim, complete, and release Telegram webhook updates."""

    def __init__(
        self,
        *,
        redis_client_provider: RedisProvider | None = None,
        replay_key_builder: ReplayKeyBuilder | None = None,
        environ: Mapping[str, str] | None = None,
        max_memory_updates: int = 50_000,
    ) -> None:
        self._redis_client_provider = redis_client_provider or (lambda: None)
        self._replay_key_builder = replay_key_builder or (
            lambda update_id: f"tg_update_seen:{update_id}"
        )
        self._environ = os.environ if environ is None else environ
        self._max_memory_updates = max(100, int(max_memory_updates))
        self.memory: OrderedDict[int, MemoryRecord] = OrderedDict()
        self.memory_lock = threading.RLock()
        self._warning_keys: set[str] = set()
        self._warning_lock = threading.Lock()

    def configure(
        self,
        *,
        redis_client_provider: RedisProvider | None = None,
        replay_key_builder: ReplayKeyBuilder | None = None,
    ) -> None:
        """Update runtime dependencies without replacing in-memory state."""

        if redis_client_provider is not None:
            self._redis_client_provider = redis_client_provider
        if replay_key_builder is not None:
            self._replay_key_builder = replay_key_builder

    def replay_ttl_seconds(self) -> int:
        return _bounded_env_int(
            self._environ,
            "WEBHOOK_REPLAY_TTL_S",
            600,
            minimum=60,
            maximum=86_400,
        )

    def processing_ttl_seconds(self) -> int:
        return _bounded_env_int(
            self._environ,
            "WEBHOOK_PROCESSING_TTL_S",
            120,
            minimum=15,
            maximum=self.replay_ttl_seconds(),
        )

    @staticmethod
    def update_id(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def replay_key(self, update_id: int) -> str:
        return self._replay_key_builder(int(update_id))

    def trim_memory_locked(self, now: float, ttl: int) -> None:
        """Trim stale/overflow entries while ``memory_lock`` is held."""

        stale_before = now - ttl
        for old_update_id, (_state, old_ts, _token) in list(self.memory.items())[
            :5_000
        ]:
            if old_ts < stale_before:
                self.memory.pop(old_update_id, None)
        while len(self.memory) > self._max_memory_updates:
            self.memory.popitem(last=False)

    def _redis_client(self) -> Any | None:
        try:
            return self._redis_client_provider()
        except Exception as exc:
            self._warn_once("provider", "Redis provider failed; using memory: %s", exc)
            return None

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        with self._warning_lock:
            if key in self._warning_keys:
                return
            self._warning_keys.add(key)
        logger.warning(message, *args)

    @staticmethod
    def _result(
        state: str,
        token: str | None,
        *,
        include_token: bool,
    ) -> ClaimResult:
        return (state, token) if include_token else state

    async def claim(
        self,
        update_id: Any,
        *,
        include_token: bool = False,
    ) -> ClaimResult:
        """Return claimed, processing, completed, or invalid."""

        uid = self.update_id(update_id)
        if uid is None:
            return self._result("invalid", None, include_token=include_token)

        claim_token = secrets.token_urlsafe(18) if include_token else None
        processing_value = (
            f"processing:{claim_token}" if claim_token is not None else "processing"
        )
        redis_client = self._redis_client()
        if redis_client is not None:
            key = self.replay_key(uid)
            try:
                created = await asyncio.to_thread(
                    redis_client.set,
                    key,
                    processing_value,
                    ex=self.processing_ttl_seconds(),
                    nx=True,
                )
                if created:
                    return self._result(
                        "claimed", claim_token, include_token=include_token
                    )
                raw = await asyncio.to_thread(redis_client.get, key)
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")
                state = "completed" if str(raw or "").lower() == "done" else "processing"
                return self._result(state, None, include_token=include_token)
            except Exception as exc:
                self._warn_once("claim", "Webhook replay Redis fallback: %s", exc)

        now = time.monotonic()
        ttl = self.replay_ttl_seconds()
        with self.memory_lock:
            self.trim_memory_locked(now, ttl)
            existing = self.memory.get(uid)
            if existing:
                state, _timestamp, _token = existing
                self.memory.move_to_end(uid)
                existing_state = "completed" if state == "done" else "processing"
                return self._result(existing_state, None, include_token=include_token)
            self.memory[uid] = ("processing", now, claim_token)
            self.trim_memory_locked(now, ttl)
            return self._result("claimed", claim_token, include_token=include_token)

    async def complete(
        self,
        update_id: Any,
        *,
        claim_token: str | None = None,
    ) -> bool:
        uid = self.update_id(update_id)
        if uid is None:
            return False
        redis_client = self._redis_client()
        if redis_client is not None:
            key = self.replay_key(uid)
            try:
                if claim_token is None:
                    await asyncio.to_thread(
                        redis_client.set,
                        key,
                        "done",
                        ex=self.replay_ttl_seconds(),
                    )
                else:
                    script = (
                        "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                        "redis.call('SET', KEYS[1], 'done', 'EX', ARGV[2]); "
                        "return 1 else return 0 end"
                    )
                    completed = await asyncio.to_thread(
                        redis_client.eval,
                        script,
                        1,
                        key,
                        f"processing:{claim_token}",
                        str(self.replay_ttl_seconds()),
                    )
                    if not completed:
                        return False
            except Exception as exc:
                self._warn_once(
                    "complete",
                    "Webhook replay completion Redis fallback: %s",
                    exc,
                )

        now = time.monotonic()
        with self.memory_lock:
            existing = self.memory.get(uid)
            if (
                claim_token is not None
                and existing is not None
                and existing[2] not in {None, claim_token}
            ):
                return False
            self.memory[uid] = ("done", now, None)
            self.memory.move_to_end(uid)
            self.trim_memory_locked(now, self.replay_ttl_seconds())
        return True

    async def release(
        self,
        update_id: Any,
        *,
        claim_token: str | None = None,
    ) -> bool:
        uid = self.update_id(update_id)
        if uid is None:
            return False
        released = True
        redis_client = self._redis_client()
        if redis_client is not None:
            key = self.replay_key(uid)
            script = (
                "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                "return redis.call('DEL', KEYS[1]) else return 0 end"
            )
            expected = (
                f"processing:{claim_token}"
                if claim_token is not None
                else "processing"
            )
            try:
                released = bool(
                    await asyncio.to_thread(
                        redis_client.eval,
                        script,
                        1,
                        key,
                        expected,
                    )
                )
            except Exception as exc:
                self._warn_once(
                    "release",
                    "Webhook replay release Redis fallback: %s",
                    exc,
                )
        with self.memory_lock:
            existing = self.memory.get(uid)
            owned = (
                existing
                and existing[0] == "processing"
                and (claim_token is None or existing[2] == claim_token)
            )
            if owned:
                self.memory.pop(uid, None)
            elif existing is not None and claim_token is not None:
                released = False
        return released


_DEFAULT_DEDUPLICATOR = WebhookUpdateDeduplicator()
WEBHOOK_UPDATE_MEMORY = _DEFAULT_DEDUPLICATOR.memory
WEBHOOK_UPDATE_MEMORY_LOCK = _DEFAULT_DEDUPLICATOR.memory_lock


def configure_webhook_deduplicator(
    *,
    redis_client_provider: RedisProvider | None = None,
    replay_key_builder: ReplayKeyBuilder | None = None,
) -> WebhookUpdateDeduplicator:
    _DEFAULT_DEDUPLICATOR.configure(
        redis_client_provider=redis_client_provider,
        replay_key_builder=replay_key_builder,
    )
    return _DEFAULT_DEDUPLICATOR


def _webhook_replay_ttl_seconds() -> int:
    return _DEFAULT_DEDUPLICATOR.replay_ttl_seconds()


def _webhook_processing_ttl_seconds() -> int:
    return _DEFAULT_DEDUPLICATOR.processing_ttl_seconds()


def _telegram_webhook_update_id(update_id: Any) -> int | None:
    return _DEFAULT_DEDUPLICATOR.update_id(update_id)


def _telegram_webhook_replay_key(update_id: int) -> str:
    return _DEFAULT_DEDUPLICATOR.replay_key(update_id)


def _trim_webhook_memory_locked(now: float, ttl: int) -> None:
    _DEFAULT_DEDUPLICATOR.trim_memory_locked(now, ttl)


async def _telegram_webhook_update_claim(
    update_id: Any,
    *,
    include_token: bool = False,
) -> ClaimResult:
    return await _DEFAULT_DEDUPLICATOR.claim(
        update_id,
        include_token=include_token,
    )


async def _telegram_webhook_update_complete(
    update_id: Any,
    *,
    claim_token: str | None = None,
) -> bool:
    return await _DEFAULT_DEDUPLICATOR.complete(update_id, claim_token=claim_token)


async def _telegram_webhook_update_release(
    update_id: Any,
    *,
    claim_token: str | None = None,
) -> bool:
    return await _DEFAULT_DEDUPLICATOR.release(update_id, claim_token=claim_token)


__all__ = [
    "WEBHOOK_UPDATE_MEMORY",
    "WEBHOOK_UPDATE_MEMORY_LOCK",
    "WebhookUpdateDeduplicator",
    "_telegram_webhook_replay_key",
    "_telegram_webhook_update_claim",
    "_telegram_webhook_update_complete",
    "_telegram_webhook_update_id",
    "_telegram_webhook_update_release",
    "_trim_webhook_memory_locked",
    "_webhook_processing_ttl_seconds",
    "_webhook_replay_ttl_seconds",
    "configure_webhook_deduplicator",
]
