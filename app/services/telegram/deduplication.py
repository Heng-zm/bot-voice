"""Process-local Telegram webhook replay protection.

The application runs as one web/Telegram process, so a distributed Redis lease is
neither required nor desirable.  This module keeps a bounded in-memory replay
window and uses short processing leases so a crashed/cancelled handler can be
retried quickly.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

ReplayState = Literal["processing", "done"]


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _webhook_replay_ttl_seconds() -> int:
    return _env_int("WEBHOOK_REPLAY_TTL_S", 600, minimum=60, maximum=86400)


def _webhook_processing_ttl_seconds() -> int:
    return _env_int(
        "WEBHOOK_PROCESSING_TTL_S",
        120,
        minimum=15,
        maximum=_webhook_replay_ttl_seconds(),
    )


def _webhook_replay_max_entries() -> int:
    return _env_int("WEBHOOK_REPLAY_MAX_ENTRIES", 50_000, minimum=1_000, maximum=250_000)


def _telegram_webhook_update_id(update_id: Any) -> int | None:
    try:
        return int(update_id)
    except (TypeError, ValueError, OverflowError):
        return None


def _telegram_webhook_replay_key(uid: int) -> str:
    """Compatibility/debug key; replay state itself is no longer stored in Redis."""
    return f"tg_update_seen:{int(uid)}"


@dataclass(slots=True)
class _Lease:
    state: ReplayState
    created_at: float
    token: str | None = None


class WebhookReplayStore:
    """Bounded replay window with ownership-aware processing leases."""

    def __init__(self) -> None:
        self._items: OrderedDict[int, _Lease] = OrderedDict()
        self._lock = threading.RLock()
        self._claims = 0
        self._duplicates = 0
        self._reclaimed = 0

    @staticmethod
    def _expired(lease: _Lease, now: float) -> bool:
        ttl = (
            _webhook_processing_ttl_seconds()
            if lease.state == "processing"
            else _webhook_replay_ttl_seconds()
        )
        return now - lease.created_at >= ttl

    def _expire_uid_locked(self, uid: int, now: float) -> None:
        lease = self._items.get(uid)
        if lease is None or not self._expired(lease, now):
            return
        self._items.pop(uid, None)
        if lease.state == "processing":
            self._reclaimed += 1

    def _trim_locked(self, now: float) -> None:
        for uid, lease in list(self._items.items())[:10_000]:
            if self._expired(lease, now):
                self._items.pop(uid, None)
                if lease.state == "processing":
                    self._reclaimed += 1
        max_entries = _webhook_replay_max_entries()
        while len(self._items) > max_entries:
            self._items.popitem(last=False)

    def claim(
        self,
        update_id: Any,
        *,
        include_token: bool = False,
    ) -> str | tuple[str, str | None]:
        def result(state: str, token: str | None = None) -> str | tuple[str, str | None]:
            return (state, token) if include_token else state

        uid = _telegram_webhook_update_id(update_id)
        if uid is None:
            return result("invalid")

        now = time.monotonic()
        with self._lock:
            self._trim_locked(now)
            self._expire_uid_locked(uid, now)
            existing = self._items.get(uid)
            if existing is not None:
                self._items.move_to_end(uid)
                self._duplicates += 1
                return result("completed" if existing.state == "done" else "processing")

            token = secrets.token_urlsafe(18) if include_token else None
            self._items[uid] = _Lease("processing", now, token)
            self._items.move_to_end(uid)
            self._claims += 1
            self._trim_locked(now)
            return result("claimed", token)

    def complete(self, update_id: Any, *, claim_token: str | None = None) -> bool:
        uid = _telegram_webhook_update_id(update_id)
        if uid is None:
            return False
        now = time.monotonic()
        with self._lock:
            self._trim_locked(now)
            self._expire_uid_locked(uid, now)
            existing = self._items.get(uid)
            if claim_token is not None:
                if existing is None or existing.state != "processing" or existing.token != claim_token:
                    return False
            self._items[uid] = _Lease("done", now, None)
            self._items.move_to_end(uid)
            self._trim_locked(now)
            return True

    def release(self, update_id: Any, *, claim_token: str | None = None) -> bool:
        uid = _telegram_webhook_update_id(update_id)
        if uid is None:
            return False
        now = time.monotonic()
        with self._lock:
            self._trim_locked(now)
            self._expire_uid_locked(uid, now)
            existing = self._items.get(uid)
            if existing is None:
                return True
            if existing.state != "processing":
                return False
            if claim_token is not None and existing.token != claim_token:
                return False
            self._items.pop(uid, None)
            return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._claims = 0
            self._duplicates = 0
            self._reclaimed = 0

    def discard(self, update_id: Any) -> None:
        uid = _telegram_webhook_update_id(update_id)
        if uid is None:
            return
        with self._lock:
            self._items.pop(uid, None)

    def snapshot(self) -> dict[str, int]:
        now = time.monotonic()
        with self._lock:
            self._trim_locked(now)
            processing = sum(1 for item in self._items.values() if item.state == "processing")
            completed = len(self._items) - processing
            return {
                "entries": len(self._items),
                "processing": processing,
                "completed": completed,
                "claims": self._claims,
                "duplicates": self._duplicates,
                "expired_processing_reclaimed": self._reclaimed,
                "processing_ttl_s": _webhook_processing_ttl_seconds(),
                "replay_ttl_s": _webhook_replay_ttl_seconds(),
                "max_entries": _webhook_replay_max_entries(),
            }


_STORE = WebhookReplayStore()


async def _telegram_webhook_update_claim(
    update_id: Any,
    *,
    include_token: bool = False,
) -> str | tuple[str, str | None]:
    return _STORE.claim(update_id, include_token=include_token)


async def _telegram_webhook_update_complete(
    update_id: Any,
    *,
    claim_token: str | None = None,
) -> bool:
    return _STORE.complete(update_id, claim_token=claim_token)


async def _telegram_webhook_update_release(
    update_id: Any,
    *,
    claim_token: str | None = None,
) -> bool:
    return _STORE.release(update_id, claim_token=claim_token)


def _trim_webhook_memory_locked(now: float, ttl: int) -> None:
    """Deprecated compatibility hook for older tests/imports.

    The native store owns trimming.  Calling this function simply triggers a
    trim using the current configured TTLs; ``ttl`` is accepted for API parity.
    """
    del now, ttl
    _STORE.snapshot()


def get_webhook_replay_snapshot() -> dict[str, int]:
    return _STORE.snapshot()


def reset_webhook_replay_store() -> None:
    _STORE.clear()


__all__ = [
    "WebhookReplayStore",
    "_telegram_webhook_replay_key",
    "_telegram_webhook_update_claim",
    "_telegram_webhook_update_complete",
    "_telegram_webhook_update_id",
    "_telegram_webhook_update_release",
    "_trim_webhook_memory_locked",
    "get_webhook_replay_snapshot",
    "reset_webhook_replay_store",
]
