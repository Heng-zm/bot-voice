"""Redis-backed idempotent Telegram result delivery.

A known progress message ID is edited in place whenever possible. Retrying the
same job therefore converges on one Telegram message instead of sending a new
result for every worker attempt.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

try:
    from telegram.error import BadRequest
except Exception:  # pragma: no cover - optional in isolated tooling
    class BadRequest(Exception):
        """Fallback used when python-telegram-bot is not installed."""


_CLAIM_SCRIPT = """
-- bot_voice:delivery_claim_v1
local state = redis.call('HGET', KEYS[1], 'state')
if state == 'completed' then
  return {2, redis.call('HGET', KEYS[1], 'result') or ''}
end
local deadline = tonumber(redis.call('HGET', KEYS[1], 'lease_deadline') or '0')
if state == 'processing' and deadline > tonumber(ARGV[1]) then
  return {0, ''}
end
redis.call(
  'HSET', KEYS[1],
  'state', 'processing',
  'lease_token', ARGV[2],
  'lease_deadline', ARGV[3],
  'updated_at', ARGV[1]
)
redis.call('HINCRBY', KEYS[1], 'attempts', 1)
redis.call('EXPIRE', KEYS[1], ARGV[4])
return {1, ''}
""".strip()

_COMPLETE_SCRIPT = """
-- bot_voice:delivery_complete_v1
if redis.call('HGET', KEYS[1], 'state') ~= 'processing' then
  return 0
end
if redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call(
  'HSET', KEYS[1],
  'state', 'completed',
  'result', ARGV[2],
  'updated_at', ARGV[3]
)
redis.call('HDEL', KEYS[1], 'lease_token', 'lease_deadline')
redis.call('EXPIRE', KEYS[1], ARGV[4])
return 1
""".strip()

_RELEASE_SCRIPT = """
-- bot_voice:delivery_release_v1
if redis.call('HGET', KEYS[1], 'state') ~= 'processing' then
  return 0
end
if redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call(
  'HSET', KEYS[1],
  'state', 'failed',
  'last_error', ARGV[2],
  'updated_at', ARGV[3]
)
redis.call('HDEL', KEYS[1], 'lease_token', 'lease_deadline')
redis.call('EXPIRE', KEYS[1], ARGV[4])
return 1
""".strip()


class TelegramDeliveryError(RuntimeError):
    """Base delivery error."""


class TelegramDeliveryBusy(TelegramDeliveryError):
    """Raised when another worker still owns the delivery lease."""


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    status: str
    token: str = ""
    result: dict[str, Any] | None = None


class RedisDeliveryStore:
    """Small delivery state machine with expiring leases and retained results."""

    def __init__(
        self,
        redis_client: Any,
        *,
        redis_prefix: str = "tgbot",
        lease_seconds: float = 90.0,
        retention_seconds: int = 604_800,
    ) -> None:
        if redis_client is None:
            raise TelegramDeliveryError("Redis is required for idempotent delivery.")
        prefix = str(redis_prefix or "tgbot").strip().strip(":") or "tgbot"
        self.redis = redis_client
        self.prefix = f"{prefix}:delivery:v1"
        self.lease_seconds = max(15.0, min(3_600.0, float(lease_seconds)))
        self.retention_seconds = max(300, min(2_592_000, int(retention_seconds)))

    def key(self, idempotency_key: str) -> str:
        clean = str(idempotency_key or "").strip()
        if not clean:
            raise ValueError("Delivery idempotency key is required.")
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        return f"{self.prefix}:{digest}"

    async def _call(self, method: str, *args: Any) -> Any:
        try:
            op = getattr(self.redis, method)
            if inspect.iscoroutinefunction(op):
                return await op(*args)
            # Calling a synchronous Redis client on the event-loop thread both
            # blocks other jobs and, previously, caused the command to be run a
            # second time in ``to_thread``.  Some proxy clients are not marked
            # as coroutine functions but still return an awaitable, so retain
            # that compatibility after making exactly one call.
            res = await asyncio.to_thread(op, *args)
            if inspect.isawaitable(res):
                return await res
            return res
        except Exception as exc:
            raise TelegramDeliveryError(f"Redis delivery {method} failed.") from exc

    @staticmethod
    def _decode(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value or "")

    async def claim(self, idempotency_key: str) -> DeliveryClaim:
        now = time.time()
        token = secrets.token_urlsafe(24)
        raw = await self._call(
            "eval",
            _CLAIM_SCRIPT,
            1,
            self.key(idempotency_key),
            str(now),
            token,
            str(now + self.lease_seconds),
            str(self.retention_seconds),
        )
        values = list(raw or ())
        if len(values) != 2:
            raise TelegramDeliveryError("Redis returned an invalid delivery claim.")
        status = int(values[0])
        if status == 2:
            result_raw = self._decode(values[1])
            try:
                result = json.loads(result_raw) if result_raw else {}
            except json.JSONDecodeError as exc:
                raise TelegramDeliveryError("Stored delivery result is invalid.") from exc
            return DeliveryClaim("completed", result=result)
        if status == 1:
            return DeliveryClaim("claimed", token=token)
        return DeliveryClaim("busy")

    async def complete(
        self,
        idempotency_key: str,
        token: str,
        result: dict[str, Any],
    ) -> bool:
        payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        changed = await self._call(
            "eval",
            _COMPLETE_SCRIPT,
            1,
            self.key(idempotency_key),
            token,
            payload,
            str(time.time()),
            str(self.retention_seconds),
        )
        return bool(changed)

    async def release(
        self,
        idempotency_key: str,
        token: str,
        error: BaseException | str,
    ) -> bool:
        changed = await self._call(
            "eval",
            _RELEASE_SCRIPT,
            1,
            self.key(idempotency_key),
            token,
            str(error)[:500],
            str(time.time()),
            str(self.retention_seconds),
        )
        return bool(changed)


class MemoryDeliveryStore:
    """Process-local delivery leases for explicit Redis-disabled mode."""

    def __init__(
        self,
        *,
        lease_seconds: float = 90.0,
        retention_seconds: int = 604_800,
    ) -> None:
        self.lease_seconds = max(15.0, min(3_600.0, float(lease_seconds)))
        self.retention_seconds = max(300, min(2_592_000, int(retention_seconds)))
        self._states: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def claim(self, idempotency_key: str) -> DeliveryClaim:
        key = str(idempotency_key or "").strip()
        if not key:
            raise ValueError("Delivery idempotency key is required.")
        now = time.time()
        async with self._lock:
            state = self._states.get(key)
            if state and state.get("state") == "completed":
                completed_at = float(state.get("completed_at") or now)
                if completed_at + self.retention_seconds > now:
                    return DeliveryClaim("completed", result=dict(state.get("result") or {}))
                self._states.pop(key, None)
                state = None
            if (
                state
                and state.get("state") == "processing"
                and float(state.get("lease_deadline") or 0.0) > now
            ):
                return DeliveryClaim("busy")
            token = secrets.token_urlsafe(24)
            self._states[key] = {
                "state": "processing",
                "token": token,
                "lease_deadline": now + self.lease_seconds,
            }
            return DeliveryClaim("claimed", token=token)

    async def complete(
        self,
        idempotency_key: str,
        token: str,
        result: dict[str, Any],
    ) -> bool:
        async with self._lock:
            state = self._states.get(str(idempotency_key))
            if not state or state.get("state") != "processing" or state.get("token") != token:
                return False
            self._states[str(idempotency_key)] = {
                "state": "completed",
                "result": dict(result),
                "completed_at": time.time(),
            }
            return True

    async def release(
        self,
        idempotency_key: str,
        token: str,
        error: BaseException | str,
    ) -> bool:
        del error
        async with self._lock:
            key = str(idempotency_key)
            state = self._states.get(key)
            if not state or state.get("state") != "processing" or state.get("token") != token:
                return False
            self._states.pop(key, None)
            return True

class IdempotentTelegramDelivery:
    """Edit one known result message, or send once when no target exists."""

    def __init__(self, store: RedisDeliveryStore | MemoryDeliveryStore) -> None:
        self.store = store

    async def deliver_text(
        self,
        *,
        bot: Any,
        idempotency_key: str,
        chat_id: int,
        text: str,
        progress_message_id: int | None,
        reply_to_message_id: int | None = None,
        parse_mode: str | None = None,
        reply_markup: Any | None = None,
        disable_web_page_preview: bool = True,
    ) -> dict[str, Any]:
        claim = await self.store.claim(idempotency_key)
        if claim.status == "completed":
            return dict(claim.result or {})
        if claim.status != "claimed":
            raise TelegramDeliveryBusy("Another worker is delivering this result.")

        try:
            message_id = int(progress_message_id or 0)
            if message_id > 0:
                try:
                    message = await bot.edit_message_text(
                        chat_id=int(chat_id),
                        message_id=message_id,
                        text=str(text),
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_web_page_preview=disable_web_page_preview,
                    )
                    resolved_id = int(getattr(message, "message_id", 0) or message_id)
                except BadRequest as exc:
                    if "message is not modified" not in str(exc).lower():
                        raise
                    resolved_id = message_id
            else:
                kwargs: dict[str, Any] = {
                    "chat_id": int(chat_id),
                    "text": str(text),
                    "parse_mode": parse_mode,
                    "reply_markup": reply_markup,
                    "disable_web_page_preview": disable_web_page_preview,
                }
                if reply_to_message_id:
                    kwargs["reply_to_message_id"] = int(reply_to_message_id)
                message = await bot.send_message(**kwargs)
                resolved_id = int(getattr(message, "message_id", 0) or 0)

            result = {
                "chat_id": int(chat_id),
                "message_id": resolved_id,
                "mode": "edit" if message_id > 0 else "send",
                "delivered_at": time.time(),
            }
            if not await self.store.complete(idempotency_key, claim.token, result):
                raise TelegramDeliveryError("Delivery lease was lost before completion.")
            return result
        except BaseException as exc:
            with suppress(Exception):
                await self.store.release(idempotency_key, claim.token, exc)
            raise

    async def deliver_voice(
        self,
        *,
        bot: Any,
        idempotency_key: str,
        chat_id: int,
        voice: Any,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        """Send a voice result once and reuse its stored result on retries."""

        claim = await self.store.claim(idempotency_key)
        if claim.status == "completed":
            return dict(claim.result or {})
        if claim.status != "claimed":
            raise TelegramDeliveryBusy("Another worker is delivering this voice result.")

        try:
            kwargs: dict[str, Any] = {
                "chat_id": int(chat_id),
                "voice": voice,
            }
            if reply_to_message_id:
                kwargs["reply_to_message_id"] = int(reply_to_message_id)
            message = await bot.send_voice(**kwargs)
            result = {
                "chat_id": int(chat_id),
                "message_id": int(getattr(message, "message_id", 0) or 0),
                "mode": "send_voice",
                "delivered_at": time.time(),
            }
            if not await self.store.complete(idempotency_key, claim.token, result):
                raise TelegramDeliveryError("Voice delivery lease was lost before completion.")
            return result
        except BaseException as exc:
            with suppress(Exception):
                await self.store.release(idempotency_key, claim.token, exc)
            raise


_DELIVERY: IdempotentTelegramDelivery | None = None


def configure_telegram_delivery(
    redis_client: Any | None,
    *,
    redis_prefix: str = "tgbot",
    memory_fallback: bool = False,
) -> IdempotentTelegramDelivery | None:
    global _DELIVERY
    if redis_client is None:
        _DELIVERY = (
            IdempotentTelegramDelivery(MemoryDeliveryStore())
            if memory_fallback
            else None
        )
        return _DELIVERY
    _DELIVERY = IdempotentTelegramDelivery(
        RedisDeliveryStore(redis_client, redis_prefix=redis_prefix)
    )
    return _DELIVERY


def get_telegram_delivery() -> IdempotentTelegramDelivery:
    if _DELIVERY is None:
        raise TelegramDeliveryError("Telegram delivery is not configured.")
    return _DELIVERY


__all__ = [
    "DeliveryClaim",
    "IdempotentTelegramDelivery",
    "MemoryDeliveryStore",
    "RedisDeliveryStore",
    "TelegramDeliveryBusy",
    "TelegramDeliveryError",
    "configure_telegram_delivery",
    "get_telegram_delivery",
]
