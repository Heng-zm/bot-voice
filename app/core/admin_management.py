"""Redis-backed administrator management with confirmation and audit records."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any

_ADD_SCRIPT = """
-- bot_voice:admin_add_v1
local confirmation = redis.call('GET', KEYS[1])
if not confirmation or confirmation ~= ARGV[1] then
  return {-2, 0}
end
redis.call('DEL', KEYS[1])
local changed = redis.call('SADD', KEYS[2], ARGV[3])
redis.call(
  'XADD', KEYS[3], 'MAXLEN', '~', ARGV[5], '*',
  'timestamp', ARGV[2],
  'actor_id', ARGV[4],
  'action', 'admin_add',
  'target_id', ARGV[3],
  'changed', tostring(changed)
)
return {1, changed}
""".strip()

_REMOVE_SCRIPT = """
-- bot_voice:admin_remove_v1
local confirmation = redis.call('GET', KEYS[1])
if not confirmation or confirmation ~= ARGV[1] then
  return {-2, 0}
end
redis.call('DEL', KEYS[1])
if redis.call('SISMEMBER', KEYS[2], ARGV[3]) == 0 then
  redis.call(
    'XADD', KEYS[3], 'MAXLEN', '~', ARGV[5], '*',
    'timestamp', ARGV[2],
    'actor_id', ARGV[4],
    'action', 'admin_remove',
    'target_id', ARGV[3],
    'changed', '0'
  )
  return {1, 0}
end
if redis.call('SCARD', KEYS[2]) <= 1 then
  redis.call(
    'XADD', KEYS[3], 'MAXLEN', '~', ARGV[5], '*',
    'timestamp', ARGV[2],
    'actor_id', ARGV[4],
    'action', 'admin_remove_denied_final',
    'target_id', ARGV[3],
    'changed', '0'
  )
  return {-1, 0}
end
local changed = redis.call('SREM', KEYS[2], ARGV[3])
redis.call(
  'XADD', KEYS[3], 'MAXLEN', '~', ARGV[5], '*',
  'timestamp', ARGV[2],
  'actor_id', ARGV[4],
  'action', 'admin_remove',
  'target_id', ARGV[3],
  'changed', tostring(changed)
)
return {1, changed}
""".strip()


class AdminManagementError(RuntimeError):
    """Base error for administrator management failures."""


class AdminConfirmationError(AdminManagementError):
    """Raised when a confirmation token is missing, expired, or mismatched."""


class LastAdministratorError(AdminManagementError):
    """Raised when an operation would remove the final administrator."""


@dataclass(frozen=True, slots=True)
class AdminMutation:
    action: str
    actor_id: int
    target_id: int
    changed: bool


class RedisAdminManager:
    """Atomically manage the Telegram administrator set and its audit stream."""

    def __init__(
        self,
        redis_client: Any,
        *,
        redis_prefix: str = "tgbot",
        confirmation_ttl_seconds: int = 300,
        audit_max_length: int = 10_000,
    ) -> None:
        prefix = str(redis_prefix or "tgbot").strip().strip(":") or "tgbot"
        self.redis = redis_client
        self.admins_key = f"{prefix}:security:admin_user_ids:v1"
        self.confirmation_prefix = f"{prefix}:security:admin_confirm:v1"
        self.audit_key = f"{prefix}:security:admin_audit:v1"
        self.confirmation_ttl_seconds = max(
            30,
            min(900, int(confirmation_ttl_seconds)),
        )
        self.audit_max_length = max(100, min(100_000, int(audit_max_length)))

    @staticmethod
    def _user_id(value: Any) -> int:
        try:
            user_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Administrator user ID must be an integer.") from exc
        if user_id <= 0 or user_id >= 2**63:
            raise ValueError("Administrator user ID is outside the valid range.")
        return user_id

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        return str(value)

    def _confirmation_key(self, token: str) -> str:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return f"{self.confirmation_prefix}:{digest}"

    @staticmethod
    def _confirmation_value(action: str, actor_id: int, target_id: int) -> str:
        return f"{action}:{actor_id}:{target_id}"

    def _require_redis(self) -> Any:
        if self.redis is None:
            raise AdminManagementError(
                "Redis is required for administrator management."
            )
        return self.redis

    def create_confirmation_sync(
        self,
        *,
        action: str,
        actor_id: int,
        target_id: int,
    ) -> tuple[str, int]:
        action = str(action or "").strip().lower()
        if action not in {"add", "remove"}:
            raise ValueError("Administrator action must be add or remove.")
        actor = self._user_id(actor_id)
        target = self._user_id(target_id)
        client = self._require_redis()
        try:
            for _attempt in range(3):
                token = secrets.token_urlsafe(32)
                stored = client.set(
                    self._confirmation_key(token),
                    self._confirmation_value(action, actor, target),
                    nx=True,
                    ex=self.confirmation_ttl_seconds,
                )
                if stored:
                    return token, self.confirmation_ttl_seconds
        except Exception as exc:
            raise AdminManagementError(
                "Redis could not create an administrator confirmation."
            ) from exc
        raise AdminManagementError("Could not create a unique confirmation token.")

    async def create_confirmation(
        self,
        *,
        action: str,
        actor_id: int,
        target_id: int,
    ) -> tuple[str, int]:
        return await asyncio.to_thread(
            self.create_confirmation_sync,
            action=action,
            actor_id=actor_id,
            target_id=target_id,
        )

    @staticmethod
    def _mutation_result(raw: Any) -> tuple[int, bool]:
        values = list(raw or ())
        if len(values) != 2:
            raise AdminManagementError(
                "Redis returned an invalid administrator mutation result."
            )
        return int(values[0]), bool(int(values[1]))

    def _mutate_sync(
        self,
        *,
        action: str,
        actor_id: int,
        target_id: int,
        confirmation_token: str,
    ) -> AdminMutation:
        actor = self._user_id(actor_id)
        target = self._user_id(target_id)
        token = str(confirmation_token or "").strip()
        if len(token) < 32 or len(token) > 256:
            raise AdminConfirmationError("Confirmation token is invalid or expired.")
        expected = self._confirmation_value(action, actor, target)
        client = self._require_redis()
        script = _ADD_SCRIPT if action == "add" else _REMOVE_SCRIPT
        try:
            raw = client.eval(
                script,
                3,
                self._confirmation_key(token),
                self.admins_key,
                self.audit_key,
                expected,
                str(time.time_ns()),
                str(target),
                str(actor),
                str(self.audit_max_length),
            )
        except Exception as exc:
            raise AdminManagementError(
                "Redis could not update the administrator allowlist."
            ) from exc
        status, changed = self._mutation_result(raw)
        if status == -2:
            raise AdminConfirmationError(
                "Confirmation token is invalid, expired, or belongs to another action."
            )
        if status == -1:
            raise LastAdministratorError(
                "The final administrator cannot be removed."
            )
        if status != 1:
            raise AdminManagementError("Administrator mutation failed.")
        return AdminMutation(action, actor, target, changed)

    async def add(
        self,
        *,
        actor_id: int,
        target_id: int,
        confirmation_token: str,
    ) -> AdminMutation:
        return await asyncio.to_thread(
            self._mutate_sync,
            action="add",
            actor_id=actor_id,
            target_id=target_id,
            confirmation_token=confirmation_token,
        )

    async def remove(
        self,
        *,
        actor_id: int,
        target_id: int,
        confirmation_token: str,
    ) -> AdminMutation:
        return await asyncio.to_thread(
            self._mutate_sync,
            action="remove",
            actor_id=actor_id,
            target_id=target_id,
            confirmation_token=confirmation_token,
        )

    def list_ids_sync(self) -> tuple[int, ...]:
        try:
            members = self._require_redis().smembers(self.admins_key) or ()
        except Exception as exc:
            raise AdminManagementError(
                "Redis could not load the administrator allowlist."
            ) from exc
        ids: set[int] = set()
        for value in members:
            try:
                ids.add(self._user_id(self._decode(value)))
            except ValueError:
                continue
        return tuple(sorted(ids))

    async def list_ids(self) -> tuple[int, ...]:
        return await asyncio.to_thread(self.list_ids_sync)

    def audit_sync(self, *, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(500, int(limit)))
        try:
            entries = self._require_redis().xrevrange(
                self.audit_key,
                max="+",
                min="-",
                count=safe_limit,
            )
        except Exception as exc:
            raise AdminManagementError(
                "Redis could not load the administrator audit trail."
            ) from exc
        result: list[dict[str, Any]] = []
        for entry_id, fields in entries or ():
            decoded = {
                self._decode(key): self._decode(value)
                for key, value in dict(fields).items()
            }
            result.append(
                {
                    "id": self._decode(entry_id),
                    "timestamp_ns": int(decoded.get("timestamp") or 0),
                    "actor_id": int(decoded.get("actor_id") or 0),
                    "action": decoded.get("action", ""),
                    "target_id": int(decoded.get("target_id") or 0),
                    "changed": decoded.get("changed") == "1",
                }
            )
        return result

    async def audit(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.audit_sync, limit=limit)


__all__ = [
    "AdminConfirmationError",
    "AdminManagementError",
    "AdminMutation",
    "LastAdministratorError",
    "RedisAdminManager",
]
