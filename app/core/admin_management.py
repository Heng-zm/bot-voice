"""Single-process administrator management persisted in Supabase bot_settings."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.core.telegram_auth import TelegramAdminAuthorizer, get_telegram_admin_authorizer
from app.services.settings.store import SettingsStore, get_settings_store

_AUDIT_KEY = "security:admin_audit:v2"


class AdminManagementError(RuntimeError):
    pass


class AdminConfirmationError(AdminManagementError):
    pass


class LastAdministratorError(AdminManagementError):
    pass


@dataclass(frozen=True)
class AdminMutationResult:
    action: str
    target_id: int
    changed: bool
    persistent: bool


class SupabaseAdminManager:
    """Admin mutations with short-lived, process-local confirmation tokens."""

    _confirmations: dict[str, tuple[str, int, int, float]] = {}
    _confirmation_lock = asyncio.Lock()
    _audit_memory: deque[dict[str, Any]] = deque(maxlen=200)

    def __init__(
        self,
        settings_store: SettingsStore | None = None,
        authorizer: TelegramAdminAuthorizer | None = None,
        *,
        confirmation_ttl_seconds: int = 120,
    ) -> None:
        self.store = settings_store or get_settings_store()
        self.authorizer = authorizer or get_telegram_admin_authorizer()
        self.confirmation_ttl_seconds = max(30, min(600, int(confirmation_ttl_seconds)))

    async def list_ids(self) -> tuple[int, ...]:
        return tuple(sorted(await self.authorizer.load_ids(force=True)))

    async def create_confirmation(self, *, action: str, actor_id: int, target_id: int) -> tuple[str, int]:
        action = str(action or "").strip().lower()
        if action not in {"add", "remove"}:
            raise AdminConfirmationError("Unsupported administrator action.")
        actor_id, target_id = int(actor_id), int(target_id)
        if actor_id <= 0 or target_id <= 0:
            raise AdminConfirmationError("Administrator IDs must be positive integers.")
        current = set(await self.authorizer.load_ids(force=True))
        if actor_id not in current:
            raise AdminConfirmationError("The requesting administrator is no longer authorized.")
        if action == "remove" and target_id in current and len(current) <= 1:
            raise LastAdministratorError("The last administrator cannot be removed.")
        token = secrets.token_urlsafe(32)
        expires_at = time.monotonic() + self.confirmation_ttl_seconds
        async with self._confirmation_lock:
            self._prune_confirmations_locked()
            self._confirmations[token] = (action, actor_id, target_id, expires_at)
        return token, self.confirmation_ttl_seconds

    async def add(self, *, actor_id: int, target_id: int, confirmation_token: str) -> AdminMutationResult:
        await self._consume_confirmation("add", actor_id, target_id, confirmation_token)
        current = set(await self.authorizer.load_ids(force=True))
        changed = int(target_id) not in current
        current.add(int(target_id))
        persistent = await self.authorizer.save_ids(current, updated_by=int(actor_id))
        await self._audit("add", actor_id, target_id, changed, persistent)
        return AdminMutationResult("add", int(target_id), changed, persistent)

    async def remove(self, *, actor_id: int, target_id: int, confirmation_token: str) -> AdminMutationResult:
        await self._consume_confirmation("remove", actor_id, target_id, confirmation_token)
        current = set(await self.authorizer.load_ids(force=True))
        target_id = int(target_id)
        if target_id in current and len(current) <= 1:
            raise LastAdministratorError("The last administrator cannot be removed.")
        changed = target_id in current
        current.discard(target_id)
        if not current:
            raise LastAdministratorError("The last administrator cannot be removed.")
        persistent = await self.authorizer.save_ids(current, updated_by=int(actor_id))
        await self._audit("remove", actor_id, target_id, changed, persistent)
        return AdminMutationResult("remove", target_id, changed, persistent)

    async def audit(self, *, limit: int = 100) -> list[dict[str, Any]]:
        payload = await self.store.get_json(_AUDIT_KEY, [])
        if isinstance(payload, list):
            return list(payload)[-max(1, min(500, int(limit))):][::-1]
        return list(self._audit_memory)[-max(1, min(500, int(limit))):][::-1]

    async def _consume_confirmation(self, action: str, actor_id: int, target_id: int, token: str) -> None:
        token = str(token or "").strip()
        if not token:
            raise AdminConfirmationError("Confirmation token is required.")
        async with self._confirmation_lock:
            self._prune_confirmations_locked()
            record = self._confirmations.pop(token, None)
        expected = (action, int(actor_id), int(target_id))
        if record is None or record[:3] != expected:
            raise AdminConfirmationError("Confirmation token is invalid, expired, or already used.")

    def _prune_confirmations_locked(self) -> None:
        now = time.monotonic()
        expired = [token for token, record in self._confirmations.items() if record[3] <= now]
        for token in expired:
            self._confirmations.pop(token, None)

    async def _audit(self, action: str, actor_id: int, target_id: int, changed: bool, persistent: bool) -> None:
        entry = {
            "action": action,
            "actor_id": int(actor_id),
            "target_id": int(target_id),
            "changed": bool(changed),
            "persistent": bool(persistent),
            "timestamp": int(time.time()),
        }
        self._audit_memory.append(entry)
        previous = await self.store.get_json(_AUDIT_KEY, [])
        rows = list(previous) if isinstance(previous, list) else []
        rows.append(entry)
        await self.store.set_json(_AUDIT_KEY, rows[-200:], updated_by=int(actor_id))


__all__ = [
    "AdminConfirmationError",
    "AdminManagementError",
    "AdminMutationResult",
    "LastAdministratorError",
    "SupabaseAdminManager",
]
