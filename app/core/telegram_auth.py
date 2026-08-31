"""Supabase-backed Telegram administrator policy for bot commands."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.services.settings.store import SettingsStore, get_settings_store

_ADMIN_KEY = "security:admin_user_ids:v2"


class TelegramAdminAuthorizer:
    """Maintain a short-lived administrator ID snapshot for command guards."""

    def __init__(self, *, cache_ttl_seconds: float = 5.0) -> None:
        self.store: SettingsStore = get_settings_store()
        self.fallback_admin_ids: frozenset[int] = frozenset()
        self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))
        self._cache: frozenset[int] | None = None
        self._cache_at = 0.0
        self._lock = asyncio.Lock()

    def configure(
        self,
        *,
        settings_store: SettingsStore | None = None,
        fallback_admin_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...] = (),
        **_ignored: Any,
    ) -> TelegramAdminAuthorizer:
        self.store = settings_store or get_settings_store()
        self.fallback_admin_ids = frozenset(
            int(value) for value in fallback_admin_ids if int(value) > 0
        )
        self.invalidate()
        return self

    def invalidate(self) -> None:
        self._cache = None
        self._cache_at = 0.0

    def is_admin_sync(self, user_id: int) -> bool:
        """Check the loaded admin snapshot without blocking Telegram's loop."""
        try:
            candidate = int(user_id)
        except (TypeError, ValueError):
            return False
        if candidate <= 0:
            return False
        snapshot = self._cache
        if snapshot is not None:
            return candidate in snapshot
        return candidate in self.fallback_admin_ids

    async def load_ids(self, *, force: bool = False) -> frozenset[int]:
        now = time.monotonic()
        if not force and self._cache is not None and now - self._cache_at < self.cache_ttl_seconds:
            return self._cache
        async with self._lock:
            now = time.monotonic()
            if not force and self._cache is not None and now - self._cache_at < self.cache_ttl_seconds:
                return self._cache
            payload = await self.store.get_json(_ADMIN_KEY, [])
            ids: set[int] = set()
            if isinstance(payload, list):
                for value in payload:
                    try:
                        admin_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if admin_id > 0:
                        ids.add(admin_id)
            if not ids and self.fallback_admin_ids:
                ids.update(self.fallback_admin_ids)
                await self.store.set_json(_ADMIN_KEY, sorted(ids))
            self._cache = frozenset(ids)
            self._cache_at = now
            return self._cache

    async def save_ids(
        self,
        ids: set[int] | frozenset[int],
        *,
        updated_by: int | None = None,
    ) -> bool:
        clean = sorted({int(value) for value in ids if int(value) > 0})
        persistent = await self.store.set_json(_ADMIN_KEY, clean, updated_by=updated_by)
        self._cache = frozenset(clean)
        self._cache_at = time.monotonic()
        return persistent


_AUTHORIZER = TelegramAdminAuthorizer()


def configure_telegram_admin_authorizer(**kwargs: Any) -> TelegramAdminAuthorizer:
    return _AUTHORIZER.configure(**kwargs)


def get_telegram_admin_authorizer() -> TelegramAdminAuthorizer:
    return _AUTHORIZER


__all__ = [
    "TelegramAdminAuthorizer",
    "configure_telegram_admin_authorizer",
    "get_telegram_admin_authorizer",
]
