"""Supabase-backed Telegram administrator policy for bot commands."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from typing import Any, Iterable

from app.services.settings.store import SettingsStore, get_settings_store

logger = logging.getLogger(__name__)

_ADMIN_KEY = "security:admin_user_ids:v2"


def _clean_admin_ids(raw_values: Any) -> set[int]:
    """Extract and validate positive integer admin IDs from various formats."""
    result: set[int] = set()
    if not raw_values:
        return result

    items: Iterable[Any]
    if isinstance(raw_values, str):
        items = raw_values.split(",")
    elif isinstance(raw_values, (set, frozenset, list, tuple)):
        items = raw_values
    else:
        items = [raw_values]

    for item in items:
        with suppress(ValueError, TypeError):
            val = int(str(item).strip())
            if val > 0:
                result.add(val)
    return result


def _discover_default_admin_ids() -> frozenset[int]:
    """Discover fallback admin IDs from environment variables and application SETTINGS."""
    found: set[int] = set()

    # 1. Environment variables
    for env_key in ("ADMIN_IDS", "TELEGRAM_ADMIN_IDS", "BOT_ADMIN_IDS"):
        env_val = os.getenv(env_key, "").strip()
        if env_val:
            found.update(_clean_admin_ids(env_val))

    # 2. Core settings store configuration
    with suppress(Exception):
        from app.core.config import SETTINGS

        for attr in ("ADMIN_IDS", "TELEGRAM_ADMIN_IDS"):
            settings_val = getattr(SETTINGS, attr, None)
            if settings_val:
                found.update(_clean_admin_ids(settings_val))

    return frozenset(found)


class TelegramAdminAuthorizer:
    """Maintain a short-lived administrator ID snapshot for command guards."""

    def __init__(self, *, cache_ttl_seconds: float = 5.0) -> None:
        self._store: SettingsStore | None = None
        self.fallback_admin_ids: frozenset[int] = _discover_default_admin_ids()
        self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))
        self._cache: frozenset[int] | None = None
        self._cache_at = 0.0
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    @property
    def store(self) -> SettingsStore:
        """Lazily retrieve settings store instance."""
        if self._store is None:
            self._store = get_settings_store()
        return self._store

    @store.setter
    def store(self, value: SettingsStore) -> None:
        self._store = value

    def _get_async_lock(self) -> asyncio.Lock:
        """Retrieve loop-aware asyncio lock, creating a new one if event loop restarted."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop != current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        return self._lock

    def configure(
        self,
        *,
        settings_store: SettingsStore | None = None,
        fallback_admin_ids: Any = (),
        cache_ttl_seconds: float | None = None,
        **_ignored: Any,
    ) -> TelegramAdminAuthorizer:
        """Configure runtime store and default fallback admin credentials."""
        if settings_store is not None:
            self.store = settings_store

        cleaned = _clean_admin_ids(fallback_admin_ids)
        if cleaned:
            self.fallback_admin_ids = frozenset(cleaned)
        elif not self.fallback_admin_ids:
            self.fallback_admin_ids = _discover_default_admin_ids()

        if cache_ttl_seconds is not None:
            self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))

        self.invalidate()
        return self

    def invalidate(self) -> None:
        """Invalidate in-memory cache to force a fresh reload on next query."""
        self._cache = None
        self._cache_at = 0.0

    def is_admin_sync(self, user_id: int | str | None) -> bool:
        """Synchronously check admin status without blocking Telegram's event loop."""
        try:
            candidate = int(user_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

        if candidate <= 0:
            return False

        # If cache is primed, verify against snapshot
        snapshot = self._cache
        if snapshot is not None:
            return candidate in snapshot

        # If cache has not yet been loaded, fall back to environment credentials
        return candidate in self.fallback_admin_ids

    async def is_admin(self, user_id: int | str | None) -> bool:
        """Asynchronously check admin status, ensuring the cache is active and fresh."""
        try:
            candidate = int(user_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

        if candidate <= 0:
            return False

        admin_ids = await self.load_ids()
        return candidate in admin_ids

    async def load_ids(self, *, force: bool = False) -> frozenset[int]:
        """Load administrative IDs from Supabase/database with TTL caching."""
        now = time.monotonic()
        if not force and self._cache is not None and (now - self._cache_at) < self.cache_ttl_seconds:
            return self._cache

        lock = self._get_async_lock()
        async with lock:
            now = time.monotonic()
            if not force and self._cache is not None and (now - self._cache_at) < self.cache_ttl_seconds:
                return self._cache

            ids: set[int] = set()
            try:
                payload = await self.store.get_json(_ADMIN_KEY, [])
                if isinstance(payload, list):
                    for value in payload:
                        with suppress(ValueError, TypeError):
                            admin_id = int(value)
                            if admin_id > 0:
                                ids.add(admin_id)
            except Exception as exc:
                logger.warning("Failed to fetch admin IDs from store: %s", exc)
                # Resilient fallback: return existing cache or fallback IDs
                if self._cache is not None:
                    return self._cache
                return self.fallback_admin_ids

            # Seed default admins from environment if store was empty
            if not ids and self.fallback_admin_ids:
                ids.update(self.fallback_admin_ids)
                with suppress(Exception):
                    await self.store.set_json(_ADMIN_KEY, sorted(ids))

            self._cache = frozenset(ids)
            self._cache_at = now
            return self._cache

    async def save_ids(
        self,
        ids: Iterable[int | str],
        *,
        updated_by: int | None = None,
    ) -> bool:
        """Persist administrative IDs to Supabase and update local cache."""
        clean_set = _clean_admin_ids(ids)
        if not clean_set and self.fallback_admin_ids:
            clean_set.update(self.fallback_admin_ids)

        clean_list = sorted(clean_set)
        persistent = False
        try:
            persistent = await self.store.set_json(_ADMIN_KEY, clean_list, updated_by=updated_by)
        except Exception as exc:
            logger.error("Failed to persist updated admin IDs: %s", exc)

        self._cache = frozenset(clean_list)
        self._cache_at = time.monotonic()
        return persistent

    async def add_admin(self, user_id: int | str, *, updated_by: int | None = None) -> bool:
        """Add an administrator ID to the policy."""
        try:
            target = int(user_id)
        except (ValueError, TypeError):
            return False

        if target <= 0:
            return False

        current_ids = set(await self.load_ids())
        if target in current_ids:
            return True

        current_ids.add(target)
        return await self.save_ids(current_ids, updated_by=updated_by)

    async def remove_admin(self, user_id: int | str, *, updated_by: int | None = None) -> bool:
        """Remove an administrator ID (with protection against emptying all admins)."""
        try:
            target = int(user_id)
        except (ValueError, TypeError):
            return False

        current_ids = set(await self.load_ids())
        if target not in current_ids:
            return True

        # Lockout protection: Do not allow removing the only remaining administrator
        if len(current_ids) <= 1:
            logger.warning("Attempted to remove the only remaining admin ID: %s", target)
            return False

        current_ids.remove(target)
        return await self.save_ids(current_ids, updated_by=updated_by)


# Global singleton instance
_AUTHORIZER = TelegramAdminAuthorizer()


def configure_telegram_admin_authorizer(**kwargs: Any) -> TelegramAdminAuthorizer:
    return _AUTHORIZER.configure(**kwargs)


def get_telegram_admin_authorizer() -> TelegramAdminAuthorizer:
    return _AUTHORIZER


def is_telegram_admin(user_id: int | str | None) -> bool:
    """Convenience synchronous check against active admin policy."""
    return _AUTHORIZER.is_admin_sync(user_id)


async def is_telegram_admin_async(user_id: int | str | None) -> bool:
    """Convenience asynchronous check against active admin policy."""
    return await _AUTHORIZER.is_admin(user_id)


__all__ = [
    "TelegramAdminAuthorizer",
    "configure_telegram_admin_authorizer",
    "get_telegram_admin_authorizer",
    "is_telegram_admin",
    "is_telegram_admin_async",
]