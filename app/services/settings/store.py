"""Process-wide runtime settings store.

Persists small control-plane values in the Supabase ``bot_settings`` table
with an in-memory TTL cache and fallback for local development.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


class SettingsStoreError(RuntimeError):
    """Raised when a persistent settings operation cannot be completed."""


@dataclass(frozen=True)
class SettingsStoreStatus:
    backend: str
    persistent: bool
    configured: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "persistent": self.persistent,
            "configured": self.configured,
        }


class SettingsStore:
    """Async facade over the synchronous Supabase client with local TTL caching."""

    def __init__(self, supabase_client: Any | None = None, *, cache_ttl_seconds: float = 5.0) -> None:
        self.supabase = supabase_client
        self._memory: dict[str, str] = {}
        self._cache_ts: dict[str, float] = {}
        self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily initialize and re-bind asyncio lock to the active event loop."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop != current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        return self._lock

    def _get_supabase(self) -> Any | None:
        """Retrieve client or lazily discover it from project modules."""
        if self.supabase is not None:
            return self.supabase

        with suppress(Exception):
            from app.core.supabase import get_supabase_client

            client = get_supabase_client()
            if client is not None:
                self.supabase = client
                return self.supabase

        with suppress(Exception):
            from app import legacy

            client = getattr(legacy, "_supabase", None) or getattr(legacy, "supabase", None)
            if client is not None:
                self.supabase = client
                return self.supabase

        return None

    def configure(
        self,
        supabase_client: Any | None = None,
        *,
        cache_ttl_seconds: float | None = None,
    ) -> SettingsStore:
        """Update active configuration in place without breaking existing references."""
        if supabase_client is not None:
            self.supabase = supabase_client
        if cache_ttl_seconds is not None:
            self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))
        return self

    @property
    def status(self) -> SettingsStoreStatus:
        client = self._get_supabase()
        persistent = client is not None
        return SettingsStoreStatus(
            backend="supabase" if persistent else "memory",
            persistent=persistent,
            configured=True,
        )

    @staticmethod
    def _clean_key(key: str) -> str:
        clean = str(key or "").strip()
        if not clean or len(clean) > 160:
            raise ValueError("Settings key must contain 1-160 characters.")
        return clean

    def clear_cache(self) -> None:
        """Clear all in-memory cached entries."""
        self._memory.clear()
        self._cache_ts.clear()

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate a specific key or all keys from cache."""
        if key is None:
            self.clear_cache()
            return
        clean = self._clean_key(key)
        self._cache_ts.pop(clean, None)
        self._memory.pop(clean, None)

    async def get_text(self, key: str, default: str = "") -> str:
        """Retrieve string setting with memory cache and negative-caching protection."""
        clean = self._clean_key(key)
        now = time.monotonic()

        # Cache Hit Check (both positive hits and negative misses within TTL)
        if clean in self._cache_ts and (now - self._cache_ts[clean] < self.cache_ttl_seconds):
            return self._memory.get(clean, default)

        supabase = self._get_supabase()
        if supabase is not None:
            try:
                value = await asyncio.to_thread(self._read_sync, clean)
                self._cache_ts[clean] = now
                if value is not None:
                    self._memory[clean] = value
                    return value
            except Exception as exc:
                logger.warning("Settings read fell back to memory key=%s: %s", clean, exc)

        # Cache negative lookup timestamp to prevent repeating DB queries for absent keys
        self._cache_ts[clean] = now
        return self._memory.get(clean, default)

    async def get_many_text(
        self,
        keys: Iterable[str],
        default: str = "",
    ) -> dict[str, str]:
        """Load multiple settings in a single batch query with TTL caching."""
        clean_keys = tuple(dict.fromkeys(self._clean_key(key) for key in keys))
        if not clean_keys:
            return {}

        now = time.monotonic()
        stale_or_missing = [
            k
            for k in clean_keys
            if k not in self._cache_ts or (now - self._cache_ts[k] >= self.cache_ttl_seconds)
        ]

        values: dict[str, str] = {}
        supabase = self._get_supabase()

        if stale_or_missing and supabase is not None:
            try:
                fetched = await asyncio.to_thread(self._read_many_sync, tuple(stale_or_missing))
                # Update timestamps for all requested keys (including absent keys)
                for k in stale_or_missing:
                    self._cache_ts[k] = now
                for k, v in fetched.items():
                    self._memory[k] = v
                values.update(fetched)
            except Exception as exc:
                logger.warning("Settings batch read fell back to memory keys=%d: %s", len(stale_or_missing), exc)

        return {
            key: values.get(key, self._memory.get(key, default))
            for key in clean_keys
        }

    async def set_text(
        self,
        key: str,
        value: Any,
        *,
        updated_by: int | None = None,
    ) -> bool:
        """Persist setting string to Supabase and update local cache."""
        clean = self._clean_key(key)
        text = str(value)
        now = time.monotonic()
        lock = self._get_lock()

        async with lock:
            self._memory[clean] = text
            self._cache_ts[clean] = now
            supabase = self._get_supabase()

            if supabase is not None:
                try:
                    await asyncio.to_thread(self._write_sync, clean, text, updated_by)
                    return True
                except Exception as exc:
                    logger.warning("Settings write fell back to memory key=%s: %s", clean, exc)
                    return False

            return True

    async def get_json(self, key: str, default: Any = None) -> Any:
        """Retrieve and parse JSON setting."""
        raw = await self.get_text(key, "")
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid JSON settings value key=%s", key)
            return default

    async def set_json(
        self,
        key: str,
        value: Any,
        *,
        updated_by: int | None = None,
    ) -> bool:
        """Serialize and persist value as JSON string."""
        return await self.set_text(
            key,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            updated_by=updated_by,
        )

    async def get_bool(self, key: str, default: bool = False) -> bool:
        """Retrieve setting as a boolean value."""
        raw = await self.get_text(key, "")
        if not raw:
            return default
        val = raw.strip().lower()
        if val in ("true", "1", "yes", "on", "enable", "enabled"):
            return True
        if val in ("false", "0", "no", "off", "disable", "disabled"):
            return False
        return default

    async def set_bool(self, key: str, value: bool, *, updated_by: int | None = None) -> bool:
        """Persist boolean setting."""
        return await self.set_text(key, "true" if value else "false", updated_by=updated_by)

    async def get_int(self, key: str, default: int = 0) -> int:
        """Retrieve setting as an integer."""
        raw = await self.get_text(key, "")
        if not raw:
            return default
        try:
            return int(raw.strip())
        except (ValueError, TypeError):
            return default

    async def set_int(self, key: str, value: int, *, updated_by: int | None = None) -> bool:
        """Persist integer setting."""
        return await self.set_text(key, str(int(value)), updated_by=updated_by)

    async def get_float(self, key: str, default: float = 0.0) -> float:
        """Retrieve setting as a float."""
        raw = await self.get_text(key, "")
        if not raw:
            return default
        try:
            return float(raw.strip())
        except (ValueError, TypeError):
            return default

    async def set_float(self, key: str, value: float, *, updated_by: int | None = None) -> bool:
        """Persist float setting."""
        return await self.set_text(key, str(float(value)), updated_by=updated_by)

    async def delete_setting(self, key: str) -> bool:
        """Delete setting from Supabase and cache."""
        clean = self._clean_key(key)
        lock = self._get_lock()

        async with lock:
            self._memory.pop(clean, None)
            self._cache_ts.pop(clean, None)
            supabase = self._get_supabase()

            if supabase is not None:
                try:
                    await asyncio.to_thread(
                        lambda: supabase.table("bot_settings").delete().eq("key", clean).execute()
                    )
                    return True
                except Exception as exc:
                    logger.warning("Settings delete error key=%s: %s", clean, exc)
                    return False

            return True

    # Standard alias
    delete = delete_setting

    def get_text_sync(self, key: str, default: str = "") -> str:
        """Synchronously check memory cache with fallback DB query on cache miss."""
        clean = self._clean_key(key)
        now = time.monotonic()

        if clean in self._cache_ts and (now - self._cache_ts.get(clean, 0.0) < self.cache_ttl_seconds):
            return self._memory.get(clean, default)

        supabase = self._get_supabase()
        if supabase is not None:
            try:
                val = self._read_sync(clean)
                self._cache_ts[clean] = now
                if val is not None:
                    self._memory[clean] = val
                    return val
            except Exception as exc:
                logger.warning("get_text_sync fell back to memory key=%s: %s", clean, exc)

        self._cache_ts[clean] = now
        return self._memory.get(clean, default)

    async def preload_all(self) -> dict[str, str]:
        """Pre-warm memory cache with all settings from the database in a single query."""
        supabase = self._get_supabase()
        if supabase is None:
            return dict(self._memory)

        def _fetch_all() -> dict[str, str]:
            res = (
                supabase.table("bot_settings")
                .select("key,value")
                .limit(1000)
                .execute()
            )
            items: dict[str, str] = {}
            for row in list(getattr(res, "data", None) or []):
                k = str(row.get("key") or "").strip()
                if k:
                    items[k] = str(row.get("value") or "")
            return items

        try:
            now = time.monotonic()
            data = await asyncio.to_thread(_fetch_all)
            lock = self._get_lock()
            async with lock:
                self._memory.update(data)
                for k in data:
                    self._cache_ts[k] = now
            logger.info("Preloaded %d settings from database into memory.", len(data))
            return data
        except Exception as exc:
            logger.warning("Failed to preload settings: %s", exc)
            return dict(self._memory)

    def _read_sync(self, key: str) -> str | None:
        supabase = self._get_supabase()
        if supabase is None:
            return None

        result = (
            supabase.table("bot_settings")
            .select("value")
            .eq("key", key)
            .limit(1)
            .execute()
        )
        rows = list(getattr(result, "data", None) or [])
        if not rows:
            return None
        return str(rows[0].get("value") or "")

    def _read_many_sync(self, keys: tuple[str, ...]) -> dict[str, str]:
        supabase = self._get_supabase()
        if supabase is None:
            return {}

        result = (
            supabase.table("bot_settings")
            .select("key,value")
            .in_("key", list(keys))
            .execute()
        )
        values: dict[str, str] = {}
        target_keys = set(keys)
        for row in list(getattr(result, "data", None) or []):
            key = str(row.get("key") or "").strip()
            if key in target_keys:
                values[key] = str(row.get("value") or "")
        return values

    def _write_sync(self, key: str, value: str, updated_by: int | None) -> None:
        supabase = self._get_supabase()
        if supabase is None:
            return

        payload: dict[str, Any] = {
            "key": key,
            "value": value,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if updated_by is not None:
            payload["updated_by"] = int(updated_by)

        supabase.table("bot_settings").upsert(
            payload,
            on_conflict="key",
        ).execute()


# Global singleton instance
_STORE = SettingsStore()


def configure_settings_store(
    supabase_client: Any | None = None,
    *,
    cache_ttl_seconds: float = 5.0,
) -> SettingsStore:
    """Configure or update the global settings store singleton in place."""
    return _STORE.configure(supabase_client, cache_ttl_seconds=cache_ttl_seconds)


def get_settings_store() -> SettingsStore:
    """Retrieve global settings store instance."""
    return _STORE


def reset_settings_store() -> None:
    """Reset the settings store cache and client connection."""
    global _STORE
    _STORE.clear_cache()
    _STORE.supabase = None


__all__ = [
    "SettingsStore",
    "SettingsStoreError",
    "SettingsStoreStatus",
    "configure_settings_store",
    "get_settings_store",
    "reset_settings_store",
]