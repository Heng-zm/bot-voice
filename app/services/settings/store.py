"""Process-wide runtime settings store.

Redis used to coordinate admin IDs, runtime overrides and CORS policy.  The bot
now runs as one process and persists these small control-plane values in the
existing Supabase ``bot_settings`` table.  A bounded in-memory fallback keeps
local development usable when Supabase is intentionally not configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
    """Async facade over the synchronous Supabase client."""

    def __init__(self, supabase_client: Any | None = None) -> None:
        self.supabase = supabase_client
        self._memory: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @property
    def status(self) -> SettingsStoreStatus:
        persistent = self.supabase is not None
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

    async def get_text(self, key: str, default: str = "") -> str:
        clean = self._clean_key(key)
        if self.supabase is not None:
            try:
                value = await asyncio.to_thread(self._read_sync, clean)
                if value is not None:
                    self._memory[clean] = value
                    return value
            except Exception as exc:  # noqa: BLE001 - graceful DB degradation
                logger.warning("Settings read fell back to memory key=%s: %s", clean, exc)
        return self._memory.get(clean, default)

    async def set_text(
        self,
        key: str,
        value: Any,
        *,
        updated_by: int | None = None,
    ) -> bool:
        clean = self._clean_key(key)
        text = str(value)
        async with self._lock:
            if self.supabase is not None:
                try:
                    await asyncio.to_thread(
                        self._write_sync,
                        clean,
                        text,
                        updated_by,
                    )
                    self._memory[clean] = text
                    return True
                except Exception as exc:  # noqa: BLE001 - memory fallback is deliberate
                    logger.warning("Settings write fell back to memory key=%s: %s", clean, exc)
            self._memory[clean] = text
            return False

    async def get_json(self, key: str, default: Any) -> Any:
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
        return await self.set_text(
            key,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            updated_by=updated_by,
        )

    def _read_sync(self, key: str) -> str | None:
        result = (
            self.supabase.table("bot_settings")
            .select("value")
            .eq("key", key)
            .limit(1)
            .execute()
        )
        rows = list(getattr(result, "data", None) or [])
        if not rows:
            return None
        return str(rows[0].get("value") or "")

    def _write_sync(self, key: str, value: str, updated_by: int | None) -> None:
        payload: dict[str, Any] = {
            "key": key,
            "value": value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if updated_by is not None:
            payload["updated_by"] = int(updated_by)
        self.supabase.table("bot_settings").upsert(
            payload,
            on_conflict="key",
        ).execute()


_STORE = SettingsStore()


def configure_settings_store(supabase_client: Any | None) -> SettingsStore:
    global _STORE
    _STORE = SettingsStore(supabase_client)
    return _STORE


def get_settings_store() -> SettingsStore:
    return _STORE


def reset_settings_store() -> None:
    global _STORE
    _STORE = SettingsStore()


__all__ = [
    "SettingsStore",
    "SettingsStoreError",
    "SettingsStoreStatus",
    "configure_settings_store",
    "get_settings_store",
    "reset_settings_store",
]
