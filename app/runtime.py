"""Single-process application runtime.

The dedicated Redis queue/worker topology has been removed.  FastAPI, Telegram,
admin controls and provider execution now share one process; small control-plane
state is persisted through the Supabase-backed settings store.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from typing import Any, Literal

from fastapi import FastAPI

from app import legacy as _legacy
from app.core.cors import configure_dynamic_cors_store, get_dynamic_cors_store
from app.core.telegram_auth import configure_telegram_admin_authorizer
from app.services.ai.providers import get_provider_manager
from app.services.settings.store import (
    SettingsStore,
    configure_settings_store,
    reset_settings_store,
)
from app.services.telegram.deduplication import get_webhook_replay_snapshot
from app.services.telegram.workloads import get_telegram_workload_limiter

logger = logging.getLogger(__name__)
RuntimeRole = Literal["web", "combined"]


def _clean_role(value: str) -> RuntimeRole:
    role = str(value or "combined").strip().lower()
    if role == "worker":
        raise ValueError(
            "PROCESS_ROLE=worker is no longer supported. Remove the worker service "
            "and run app.main as a single web/Telegram process."
        )
    if role not in {"web", "combined"}:
        raise ValueError("Runtime role must be web or combined.")
    return role  # type: ignore[return-value]


class RuntimeContext:
    """Own startup/shutdown of the single-process runtime exactly once."""

    def __init__(self, legacy: Any = _legacy) -> None:
        self.legacy = legacy
        self.settings: Any | None = None
        self.supabase: Any | None = None
        self.security: dict[str, Any] = {}
        self.settings_store: SettingsStore | None = None
        self.cors_store: Any | None = None
        self.admin_authorizer: Any | None = None
        self.role: RuntimeRole | None = None
        self.started = False
        self.started_at = 0.0
        self._owners: dict[str, RuntimeRole] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        application: FastAPI | None = None,
        *,
        owner: str,
        role: str | None = None,
    ) -> None:
        clean_owner = str(owner or "runtime").strip()[:64] or "runtime"
        requested_role = _clean_role(role or os.getenv("PROCESS_ROLE", "combined"))
        async with self._lock:
            if clean_owner in self._owners:
                return
            if self.started:
                self._owners[clean_owner] = requested_role
                self.role = self._effective_role()
                if application is not None:
                    await self._ensure_web_services(application)
                return

            try:
                self.legacy.load_dotenv()
                self.legacy._refresh_arch_runtime_settings()
                self.legacy._init_clients()
                await self.legacy._init_async_clients()

                # Redis is intentionally disabled even when an old deployment
                # still has REDIS_URL configured. Existing cache helpers degrade
                # to memory/Supabase paths in legacy.py.
                redis_client = getattr(self.legacy, "redis_client", None)
                if redis_client is not None:
                    close = getattr(redis_client, "close", None)
                    if callable(close):
                        with suppress(Exception):
                            close()
                    self.legacy.redis_client = None
                    logger.warning("REDIS_URL is ignored by the single-process runtime.")

                self.settings = self.legacy.SETTINGS
                self.supabase = self.legacy.supabase
                self.settings_store = configure_settings_store(self.supabase)

                # Compatibility name now restores from Supabase bot_settings.
                with suppress(Exception):
                    await self.legacy._restore_run_state()
                self.security = dict(await self.legacy._bootstrap_runtime_security() or {})

                self.started_at = time.time()
                self.started = True
                self._owners[clean_owner] = requested_role
                self.role = requested_role

                if application is not None:
                    await self._ensure_web_services(application)

                logger.info(
                    "Single-process runtime started owner=%s role=%s settings_backend=%s",
                    clean_owner,
                    requested_role,
                    self.settings_store.status.backend,
                )
            except BaseException:
                await self._stop_unlocked()
                raise

    def _effective_role(self) -> RuntimeRole:
        return "combined" if "combined" in self._owners.values() else "web"

    async def _ensure_web_services(self, application: FastAPI) -> None:
        if self.settings_store is None:
            raise RuntimeError("Runtime settings store is unavailable.")
        if self.cors_store is None:
            self.cors_store = configure_dynamic_cors_store(
                settings_store=self.settings_store,
            )
            cors_snapshot = await self.cors_store.load(force=True)
            self.admin_authorizer = configure_telegram_admin_authorizer(
                settings_store=self.settings_store,
                fallback_admin_ids=getattr(self.legacy, "ADMIN_IDS", set()),
            )
            admin_ids = await self.admin_authorizer.load_ids(force=True)
            if admin_ids:
                self.legacy.ADMIN_IDS.update(admin_ids)
            bootstrap_admin = getattr(self.legacy, "_runtime_admin_bootstrap_state", None)
            if callable(bootstrap_admin):
                bootstrap_admin()
            application.state.dynamic_cors = cors_snapshot.as_dict()

        application.state.runtime = self
        application.state.runtime_security = self.security
        application.state.provider_scope = get_provider_manager().metadata()
        application.state.settings_store = self.settings_store.status.as_dict()

    async def stop(self, *, owner: str) -> None:
        clean_owner = str(owner or "runtime").strip()[:64] or "runtime"
        async with self._lock:
            self._owners.pop(clean_owner, None)
            if self._owners:
                self.role = self._effective_role()
                return
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        if self.cors_store is not None:
            with suppress(Exception):
                self.cors_store.close()
        else:
            with suppress(Exception):
                get_dynamic_cors_store().close()
        reset_settings_store()
        self.settings = None
        self.supabase = None
        self.security = {}
        self.settings_store = None
        self.cors_store = None
        self.admin_authorizer = None
        self.role = None
        self.started = False
        self.started_at = 0.0
        self._owners.clear()

    def snapshot(self) -> dict[str, Any]:
        store_status = (
            self.settings_store.status.as_dict()
            if self.settings_store is not None
            else {"backend": "unconfigured", "persistent": False, "configured": False}
        )
        return {
            "started": self.started,
            "started_at": self.started_at or None,
            "role": self.role,
            "owners": sorted(self._owners),
            "supabase": self.supabase is not None,
            "security": bool(self.security),
            "settings_store": store_status,
            "architecture": "single_process",
            "worker_removed": True,
            "redis_removed": True,
            "providers": get_provider_manager().metadata(),
            "telegram_workloads": get_telegram_workload_limiter().snapshot(),
            "webhook_replay": get_webhook_replay_snapshot(),
        }


_RUNTIME = RuntimeContext()


def get_runtime_context() -> RuntimeContext:
    return _RUNTIME


__all__ = ["RuntimeContext", "get_runtime_context"]
