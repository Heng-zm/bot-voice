"""Single idempotent startup/shutdown boundary for the combined application."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from typing import Any

from fastapi import FastAPI

from app import legacy as _legacy
from app.core.cors import configure_dynamic_cors_store, get_dynamic_cors_store
from app.core.security import get_runtime_secret_manager
from app.core.telegram_auth import configure_telegram_admin_authorizer
from app.services.ai.providers import get_provider_manager
from app.services.jobs.handlers import BotJobHandlers, build_bot_job_handlers
from app.services.jobs.runtime import (
    configure_job_queue,
    job_worker_snapshot,
    start_job_workers,
    stop_job_workers,
)

logger = logging.getLogger(__name__)


class RuntimeContext:
    """Own external clients, security stores, queues, and workers once per process."""

    def __init__(self, legacy: Any = _legacy) -> None:
        self.legacy = legacy
        self.settings: Any | None = None
        self.redis: Any | None = None
        self.supabase: Any | None = None
        self.security: dict[str, Any] = {}
        self.cors_store: Any | None = None
        self.admin_authorizer: Any | None = None
        self.job_queue: Any | None = None
        self.job_handlers: BotJobHandlers | None = None
        self.started = False
        self.started_at = 0.0
        self._owners: set[str] = set()
        self._lock = asyncio.Lock()

    async def start(self, application: FastAPI, *, owner: str) -> None:
        clean_owner = str(owner or "runtime").strip()[:64] or "runtime"
        async with self._lock:
            if clean_owner in self._owners:
                return
            if self.started:
                self._owners.add(clean_owner)
                return

            try:
                self.legacy.load_dotenv()
                self.legacy._refresh_arch_runtime_settings()
                self.legacy._init_clients()
                await self.legacy._init_async_clients()
                try:
                    await self.legacy._restore_run_state_from_redis()
                except Exception as exc:  # noqa: BLE001 - optional persisted state
                    self.legacy.webhook_logger.warning(
                        "Runtime state restore failed during startup: %s",
                        exc,
                    )

                security_status = await self.legacy._bootstrap_runtime_security()
                secret_manager = get_runtime_secret_manager()
                redis_client = self.legacy.redis_client or secret_manager.redis_client
                redis_url = str(
                    getattr(self.legacy, "REDIS_URL", "")
                    or getattr(self.legacy.SETTINGS, "REDIS_URL", "")
                    or ""
                )
                redis_prefix = str(
                    getattr(self.legacy, "REDIS_CACHE_PREFIX", "") or "tgbot"
                )

                cors_store = configure_dynamic_cors_store(
                    redis_client=redis_client,
                    redis_url=redis_url,
                    supabase_client=self.legacy.supabase,
                )
                cors_snapshot = await cors_store.load(force=True)

                admin_authorizer = configure_telegram_admin_authorizer(
                    redis_client=redis_client,
                    fallback_admin_ids=getattr(self.legacy, "ADMIN_IDS", set()),
                )
                redis_admin_ids = await admin_authorizer.load_ids(force=True)
                if redis_admin_ids:
                    self.legacy.ADMIN_IDS.update(redis_admin_ids)
                bootstrap_admin = getattr(
                    self.legacy,
                    "_runtime_admin_bootstrap_state",
                    None,
                )
                if callable(bootstrap_admin):
                    bootstrap_admin()

                self.settings = self.legacy.SETTINGS
                self.redis = redis_client
                self.supabase = self.legacy.supabase
                self.security = dict(security_status or {})
                self.cors_store = cors_store
                self.admin_authorizer = admin_authorizer

                queue = configure_job_queue(
                    redis_client,
                    redis_prefix=redis_prefix,
                    max_queued_jobs=int(
                        os.getenv("BOT_JOB_QUEUE_MAX", "1000") or 1000
                    ),
                )
                if queue is None:
                    raise RuntimeError(
                        "Redis is required for the durable production job queue."
                    )
                self.job_queue = queue

                handlers = build_bot_job_handlers(self.legacy)
                self.job_handlers = handlers
                await start_job_workers(
                    handlers.mapping(),
                    worker_count=int(os.getenv("BOT_JOB_WORKERS", "2") or 2),
                )
                self.started_at = time.time()
                self.started = True
                self._owners.add(clean_owner)

                application.state.runtime = self
                application.state.runtime_security = self.security
                application.state.dynamic_cors = cors_snapshot.as_dict()
                application.state.job_queue = queue
                application.state.provider_scope = get_provider_manager().metadata()
                logger.info(
                    "Runtime started owner=%s workers=%s provider_scope=process",
                    clean_owner,
                    job_worker_snapshot().get("count"),
                )
            except BaseException:
                await self._stop_unlocked()
                raise

    async def stop(self, *, owner: str) -> None:
        clean_owner = str(owner or "runtime").strip()[:64] or "runtime"
        async with self._lock:
            self._owners.discard(clean_owner)
            if self._owners:
                return
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        await stop_job_workers()
        handlers, self.job_handlers = self.job_handlers, None
        if handlers is not None:
            await handlers.close()
        configure_job_queue(None)

        if self.cors_store is not None:
            with suppress(Exception):
                self.cors_store.close()
        else:
            with suppress(Exception):
                get_dynamic_cors_store().close()
        with suppress(Exception):
            get_runtime_secret_manager().close()

        redis_client = self.redis
        if redis_client is not None:
            close = getattr(redis_client, "close", None)
            if callable(close):
                with suppress(Exception):
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result

        self.settings = None
        self.redis = None
        self.supabase = None
        self.security = {}
        self.cors_store = None
        self.admin_authorizer = None
        self.job_queue = None
        self.started = False
        self.started_at = 0.0
        self._owners.clear()

    def snapshot(self) -> dict[str, Any]:
        workers = job_worker_snapshot()
        return {
            "started": self.started,
            "started_at": self.started_at or None,
            "owners": sorted(self._owners),
            "redis": self.redis is not None,
            "supabase": self.supabase is not None,
            "security": bool(self.security),
            "job_queue": self.job_queue is not None,
            "workers": workers,
            "providers": get_provider_manager().metadata(),
        }


_RUNTIME = RuntimeContext()


def get_runtime_context() -> RuntimeContext:
    return _RUNTIME


__all__ = ["RuntimeContext", "get_runtime_context"]
