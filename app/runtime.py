"""Single idempotent startup/shutdown boundary for web and worker roles."""

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
from app.core.security import get_runtime_secret_manager
from app.core.telegram_auth import configure_telegram_admin_authorizer
from app.services.ai.providers import get_provider_manager
from app.services.artifacts.storage import (
    ArtifactService,
    configure_artifact_service,
    reset_artifact_service,
)
from app.services.jobs.handlers import BotJobHandlers, build_bot_job_handlers
from app.services.jobs.runtime import (
    configure_job_queue,
    job_worker_snapshot,
    start_job_workers,
    stop_job_workers,
)
from app.services.telegram.delivery import (
    IdempotentTelegramDelivery,
    configure_telegram_delivery,
)

logger = logging.getLogger(__name__)
RuntimeRole = Literal["web", "worker", "combined"]


def _clean_role(value: str) -> RuntimeRole:
    role = str(value or "combined").strip().lower()
    if role not in {"web", "worker", "combined"}:
        raise ValueError("Runtime role must be web, worker, or combined.")
    return role  # type: ignore[return-value]


def _role_has_workers(role: RuntimeRole) -> bool:
    return role in {"worker", "combined"}


def _role_has_web(role: RuntimeRole) -> bool:
    return role in {"web", "combined"}


class RuntimeContext:
    """Own external clients, queues, artifacts, delivery, and workers once."""

    def __init__(self, legacy: Any = _legacy) -> None:
        self.legacy = legacy
        self.settings: Any | None = None
        self.redis: Any | None = None
        self.supabase: Any | None = None
        self.security: dict[str, Any] = {}
        self.cors_store: Any | None = None
        self.admin_authorizer: Any | None = None
        self.job_queue: Any | None = None
        self.artifacts: ArtifactService | None = None
        self.delivery: IdempotentTelegramDelivery | None = None
        self.job_handlers: BotJobHandlers | None = None
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
        requested_role = _clean_role(
            role or os.getenv("PROCESS_ROLE", "combined") or "combined"
        )
        async with self._lock:
            if clean_owner in self._owners:
                return
            if self.started:
                self._owners[clean_owner] = requested_role
                if _role_has_workers(requested_role):
                    await self._ensure_workers()
                if _role_has_web(requested_role) and application is not None:
                    await self._ensure_web_services(application)
                self.role = self._effective_role()
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
                if redis_client is None:
                    raise RuntimeError("Redis is required for the durable runtime.")
                redis_prefix = str(
                    getattr(self.legacy, "REDIS_CACHE_PREFIX", "") or "tgbot"
                )

                self.settings = self.legacy.SETTINGS
                self.redis = redis_client
                self.supabase = self.legacy.supabase
                self.security = dict(security_status or {})
                self.job_queue = configure_job_queue(
                    redis_client,
                    redis_prefix=redis_prefix,
                    max_queued_jobs=int(
                        os.getenv("BOT_JOB_QUEUE_MAX", "1000") or 1000
                    ),
                )
                if self.job_queue is None:
                    raise RuntimeError("Could not configure the durable job queue.")
                self.artifacts = configure_artifact_service(
                    supabase_client=self.supabase,
                    role=requested_role,
                )
                self.delivery = configure_telegram_delivery(
                    redis_client,
                    redis_prefix=redis_prefix,
                )
                if self.delivery is None:
                    raise RuntimeError("Could not configure Telegram delivery.")

                self.started_at = time.time()
                self.started = True
                self._owners[clean_owner] = requested_role
                self.role = requested_role

                if _role_has_web(requested_role) and application is not None:
                    await self._ensure_web_services(application)
                if _role_has_workers(requested_role):
                    await self._ensure_workers()

                logger.info(
                    "Runtime started owner=%s role=%s workers=%s artifact_backend=%s",
                    clean_owner,
                    requested_role,
                    job_worker_snapshot().get("count"),
                    self.artifacts.backend,
                )
            except BaseException:
                await self._stop_unlocked()
                raise

    def _effective_role(self) -> RuntimeRole:
        roles = set(self._owners.values())
        if "combined" in roles or {"web", "worker"}.issubset(roles):
            return "combined"
        if "worker" in roles:
            return "worker"
        return "web"

    async def _ensure_web_services(self, application: FastAPI) -> None:
        if self.redis is None:
            raise RuntimeError("Runtime Redis is unavailable.")
        if self.cors_store is None:
            redis_url = str(
                getattr(self.legacy, "REDIS_URL", "")
                or getattr(self.legacy.SETTINGS, "REDIS_URL", "")
                or ""
            )
            self.cors_store = configure_dynamic_cors_store(
                redis_client=self.redis,
                redis_url=redis_url,
                supabase_client=self.supabase,
            )
            cors_snapshot = await self.cors_store.load(force=True)
            self.admin_authorizer = configure_telegram_admin_authorizer(
                redis_client=self.redis,
                fallback_admin_ids=getattr(self.legacy, "ADMIN_IDS", set()),
            )
            redis_admin_ids = await self.admin_authorizer.load_ids(force=True)
            if redis_admin_ids:
                self.legacy.ADMIN_IDS.update(redis_admin_ids)
            bootstrap_admin = getattr(self.legacy, "_runtime_admin_bootstrap_state", None)
            if callable(bootstrap_admin):
                bootstrap_admin()
            application.state.dynamic_cors = cors_snapshot.as_dict()

        application.state.runtime = self
        application.state.runtime_security = self.security
        application.state.job_queue = self.job_queue
        application.state.provider_scope = get_provider_manager().metadata()
        application.state.artifact_storage = {
            "backend": self.artifacts.backend if self.artifacts else "unconfigured",
            "shared": self.artifacts.shared if self.artifacts else False,
        }

    async def _ensure_workers(self) -> None:
        if self.job_handlers is not None and job_worker_snapshot().get("count", 0):
            return
        if self.artifacts is None or self.delivery is None:
            raise RuntimeError("Worker dependencies are not configured.")
        handlers = build_bot_job_handlers(
            self.legacy,
            artifacts=self.artifacts,
            delivery=self.delivery,
        )
        self.job_handlers = handlers
        await start_job_workers(
            handlers.mapping(),
            worker_count=int(os.getenv("BOT_JOB_WORKERS", "2") or 2),
        )

    async def stop(self, *, owner: str) -> None:
        clean_owner = str(owner or "runtime").strip()[:64] or "runtime"
        async with self._lock:
            self._owners.pop(clean_owner, None)
            if self._owners:
                self.role = self._effective_role()
                return
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        await stop_job_workers()
        handlers, self.job_handlers = self.job_handlers, None
        if handlers is not None:
            await handlers.close()
        configure_job_queue(None)
        configure_telegram_delivery(None)
        reset_artifact_service()

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
        self.artifacts = None
        self.delivery = None
        self.role = None
        self.started = False
        self.started_at = 0.0
        self._owners.clear()

    def snapshot(self) -> dict[str, Any]:
        workers = job_worker_snapshot()
        expects_workers = bool(self.role and _role_has_workers(self.role))
        return {
            "started": self.started,
            "started_at": self.started_at or None,
            "role": self.role,
            "owners": sorted(self._owners),
            "redis": self.redis is not None,
            "supabase": self.supabase is not None,
            "security": bool(self.security),
            "job_queue": self.job_queue is not None,
            "artifacts": {
                "configured": self.artifacts is not None,
                "backend": self.artifacts.backend if self.artifacts else None,
                "shared": self.artifacts.shared if self.artifacts else False,
            },
            "delivery": self.delivery is not None,
            "expects_workers": expects_workers,
            "workers": workers,
            "providers": get_provider_manager().metadata(),
        }


_RUNTIME = RuntimeContext()


def get_runtime_context() -> RuntimeContext:
    return _RUNTIME


__all__ = ["RuntimeContext", "RuntimeRole", "get_runtime_context"]
