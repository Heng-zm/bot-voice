"""Core ASGI and combined Telegram-bot entry point."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

from app import legacy as _legacy
from app.api.v1.admin import router as admin_router
from app.api.v1.admin_cors import router as admin_cors_router
from app.api.v1.admin_runtime import router as admin_runtime_router
from app.api.v1.admin_users import router as admin_users_router
from app.core.cors import DynamicCORSMiddleware, get_dynamic_cors_store
from app.runtime import get_runtime_context
from app.services.build_info import get_build_info
from app.services.incidents import (
    configure_incident_alert_handler,
    record_component_event,
    send_incident_alert,
)
from app.services.monitoring import discover_public_url, sanitize_monitor_text
from app.services.supervision import (
    ComponentSupervisor,
    SupervisorPolicy,
    is_configuration_failure,
)

# The legacy FastAPI object remains the compatibility shell during extraction.
# All new startup ownership now lives in RuntimeContext rather than legacy.py.
app: FastAPI = _legacy.app
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ADMIN_STATIC_DIR = _PROJECT_ROOT / "static" / "admin"
_ADMIN_INDEX_FILE = _ADMIN_STATIC_DIR / "index.html"
_ADMIN_CSP = (
    "default-src 'self'; "
    "base-uri 'none'; "
    "object-src 'none'; "
    "form-action 'self'; "
    "frame-ancestors https://web.telegram.org https://*.telegram.org; "
    "script-src 'self' https://telegram.org; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: https:; "
    "connect-src 'self'"
)

_original_lifespan_context = app.router.lifespan_context


@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[dict | None]:
    """Acquire the shared runtime once for this ASGI lifespan."""

    runtime = get_runtime_context()
    async with _original_lifespan_context(application) as lifespan_state:
        await runtime.start(application, owner="asgi", role="web")
        try:
            yield lifespan_state
        finally:
            await runtime.stop(owner="asgi")


def _runtime_ready_response() -> JSONResponse:
    """Build the readiness response without entering the route stack."""

    snapshot = get_runtime_context().snapshot()
    workers = snapshot["workers"]
    redis_ready = not snapshot.get("redis_enabled", True) or snapshot["redis"]
    ready = bool(
        snapshot["started"]
        and redis_ready
        and snapshot["security"]
        and snapshot["job_queue"]
        and snapshot["artifacts"].get("configured")
        and snapshot["delivery"]
        and (not snapshot["expects_workers"] or workers.get("healthy"))
    )
    return JSONResponse(
        {
            "ok": ready,
            "web": True,
            "status": "ready" if ready else "starting",
            "runtime_started": snapshot["started"],
            "redis": snapshot["redis"],
            "job_queue": snapshot["job_queue"],
            "job_queue_backend": snapshot["job_queue_backend"],
            "job_queue_durable": snapshot["job_queue_durable"],
            "role": snapshot.get("role"),
            "artifact_storage": snapshot.get("artifacts"),
            "delivery": snapshot.get("delivery"),
            "workers": {
                "count": workers.get("count", 0),
                "alive": workers.get("alive", 0),
                "healthy": workers.get("healthy", False),
                "accepting": workers.get("accepting", False),
            },
        },
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


class RuntimeReadyMiddleware:
    """Serve readiness through a minimal ASGI path.

    A function-based FastAPI middleware uses ``BaseHTTPMiddleware``, which
    creates an extra task/stream boundary for every request. Health probes are
    frequent, so a small pure-ASGI branch keeps both probes and unrelated API
    responses off that additional scheduling path.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") == "/readyz":
            await _runtime_ready_response()(scope, receive, send)
            return
        await self.app(scope, receive, send)


class ServerFastPathMiddleware:
    """Consolidate request telemetry and lightweight guards in one ASGI layer."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @staticmethod
    def _request_id(value: str) -> str:
        candidate = str(value or "").strip()[:64]
        if candidate and all(
            character.isascii()
            and (character.isalnum() or character in "-._:")
            for character in candidate
        ):
            return candidate
        return secrets.token_hex(8)

    @staticmethod
    def _backend_disabled_response() -> _legacy._FastJSONResponse:
        return _legacy._FastJSONResponse(
            {
                "ok": False,
                "code": "admin_frontend_disabled",
                "error": (
                    "HTML admin dashboard pages are disabled. Deploy the frontend "
                    "separately and call this backend API."
                ),
                "backend_only": True,
                "frontend_url": _legacy._admin_frontend_url(),
                "auth": {
                    "login": "/api/admin/auth/login",
                    "me": "/api/admin/auth/me",
                    "logout": "/api/admin/auth/logout",
                },
                "endpoints": {
                    "health": "/readyz",
                    "bootstrap": "/api/admin/bootstrap",
                    "status": "/admin/status.json",
                    "live": "/admin/live.json",
                    "performance": "/admin/performance.json",
                    "errors": "/admin/errors.json",
                    "broadcast_jobs": "/admin/broadcast/jobs.json",
                    "report_pdf": "/admin/report.pdf",
                },
            },
            status_code=410,
        )

    async def _guard_response(
        self,
        scope: Scope,
        receive: Receive,
        method: str,
        path: str,
        headers: Headers,
    ) -> _legacy._FastJSONResponse | None:
        if (
            _legacy._admin_origin_guard_enabled()
            and method in {"POST", "PUT", "PATCH", "DELETE"}
            and (path.startswith("/api/admin") or path.startswith("/admin/"))
        ):
            origin = _legacy._normalise_origin(headers.get("origin") or "")
            if origin and not _legacy._admin_origin_allowed(origin):
                logging.getLogger(__name__).warning(
                    "Blocked admin API request from disallowed origin "
                    "origin=%s method=%s path=%s",
                    origin,
                    method,
                    path,
                )
                return _legacy._FastJSONResponse(
                    _legacy._admin_origin_guard_payload(origin),
                    status_code=403,
                )

        if _legacy._backend_only_mode() and _legacy._is_legacy_admin_frontend_request(
            path,
            method,
        ):
            return self._backend_disabled_response()

        if path.startswith("/ai-assistant") and not await _legacy._api_rate_limit_allowed(
            Request(scope, receive=receive)
        ):
            return _legacy._FastJSONResponse(
                {
                    "ok": False,
                    "error": "Too many requests. Please slow down.",
                    "code": 429,
                },
                status_code=429,
            )
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method") or "GET").upper()
        path = str(scope.get("path") or "")
        headers = Headers(scope=scope)
        guarded = await self._guard_response(scope, receive, method, path, headers)

        request_id = self._request_id(headers.get("x-request-id") or "")
        started = time.monotonic()
        with _legacy._WEB_ACTIVE_REQUESTS_LOCK:
            _legacy._WEB_ACTIVE_REQUESTS += 1
            active_now = _legacy._WEB_ACTIVE_REQUESTS
        response_started = False

        def finish_response(message) -> None:
            nonlocal response_started
            if response_started:
                return
            response_started = True
            with _legacy._WEB_ACTIVE_REQUESTS_LOCK:
                _legacy._WEB_ACTIVE_REQUESTS = max(
                    0,
                    _legacy._WEB_ACTIVE_REQUESTS - 1,
                )

            elapsed_ms = (time.monotonic() - started) * 1000.0
            mutable = MutableHeaders(scope=message)
            mutable.setdefault("X-Request-ID", request_id)
            mutable.setdefault("X-Response-Time-ms", f"{elapsed_ms:.1f}")
            mutable.setdefault("X-Active-Requests", str(active_now))

            slow_threshold_ms = _legacy.WEB_SLOW_REQUEST_MS
            if path.startswith("/tg-webhook-") or path == "/telegram-webhook":
                slow_threshold_ms = max(
                    _legacy.WEB_SLOW_REQUEST_MS,
                    _legacy._env_float(
                        "WEBHOOK_SLOW_REQUEST_MS",
                        10_000.0,
                        minimum=1_000.0,
                        maximum=120_000.0,
                    ),
                )
            if elapsed_ms < slow_threshold_ms:
                return
            item = {
                "ts": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "method": method,
                "path": path,
                "status": int(message.get("status") or 0),
                "elapsed_ms": round(elapsed_ms, 1),
                "active": int(active_now),
            }
            _legacy._WEB_SLOW_REQUESTS.appendleft(item)
            logging.getLogger(__name__).warning(
                "Slow web request request_id=%s method=%s path=%s status=%s "
                "elapsed_ms=%.1f active=%s",
                request_id,
                method,
                path,
                item["status"],
                elapsed_ms,
                active_now,
            )

        async def send_with_timing(message) -> None:
            if message["type"] == "http.response.start":
                finish_response(message)
            await send(message)

        try:
            if guarded is not None:
                await guarded(scope, receive, send_with_timing)
            else:
                await self.app(scope, receive, send_with_timing)
        except Exception:
            logging.getLogger(__name__).exception(
                "Unhandled web request error request_id=%s method=%s path=%s active=%s",
                request_id,
                method,
                path,
                active_now,
            )
            raise
        finally:
            if not response_started:
                with _legacy._WEB_ACTIVE_REQUESTS_LOCK:
                    _legacy._WEB_ACTIVE_REQUESTS = max(
                        0,
                        _legacy._WEB_ACTIVE_REQUESTS - 1,
                    )


if not getattr(app.state, "_runtime_bootstrap_installed", False):
    # These compatibility guards were originally registered with FastAPI's
    # function decorator, producing four nested BaseHTTPMiddleware task/stream
    # boundaries. ServerFastPathMiddleware preserves their behavior in one
    # pure-ASGI pass.
    legacy_dispatches = {
        _legacy._admin_origin_guard_middleware,
        _legacy._backend_only_legacy_dashboard_guard,
        _legacy._hot_api_rate_limit_middleware,
        _legacy._web_request_id_and_timing,
    }
    app.user_middleware[:] = [
        middleware
        for middleware in app.user_middleware
        if middleware.kwargs.get("dispatch") not in legacy_dispatches
    ]

    app.include_router(admin_router)
    app.include_router(admin_cors_router)
    app.include_router(admin_runtime_router)
    app.include_router(admin_users_router)
    app.mount(
        "/miniapp/admin/assets",
        StaticFiles(directory=str(_ADMIN_STATIC_DIR), check_dir=False),
        name="admin-miniapp-assets",
    )

    @app.get("/miniapp/admin", include_in_schema=False)
    @app.get("/miniapp/admin/", include_in_schema=False)
    async def telegram_admin_mini_app() -> FileResponse:
        """Serve the public shell; every data request is authenticated."""

        return FileResponse(
            _ADMIN_INDEX_FILE,
            media_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": _ADMIN_CSP,
                "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
                "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/version", tags=["health"])
    @app.get("/api/version", tags=["health"])
    async def version_metadata() -> dict:
        snapshot = get_runtime_context().snapshot()
        workers = snapshot["workers"]
        return {
            "ok": True,
            "build": get_build_info(
                role=snapshot.get("role"),
                started_at=snapshot.get("started_at"),
            ),
            "runtime": {
                "started": bool(snapshot.get("started")),
                "role": snapshot.get("role") or "combined",
                "workers": {
                    "count": int(workers.get("count", 0) or 0),
                    "alive": int(workers.get("alive", 0) or 0),
                    "healthy": bool(workers.get("healthy", False)),
                },
            },
        }

    app.add_middleware(ServerFastPathMiddleware)
    app.add_middleware(
        DynamicCORSMiddleware,
        store=get_dynamic_cors_store(),
        max_age=60,
    )
    app.add_middleware(RuntimeReadyMiddleware)
    app.router.lifespan_context = application_lifespan
    app.state._runtime_bootstrap_installed = True


def create_app() -> FastAPI:
    """Return the configured ASGI application."""

    return app


async def _send_admin_incident_alert(event: dict) -> None:
    """Deliver a bounded incident alert even when the PTB component is down."""

    token = str(getattr(_legacy, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    admin_ids = sorted(
        int(value)
        for value in getattr(_legacy, "ADMIN_IDS", set())
        if str(value).strip().lstrip("-").isdigit()
    )
    if not token or not admin_ids:
        return
    severity = str(event.get("severity") or "info").lower()
    icon = {
        "critical": "🚨",
        "error": "❌",
        "warning": "⚠️",
        "info": "✅",
    }.get(severity, "ℹ️")
    public_url = discover_public_url().get("url") or "not detected"
    text = sanitize_monitor_text(
        f"{icon} Bot incident\n"
        f"Component: {event.get('component') or 'runtime'}\n"
        f"Event: {event.get('event') or 'status'}\n"
        f"State: {event.get('state') or 'unknown'}\n"
        f"Detail: {event.get('message') or '-'}\n"
        f"Public URL: {public_url}",
        limit=1800,
    )
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    timeout = _legacy.httpx.Timeout(10.0)
    async with _legacy.httpx.AsyncClient(timeout=timeout) as client:
        for admin_id in admin_ids:
            try:
                response = await client.post(
                    api_url,
                    json={
                        "chat_id": admin_id,
                        "text": text,
                        "disable_web_page_preview": True,
                    },
                )
                if response.status_code >= 400:
                    _legacy.logger.warning(
                        "Incident alert delivery failed admin_id=%s status=%s.",
                        admin_id,
                        response.status_code,
                    )
            except Exception as exc:  # notification must not break recovery
                _legacy.logger.warning(
                    "Incident alert delivery failed admin_id=%s: %s",
                    admin_id,
                    sanitize_monitor_text(exc, limit=240),
                )


async def _combined_main_once() -> None:
    """Run web, Telegram, schedulers, and workers under one RuntimeContext."""

    runtime = get_runtime_context()
    await runtime.start(app, owner="combined", role="combined")
    configure_incident_alert_handler(_send_admin_incident_alert)

    _legacy.logger.info(
        "Combined runtime starting provider=%s bot_mode=%s durable_workers=%s",
        getattr(_legacy, "AI_PROVIDER", "unknown"),
        _legacy._run_state_bot_mode(),
        runtime.snapshot()["workers"].get("count", 0),
    )

    _legacy._start_web_broadcast_queue_workers()
    keepalive_stop = asyncio.Event()
    component_stop = asyncio.Event()
    policy = SupervisorPolicy(
        base_backoff_seconds=1.0,
        max_backoff_seconds=60.0,
        stable_run_seconds=60.0,
        max_configuration_failures=3,
    )
    web_supervisor = ComponentSupervisor("web", _legacy.run_fastapi, policy=policy)
    telegram_supervisor = ComponentSupervisor(
        "telegram",
        _legacy._run_bot,
        policy=policy,
    )
    component_tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(web_supervisor.run(component_stop), name="supervisor-web"),
        asyncio.create_task(
            telegram_supervisor.run(component_stop),
            name="supervisor-telegram",
        ),
    ]
    tasks: list[asyncio.Task[None]] = [
        *component_tasks,
        asyncio.create_task(
            _legacy._run_startup_background_checks(),
            name="startup-background-checks",
        ),
    ]
    if discover_public_url().get("url"):
        tasks.append(
            asyncio.create_task(
                _legacy.keep_alive_async(keepalive_stop),
                name="async-keep-alive",
            )
        )

    try:
        # Web and Telegram own independent recovery loops. One failed component
        # no longer tears down its healthy sibling.
        await asyncio.gather(*component_tasks)
    finally:
        component_stop.set()
        keepalive_stop.set()
        await _legacy._stop_web_broadcast_queue_workers()
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        await runtime.stop(owner="combined")


async def _wait_for_critical_tasks(tasks: list[asyncio.Task[None]]) -> None:
    """Return when a critical service stops, propagating its exception."""

    if not tasks:
        raise ValueError("At least one critical task is required.")
    done, _pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in done:
        if task.cancelled():
            continue
        error = task.exception()
        if error is not None:
            raise error


def main() -> None:
    """Run startup with bounded backoff and a configuration circuit breaker."""

    failure_streak = 0
    configuration_failures = 0
    while True:
        try:
            asyncio.run(_combined_main_once())
        except KeyboardInterrupt:
            _legacy.logger.info("Shutdown requested.")
            break
        except Exception as exc:  # noqa: BLE001 - process supervisor boundary
            failure_streak += 1
            configuration_failure = is_configuration_failure(exc)
            if configuration_failure:
                configuration_failures += 1
            else:
                configuration_failures = 0
            circuit_open = configuration_failures >= 3
            delay = min(60.0, float(2 ** min(failure_streak - 1, 6)))
            event = record_component_event(
                "runtime",
                "startup_failed",
                severity="critical" if circuit_open else "error",
                message=f"{type(exc).__name__}: {exc}",
                state="circuit_open" if circuit_open else "backoff",
                restart_count=failure_streak,
                consecutive_failures=failure_streak,
                next_retry_seconds=None if circuit_open else delay,
                configuration_failure=configuration_failure,
            )
            # The component supervisors normally send these alerts. This
            # boundary also covers failures that happen before they start.
            configure_incident_alert_handler(_send_admin_incident_alert)
            with suppress(Exception):
                asyncio.run(send_incident_alert(event))
            _legacy.logger.error(
                "Runtime startup failed: %s — %s",
                exc,
                (
                    "automatic restarts stopped"
                    if circuit_open
                    else f"retrying in {delay:g}s"
                ),
                exc_info=True,
            )
            if circuit_open:
                break
            time.sleep(delay)
        else:
            failure_streak = 0
            configuration_failures = 0
            _legacy.logger.warning("Runtime stopped — restarting in 1s...")
            time.sleep(1)


def __getattr__(name: str):
    """Keep uncommon legacy imports working during staged migration."""

    return getattr(_legacy, name)


if __name__ == "__main__":
    main()
