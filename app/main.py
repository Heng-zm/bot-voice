"""Core ASGI and combined Telegram-bot entry point."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import legacy as _legacy
from app.api.v1.admin import router as admin_router
from app.api.v1.admin_cors import router as admin_cors_router
from app.api.v1.admin_runtime import router as admin_runtime_router
from app.api.v1.admin_users import router as admin_users_router
from app.core.cors import DynamicCORSMiddleware, get_dynamic_cors_store
from app.runtime import get_runtime_context

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


async def _runtime_ready_middleware(request: Request, call_next):
    if request.url.path != "/readyz":
        return await call_next(request)
    snapshot = get_runtime_context().snapshot()
    workers = snapshot["workers"]
    ready = bool(
        snapshot["started"]
        and snapshot["redis"]
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


if not getattr(app.state, "_runtime_bootstrap_installed", False):
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

    app.add_middleware(
        DynamicCORSMiddleware,
        store=get_dynamic_cors_store(),
        max_age=60,
    )
    app.middleware("http")(_runtime_ready_middleware)
    app.router.lifespan_context = application_lifespan
    app.state._runtime_bootstrap_installed = True


def create_app() -> FastAPI:
    """Return the configured ASGI application."""

    return app


async def _combined_main_once() -> None:
    """Run web, Telegram, schedulers, and workers under one RuntimeContext."""

    runtime = get_runtime_context()
    await runtime.start(app, owner="combined", role="combined")

    _legacy.logger.info(
        "Combined runtime starting provider=%s bot_mode=%s durable_workers=%s",
        getattr(_legacy, "AI_PROVIDER", "unknown"),
        _legacy._run_state_bot_mode(),
        runtime.snapshot()["workers"].get("count", 0),
    )

    _legacy._start_web_broadcast_queue_workers()
    keepalive_stop = asyncio.Event()
    web_task = asyncio.create_task(_legacy.run_fastapi(), name="fastapi-web")
    bot_task = asyncio.create_task(_legacy._run_bot(), name="telegram-bot")
    tasks: list[asyncio.Task[None]] = [
        web_task,
        bot_task,
        asyncio.create_task(
            _legacy._run_startup_background_checks(),
            name="startup-background-checks",
        ),
    ]
    if str(
        os.environ.get("RENDER_EXTERNAL_URL")
        or getattr(_legacy.SETTINGS, "RENDER_EXTERNAL_URL", "")
        or ""
    ).strip():
        tasks.append(
            asyncio.create_task(
                _legacy.keep_alive_async(keepalive_stop),
                name="async-keep-alive",
            )
        )

    try:
        # Startup checks are finite by design, while the web and Telegram
        # loops are both critical long-running services. If either critical
        # loop returns normally, tear down the partial runtime so main() can
        # restart the complete service instead of remaining half alive.
        await _wait_for_critical_tasks([web_task, bot_task])
    finally:
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
    """Run the combined lifecycle with the existing crash-restart policy."""

    while True:
        try:
            asyncio.run(_combined_main_once())
        except KeyboardInterrupt:
            _legacy.logger.info("Shutdown requested.")
            break
        except Exception as exc:  # noqa: BLE001 - process supervisor boundary
            _legacy.logger.error(
                "Runtime crashed: %s — restarting in 5s...",
                exc,
                exc_info=True,
            )
            time.sleep(5)
        else:
            _legacy.logger.warning("Runtime stopped — restarting in 5s...")
            time.sleep(5)


def __getattr__(name: str):
    """Keep uncommon legacy imports working during staged migration."""

    return getattr(_legacy, name)


if __name__ == "__main__":
    main()
