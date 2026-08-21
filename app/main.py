"""Core ASGI and combined Telegram-bot entry point."""

from __future__ import annotations

# Support deployment panels that execute this file directly as
# ``python app/main.py``. In that mode Python places ``app/`` itself on
# ``sys.path``; add the repository root before importing the ``app`` package.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    _repo_root = str(_BootstrapPath(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import asyncio
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
from app.api.v1.admin_v2 import router as admin_v2_router
from app.core.cors import DynamicCORSMiddleware
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
    "style-src 'self'; "
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
    ready = bool(snapshot["started"] and snapshot["security"])
    return JSONResponse(
        {
            "ok": ready,
            "web": True,
            "status": "ready" if ready else "starting",
            "runtime_started": snapshot["started"],
            "role": snapshot.get("role"),
            "supabase": snapshot.get("supabase"),
            "settings_store": snapshot.get("settings_store"),
            "telegram": snapshot.get("telegram"),
            "architecture": snapshot.get("architecture"),
            "redis_removed": True,
            "worker_removed": True,
        },
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


if not getattr(app.state, "_runtime_bootstrap_installed", False):
    app.include_router(admin_router)
    app.include_router(admin_cors_router)
    app.include_router(admin_runtime_router)
    app.include_router(admin_users_router)
    app.include_router(admin_v2_router)
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
        max_age=60,
    )
    app.middleware("http")(_runtime_ready_middleware)
    app.router.lifespan_context = application_lifespan
    app.state._runtime_bootstrap_installed = True


def create_app() -> FastAPI:
    """Return the configured ASGI application."""

    return app


async def _combined_main_once() -> None:
    """Run web, Telegram and schedulers under one RuntimeContext."""

    runtime = get_runtime_context()
    await runtime.start(app, owner="combined", role="combined")

    _legacy.logger.info(
        "Single-process runtime starting provider=%s bot_mode=%s",
        getattr(_legacy, "AI_PROVIDER", "unknown"),
        _legacy._run_state_bot_mode(),
    )

    _legacy._start_web_broadcast_queue_workers()
    critical_tasks: list[asyncio.Task] = [
        asyncio.create_task(_legacy.run_fastapi(), name="fastapi-web"),
        asyncio.create_task(_legacy._run_bot(), name="telegram-bot"),
    ]
    auxiliary_tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _legacy._run_startup_background_checks(),
            name="startup-background-checks",
        )
    ]
    tasks = [*critical_tasks, *auxiliary_tasks]

    try:
        # FastAPI and Telegram are both required for the combined process.
        # FIRST_EXCEPTION can leave the process half-alive when either service
        # returns normally, so treat completion of either critical task as a
        # supervisor event and restart the whole process.
        done, _pending = await asyncio.wait(
            critical_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                raise error
        names = ", ".join(sorted(task.get_name() for task in done)) or "unknown"
        raise RuntimeError(f"Critical runtime service stopped unexpectedly: {names}")
    finally:
        await _legacy._stop_web_broadcast_queue_workers()
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        await runtime.stop(owner="combined")


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
