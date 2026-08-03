"""Core ASGI and combined Telegram-bot entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import legacy as _legacy
from app.api.v1.admin import router as admin_router
from app.api.v1.admin_cors import router as admin_cors_router
from app.api.v1.admin_runtime import router as admin_runtime_router
from app.api.v1.admin_users import router as admin_users_router
from app.core.cors import (
    DynamicCORSMiddleware,
    configure_dynamic_cors_store,
    get_dynamic_cors_store,
)
from app.core.security import get_runtime_secret_manager
from app.core.telegram_auth import configure_telegram_admin_authorizer
from app.services.jobs.runtime import configure_job_queue

# ASGI servers can use ``uvicorn app.main:app``.  The combined production
# process still uses ``python -m app.main`` so Telegram and background workers
# retain their existing lifecycle while the monolith is migrated in stages.
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


async def _initialize_runtime_services(application: FastAPI) -> None:
    """Initialize external clients, persistent secrets, and dynamic CORS."""

    _legacy.load_dotenv()
    if not str(getattr(_legacy, "REDIS_URL", "") or "").strip():
        _legacy._refresh_arch_runtime_settings()
        _legacy._init_clients()
        await _legacy._init_async_clients()
        try:
            await _legacy._restore_run_state_from_redis()
        except Exception as exc:  # noqa: BLE001 - existing optional restore path
            _legacy.webhook_logger.warning(
                "Runtime state restore failed during ASGI startup: %s",
                exc,
            )

    security_status = await _legacy._bootstrap_runtime_security()
    secret_manager = get_runtime_secret_manager()
    cors_store = configure_dynamic_cors_store(
        redis_client=_legacy.redis_client or secret_manager.redis_client,
        redis_url=str(
            getattr(_legacy, "REDIS_URL", "")
            or getattr(_legacy.SETTINGS, "REDIS_URL", "")
            or ""
        ),
        supabase_client=_legacy.supabase,
    )
    cors_snapshot = await cors_store.load(force=True)
    admin_authorizer = configure_telegram_admin_authorizer(
        redis_client=_legacy.redis_client or secret_manager.redis_client,
        fallback_admin_ids=getattr(_legacy, "ADMIN_IDS", set()),
    )
    redis_admin_ids = await admin_authorizer.load_ids(force=True)
    if redis_admin_ids:
        _legacy.ADMIN_IDS.update(redis_admin_ids)
    application.state.runtime_security = security_status
    application.state.dynamic_cors = cors_snapshot.as_dict()
    application.state.job_queue = configure_job_queue(
        _legacy.redis_client or secret_manager.redis_client,
        redis_prefix=str(
            getattr(_legacy, "REDIS_CACHE_PREFIX", "")
            or "tgbot"
        ),
    )


_original_lifespan_context = app.router.lifespan_context


@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[dict | None]:
    """Run persistent security initialization before accepting requests."""

    async with _original_lifespan_context(application) as lifespan_state:
        await _initialize_runtime_services(application)
        try:
            yield lifespan_state
        finally:
            configure_job_queue(None)
            get_dynamic_cors_store().close()
            get_runtime_secret_manager().close()


if not getattr(app.state, "_dynamic_security_installed", False):
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
    app.router.lifespan_context = application_lifespan
    app.state._dynamic_security_installed = True


def create_app() -> FastAPI:
    """Return the configured ASGI application."""
    return app


def main() -> None:
    """Run the existing combined web, Telegram, and scheduler lifecycle."""
    _legacy.main()


def __getattr__(name: str):
    """Keep uncommon legacy imports working during the staged migration."""
    return getattr(_legacy, name)


if __name__ == "__main__":
    main()
