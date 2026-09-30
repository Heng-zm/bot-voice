"""Telegram Webhook and automated registration endpoints."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app import legacy
from app.core.config import get_detected_webhook_url
from app.core.security import validate_api_key
from app.services.telegram.dispatcher import get_telegram_dispatcher

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Telegram Webhook & Lifecycle"])


def _resolve_telegram_app() -> Any | None:
    """Resolve active Telegram Application instance from modern runner or legacy."""
    with suppress(Exception):
        from app.bot import get_global_telegram_app

        bot_app = get_global_telegram_app()
        if bot_app is not None and getattr(bot_app, "bot", None) is not None:
            return bot_app

    legacy_app = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
    if legacy_app is not None and getattr(legacy_app, "bot", None) is not None:
        return legacy_app

    return None


def _check_auth(
    x_api_key: str | None = None,
    authorization: str | None = None,
    api_key_query: str | None = None,
) -> None:
    """Validate API key from headers or optional query parameter fallback."""
    effective_key = x_api_key or api_key_query
    if not validate_api_key(effective_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")


@router.get("/webhook", include_in_schema=False)
@router.get("/telegram/webhook", include_in_schema=False)
async def telegram_webhook_probe() -> JSONResponse:
    """Lightweight health probe endpoint to prevent 405 errors from uptime monitors."""
    return JSONResponse({
        "status": "active",
        "service": "Telegram Webhook Receiver",
        "mode": "ready",
    })


@router.post("/webhook")
@router.post("/telegram/webhook")
@router.post("/tg-webhook-{path_token:path}")
@router.post("/tg-webhook/{path_token:path}")
@router.post("/webhook/{path_token:path}")
async def telegram_webhook(
    request: Request,
    path_token: str | None = None,
) -> Any:
    """Handle incoming Telegram webhook updates with deduplication and state tracking."""
    return await get_telegram_dispatcher().dispatch_webhook_request(request, path_token)


@router.get("/dispatcher/metrics")
@router.get("/dispatcher-status")
async def get_dispatcher_metrics(
    request: Request,
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Return real-time metrics and worker health of the Telegram update dispatcher (Protected)."""
    _check_auth(x_api_key, authorization, api_key)
    try:
        metrics = get_telegram_dispatcher().get_metrics()
        return JSONResponse(metrics)
    except Exception as exc:
        logger.error("Failed to retrieve dispatcher metrics: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/setup-webhook")
@router.post("/setup-webhook")
async def trigger_setup_webhook(
    request: Request,
    url: str | None = Query(default=None, description="Explicit webhook URL override"),
    drop_pending_updates: bool = Query(default=False, description="Drop queued updates on Telegram"),
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Manually force-register detected or specified URL as Telegram Webhook (Protected)."""
    _check_auth(x_api_key, authorization, api_key)

    target_url = (url or get_detected_webhook_url() or "").strip()
    if not target_url:
        raise HTTPException(
            status_code=400,
            detail="No valid WEBHOOK_URL found in environment or parameters",
        )

    # Normalize endpoint URL to avoid /webhook/webhook duplication
    endpoint_url = target_url if target_url.endswith("/webhook") else f"{target_url.rstrip('/')}/webhook"

    from app.main import auto_setup_webhook

    # Temporarily set drop_pending_updates environment flag if requested
    if drop_pending_updates:
        request.scope.setdefault("env_overrides", {})["DROP_PENDING_UPDATES"] = "true"

    ok = await auto_setup_webhook(target_url)
    return JSONResponse({
        "success": ok,
        "webhook_url": endpoint_url,
        "drop_pending_updates": drop_pending_updates,
    })


@router.get("/delete-webhook")
@router.post("/delete-webhook")
async def trigger_delete_webhook(
    request: Request,
    drop_pending_updates: bool = Query(default=False, description="Drop pending updates during deletion"),
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Remove active Telegram Webhook to enable local testing or polling mode (Protected)."""
    _check_auth(x_api_key, authorization, api_key)

    app_instance = _resolve_telegram_app()
    if app_instance is None or getattr(app_instance, "bot", None) is None:
        return JSONResponse({"error": "Telegram application not ready"}, status_code=503)

    try:
        ok = await app_instance.bot.delete_webhook(drop_pending_updates=drop_pending_updates)
        logger.info("Webhook deleted via API endpoint (drop_pending=%s)", drop_pending_updates)
        return JSONResponse({
            "success": ok,
            "message": "Telegram webhook removed successfully.",
            "drop_pending_updates": drop_pending_updates,
        })
    except Exception as exc:
        logger.error("Failed to delete webhook: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)


@router.get("/auto-register")
@router.post("/auto-register")
async def trigger_auto_register(
    request: Request,
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Trigger complete auto-registration sequence and return status (Protected)."""
    _check_auth(x_api_key, authorization, api_key)

    from app.main import auto_register_all

    app_instance = _resolve_telegram_app()
    res = await auto_register_all(app_instance)
    return JSONResponse(res)


@router.get("/webhook-info")
async def get_webhook_info(
    request: Request,
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Query Telegram for active webhook status, errors, and pending update count (Protected)."""
    _check_auth(x_api_key, authorization, api_key)

    app_instance = _resolve_telegram_app()
    if app_instance is None or getattr(app_instance, "bot", None) is None:
        return JSONResponse({"error": "Telegram application not ready"}, status_code=503)

    try:
        info = await app_instance.bot.get_webhook_info()
        return JSONResponse({
            "url": info.url or "",
            "has_custom_certificate": info.has_custom_certificate,
            "pending_update_count": info.pending_update_count,
            "ip_address": getattr(info, "ip_address", None),
            "last_error_date": str(info.last_error_date) if info.last_error_date else None,
            "last_error_message": info.last_error_message,
            "last_synchronization_error_date": (
                str(getattr(info, "last_synchronization_error_date", None))
                if getattr(info, "last_synchronization_error_date", None)
                else None
            ),
            "max_connections": info.max_connections,
            "allowed_updates": info.allowed_updates,
        })
    except Exception as exc:
        logger.error("Failed to fetch webhook info from Telegram: %s", exc)
        return JSONResponse({"error": f"Telegram API error: {exc}"}, status_code=502)


__all__ = ["router"]