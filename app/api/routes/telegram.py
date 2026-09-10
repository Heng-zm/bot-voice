"""Telegram Webhook and automated registration endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app import legacy
from app.core.config import get_detected_webhook_url
from app.core.security import validate_api_key
from app.services.telegram.dispatcher import get_telegram_dispatcher

router = APIRouter(tags=["Telegram Webhook & Lifecycle"])


@router.post("/webhook")
@router.post("/telegram/webhook")
@router.post("/tg-webhook-{path_token:path}")
@router.post("/tg-webhook/{path_token:path}")
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
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Return real-time metrics and worker health of the Telegram update dispatcher."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")
    return JSONResponse(get_telegram_dispatcher().get_metrics())


@router.get("/setup-webhook")
@router.post("/setup-webhook")
async def trigger_setup_webhook(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Manually force-register the detected URL as the Telegram Webhook (Protected)."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")
    url = get_detected_webhook_url()
    if not url:
        raise HTTPException(status_code=400, detail="No valid WEBHOOK_URL found in environment")
    from app.main import auto_setup_webhook

    ok = await auto_setup_webhook(url)
    return JSONResponse({
        "success": ok,
        "webhook_url": f"{url.rstrip('/')}/webhook" if url else "none",
    })


@router.get("/auto-register")
@router.post("/auto-register")
async def trigger_auto_register(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Trigger complete auto-registration sequence and return status (Protected)."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")
    from app.main import auto_register_all

    res = await auto_register_all()
    return JSONResponse(res)


@router.get("/webhook-info")
async def get_webhook_info(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Query Telegram for the active webhook status and pending update count (Protected)."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")
    app_instance = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
    if app_instance is None or getattr(app_instance, "bot", None) is None:
        return JSONResponse({"error": "Telegram application not ready"}, status_code=503)
    info = await app_instance.bot.get_webhook_info()
    return JSONResponse({
        "url": info.url,
        "has_custom_certificate": info.has_custom_certificate,
        "pending_update_count": info.pending_update_count,
        "last_error_date": str(info.last_error_date) if info.last_error_date else None,
        "last_error_message": info.last_error_message,
        "max_connections": info.max_connections,
    })


__all__ = ["router"]
