"""System health and metrics telemetry endpoints."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app import legacy
from app.core.config import get_detected_webhook_url
from app.core.security import get_allowed_api_keys, validate_api_key

router = APIRouter(tags=["System & Telemetry"])


@router.get("/")
@router.head("/")
@router.get("/healthz")
@router.head("/healthz")
@router.get("/ping")
@router.head("/ping")
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint reporting live service and webhook status."""
    captured_url = get_detected_webhook_url()
    instance_id = (
        os.environ.get("ANAJAK_INSTANCE_ID")
        or os.environ.get("RENDER_INSTANCE_ID")
        or os.environ.get("HOSTNAME")
        or f"node-{os.getpid()}"
    )
    return JSONResponse({
        "status": "ok",
        "service": "telegram-bot-voice",
        "instance_id": instance_id,
        "webhook_url": f"{captured_url}/webhook" if captured_url else "polling_mode",
        "api_url": f"{captured_url}/ai-assistant" if captured_url else "/ai-assistant",
        "tts_url": f"{captured_url}/tts" if captured_url else "/tts",
    })


@router.get("/system")
@router.get("/metrics")
@router.get("/api/system")
@router.get("/api/metrics")
async def system_metrics_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Live system telemetry, resource health, and multi-provider stats (Protected if API keys configured)."""
    allowed_keys = get_allowed_api_keys()
    if allowed_keys and not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")

    snapshot_func = getattr(legacy, "_system_metrics_snapshot", None)
    if snapshot_func:
        loop = asyncio.get_running_loop()
        snapshot = await loop.run_in_executor(None, snapshot_func)
        return JSONResponse(snapshot)
    return JSONResponse({"status": "running", "version": "4.2.0"})


__all__ = ["router"]
