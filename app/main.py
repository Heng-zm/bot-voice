"""Telegram Voice Bot entry point with Auto Webhook URL capture and FastAPI endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from telegram import Update

from app import legacy

logger = logging.getLogger("app.webhook")

_DETECTED_WEBHOOK_URL: str = ""


def get_detected_webhook_url() -> str:
    """Retrieve auto-detected webhook URL from platform env variables."""
    global _DETECTED_WEBHOOK_URL
    if _DETECTED_WEBHOOK_URL:
        return _DETECTED_WEBHOOK_URL
    for env_var in (
        "WEBHOOK_URL",
        "RENDER_EXTERNAL_URL",
        "RAILWAY_STATIC_URL",
        "RAILWAY_PUBLIC_DOMAIN",
        "KOYEB_PUBLIC_DOMAIN",
        "VERCEL_URL",
    ):
        val = (os.environ.get(env_var) or "").strip().rstrip("/")
        if val:
            if not val.startswith("http://") and not val.startswith("https://"):
                val = f"https://{val}"
            _DETECTED_WEBHOOK_URL = val
            return val
    return ""


async def auto_setup_webhook(app_url: str) -> bool:
    """Automatically register the captured webhook URL with Telegram."""
    global _DETECTED_WEBHOOK_URL
    clean_url = app_url.rstrip("/")
    if not clean_url.startswith("https://"):
        return False

    _DETECTED_WEBHOOK_URL = clean_url
    webhook_endpoint = f"{clean_url}/webhook"

    app_instance = None
    for _ in range(30):
        app_instance = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
        if app_instance is not None and getattr(app_instance, "bot", None) is not None:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        logger.warning("Could not auto-register webhook: Telegram application not ready.")
        return False

    secret = (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or "").strip()
    for attempt in range(5):
        try:
            await app_instance.bot.set_webhook(
                url=webhook_endpoint,
                secret_token=secret or None,
                allowed_updates=["message", "edited_message", "callback_query"],
                drop_pending_updates=False,
            )
            logger.info("Auto-captured and registered Telegram Webhook at %s", webhook_endpoint)
            return True
        except Exception as exc:
            retry_after = getattr(exc, "retry_after", None)
            delay = (float(retry_after) + 0.5) if retry_after is not None else (attempt + 1.0)
            logger.warning("auto_setup_webhook retry (%ss): %s", delay, exc)
            await asyncio.sleep(delay)
    return False


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Start the Telegram Bot runner and auto-capture webhook URL."""
    bot_task = asyncio.create_task(legacy._async_main_once(), name="telegram-bot-runner")

    async def _setup_known_webhook() -> None:
        url = get_detected_webhook_url()
        if url:
            await auto_setup_webhook(url)

    webhook_task = asyncio.create_task(_setup_known_webhook(), name="webhook-auto-setup")

    try:
        yield
    finally:
        webhook_task.cancel()
        bot_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await bot_task


app = FastAPI(title="Telegram Bot Voice API", lifespan=lifespan)


@app.get("/")
@app.get("/healthz")
@app.get("/ping")
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint that auto-captures host header for Webhook URL."""
    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    if host_header and not host_header.startswith("localhost") and not host_header.startswith("127.0.0.1"):
        detected = f"{proto}://{host_header}"
        if not get_detected_webhook_url():
            asyncio.create_task(auto_setup_webhook(detected))

    return JSONResponse({
        "status": "ok",
        "service": "telegram-bot-voice",
        "webhook_url": get_detected_webhook_url() or "polling_mode",
    })


@app.get("/setup-webhook")
async def trigger_setup_webhook(request: Request) -> JSONResponse:
    """Manually force-register the current URL as the Telegram Webhook."""
    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    url = f"{proto}://{host_header}" if host_header and not host_header.startswith("localhost") else get_detected_webhook_url()
    ok = await auto_setup_webhook(url)
    return JSONResponse({
        "success": ok,
        "webhook_url": f"{url.rstrip('/')}/webhook" if url else "none",
    })


@app.get("/webhook-info")
async def get_webhook_info() -> JSONResponse:
    """Query Telegram for the active webhook status and pending update count."""
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


@app.post("/webhook")
@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> JSONResponse:
    """Handle incoming Telegram webhook updates with secret token verification."""
    expected_secret = (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or "").strip()
    if expected_secret and x_telegram_bot_api_secret_token != expected_secret:
        raise HTTPException(status_code=403, detail="Invalid secret token")

    telegram_app = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
    if telegram_app is None:
        raise HTTPException(status_code=503, detail="Telegram application is initializing")

    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        if update is not None:
            await telegram_app.process_update(update)
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


def main() -> None:
    legacy.main()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
