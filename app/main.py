"""Telegram Voice Bot entry point with FastAPI Webhook and Health check endpoints."""

from __future__ import annotations

import asyncio
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

from app.bot import get_global_telegram_app, main, run_bot_async


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Start the Telegram Bot runner in the background when running under Uvicorn."""
    bot_task = asyncio.create_task(run_bot_async(), name="telegram-bot-runner")
    try:
        yield
    finally:
        bot_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await bot_task


app = FastAPI(title="Telegram Bot Voice API", lifespan=lifespan)


@app.get("/")
@app.get("/healthz")
@app.get("/ping")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "telegram-bot-voice"})


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

    telegram_app = get_global_telegram_app()
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


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
