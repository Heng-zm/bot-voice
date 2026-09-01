"""Telegram Voice Bot entry point with FastAPI health check support."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import AsyncGenerator

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.bot import main, run_bot_async


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Start the Telegram Bot in the background when running under Uvicorn."""
    bot_task = asyncio.create_task(run_bot_async(), name="telegram-bot-runner")
    try:
        yield
    finally:
        bot_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await bot_task


app = FastAPI(title="Telegram Bot Voice Health API", lifespan=lifespan)


@app.get("/")
@app.get("/healthz")
@app.get("/ping")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "telegram-bot-voice"})


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
