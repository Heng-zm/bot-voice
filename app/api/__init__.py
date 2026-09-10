"""Central API router registry aggregating all service routes."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.ai import router as ai_router
from app.api.routes.system import router as system_router
from app.api.routes.telegram import router as telegram_router
from app.api.routes.tts import router as tts_router

api_router = APIRouter()

# Register sub-routers
api_router.include_router(system_router)
api_router.include_router(telegram_router)
api_router.include_router(tts_router)
api_router.include_router(ai_router)

__all__ = [
    "api_router",
    "ai_router",
    "system_router",
    "telegram_router",
    "tts_router",
]
