"""Central API router registry aggregating all service routes."""

from __future__ import annotations

try:
    from fastapi import APIRouter
except (ImportError, ModuleNotFoundError, AttributeError):
    class APIRouter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            self.routes = []
        def __getattr__(self, name: str):
            return lambda *args, **kwargs: (lambda f: f)
        def include_router(self, *args, **kwargs) -> None:
            pass

from app.api.health import router as health_router
from app.api.logs import router as logs_router
from app.api.routes.ai import router as ai_router
from app.api.tts import router as tts_router
from app.api.webhook import router as webhook_router

api_router = APIRouter()

# Register sub-routers
api_router.include_router(health_router)
api_router.include_router(logs_router)
api_router.include_router(webhook_router)
api_router.include_router(tts_router)
api_router.include_router(ai_router)

__all__ = [
    "ai_router",
    "api_router",
    "health_router",
    "logs_router",
    "tts_router",
    "webhook_router",
]
