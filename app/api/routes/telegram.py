"""Backward compatibility shim for app.api.routes.telegram."""

from __future__ import annotations

from app.api.webhook import (
    router,
    telegram_webhook_handler,
    telegram_webhook_probe,
)

__all__ = [
    "router",
    "telegram_webhook_handler",
    "telegram_webhook_probe",
]