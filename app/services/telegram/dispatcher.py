"""Backward compatibility shim for app.services.telegram.dispatcher."""

from __future__ import annotations

from app.bot.dispatcher import (
    TelegramDispatcher,
    get_telegram_dispatcher,
)

__all__ = [
    "TelegramDispatcher",
    "get_telegram_dispatcher",
]
