"""Telegram user security, command guards, and rate-limited security notifications."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress
from typing import Any

try:
    from telegram import Update
except (ImportError, ModuleNotFoundError):
    Update = Any  # type: ignore[assignment,misc]

from app._legacy_bridge import legacy_module

ADMIN_ONLY_COMMANDS: frozenset[str] = frozenset({
    "admin",
    "stats",
    "broadcast",
    "schedule",
    "schedules",
    "cancelschedule",
    "runtime",
    "health",
    "api",
    "botsettings",
    "users",
    "chat",
    "endchat",
})

_ADMIN_SECURITY_NOTICE_MEMORY: dict[str, float] = {}
_ADMIN_SECURITY_NOTICE_MEMORY_LOCK = threading.RLock()


def telegram_command_name(update: Update) -> str:
    """Extract lowercase command name without leading slash or @bot suffix."""
    msg = update.effective_message
    text = str(getattr(msg, "text", None) or getattr(msg, "caption", None) or "").strip()
    if not text.startswith("/"):
        return ""
    return text[1:].split(maxsplit=1)[0].split("@", 1)[0].lower()


async def security_notice_once(
    update: Update,
    key: str,
    text: str,
    *,
    alert: bool = False,
    cooldown_seconds: float = 5.0,
) -> None:
    """Send rate-limited security rejection notice to prevent alert spam."""
    now = time.monotonic()
    cooldown = max(0.5, float(cooldown_seconds))

    with _ADMIN_SECURITY_NOTICE_MEMORY_LOCK:
        last = _ADMIN_SECURITY_NOTICE_MEMORY.get(key, 0.0)
        if now - last < cooldown:
            return
        _ADMIN_SECURITY_NOTICE_MEMORY[key] = now

        # Evict stale entries when map exceeds threshold
        if len(_ADMIN_SECURITY_NOTICE_MEMORY) > 10_000:
            stale_before = now - max(cooldown * 4, 300.0)
            for old_key, old_ts in list(_ADMIN_SECURITY_NOTICE_MEMORY.items()):
                if old_ts < stale_before:
                    _ADMIN_SECURITY_NOTICE_MEMORY.pop(old_key, None)
            while len(_ADMIN_SECURITY_NOTICE_MEMORY) > 10_000:
                _ADMIN_SECURITY_NOTICE_MEMORY.pop(next(iter(_ADMIN_SECURITY_NOTICE_MEMORY)), None)

    query = update.callback_query
    if query is not None:
        with suppress(Exception):
            await query.answer(text, show_alert=alert)
        return

    msg = update.effective_message
    if msg is not None:
        with suppress(Exception):
            await msg.reply_text(text)


async def is_user_blocked(user_id: int) -> bool:
    """Return block state without scheduling a worker for cache hits.

    The synchronous database helper owns the authoritative bounded TTL cache.
    Checking that cache on the event loop is cheap and thread-safe; only a true
    cache miss needs the bounded database executor.
    """
    clean_user_id = int(user_id)
    legacy = legacy_module()
    cached = legacy._blocked_cache_get(clean_user_id)
    if cached is not None:
        return bool(cached)
    return bool(
        await asyncio.get_running_loop().run_in_executor(
            legacy._DB_EXECUTOR,
            legacy.db_user_is_blocked,
            clean_user_id,
        )
    )


# Compatibility aliases
_ADMIN_ONLY_COMMANDS = ADMIN_ONLY_COMMANDS
_telegram_command_name = telegram_command_name
_security_notice_once = security_notice_once

__all__ = [
    "ADMIN_ONLY_COMMANDS",
    "_ADMIN_ONLY_COMMANDS",
    "_security_notice_once",
    "_telegram_command_name",
    "is_user_blocked",
    "security_notice_once",
    "telegram_command_name",
]
