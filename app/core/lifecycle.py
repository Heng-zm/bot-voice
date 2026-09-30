"""Application lifecycle management, service supervisors, and graceful shutdown coordinators."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import urllib.request
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any

from app.config.settings import SETTINGS, get_detected_webhook_url
from app.core.concurrency.queue import get_task_manager

logger = logging.getLogger("app.lifecycle")


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _handle_task_result(task: asyncio.Task[Any]) -> None:
    try:
        exc = task.exception()
        if exc and not isinstance(exc, asyncio.CancelledError):
            logger.error("Background task '%s' failed: %s", task.get_name(), exc, exc_info=exc)
    except asyncio.CancelledError:
        pass


def safe_create_task(coro_or_func: Any, name: str, *args: Any, **kwargs: Any) -> asyncio.Task[Any] | None:
    try:
        res = coro_or_func(*args, **kwargs) if callable(coro_or_func) else coro_or_func
        if asyncio.iscoroutine(res):
            task = asyncio.create_task(res, name=name)
            task.add_done_callback(_handle_task_result)
            return task
        if isinstance(res, asyncio.Task):
            res.add_done_callback(_handle_task_result)
            return res
    except Exception as exc:
        logger.warning("Could not launch background task '%s': %s", name, exc)
    return None


def resolve_telegram_app() -> Any | None:
    """Resolve active Telegram Application instance across modern runner and legacy runtime."""
    with suppress(Exception):
        from app.bot import get_global_telegram_app

        bot_app = get_global_telegram_app()
        if bot_app is not None and getattr(bot_app, "bot", None) is not None:
            return bot_app

    with suppress(Exception):
        from app import legacy

        legacy_app = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
        if legacy_app is not None and getattr(legacy_app, "bot", None) is not None:
            return legacy_app

    return None


async def auto_setup_webhook(app_url: str) -> bool:
    """Automatically register the captured webhook URL with Telegram."""
    bot_mode = (
        os.environ.get("BOT_MODE") or getattr(SETTINGS, "BOT_MODE", "") or ""
    ).strip().upper()
    if bot_mode == "POLLING":
        logger.info("auto_setup_webhook aborted because BOT_MODE=POLLING.")
        return False

    clean_url = app_url.rstrip("/")
    if not clean_url.startswith("https://"):
        return False

    webhook_endpoint = clean_url if clean_url.endswith("/webhook") else f"{clean_url}/webhook"

    app_instance = None
    for _ in range(30):
        app_instance = resolve_telegram_app()
        if app_instance is not None:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        logger.warning("Could not auto-register webhook: Telegram application not ready.")
        return False

    try:
        secret = (
            os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN")
            or getattr(SETTINGS, "TELEGRAM_WEBHOOK_SECRET_TOKEN", "")
            or ""
        ).strip()
        allowed_updates = ["message", "edited_message", "callback_query", "inline_query", "channel_post", "my_chat_member"]
        await app_instance.bot.set_webhook(
            url=webhook_endpoint,
            secret_token=secret or None,
            allowed_updates=allowed_updates,
            drop_pending_updates=False,
        )
        logger.info("✅ Webhook auto-registered successfully: %s", webhook_endpoint)
        return True
    except Exception as exc:
        logger.error("Failed to auto-register webhook: %s", exc)
        return False


async def auto_register_bot_commands() -> bool:
    """Register default Telegram slash commands."""
    from telegram import BotCommand

    app_instance = None
    for _ in range(30):
        app_instance = resolve_telegram_app()
        if app_instance is not None:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        return False

    commands = [
        BotCommand("start", "Start the bot & main menu"),
        BotCommand("help", "Help & guide"),
        BotCommand("tts", "Text-to-speech voice generation"),
        BotCommand("ask", "AI chat assistant"),
        BotCommand("latest", "Latest news & audio articles"),
        BotCommand("podcast", "Daily morning podcast audio"),
        BotCommand("donate", "Support with ABA KHQR"),
    ]
    try:
        await app_instance.bot.set_my_commands(commands)
        logger.info("Bot commands registered successfully.")
        return True
    except Exception as exc:
        logger.warning("Could not register bot commands: %s", exc)
        return False


async def auto_register_all() -> None:
    """Run all automated registrations (webhook, bot commands) asynchronously."""
    url = get_detected_webhook_url()
    if url:
        await auto_setup_webhook(url)
    await auto_register_bot_commands()


async def keep_awake() -> None:
    """Periodically ping self to prevent cloud container sleep/spin-down."""
    try:
        await asyncio.sleep(60)
        while True:
            url = get_detected_webhook_url()
            if url:
                health_url = f"{url.rstrip('/')}/healthz"
                try:
                    req = urllib.request.Request(health_url, headers={"User-Agent": "BotVoice-KeepAwake/1.0"})
                    await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=10.0).close())
                except Exception as e:
                    logger.debug("Keep-awake ping suppressed: %s", e)
            await asyncio.sleep(600)
    except asyncio.CancelledError:
        pass


async def periodic_system_maintenance() -> None:
    """Periodic maintenance (cache, temp files, GC) every 2 hours."""
    try:
        await asyncio.sleep(120)
        while True:
            try:
                gc.collect()
                logger.info("Periodic maintenance garbage collection complete.")
            except Exception as exc:
                logger.warning("Periodic maintenance error: %s", exc)
            await asyncio.sleep(7200)
    except asyncio.CancelledError:
        pass


__all__ = [
    "auto_register_all",
    "auto_register_bot_commands",
    "auto_setup_webhook",
    "keep_awake",
    "periodic_system_maintenance",
    "resolve_telegram_app",
    "safe_create_task",
]
