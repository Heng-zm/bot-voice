"""Telegram Voice Bot entry point with Automated Registration, AI Assistant, and TTS APIs."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from dotenv import load_dotenv

    _root_env = Path(__file__).resolve().parent.parent / ".env"
    if _root_env.is_file():
        load_dotenv(dotenv_path=_root_env, override=False)
    else:
        load_dotenv(override=False)
except Exception:
    pass

from fastapi import FastAPI
from telegram import BotCommand

from app import legacy
from app.api import api_router
from app.core.config import get_detected_webhook_url
from app.core.security import (
    get_allowed_api_keys as _get_allowed_api_keys,
    validate_api_key as _validate_api_key,
)

logger = logging.getLogger("app.webhook")



async def auto_setup_webhook(app_url: str) -> bool:
    """Automatically register the captured webhook URL with Telegram."""
    global _DETECTED_WEBHOOK_URL
    bot_mode = os.environ.get("BOT_MODE", "").strip().upper()
    if bot_mode == "POLLING":
        logger.info("auto_setup_webhook aborted because BOT_MODE=POLLING.")
        return False

    clean_url = app_url.rstrip("/")
    if not clean_url.startswith("https://"):
        return False

    _DETECTED_WEBHOOK_URL = clean_url
    webhook_endpoint = f"{clean_url}/webhook"

    app_instance = None
    for _ in range(30):
        app_instance = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
        is_ready = getattr(legacy, "_TELEGRAM_APP_READY", False)
        secret_ready = bool(getattr(legacy, "_runtime_webhook_secret_token", lambda: "")())
        if app_instance is not None and getattr(app_instance, "bot", None) is not None and is_ready and secret_ready:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        logger.warning("Could not auto-register webhook: Telegram application not ready.")
        return False


    secret_func = getattr(legacy, "_runtime_webhook_secret_token", None)
    secret = secret_func() if secret_func else (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or "").strip()
    allowed_updates_func = getattr(legacy, "_telegram_allowed_updates", None)
    allowed_updates = (
        allowed_updates_func()
        if allowed_updates_func
        else ["message", "edited_message", "callback_query", "channel_post"]
    )

    for attempt in range(5):
        try:
            await app_instance.bot.set_webhook(
                url=webhook_endpoint,
                secret_token=secret or None,
                allowed_updates=allowed_updates,
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


async def auto_register_bot_commands() -> bool:
    """Automatically register Telegram Menu commands with Telegram API."""
    app_instance = None
    for _ in range(30):
        app_instance = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
        if app_instance is not None and getattr(app_instance, "bot", None) is not None:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        return False

    user_commands = [
        BotCommand("start", "🚀 ចាប់ផ្ដើម / Start Bot"),
        BotCommand("help", "📖 របៀបប្រើប្រាស់ / Help"),
        BotCommand("ask", "🤖 សួរ AI / Ask AI"),
        BotCommand("translate", "🌐 បកប្រែជាភាសាខ្មែរ / Translate"),
        BotCommand("summary", "📝 សង្ខេបអត្ថបទវែង / Summarize"),
        BotCommand("narrate", "📰 អានអត្ថបទពីតំណភ្ជាប់ / Narrate URL"),
        BotCommand("myprefs", "⚙️ កំណត់សំឡេង & ល្បឿន / Settings"),
        BotCommand("ttsmodel", "🎙️ ជ្រើសរើសម៉ូដែល TTS / TTS Engine"),
        BotCommand("clear", "🗑️ សម្អាតប្រវត្តិ / Clear Chat"),
        BotCommand("unlock", "🔓 ដោះសោររង់ចាំ / Force Unlock"),
        BotCommand("security", "🔐 សុវត្ថិភាព / Security Status"),
        BotCommand("privacy", "🔒 ឯកជនភាព / Privacy Policy"),
        BotCommand("deleteme", "⚠️ លុបទិន្នន័យ / Delete My Data"),
        BotCommand("feedback", "💡 ផ្ញើមតិកែលម្អ / Feedback"),
        BotCommand("system", "📊 ព័ត៌មានប្រព័ន្ធ / System Status"),
        BotCommand("cancel", "🛑 បោះបង់ / Cancel Action"),
        BotCommand("admin", "👑 ផ្ទាំងគ្រប់គ្រង / Admin Panel"),
    ]

    admin_commands = [
        *user_commands,
        BotCommand("health", "🏥 ស្ថានភាពម៉ាស៊ីន / Health Check"),
        BotCommand("stats", "📈 ស្ថិតិបូតទូទៅ / Bot Stats"),
        BotCommand("broadcast", "📢 ផ្ញើសារជូនដំណឹង / Broadcast"),
        BotCommand("schedule", "📅 បង្កើតកាលវិភាគ / Schedule Broadcast"),
        BotCommand("schedules", "📋 បញ្ជីកាលវិភាគ / List Schedules"),
        BotCommand("cancelschedule", "❌ បោះបង់កាលវិភាគ / Cancel Schedule"),
        BotCommand("users", "👥 គ្រប់គ្រងអ្នកប្រើ / Manage Users"),
        BotCommand("botsettings", "🛠️ កំណត់រចនាសម្ព័ន្ធ / Bot Config"),
        BotCommand("api", "🔑 គ្រប់គ្រង API / API Keys"),
        BotCommand("runtime", "⚡ ព័ត៌មាន Runtime / Telemetry"),
    ]

    try:
        from telegram import BotCommandScopeChat, BotCommandScopeDefault

        await app_instance.bot.set_my_commands(user_commands, scope=BotCommandScopeDefault())
        logger.info("Auto-registered %s Telegram default commands in menu.", len(user_commands))

        # Register extended admin commands for configured admins
        admin_ids: set[int] = set()
        legacy_admins = getattr(legacy, "ADMIN_IDS", None)
        if isinstance(legacy_admins, (set, list, tuple)):
            admin_ids.update(int(aid) for aid in legacy_admins if str(aid).isdigit())
        for env_aid in os.environ.get("ADMIN_IDS", "").split(","):
            env_aid = env_aid.strip()
            if env_aid.isdigit():
                admin_ids.add(int(env_aid))

        for aid in admin_ids:
            with suppress(Exception):
                await app_instance.bot.set_my_commands(admin_commands, scope=BotCommandScopeChat(aid))

        return True
    except Exception as exc:
        logger.warning("Failed to auto-register bot commands: %s", exc)
        return False


async def auto_register_all() -> dict[str, Any]:
    """Execute complete automated registration sequence on startup."""
    results: dict[str, Any] = {}

    # 1. Register Webhook or Fallback to Polling
    bot_mode = os.environ.get("BOT_MODE", "").strip().upper()
    if bot_mode == "POLLING":
        results["webhook"] = False
        results["mode"] = "POLLING"
        logger.info("BOT_MODE=POLLING explicitly configured; webhook registration skipped.")
    else:
        url = get_detected_webhook_url()
        if url:
            webhook_ok = await auto_setup_webhook(url)
            results["webhook"] = webhook_ok
            results["webhook_url"] = f"{url.rstrip('/')}/webhook"
            if not webhook_ok:
                logger.warning("Webhook registration failed (unresolvable host); falling back to POLLING mode.")
                try:
                    if hasattr(legacy, "_switch_telegram_runtime_mode"):
                        await legacy._switch_telegram_runtime_mode("POLLING")
                        results["fallback_mode"] = "POLLING"
                except Exception as e:
                    logger.error("Failed to switch to POLLING mode: %s", e)
        else:
            results["webhook"] = False
            results["mode"] = "POLLING"
            try:
                if hasattr(legacy, "_run_state_bot_mode") and legacy._run_state_bot_mode() == "WEBHOOK":
                    await legacy._switch_telegram_runtime_mode("POLLING")
            except Exception as e:
                logger.debug("Ensure polling mode error: %s", e)

    # 2. Register Commands Menu
    results["commands"] = await auto_register_bot_commands()

    # 3. Warm up Settings & TTS
    with suppress(Exception):
        from app.services.settings.store import get_settings_store
        from app.services.tts.voices import get_default_tts_model

        await get_settings_store().get_text("DEFAULT_TTS_MODEL", "")
        results["default_tts_model"] = get_default_tts_model()

    return results


async def keep_awake() -> None:
    """Ping the public URL every 5 minutes to prevent Render Free Tier hibernation."""
    import httpx

    try:
        # Wait for server to fully start
        await asyncio.sleep(60)

        async with httpx.AsyncClient() as client:
            while True:
                bot_mode = os.environ.get("BOT_MODE", "").strip().upper()
                url = get_detected_webhook_url()
                if url and bot_mode != "POLLING":
                    health_url = f"{url.rstrip('/')}/healthz"
                    try:
                        await client.get(health_url, timeout=10.0)
                        logger.debug("Keep-awake ping sent to %s", health_url)
                    except Exception as e:
                        logger.debug("Keep-awake ping suppressed: %s", e)

                # Sleep for 10 minutes (Render sleeps after 15m)
                await asyncio.sleep(600)
    except asyncio.CancelledError:
        logger.debug("keep_awake background task stopped cleanly.")


async def periodic_database_pruner() -> None:
    """Run database cleanup every 24 hours."""
    try:
        await asyncio.sleep(120)  # wait 2 minutes after startup
        while True:
            try:
                if hasattr(legacy, "db_run_periodic_pruning"):
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, legacy.db_run_periodic_pruning)
            except Exception as exc:
                logger.warning("Periodic DB pruning error: %s", exc)
            await asyncio.sleep(86400)
    except asyncio.CancelledError:
        logger.debug("periodic_database_pruner background task stopped cleanly.")


async def bot_runner_supervisor() -> None:
    """Supervise the Telegram bot runner and restart it automatically if it ever exits."""
    backoff = 5
    while True:
        try:
            logger.info("Starting Telegram bot runner under supervisor...")
            await legacy._async_main_once()
            logger.warning("Telegram bot runner exited. Restarting in %ss...", backoff)
        except asyncio.CancelledError:
            logger.info("Telegram bot runner supervisor received cancellation.")
            raise
        except Exception as exc:
            logger.error("Telegram bot runner crashed: %s. Restarting in %ss...", exc, backoff, exc_info=True)
        await asyncio.sleep(backoff)


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Start the Telegram Bot runner and auto-register all services."""
    bot_task = asyncio.create_task(bot_runner_supervisor(), name="telegram-bot-runner")
    auto_reg_task = asyncio.create_task(auto_register_all(), name="auto-register-all")
    keep_awake_task = asyncio.create_task(keep_awake(), name="keep-awake")
    db_prune_task = asyncio.create_task(periodic_database_pruner(), name="db-pruner")

    tasks = (db_prune_task, keep_awake_task, auto_reg_task, bot_task)
    try:
        yield
    finally:
        with suppress(Exception):
            from app.services.telegram.dispatcher import get_telegram_dispatcher

            await get_telegram_dispatcher().drain(timeout=10.0)

        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t


app = FastAPI(
    title="Telegram Bot Voice & AI Assistant Suite",
    description="Multilingual Voice Synthesis, AI Vision OCR, and Gemini Assistant API.",
    version="4.2.0",
    lifespan=lifespan,
)

# Include modular API routers
app.include_router(api_router)


def main() -> None:
    """Run the FastAPI application with Uvicorn on $PORT."""
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")  # noqa: S104



__all__ = [
    "app",
    "main",
    "auto_register_all",
    "auto_register_bot_commands",
    "auto_setup_webhook",
    "bot_runner_supervisor",
    "get_detected_webhook_url",
    "lifespan",
    "_validate_api_key",
    "_get_allowed_api_keys",
]


if __name__ == "__main__":
    main()
