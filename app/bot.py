"""Modern, fully modular Telegram Voice Bot runner.

Replaces legacy runtime with pure service-oriented architecture.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from typing import Any

from telegram import Update
from telegram.ext import Application

from app.core.config import SETTINGS
from app.services.ai.gemini import GEMINI_MODEL_DEFAULT
from app.services.settings.store import get_settings_store
from app.services.telegram.routing import register_telegram_handlers
from app.services.tts.voices import get_default_tts_model, tts_model_label

logger = logging.getLogger("app.bot")


def build_telegram_application(token: str) -> Application:
    """Build Telegram Application and register all modular handlers."""
    clean_token = str(token or "").strip()
    if not clean_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is empty or not configured.")

    builder = (
        Application.builder()
        .token(clean_token)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
    )
    if hasattr(builder, "concurrent_updates"):
        builder = builder.concurrent_updates(4)
    if hasattr(builder, "connection_pool_size"):
        builder = builder.connection_pool_size(24)

    app = builder.build()
    register_telegram_handlers(app, bot_mode="POLLING")
    return app


async def run_bot_async() -> None:
    """Initialize and run the Telegram long-polling bot session."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or getattr(SETTINGS, "TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is missing. Please set it in .env or your hosting panel.")
        return

    # 1. Warm up settings store & retrieve active runtime defaults
    with suppress(Exception):
        store = get_settings_store()
        await store.get_text("DEFAULT_TTS_MODEL", "")

    default_model = get_default_tts_model()
    model_name = tts_model_label(default_model)

    print(
        f"🤖 Bot Voice is starting... [AI: {GEMINI_MODEL_DEFAULT} | "
        f"TTS: {model_name} | Mode: POLLING | Storage: Supabase]"
    )

    from app.services.health import start_health_server
    health_task = asyncio.create_task(start_health_server(), name="health-server")

    app = build_telegram_application(token)

    async with app:

        await app.start()
        updater = getattr(app, "updater", None)
        if updater is not None:
            await updater.start_polling(
                allowed_updates=["message", "edited_message", "callback_query"],
                drop_pending_updates=False,
            )
            logger.info("Telegram polling started successfully.")

        # Keep running until cancelled
        try:
            while True:
                await asyncio.sleep(3600)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Bot shutdown requested.")
        finally:
            if updater is not None:
                with suppress(Exception):
                    await updater.stop()
            with suppress(Exception):
                await app.stop()


def main() -> None:
    """Main process entrypoint with auto-restart protection."""
    while True:
        try:
            asyncio.run(run_bot_async())
        except KeyboardInterrupt:
            logger.info("Shutdown completed.")
            break
        except Exception as exc:
            logger.error("Runtime encountered an error: %s — restarting in 5s...", exc, exc_info=True)
            time.sleep(5)
        else:
            time.sleep(5)


if __name__ == "__main__":
    main()
