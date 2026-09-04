"""Modern, fully modular Telegram Voice Bot runner.

Supports both high-speed Webhook mode and long-polling mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress

from telegram.ext import Application

from app.core.config import SETTINGS
from app.services.ai.gemini import GEMINI_MODEL_DEFAULT
from app.services.health import start_health_server
from app.services.settings.store import get_settings_store
from app.services.telegram.routing import register_telegram_handlers
from app.services.tts.voices import get_default_tts_model, tts_model_label

logger = logging.getLogger("app.bot")

_GLOBAL_TELEGRAM_APP: Application | None = None


def get_global_telegram_app() -> Application | None:
    return _GLOBAL_TELEGRAM_APP


def build_telegram_application(token: str, bot_mode: str = "POLLING") -> Application:
    """Build Telegram Application and register all modular handlers."""
    global _GLOBAL_TELEGRAM_APP
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
    register_telegram_handlers(app, bot_mode=bot_mode)
    _GLOBAL_TELEGRAM_APP = app
    return app


async def run_bot_async() -> None:
    """Initialize and run the Telegram bot in Webhook or Polling mode."""
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

    webhook_url = (os.environ.get("WEBHOOK_URL") or "").strip().rstrip("/")
    bot_mode = "WEBHOOK" if os.environ.get("BOT_MODE", "").strip().upper() == "WEBHOOK" and webhook_url else "POLLING"
    secret_token = (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or "").strip()

    print(
        f"🤖 Bot Voice is starting... [AI: {GEMINI_MODEL_DEFAULT} | "
        f"TTS: {model_name} | Mode: {bot_mode} | Storage: Supabase]"
    )

    asyncio.create_task(start_health_server(), name="health-server")

    app = build_telegram_application(token, bot_mode=bot_mode)

    async with app:
        await app.start()
        try:
            if bot_mode == "WEBHOOK" and webhook_url:
                webhook_endpoint = f"{webhook_url}/webhook"
                logger.info("Configuring Telegram Webhook at %s", webhook_endpoint)
                for attempt in range(5):
                    try:
                        await app.bot.set_webhook(
                            url=webhook_endpoint,
                            secret_token=secret_token or None,
                            allowed_updates=["message", "edited_message", "callback_query"],
                            drop_pending_updates=False,
                        )
                        logger.info("Telegram Webhook set successfully.")
                        break
                    except Exception as exc:
                        retry_after = getattr(exc, "retry_after", None)
                        delay = (float(retry_after) + 0.5) if retry_after is not None else (attempt + 1.0)
                        logger.warning("set_webhook flood control / retry (%ss): %s", delay, exc)
                        await asyncio.sleep(delay)
            else:
                for attempt in range(5):
                    try:
                        await app.bot.delete_webhook(drop_pending_updates=False)
                        break
                    except Exception as exc:
                        retry_after = getattr(exc, "retry_after", None)
                        delay = (float(retry_after) + 0.5) if retry_after is not None else (attempt + 1.0)
                        logger.warning("delete_webhook flood control / retry (%ss): %s", delay, exc)
                        await asyncio.sleep(delay)

                updater = getattr(app, "updater", None)
                if updater is not None:
                    await updater.start_polling(
                        allowed_updates=["message", "edited_message", "callback_query"],
                        drop_pending_updates=False,
                        bootstrap_retries=-1,
                        timeout=20,
                    )
                    logger.info("Telegram polling started successfully.")

            # Keep running until cancelled
            stop_event = asyncio.Event()
            await stop_event.wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Bot shutdown requested.")
        finally:
            updater = getattr(app, "updater", None)
            if updater is not None and getattr(updater, "running", False):
                with suppress(Exception):
                    await updater.stop()
            if getattr(app, "running", False):
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
