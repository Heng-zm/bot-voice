"""Modern, fully modular Telegram Voice Bot runner.

Supports both high-speed Webhook mode and long-polling mode.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from typing import Any

from telegram.ext import Application

from app.core.config import SETTINGS
from app.services.ai.gemini import GEMINI_MODEL_DEFAULT
from app.services.health import start_health_server
from app.services.settings.store import get_settings_store
from app.services.telegram.routing import register_telegram_handlers
from app.services.tts.voices import get_default_tts_model, tts_model_label
from app.utils.logging import install_telegram_polling_filter

install_telegram_polling_filter()

logger = logging.getLogger("app.bot")

_GLOBAL_TELEGRAM_APP: Application | None = None
_SHUTDOWN_REQUESTED: bool = False

DEFAULT_ALLOWED_UPDATES = [
    "message",
    "edited_message",
    "callback_query",
    "inline_query",
    "channel_post",
    "my_chat_member",
]


def _safe_int(value: Any, default: int) -> int:
    """Safely convert a value to an integer with a fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _handle_task_result(task: asyncio.Task) -> None:
    """Callback to log unexpected exceptions from background tasks."""
    try:
        exc = task.exception()
        if exc and not isinstance(exc, asyncio.CancelledError):
            logger.error("Background task '%s' failed: %s", task.get_name(), exc, exc_info=exc)
    except asyncio.CancelledError:
        pass


def get_global_telegram_app() -> Application | None:
    """Retrieve the active Telegram Application singleton."""
    if _GLOBAL_TELEGRAM_APP is not None:
        return _GLOBAL_TELEGRAM_APP
    try:
        from app import legacy
        return getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
    except Exception:
        return None


def build_telegram_application(token: str, bot_mode: str = "POLLING") -> Application:
    """Build Telegram Application and register all modular handlers."""
    global _GLOBAL_TELEGRAM_APP
    clean_token = str(token or "").strip()
    if not clean_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is empty or not configured.")

    builder = (
        Application.builder()
        .token(clean_token)
        .connect_timeout(30.0)
        .read_timeout(60.0)
        .write_timeout(120.0)
        .pool_timeout(60.0)
    )

    if hasattr(builder, "get_updates_connect_timeout"):
        builder = builder.get_updates_connect_timeout(15.0)
    if hasattr(builder, "get_updates_read_timeout"):
        builder = builder.get_updates_read_timeout(35.0)
    if hasattr(builder, "get_updates_write_timeout"):
        builder = builder.get_updates_write_timeout(15.0)
    if hasattr(builder, "get_updates_pool_timeout"):
        builder = builder.get_updates_pool_timeout(15.0)

    raw_concurrent = os.environ.get("TELEGRAM_CONCURRENT_UPDATES") or getattr(
        SETTINGS, "TELEGRAM_CONCURRENT_UPDATES", 32
    )
    concurrent_updates = max(4, min(128, _safe_int(raw_concurrent, 32)))

    raw_pool = os.environ.get("TELEGRAM_CONNECTION_POOL_SIZE") or getattr(
        SETTINGS, "TELEGRAM_CONNECTION_POOL_SIZE", 64
    )
    pool_size = max(16, min(256, _safe_int(raw_pool, 64)))

    # Ensure connection pool is sufficiently sized for concurrent processing
    pool_size = max(pool_size, concurrent_updates + 8)
    get_updates_pool_size = min(16, max(4, pool_size // 4))

    if hasattr(builder, "get_updates_connection_pool_size"):
        builder = builder.get_updates_connection_pool_size(get_updates_pool_size)
    if hasattr(builder, "concurrent_updates"):
        builder = builder.concurrent_updates(concurrent_updates)
    if hasattr(builder, "connection_pool_size"):
        builder = builder.connection_pool_size(pool_size)

    app = builder.build()
    register_telegram_handlers(app, bot_mode=bot_mode)
    _GLOBAL_TELEGRAM_APP = app

    with suppress(Exception):
        from app import legacy
        legacy.telegram_application = app
        legacy._TELEGRAM_APP = app

    return app


async def run_bot_async() -> bool:
    """Initialize and run the Telegram bot in Webhook or Polling mode.

    Returns:
        bool: True if execution completed as part of a clean shutdown,
              False if terminated due to fatal configuration errors.
    """
    global _GLOBAL_TELEGRAM_APP, _SHUTDOWN_REQUESTED

    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or getattr(SETTINGS, "TELEGRAM_BOT_TOKEN", None)
        or ""
    ).strip()

    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN is missing. Please set it in .env or your hosting panel.")
        return False

    # 1. Warm up settings store & retrieve active runtime defaults
    with suppress(Exception):
        store = get_settings_store()
        if hasattr(store, "get_text"):
            res = store.get_text("DEFAULT_TTS_MODEL", "")
            if asyncio.iscoroutine(res):
                await res

    try:
        default_model = get_default_tts_model()
        model_name = tts_model_label(default_model)
    except Exception as exc:
        logger.warning("Could not determine default TTS model (%s), using fallback.", exc)
        model_name = "Default"

    raw_url = os.environ.get("WEBHOOK_URL") or getattr(SETTINGS, "WEBHOOK_URL", "")
    webhook_url = str(raw_url or "").strip().rstrip("/")

    configured_mode = (
        os.environ.get("BOT_MODE") or getattr(SETTINGS, "BOT_MODE", "") or ""
    ).strip().upper()

    if configured_mode == "WEBHOOK" and not webhook_url:
        logger.warning("BOT_MODE=WEBHOOK is specified, but WEBHOOK_URL is missing. Falling back to POLLING mode.")

    bot_mode = "WEBHOOK" if configured_mode == "WEBHOOK" and webhook_url else "POLLING"
    secret_token = (
        os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN")
        or getattr(SETTINGS, "TELEGRAM_WEBHOOK_SECRET_TOKEN", None)
        or ""
    ).strip() or None

    drop_pending_updates = str(
        os.environ.get("DROP_PENDING_UPDATES")
        or getattr(SETTINGS, "DROP_PENDING_UPDATES", "false")
    ).strip().lower() in ("true", "1", "yes")

    print(
        f"🤖 Bot Voice is starting... [AI: {GEMINI_MODEL_DEFAULT} | "
        f"TTS: {model_name} | Mode: {bot_mode} | Storage: Supabase]"
    )

    # 2. Build application and assign singleton before launching dependent servers
    app = build_telegram_application(token, bot_mode=bot_mode)

    # 3. Setup signal handling for graceful container termination (SIGTERM/SIGINT)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _trigger_shutdown() -> None:
        global _SHUTDOWN_REQUESTED
        _SHUTDOWN_REQUESTED = True
        logger.info("Shutdown signal received. Initiating graceful teardown...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _trigger_shutdown)
        except (NotImplementedError, RuntimeError):
            # Not supported on Windows Proactor EventLoop or non-main threads
            pass

    # 4. Start background helpers
    health_task: asyncio.Task | None = None
    with suppress(Exception):
        try:
            h_coro = start_health_server(app)
        except TypeError:
            h_coro = start_health_server()
        if asyncio.iscoroutine(h_coro):
            health_task = asyncio.create_task(h_coro, name="health-server")
            health_task.add_done_callback(_handle_task_result)

    batcher_task: asyncio.Task | None = None
    with suppress(Exception):
        from app.services.tasks.batcher import get_database_batcher
        batcher = get_database_batcher()
        if hasattr(batcher, "start_worker"):
            b_res = batcher.start_worker()
            if asyncio.iscoroutine(b_res):
                batcher_task = asyncio.create_task(b_res, name="db-batcher-worker")
            elif isinstance(b_res, asyncio.Task):
                batcher_task = b_res
            if isinstance(batcher_task, asyncio.Task):
                batcher_task.add_done_callback(_handle_task_result)

    bakong_task: asyncio.Task | None = None
    cmd_task: asyncio.Task | None = None

    async with app:
        await app.start()
        try:
            if bot_mode == "WEBHOOK" and webhook_url:
                webhook_endpoint = webhook_url if webhook_url.endswith("/webhook") else f"{webhook_url}/webhook"
                logger.info("Configuring Telegram Webhook at %s", webhook_endpoint)
                webhook_configured = False

                for attempt in range(1, 6):
                    try:
                        await app.bot.set_webhook(
                            url=webhook_endpoint,
                            secret_token=secret_token,
                            allowed_updates=DEFAULT_ALLOWED_UPDATES,
                            drop_pending_updates=drop_pending_updates,
                        )
                        logger.info("Telegram Webhook set successfully.")
                        webhook_configured = True
                        break
                    except Exception as exc:
                        retry_after = getattr(exc, "retry_after", None)
                        delay = (float(retry_after) + 0.5) if retry_after is not None else float(attempt)
                        logger.warning("set_webhook retry %d/5 in %.1fs: %s", attempt, delay, exc)
                        if attempt == 5:
                            raise RuntimeError(f"Failed to configure Webhook after 5 attempts: {exc}") from exc
                        await asyncio.sleep(delay)

            else:
                for attempt in range(1, 6):
                    try:
                        await app.bot.delete_webhook(drop_pending_updates=drop_pending_updates)
                        logger.info("Telegram webhook removed. Polling mode ready.")
                        break
                    except Exception as exc:
                        retry_after = getattr(exc, "retry_after", None)
                        delay = (float(retry_after) + 0.5) if retry_after is not None else float(attempt)
                        logger.warning("delete_webhook retry %d/5 in %.1fs: %s", attempt, delay, exc)
                        if attempt == 5:
                            logger.error("Could not clear webhook before starting polling: %s", exc)
                            raise
                        await asyncio.sleep(delay)

                updater = getattr(app, "updater", None)
                if updater is not None:
                    await updater.start_polling(
                        allowed_updates=DEFAULT_ALLOWED_UPDATES,
                        drop_pending_updates=drop_pending_updates,
                        bootstrap_retries=-1,
                        timeout=20,
                    )
                    logger.info("Telegram polling started successfully.")

            # Register bot commands menu asynchronously
            with suppress(Exception):
                from app.main import auto_register_bot_commands
                try:
                    c_coro = auto_register_bot_commands(app)
                except TypeError:
                    c_coro = auto_register_bot_commands()
                if asyncio.iscoroutine(c_coro):
                    cmd_task = asyncio.create_task(c_coro, name="bot-auto-register-commands")
                    cmd_task.add_done_callback(_handle_task_result)

            # Start Bakong continuous donations checker
            with suppress(Exception):
                from app.services.donation.handlers import periodic_bakong_auto_checker
                try:
                    bk_coro = periodic_bakong_auto_checker(app)
                except TypeError:
                    bk_coro = periodic_bakong_auto_checker()
                if asyncio.iscoroutine(bk_coro):
                    bakong_task = asyncio.create_task(bk_coro, name="bakong-auto-checker")
                    bakong_task.add_done_callback(_handle_task_result)

            # Block until shutdown signal is received
            await stop_event.wait()

        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Bot execution cancelled.")
        finally:
            logger.info("Executing graceful teardown sequence...")

            # 1. Stop receiving new updates first to prevent new workload during drain
            updater = getattr(app, "updater", None)
            if updater is not None and getattr(updater, "running", False):
                with suppress(Exception):
                    await updater.stop()

            # 2. Stop application update processor
            if getattr(app, "running", False):
                with suppress(Exception):
                    await app.stop()

            # 3. Stop background continuous tasks
            for task in (bakong_task, cmd_task):
                if task and not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task

            # 4. Drain queued tasks and batched DB writes cleanly
            with suppress(Exception):
                from app.services.tasks.queue import get_task_manager
                await get_task_manager().drain(timeout=5.0)

            with suppress(Exception):
                from app.services.tasks.batcher import get_database_batcher
                await get_database_batcher().drain(timeout=5.0)

            # 5. Cancel batcher and health workers
            for task in (batcher_task, health_task):
                if task and not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task

            # 6. Clean up global references and signal handlers
            _GLOBAL_TELEGRAM_APP = None
            with suppress(Exception):
                from app import legacy
                legacy.telegram_application = None
                legacy._TELEGRAM_APP = None

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass

    return True


def main() -> None:
    """Main process entrypoint with fault-tolerant crash auto-recovery."""
    global _SHUTDOWN_REQUESTED

    while not _SHUTDOWN_REQUESTED:
        try:
            success = asyncio.run(run_bot_async())
            if not success or _SHUTDOWN_REQUESTED:
                # Fatal configuration or intentional stop -> do not loop
                break
        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Exiting.")
            break
        except Exception as exc:
            logger.error(
                "Runtime encountered an unexpected error: %s — restarting in 5s...",
                exc,
                exc_info=True,
            )
            time.sleep(5)

    logger.info("Bot process exited cleanly.")


if __name__ == "__main__":
    main()