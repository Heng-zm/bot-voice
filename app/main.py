"""Telegram Voice Bot entry point with Automated Registration, AI Assistant, and TTS APIs."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from telegram import BotCommand

from app import legacy
from app.api import api_router
from app.core.config import SETTINGS, get_detected_webhook_url
from app.core.logging_middleware import request_lifecycle_logging_middleware
from app.core.security import (
    get_allowed_api_keys as _get_allowed_api_keys,
)
from app.core.security import (
    validate_api_key as _validate_api_key,
)
from app.utils.logging import configure_server_logging, install_telegram_polling_filter

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

with suppress(Exception):
    from dotenv import load_dotenv

    _root_env = Path(__file__).resolve().parent.parent / ".env"
    if _root_env.is_file():
        load_dotenv(dotenv_path=_root_env, override=False)
    else:
        load_dotenv(override=False)

_default_response_class = None
with suppress(Exception):
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse

    _default_response_class = ORJSONResponse

with suppress(Exception):
    gc.set_threshold(50000, 10, 10)

configure_server_logging()
install_telegram_polling_filter()

logger = logging.getLogger("app.webhook")


def _safe_int(value: Any, default: int) -> int:
    """Safely convert a value to an integer."""
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


def _safe_create_task(coro_or_func: Any, name: str, *args: Any, **kwargs: Any) -> asyncio.Task | None:
    """Safely execute and wrap a coroutine or callable into a supervised asyncio.Task."""
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


def _resolve_telegram_app() -> Any | None:
    """Resolve active Telegram Application instance from modern runner or legacy."""
    with suppress(Exception):
        from app.bot import get_global_telegram_app

        bot_app = get_global_telegram_app()
        if bot_app is not None and getattr(bot_app, "bot", None) is not None:
            return bot_app

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
        app_instance = _resolve_telegram_app()
        is_ready = getattr(legacy, "_TELEGRAM_APP_READY", True)
        if app_instance is not None and is_ready:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        logger.warning("Could not auto-register webhook: Telegram application not ready.")
        return False

    secret_func = getattr(legacy, "_runtime_webhook_secret_token", None)
    secret = (
        secret_func()
        if secret_func
        else (
            os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN")
            or getattr(SETTINGS, "TELEGRAM_WEBHOOK_SECRET_TOKEN", "")
            or ""
        ).strip()
    )

    allowed_updates_func = getattr(legacy, "_telegram_allowed_updates", None)
    allowed_updates = (
        allowed_updates_func()
        if allowed_updates_func
        else ["message", "edited_message", "callback_query", "inline_query", "channel_post", "my_chat_member"]
    )

    drop_pending = str(
        os.environ.get("DROP_PENDING_UPDATES")
        or getattr(SETTINGS, "DROP_PENDING_UPDATES", "false")
    ).strip().lower() in ("true", "1", "yes")

    for attempt in range(1, 6):
        try:
            await app_instance.bot.set_webhook(
                url=webhook_endpoint,
                secret_token=secret or None,
                allowed_updates=allowed_updates,
                drop_pending_updates=drop_pending,
            )
            logger.info("Auto-captured and registered Telegram Webhook at %s", webhook_endpoint)
            return True
        except Exception as exc:
            retry_after = getattr(exc, "retry_after", None)
            delay = (float(retry_after) + 0.5) if retry_after is not None else float(attempt)
            logger.warning("auto_setup_webhook retry %d/5 (%ss): %s", attempt, delay, exc)
            await asyncio.sleep(delay)

    logger.error("Failed to set webhook at %s after 5 attempts.", webhook_endpoint)
    return False


async def auto_register_bot_commands(app_instance: Any = None) -> bool:
    """Automatically register Telegram Menu commands with Telegram API."""
    if app_instance is None:
        for _ in range(30):
            app_instance = _resolve_telegram_app()
            if app_instance is not None and getattr(app_instance, "bot", None) is not None:
                break
            await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        logger.warning("Could not auto-register bot commands: Telegram application not available.")
        return False

    user_commands = [
        BotCommand("start", "🚀 ចាប់ផ្តើម / Start Bot"),
        BotCommand("help", "📖 ជំនួយ / Help"),
        BotCommand("menu", "📋 ម៉ឺនុយរហ័ស / Quick Menu"),
        BotCommand("ask", "💬 សួរ AI / Ask AI"),
        BotCommand("translate", "🌐 បកប្រែភាសា / Translate"),
        BotCommand("summary", "📝 សង្ខេបអត្ថបទ / Summarize"),
        BotCommand("narrate", "🎙️ អានអត្ថបទគេហទំព័រ / Narrate URL"),
        BotCommand("tiktok", "📥 ទាញយក TikTok / TikTok Downloader"),
        BotCommand("myprefs", "⚙️ ការកំណត់ & បំណង / Settings"),
        BotCommand("ttsmodel", "🗣️ ម៉ាស៊ីនសម្លេង TTS / TTS Engine"),
        BotCommand("clear", "🗑️ សម្អាតប្រវត្តិសារ / Clear Chat"),
        BotCommand("unlock", "🔓 ដោះសោរ / Force Unlock"),
        BotCommand("security", "🛡️ ស្ថានភាពសុវត្ថិភាព / Security Status"),
        BotCommand("privacy", "🔒 គោលការណ៍ឯកជនភាព / Privacy Policy"),
        BotCommand("deleteme", "⚠️ លុបទិន្នន័យខ្ញុំ / Delete My Data"),
        BotCommand("feedback", "💌 មតិកែលម្អ / Feedback"),
        BotCommand("system", "📊 ស្ថានភាពប្រព័ន្ធ / System Status"),
        BotCommand("donate", "☕ ជូនកាហ្វេ / Buy Coffee"),
        BotCommand("donors", "🏆 តារាងអ្នកឧបត្ថម្ភ / Hall of Fame"),
        BotCommand("cancel", "🚫 បោះបង់សកម្មភាព / Cancel Action"),
        BotCommand("admin", "👑 ផ្ទាំងគ្រប់គ្រង / Admin Panel"),
    ]

    admin_commands = [
        *user_commands,
        BotCommand("health", "🩺 ពិនិត្យសុខភាពប្រព័ន្ធ / Health Check"),
        BotCommand("stats", "📈 ស្ថិតិប្រព័ន្ធ / Bot Stats"),
        BotCommand("broadcast", "📢 ផ្សាយដំណឹង / Broadcast"),
        BotCommand("schedule", "📅 កាលវិភាគផ្សាយ / Schedule Broadcast"),
        BotCommand("schedules", "📋 បញ្ជីកាលវិភាគ / List Schedules"),
        BotCommand("cancelschedule", "❌ លុបកាលវិភាគ / Cancel Schedule"),
        BotCommand("users", "👥 គ្រប់គ្រងអ្នកប្រើ / Manage Users"),
        BotCommand("botsettings", "⚙️ កំណត់រចនាសម្ព័ន្ធ / Bot Config"),
        BotCommand("api", "🔑 គ្រប់គ្រង API Keys / API Keys"),
        BotCommand("runtime", "⏱️ ស្ថានភាព Runtime / Telemetry"),
        BotCommand("dbstatus", "🗄️ ស្ថានភាព DB / Database Status"),
        BotCommand("dbbackup", "💾 បម្រុងទុក DB / Database Backup"),
        BotCommand("migrate", "🚀 ធ្វើបច្ចុប្បន្នភាព DB / Database Migration"),
        BotCommand("podcast", "🎙️ ផតខាស់ប្រចាំថ្ងៃ / Daily Podcast"),
        BotCommand("bakongstatus", "🇰🇭 ស្ថានភាព Bakong / Bakong Gateway"),
        BotCommand("adddonor", "➕ បន្ថែមអ្នកឧបត្ថម្ភ / Add Donor"),
        BotCommand("testblessing", "🎉 សាកល្បងពរជ័យ / Test Blessing"),
    ]

    try:
        from telegram import BotCommandScopeChat, BotCommandScopeDefault

        await app_instance.bot.set_my_commands(user_commands, scope=BotCommandScopeDefault())
        logger.info("Auto-registered %s Telegram default commands in menu.", len(user_commands))

        # Register extended admin commands for configured admins
        admin_ids: set[int] = set()
        for source in (getattr(legacy, "ADMIN_IDS", None), getattr(SETTINGS, "ADMIN_IDS", None)):
            if isinstance(source, (set, list, tuple)):
                admin_ids.update(int(aid) for aid in source if str(aid).lstrip("-").isdigit())
            elif isinstance(source, str):
                for aid in source.split(","):
                    if aid.strip().lstrip("-").isdigit():
                        admin_ids.add(int(aid.strip()))

        for env_aid in os.environ.get("ADMIN_IDS", "").split(","):
            env_aid = env_aid.strip()
            if env_aid.lstrip("-").isdigit():
                admin_ids.add(int(env_aid))

        for aid in admin_ids:
            with suppress(Exception):
                await app_instance.bot.set_my_commands(admin_commands, scope=BotCommandScopeChat(aid))

        return True
    except Exception as exc:
        logger.warning("Failed to auto-register bot commands: %s", exc)
        return False


async def auto_register_all(app_instance: Any = None) -> dict[str, Any]:
    """Execute complete automated registration sequence on startup."""
    results: dict[str, Any] = {}

    bot_mode = (
        os.environ.get("BOT_MODE") or getattr(SETTINGS, "BOT_MODE", "") or ""
    ).strip().upper()

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
                logger.warning("Webhook registration failed; attempting fallback to POLLING mode.")
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

    results["commands"] = await auto_register_bot_commands(app_instance)

    with suppress(Exception):
        from app.services.settings.store import get_settings_store
        from app.services.tts.voices import get_default_tts_model

        store = get_settings_store()
        if hasattr(store, "get_text"):
            res = store.get_text("DEFAULT_TTS_MODEL", "")
            if asyncio.iscoroutine(res):
                await res
        results["default_tts_model"] = get_default_tts_model()

    with suppress(Exception):
        if hasattr(legacy, "db_unblock_all_users"):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, legacy.db_unblock_all_users)

    return results


async def keep_awake() -> None:
    """Ping the public health endpoint every 10 minutes to prevent Free Tier hibernation."""
    try:
        await asyncio.sleep(60)

        _httpx = None
        with suppress(ImportError, ModuleNotFoundError):
            import httpx
            _httpx = httpx

        while True:
            url = get_detected_webhook_url()
            keep_awake_enabled = str(os.environ.get("KEEP_AWAKE", "true")).lower() in ("true", "1", "yes")

            if url and keep_awake_enabled:
                health_url = f"{url.rstrip('/')}/healthz"
                try:
                    if _httpx is not None:
                        async with _httpx.AsyncClient(timeout=10.0) as client:
                            await client.get(health_url)
                    else:
                        import urllib.request

                        req = urllib.request.Request(health_url, headers={"User-Agent": "BotVoice-KeepAwake/1.0"})
                        await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=10.0).close())
                    logger.debug("Keep-awake ping sent to %s", health_url)
                except Exception as e:
                    logger.debug("Keep-awake ping suppressed: %s", e)

            await asyncio.sleep(600)
    except asyncio.CancelledError:
        logger.debug("keep_awake background task stopped cleanly.")


async def periodic_system_maintenance() -> None:
    """Run periodic maintenance (temp files, cache, GC) every 2 hours, and DB pruning daily."""
    try:
        await asyncio.sleep(120)
        last_db_prune = 0.0
        while True:
            try:
                from app.services.admin.optimization import run_system_optimization_async

                now = time.monotonic()
                should_prune_db = (now - last_db_prune) >= 86400.0 or last_db_prune == 0.0
                stats = await run_system_optimization_async(prune_db=should_prune_db)
                if should_prune_db:
                    last_db_prune = now
                gc.collect()
                logger.info("Periodic system maintenance complete: %s", stats)
            except Exception as exc:
                logger.warning("Periodic system maintenance error: %s", exc)
            await asyncio.sleep(7200)
    except asyncio.CancelledError:
        logger.debug("periodic_system_maintenance background task stopped cleanly.")


periodic_database_pruner = periodic_system_maintenance


async def bot_runner_supervisor() -> None:
    """Supervise the Telegram bot runner and restart it automatically if it crashes."""
    backoff = 5
    while True:
        try:
            logger.info("Starting Telegram bot runner under supervisor...")
            if hasattr(legacy, "_async_main_once"):
                await legacy._async_main_once()
            else:
                from app.bot import run_bot_async

                await run_bot_async()
            logger.warning("Telegram bot runner exited cleanly. Restarting in %ss...", backoff)
        except asyncio.CancelledError:
            logger.info("Telegram bot runner supervisor received cancellation.")
            raise
        except Exception as exc:
            logger.error("Telegram bot runner crashed: %s. Restarting in %ss...", exc, backoff, exc_info=True)
        await asyncio.sleep(backoff)


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Start the Telegram Bot runner and manage life cycles of all background services."""
    from app.services.tasks.batcher import get_database_batcher
    from app.services.tasks.queue import get_task_manager

    # 1. Start database write buffer worker safely
    batcher = get_database_batcher()
    batcher_task: asyncio.Task | None = None
    if hasattr(batcher, "start_worker"):
        batcher_task = _safe_create_task(batcher.start_worker, "db-batcher-worker")

    # 2. Start bot runner supervisor and background schedulers
    bot_task = _safe_create_task(bot_runner_supervisor, "telegram-bot-runner")
    auto_reg_task = _safe_create_task(auto_register_all, "auto-register-all")
    keep_awake_task = _safe_create_task(keep_awake, "keep-awake")
    maintenance_task = _safe_create_task(periodic_system_maintenance, "system-maintenance")

    # 3. Start auxiliary tool services safely (fault-tolerant)
    temp_mail_task: asyncio.Task | None = None
    with suppress(Exception):
        from app.services.tools.temp_mail import temp_mail_manager

        temp_mail_task = _safe_create_task(temp_mail_manager.poll_inboxes, "temp-mail-poller")

    bakong_task: asyncio.Task | None = None
    with suppress(Exception):
        from app.services.donation.handlers import periodic_bakong_auto_checker

        bakong_task = _safe_create_task(periodic_bakong_auto_checker, "bakong-auto-checker")

    podcast_task: asyncio.Task | None = None
    with suppress(Exception):
        from app.services.podcast import periodic_podcast_scheduler

        podcast_task = _safe_create_task(periodic_podcast_scheduler, "podcast-scheduler")

    article_monitor_task: asyncio.Task | None = None
    with suppress(Exception):
        from app.services.ai.article_monitor import periodic_article_monitor_scheduler

        article_monitor_task = _safe_create_task(
            lambda: periodic_article_monitor_scheduler(180.0), "article-monitor"
        )

    try:
        yield
    finally:
        logger.info("FastAPI lifespan shutdown initiated. Executing ordered teardown...")

        # Step 1: Cancel active producers/runners so no new tasks or DB entries are created
        producers = [
            t
            for t in (
                bot_task,
                temp_mail_task,
                bakong_task,
                podcast_task,
                article_monitor_task,
                keep_awake_task,
                auto_reg_task,
                maintenance_task,
            )
            if t is not None and not t.done()
        ]
        for t in producers:
            t.cancel()
        for t in producers:
            with suppress(asyncio.CancelledError, Exception):
                await t

        # Step 2: Drain outbound Telegram dispatchers and task manager
        with suppress(Exception):
            from app.services.telegram.dispatcher import get_telegram_dispatcher

            await get_telegram_dispatcher().drain(timeout=10.0)

        with suppress(Exception):
            await get_task_manager().drain(timeout=10.0)

        # Step 3: Drain batched database writes to avoid data loss
        with suppress(Exception):
            await get_database_batcher().drain(timeout=5.0)

        # Step 4: Cancel batcher worker
        if batcher_task is not None and not batcher_task.done():
            batcher_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await batcher_task

        logger.info("FastAPI lifespan teardown completed cleanly.")


fastapi_kwargs: dict[str, Any] = {
    "title": "Telegram Bot Voice & AI Assistant Suite",
    "description": "Multilingual Voice Synthesis, AI Vision OCR, and Gemini Assistant API.",
    "version": "4.2.0",
    "lifespan": lifespan,
}
if _default_response_class is not None:
    fastapi_kwargs["default_response_class"] = _default_response_class

app = FastAPI(**fastapi_kwargs)


@app.get("/healthz", include_in_schema=False)
@app.get("/health", include_in_schema=False)
async def health_check() -> dict[str, Any]:
    """Lightweight root health probe endpoint for container orchestrators."""
    return {"status": "ok", "timestamp": time.time()}


# Register request lifecycle logging middleware
app.middleware("http")(request_lifecycle_logging_middleware)

# Include modular API routers
app.include_router(api_router)


def main() -> None:
    """Run the FastAPI application with Uvicorn on $PORT with single-process guarantee."""
    import uvicorn
    from app.utils.logging import configure_server_logging

    configure_server_logging()
    port = _safe_int(os.environ.get("PORT") or getattr(SETTINGS, "PORT", 8080), 8080)
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info", workers=1)  # noqa: S104


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