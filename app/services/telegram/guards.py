"""Extracted Telegram handler implementations.

Security gates, rate-limiting guards, stale-update droppers, and centralized error handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import suppress
from typing import Any, Callable

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop, ContextTypes

from app import legacy
from app.services.telegram.flow import is_nonfatal_telegram_edit_error

# Transitional V4.1 modules bind remaining legacy helpers at runtime.
# ruff: noqa: F821
try:
    from app.services.telegram._legacy_runtime import legacy_bound_handler
except ImportError:
    def legacy_bound_handler(fn: Callable) -> Callable:
        return fn

logger = logging.getLogger("app.guards")

# Module startup reference timestamp for stale update filtering
_BOT_START_TIME: float = time.time()
_STALE_GRACE_S: float = float(os.getenv("STALE_UPDATE_GRACE_SECONDS", "5.0"))
USER_RATE_LIMIT_NOTICE_COOLDOWN_S: float = 15.0

# In-memory sliding-window rate limit state
_RATE_LIMIT_LOCK = threading.RLock()
_RATE_LIMIT_TIMESTAMPS: dict[str, list[float]] = {}
_RATE_LIMIT_NOTICES: dict[str, float] = {}
_FLOOD_COOLDOWNS: dict[str, float] = {}
_VIOLATION_COUNTS: dict[str, int] = {}

# Standard administrative commands
_DEFAULT_ADMIN_COMMANDS = frozenset({
    "admin", "stats", "health", "broadcast", "schedule", "schedules",
    "cancelschedule", "dbstatus", "database", "dbbackup", "backup",
    "migrate", "api", "botsettings", "users", "chat", "endchat",
    "runtime", "bakongstatus", "adddonor", "testblessing",
})


def _is_admin(user_id: int) -> bool:
    """Check administrator status across authorizer policy, environment, and legacy."""
    if not user_id:
        return False

    with suppress(Exception):
        from app.core.telegram_auth import is_telegram_admin
        if is_telegram_admin(user_id):
            return True

    is_admin_fn = getattr(legacy, "_is_admin", getattr(legacy, "is_admin", None))
    if callable(is_admin_fn):
        with suppress(Exception):
            if is_admin_fn(user_id):
                return True

    admin_ids: set[int] = set()
    with suppress(Exception):
        from app.core.config import SETTINGS
        for src in (getattr(SETTINGS, "ADMIN_IDS", None),):
            if isinstance(src, (set, list, tuple)):
                admin_ids.update(int(a) for a in src if str(a).lstrip("-").isdigit())
            elif isinstance(src, str):
                admin_ids.update(int(a.strip()) for a in src.split(",") if a.strip().lstrip("-").isdigit())

    for env_a in os.environ.get("ADMIN_IDS", "").split(","):
        if env_a.strip().lstrip("-").isdigit():
            admin_ids.add(int(env_a.strip()))

    return user_id in admin_ids


def _metric_inc(name: str, **kwargs: Any) -> None:
    """Record metric counter safely."""
    inc_fn = getattr(legacy, "_metric_inc", None)
    if callable(inc_fn):
        with suppress(Exception):
            inc_fn(name, **kwargs)


def _get_rate_limit_key(update: Update) -> str:
    """Generate a stable tracking key for incoming requests."""
    user = update.effective_user
    if user:
        return f"user:{user.id}"
    chat = update.effective_chat
    if chat:
        return f"chat:{chat.id}"
    return f"upd:{getattr(update, 'update_id', 0)}"


async def safe_send(coro_or_fn: Any) -> Any:
    """Execute Telegram send operation with automatic recovery from errors."""
    try:
        res = coro_or_fn() if callable(coro_or_fn) else coro_or_fn
        if asyncio.iscoroutine(res):
            return await res
        return res
    except BadRequest as b_err:
        err_msg = str(b_err).lower()
        if "can't parse entities" in err_msg or "tag" in err_msg:
            logger.warning("Telegram HTML parse error: %s", b_err)
            return None
        logger.error("safe_send BadRequest: %s", b_err)
        return None
    except Exception as exc:
        logger.debug("safe_send failed: %s", exc)
        return None


def _get_command_name(update: Update) -> str:
    """Extract command name from an update."""
    with suppress(Exception):
        from app.services.telegram.security import _telegram_command_name
        cmd = _telegram_command_name(update)
        if cmd:
            return cmd.lower().lstrip("/")

    msg = update.effective_message
    if msg and getattr(msg, "text", None) and msg.text.startswith("/"):
        return msg.text.split()[0].lstrip("/").split("@")[0].lower()
    return ""


@legacy_bound_handler
async def _telegram_rate_limit_guard(update: Any, context: Any) -> None:
    if not isinstance(update, Update):
        return

    # Channel posts are managed broadcasts; do not throttle them
    if getattr(update, "channel_post", None) is not None:
        return

    user = getattr(update, "effective_user", None)
    if user and _is_admin(int(user.id)):
        # Admins are exempt from rate limiting for emergency control
        return

    key = _get_rate_limit_key(update)
    now = time.monotonic()

    # 1. Check Anti-Flood Cooldown
    with _RATE_LIMIT_LOCK:
        cooldown_until = _FLOOD_COOLDOWNS.get(key, 0.0)
        if now < cooldown_until:
            _metric_inc("flood_blocked")
            raise ApplicationHandlerStop

    # 2. Check rate limit sliding window (default: 5 requests per 3.0s)
    rate_lim = getattr(legacy, "_run_state_user_rate_limit", lambda: 5)()
    rate_win = getattr(legacy, "_run_state_user_rate_window", lambda: 3.0)()

    allowed = True
    with _RATE_LIMIT_LOCK:
        history = _RATE_LIMIT_TIMESTAMPS.setdefault(key, [])
        threshold = now - rate_win
        _RATE_LIMIT_TIMESTAMPS[key] = [t for t in history if t > threshold]

        if len(_RATE_LIMIT_TIMESTAMPS[key]) >= rate_lim:
            allowed = False
        else:
            _RATE_LIMIT_TIMESTAMPS[key].append(now)

    if allowed:
        return

    _metric_inc("rate_limited")
    should_send_notice = False
    flood_detected = False

    with _RATE_LIMIT_LOCK:
        violations = _VIOLATION_COUNTS.get(key, 0) + 1
        _VIOLATION_COUNTS[key] = violations

        if violations >= 6:
            # Excessive rapid queries: progressive cooldown
            cooldown_s = min(300.0, 45.0 * (violations - 5))
            _FLOOD_COOLDOWNS[key] = now + cooldown_s
            flood_detected = True

        last_notice = _RATE_LIMIT_NOTICES.get(key, 0.0)
        if now - last_notice >= USER_RATE_LIMIT_NOTICE_COOLDOWN_S:
            _RATE_LIMIT_NOTICES[key] = now
            should_send_notice = True

            # Prune memory if cache gets excessively large
            if len(_RATE_LIMIT_NOTICES) > 10_000:
                stale_cutoff = now - 300.0
                for k, ts in list(_RATE_LIMIT_NOTICES.items()):
                    if ts < stale_cutoff:
                        _RATE_LIMIT_NOTICES.pop(k, None)
                        _RATE_LIMIT_TIMESTAMPS.pop(k, None)

    if should_send_notice:
        chat = getattr(update, "effective_chat", None)
        msg = getattr(update, "effective_message", None)
        if msg is not None and getattr(chat, "type", None) != "channel":
            with suppress(Exception):
                if flood_detected:
                    await safe_send(lambda: msg.reply_text(
                        "🛑 <b>ការផ្ញើសារលឿនពេក (Anti-Spam)</b>\n\n"
                        "ប្រព័ន្ធបានដាក់ Cooldown បណ្តោះអាសន្ន ៤៥ វិនាទី។ សូមរង់ចាំបន្តិចមុនពេលផ្ញើសារបន្ទាប់។",
                        parse_mode="HTML"
                    ))
                else:
                    await safe_send(lambda: msg.reply_text(
                        "⚠️ <b>សូមកុំផ្ញើសារញឹកញាប់ពេក!</b>\n\n"
                        "ប្រព័ន្ធបានកំណត់កម្រិតផ្ញើសារ (Rate Limit)។ សូមរង់ចាំបន្តិចមុននឹងផ្ញើបន្ត។",
                        parse_mode="HTML"
                    ))
    raise ApplicationHandlerStop


@legacy_bound_handler
async def _telegram_user_security_guard(update: Any, context: Any) -> None:
    """Global user-safety gate protecting admin endpoints from unauthorized access."""
    if not isinstance(update, Update):
        return
    user = update.effective_user
    if user is None:
        return
    user_id = int(user.id)
    if _is_admin(user_id):
        return

    # Check restricted admin callback prefixes
    query = update.callback_query
    data = str(getattr(query, "data", "") or "") if query is not None else ""
    admin_callback_prefixes = (
        "admin_", "needs_", "api_", "rtadmin_", "user_", "users_",
        "history_", "sched_", "bc_", "admin_report_", "cfg_", "cfg_cat:", "cfg_set:",
    )

    admin_guard_enabled = str(os.getenv("ADMIN_CALLBACK_GUARD_ENABLED", "true")).lower() in ("true", "1", "yes")
    if admin_guard_enabled and data.startswith(admin_callback_prefixes):
        _metric_inc("admin_denied")
        if query:
            with suppress(Exception):
                await query.answer("⛔ សម្រាប់អ្នកគ្រប់គ្រងប៉ុណ្ណោះ (Admin only)។", show_alert=True)
        raise ApplicationHandlerStop

    # Check restricted admin commands
    cmd = _get_command_name(update)
    admin_commands = getattr(legacy, "_ADMIN_ONLY_COMMANDS", _DEFAULT_ADMIN_COMMANDS)
    if cmd in admin_commands:
        _metric_inc("admin_denied")
        msg = update.effective_message
        if msg:
            with suppress(Exception):
                await safe_send(lambda: msg.reply_text("⛔ ពាក្យបញ្ជានេះសម្រាប់ Admin ប៉ុណ្ណោះ។"))
        raise ApplicationHandlerStop


@legacy_bound_handler
async def _drop_stale_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Drop outdated message updates received while the bot was offline in polling mode."""
    # Never drop updates in Webhook mode
    bot_mode = os.getenv("BOT_MODE", "").strip().upper()
    if bot_mode == "WEBHOOK":
        return

    run_state_mode = getattr(legacy, "_run_state_bot_mode", None)
    if callable(run_state_mode) and run_state_mode() == "WEBHOOK":
        return

    if _BOT_START_TIME == 0.0:
        return

    # Only filter messages, not callback queries
    msg = update.message or update.edited_message or update.channel_post
    if msg and getattr(msg, "date", None):
        msg_ts = msg.date.timestamp()
        if msg_ts < (_BOT_START_TIME - _STALE_GRACE_S):
            logger.debug("Dropping stale message update (id=%s, age=%.1fs)", update.update_id, _BOT_START_TIME - msg_ts)
            raise ApplicationHandlerStop


@legacy_bound_handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Centralized exception handling for all Telegram updates."""
    err = getattr(context, "error", None)
    if err is None:
        return

    update_id = getattr(update, "update_id", None) if update else None
    t0 = getattr(update, "_telemetry_start_time", None) if update else None
    dur_ms = (time.monotonic() - t0) * 1000.0 if t0 else 0.0

    # 1. Non-fatal conditions (Handler stop or message unedited)
    if isinstance(err, ApplicationHandlerStop) or is_nonfatal_telegram_edit_error(err):
        if update_id is not None:
            with suppress(Exception):
                from app.services.telegram.telemetry import record_telegram_update_complete

                record_telegram_update_complete(
                    update_id,
                    success=True,
                    duration_ms=dur_ms,
                    detail="Handler stopped / non-fatal",
                )
        logger.debug("Ignored non-fatal Telegram handler condition: %s", err)
        return

    # 2. Transient network errors (Gateway timeout, connection reset, etc.)
    err_str = str(err).lower()
    err_cls = type(err).__name__.lower()

    is_transient_network = any(term in err_str or term in err_cls for term in (
        "bad gateway", "gateway timeout", "timed out", "timedout", "readtimeout",
        "connecttimeout", "connect timeout", "pooltimeout", "pool timeout",
        "remoteprotocolerror", "remote protocol error", "server disconnected",
        "connection reset", "connection closed", "connection refused",
        "network is unreachable", "networkerror", "network error",
        "transport error", "transporterror", "502", "504",
    ))

    if not is_transient_network:
        try:
            import httpx
            if isinstance(err, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError)):
                is_transient_network = True
        except Exception:
            pass

    if is_transient_network:
        if update_id is not None:
            with suppress(Exception):
                from app.services.telegram.telemetry import record_telegram_update_complete

                record_telegram_update_complete(
                    update_id,
                    success=False,
                    duration_ms=dur_ms,
                    detail="Transient network issue",
                )
        logger.warning("Telegram transient upstream network issue in handler: %s", err)
        return

    # 3. Log unhandled exceptions
    _metric_inc("errors")
    rec_err = getattr(legacy, "_record_admin_error", None)
    if callable(rec_err):
        with suppress(Exception):
            rec_err("telegram_error_handler", str(err), level="ERROR", context=type(update).__name__)

    if update_id is not None:
        with suppress(Exception):
            from app.services.telegram.telemetry import record_telegram_update_complete

            record_telegram_update_complete(
                update_id,
                success=False,
                duration_ms=dur_ms,
                detail=str(err)[:60],
            )

    logger.error("Unhandled Telegram exception: %s", err, exc_info=err)

    # 4. User feedback
    if isinstance(update, Update):
        if update.callback_query:
            with suppress(Exception):
                await update.callback_query.answer("⚠️ មានបញ្ហាបច្ចេកទេស។ សូមព្យាយាមម្តងទៀត។", show_alert=True)
        elif update.effective_message:
            chat = getattr(update, "effective_chat", None)
            if getattr(chat, "type", None) == "private":
                with suppress(Exception):
                    await safe_send(lambda: update.effective_message.reply_text(
                        "⚠️ មានបញ្ហាបច្ចេកទេស។ Bot នៅដំណើរការ — សូមព្យាយាមម្តងទៀត។"
                    ))


def __getattr__(name: str) -> Any:
    """Fallback dynamically to app.legacy or _legacy_runtime for transitional symbols."""
    with suppress(Exception):
        if hasattr(legacy, name):
            return getattr(legacy, name)
    with suppress(Exception):
        import app.services.telegram._legacy_runtime as lr
        if hasattr(lr, name):
            return getattr(lr, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    '_drop_stale_updates',
    '_telegram_rate_limit_guard',
    '_telegram_user_security_guard',
    'error_handler',
]