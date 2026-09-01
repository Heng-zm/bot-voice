"""Extracted Telegram handler implementations.

These are live runtime handlers; app.legacy now contains compatibility wrappers only.
"""

from __future__ import annotations

# Transitional V4.1 modules bind remaining legacy helpers at runtime.
# ruff: noqa: F821
from app.services.telegram._legacy_runtime import legacy_bound_handler
from app.services.telegram.security import (
    _ADMIN_ONLY_COMMANDS,
    _security_notice_once,
    _telegram_command_name,
    is_user_blocked,
)


@legacy_bound_handler
async def _telegram_rate_limit_guard(update: Any, context: Any) -> None:
    if not isinstance(update, Update):
        return
    key = _update_rate_limit_key(update)
    allowed, _remaining = await _rate_limit_check(key, _run_state_user_rate_limit(), _run_state_user_rate_window())
    if allowed:
        return
    _metric_inc("rate_limited")
    now = time.monotonic()
    should_send_notice = False
    with _RATE_LIMIT_MEMORY_THREAD_LOCK:
        last = _RATE_LIMIT_NOTICE_MEMORY.get(key, 0.0)
        if now - last >= USER_RATE_LIMIT_NOTICE_COOLDOWN_S:
            _RATE_LIMIT_NOTICE_MEMORY[key] = now
            should_send_notice = True
            if len(_RATE_LIMIT_NOTICE_MEMORY) > 10_000:
                stale_before = now - max(USER_RATE_LIMIT_NOTICE_COOLDOWN_S * 4, 300.0)
                for old_key, old_ts in list(_RATE_LIMIT_NOTICE_MEMORY.items()):
                    if old_ts < stale_before:
                        _RATE_LIMIT_NOTICE_MEMORY.pop(old_key, None)
                while len(_RATE_LIMIT_NOTICE_MEMORY) > 10_000:
                    _RATE_LIMIT_NOTICE_MEMORY.pop(next(iter(_RATE_LIMIT_NOTICE_MEMORY)), None)
    if should_send_notice:
        msg = getattr(update, "effective_message", None)
        if msg is not None:
            with suppress(Exception):
                await msg.reply_text(_runtime_admin_notice_text())
    raise ApplicationHandlerStop


@legacy_bound_handler
async def _telegram_user_security_guard(update: Any, context: Any) -> None:
    """Cheap global user-safety gate before expensive handlers.

    - Blocks non-admin access to admin commands/callbacks.
    - Stops blocked users before OCR/TTS/AI work begins.
    - Uses cached blocked lookups to avoid DB pressure under normal traffic.
    """
    if not isinstance(update, Update):
        return
    user = update.effective_user
    if user is None:
        return
    user_id = int(user.id)
    if _is_admin(user_id):
        return

    query = update.callback_query
    data = str(getattr(query, "data", "") or "") if query is not None else ""
    admin_callback_prefixes = ("admin_", "needs_", "api_", "rtadmin_", "user_", "users_", "history_", "sched_", "bc_", "admin_report_")
    if _env_bool("ADMIN_CALLBACK_GUARD_ENABLED", True) and data.startswith(admin_callback_prefixes):
        _metric_inc("admin_denied")
        await _security_notice_once(update, f"admin_cb:{user_id}", '⛔ សម្រាប់អ្នកគ្រប់គ្រងប៉ុណ្ណោះ។', alert=True)
        raise ApplicationHandlerStop

    cmd = _telegram_command_name(update)
    if cmd in _ADMIN_ONLY_COMMANDS:
        _metric_inc("admin_denied")
        await _security_notice_once(update, f"admin_cmd:{user_id}", "⛔ ពាក្យបញ្ជានេះសម្រាប់ Admin ប៉ុណ្ណោះ។")
        raise ApplicationHandlerStop
    if cmd in {"security", "privacy", "deleteme"}:
        # Privacy/self-service commands stay available even if the user is blocked.
        return

    # Cache hits stay on the event loop; only a Supabase miss enters the bounded
    # database executor.
    try:
        blocked = await is_user_blocked(user_id)
    except Exception as exc:
        blocked = False
        _log_once(logging.WARNING, f"blocked_guard_failed:{user_id}", "Blocked-user guard skipped user=%s: %s", user_id, exc)
    if blocked:
        _metric_inc("blocked_hits")
        await _security_notice_once(update, f"blocked:{user_id}", "⛔ អ្នកត្រូវបាន Block មិនអាចប្រើ Bot នេះបានទេ។")
        raise ApplicationHandlerStop


@legacy_bound_handler
async def _drop_stale_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Drop old message updates received before bot started.

    FIX: For callback_query updates, we intentionally do NOT drop them based on
    message date — the message date is when the original message was sent, not
    when the user tapped the button. Dropping callbacks based on message age
    would break buttons on messages sent before the bot restarted. We only drop
    stale *message* updates.
    """
    # Webhook mode must not drop old message timestamps. During cold starts,
    # this app returns 503 until Telegram is ready, and Telegram may retry the
    # same update later with the original message date.  Dropping it here would
    # lose user messages after a slow Render restart/deploy.
    if "_run_state_bot_mode" in globals() and _run_state_bot_mode() == "WEBHOOK":
        return

    if _BOT_START_TIME == 0.0:
        return

    # Only filter plain messages (not callbacks)
    msg = update.message or update.edited_message
    if msg and getattr(msg, "date", None) and msg.date.timestamp() < (
        _BOT_START_TIME - _STALE_GRACE_S
    ):
        logger.debug(f"Dropping stale message update (id={update.update_id})")
        raise ApplicationHandlerStop


@legacy_bound_handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    err = getattr(context, "error", None)
    if err is None:
        return

    if isinstance(err, ApplicationHandlerStop) or _is_nonfatal_telegram_edit_error(err):
        logger.debug("Ignored non-fatal Telegram handler condition: %s", err)
        return

    _metric_inc("errors")
    _record_admin_error("telegram_error_handler", str(err), level="ERROR", context=type(update).__name__)
    logger.error(
        "Unhandled exception: %s",
        err,
        exc_info=(type(err), err, getattr(err, "__traceback__", None)),
    )

    if isinstance(update, Update) and update.effective_message:
        with suppress(Exception):
            await safe_send(lambda: update.effective_message.reply_text(
                "⚠️ មានបញ្ហាបច្ចេកទេស។ Bot នៅដំណើរការ — សូមព្យាយាមម្តងទៀត។"
            ))


__all__ = [
    '_telegram_rate_limit_guard',
    '_telegram_user_security_guard',
    '_drop_stale_updates',
    'error_handler'
]
