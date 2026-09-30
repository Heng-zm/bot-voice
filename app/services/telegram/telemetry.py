"""Real-time Telegram Update Request Logging and Telemetry Guard.

Captures all incoming Telegram updates across Webhook and Polling modes,
logs them to the server console unbuffered, and registers them in the RequestLogStore.
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from app.core.logging_middleware import get_request_log_store

logger = logging.getLogger("app.telemetry")


def extract_update_info(update: Update) -> dict[str, Any]:
    """Extract structured, human-readable details from any Telegram Update."""
    update_id = getattr(update, "update_id", None) or 0
    from_user = getattr(update, "effective_user", None)
    chat = getattr(update, "effective_chat", None)

    username = f"@{from_user.username}" if from_user and getattr(from_user, "username", None) else None
    user_id = getattr(from_user, "id", 0) if from_user else 0
    first_name = getattr(from_user, "first_name", "") if from_user else ""

    if username:
        client_str = f"{username} (ID: {user_id})"
    elif first_name:
        client_str = f"{first_name} (ID: {user_id})"
    elif user_id:
        client_str = f"User_{user_id} (ID: {user_id})"
    else:
        client_str = "Anonymous"

    method = "UPDATE"
    path = f"update_{update_id}"
    detail = ""

    # 1. Standard Messages
    if getattr(update, "message", None):
        msg = update.message
        if getattr(msg, "text", None):
            text = str(msg.text).strip()
            if text.startswith("/"):
                method = "COMMAND"
                path = text.split()[0]
                detail = text
            else:
                method = "MESSAGE"
                path = text[:35] + ("..." if len(text) > 35 else "")
                detail = text
        elif getattr(msg, "voice", None):
            method = "VOICE"
            path = "voice_note"
            dur = getattr(msg.voice, "duration", 0)
            detail = f"duration={dur}s"
        elif getattr(msg, "audio", None):
            method = "AUDIO"
            path = getattr(msg.audio, "file_name", "audio_file") or "audio_file"
            detail = getattr(msg.audio, "title", "") or ""
        elif getattr(msg, "video", None):
            method = "VIDEO"
            path = getattr(msg.video, "file_name", "video") or "video"
            dur = getattr(msg.video, "duration", 0)
            detail = f"duration={dur}s size={getattr(msg.video, 'file_size', 0)}B"
        elif getattr(msg, "video_note", None):
            method = "VIDEO_NOTE"
            path = "video_note"
            detail = f"duration={getattr(msg.video_note, 'duration', 0)}s"
        elif getattr(msg, "photo", None):
            method = "PHOTO"
            path = "photo_upload"
            detail = f"{len(msg.photo)} sizes"
        elif getattr(msg, "document", None):
            method = "DOCUMENT"
            path = getattr(msg.document, "file_name", "document") or "document"
            detail = f"size={getattr(msg.document, 'file_size', 0)}B"
        elif getattr(msg, "sticker", None):
            method = "STICKER"
            emoji = getattr(msg.sticker, "emoji", "")
            path = f"sticker:{emoji}" if emoji else "sticker"
            detail = getattr(msg.sticker, "set_name", "") or "sticker"
        elif getattr(msg, "location", None):
            method = "LOCATION"
            path = "location"
            detail = f"lat={getattr(msg.location, 'latitude', 0)},lon={getattr(msg.location, 'longitude', 0)}"
        elif getattr(msg, "contact", None):
            method = "CONTACT"
            path = "contact"
            detail = f"phone={getattr(msg.contact, 'phone_number', '')}"
        elif getattr(msg, "successful_payment", None):
            sp = msg.successful_payment
            method = "PAYMENT"
            path = f"{getattr(sp, 'total_amount', 0)} {getattr(sp, 'currency', '')}"
            detail = f"payload={getattr(sp, 'invoice_payload', '')}"
        else:
            method = "MESSAGE"
            path = "other_media"

    # 2. Interactive & Inline Callbacks
    elif getattr(update, "callback_query", None):
        cb = update.callback_query
        method = "CALLBACK"
        path = str(getattr(cb, "data", "") or "")
        detail = f"callback_data={path}"
    elif getattr(update, "channel_post", None):
        post = update.channel_post
        method = "CHANNEL_POST"
        path = (str(getattr(post, "text", "") or "media"))[:35]
        detail = str(getattr(post, "text", "") or "")
    elif getattr(update, "inline_query", None):
        iq = update.inline_query
        method = "INLINE_QUERY"
        path = str(getattr(iq, "query", "") or "")[:35]
        detail = str(getattr(iq, "query", "") or "")
    elif getattr(update, "edited_message", None):
        emsg = update.edited_message
        method = "EDITED_MSG"
        text = str(getattr(emsg, "text", "") or "")
        path = (text[:35] + ("..." if len(text) > 35 else "")) if text else "edited_media"
        detail = text or "edited_media"
    elif getattr(update, "edited_channel_post", None):
        post = update.edited_channel_post
        method = "EDITED_POST"
        path = (str(getattr(post, "text", "") or "edited_media"))[:35]
        detail = str(getattr(post, "text", "") or "")
    elif getattr(update, "my_chat_member", None) or getattr(update, "chat_member", None):
        method = "CHAT_MEMBER"
        cm = getattr(update, "my_chat_member", None) or getattr(update, "chat_member", None)
        new_status = getattr(getattr(cm, "new_chat_member", None), "status", "unknown")
        path = f"status:{new_status}"
        detail = f"chat_member_update:{new_status}"
    elif getattr(update, "chosen_inline_result", None):
        cir = update.chosen_inline_result
        method = "CHOSEN_INLINE"
        path = str(getattr(cir, "result_id", "") or "")[:35]
        detail = str(getattr(cir, "query", "") or "")

    # 5. Payments
    elif getattr(update, "pre_checkout_query", None):
        pcq = update.pre_checkout_query
        method = "PRE_CHECKOUT"
        path = str(getattr(pcq, "invoice_payload", "") or "")[:35]
        detail = f"amount={getattr(pcq, 'total_amount', 0)}"

    return {
        "update_id": update_id,
        "method": method,
        "path": path,
        "client": client_str,
        "detail": detail[:120],
        "chat_id": getattr(chat, "id", None),
    }


async def telegram_request_telemetry_guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """TypeHandler running at priority group=-4 to log and record every incoming Telegram request."""
    try:
        setattr(update, "_telemetry_start_time", time.monotonic())
        info = extract_update_info(update)
        req_id = f"tg-{info['update_id']}"

        # 1. Real-time console log
        logger.info(
            "📥 [TG REQ] %s: %s | from %s | update_id=%s",
            info["method"],
            info["path"],
            info["client"],
            info["update_id"],
        )

        # 2. Record start in RequestLogStore
        store = get_request_log_store()
        store.record_start(
            request_id=req_id,
            category="TELEGRAM",
            method=info["method"],
            path=info["path"],
            client=info["client"],
            detail=info["detail"],
        )
    except Exception as exc:
        logger.debug("telemetry_guard start logging suppressed error: %s", exc)


async def telegram_request_telemetry_complete_guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """TypeHandler running at trailing priority group=100 to record completion in Polling mode."""
    update_id = getattr(update, "update_id", None)
    if update_id is None:
        return
    t0 = getattr(update, "_telemetry_start_time", None)
    dur_ms = (time.monotonic() - t0) * 1000.0 if t0 else 0.0
    record_telegram_update_complete(update_id, success=True, duration_ms=dur_ms, update=update)


async def telegram_telemetry_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for Telegram Application capturing unhandled exceptions in telemetry."""
    err = context.error
    update_id = getattr(update, "update_id", None) if update else "unknown"
    t0 = getattr(update, "_telemetry_start_time", None) if update else None
    dur_ms = (time.monotonic() - t0) * 1000.0 if t0 else 0.0

    err_str = str(err)[:120] if err else "Unhandled exception"
    record_telegram_update_complete(
        update_id,
        success=False,
        duration_ms=dur_ms,
        detail=err_str,
        update=update if isinstance(update, Update) else None,
    )


def record_telegram_update_complete(
    update_id: int | str,
    success: bool = True,
    duration_ms: float = 0.0,
    detail: str = "",
    update: Update | None = None,
) -> None:
    """Record completion of a Telegram update processing lifecycle."""
    store = get_request_log_store()
    req_id = f"tg-{update_id}" if not str(update_id).startswith("tg-") else str(update_id)

    # Recalculate duration if missing but start timestamp is available on update
    if duration_ms <= 0.0 and update is not None:
        t0 = getattr(update, "_telemetry_start_time", None)
        if t0:
            duration_ms = (time.monotonic() - t0) * 1000.0

    # Defensive check against duplicate completions
    with suppress(Exception):
        if hasattr(store, "_lock"):
            with store._lock:
                record = getattr(store, "_records_map", {}).get(req_id)
                if record is None and hasattr(store, "_buffer"):
                    for r in store._buffer:
                        if getattr(r, "id", None) == req_id:
                            record = r
                            break
                if record is not None and getattr(record, "status", "") != "PENDING":
                    return

    status = "SUCCESS" if success else "FAILED"
    code = 200 if success else 500

    with suppress(Exception):
        store.record_complete(
            request_id=req_id,
            status=status,
            status_code=code,
            duration_ms=duration_ms,
            detail=detail,
        )

    icon = "⚡" if success else "❌"
    logger.info(
        "%s [TG RES] %s | update_id=%s in %.2fms %s",
        icon,
        status,
        update_id,
        duration_ms,
        f"({detail})" if detail else "",
    )


def install_telemetry_handlers(app: Any) -> None:
    """Register telemetry guard handlers and error tracking onto a Telegram Application."""
    from telegram.ext import TypeHandler

    app.add_handler(TypeHandler(Update, telegram_request_telemetry_guard), group=-4)
    app.add_handler(TypeHandler(Update, telegram_request_telemetry_complete_guard), group=100)
    app.add_error_handler(telegram_telemetry_error_handler)


def get_scaling_telemetry_snapshot() -> dict[str, Any]:
    """Return consolidated telemetry across dispatcher, workers, queue, and database batcher."""
    snapshot: dict[str, Any] = {}
    with suppress(Exception):
        from app.services.telegram.dispatcher import get_telegram_dispatcher

        snapshot["dispatcher"] = get_telegram_dispatcher().get_metrics()
    with suppress(Exception):
        from app.services.tasks.queue import get_task_manager

        snapshot["tasks"] = get_task_manager().get_metrics()
    with suppress(Exception):
        from app.services.tasks.batcher import get_database_batcher

        snapshot["batcher"] = get_database_batcher().get_metrics()
    with suppress(Exception):
        from app.services.telegram.workloads import get_telegram_workload_limiter

        snapshot["workloads"] = get_telegram_workload_limiter().snapshot()
    with suppress(Exception):
        store = get_request_log_store()
        if hasattr(store, "get_metrics"):
            snapshot["request_store"] = store.get_metrics()

    return snapshot


__all__ = [
    "extract_update_info",
    "get_scaling_telemetry_snapshot",
    "install_telemetry_handlers",
    "record_telegram_update_complete",
    "telegram_request_telemetry_complete_guard",
    "telegram_request_telemetry_error_handler",
    "telegram_request_telemetry_guard",
]