"""Telegram handler registration for the single-process application."""

from __future__ import annotations

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

from app.services.telegram.callbacks import (
    _runtime_admin_callback,
    broadcast_callback,
    on_callback,
    sched_callback,
    users_page_callback,
)
from app.services.telegram.commands import (
    admin_stats,
    broadcast_start,
    cmd_admin,
    cmd_api,
    cmd_botsettings,
    cmd_cancel,
    cmd_cancelschedule,
    cmd_chat,
    cmd_clear,
    cmd_delete_my_data,
    cmd_endchat,
    cmd_feature_request,
    cmd_health,
    cmd_myprefs,
    cmd_privacy,
    cmd_runtime,
    cmd_schedule,
    cmd_schedules,
    cmd_security,
    cmd_ttsmodel,
    cmd_users,
    on_help,
    on_start,
)
from app.services.telegram.guards import (
    _drop_stale_updates,
    _telegram_rate_limit_guard,
    _telegram_user_security_guard,
    error_handler,
)
from app.services.telegram.media import (
    on_any_media,
    on_audio_file,
    on_photo,
    on_text,
    on_voice,
)


def register_telegram_handlers(application: Application, *, bot_mode: str) -> None:
    """Register all Telegram handlers in deterministic priority order."""

    application.add_handler(TypeHandler(Update, _telegram_rate_limit_guard), group=-3)
    application.add_handler(TypeHandler(Update, _telegram_user_security_guard), group=-2)
    if str(bot_mode).upper() != "WEBHOOK":
        application.add_handler(TypeHandler(Update, _drop_stale_updates), group=-1)

    command_handlers = (
        ("start", on_start),
        ("help", on_help),
        ("myprefs", cmd_myprefs),
        ("ttsmodel", cmd_ttsmodel),
        ("clear", cmd_clear),
        ("security", cmd_security),
        ("privacy", cmd_privacy),
        ("deleteme", cmd_delete_my_data),
        ("broadcast", broadcast_start),
        ("schedule", cmd_schedule),
        ("schedules", cmd_schedules),
        ("cancelschedule", cmd_cancelschedule),
        ("cancel", cmd_cancel),
        ("stats", admin_stats),
        ("health", cmd_health),
        ("admin", cmd_admin),
        ("need", cmd_feature_request),
        ("feedback", cmd_feature_request),
        ("request_feature", cmd_feature_request),
        ("runtime", cmd_runtime),
        ("api", cmd_api),
        ("botsettings", cmd_botsettings),
        ("users", cmd_users),
        ("chat", cmd_chat),
        ("endchat", cmd_endchat),
    )
    for command, callback in command_handlers:
        application.add_handler(CommandHandler(command, callback))

    application.add_handler(CallbackQueryHandler(broadcast_callback, pattern=r"^bc_"))
    application.add_handler(
        CallbackQueryHandler(
            users_page_callback,
            pattern=(
                r"^(?:users_(?:page:\d+|search(?:_page:\d+)?|close)|noop|"
                r"history_(?:page:\d+|refresh|close|user:\d+(?::\d+)?)|"
                r"user_(?:view|chat|block|unblock|resetprefs|clearhist|history):"
                r"\d+(?::[psh]\d+)?(?::\d+)?)$"
            ),
        )
    )
    application.add_handler(CallbackQueryHandler(sched_callback, pattern=r"^sched_"))
    application.add_handler(
        CallbackQueryHandler(_runtime_admin_callback, pattern=r"^rtadmin_")
    )
    application.add_handler(CallbackQueryHandler(on_callback))

    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    application.add_handler(
        MessageHandler((filters.Document.ALL | filters.AUDIO) & ~filters.VOICE, on_audio_file)
    )
    application.add_handler(
        MessageHandler(filters.Sticker.ALL | filters.VIDEO | filters.VIDEO_NOTE, on_any_media)
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.add_error_handler(error_handler)


__all__ = ["register_telegram_handlers"]
