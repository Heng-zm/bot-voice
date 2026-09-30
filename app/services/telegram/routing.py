"""Telegram handler registration for the single-process application."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    InlineQueryHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

from app import legacy
from app.services.donation import (
    cmd_adddonor,
    cmd_donate,
    cmd_donors,
    cmd_testblessing,
    donation_callback,
)
from app.services.podcast import cmd_podcast, podcast_callback
from app.services.telegram.callbacks import (
    _runtime_admin_callback,
    article_callback,
    broadcast_callback,
    facebook_callback,
    instagram_callback,
    on_callback,
    sched_callback,
    tiktok_callback,
    users_page_callback,
    youtube_callback,
)
from app.services.telegram.channel import on_channel_post
from app.services.telegram.commands import (
    admin_stats,
    broadcast_start,
    cmd_admin,
    cmd_api,
    cmd_article_scan,
    cmd_article_sources,
    cmd_ask,
    cmd_bakongstatus,
    cmd_botsettings,
    cmd_cancel,
    cmd_cancelschedule,
    cmd_chat,
    cmd_checkpay,
    cmd_clear,
    cmd_dbbackup,
    cmd_dbstatus,
    cmd_delete_my_data,
    cmd_email,
    cmd_endchat,
    cmd_facebook,
    cmd_feature_request,
    cmd_health,
    cmd_instagram,
    cmd_khqr,
    cmd_menu,
    cmd_migrate,
    cmd_mode,
    cmd_myprefs,
    cmd_narrate,
    cmd_privacy,
    cmd_runtime,
    cmd_schedule,
    cmd_schedules,
    cmd_security,
    cmd_speed,
    cmd_summary,
    cmd_system,
    cmd_tiktok,
    cmd_translate,
    cmd_ttsmodel,
    cmd_unlock,
    cmd_users,
    cmd_voice,
    cmd_youtube,
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
from app.services.telegram.telemetry import (
    telegram_request_telemetry_complete_guard,
    telegram_request_telemetry_guard,
)

logger = logging.getLogger(__name__)


def register_telegram_handlers(application: Application, *, bot_mode: str) -> None:
    """Register all Telegram handlers in deterministic priority order."""

    # ── Group -4 to -1: Guards & Interceptors ──────────────────────────────────
    # Priority -4: Telemetry Start Guard
    application.add_handler(TypeHandler(Update, telegram_request_telemetry_guard), group=-4)

    # Priority -3: Drop stale updates on startup in Polling mode before rate limits
    if str(bot_mode).upper() != "WEBHOOK":
        application.add_handler(TypeHandler(Update, _drop_stale_updates), group=-3)

    # Priority -2: Rate Limiting Guard
    application.add_handler(TypeHandler(Update, _telegram_rate_limit_guard), group=-2)

    # Priority -1: User Authorization & Security Guard
    application.add_handler(TypeHandler(Update, _telegram_user_security_guard), group=-1)

    # ── Group 0: Core Command Handlers ─────────────────────────────────────────
    command_handlers = (
        # General & Help
        ("start", on_start),
        ("help", on_help),
        ("menu", cmd_menu),
        ("mode", cmd_mode),
        ("botmode", cmd_mode),
        # User Preferences & Audio
        ("myprefs", cmd_myprefs),
        ("settings", cmd_myprefs),
        ("setting", cmd_myprefs),
        ("profile", cmd_myprefs),
        ("speed", cmd_speed),
        ("voice", cmd_voice),
        ("ttsmodel", cmd_ttsmodel),
        ("tts", cmd_ttsmodel),
        # Session Management & Security
        ("clear", cmd_clear),
        ("clean", cmd_clear),
        ("security", cmd_security),
        ("privacy", cmd_privacy),
        ("terms", cmd_privacy),
        ("deleteme", cmd_delete_my_data),
        ("unlock", cmd_unlock),
        ("reset", cmd_unlock),
        ("unblock", cmd_unlock),
        ("cancel", cmd_cancel),
        ("stop", cmd_cancel),
        # Administration & Telemetry
        ("admin", cmd_admin),
        ("stats", admin_stats),
        ("health", cmd_health),
        ("runtime", cmd_runtime),
        ("system", cmd_system),
        ("metrics", cmd_system),
        ("broadcast", broadcast_start),
        ("schedule", cmd_schedule),
        ("schedules", cmd_schedules),
        ("cancelschedule", cmd_cancelschedule),
        ("dbstatus", cmd_dbstatus),
        ("database", cmd_dbstatus),
        ("dbbackup", cmd_dbbackup),
        ("backup", cmd_dbbackup),
        ("migrate", cmd_migrate),
        ("api", cmd_api),
        ("botsettings", cmd_botsettings),
        ("users", cmd_users),
        ("chat", cmd_chat),
        ("endchat", cmd_endchat),
        # Feature Requests & Tools
        ("need", cmd_feature_request),
        ("feedback", cmd_feature_request),
        ("request_feature", cmd_feature_request),
        ("email", cmd_email),
        ("tempmail", cmd_email),
        # AI & Reading
        ("ask", cmd_ask),
        ("ai", cmd_ask),
        ("translate", cmd_translate),
        ("trans", cmd_translate),
        ("summary", cmd_summary),
        ("sum", cmd_summary),
        ("narrate", cmd_narrate),
        ("read", cmd_narrate),
        ("article_sources", cmd_article_sources),
        ("article_source", cmd_article_sources),
        ("articlesources", cmd_article_sources),
        ("articlesource", cmd_article_sources),
        ("article_scan", cmd_article_scan),
        ("articlescan", cmd_article_scan),
        # Donations & Bakong
        ("donate", cmd_donate),
        ("coffee", cmd_donate),
        ("pay", cmd_donate),
        ("donors", cmd_donors),
        ("donor", cmd_donors),
        ("halloffame", cmd_donors),
        ("adddonor", cmd_adddonor),
        ("testblessing", cmd_testblessing),
        ("bakongstatus", cmd_bakongstatus),
        ("bakong", cmd_bakongstatus),
        ("checkpay", cmd_checkpay),
        ("checktx", cmd_checkpay),
        ("bakongcheck", cmd_checkpay),
        ("khqr", cmd_khqr),
        ("setkhqr", cmd_khqr),
        # Podcasts
        ("podcast", cmd_podcast),
        ("morning", cmd_podcast),
        ("dailypodcast", cmd_podcast),
        # Downloaders
        ("tiktok", cmd_tiktok),
        ("tt", cmd_tiktok),
        ("facebook", cmd_facebook),
        ("fb", cmd_facebook),
        ("instagram", cmd_instagram),
        ("ig", cmd_instagram),
        ("youtube", cmd_youtube),
        ("yt", cmd_youtube),
    )
    for command, callback in command_handlers:
        application.add_handler(CommandHandler(command, callback))

    # ── Group 0: Callback Query Handlers ───────────────────────────────────────
    # Admin full controller dispatcher
    try:
        from app.services.admin.callbacks import handle_admin_callback

        async def _admin_cb_adapter(update: Update, context: ContextTypes.DEFAULT_TYPE):
            query = update.callback_query
            if query:
                user_id = query.from_user.id if query.from_user else 0
                await handle_admin_callback(query, user_id, context, query.data or "")

        application.add_handler(CallbackQueryHandler(_admin_cb_adapter, pattern=r"^admin_"))
    except ImportError:
        pass

    application.add_handler(CallbackQueryHandler(broadcast_callback, pattern=r"^bc_"))
    application.add_handler(
        CallbackQueryHandler(
            users_page_callback,
            pattern=r"^(?:users_|user_|history_|noop)",
        )
    )
    application.add_handler(CallbackQueryHandler(sched_callback, pattern=r"^sched_"))
    application.add_handler(CallbackQueryHandler(_runtime_admin_callback, pattern=r"^rtadmin_"))
    application.add_handler(CallbackQueryHandler(donation_callback, pattern=r"^(?:donate_|khqr_|checkpay)"))
    application.add_handler(CallbackQueryHandler(podcast_callback, pattern=r"^podcast_"))
    application.add_handler(CallbackQueryHandler(tiktok_callback, pattern=r"^tt_"))
    application.add_handler(CallbackQueryHandler(facebook_callback, pattern=r"^fb_"))
    application.add_handler(CallbackQueryHandler(instagram_callback, pattern=r"^ig_"))
    application.add_handler(CallbackQueryHandler(youtube_callback, pattern=r"^yt_"))
    application.add_handler(CallbackQueryHandler(article_callback, pattern=r"^art_"))
    application.add_handler(CallbackQueryHandler(on_callback))

    # ── Group 0: Channel & Message Handlers ─────────────────────────────────────
    # Channel posts
    application.add_handler(
        MessageHandler(
            filters.ChatType.CHANNEL,
            on_channel_post,
        )
    )

    # User Media & Text Message Handlers
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.ChatType.CHANNEL, on_photo))
    application.add_handler(MessageHandler(filters.VOICE & ~filters.ChatType.CHANNEL, on_voice))
    application.add_handler(
        MessageHandler(
            (filters.Document.ALL | filters.AUDIO) & ~filters.VOICE & ~filters.ChatType.CHANNEL,
            on_audio_file,
        )
    )
    application.add_handler(
        MessageHandler(
            (filters.Sticker.ALL | filters.VIDEO | filters.VIDEO_NOTE) & ~filters.ChatType.CHANNEL,
            on_any_media,
        )
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.ChatType.CHANNEL, on_text)
    )

    # ── Optional: Chat Member & Inline Handlers ────────────────────────────────
    with suppress(Exception):
        on_my_chat_member = getattr(legacy, "on_my_chat_member", None)
        if callable(on_my_chat_member):
            application.add_handler(
                ChatMemberHandler(on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER)
            )

    with suppress(Exception):
        on_inline_query = getattr(legacy, "on_inline_query", None)
        if callable(on_inline_query):
            application.add_handler(InlineQueryHandler(on_inline_query))

    # ── Group 100: Telemetry Trailing Completion & Error Handling ──────────────
    application.add_handler(
        TypeHandler(Update, telegram_request_telemetry_complete_guard),
        group=100,
    )
    application.add_error_handler(error_handler)


__all__ = ["register_telegram_handlers"]