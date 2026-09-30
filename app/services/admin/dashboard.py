"""Admin Bot Controller Dashboard and Keyboards (Full Option).

Provides mobile-friendly Telegram inline keyboards and rich telemetry summaries
for administrative management: AI stack, morning podcast, Bakong KHQR,
bot modes, user CRM, database operations, and system optimization.
"""

from __future__ import annotations

import html
import logging
import os
import time
from contextlib import suppress
from datetime import datetime
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)

# Re-usable navigation buttons
BTN_BACK_ADMIN = InlineKeyboardButton("⬅️ Admin Home", callback_data="admin_home")
BTN_CLOSE_ADMIN = InlineKeyboardButton("❌ បិទ", callback_data="admin_close")


def get_admin_dashboard_full_kb() -> InlineKeyboardMarkup:
    """Return the comprehensive, full-option Admin Control Center keyboard.

    Maintains 100% backward compatibility with all existing test suites while
    providing dedicated 1-tap entries for new core services.
    """
    rows = [
        [
            InlineKeyboardButton("🤖 Bot Config & AI", callback_data="admin_bot_config"),
            InlineKeyboardButton("🎯 Bot Mode", callback_data="admin_bot_mode"),
        ],
        [
            InlineKeyboardButton("📻 Morning Podcast", callback_data="admin_podcast"),
            InlineKeyboardButton("💳 Bakong & KHQR", callback_data="admin_bakong"),
        ],
        [
            InlineKeyboardButton("📢 Broadcast", callback_data="admin_broadcast"),
            InlineKeyboardButton("🗓 Schedules", callback_data="admin_schedules"),
        ],
        [
            InlineKeyboardButton("👥 Users & CRM", callback_data="admin_users"),
            InlineKeyboardButton("🧠 User Needs", callback_data="admin_user_needs"),
        ],
        [
            InlineKeyboardButton("🩺 Health", callback_data="admin_health"),
            InlineKeyboardButton("🚨 Error Inbox", callback_data="admin_errors"),
        ],
        [
            InlineKeyboardButton("🗄️ Database", callback_data="admin_db"),
            InlineKeyboardButton("⚡ Optimize", callback_data="admin_optimize"),
        ],
        [
            InlineKeyboardButton("🗞️ Web News Scanner", callback_data="admin_web_news"),
        ],
        [
            InlineKeyboardButton("🎨 UI & Buttons", callback_data="admin_ui_hub"),
            InlineKeyboardButton("📄 PDF Report", callback_data="admin_report"),
        ],
        [
            InlineKeyboardButton("📜 Live Logs", callback_data="admin_live_logs"),
            InlineKeyboardButton("📈 Stats", callback_data="admin_stats"),
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="admin_home"),
            InlineKeyboardButton("❌ បិទ (Close)", callback_data="admin_close"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


def get_admin_podcast_kb() -> InlineKeyboardMarkup:
    """Keyboard for Daily Morning Podcast administrative control."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🚀 ផ្សាយទៅ Subscribers ឥឡូវ", callback_data="admin_podcast_broadcast"),
        ],
        [
            InlineKeyboardButton("🧪 ផ្ញើតេស្តមកខ្ញុំ (Test Me)", callback_data="admin_podcast_test"),
            InlineKeyboardButton("🔄 ទាញយកព័ត៌មានថ្មី (Force Cache)", callback_data="admin_podcast_refresh"),
        ],
        [
            InlineKeyboardButton("🎙️ សំឡេងស្រី", callback_data="admin_podcast_voice_f"),
            InlineKeyboardButton("🎙️ សំឡេងប្រុស", callback_data="admin_podcast_voice_m"),
            InlineKeyboardButton("🎵 MP3 Track", callback_data="admin_podcast_mp3"),
        ],
        [
            InlineKeyboardButton("👥 បញ្ជី Subscribers", callback_data="admin_podcast_subs"),
            InlineKeyboardButton("🔄 ពិនិត្យឡើងវិញ", callback_data="admin_podcast"),
        ],
        [
            BTN_BACK_ADMIN,
            BTN_CLOSE_ADMIN,
        ],
    ])


def get_admin_bakong_kb() -> InlineKeyboardMarkup:
    """Keyboard for Bakong KHQR Gateway and Payment administration."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 តេស្ត Bakong Gateway Ping", callback_data="admin_bakong_ping"),
        ],
        [
            InlineKeyboardButton("🖼 មើល QR Code Preview", callback_data="admin_bakong_preview_qr"),
            InlineKeyboardButton("⏳ សំណើកំពុងរង់ចាំ", callback_data="admin_bakong_pending"),
        ],
        [
            InlineKeyboardButton("🏆 តារាងអ្នកឧបត្ថម្ភ (Donors)", callback_data="admin_bakong_donors"),
            InlineKeyboardButton("⚙️ ព័ត៌មាន Merchant", callback_data="admin_bakong_config"),
        ],
        [
            BTN_BACK_ADMIN,
            BTN_CLOSE_ADMIN,
        ],
    ])


def get_admin_bot_mode_kb(current_mode: str = "auto") -> InlineKeyboardMarkup:
    """Keyboard for switching default Bot Interaction Mode."""
    curr = (current_mode or "auto").lower()

    def _mark(mode_val: str, label: str) -> str:
        return f"✅ {label}" if curr == mode_val else label

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(_mark("auto", "⚡ ស្វ័យប្រវត្តិ (Auto-Detect)"), callback_data="admin_mode_set:auto"),
        ],
        [
            InlineKeyboardButton(_mark("tts", "🎙️ អានសំឡេងផ្ទាល់ (TTS Only)"), callback_data="admin_mode_set:tts"),
        ],
        [
            InlineKeyboardButton(_mark("ai_chat", "🤖 សួរឆ្លើយ AI (AI Chat Only)"), callback_data="admin_mode_set:ai_chat"),
        ],
        [
            BTN_BACK_ADMIN,
            BTN_CLOSE_ADMIN,
        ],
    ])


def get_admin_ui_hub_kb() -> InlineKeyboardMarkup:
    """Keyboard for UI, Welcome Message, and Button Customization Hub."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👋 សារស្វាគមន៍ (Welcome Text)", callback_data="admin_welcome"),
            InlineKeyboardButton("🖼 រូបភាពស្វាគមន៍ (Photo)", callback_data="admin_welcome_set_photo"),
        ],
        [
            InlineKeyboardButton("🎛️ កែប្រែអក្សរប៊ូតុង (Button Editor)", callback_data="admin_btn_editor"),
            InlineKeyboardButton("🎙️ ម៉ូដែល TTS លំនាំដើម", callback_data="admin_tts_models"),
        ],
        [
            InlineKeyboardButton("👁️ មើលសារស្វាគមន៍ (Preview)", callback_data="admin_welcome_preview"),
            InlineKeyboardButton("🗑️ លុបរូបភាពស្វាគមន៍", callback_data="admin_welcome_remove_photo"),
        ],
        [
            BTN_BACK_ADMIN,
            BTN_CLOSE_ADMIN,
        ],
    ])


def get_admin_quick_actions_kb() -> InlineKeyboardMarkup:
    """Keyboard for One-Tap Administrative Quick Operations."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚡ Optimize System (GC + DB)", callback_data="admin_quick_optimize"),
            InlineKeyboardButton("🧹 សម្អាត Audio Cache", callback_data="admin_quick_flush_cache"),
        ],
        [
            InlineKeyboardButton("🚨 សម្អាត Error Inbox", callback_data="admin_quick_clear_errors"),
            InlineKeyboardButton("🔄 Refresh Status", callback_data="admin_quick_actions"),
        ],
        [
            BTN_BACK_ADMIN,
            BTN_CLOSE_ADMIN,
        ],
    ])


async def build_admin_home_full_text(admin_id: int, title: str = "👑 <b>ផ្ទាំងគ្រប់គ្រងកំពូល / Admin Control Center</b>") -> str:
    """Build the comprehensive, full-option status summary text for Admin Home."""
    from app import legacy

    counts = await legacy._admin_summary_counts(admin_id)
    settings, settings_status = await legacy.get_bot_settings_async()
    mode = legacy._run_state_bot_mode() if hasattr(legacy, "_run_state_bot_mode") else str(globals().get("BOT_MODE", "POLLING"))
    
    telegram_runtime = legacy._telegram_runtime_status_snapshot()
    telegram_status = str(telegram_runtime.get("status") or "online").lower()
    telegram_status_labels = {
        "loading": "⏳ Loading",
        "online": "✅ Online",
        "standby": "🟡 Standby",
        "stopping": "🛑 Stopping",
        "error": "❌ Error",
        "offline": "⚫ Offline",
    }
    telegram_status_label = telegram_status_labels.get(telegram_status, f"⚠️ {telegram_status.title()}")

    maintenance = legacy._setting_bool_from(settings, "maintenance_mode", False)
    tts_on = legacy._setting_bool_from(settings, "tts_enabled", True)
    ocr_on = legacy._setting_bool_from(settings, "ocr_enabled", True)
    channel_on = legacy._setting_bool_from(settings, "channel_narrator_enabled", True)

    error_count = 0
    with suppress(Exception):
        error_count = int(legacy._admin_error_center_total_count())

    optimize_score = "?"
    with suppress(Exception):
        optimize_score = str(legacy._optimization_score(legacy._runtime_performance_snapshot(light=True))[0])

    alerts = await legacy._admin_smart_alert_lines(counts)
    alert_text = "\n".join(f"• {line}" for line in alerts) if alerts else "• ✅ ប្រព័ន្ធស្នូលដំណើរការល្អទាំងអស់"

    redis_active = bool(os.environ.get("REDIS_URL") or getattr(legacy.SETTINGS, "REDIS_URL", ""))
    from app.services.tts.voices import get_default_tts_model, tts_model_label
    default_tts = tts_model_label(get_default_tts_model())
    active_gemini = legacy._setting_raw_from(settings, "GEMINI_MODEL", getattr(legacy.SETTINGS, "GEMINI_MODEL", "gemini-2.5-flash"))
    db_ok = bool(legacy.supabase and settings_status.get("db_ok"))

    # Audio Cache Metrics
    tts_cache_line = ""
    with suppress(Exception):
        from app.services.tts.cache import get_tts_cache_summary
        csum = get_tts_cache_summary()
        fid_cnt = csum.get("file_id", {}).get("items", 0)
        aud_mb = csum.get("audio", {}).get("current_mb", 0.0)
        fid_hr = csum.get("file_id", {}).get("hit_rate_pct", 0.0)
        tts_cache_line = f"\n• Audio Cache: <b>{fid_cnt:,}</b> CDN IDs · <b>{aud_mb:.1f} MB</b> · Hit: <b>{fid_hr}%</b>"

    # Podcast Status Telemetry
    podcast_info = "0 Subscribers"
    with suppress(Exception):
        from app.services.podcast.store import podcast_store
        sub_cnt = podcast_store.count()
        last_date = podcast_store.get_last_broadcast_date() or "មិនទាន់មាន"
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_status = "ផ្សាយរួច ✅" if last_date == today_str else "រង់ចាំ ⏳ (7:00 AM)"
        podcast_info = f"<b>{sub_cnt}</b> នាក់ · ថ្ងៃនេះ: <b>{today_status}</b>"

    # Bakong KHQR Telemetry
    bakong_info = "Bakong Open API"
    with suppress(Exception):
        from app.services.donation.khqr import get_khqr_config
        cfg = get_khqr_config()
        m_name = cfg.get("merchant_name") or "Bot Support"
        bakong_info = f"<code>{html.escape(m_name)}</code> (Static fallback ready ✅)"

    # Default Bot Interaction Mode
    bot_mode_val = legacy._setting_raw_from(settings, "DEFAULT_BOT_MODE", "auto").lower()
    mode_display = {
        "auto": "⚡ Auto-Detect",
        "tts": "🎙️ TTS Only",
        "ai_chat": "🤖 AI Chat Only",
    }.get(bot_mode_val, "⚡ Auto-Detect")

    return (
        f"{title}\n"
        f"<code>/admin › Full Option Control Hub</code>\n\n"
        f"⚡ <b>ប្រព័ន្ធដំណើរការ (System):</b> {telegram_status_label} · <b>Mode:</b> <code>{html.escape(str(mode))}</code>\n"
        f"⏱ <b>Uptime:</b> <code>{html.escape(legacy._format_uptime())}</code> · <b>Health:</b> <code>{html.escape(optimize_score)}/100</code> · <b>Errors:</b> <code>{error_count}</code>\n\n"
        f"🤖 <b>AI & Speech Stack</b>\n"
        f"• Default Bot Mode: <b>{mode_display}</b>\n"
        f"• Gemini Model: <code>{html.escape(active_gemini)}</code>\n"
        f"• Default TTS: <b>{html.escape(default_tts)}</b> · Voice: <b>{'ON ✅' if tts_on else 'OFF ⚠️'}</b>\n"
        f"• Channel Narrator: <b>{'ON ✅' if channel_on else 'OFF ⚠️'}</b> · OCR: <b>{'ON ✅' if ocr_on else 'OFF ⚠️'}</b>{tts_cache_line}\n"
        f"• Storage: Supabase <b>{legacy._ok_bad(db_ok, 'OK', 'WARN')}</b> · Redis <b>{legacy._ok_bad(redis_active, 'OK', 'LOCAL')}</b>\n\n"
        f"📻 <b>Morning Podcast (ព័ត៌មានពេលព្រឹក)</b>\n"
        f"• {podcast_info}\n\n"
        f"💳 <b>Bakong KHQR Gateway</b>\n"
        f"• {bakong_info}\n\n"
        f"📊 <b>Audience & Activity</b>\n"
        f"• Users: <b>{int(counts.get('total_users') or 0):,}</b> · Broadcasts: <b>{int(counts.get('total_broadcasts') or 0):,}</b> · Schedules: <b>{int(counts.get('pending_sched') or 0):,}</b>\n"
        f"• Maintenance Mode: <b>{'ACTIVE ⚠️' if maintenance else 'NORMAL ✅'}</b>\n\n"
        f"🔔 <b>System Alerts:</b>\n"
        f"{alert_text}\n\n"
        "💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីគ្រប់គ្រងមុខងារទាំងអស់ (Full Option)៖</i>"
    )


def build_admin_podcast_text() -> str:
    """Build overview text for the Morning Podcast Administrative Panel."""
    from app.services.podcast.store import podcast_store
    from app.services.podcast.handlers import (
        get_cached_podcast_voice_file_id,
        get_cached_podcast_mp3_file_id,
        get_source_banner_bytes,
    )

    subscribers = podcast_store.get_all_subscribers()
    last_date = podcast_store.get_last_broadcast_date() or "មិនទាន់មានទិន្នន័យ"
    today_str = datetime.now().strftime("%Y-%m-%d")
    is_sent_today = (last_date == today_str)

    today_status = "✅ បានផ្សាយរួចរាល់សម្រាប់ថ្ងៃនេះ" if is_sent_today else "⏳ មិនទាន់ផ្សាយ (កំណត់ម៉ោង 07:00 AM UTC+7)"

    vf_cached = "✅ រក្សាទុក (Cached)" if get_cached_podcast_voice_file_id("female") else "⏳ មិនទាន់មាន"
    vm_cached = "✅ រក្សាទុក (Cached)" if get_cached_podcast_voice_file_id("male") else "⏳ មិនទាន់មាន"
    mp3_cached = "✅ រក្សាទុក (Cached)" if get_cached_podcast_mp3_file_id("female") else "⏳ មិនទាន់មាន"
    cover_cached = "✅ រួចរាល់ (Cover Banner)" if get_source_banner_bytes() else "🖼️ Dynamic Generated"

    return (
        "📻 <b>ផ្ទាំងគ្រប់គ្រង Daily Morning Podcast (Admin Hub)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👥 <b>អ្នកជាវសរុប (Subscribers):</b> <b>{len(subscribers)}</b> នាក់\n"
        f"📅 <b>កាលបរិច្ឆេទផ្សាយចុងក្រោយ:</b> <code>{html.escape(last_date)}</code>\n"
        f"📊 <b>ស្ថានភាពថ្ងៃនេះ ({today_str}):</b> {today_status}\n"
        "⏰ <b>កាលវិភាគស្វ័យប្រវត្ត:</b> ម៉ោង <b>07:00 ព្រឹក</b> (ម៉ោងនៅកម្ពុជា)\n"
        "🎙️ <b>បច្ចេកវិទ្យា:</b> Gemini AI News Briefing + Edge TTS High-Quality\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚡ <b>ស្ថានភាព Cache & Media File IDs:</b>\n"
        f"• 🎙️ សំឡេងស្រី (Female Voice)៖ {vf_cached}\n"
        f"• 🎙️ សំឡេងប្រុស (Male Voice)៖ {vm_cached}\n"
        f"• 🎵 MP3 Audio File៖ {mp3_cached}\n"
        f"• 🖼️ រូបភាពផ្ទាំងព័ត៌មាន (Banner)៖ {cover_cached}\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <b>ជម្រើសបញ្ជា (Control Options):</b>\n"
        "• <b>ផ្សាយទៅ Subscribers ឥឡូវ</b> ៖ បញ្ជូនព័ត៌មានទៅកាន់អ្នកជាវទាំងអស់ភ្លាមៗ\n"
        "• <b>ផ្ញើតេស្តមកខ្ញុំ</b> ៖ ផ្ញើទាំងកាតព័ត៌មាន និងសំឡេង Voice មកកាន់ Chat នេះ\n"
        "• <b>តេស្តសំឡេង/MP3</b> ៖ ស្តាប់សំឡេងស្រី សំឡេងប្រុស ឬទាញយក MP3 ផ្ទាល់\n"
        "• <b>ទាញយកព័ត៌មានថ្មី</b> ៖ បង្ខំ Gemini ស្រង់ និងសង្ខេបព័ត៌មានថ្មី (Force Cache Refresh)"
    )


async def build_admin_bakong_text() -> str:
    """Build overview text for Bakong KHQR Gateway administrative panel."""
    from app.services.donation import bakong_api
    from app.services.donation.handlers import _PENDING_DONATIONS, _PENDING_LOCK
    from app.services.donation.khqr import (
        DEFAULT_BAKONG_ACCOUNT_ID,
        DEFAULT_BAKONG_MERCHANT_NAME,
        get_khqr_config,
        get_static_khqr_card,
    )

    cfg = get_khqr_config()
    acc_id = cfg.get("account_id") or DEFAULT_BAKONG_ACCOUNT_ID
    m_name = cfg.get("merchant_name") or DEFAULT_BAKONG_MERCHANT_NAME

    token_info = bakong_api.decode_token_payload()
    merchant_id = token_info.get("merchant_id") or "N/A"
    expires_at = token_info.get("expires_at") or "N/A"
    is_expired = token_info.get("is_expired", False)

    static_card_ok = bool(await get_static_khqr_card())

    with _PENDING_LOCK:
        pending_count = len(_PENDING_DONATIONS)

    return (
        "💳 <b>ផ្ទាំងគ្រប់គ្រង Bakong KHQR & Payments (Admin Hub)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Merchant Name:</b> <code>{html.escape(m_name)}</code>\n"
        f"🆔 <b>Bakong Account ID:</b> <code>{html.escape(acc_id)}</code>\n"
        f"🏢 <b>Developer ID:</b> <code>{html.escape(merchant_id)}</code>\n"
        f"📅 <b>Token Expires At:</b> <code>{html.escape(expires_at)}</code> "
        f"({'⚠️ ផុតកំណត់' if is_expired else '✅ សកម្ម'})\n"
        f"🖼 <b>Static Card Fallback:</b> {'✅ រួចរាល់ (asset/my_khqr.webp)' if static_card_ok else '⚠️ រកមិនឃើញ'}\n"
        f"⏳ <b>សំណើកំពុងរង់ចាំពិនិត្យ (Pending):</b> <b>{pending_count}</b> ករណី\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីតេស្ត Gateway ឬពិនិត្យមើល QR Code៖</i>"
    )


def build_admin_bot_mode_text(current_mode: str = "auto") -> str:
    """Build text explaining the active Bot Interaction Mode."""
    curr = (current_mode or "auto").lower()
    descriptions = {
        "auto": "⚡ <b>ស្វ័យប្រវត្តិ (Smart Auto-Detect)</b>\n"
                "• Bot វិភាគដោយស្វ័យប្រវត្តិ៖ បើសំណួរ/ការសួរសុខទុក្ខ ឆ្លើយឆ្លងជាសំឡេងតាម AI Chat; "
                "បើសារទូទៅ ឬអត្ថបទវែង នឹងអានជាសំឡេងផ្ទាល់ (TTS Direct)។\n"
                "• User អាច override ដោយប្រើ prefix <code>?</code> (បង្ខំ AI) ឬ <code>!</code> (បង្ខំ TTS)។",
        "tts": "🎙️ <b>អានសំឡេងផ្ទាល់ (TTS Voice Only)</b>\n"
               "• គ្រប់សារអត្ថបទទាំងអស់ត្រូវបានបម្លែងជាសំឡេង Voice Note ភ្លាមៗ (មិនឆ្លើយ AI ឡើយ)។",
        "ai_chat": "🤖 <b>សួរឆ្លើយ AI (Smart AI Voice Chat Only)</b>\n"
                   "• គ្រប់សារអត្ថបទទាំងអស់ត្រូវបានបញ្ជូនទៅកាន់ Gemini AI ដើម្បីឆ្លើយ និងអានចម្លើយជាសំឡេង។",
    }
    desc = descriptions.get(curr, descriptions["auto"])

    return (
        "🎯 <b>ការកំណត់របៀបដំណើរការរបស់ Bot (Default Bot Mode)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"របៀបបច្ចុប្បន្ន៖ <b>{curr.upper()}</b>\n\n"
        f"{desc}\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីប្តូររបៀបដំណើរការលំនាំដើមរបស់ Bot៖</i>"
    )


async def build_admin_ui_hub_text() -> str:
    """Build overview text for UI & Customization Hub."""
    from app import legacy

    settings, _ = await legacy.get_bot_settings_async()
    has_photo = bool(legacy._setting_raw_from(settings, "welcome_photo_file_id", ""))
    welcome_text_custom = bool(legacy._setting_raw_from(settings, "welcome_custom_text", ""))

    from app.services.telegram.buttons import DEFAULT_BUTTON_LABELS

    return (
        "🎨 <b>ផ្ទាំងគ្រប់គ្រង UI & ប៊ូតុង Telegram (Customization Hub)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👋 <b>Welcome Banner Photo:</b> {'✅ កំណត់រួច' if has_photo else '⚪ គ្មាន (អត្ថបទសុទ្ធ)'}\n"
        f"📝 <b>Welcome Custom Text:</b> {'✅ កំណត់រួច' if welcome_text_custom else '⚪ តម្លៃដើម (Default)'}\n"
        f"🎛️ <b>Dynamic Button Labels:</b> គាំទ្រ {len(DEFAULT_BUTTON_LABELS)} ប៊ូតុង\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <i>អ្នកអាចផ្លាស់ប្តូររូបភាព Welcome, អត្ថបទស្វាគមន៍, និងអក្សរលើប៊ូតុងទាំងអស់ដោយផ្ទាល់ពីទីនេះ។</i>"
    )


def build_admin_quick_actions_text() -> str:
    """Build overview text for One-Tap Quick Operations."""
    return (
        "⚡ <b>ផ្ទាំងបញ្ជាល្បឿន និងសម្អាតប្រព័ន្ធ (One-Tap Quick Operations)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "• <b>Optimize System:</b> រត់ Garbage Collection (GC) បង្កើនអង្គចងចាំ RAM, តេស្ត Latency របស់ Redis & Supabase\n"
        "• <b>សម្អាត Audio Cache:</b> Flush L1 Memory Cache និង CDN File IDs ដើម្បីផ្ទុកឡើងវិញថ្មីស្រឡាង\n"
        "• <b>សម្អាត Error Inbox:</b> Reset បញ្ជីកំហុសដែលចាប់បានក្នុងអង្គចងចាំ (Memory Log Clearing)\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីអនុវត្តសកម្មភាពភ្លាមៗ៖</i>"
    )


def get_admin_live_logs_kb(dashboard_url: str | None = None, only_errors: bool = False) -> InlineKeyboardMarkup:
    """Keyboard for Admin Live Logs viewer."""
    rows = []
    if dashboard_url:
        rows.append([InlineKeyboardButton("🌐 បើក Live Web Dashboard", url=dashboard_url)])

    toggle_btn = (
        InlineKeyboardButton("📋 បង្ហាញ Request ទាំងអស់", callback_data="admin_live_logs")
        if only_errors
        else InlineKeyboardButton("🚨 បង្ហាញតែ Error (Errors Only)", callback_data="admin_live_logs:errors")
    )
    rows.append([
        InlineKeyboardButton("🔄 Refresh Logs", callback_data=f"admin_live_logs{':errors' if only_errors else ''}"),
        toggle_btn,
    ])
    rows.append([
        BTN_BACK_ADMIN,
        BTN_CLOSE_ADMIN,
    ])
    return InlineKeyboardMarkup(rows)


def build_admin_live_logs_text(only_errors: bool = False) -> str:
    """Build formatted message text showing recent 10-15 real-time server requests."""
    from app.core.logging_middleware import get_request_log_store

    store = get_request_log_store()
    records = store.get_recent(limit=10, only_errors=only_errors)
    metrics = store.get_metrics()

    title = "🚨 <b>ADMIN REAL-TIME ERROR LOGS</b>" if only_errors else "📜 <b>ADMIN REAL-TIME REQUEST LOGS</b>"
    lines = [
        title,
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"⚡ <b>Active:</b> <code>{metrics['active_requests']}</code> | 📊 <b>Total:</b> <code>{metrics['total_requests']}</code>",
        f"⏱️ <b>Avg Latency:</b> <code>{metrics['average_latency_ms']}ms</code> | 🚀 <b>RPM:</b> <code>{metrics['requests_per_minute']}</code>",
        f"🚨 <b>Errors:</b> <code>{metrics['total_errors']}</code> | 🌐 <b>HTTP:</b> <code>{metrics['total_http']}</code> | 🤖 <b>TG:</b> <code>{metrics['total_telegram']}</code>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not records:
        lines.append("<i>មិនទាន់មាន Request ក្នុងប្រព័ន្ធនៅឡើយទេ។</i>")
    else:
        for r in records[:8]:
            status_code = r.get("status_code", 200)
            status_str = r.get("status", "")
            is_err = status_code >= 400 or "FAIL" in status_str or "ERROR" in status_str
            icon = "❌" if is_err else ("⏳" if status_str == "PENDING" else "✅")
            dur = f"{r.get('duration_ms', 0.0):.1f}ms" if r.get("duration_ms", 0.0) > 0 else ("⏳ pending" if status_str == "PENDING" else "-")
            cat = r.get("category", "")
            method = r.get("method", "")
            path = html.escape(str(r.get("path", ""))[:28])
            client = html.escape(str(r.get("client", ""))[:25])
            time_s = r.get("time_str", "")

            lines.append(f"{icon} <code>{time_s}</code> [<b>{cat} {method}</b>] ({dur})")
            lines.append(f"  ↳ <code>{path}</code>")
            lines.append(f"  ↳ 👤 {client}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("💡 <i>ចុច Refresh ដើម្បីទាញយកទិន្នន័យចុងក្រោយ ឬបើក Web Dashboard</i>")
    return "\n".join(lines)



def get_admin_web_news_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 ស្កេនព័ត៌មាន (Scan Now)", callback_data="admin_web_news_scan")],
        [InlineKeyboardButton("📋 ប្រភពព័ត៌មាន (View Sources)", callback_data="admin_web_news_sources")],
        [BTN_BACK_ADMIN, BTN_CLOSE_ADMIN],
    ])

async def build_admin_web_news_text() -> str:
    stats = {}
    with suppress(Exception):
        from app.services.ai.article_storage import get_article_storage_stats
        stats = await get_article_storage_stats()

    sources_active = stats.get("sources_active", 0)
    sources_total = stats.get("sources_total", 0)
    pending = stats.get("pending_articles", 0)
    sent = stats.get("sent_articles", 0)
    rejected = stats.get("rejected_articles", 0)

    lines = [
        "🗞️ <b>Web News Scanner (Admin Panel)</b>",
        "",
        "📊 <b>ទិន្នន័យរួម (Telemetry Metrics)</b>:",
        f"• ប្រភព (Sources): <b>{sources_active}</b> active / {sources_total} total",
        f"• រង់ចាំការអនុម័ត (Pending): <b>{pending}</b> articles",
        f"• ផ្សាយរួច (Sent): <b>{sent}</b> articles",
        f"• ច្រានចោល (Rejected): <b>{rejected}</b> articles",
        "",
        "សូមជ្រើសរើសជម្រើសខាងក្រោម៖"
    ]
    return "\n".join(lines)
