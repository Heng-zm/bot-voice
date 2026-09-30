"""Interactive Telegram UI components (Keyboards, Menus, Quick Guides)."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# ============================================================================
# MAIN MENU BUTTON CONSTANTS (11 Core Features)
# ============================================================================
BTN_COPY = "📋 Copy"
BTN_UPSCALE = "🖼️ ច្បាស់"
BTN_REMOVE_BG = "🖼️ លុបBG"

BTN_EXCHANGE = "📈 ហាងឆេង"
BTN_PDF = "📄 PDF"
BTN_FOOD = "🥘 ឆែកអាហារ"

BTN_HOMEWORK = "🤖 លំហាត់"
BTN_AI_CHAT = "💬 ឆាតAi"
BTN_HOMEWORK_EXPLAIN = "👨‍🏫 លំហាត់-ពន្យល់"

BTN_DEPOSIT = "👤 ដាក់លុយចូល Bot"
BTN_SERVICES = "⚡ សេវាកម្ម"

# Sub-menu button constants
BTN_CHAT_FEMALE = "🎙 ឆាតជាមួយស្រី"
BTN_CHAT_MALE = "🎙 ឆាតជាមួយប្រុស"
EXIT_PREFIX = "« ចេញពីមុខងារ:"

# Quick Menu Button Constants
MENU_BTN_TTS = "🎙️ បម្លែងសំឡេង (TTS)"
MENU_BTN_AI = "🤖 សួរឆ្លើយ AI"
MENU_BTN_TIKTOK = "🎬 ទាញយក TikTok"
MENU_BTN_PODCAST = "🎙️ Podcast ព័ត៌មាន"
MENU_BTN_SETTINGS = "⚙️ ការកំណត់"
MENU_BTN_HELP = "📖 របៀបប្រើ (Help)"


# ============================================================================
# INLINE KEYBOARDS (Legacy Compatibility & Welcome Panels)
# ============================================================================

def get_clean_inline_menu_kb() -> InlineKeyboardMarkup:
    """Return a clean inline menu keyboard with close/back buttons.

    Required by app.legacy get_welcome_kb().
    """
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🏠 ម៉ឺនុយដើម (Menu)", callback_data="welcome_menu"),
            InlineKeyboardButton("❌ បិទ (Close)", callback_data="welcome_back"),
        ]
    ])


def get_welcome_kb() -> InlineKeyboardMarkup:
    """Return the welcome message inline keyboard."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚙️ ការកំណត់ / Settings", callback_data="welcome_profile"),
            InlineKeyboardButton("🤖 ម៉ូដែល TTS", callback_data="show_tts_model"),
        ],
        [
            InlineKeyboardButton("☕ ឧបត្ថម្ភកាហ្វេ", callback_data="donate_menu"),
            InlineKeyboardButton("🏆 តារាងកិត្តិយស", callback_data="donate_halloffame"),
        ],
        [
            InlineKeyboardButton("📢 Channel ព័ត៌មាន", url="https://t.me/m11mmm112"),
        ],
        [
            InlineKeyboardButton("📖 របៀបប្រើ (Help)", callback_data="welcome_help"),
            InlineKeyboardButton("❌ បិទ (Close)", callback_data="welcome_back"),
        ],
    ])


def get_media_hub_kb() -> InlineKeyboardMarkup:
    """Interactive inline panel for Media & AI Vision conversion tools."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🖼️ បង្កើនគុណភាព (Upscale)", callback_data="action_upscale"),
            InlineKeyboardButton("✂️ លុបផ្ទៃខាងក្រោយ (BG)", callback_data="action_remove_bg"),
        ],
        [
            InlineKeyboardButton("📝 ទាញអត្ថបទ (OCR)", callback_data="action_ocr"),
            InlineKeyboardButton("🎙️ បម្លែងជាសំឡេង (TTS)", callback_data="action_tts"),
        ],
        [
            InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ (Back)", callback_data="welcome_menu"),
        ],
    ])


def get_services_overview_kb() -> InlineKeyboardMarkup:
    """Interactive catalog of all bot capabilities."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🤖 AI Chat & សំណួរ", callback_data="guide_ai_chat"),
            InlineKeyboardButton("🎙️ សំឡេង & TTS", callback_data="guide_tts"),
        ],
        [
            InlineKeyboardButton("📈 ហាងឆេងរូបិយប័ណ្ណ", callback_data="action_exchange"),
            InlineKeyboardButton("📄 គ្រប់គ្រង PDF", callback_data="action_pdf"),
        ],
        [
            InlineKeyboardButton("🥘 វិភាគអាហារ & កាឡូរី", callback_data="action_food"),
            InlineKeyboardButton("👨‍🏫 ដោះស្រាយលំហាត់", callback_data="action_homework"),
        ],
        [
            InlineKeyboardButton("☕ ឧបត្ថម្ភ / បញ្ចូលលុយ", callback_data="donate_menu"),
            InlineKeyboardButton("⚙️ ការកំណត់", callback_data="welcome_profile"),
        ],
        [
            InlineKeyboardButton("❌ បិទ", callback_data="welcome_back"),
        ],
    ])


# ============================================================================
# REPLY KEYBOARDS (Bottom Dynamic Menus)
# ============================================================================

def get_quick_reply_keyboard() -> ReplyKeyboardMarkup:
    """Return the dynamic quick reply menu respecting feature toggles."""
    from app.core.features import is_podcast_enabled, is_tiktok_enabled

    rows: list[list[KeyboardButton]] = [
        [KeyboardButton(MENU_BTN_TTS), KeyboardButton(MENU_BTN_AI)],
    ]

    row_media: list[KeyboardButton] = []
    if is_tiktok_enabled():
        row_media.append(KeyboardButton(MENU_BTN_TIKTOK))
    if is_podcast_enabled():
        row_media.append(KeyboardButton(MENU_BTN_PODCAST))
    if row_media:
        rows.append(row_media)

    rows.append([KeyboardButton(MENU_BTN_SETTINGS), KeyboardButton(MENU_BTN_HELP)])

    return ReplyKeyboardMarkup(
        rows,
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="សូមជ្រើសរើសមុខងារ ឬផ្ញើសារ/សំឡេង...",
    )


def get_exit_keyboard(feature_name: str) -> ReplyKeyboardMarkup:
    """Return a single-action reply keyboard to cleanly exit a specialized mode."""
    clean_name = feature_name.strip()
    return ReplyKeyboardMarkup(
        [[KeyboardButton(f"{EXIT_PREFIX} {clean_name}")]],
        resize_keyboard=True,
        is_persistent=True,
    )


def get_ai_chat_keyboard() -> ReplyKeyboardMarkup:
    """Return keyboard for AI Chat assistant mode with voice persona toggles."""
    return ReplyKeyboardMarkup([
        [KeyboardButton(BTN_CHAT_FEMALE), KeyboardButton(BTN_CHAT_MALE)],
        [KeyboardButton(f"{EXIT_PREFIX} ឆាតAi")],
    ], resize_keyboard=True, is_persistent=True)


def is_exit_button(text: str | None) -> bool:
    """Check if an incoming button text is a mode exit request."""
    if not text:
        return False
    clean = text.strip()
    return clean.startswith(EXIT_PREFIX) or clean in ("« ចេញ", "❌ ចេញ", "បោះបង់", "/cancel")


def extract_exit_feature(text: str | None) -> str:
    """Extract feature name from an exit button string."""
    if not text:
        return ""
    clean = text.strip()
    if clean.startswith(EXIT_PREFIX):
        return clean.replace(EXIT_PREFIX, "", 1).strip()
    return ""


# ============================================================================
# QUICK GUIDES & HELP PANELS
# ============================================================================

async def send_tts_quick_guide(message: Any, user_id: int | None = None) -> None:
    """Send an interactive quick guide for the Text-to-Speech engine."""
    if not message:
        return

    text = (
        "🎙️ <b>ការណែនាំមុខងារបម្លែងអត្ថបទទៅជាសំឡេង (TTS Quick Guide)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "• <b>របៀបប្រើ៖</b> ផ្ញើអត្ថបទខ្មែរ ឬអន្តរជាតិមកកាន់ Bot នោះ Bot នឹងបង្កើត Voice Note ជូនភ្លាមៗ\n"
        "• <b>ប្តូរសំឡេង៖</b> ប្រើពាក្យបញ្ជា <code>/voice</code> ដើម្បីជ្រើសរើសសំឡេងប្រុស ឬស្រី\n"
        "• <b>ល្បឿនអាន៖</b> ប្រើ <code>/speed 1.25</code> ដើម្បីសារ៉េល្បឿនលឿន ឬយឺត\n"
        "• <b>ម៉ូដែលសំឡេង៖</b> ប្រើ <code>/ttsmodel</code> (Kiri Khmer, Gemini AI, Edge TTS)\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីកំណត់លក្ខណៈសំឡេងរបស់អ្នក៖</i>"
    )

    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚙️ ការកំណត់សំឡេង", callback_data="welcome_profile"),
            InlineKeyboardButton("🤖 ម៉ូដែល TTS", callback_data="show_tts_model"),
        ],
        [
            InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ", callback_data="welcome_menu"),
        ],
    ])

    with suppress(Exception):
        await message.reply_text(text, parse_mode="HTML", reply_markup=kb)


async def send_ai_chat_quick_guide(message: Any, user_id: int | None = None) -> None:
    """Send an interactive quick guide for the AI Assistant Chat mode."""
    if not message:
        return

    text = (
        "💬 <b>ការណែនាំអំពីមុខងារសួរឆ្លើយជាមួយ AI Assistant (AI Chat)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "• <b>ការសាកសួរ៖</b> អ្នកអាចសួរសំណួរជា <b>អត្ថបទ</b> ឬផ្ញើជា <b>Voice Message</b> ដោយផ្ទាល់\n"
        "• <b>សួររហ័ស៖</b> វាយ <code>/ask [សំណួរ]</code> នៅកន្លែងណាក៏បាន\n"
        "• <b>របៀបឆ្លើយ៖</b> អ្នកអាចជ្រើសរើសឱ្យ AI ឆ្លើយជាសំឡេងស្រី ឬប្រុស\n"
        "• <b>សម្អាតប្រវត្តិ៖</b> វាយ <code>/clear</code> ដើម្បីចាប់ផ្ដើមការសន្ទនាថ្មី\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 <i>សូមជ្រើសរើសតួអង្គ AI ខាងក្រោមដើម្បីចាប់ផ្ដើម៖</i>"
    )

    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👩 សំឡេងស្រី (Female)", callback_data="set_gender:female"),
            InlineKeyboardButton("👨 សំឡេងប្រុស (Male)", callback_data="set_gender:male"),
        ],
        [
            InlineKeyboardButton("🗑️ សម្អាតបរិបទសន្ទនា", callback_data="action_clear_chat"),
            InlineKeyboardButton("🔙 ត្រឡប់", callback_data="welcome_menu"),
        ],
    ])

    with suppress(Exception):
        await message.reply_text(text, parse_mode="HTML", reply_markup=kb)


# ============================================================================
# MENU COMMAND HANDLERS
# ============================================================================

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display the primary dynamic reply keyboard menu."""
    msg = update.effective_message
    if not msg:
        return

    if update.callback_query:
        with suppress(Exception):
            await update.callback_query.answer()

    welcome_text = (
        "🌟 <b>សូមស្វាគមន៍មកកាន់ ម៉ឺនុយរហ័ស (Quick Menu)!</b>\n\n"
        "សូមជ្រើសរើសមុខងារដែលអ្នកចង់ប្រើប្រាស់តាមរយៈប៊ូតុងខាងក្រោម៖"
    )

    try:
        await msg.reply_text(
            welcome_text,
            parse_mode="HTML",
            reply_markup=get_quick_reply_keyboard(),
        )
    except Exception as exc:
        logger.error("Failed to send quick reply menu: %s", exc)


# ============================================================================
# DYNAMIC FALLBACK RESOLVER (PEP 562)
# ============================================================================

def __getattr__(name: str) -> Any:
    """Safety fallback to prevent crashes if any module imports an unexpected legacy keyboard."""
    if name.endswith("_kb"):
        logger.warning("Dynamic fallback invoked for missing keyboard '%s' in menu.py", name)
        return get_clean_inline_menu_kb
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


get_main_menu_keyboard = get_quick_reply_keyboard
get_admin_menu_keyboard = get_clean_inline_menu_kb

__all__ = [
    # Button Labels
    "BTN_COPY",
    "BTN_UPSCALE",
    "BTN_REMOVE_BG",
    "BTN_EXCHANGE",
    "BTN_PDF",
    "BTN_FOOD",
    "BTN_HOMEWORK",
    "BTN_AI_CHAT",
    "BTN_HOMEWORK_EXPLAIN",
    "BTN_DEPOSIT",
    "BTN_SERVICES",
    "BTN_CHAT_FEMALE",
    "BTN_CHAT_MALE",
    "EXIT_PREFIX",
    "MENU_BTN_TTS",
    "MENU_BTN_AI",
    "MENU_BTN_TIKTOK",
    "MENU_BTN_PODCAST",
    "MENU_BTN_SETTINGS",
    "MENU_BTN_HELP",
    # Reply Keyboards
    "get_admin_menu_keyboard",
    "get_ai_chat_keyboard",
    "get_clean_inline_menu_kb",
    "get_exit_keyboard",
    "get_main_menu_keyboard",
    "get_media_hub_kb",
    "get_quick_reply_keyboard",
    "get_services_overview_kb",
    "get_welcome_kb",
    "is_exit_button",
    "extract_exit_feature",
    # Guides & Handlers
    "cmd_menu",
    "send_tts_quick_guide",
    "send_ai_chat_quick_guide",
]