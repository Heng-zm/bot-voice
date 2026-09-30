"""Customizable button labels for Telegram Bot Voice.

Enables administrators to customize all user-facing button text dynamically
via /admin without code modifications or redeployment.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress
from typing import Any

from telegram import InlineKeyboardButton, KeyboardButton

logger = logging.getLogger(__name__)

# ============================================================================
# DEFAULT BUTTON LABELS (Bilingual / Khmer with native emojis)
# ============================================================================
DEFAULT_BUTTON_LABELS: dict[str, str] = {
    # Main Dynamic Menu (11 Core Features)
    "btn_copy": "📋 Copy",
    "btn_upscale": "🖼️ ច្បាស់",
    "btn_remove_bg": "🖼️ លុបBG",
    "btn_exchange": "📈 ហាងឆេង",
    "btn_pdf": "📄 PDF",
    "btn_food": "🥘 ឆែកអាហារ",
    "btn_homework": "🤖 លំហាត់",
    "btn_ai_chat": "💬 ឆាតAi",
    "btn_homework_explain": "👨‍🏫 លំហាត់-ពន្យល់",
    "btn_deposit": "👤 ដាក់លុយចូល Bot",
    "btn_services": "⚡ សេវាកម្ម",
    # Sub-Menu Toggles
    "btn_chat_female": "🎙 ឆាតជាមួយស្រី",
    "btn_chat_male": "🎙 ឆាតជាមួយប្រុស",
    # Voice & Playback Controls
    "btn_female": "👩 សំឡេងស្រី",
    "btn_male": "👨 សំឡេងប្រុស",
    "btn_speed": "🎚️ ល្បឿនសំឡេង",
    "btn_tts_model": "🤖 ម៉ូដែល TTS",
    "btn_audio_tts": "📢 បំលែងជាសំឡេង",
    "btn_ocr_read": "▶️ អានអត្ថបទ",
    "btn_ai_read": "📢 AI អាន",
    "btn_podcast": "📻 ព័ត៌មានពេលព្រឹក",
    # Navigation & Actions
    "btn_back": "🔙 ត្រឡប់",
    "btn_delete": "🗑️ លុប",
    "btn_cancel": "🚫 បោះបង់",
    "btn_confirm": "✅ យល់ព្រម",
    "btn_close": "✖️ បិទ",
    "btn_refresh": "🔄 ផ្ទុកឡើងវិញ",
    # Information & Settings
    "btn_welcome_profile": "⚙️ ការកំណត់",
    "btn_settings": "⚙️ ការកំណត់",
    "btn_help": "📖 របៀបប្រើ",
    "btn_donate": "☕ ឧបត្ថម្ភកាហ្វេ",
    "btn_halloffame": "🏆 តារាងកិត្តិយស",
    "btn_channel": "📢 Channel",
    "btn_ask_ai": "💬 សួរ AI",
    "btn_translate": "🌐 បកប្រែ",
    "btn_summary": "📝 សង្ខេប",
    "btn_narrate": "🎙️ អានគេហទំព័រ",
    "btn_tiktok": "📥 ទាញយក TikTok",
    "btn_clear": "🗑️ សម្អាតសារ",
    "btn_unlock": "🔓 ដោះសោរ",
    "btn_security": "🛡️ សុវត្ថិភាព",
    "btn_privacy": "🔒 ឯកជនភាព",
    "btn_feedback": "💌 មតិកែលម្អ",
}

# Aliases for direct imports from menu compatibility
BTN_COPY = DEFAULT_BUTTON_LABELS["btn_copy"]
BTN_UPSCALE = DEFAULT_BUTTON_LABELS["btn_upscale"]
BTN_REMOVE_BG = DEFAULT_BUTTON_LABELS["btn_remove_bg"]
BTN_EXCHANGE = DEFAULT_BUTTON_LABELS["btn_exchange"]
BTN_PDF = DEFAULT_BUTTON_LABELS["btn_pdf"]
BTN_FOOD = DEFAULT_BUTTON_LABELS["btn_food"]
BTN_HOMEWORK = DEFAULT_BUTTON_LABELS["btn_homework"]
BTN_AI_CHAT = DEFAULT_BUTTON_LABELS["btn_ai_chat"]
BTN_HOMEWORK_EXPLAIN = DEFAULT_BUTTON_LABELS["btn_homework_explain"]
BTN_DEPOSIT = DEFAULT_BUTTON_LABELS["btn_deposit"]
BTN_SERVICES = DEFAULT_BUTTON_LABELS["btn_services"]
BTN_CHAT_FEMALE = DEFAULT_BUTTON_LABELS["btn_chat_female"]
BTN_CHAT_MALE = DEFAULT_BUTTON_LABELS["btn_chat_male"]

# Thread-safe cache with TTL for custom button labels
_CUSTOM_BUTTONS_CACHE: dict[str, tuple[str, float]] = {}
_CUSTOM_BUTTONS_LOCK = threading.RLock()
_BUTTONS_CACHE_TTL_S = 300.0  # 5 minutes


def _normalize_key(key: str) -> str:
    """Normalize button key by lowercasing and stripping redundant prefixes."""
    clean = str(key or "").strip().lower()
    if clean.startswith("btn:"):
        clean = clean[4:].strip()
    return clean


def _resolve_default(norm_key: str, default: str | None = None) -> str:
    """Retrieve default label supporting both 'btn_xyz' and 'xyz' keys."""
    if default is not None:
        return default
    if norm_key in DEFAULT_BUTTON_LABELS:
        return DEFAULT_BUTTON_LABELS[norm_key]
    with_prefix = f"btn_{norm_key}"
    if with_prefix in DEFAULT_BUTTON_LABELS:
        return DEFAULT_BUTTON_LABELS[with_prefix]
    return norm_key


def clear_button_labels_cache() -> None:
    """Clear all cached custom button labels."""
    with _CUSTOM_BUTTONS_LOCK:
        _CUSTOM_BUTTONS_CACHE.clear()


def get_button_label(key: str, default: str | None = None) -> str:
    """Get button text with in-memory caching and fallback to default."""
    norm_key = _normalize_key(key)
    now = time.monotonic()

    # 1. Fast cache hit check
    with _CUSTOM_BUTTONS_LOCK:
        entry = _CUSTOM_BUTTONS_CACHE.get(norm_key)
        if entry is not None and (now - entry[1]) < _BUTTONS_CACHE_TTL_S:
            return entry[0]

    fallback = _resolve_default(norm_key, default)

    # 2. Check settings store synchronously if available
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        if hasattr(store, "get_text_sync"):
            val = store.get_text_sync(f"btn:{norm_key}", "")
            clean_val = str(val or "").strip()
            resolved = clean_val if clean_val else fallback
            with _CUSTOM_BUTTONS_LOCK:
                _CUSTOM_BUTTONS_CACHE[norm_key] = (resolved, now)
            return resolved

    # 3. Cache the fallback to prevent repeated DB lookups on uncustomized buttons
    with _CUSTOM_BUTTONS_LOCK:
        _CUSTOM_BUTTONS_CACHE[norm_key] = (fallback, now)

    return fallback


async def get_button_label_async(key: str, default: str | None = None) -> str:
    """Async variant for retrieving button label with persistence caching."""
    norm_key = _normalize_key(key)
    now = time.monotonic()

    # 1. Fast cache hit check
    with _CUSTOM_BUTTONS_LOCK:
        entry = _CUSTOM_BUTTONS_CACHE.get(norm_key)
        if entry is not None and (now - entry[1]) < _BUTTONS_CACHE_TTL_S:
            return entry[0]

    fallback = _resolve_default(norm_key, default)

    # 2. Retrieve override from database
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        val = await store.get_text(f"btn:{norm_key}", "")
        clean_val = str(val or "").strip()
        resolved = clean_val if clean_val else fallback

        with _CUSTOM_BUTTONS_LOCK:
            _CUSTOM_BUTTONS_CACHE[norm_key] = (resolved, now)
        return resolved

    # 3. Cache fallback on error or absent setting
    with _CUSTOM_BUTTONS_LOCK:
        _CUSTOM_BUTTONS_CACHE[norm_key] = (fallback, now)

    return fallback


async def set_button_label(key: str, value: str) -> bool:
    """Save custom button text to the database and update cache."""
    norm_key = _normalize_key(key)
    clean_val = str(value or "").strip()

    if not norm_key or not clean_val:
        return False

    if len(clean_val) > 100:
        clean_val = clean_val[:100].strip()

    try:
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        await store.set_text(f"btn:{norm_key}", clean_val)

        with _CUSTOM_BUTTONS_LOCK:
            _CUSTOM_BUTTONS_CACHE[norm_key] = (clean_val, time.monotonic())

        logger.info("Custom button label set for '%s': '%s'", norm_key, clean_val)
        return True
    except Exception as exc:
        logger.warning("Failed to persist custom button %s: %s", norm_key, exc)
        return False


async def reset_button_label(key: str) -> bool:
    """Reset a single button to its factory default text."""
    norm_key = _normalize_key(key)
    if not norm_key:
        return False

    try:
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        db_key = f"btn:{norm_key}"

        if hasattr(store, "delete_setting"):
            await store.delete_setting(db_key)
        elif hasattr(store, "delete"):
            await store.delete(db_key)
        else:
            await store.set_text(db_key, "")

        default_text = _resolve_default(norm_key)
        with _CUSTOM_BUTTONS_LOCK:
            _CUSTOM_BUTTONS_CACHE[norm_key] = (default_text, time.monotonic())

        logger.info("Button label reset to default for '%s'", norm_key)
        return True
    except Exception as exc:
        logger.warning("Failed to reset custom button %s: %s", norm_key, exc)
        return False


async def reset_all_button_labels() -> int:
    """Reset all custom buttons to factory defaults in parallel."""
    count = 0
    keys = list(DEFAULT_BUTTON_LABELS.keys())
    for key in keys:
        if await reset_button_label(key):
            count += 1
    clear_button_labels_cache()
    return count


async def preload_button_labels_async() -> None:
    """Pre-warm button label cache on startup in a single batch query."""
    now = time.monotonic()
    try:
        from app.services.settings.store import get_settings_store

        store = get_settings_store()
        db_keys = [f"btn:{k}" for k in DEFAULT_BUTTON_LABELS]

        # Use batch lookup if available
        if hasattr(store, "get_many_text"):
            fetched = await store.get_many_text(db_keys, default="")
            with _CUSTOM_BUTTONS_LOCK:
                for key, default_text in DEFAULT_BUTTON_LABELS.items():
                    val = str(fetched.get(f"btn:{key}") or "").strip()
                    _CUSTOM_BUTTONS_CACHE[key] = (val if val else default_text, now)
        else:
            for key, default_text in DEFAULT_BUTTON_LABELS.items():
                val = await store.get_text(f"btn:{key}", "")
                clean = str(val or "").strip()
                with _CUSTOM_BUTTONS_LOCK:
                    _CUSTOM_BUTTONS_CACHE[key] = (clean if clean else default_text, now)

        logger.info("Pre-warmed %d button labels into memory.", len(DEFAULT_BUTTON_LABELS))
    except Exception as exc:
        logger.warning("Could not pre-warm button labels: %s", exc)


def is_button_customized(key: str) -> bool:
    """Check whether a button currently has an active custom override."""
    norm_key = _normalize_key(key)
    current_label = get_button_label(norm_key)
    default_label = _resolve_default(norm_key)
    return current_label != default_label


def get_customized_buttons() -> dict[str, str]:
    """Return dictionary of all buttons that currently have custom overrides."""
    custom: dict[str, str] = {}
    for key, default_text in DEFAULT_BUTTON_LABELS.items():
        val = get_button_label(key)
        if val != default_text:
            custom[key] = val
    return custom


async def get_customized_buttons_async() -> dict[str, str]:
    """Return dictionary of all buttons with custom overrides (asynchronous)."""
    custom: dict[str, str] = {}
    for key, default_text in DEFAULT_BUTTON_LABELS.items():
        val = await get_button_label_async(key)
        if val != default_text:
            custom[key] = val
    return custom


def get_all_button_labels() -> dict[str, str]:
    """Return dictionary of all active button labels (synchronous)."""
    result: dict[str, str] = {}
    for key, default_text in DEFAULT_BUTTON_LABELS.items():
        result[key] = get_button_label(key, default_text)
    return result


async def get_all_button_labels_async() -> dict[str, str]:
    """Return dictionary of all active button labels (asynchronous)."""
    result: dict[str, str] = {}
    for key, default_text in DEFAULT_BUTTON_LABELS.items():
        result[key] = await get_button_label_async(key, default_text)
    return result


def make_inline_button(
    key: str,
    callback_data: str | None = None,
    *,
    url: str | None = None,
    default: str | None = None,
) -> InlineKeyboardButton:
    """Create a Telegram InlineKeyboardButton with dynamic label resolution."""
    label = get_button_label(key, default)
    if url:
        return InlineKeyboardButton(label, url=url)
    return InlineKeyboardButton(label, callback_data=callback_data or key)


def make_keyboard_button(key: str, default: str | None = None) -> KeyboardButton:
    """Create a Telegram Reply KeyboardButton with dynamic label resolution."""
    return KeyboardButton(get_button_label(key, default))


__all__ = [
    # Constants
    "BTN_AI_CHAT",
    "BTN_CHAT_FEMALE",
    "BTN_CHAT_MALE",
    "BTN_COPY",
    "BTN_DEPOSIT",
    "BTN_EXCHANGE",
    "BTN_FOOD",
    "BTN_HOMEWORK",
    "BTN_HOMEWORK_EXPLAIN",
    "BTN_PDF",
    "BTN_REMOVE_BG",
    "BTN_SERVICES",
    "BTN_UPSCALE",
    "DEFAULT_BUTTON_LABELS",
    # Functions
    "clear_button_labels_cache",
    "get_all_button_labels",
    "get_all_button_labels_async",
    "get_button_label",
    "get_button_label_async",
    "get_customized_buttons",
    "get_customized_buttons_async",
    "is_button_customized",
    "make_inline_button",
    "make_keyboard_button",
    "preload_button_labels_async",
    "reset_all_button_labels",
    "reset_button_label",
    "set_button_label",
]