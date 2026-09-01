"""Customizable button labels for Telegram Bot Voice.

Enables administrators to customize all user-facing button text dynamically
via /admin without code modifications or redeployment.
"""

from __future__ import annotations

import logging
from contextlib import suppress

logger = logging.getLogger(__name__)

# Default button texts across all user flows
DEFAULT_BUTTON_LABELS: dict[str, str] = {
    "btn_female": "👩 សំឡេងស្រី",
    "btn_male": "👨 សំឡេងប្រុស",
    "btn_speed": "🎚️ ល្បឿនសំឡេង",
    "btn_tts_model": "🤖 ម៉ូដែល TTS",
    "btn_back": "🔙 ត្រឡប់",
    "btn_ai_read": "📢 AI អាន",
    "btn_delete": "🗑️ លុប",
    "btn_ocr_read": "▶️ អាន",
    "btn_audio_tts": "📢 បំលែងទៅសំឡេង TTS",
    "btn_welcome_profile": "👤 User Profile",
}

# Cache for loaded custom labels
_CUSTOM_BUTTONS_CACHE: dict[str, str] = {}


def get_button_label(key: str, default: str | None = None) -> str:
    """Get button text with dynamic Supabase override."""
    if key in _CUSTOM_BUTTONS_CACHE:
        return _CUSTOM_BUTTONS_CACHE[key]

    fallback = default if default is not None else DEFAULT_BUTTON_LABELS.get(key, key)
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        val = get_settings_store().get_text_sync(f"btn:{key}", "")
        if val:
            _CUSTOM_BUTTONS_CACHE[key] = val
            return val

    return fallback


async def get_button_label_async(key: str, default: str | None = None) -> str:
    """Async variant for getting button label."""
    if key in _CUSTOM_BUTTONS_CACHE:
        return _CUSTOM_BUTTONS_CACHE[key]

    fallback = default if default is not None else DEFAULT_BUTTON_LABELS.get(key, key)
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        val = await get_settings_store().get_text(f"btn:{key}", "")
        if val:
            _CUSTOM_BUTTONS_CACHE[key] = val
            return val

    return fallback


async def set_button_label(key: str, value: str) -> bool:
    """Save custom button text to Supabase bot_settings."""
    clean_val = str(value or "").strip()
    if not clean_val:
        return False

    try:
        from app.services.settings.store import get_settings_store

        await get_settings_store().set_text(f"btn:{key}", clean_val)
        _CUSTOM_BUTTONS_CACHE[key] = clean_val
        return True
    except Exception as exc:
        logger.warning("Failed to persist custom button %s: %s", key, exc)
        return False


async def reset_button_label(key: str) -> bool:
    """Reset single button to default text."""
    try:
        from app.services.settings.store import get_settings_store

        await get_settings_store().delete_setting(f"btn:{key}")
        _CUSTOM_BUTTONS_CACHE.pop(key, None)
        return True
    except Exception as exc:
        logger.warning("Failed to reset custom button %s: %s", key, exc)
        return False


def get_all_button_labels() -> dict[str, str]:
    """Return dictionary of all active button labels."""
    result = {}
    for key, default_text in DEFAULT_BUTTON_LABELS.items():
        result[key] = get_button_label(key, default_text)
    return result


__all__ = [
    "DEFAULT_BUTTON_LABELS",
    "get_all_button_labels",
    "get_button_label",
    "get_button_label_async",
    "reset_button_label",
    "set_button_label",
]
