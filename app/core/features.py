"""Centralized Feature Toggle management for Bot Voice.

Enables fine-grained control over individual capabilities (e.g. TikTok downloader,
Morning Podcast, Bakong donations, Article narration, OCR, Audio transcription, AI chat, TTS)
via environment variables or runtime overrides.
"""

from __future__ import annotations

import os
import threading
from typing import Any

_FEATURE_LOCK = threading.RLock()
_FEATURE_OVERRIDES: dict[str, bool] = {}

# Canonical feature names
FEATURE_TIKTOK = "tiktok"
FEATURE_FACEBOOK = "facebook"
FEATURE_INSTAGRAM = "instagram"
FEATURE_YOUTUBE = "youtube"
FEATURE_PODCAST = "podcast"
FEATURE_DONATION = "donation"
FEATURE_CHANNEL_NARRATOR = "channel_narrator"
FEATURE_ARTICLE_READER = "article_reader"
FEATURE_OCR = "ocr"
FEATURE_AUDIO_TRANSCRIPTION = "audio_transcription"
FEATURE_AI_CHAT = "ai_chat"
FEATURE_TTS = "tts"

_ALL_FEATURES = (
    FEATURE_TIKTOK,
    FEATURE_FACEBOOK,
    FEATURE_INSTAGRAM,
    FEATURE_YOUTUBE,
    FEATURE_PODCAST,
    FEATURE_DONATION,
    FEATURE_CHANNEL_NARRATOR,
    FEATURE_ARTICLE_READER,
    FEATURE_OCR,
    FEATURE_AUDIO_TRANSCRIPTION,
    FEATURE_AI_CHAT,
    FEATURE_TTS,
)


def _parse_bool(val: Any, default: bool = True) -> bool:
    """Parse truthy/falsy representations into a strict boolean."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if s in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    return default


def is_feature_enabled(feature_name: str, default: bool = True) -> bool:
    """Check if a named feature is enabled.

    Priority order:
    1. In-memory runtime override (set via `set_feature_override`)
    2. Environment variable: ENABLE_<FEATURE>
    3. Environment variable: <FEATURE>_ENABLED
    4. AppSettings attribute: ENABLE_<FEATURE> or <FEATURE>_ENABLED
    5. Provided default (default: True)
    """
    canon = feature_name.strip().lower().replace("-", "_")

    with _FEATURE_LOCK:
        if canon in _FEATURE_OVERRIDES:
            return _FEATURE_OVERRIDES[canon]

    upper = canon.upper()
    env_keys = (
        f"ENABLE_{upper}",
        f"{upper}_ENABLED",
    )

    for k in env_keys:
        if k in os.environ:
            return _parse_bool(os.environ[k], default)

    # Check AppSettings
    try:
        from app.core.config import SETTINGS

        for k in env_keys:
            if hasattr(SETTINGS, k):
                return _parse_bool(getattr(SETTINGS, k), default)
    except Exception:
        pass

    return default


def set_feature_override(feature_name: str, enabled: bool | None) -> None:
    """Set or clear a runtime feature override (useful for testing or dynamic admin controls)."""
    canon = feature_name.strip().lower().replace("-", "_")
    with _FEATURE_LOCK:
        if enabled is None:
            _FEATURE_OVERRIDES.pop(canon, None)
        else:
            _FEATURE_OVERRIDES[canon] = bool(enabled)


def reset_feature_overrides() -> None:
    """Clear all runtime feature overrides."""
    with _FEATURE_LOCK:
        _FEATURE_OVERRIDES.clear()


def get_feature_flags() -> dict[str, bool]:
    """Return a dictionary of all canonical feature flags and their current states."""
    return {feat: is_feature_enabled(feat) for feat in _ALL_FEATURES}


# ── Canonical Convenience Checkers ───────────────────────────────────────────

def is_tiktok_enabled() -> bool:
    """Check if TikTok downloader is enabled (ENABLE_TIKTOK)."""
    return is_feature_enabled(FEATURE_TIKTOK, default=True)


def is_facebook_enabled() -> bool:
    """Check if Facebook downloader is enabled (ENABLE_FACEBOOK)."""
    return is_feature_enabled(FEATURE_FACEBOOK, default=True)


def is_instagram_enabled() -> bool:
    """Check if Instagram downloader is enabled (ENABLE_INSTAGRAM)."""
    return is_feature_enabled(FEATURE_INSTAGRAM, default=True)


def is_youtube_enabled() -> bool:
    """Check if YouTube downloader is enabled (ENABLE_YOUTUBE)."""
    return is_feature_enabled(FEATURE_YOUTUBE, default=True)


def is_podcast_enabled() -> bool:
    """Check if Daily Morning Podcast is enabled (ENABLE_PODCAST)."""
    return is_feature_enabled(FEATURE_PODCAST, default=True)


def is_donation_enabled() -> bool:
    """Check if Bakong KHQR Donation is enabled (ENABLE_DONATION)."""
    return is_feature_enabled(FEATURE_DONATION, default=True)


def is_channel_narrator_enabled() -> bool:
    """Check if Channel Auto-Narrator is enabled (ENABLE_CHANNEL_NARRATOR / CHANNEL_NARRATOR_ENABLED)."""
    return is_feature_enabled(FEATURE_CHANNEL_NARRATOR, default=True)


def is_article_reader_enabled() -> bool:
    """Check if Web Article Narration is enabled (ENABLE_ARTICLE_READER)."""
    return is_feature_enabled(FEATURE_ARTICLE_READER, default=True)


def is_ocr_enabled() -> bool:
    """Check if Vision OCR is enabled (ENABLE_OCR)."""
    return is_feature_enabled(FEATURE_OCR, default=True)


def is_audio_transcription_enabled() -> bool:
    """Check if Audio & Voice Transcription is enabled (ENABLE_AUDIO_TRANSCRIPTION)."""
    return is_feature_enabled(FEATURE_AUDIO_TRANSCRIPTION, default=True)


def is_ai_chat_enabled() -> bool:
    """Check if AI Chat with Gemini is enabled (ENABLE_AI_CHAT)."""
    return is_feature_enabled(FEATURE_AI_CHAT, default=True)


def is_tts_enabled() -> bool:
    """Check if core Text-to-Speech is enabled (ENABLE_TTS)."""
    return is_feature_enabled(FEATURE_TTS, default=True)


__all__ = [
    "FEATURE_TIKTOK",
    "FEATURE_FACEBOOK",
    "FEATURE_INSTAGRAM",
    "FEATURE_YOUTUBE",
    "FEATURE_PODCAST",
    "FEATURE_DONATION",
    "FEATURE_CHANNEL_NARRATOR",
    "FEATURE_ARTICLE_READER",
    "FEATURE_OCR",
    "FEATURE_AUDIO_TRANSCRIPTION",
    "FEATURE_AI_CHAT",
    "FEATURE_TTS",
    "is_feature_enabled",
    "set_feature_override",
    "reset_feature_overrides",
    "get_feature_flags",
    "is_tiktok_enabled",
    "is_facebook_enabled",
    "is_instagram_enabled",
    "is_youtube_enabled",
    "is_podcast_enabled",
    "is_donation_enabled",
    "is_channel_narrator_enabled",
    "is_article_reader_enabled",
    "is_ocr_enabled",
    "is_audio_transcription_enabled",
    "is_ai_chat_enabled",
    "is_tts_enabled",
]
