"""Centralized application configuration and logging settings."""

from __future__ import annotations

from app.config.logging import (
    TRANSIENT_NETWORK_ERRORS,
    TelegramPollingNetworkFilter,
    configure_server_logging,
    install_telegram_polling_filter,
    is_transient_network_error,
)
from app.config.settings import (
    SETTINGS,
    AppSettings,
    get_detected_webhook_url,
    get_feature_flags,
    is_ai_chat_enabled,
    is_article_reader_enabled,
    is_audio_transcription_enabled,
    is_channel_narrator_enabled,
    is_donation_enabled,
    is_feature_enabled,
    is_ocr_enabled,
    is_podcast_enabled,
    is_tiktok_enabled,
    is_tts_enabled,
    reset_feature_overrides,
    set_feature_override,
)

__all__ = [
    "SETTINGS",
    "AppSettings",
    "TRANSIENT_NETWORK_ERRORS",
    "TelegramPollingNetworkFilter",
    "configure_server_logging",
    "get_detected_webhook_url",
    "get_feature_flags",
    "install_telegram_polling_filter",
    "is_ai_chat_enabled",
    "is_article_reader_enabled",
    "is_audio_transcription_enabled",
    "is_channel_narrator_enabled",
    "is_donation_enabled",
    "is_feature_enabled",
    "is_ocr_enabled",
    "is_podcast_enabled",
    "is_tiktok_enabled",
    "is_transient_network_error",
    "is_tts_enabled",
    "reset_feature_overrides",
    "set_feature_override",
]
