"""Application settings and environment configuration."""

from __future__ import annotations

import os

try:
    from pydantic import field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class AppSettings(BaseSettings):
        """Core runtime configuration loaded from environment and .env."""

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        @field_validator("GEMINI_MODEL", mode="before")
        @classmethod
        def _normalize_gemini_model(cls, v: Any) -> str:
            from app.services.ai.gemini import normalize_gemini_model

            return normalize_gemini_model(str(v or "gemini-2.5-flash"))

        TELEGRAM_BOT_TOKEN: str = ""
        ADMIN_IDS: str = ""
        GEMINI_API_KEY: str = ""
        GEMINI_MODEL: str = "gemini-2.5-flash"
        HF_TOKEN: str = ""
        SUPABASE_URL: str = ""
        SUPABASE_KEY: str = ""
        SUPABASE_SERVICE_ROLE_KEY: str = ""
        REDIS_URL: str = ""
        UPSTASH_VECTOR_REST_URL: str = ""
        UPSTASH_VECTOR_REST_TOKEN: str = ""
        PORT: int = 8080
        TELEGRAM_ALLOWED_UPDATES: str = "message,edited_message,callback_query,channel_post"
        TELEGRAM_CONCURRENT_UPDATES: int = 32
        TELEGRAM_CONNECTION_POOL_SIZE: int = 64
        DISPATCHER_MAX_CONCURRENCY: int = 64
        DISPATCHER_MAX_QUEUE_DEPTH: int = 500
        MAX_CONCURRENT_TTS_USERS: int = 16
        DB_EXECUTOR_MAX_WORKERS: int = 12
        BROADCAST_BATCH_SIZE: int = 25
        WEBHOOK_URL: str = ""
        ANAJAK_URL: str = ""
        ANAJAK_PUBLIC_URL: str = ""

        CHANNEL_NARRATOR_ENABLED: bool = True
        CHANNEL_NARRATOR_GENDER: str = "female"
        CHANNEL_NARRATOR_SPEED: float = 1.0
        CHANNEL_NARRATOR_MODEL: str = "auto"
        CHANNEL_NARRATOR_MAX_CHARS: int = 2000
        CHANNEL_NARRATOR_SHOW_BUTTONS: bool = False
        ALLOWED_CHANNEL_IDS: str = ""

        # Feature Toggles (set to False in .env to disable)
        ENABLE_TIKTOK: bool = True
        ENABLE_FACEBOOK: bool = True
        ENABLE_INSTAGRAM: bool = True
        ENABLE_YOUTUBE: bool = True
        ENABLE_PODCAST: bool = True
        ENABLE_DONATION: bool = True
        ENABLE_CHANNEL_NARRATOR: bool = True
        ENABLE_ARTICLE_READER: bool = True
        ENABLE_OCR: bool = True
        ENABLE_AUDIO_TRANSCRIPTION: bool = True
        ENABLE_AI_CHAT: bool = True
        ENABLE_TTS: bool = True

except (ImportError, ModuleNotFoundError):
    class AppSettings:  # type: ignore[no-redef]
        """Fallback runtime configuration loaded directly from os.environ."""

        def __init__(self) -> None:
            self.TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            self.ADMIN_IDS: str = os.environ.get("ADMIN_IDS", "")
            from app.services.ai.gemini import normalize_gemini_model

            self.GEMINI_MODEL: str = normalize_gemini_model(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
            self.HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
            self.SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
            self.SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
            self.SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
            self.REDIS_URL: str = os.environ.get("REDIS_URL", "")
            self.UPSTASH_VECTOR_REST_URL: str = os.environ.get("UPSTASH_VECTOR_REST_URL", "")
            self.UPSTASH_VECTOR_REST_TOKEN: str = os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "")
            self.PORT: int = int(os.environ.get("PORT", "8080") or 8080)
            self.TELEGRAM_ALLOWED_UPDATES: str = os.environ.get(
                "TELEGRAM_ALLOWED_UPDATES", "message,edited_message,callback_query,channel_post"
            )
            self.TELEGRAM_CONCURRENT_UPDATES: int = int(os.environ.get("TELEGRAM_CONCURRENT_UPDATES", "32") or 32)
            self.TELEGRAM_CONNECTION_POOL_SIZE: int = int(os.environ.get("TELEGRAM_CONNECTION_POOL_SIZE", "64") or 64)
            self.DISPATCHER_MAX_CONCURRENCY: int = int(os.environ.get("DISPATCHER_MAX_CONCURRENCY", "64") or 64)
            self.DISPATCHER_MAX_QUEUE_DEPTH: int = int(os.environ.get("DISPATCHER_MAX_QUEUE_DEPTH", "500") or 500)
            self.MAX_CONCURRENT_TTS_USERS: int = int(os.environ.get("MAX_CONCURRENT_TTS_USERS", "16") or 16)
            self.DB_EXECUTOR_MAX_WORKERS: int = int(os.environ.get("DB_EXECUTOR_MAX_WORKERS", "12") or 12)
            self.BROADCAST_BATCH_SIZE: int = int(os.environ.get("BROADCAST_BATCH_SIZE", "25") or 25)
            self.WEBHOOK_URL: str = os.environ.get("WEBHOOK_URL", "")
            self.ANAJAK_URL: str = os.environ.get("ANAJAK_URL", "")
            self.ANAJAK_PUBLIC_URL: str = os.environ.get("ANAJAK_PUBLIC_URL", "")
            self.CHANNEL_NARRATOR_ENABLED: bool = os.environ.get(
                "CHANNEL_NARRATOR_ENABLED", "true"
            ).lower() in ("1", "true", "yes")
            self.CHANNEL_NARRATOR_GENDER: str = os.environ.get("CHANNEL_NARRATOR_GENDER", "female")
            self.CHANNEL_NARRATOR_SPEED: float = float(os.environ.get("CHANNEL_NARRATOR_SPEED", "1.0") or 1.0)
            self.CHANNEL_NARRATOR_MODEL: str = os.environ.get("CHANNEL_NARRATOR_MODEL", "auto")
            self.CHANNEL_NARRATOR_MAX_CHARS: int = int(os.environ.get("CHANNEL_NARRATOR_MAX_CHARS", "2000") or 2000)
            self.CHANNEL_NARRATOR_SHOW_BUTTONS: bool = os.environ.get(
                "CHANNEL_NARRATOR_SHOW_BUTTONS", "false"
            ).lower() in ("1", "true", "yes")
            self.ALLOWED_CHANNEL_IDS: str = os.environ.get("ALLOWED_CHANNEL_IDS", "")

            def _b(k: str, default_val: str = "true") -> bool:
                return os.environ.get(k, default_val).strip().lower() in ("1", "true", "yes", "on")

            self.ENABLE_TIKTOK: bool = _b("ENABLE_TIKTOK", os.environ.get("TIKTOK_ENABLED", "true"))
            self.ENABLE_FACEBOOK: bool = _b("ENABLE_FACEBOOK", os.environ.get("FACEBOOK_ENABLED", "true"))
            self.ENABLE_INSTAGRAM: bool = _b("ENABLE_INSTAGRAM", os.environ.get("INSTAGRAM_ENABLED", "true"))
            self.ENABLE_YOUTUBE: bool = _b("ENABLE_YOUTUBE", os.environ.get("YOUTUBE_ENABLED", "true"))
            self.ENABLE_PODCAST: bool = _b("ENABLE_PODCAST", os.environ.get("PODCAST_ENABLED", "true"))
            self.ENABLE_DONATION: bool = _b("ENABLE_DONATION", os.environ.get("DONATION_ENABLED", "true"))
            self.ENABLE_CHANNEL_NARRATOR: bool = _b("ENABLE_CHANNEL_NARRATOR", os.environ.get("CHANNEL_NARRATOR_ENABLED", "true"))
            self.ENABLE_ARTICLE_READER: bool = _b("ENABLE_ARTICLE_READER", os.environ.get("ARTICLE_READER_ENABLED", "true"))
            self.ENABLE_OCR: bool = _b("ENABLE_OCR", os.environ.get("OCR_ENABLED", "true"))
            self.ENABLE_AUDIO_TRANSCRIPTION: bool = _b("ENABLE_AUDIO_TRANSCRIPTION", os.environ.get("AUDIO_TRANSCRIPTION_ENABLED", "true"))
            self.ENABLE_AI_CHAT: bool = _b("ENABLE_AI_CHAT", os.environ.get("AI_CHAT_ENABLED", "true"))
            self.ENABLE_TTS: bool = _b("ENABLE_TTS", os.environ.get("TTS_ENABLED", "true"))


SETTINGS = AppSettings()

_DETECTED_WEBHOOK_URL: str = ""


def get_detected_webhook_url() -> str:
    """Retrieve auto-detected webhook URL from platform env variables."""
    global _DETECTED_WEBHOOK_URL
    if _DETECTED_WEBHOOK_URL:
        return _DETECTED_WEBHOOK_URL
    for env_var in (
        "WEBHOOK_URL",
        "ANAJAK_URL",
        "ANAJAK_PUBLIC_URL",
        "ANAJAK_HOST",
        "ANAJAK_DOMAIN",
        "ANAJAK_EXTERNAL_URL",
        "PUBLIC_URL",
        "APP_URL",
        "RENDER_EXTERNAL_URL",
        "RAILWAY_STATIC_URL",
        "RAILWAY_PUBLIC_DOMAIN",
        "KOYEB_PUBLIC_DOMAIN",
        "VERCEL_URL",
    ):
        val = (os.environ.get(env_var) or "").strip().rstrip("/")
        if val:
            if not val.startswith("http://") and not val.startswith("https://"):
                val = f"https://{val}"
            _DETECTED_WEBHOOK_URL = val
            return val
    return ""


from app.core.features import (
    FEATURE_AI_CHAT,
    FEATURE_ARTICLE_READER,
    FEATURE_AUDIO_TRANSCRIPTION,
    FEATURE_CHANNEL_NARRATOR,
    FEATURE_DONATION,
    FEATURE_OCR,
    FEATURE_PODCAST,
    FEATURE_TIKTOK,
    FEATURE_TTS,
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
    "get_detected_webhook_url",
    "FEATURE_AI_CHAT",
    "FEATURE_ARTICLE_READER",
    "FEATURE_AUDIO_TRANSCRIPTION",
    "FEATURE_CHANNEL_NARRATOR",
    "FEATURE_DONATION",
    "FEATURE_OCR",
    "FEATURE_PODCAST",
    "FEATURE_TIKTOK",
    "FEATURE_TTS",
    "get_feature_flags",
    "is_ai_chat_enabled",
    "is_article_reader_enabled",
    "is_audio_transcription_enabled",
    "is_channel_narrator_enabled",
    "is_donation_enabled",
    "is_feature_enabled",
    "is_ocr_enabled",
    "is_podcast_enabled",
    "is_tiktok_enabled",
    "is_tts_enabled",
    "reset_feature_overrides",
    "set_feature_override",
]
