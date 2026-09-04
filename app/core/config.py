"""Application settings and environment configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Core runtime configuration loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    TELEGRAM_BOT_TOKEN: str = ""
    ADMIN_IDS: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    HF_TOKEN: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    REDIS_URL: str = ""
    UPSTASH_VECTOR_REST_URL: str = ""
    UPSTASH_VECTOR_REST_TOKEN: str = ""
    PORT: int = 8080
    TELEGRAM_ALLOWED_UPDATES: str = "message,edited_message,callback_query,channel_post"

    CHANNEL_NARRATOR_ENABLED: bool = True
    CHANNEL_NARRATOR_GENDER: str = "female"
    CHANNEL_NARRATOR_SPEED: float = 1.0
    CHANNEL_NARRATOR_MODEL: str = "auto"
    CHANNEL_NARRATOR_MAX_CHARS: int = 2000
    ALLOWED_CHANNEL_IDS: str = ""


SETTINGS = AppSettings()

__all__ = [
    "AppSettings",
    "SETTINGS",
]
