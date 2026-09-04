"""Application settings and environment configuration."""

from __future__ import annotations

import os

try:
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
        CHANNEL_NARRATOR_SHOW_BUTTONS: bool = False
        ALLOWED_CHANNEL_IDS: str = ""

except (ImportError, ModuleNotFoundError):
    class AppSettings:  # type: ignore[no-redef]
        """Fallback runtime configuration loaded directly from os.environ."""

        def __init__(self) -> None:
            self.TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            self.ADMIN_IDS: str = os.environ.get("ADMIN_IDS", "")
            self.GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
            self.GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
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


SETTINGS = AppSettings()

__all__ = [
    "AppSettings",
    "SETTINGS",
]
