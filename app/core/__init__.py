"""Core configuration, settings, security, and telemetry logging package."""

from __future__ import annotations

from app.core.config import SETTINGS, AppSettings
from app.core.features import (
    is_ai_chat_enabled,
    is_donation_enabled,
    is_podcast_enabled,
    is_tiktok_enabled,
)
from app.core.logging_middleware import (
    RequestLogStore,
    RequestRecord,
    get_request_log_store,
    request_lifecycle_logging_middleware,
)
from app.core.security import (
    get_allowed_api_keys,
    timing_safe_compare,
    validate_api_key,
)
from app.core.telegram_auth import is_telegram_admin

__all__ = [
    "AppSettings",
    "RequestLogStore",
    "RequestRecord",
    "SETTINGS",
    "get_allowed_api_keys",
    "get_request_log_store",
    "is_ai_chat_enabled",
    "is_donation_enabled",
    "is_podcast_enabled",
    "is_telegram_admin",
    "is_tiktok_enabled",
    "request_lifecycle_logging_middleware",
    "timing_safe_compare",
    "validate_api_key",
]
