"""AI services package."""

from __future__ import annotations

from app.services.ai.gemini import (
    GEMINI_MODEL_DEFAULT,
    detect_image_mime,
    detect_image_mime_from_bytes,
    is_retryable_gemini_error,
)
from app.services.ai.language import (
    _detect_lang,
    _detect_latin_tts_language,
    _language_display,
    _looks_like_japanese_han_phrase,
)
from app.services.ai.providers import (
    NoProviderAvailable,
    ProviderBusy,
    ProviderManager,
    ProviderPolicy,
    ProviderState,
    ProviderTimeout,
    configure_default_providers,
    get_provider_manager,
)

__all__ = [
    "GEMINI_MODEL_DEFAULT",
    "NoProviderAvailable",
    "ProviderBusy",
    "ProviderManager",
    "ProviderPolicy",
    "ProviderState",
    "ProviderTimeout",
    "_detect_lang",
    "_detect_latin_tts_language",
    "_language_display",
    "_looks_like_japanese_han_phrase",
    "configure_default_providers",
    "detect_image_mime",
    "detect_image_mime_from_bytes",
    "get_provider_manager",
    "is_retryable_gemini_error",
]
