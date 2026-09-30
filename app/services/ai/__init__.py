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
from app.services.ai.ocr import (
    OCRService,
    ask_gemini_ocr_bytes,
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

try:
    from app.services.ai.vector_store import (
        UpstashVectorStore,
        get_global_vector_store,
    )
except (ImportError, ModuleNotFoundError):
    UpstashVectorStore = None  # type: ignore[assignment,misc]

    def get_global_vector_store():  # type: ignore[misc]
        return None

__all__ = [
    "GEMINI_MODEL_DEFAULT",
    "NoProviderAvailable",
    "OCRService",
    "ProviderBusy",
    "ProviderManager",
    "ProviderPolicy",
    "ProviderState",
    "ProviderTimeout",
    "UpstashVectorStore",
    "_detect_lang",
    "_detect_latin_tts_language",
    "_language_display",
    "_looks_like_japanese_han_phrase",
    "ask_gemini_ocr_bytes",
    "configure_default_providers",
    "detect_image_mime",
    "detect_image_mime_from_bytes",
    "get_global_vector_store",
    "get_provider_manager",
    "is_retryable_gemini_error",
]
