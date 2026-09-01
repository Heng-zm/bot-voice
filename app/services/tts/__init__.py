"""Text-to-speech services package."""

from __future__ import annotations

from app.services.tts.cache import (
    TTSAudioCache,
    TTSUserHistoryTracker,
    clear_user_tts_history,
    get_global_tts_cache,
    get_global_tts_history,
    get_last_tts,
    get_last_tts_text,
    make_tts_audio_cache_key,
    set_last_tts,
    set_last_tts_text,
)
from app.services.tts.engine import (
    TTSEngine,
    get_global_tts_engine,
)
from app.services.tts.voices import (
    DEFAULT_SPEED,
    DEFAULT_TTS_MODEL,
    SPEED_OPTIONS,
    TTS_LANGUAGE_LABELS,
    TTS_MODEL_ALIASES,
    TTS_MODEL_OPTIONS,
    TTS_SUPPORTED_LANG_ORDER,
    VOICE_MAP,
    clean_tts_text,
    get_default_tts_model,
    normalize_tts_model,
    resolve_tts_voice_candidates,
    split_text_chunks,
    tts_model_label,
)

__all__ = [
    "DEFAULT_SPEED",
    "DEFAULT_TTS_MODEL",
    "SPEED_OPTIONS",
    "TTSAudioCache",
    "TTSEngine",
    "TTSUserHistoryTracker",
    "TTS_LANGUAGE_LABELS",
    "TTS_MODEL_ALIASES",
    "TTS_MODEL_OPTIONS",
    "TTS_SUPPORTED_LANG_ORDER",
    "VOICE_MAP",
    "clean_tts_text",
    "clear_user_tts_history",
    "get_default_tts_model",
    "get_global_tts_cache",
    "get_global_tts_engine",
    "get_global_tts_history",
    "get_last_tts",
    "get_last_tts_text",
    "make_tts_audio_cache_key",
    "normalize_tts_model",
    "resolve_tts_voice_candidates",
    "set_last_tts",
    "set_last_tts_text",
    "split_text_chunks",
    "tts_model_label",
]
