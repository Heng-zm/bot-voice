"""Backward compatibility shim forwarding to canonical app.cache.audio."""

from __future__ import annotations

from app.cache.audio import (
    TTSAudioCache,
    TTSFileIdCache,
    TTSSingleFlight,
    TTSUserHistoryTracker,
    clear_all_tts_caches,
    clear_user_tts_history,
    get_cached_telegram_file_id,
    get_global_tts_cache,
    get_global_tts_file_id_cache,
    get_global_tts_history,
    get_global_tts_single_flight,
    get_last_tts,
    get_last_tts_text,
    get_tts_cache_summary,
    invalidate_cached_telegram_file_id,
    make_tts_audio_cache_key,
    normalize_tts_text_for_hash,
    set_cached_telegram_file_id,
    set_last_tts,
    set_last_tts_text,
)

get_tts_cache = get_global_tts_cache

__all__ = [
    "TTSAudioCache",
    "TTSFileIdCache",
    "TTSSingleFlight",
    "TTSUserHistoryTracker",
    "clear_all_tts_caches",
    "clear_user_tts_history",
    "get_cached_telegram_file_id",
    "get_global_tts_cache",
    "get_global_tts_file_id_cache",
    "get_global_tts_history",
    "get_global_tts_single_flight",
    "get_last_tts",
    "get_last_tts_text",
    "get_tts_cache",
    "get_tts_cache_summary",
    "invalidate_cached_telegram_file_id",
    "make_tts_audio_cache_key",
    "normalize_tts_text_for_hash",
    "set_cached_telegram_file_id",
    "set_last_tts",
    "set_last_tts_text",
]
