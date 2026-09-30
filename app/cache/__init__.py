"""Unified caching package (memory, redis, audio, telegram)."""

from __future__ import annotations

from app.cache.audio import (
    TTSAudioCache,
    TTSFileIdCache,
    TTSSingleFlight,
    make_tts_audio_cache_key,
    normalize_tts_text_for_hash,
)
from app.cache.memory import MemoryCache
from app.cache.redis import get_redis_client
from app.cache.telegram import clear_user_state, get_user_state, set_user_state

__all__ = [
    "MemoryCache",
    "TTSAudioCache",
    "TTSFileIdCache",
    "TTSSingleFlight",
    "clear_user_state",
    "get_redis_client",
    "get_user_state",
    "make_tts_audio_cache_key",
    "normalize_tts_text_for_hash",
    "set_user_state",
]
