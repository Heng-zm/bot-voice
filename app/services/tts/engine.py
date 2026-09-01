"""Modular Text-to-Speech (TTS) Synthesis Engine.

Coordinates voice synthesis across Google Gemini Audio, Microsoft Edge TTS,
and Hugging Face Khmer Kiri models with automatic fallbacks and audio caching.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from app.services.tts.cache import (
    get_global_tts_cache,
    make_tts_audio_cache_key,
)
from app.services.tts.voices import (
    clean_tts_text,
    normalize_tts_model,
    resolve_tts_voice_candidates,
    split_text_chunks,
)

logger = logging.getLogger(__name__)


class TTSEngine:
    """Core TTS orchestrator handling caching, chunking, and multi-provider dispatch."""

    def __init__(self, cache_enabled: bool = True) -> None:
        self.cache_enabled = bool(cache_enabled)

    def get_cached_audio(
        self,
        text: str,
        gender: str,
        speed: float,
        model: str,
    ) -> bytes | None:
        """Check in-memory LRU audio cache for matching generated voice."""
        if not self.cache_enabled:
            return None
        key = make_tts_audio_cache_key(text, gender, speed, model)
        return get_global_tts_cache().get(key)

    def cache_audio(
        self,
        text: str,
        gender: str,
        speed: float,
        model: str,
        audio_bytes: bytes,
    ) -> None:
        """Store generated audio bytes in in-memory LRU cache."""
        if not self.cache_enabled or not audio_bytes:
            return
        key = make_tts_audio_cache_key(text, gender, speed, model)
        get_global_tts_cache().set(key, audio_bytes)


_GLOBAL_TTS_ENGINE = TTSEngine()


def get_global_tts_engine() -> TTSEngine:
    return _GLOBAL_TTS_ENGINE


__all__ = [
    "TTSEngine",
    "get_global_tts_engine",
]
