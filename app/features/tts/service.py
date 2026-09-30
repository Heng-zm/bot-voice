"""Text-to-Speech orchestration service with multi-tier caching and fallback cascade."""

from __future__ import annotations

import logging
from typing import Any

from app.cache.audio import (
    TTSAudioCache,
    make_tts_audio_cache_key,
    normalize_tts_text_for_hash,
)
from app.features.tts.providers.edge import EdgeTTSProvider
from app.features.tts.providers.gemini import GeminiTTSProvider
from app.features.tts.providers.gtts import GTTSProvider
from app.features.tts.providers.hf import HuggingFaceTTSProvider
from app.services.tts.voices import get_default_tts_model, normalize_tts_model

logger = logging.getLogger("app.features.tts.service")


class TTSService:
    """Orchestrates TTS synthesis across Edge, HuggingFace, Gemini, and gTTS providers with caching."""

    def __init__(self) -> None:
        self.edge_provider = EdgeTTSProvider()
        self.hf_provider = HuggingFaceTTSProvider()
        self.gemini_provider = GeminiTTSProvider()
        self.gtts_provider = GTTSProvider()

    async def synthesize(
        self,
        text: str,
        gender: str = "female",
        speed: float = 1.0,
        model: str = "auto",
        lang: str = "km",
        preferred_provider: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> bytes | None:
        """Synthesize text to audio bytes using the configured model and fallback order."""
        active_provider = preferred_provider or provider
        if active_provider == "edge" or model == "edge":
            edge_audio = await self.edge_provider.synthesize(text, gender=gender, speed=speed)
            if edge_audio:
                return edge_audio

        from app import legacy

        fn = getattr(legacy, "generate_voice_limited", None) or getattr(legacy, "generate_voice", None)
        if callable(fn):
            try:
                res = await fn(text, gender=gender, speed=speed, model=model)
                if isinstance(res, (bytes, bytearray)):
                    return bytes(res)
                if isinstance(res, tuple) and res and isinstance(res[0], (bytes, bytearray)):
                    return bytes(res[0])
            except Exception as exc:
                logger.warning("Legacy voice generator failed, falling back: %s", exc)

        # Fallback to Edge TTS if available
        if self.edge_provider.is_available():
            edge_audio = await self.edge_provider.synthesize(text)
            if edge_audio:
                return edge_audio

        # Fallback to gTTS
        return await self.gtts_provider.synthesize(text, lang=lang)

    @property
    def providers(self) -> list[Any]:
        return [self.edge_provider, self.hf_provider, self.gemini_provider, self.gtts_provider]


_TTS_SERVICE = TTSService()
tts_service = _TTS_SERVICE


def get_tts_service() -> TTSService:
    return _TTS_SERVICE


__all__ = ["TTSService", "get_tts_service", "tts_service"]
