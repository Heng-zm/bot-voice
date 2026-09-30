"""Google Gemini AI Voice & conversational synthesis provider."""

from __future__ import annotations

import logging
from typing import Any

from app import legacy

logger = logging.getLogger("app.features.tts.providers.gemini")


class GeminiTTSProvider:
    """Provider wrapper for Google Gemini multimodal audio synthesis."""

    name = "gemini"

    async def synthesize(self, text: str, voice_name: str = "Puck") -> bytes | None:
        try:
            fn = getattr(legacy, "_gemini_voice_synthesis_async", None)
            if callable(fn):
                return await fn(text, voice_name=voice_name)
        except Exception as exc:
            logger.warning("Gemini voice synthesis failed: %s", exc)
        return None


__all__ = ["GeminiTTSProvider"]
