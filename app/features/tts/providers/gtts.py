"""Google TTS (gTTS) fallback speech synthesis provider."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.ai.gtts_narrator import generate_khmer_gtts_audio

logger = logging.getLogger("app.features.tts.providers.gtts")


class GTTSProvider:
    """Provider wrapper for Google Translate TTS fallback."""

    name = "gtts"

    async def synthesize(self, text: str, lang: str = "km") -> bytes | None:
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, generate_khmer_gtts_audio, text, 1.0, False, lang)
        except Exception as exc:
            logger.warning("gTTS synthesis failed: %s", exc)
            return None


__all__ = ["GTTSProvider"]
