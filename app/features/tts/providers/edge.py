"""Microsoft Edge Neural TTS provider implementation."""

from __future__ import annotations

import asyncio
import io
import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger("app.features.tts.providers.edge")


class EdgeTTSProvider:
    """Provider wrapper for Microsoft Edge TTS neural voices."""

    name = "edge"

    def __init__(self) -> None:
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import edge_tts  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
        return self._available

    async def synthesize(
        self,
        text: str,
        voice: str = "km-KH-PisethNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bytes | None:
        if not self.is_available():
            logger.debug("Edge TTS not available.")
            return None

        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
            )
            buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])
            audio_data = buffer.getvalue()
            return audio_data if len(audio_data) > 0 else None
        except Exception as exc:
            logger.warning("Edge TTS synthesis failed: %s", exc)
            return None


__all__ = ["EdgeTTSProvider"]
