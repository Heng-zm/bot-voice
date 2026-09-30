"""HuggingFace Inference API TTS provider (Khmer Space neural engine)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("app.features.tts.providers.hf")


class HuggingFaceTTSProvider:
    """Provider wrapper for HuggingFace Space Khmer TTS."""

    name = "hf"

    async def synthesize(self, text: str, gender: str = "female", speed: float = 1.0) -> bytes | None:
        try:
            from app import legacy
            fn = getattr(legacy, "_hf_tts_space_predict_sync", None)
            if callable(fn):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, fn, text)
        except Exception as exc:
            logger.warning("HuggingFace Space TTS synthesis failed: %s", exc)
        return None


__all__ = ["HuggingFaceTTSProvider"]
