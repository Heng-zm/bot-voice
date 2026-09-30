"""AI Vision & Multimodal Image Understanding."""

from __future__ import annotations

import logging
from typing import Any

from app import legacy

logger = logging.getLogger("app.features.ai.vision")


async def analyze_image_async(image_bytes: bytes, prompt: str = "Describe this image in Khmer.") -> str:
    """Analyze image using Gemini Vision."""
    try:
        fn = getattr(legacy, "_gemini_vision_async", None)
        if callable(fn):
            return await fn(image_bytes, prompt=prompt)
    except Exception as exc:
        logger.warning("Gemini vision analysis failed: %s", exc)
    return "មិនអាចវិភាគរូបភាពបានទេ។"


__all__ = ["analyze_image_async"]
