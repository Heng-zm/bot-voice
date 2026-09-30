"""AI Assistant & Intelligence service facade."""

from __future__ import annotations

import logging
from typing import Any

from app.features.ai.chat import ask_gemini_async
from app.features.ai.summarizer import extract_key_points_bulleted
from app.features.ai.vision import analyze_image_async

logger = logging.getLogger("app.features.ai.service")


class AIService:
    """Facade for conversational chat, summarization, and vision analysis."""

    async def ask(self, prompt: str, user_id: int | None = None) -> str:
        return await ask_gemini_async(prompt)

    async def summarize(self, text: str, max_bullets: int = 3) -> list[str]:
        return extract_key_points_bulleted(text, num_sentences=max_bullets)

    async def vision(self, image_bytes: bytes, prompt: str = "") -> str:
        return await analyze_image_async(image_bytes, prompt=prompt)

    async def analyze_image(self, image_bytes: bytes, prompt: str = "") -> str:
        return await self.vision(image_bytes, prompt=prompt)


_AI_SERVICE = AIService()
ai_service = _AI_SERVICE


def get_ai_service() -> AIService:
    return _AI_SERVICE


__all__ = ["AIService", "ai_service", "get_ai_service"]
