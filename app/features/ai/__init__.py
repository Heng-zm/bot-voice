"""AI features package (chat, vision, summarizer)."""

from __future__ import annotations

from app.features.ai.chat import ask_gemini_async
from app.features.ai.service import AIService, get_ai_service
from app.features.ai.summarizer import extract_key_points_bulleted
from app.features.ai.vision import analyze_image_async

__all__ = [
    "AIService",
    "analyze_image_async",
    "ask_gemini_async",
    "extract_key_points_bulleted",
    "get_ai_service",
]
