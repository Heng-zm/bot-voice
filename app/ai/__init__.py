"""AI conversational, vision, OCR, and summarization services package."""

from __future__ import annotations

from app.features.ai.chat import ask_gemini_async, generate_content_with_fallback
from app.features.ai.service import AIService, ai_service, get_ai_service
from app.features.ai.summarizer import extract_key_points_bulleted
from app.features.ai.vision import analyze_image_async

__all__ = [
    "AIService",
    "ai_service",
    "analyze_image_async",
    "ask_gemini_async",
    "extract_key_points_bulleted",
    "generate_content_with_fallback",
    "get_ai_service",
]
