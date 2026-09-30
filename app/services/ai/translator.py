"""Forwarding alias from translator to article_translator for backward compatibility."""

from app.services.ai.article_translator import (
    is_khmer,
    translate_chunk,
    translate_text,
    translate_text_async,
    translate_text_sync,
)

__all__ = [
    "is_khmer",
    "translate_chunk",
    "translate_text",
    "translate_text_async",
    "translate_text_sync",
]
