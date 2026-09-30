"""Text-to-Speech feature package."""

from __future__ import annotations

from app.features.tts.handlers import handle_tts_command
from app.features.tts.service import TTSService, get_tts_service

__all__ = ["TTSService", "get_tts_service", "handle_tts_command"]
