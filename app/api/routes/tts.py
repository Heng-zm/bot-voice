"""Backward compatibility shim for app.api.routes.tts."""

from __future__ import annotations

from app.api.tts import (
    DEFAULT_TTS_SPEED,
    MAX_TTS_TEXT_LENGTH,
    SUPPORTED_MODELS,
    VALID_GENDERS,
    router,
)

__all__ = [
    "DEFAULT_TTS_SPEED",
    "MAX_TTS_TEXT_LENGTH",
    "SUPPORTED_MODELS",
    "VALID_GENDERS",
    "router",
]