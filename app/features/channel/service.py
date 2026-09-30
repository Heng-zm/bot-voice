"""Channel narration service for Telegram channels."""

from __future__ import annotations

import re
from typing import Any

from app.services.telegram.channel import (
    OPT_OUT_TAGS,
    clean_channel_text,
    is_narration_opted_out,
)


class ChannelNarratorService:
    """Service handling text normalization and business rules for channel audio narration."""

    def __init__(self) -> None:
        self.opt_out_tags = OPT_OUT_TAGS

    def is_opted_out(self, text: str) -> bool:
        """Check if post contains opt-out tags."""
        return is_narration_opted_out(text)

    def clean_text(self, raw_text: str, max_chars: int = 2000) -> str:
        """Clean post text for TTS voice generation."""
        return clean_channel_text(raw_text, max_chars=max_chars)


channel_service = ChannelNarratorService()

__all__ = [
    "ChannelNarratorService",
    "OPT_OUT_TAGS",
    "channel_service",
    "clean_channel_text",
    "is_narration_opted_out",
]
