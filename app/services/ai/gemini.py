"""Gemini client and multimodal inference helpers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

GEMINI_MODEL_DEFAULT = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def is_retryable_gemini_error(exc: BaseException | str) -> bool:
    """Determine if a Gemini API failure is transient and can be retried."""
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "unavailable",
            "high demand",
            "resource exhausted",
            "temporarily overloaded",
            "service unavailable",
            "quota exceeded",
        )
    )


def detect_image_mime_from_bytes(header: bytes) -> str:
    """Detect image MIME type from initial header bytes."""
    if len(header) >= 8 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if len(header) >= 2 and header[:2] == b"\xff\xd8":
        return "image/jpeg"
    if len(header) >= 6 and header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def detect_image_mime(path: str) -> str:
    """Read magic bytes from file and return MIME type."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(12)
        return detect_image_mime_from_bytes(header)
    except OSError:
        return "image/jpeg"


__all__ = [
    "GEMINI_MODEL_DEFAULT",
    "detect_image_mime",
    "detect_image_mime_from_bytes",
    "is_retryable_gemini_error",
]
