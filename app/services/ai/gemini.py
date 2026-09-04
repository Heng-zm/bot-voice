"""Gemini client and multimodal inference helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

GEMINI_MODEL_DEFAULT = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"


def generate_content_with_fallback(
    client: Any,
    contents: Any,
    preferred_model: str = GEMINI_MODEL_DEFAULT,
    config: Any = None,
) -> Any:
    """Generate content with automatic fallback across models if quota (429) is hit."""
    if client is None:
        raise RuntimeError("Gemini client is not configured.")

    candidates = [preferred_model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
    unique_models: list[str] = []
    for m in candidates:
        if m and m not in unique_models:
            unique_models.append(m)

    last_exc: Exception | None = None
    for model_name in unique_models:
        try:
            kwargs: dict[str, Any] = {"model": model_name, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            last_exc = exc
            err_text = str(exc)
            if any(k in err_text for k in ("429", "RESOURCE_EXHAUSTED", "quota", "Quota")):
                logger.warning(
                    "Gemini model %s quota exceeded (429); falling back to next available model...",
                    model_name,
                )
                continue
            raise exc

    if last_exc:
        raise last_exc
    raise RuntimeError("No Gemini models succeeded.")


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
    "generate_content_with_fallback",
    "is_retryable_gemini_error",
]
