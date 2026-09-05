"""Gemini client and multimodal inference helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

GEMINI_MODEL_DEFAULT = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"


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
            "rate limit",
            "ratelimit",
            "deadline exceeded",
            "overloaded",
            "connection error",
            "connect error",
            "read timeout",
            "socket error",
        )
    )


def extract_gemini_text(response: Any) -> str:
    """Safely extract text from Gemini response without ValueError on safety blocks or empty candidates."""
    if response is None:
        return ""
    try:
        val = getattr(response, "text", "")
        if val:
            return str(val).strip()
    except (ValueError, AttributeError) as exc:
        logger.debug("Gemini response.text unavailable: %s", exc)

    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            cand_text = "".join(getattr(p, "text", "") or "" for p in parts if getattr(p, "text", None))
            if cand_text.strip():
                return cand_text.strip()
    except Exception as exc:
        logger.debug("Gemini candidates text extraction error: %s", exc)
    return ""


def generate_content_with_fallback(
    client: Any,
    contents: Any,
    preferred_model: str = GEMINI_MODEL_DEFAULT,
    config: Any = None,
) -> Any:
    """Generate content with automatic fallback across models if quota, transient error, or unavailable."""
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
            err_lower = err_text.lower()
            if is_retryable_gemini_error(exc) or any(
                k in err_lower for k in ("404", "not found", "not supported", "unsupported")
            ):
                logger.warning(
                    "Gemini model %s failed (%s); falling back to next available model...",
                    model_name,
                    exc,
                )
                continue
            raise exc

    if last_exc:
        raise last_exc
    raise RuntimeError("No Gemini models succeeded.")


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
    "extract_gemini_text",
    "generate_content_with_fallback",
    "is_retryable_gemini_error",
]
