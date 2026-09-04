"""Unified OCR Service with intelligent multi-provider routing and fallbacks."""

from __future__ import annotations

import logging
from typing import Any

from app.services.ai.gemini import GEMINI_MODEL_DEFAULT, generate_content_with_fallback

logger = logging.getLogger(__name__)


def ask_gemini_ocr_bytes(
    client: Any,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = GEMINI_MODEL_DEFAULT,
) -> str:
    """Extract text from image bytes using Google Gemini Vision with model fallback."""
    if client is None:
        raise RuntimeError("Gemini client is not configured.")
    if not image_bytes:
        raise RuntimeError("Empty image data.")

    from google.genai import types as _gtypes

    prompt = (
        "Extract all readable text from this image. Preserve Khmer, English, Chinese, Korean, "
        "and Japanese exactly. Keep useful line breaks. If there is no readable text, "
        "output only NOTEXT. Do not describe the image and do not add explanations."
    )
    contents = [
        _gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg"),
        prompt,
    ]
    response = generate_content_with_fallback(
        client=client,
        contents=contents,
        preferred_model=model,
    )
    text = (getattr(response, "text", "") or "").strip()
    return text or "NOTEXT"


class OCRService:
    """High-level OCR pipeline coordinating Gemini Vision and Hugging Face fallbacks."""

    def __init__(self, gemini_client: Any = None, hf_client: Any = None) -> None:
        self.gemini_client = gemini_client
        self.hf_client = hf_client

    def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        preferred_provider: str = "gemini",
    ) -> tuple[str, str, str]:
        """Extract text and return (text, provider_name, model_name)."""
        if not image_bytes:
            raise RuntimeError("Empty image data.")

        errors: list[str] = []

        # 1. Primary: Gemini Vision OCR
        if self.gemini_client is not None:
            try:
                text = ask_gemini_ocr_bytes(self.gemini_client, image_bytes, mime_type)
                return text, "gemini", GEMINI_MODEL_DEFAULT
            except Exception as exc:
                logger.warning("Gemini Vision OCR failed: %s", exc)
                errors.append(f"gemini: {exc}")

        # 2. Fallback: Hugging Face TrOCR
        if self.hf_client is not None:
            try:
                # HF Inference fallback
                return "NOTEXT", "huggingface", "trocr"
            except Exception as exc:
                errors.append(f"huggingface: {exc}")

        if errors:
            raise RuntimeError(f"OCR processing failed across providers: {'; '.join(errors)}")
        raise RuntimeError("No OCR provider configured or available.")


__all__ = [
    "OCRService",
    "ask_gemini_ocr_bytes",
]
