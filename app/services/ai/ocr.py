"""Unified OCR Service with intelligent multi-provider routing and fallbacks."""

from __future__ import annotations

import logging
from typing import Any

from app.services.ai.gemini import (
    GEMINI_MODEL_DEFAULT,
    extract_gemini_text,
    generate_content_with_fallback,
)

logger = logging.getLogger(__name__)


def ask_gemini_ocr_bytes(
    client: Any,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str = GEMINI_MODEL_DEFAULT,
    user_prompt: str = "",
) -> str:
    """Extract text or analyze image using Google Gemini Vision with model fallback."""
    if client is None:
        raise RuntimeError("Gemini client is not configured.")
    if not image_bytes:
        raise RuntimeError("Empty image data.")

    from google.genai import types as _gtypes

    if user_prompt.strip():
        prompt = (
            f"You are an expert AI Vision assistant.\n"
            f"Please analyze this image and answer the user request precisely and clearly in natural Khmer:\n\n"
            f"USER REQUEST: {user_prompt.strip()}\n\n"
            f"Keep your explanation clear, well-structured, and suitable for spoken text-to-speech audio narration."
        )
    else:
        prompt = (
            "Extract all visible and readable text from this image with high fidelity.\n"
            "- Accurately preserve Khmer script (ព្យញ្ជនៈ, ស្រៈ, ជើង, វណ្ណយុត្តិ, លេខខ្មែរ), English, Chinese, Vietnamese, Thai, etc.\n"
            "- Maintain original structure, paragraphs, bullet points, and tables where appropriate.\n"
            "- If the image contains zero readable text, return exactly: NOTEXT.\n"
            "- Return only the extracted text without introductory or concluding remarks."
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
    text = extract_gemini_text(response)
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
        user_prompt: str = "",
    ) -> tuple[str, str, str]:
        """Extract text and return (text, provider_name, model_name)."""
        if not image_bytes:
            raise RuntimeError("Empty image data.")

        errors: list[str] = []

        # 1. Primary: Gemini Vision OCR
        if self.gemini_client is not None:
            try:
                text = ask_gemini_ocr_bytes(
                    self.gemini_client,
                    image_bytes,
                    mime_type,
                    user_prompt=user_prompt,
                )
                return text, "gemini", GEMINI_MODEL_DEFAULT
            except Exception as exc:
                logger.warning("Gemini Vision OCR failed: %s", exc)
                errors.append(f"gemini: {exc}")

        # 2. Fallback: Hugging Face TrOCR
        if self.hf_client is not None:
            try:
                if hasattr(self.hf_client, "image_to_text"):
                    result = self.hf_client.image_to_text(image=image_bytes)
                    if isinstance(result, list) and result and isinstance(result[0], dict):
                        text = str(result[0].get("generated_text") or "").strip()
                    elif isinstance(result, str):
                        text = result.strip()
                    else:
                        text = str(getattr(result, "generated_text", "") or "").strip()
                    if text:
                        return text, "huggingface", "trocr"
                elif callable(self.hf_client):
                    result = self.hf_client(image_bytes)
                    text = str(result or "").strip()
                    if text:
                        return text, "huggingface", "trocr"
                errors.append("huggingface: empty OCR text returned")
            except Exception as exc:
                errors.append(f"huggingface: {exc}")

        if errors:
            raise RuntimeError(f"OCR processing failed across providers: {'; '.join(errors)}")
        raise RuntimeError("No OCR provider configured or available.")


__all__ = [
    "OCRService",
    "ask_gemini_ocr_bytes",
]
