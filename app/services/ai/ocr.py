"""OCR media validation and file-format helpers."""

from __future__ import annotations

from pathlib import Path


def normalize_media_suffix(value: str, *, default: str) -> str:
    suffix = str(value or default).strip().lower()
    if not suffix.startswith(".") or len(suffix) > 10:
        return default
    if any(character not in ".abcdefghijklmnopqrstuvwxyz0123456789" for character in suffix):
        return default
    return suffix


def normalize_ocr_result(value: object, *, no_text_message: str) -> str:
    text = str(value or "").strip()
    if not text or text.upper() == "NOTEXT":
        return str(no_text_message)
    return text


def normalize_ocr_provider(value: object, *, default: str = "gemini") -> str:
    provider = str(value or default or "gemini").lower().strip()
    provider = {
        "huggingface": "hf",
        "hugging_face": "hf",
        "hf_ocr": "hf",
        "google": "gemini",
        "google_gemini": "gemini",
        "gemini_ocr": "gemini",
        "tesseract": "local",
        "easyocr": "local",
    }.get(provider, provider)
    return provider if provider in {"gemini", "auto", "hf", "local"} else "gemini"


def normalize_preferred_ocr_provider(
    value: object,
    *,
    default: str = "gemini",
) -> str:
    provider = str(value or default or "gemini").lower().strip()
    return "hf" if provider in {"hf", "huggingface", "hugging_face", "hf_ocr"} else "gemini"


def detect_image_mime(path: str) -> str:
    """Detect supported image MIME types from magic bytes."""

    try:
        with Path(path).open("rb") as handle:
            header = handle.read(12)
    except OSError:
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header[:2] == b"\xff\xd8":
        return "image/jpeg"
    if header[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    return "image/jpeg"


__all__ = [
    "detect_image_mime",
    "normalize_media_suffix",
    "normalize_ocr_provider",
    "normalize_ocr_result",
    "normalize_preferred_ocr_provider",
]
