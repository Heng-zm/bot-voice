"""OCR text extraction feature package."""

from __future__ import annotations

from app.features.ocr.extractor import ask_gemini_ocr_bytes
from app.features.ocr.handlers import handle_photo_ocr
from app.features.ocr.service import OCRService, get_ocr_service

__all__ = [
    "OCRService",
    "ask_gemini_ocr_bytes",
    "get_ocr_service",
    "handle_photo_ocr",
]
