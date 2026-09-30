"""Telegram interaction handlers for photo and document OCR."""

from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from app import legacy

logger = logging.getLogger("app.features.ocr.handlers")


async def handle_photo_ocr(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photos for OCR or visual analysis."""
    fn = getattr(legacy, "_handle_photo_ocr", None)
    if callable(fn):
        await fn(update, context)


__all__ = ["handle_photo_ocr"]
