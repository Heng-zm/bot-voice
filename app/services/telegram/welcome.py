"""Reusable Telegram welcome-content normalization and delivery."""

from __future__ import annotations

import logging
from typing import Any

from telegram.error import BadRequest

WELCOME_MESSAGE_SETTING_KEY = "welcome_message"
WELCOME_IMAGE_SETTING_KEY = "welcome_image_file_id"
WELCOME_MESSAGE_MAX_CHARS = 3500
WELCOME_IMAGE_FILE_ID_MAX_CHARS = 1024
WELCOME_IMAGE_CAPTION_MAX_CHARS = 1024


def normalize_welcome_message(value: Any, *, default_text: str) -> str:
    """Return a bounded plain-text welcome message."""
    text = str(value or "").replace("\x00", "").strip()
    return (text or default_text)[:WELCOME_MESSAGE_MAX_CHARS]


def normalize_welcome_image_file_id(value: Any) -> str:
    """Return a bounded Telegram photo file ID or an empty string."""
    return str(value or "").replace("\x00", "").strip()[:WELCOME_IMAGE_FILE_ID_MAX_CHARS]


async def send_welcome_content(
    message: Any,
    welcome_message: Any,
    image_file_id: Any,
    *,
    default_text: str,
    reply_markup: Any,
    safe_sender: Any,
    logger: logging.Logger | None = None,
) -> Any:
    """Send optional welcome media while respecting Telegram caption limits."""
    text = normalize_welcome_message(welcome_message, default_text=default_text)
    image = normalize_welcome_image_file_id(image_file_id)

    if image:
        try:
            if len(text) <= WELCOME_IMAGE_CAPTION_MAX_CHARS:
                return await safe_sender(lambda: message.reply_photo(
                    photo=image,
                    caption=text,
                    reply_markup=reply_markup,
                ))
            await safe_sender(lambda: message.reply_photo(photo=image))
        except BadRequest as exc:
            if logger is not None:
                logger.warning(
                    "Welcome image is unavailable; falling back to text: %s",
                    str(exc)[:300],
                )
        else:
            return await safe_sender(lambda: message.reply_text(
                text,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            ))

    return await safe_sender(lambda: message.reply_text(
        text,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    ))
