"""Telegram interaction handlers for TTS voice synthesis."""

from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from app.features.tts.service import get_tts_service

logger = logging.getLogger("app.features.tts.handlers")


async def handle_tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /tts command in Telegram."""
    if not update.message:
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text("សូមផ្ញើអត្ថបទដែលអ្នកចង់បម្លែងជាសំឡេង។ ឧទាហរណ៍៖ `/tts សួស្តី`", parse_mode="Markdown")
        return

    svc = get_tts_service()
    audio = await svc.synthesize(text)
    if audio:
        await update.message.reply_voice(voice=audio)
    else:
        await update.message.reply_text("សូមអភ័យទោស ការបង្កើតសំឡេងមិនជោគជ័យ។ សូមព្យាយាមម្តងទៀត។")


__all__ = ["handle_tts_command"]
