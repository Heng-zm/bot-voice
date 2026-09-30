"""Custom Telegram message, chat, and update filters for routing and guards."""

from __future__ import annotations

from typing import Any
from telegram import Update
from telegram.ext.filters import MessageFilter


class AdminFilter(MessageFilter):
    """Filter that matches updates from authorized administrators."""

    def filter(self, message: Any) -> bool:
        if not message or not message.from_user:
            return False
        from app.core.telegram_auth import is_telegram_admin

        return is_telegram_admin(message.from_user.id)


class MediaUrlFilter(MessageFilter):
    """Filter that matches messages containing TikTok, YouTube, Facebook, or Instagram URLs."""

    def filter(self, message: Any) -> bool:
        text = getattr(message, "text", "") or getattr(message, "caption", "") or ""
        lower = text.lower()
        return any(domain in lower for domain in ("tiktok.com", "douyin.com", "youtube.com", "youtu.be", "facebook.com", "fb.watch", "instagram.com"))


admin_filter = AdminFilter()
media_url_filter = MediaUrlFilter()

__all__ = ["AdminFilter", "MediaUrlFilter", "admin_filter", "media_url_filter"]
