"""Media downloader service facade routing URLs to platform-specific extractors."""

from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from app.features.downloader.facebook import handle_facebook_url
from app.features.downloader.instagram import handle_instagram_url
from app.features.downloader.tiktok import handle_tiktok_url
from app.features.downloader.youtube import handle_youtube_url

logger = logging.getLogger("app.features.downloader.service")


class MediaDownloaderService:
    """Detects platform and delegates media extraction to the proper platform handler."""

    @property
    def downloaders(self) -> list[Any]:
        return [handle_tiktok_url, handle_facebook_url, handle_instagram_url, handle_youtube_url]

    def detect_platform(self, url: str) -> str | None:
        lower = url.lower()
        if "tiktok.com" in lower or "douyin.com" in lower:
            return "tiktok"
        if "youtube.com" in lower or "youtu.be" in lower:
            return "youtube"
        if "facebook.com" in lower or "fb.watch" in lower:
            return "facebook"
        if "instagram.com" in lower:
            return "instagram"
        return None

    async def route_media_url(self, update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> bool:
        platform = self.detect_platform(url)
        if platform == "tiktok":
            await handle_tiktok_url(update, context, url)
            return True
        if platform == "youtube":
            await handle_youtube_url(update, context, url)
            return True
        if platform == "facebook":
            await handle_facebook_url(update, context, url)
            return True
        if platform == "instagram":
            await handle_instagram_url(update, context, url)
            return True
        return False


DownloaderService = MediaDownloaderService
_DOWNLOADER_SERVICE = MediaDownloaderService()
downloader_service = _DOWNLOADER_SERVICE


def get_downloader_service() -> MediaDownloaderService:
    return _DOWNLOADER_SERVICE


__all__ = [
    "DownloaderService",
    "MediaDownloaderService",
    "downloader_service",
    "get_downloader_service",
]
