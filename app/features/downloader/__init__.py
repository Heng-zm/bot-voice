"""Media downloader feature package for TikTok, Facebook, Instagram, and YouTube."""

from __future__ import annotations

from app.features.downloader.common import StatusCardAnimator, format_file_size, sanitize_filename
from app.features.downloader.facebook import handle_facebook_download
from app.features.downloader.instagram import handle_instagram_download
from app.features.downloader.service import MediaDownloaderService, get_downloader_service
from app.features.downloader.tiktok import handle_tiktok_download
from app.features.downloader.youtube import handle_youtube_download

# Aliases for convenience
handle_facebook_url = handle_facebook_download
handle_instagram_url = handle_instagram_download
handle_tiktok_url = handle_tiktok_download
handle_youtube_url = handle_youtube_download

__all__ = [
    "MediaDownloaderService",
    "StatusCardAnimator",
    "format_file_size",
    "get_downloader_service",
    "handle_facebook_download",
    "handle_facebook_url",
    "handle_instagram_download",
    "handle_instagram_url",
    "handle_tiktok_download",
    "handle_tiktok_url",
    "handle_youtube_download",
    "handle_youtube_url",
    "sanitize_filename",
]
