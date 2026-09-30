"""Media downloader services package."""

from app.services.downloader.facebook import (
    cmd_facebook,
    extract_facebook_url,
    facebook_callback,
    handle_facebook_download,
    is_facebook_url,
)
from app.services.downloader.instagram import (
    cmd_instagram,
    extract_instagram_url,
    handle_instagram_download,
    instagram_callback,
    is_instagram_url,
)
from app.services.downloader.tiktok import (
    cmd_tiktok,
    extract_tiktok_url,
    handle_tiktok_download,
    handle_tiktok_file_download,
    is_tiktok_url,
    tiktok_callback,
)
from app.services.downloader.youtube import (
    cmd_youtube,
    extract_youtube_url,
    handle_youtube_download,
    is_youtube_url,
    youtube_callback,
)

__all__ = [
    "cmd_facebook",
    "cmd_instagram",
    "cmd_tiktok",
    "cmd_youtube",
    "extract_facebook_url",
    "extract_instagram_url",
    "extract_tiktok_url",
    "extract_youtube_url",
    "facebook_callback",
    "handle_facebook_download",
    "handle_instagram_download",
    "handle_tiktok_download",
    "handle_tiktok_file_download",
    "handle_youtube_download",
    "instagram_callback",
    "is_facebook_url",
    "is_instagram_url",
    "is_tiktok_url",
    "is_youtube_url",
    "tiktok_callback",
    "youtube_callback",
]

