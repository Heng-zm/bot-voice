"""High-speed TikTok Downloader Service.

Features:
- No-watermark HD Video downloading with multi-mirror failover
- yt-dlp backup engine for 100% extraction uptime
- Original Sound / Music MP3 extraction with filename sanitization
- Photo Slide Carousel album support (safely handling 1 to 30 photos)
- 0ms delivery via Telegram file_id caching
- AI content summarization and translation in natural Khmer via Gemini
- Concurrency deduplication to eliminate duplicate downloads
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import suppress
import html
import io
import json
import logging
import os
import re
import tempfile
import threading
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

try:
    from telegram import (
        CallbackQuery,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        InputMediaPhoto,
        Update,
    )
    from telegram.ext import ContextTypes
except ImportError:
    class _DummyTelegramObject:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class InlineKeyboardButton(_DummyTelegramObject):
        pass

    class InlineKeyboardMarkup(_DummyTelegramObject):
        def __init__(self, inline_keyboard: Any = None, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.inline_keyboard = inline_keyboard or []

    class InputMediaPhoto(_DummyTelegramObject):
        pass

    CallbackQuery = _DummyTelegramObject  # type: ignore[misc,assignment]
    Update = _DummyTelegramObject  # type: ignore[misc,assignment]
    ContextTypes = Any  # type: ignore[misc,assignment]

logger = logging.getLogger("app.downloader.tiktok")

_TIKTOK_URL_RE = re.compile(
    r"https?://(?:(?:www|m|vm|vt)\.)?tiktok\.com/(?:t/[A-Za-z0-9_-]+|@[\w.-]+/(?:video|photo)/\d+|v/\d+\.html|[\w.-]+(?:/[\w.-]*)*)(?:\?[^\s]*)?",
    re.IGNORECASE,
)

# In-memory LRU cache for TikTok metadata and Telegram file_ids
_TIKTOK_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_TIKTOK_CACHE_LOCK = threading.RLock()
_TIKTOK_CACHE_MAX = 500

# Concurrency tracker for in-flight download tasks
_IN_FLIGHT_TASKS: set[str] = set()
_IN_FLIGHT_LOCK = threading.RLock()

# Telegram file upload limits
TELEGRAM_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB (Bot API hard limit)
TELEGRAM_MAX_URL_BYTES = 20 * 1024 * 1024     # 20 MB (Direct server handshake limit)

TIKWM_API_ENDPOINTS = (
    "https://www.tikwm.com/api/",
    "https://api.tikwm.com/api/",
    "https://tikwm.com/api/",
)


def is_tiktok_url(text: str | None) -> bool:
    """Check if the provided text contains or is a valid TikTok link."""
    if not text or not isinstance(text, str):
        return False
    return bool(_TIKTOK_URL_RE.search(text.strip()))


def extract_tiktok_url(text: str | None) -> str | None:
    """Extract the first valid TikTok URL from the input text."""
    if not text or not isinstance(text, str):
        return None
    match = _TIKTOK_URL_RE.search(text.strip())
    if match:
        return match.group(0).strip()
    return None


def get_cached_tiktok(video_id: str) -> dict[str, Any] | None:
    """Retrieve cached TikTok metadata and file_ids."""
    if not video_id:
        return None
    with _TIKTOK_CACHE_LOCK:
        item = _TIKTOK_CACHE.get(str(video_id))
        if item is not None:
            _TIKTOK_CACHE.move_to_end(str(video_id))
            return dict(item)
    return None


def cache_tiktok(video_id: str, data: dict[str, Any]) -> None:
    """Store TikTok metadata and file_ids in LRU cache."""
    if not video_id or not data:
        return
    with _TIKTOK_CACHE_LOCK:
        _TIKTOK_CACHE.pop(str(video_id), None)
        _TIKTOK_CACHE[str(video_id)] = dict(data)
        while len(_TIKTOK_CACHE) > _TIKTOK_CACHE_MAX:
            _TIKTOK_CACHE.popitem(last=False)


def clear_tiktok_cache() -> int:
    """Clear in-memory TikTok cache."""
    with _TIKTOK_CACHE_LOCK:
        count = len(_TIKTOK_CACHE)
        _TIKTOK_CACHE.clear()
        return count


def _fetch_tiktok_via_ytdlp(url: str) -> dict[str, Any] | None:
    """Fallback extractor using yt-dlp when TikWM mirrors are inaccessible."""
    try:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None

            return {
                "id": str(info.get("id") or ""),
                "title": str(info.get("title") or info.get("description") or ""),
                "play": info.get("url") or "",
                "hdplay": info.get("url") or "",
                "music": "",
                "music_title": str(info.get("track") or "Original Sound"),
                "music_author": str(info.get("artist") or info.get("uploader") or ""),
                "author_name": str(info.get("uploader") or ""),
                "author_username": str(info.get("uploader_id") or info.get("uploader") or ""),
                "images": [],
                "duration": int(info.get("duration") or 0),
                "views": int(info.get("view_count") or 0),
                "likes": int(info.get("like_count") or 0),
                "comments": int(info.get("comment_count") or 0),
                "shares": int(info.get("repost_count") or 0),
                "downloads": 0,
                "size": int(info.get("filesize") or info.get("filesize_approx") or 0),
                "sd_size": 0,
                "hd_size": int(info.get("filesize") or 0),
                "cover": str(info.get("thumbnail") or ""),
            }
    except Exception as exc:
        logger.debug("yt-dlp fallback for TikTok failed: %s", exc)
        return None


async def fetch_tiktok_data(url: str) -> dict[str, Any] | None:
    """Fetch TikTok metadata using TikWM mirror endpoints with yt-dlp fallback."""
    clean_url = extract_tiktok_url(url) or url.strip()
    if not clean_url:
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    payload = {"url": clean_url, "hd": 1}

    # 1. Try httpx across mirror endpoints
    try:
        import httpx

        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
            for endpoint in TIKWM_API_ENDPOINTS:
                try:
                    resp = await client.post(endpoint, data=payload, headers=headers)
                    if resp.status_code != 200:
                        resp = await client.get(endpoint, params=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, dict) and data.get("code") == 0:
                            raw = data.get("data") or {}
                            return _normalize_tiktok_data(raw)
                except Exception as endpoint_exc:
                    logger.debug("TikWM mirror %s failed: %s", endpoint, endpoint_exc)
    except Exception as exc:
        logger.debug("TikWM fetch via httpx failed: %s", exc)

    # 2. Resilient fallback using standard library urllib
    try:
        def _sync_fetch() -> dict[str, Any] | None:
            post_bytes = urllib.parse.urlencode(payload).encode("utf-8")
            for endpoint in TIKWM_API_ENDPOINTS:
                try:
                    req = urllib.request.Request(
                        endpoint,
                        data=post_bytes,
                        headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
                    )
                    with urllib.request.urlopen(req, timeout=18) as response:
                        if response.status == 200:
                            body = response.read().decode("utf-8", errors="replace")
                            data = json.loads(body)
                            if isinstance(data, dict) and data.get("code") == 0:
                                raw = data.get("data") or {}
                                return _normalize_tiktok_data(raw)
                except Exception as u_exc:
                    logger.debug("TikWM urllib mirror %s failed: %s", endpoint, u_exc)
            return None

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, _sync_fetch)
        if res:
            return res
    except Exception as exc:
        logger.debug("TikWM urllib fallback failed: %s", exc)

    # 3. Third-Tier: yt-dlp direct extraction
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_tiktok_via_ytdlp, clean_url)


def _normalize_tiktok_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize TikWM response dictionary into standardized bot schema."""
    video_id = str(raw.get("id") or "")
    title = str(raw.get("title") or "").strip()
    play = str(raw.get("play") or "").strip()
    hdplay = str(raw.get("hdplay") or "").strip() or play
    music = str(raw.get("music") or "").strip()
    music_info = raw.get("music_info") or {}
    music_title = str(music_info.get("title") or "Original Sound").strip()
    music_author = str(music_info.get("author") or "").strip()
    author = raw.get("author") or {}
    author_name = str(author.get("nickname") or "").strip()
    author_username = str(author.get("unique_id") or "").strip()
    images = [str(img) for img in (raw.get("images") or []) if isinstance(img, str) and img.startswith("http")]
    duration = int(raw.get("duration") or 0)
    views = int(raw.get("play_count") or 0)
    likes = int(raw.get("digg_count") or 0)
    comments = int(raw.get("comment_count") or 0)
    shares = int(raw.get("share_count") or 0)
    downloads = int(raw.get("download_count") or 0)
    sd_size = int(raw.get("size") or 0)
    hd_size = int(raw.get("hd_size") or 0)
    size = hd_size or sd_size
    cover = str(raw.get("cover") or "").strip()

    return {
        "id": video_id,
        "title": title,
        "play": play,
        "hdplay": hdplay,
        "music": music,
        "music_title": music_title,
        "music_author": music_author,
        "author_name": author_name,
        "author_username": author_username,
        "images": images,
        "duration": duration,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "downloads": downloads,
        "size": size,
        "sd_size": sd_size,
        "hd_size": hd_size,
        "cover": cover,
    }


async def download_media_bytes(
    url: str,
    max_bytes: int = TELEGRAM_MAX_UPLOAD_BYTES,
    timeout_s: float = 120.0,
) -> bytes | None:
    """Download media file bytes safely with size limits and chunking to prevent OOM."""
    if not url:
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.tiktok.com/",
    }

    try:
        import httpx

        timeout = httpx.Timeout(timeout_s, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200 and 0 < len(resp.content) <= max_bytes:
                return resp.content
    except Exception as exc:
        logger.debug("Media byte download error via httpx for %s: %s", url, exc)

    try:
        def _sync_download() -> bytes | None:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                if getattr(response, "status", 200) == 200:
                    data = response.read(max_bytes + 1)
                    if len(data) <= max_bytes:
                        return data
            return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_download)
    except Exception as exc:
        logger.debug("Media byte download error via urllib for %s: %s", url, exc)

    return None


async def download_media_to_file(
    url: str,
    dest_path: str,
    max_bytes: int = TELEGRAM_MAX_UPLOAD_BYTES,
    timeout_s: float = 180.0,
) -> int | None:
    """Stream media file directly to disk to prevent RAM/OOM spikes on large video downloads."""
    if not url:
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.tiktok.com/",
    }

    try:
        import httpx

        timeout = httpx.Timeout(timeout_s, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 200:
                    total = 0
                    with open(dest_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            total += len(chunk)
                            if total > max_bytes:
                                f.close()
                                with suppress(Exception):
                                    os.remove(dest_path)
                                return None
                            f.write(chunk)
                    if total > 0:
                        return total
    except Exception as exc:
        logger.debug("Media file stream error via httpx for %s: %s", url, exc)

    try:
        def _sync_stream() -> int | None:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                if getattr(response, "status", 200) == 200:
                    total = 0
                    with open(dest_path, "wb") as f:
                        while True:
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > max_bytes:
                                f.close()
                                with suppress(Exception):
                                    os.remove(dest_path)
                                return None
                            f.write(chunk)
                    if total > 0:
                        return total
            return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_stream)
    except Exception as exc:
        logger.debug("Media file stream error via urllib for %s: %s", url, exc)

    return None


def get_tiktok_video_kb(video_id: str, has_music: bool = True) -> InlineKeyboardMarkup:
    """Generate inline action keyboard for downloaded TikTok video."""
    buttons: list[list[InlineKeyboardButton]] = []
    top_row: list[InlineKeyboardButton] = []
    if has_music:
        top_row.append(InlineKeyboardButton("🎵 ទាញយក MP3", callback_data=f"tt_mp3:{video_id}"))
    top_row.append(InlineKeyboardButton("📁 ទាញយកជា File", callback_data=f"tt_file:{video_id}"))
    buttons.append(top_row)
    buttons.append([
        InlineKeyboardButton("🤖 AI សង្ខេប", callback_data=f"tt_ai:{video_id}"),
        InlineKeyboardButton("📊 ស្ថិតិវីដេអូ", callback_data=f"tt_stats:{video_id}"),
    ])
    buttons.append([
        InlineKeyboardButton("❌ បិទ (Close)", callback_data="close_msg"),
    ])
    return InlineKeyboardMarkup(buttons)


async def handle_tiktok_download(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    url: str,
) -> None:
    """Process TikTok download request and deliver media to user."""
    msg = update.effective_message
    if not msg:
        return

    clean_url = extract_tiktok_url(url) or url.strip()
    if not clean_url:
        await msg.reply_text("❌ សូមផ្ញើតំណភ្ជាប់ TikTok ត្រឹមត្រូវ។")
        return

    user_id = getattr(getattr(update, "effective_user", None), "id", "anon")
    task_key = f"dl:{user_id}:{clean_url}"

    with _IN_FLIGHT_LOCK:
        if task_key in _IN_FLIGHT_TASKS:
            await msg.reply_text("⏳ វីដេអូ TikTok នេះកំពុងដំណើរការទាញយកហើយ សូមរង់ចាំបន្តិច...")
            return
        _IN_FLIGHT_TASKS.add(task_key)

    if context and getattr(context, "bot", None) and hasattr(msg, "chat_id"):
        with suppress(Exception):
            await context.bot.send_chat_action(chat_id=msg.chat_id, action="upload_video")

    status_msg = None
    with suppress(Exception):
        status_msg = await msg.reply_text(
            "⚡ <b>TIKTOK ULTRA-DOWNLOADER</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "▰▰▰▱▱▱▱▱▱▱  <b>30%</b>\n"
            "📍 <b>ដំណាក់កាល:</b> <code>⠋ វិភាគតំណភ្ជាប់ (Analyzing URL)</code>\n"
            "💡 <i>កំពុងស្វែងរក & វិភាគទិន្នន័យមេឌៀ...</i>",
            parse_mode="HTML",
        )

    from app.services.telegram.formatters import StatusCardAnimator

    animator = StatusCardAnimator(
        status_msg,
        title="TIKTOK ULTRA-DOWNLOADER",
        stage="វិភាគតំណភ្ជាប់ (Analyzing URL)",
        percent=30,
        detail="កំពុងស្វែងរក & វិភាគទិន្នន័យមេឌៀ...",
    )

    try:
        animator.start()
        data = await fetch_tiktok_data(clean_url)
        if not data or not data.get("id"):
            await animator.stop()
            if status_msg and hasattr(status_msg, "edit_text"):
                with suppress(Exception):
                    await status_msg.edit_text("❌ មិនអាចទាញយកវីដេអូ TikTok នេះបានទេ។ សូមពិនិត្យមើលតំណភ្ជាប់ឡើងវិញ។")
            return

        video_id = data["id"]
        cached = get_cached_tiktok(video_id) or {}
        cached.update(data)
        cache_tiktok(video_id, cached)

        title = data.get("title") or "TikTok Video"
        author_user = data.get("author_username") or data.get("author_name") or "creator"
        author_name = data.get("author_name") or ""
        author_display = f"@{html.escape(author_user)}"
        if author_name and author_name != author_user:
            author_display += f" ({html.escape(author_name)})"

        duration = data.get("duration", 0)
        likes = data.get("likes", 0)
        views = data.get("views", 0)
        comments = data.get("comments", 0)
        shares = data.get("shares", 0)
        music_title = data.get("music_title") or "Original Sound"
        size_bytes = data.get("size", 0)

        await animator.push_state(
            stage="ទាញយកមេឌៀ (Fetching Stream)",
            percent=70,
            detail="កំពុងរៀបចំ និងបញ្ជូនមេឌៀទៅកាន់លោកអ្នក...",
            header_extra=f"👤 <b>អ្នកបង្កើត:</b> {author_display}",
        )

        # 1. Handle Photo Slide Carousel
        images = data.get("images") or []
        if images:
            caption_lines = [
                "🖼️ <b>TikTok Photo Slide</b>",
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"👤 <b>អ្នកបង្កើត:</b> {author_display}",
                f"📝 <b>ខ្លឹមសារ:</b> {html.escape(title[:260])}",
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"📸 <b>រូបភាព:</b> សរុប {len(images)} សន្លឹក",
                f"❤️ <b>Likes:</b> {likes:,} | 👁️ <b>Views:</b> {views:,}",
            ]
            if comments > 0 or shares > 0:
                caption_lines.append(f"💬 <b>Comments:</b> {comments:,} | 🔁 <b>Shares:</b> {shares:,}")
            caption = "\n".join(caption_lines)[:1020]

            # Safely handle single images vs multi-image albums
            if len(images) == 1:
                await msg.reply_photo(photo=images[0], caption=caption, parse_mode="HTML")
            else:
                # Chunk into batches of up to 10 for Telegram sendMediaGroup bounds
                photo_batches = [images[i : i + 10] for i in range(0, min(len(images), 30), 10)]
                for b_idx, batch in enumerate(photo_batches):
                    media_group = [
                        InputMediaPhoto(media=img_url, caption=caption if (b_idx == 0 and idx == 0) else None, parse_mode="HTML")
                        for idx, img_url in enumerate(batch)
                    ]
                    await msg.reply_media_group(media=media_group)
                    await asyncio.sleep(0.1)

            photo_top_row: list[InlineKeyboardButton] = []
            if data.get("music"):
                photo_top_row.append(InlineKeyboardButton("🎵 ទាញយក MP3", callback_data=f"tt_mp3:{video_id}"))
            photo_top_row.append(InlineKeyboardButton("📁 ទាញយកជា File", callback_data=f"tt_file:{video_id}"))
            kb = InlineKeyboardMarkup([
                photo_top_row,
                [
                    InlineKeyboardButton("🤖 AI សង្ខេប", callback_data=f"tt_ai:{video_id}"),
                    InlineKeyboardButton("📊 ស្ថិតិរូបភាព", callback_data=f"tt_stats:{video_id}"),
                ],
                [
                    InlineKeyboardButton("❌ បិទ (Close)", callback_data="close_msg"),
                ],
            ])
            await msg.reply_text(
                f"🎵 <b>សំឡេងដើម:</b> {html.escape(data.get('music_title', 'Original Sound'))}",
                parse_mode="HTML",
                reply_markup=kb,
            )

            await animator.stop()
            if status_msg and hasattr(status_msg, "delete"):
                with suppress(Exception):
                    await status_msg.delete()
            return

        # 2. Handle Video
        cached_vid_fid = cached.get("video_file_id")
        kb = get_tiktok_video_kb(video_id, has_music=bool(data.get("music")))

        hd_url = data.get("hdplay") or ""
        sd_url = data.get("play") or ""
        sd_size = int(data.get("sd_size") or 0)
        hd_size = int(data.get("hd_size") or 0)
        effective_size = hd_size or size_bytes

        target_play_url = hd_url or sd_url
        chosen_size = effective_size
        is_standard_fallback = False

        if hd_size > TELEGRAM_MAX_UPLOAD_BYTES:
            if sd_url and 0 < sd_size <= TELEGRAM_MAX_UPLOAD_BYTES:
                target_play_url = sd_url
                chosen_size = sd_size
                is_standard_fallback = True
            elif not sd_url or sd_size > TELEGRAM_MAX_UPLOAD_BYTES:
                dur_mins = duration // 60
                dur_secs = duration % 60
                dur_str = f"{dur_mins} នាទី {dur_secs} វិនាទី" if dur_mins > 0 else f"{duration} វិនាទី"
                size_mb = (hd_size or size_bytes) / (1024 * 1024)
                direct_url = hd_url or sd_url or clean_url

                large_caption = (
                    "📹 <b>វីដេអូ TikTok វែង (ទំហំលើសពី 50MB)</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"👤 <b>អ្នកបង្កើត:</b> {author_display}\n"
                    f"📝 <b>ខ្លឹមសារ:</b> {html.escape(title[:260])}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⏱️ <b>រយៈពេល:</b> {dur_str}\n"
                    f"📦 <b>ទំហំវីដេអូ:</b> {size_mb:.1f} MB (កម្រិត Telegram Bot អតិបរមា 50 MB)\n"
                    f"❤️ <b>Likes:</b> {likes:,} | 👁️ <b>Views:</b> {views:,}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💡 <i>ដោយសារវីដេអូនេះមានប្រវែងវែង និងទំហំលើសពី 50MB លោកអ្នកអាចចុចប៊ូតុងខាងក្រោមដើម្បីទាញយក ឬទស្សនាវីដេអូច្បាស់ដើម Full HD ដោយផ្ទាល់៖</i>"
                )[:1020]

                large_btns: list[list[InlineKeyboardButton]] = [
                    [InlineKeyboardButton(f"🌐 ទាញយកវីដេអូ HD ពេញ ({size_mb:.1f} MB)", url=direct_url)],
                ]
                if data.get("music"):
                    large_btns.append([InlineKeyboardButton("🎵 ទាញយកតែសំឡេង MP3", callback_data=f"tt_mp3:{video_id}")])
                large_btns.append([
                    InlineKeyboardButton("🤖 AI សង្ខេប", callback_data=f"tt_ai:{video_id}"),
                    InlineKeyboardButton("📊 ស្ថិតិវីដេអូ", callback_data=f"tt_stats:{video_id}"),
                ])
                large_btns.append([InlineKeyboardButton("❌ បិទ (Close)", callback_data="close_msg")])

                await animator.stop()
                if status_msg and hasattr(status_msg, "delete"):
                    with suppress(Exception):
                        await status_msg.delete()

                await msg.reply_text(
                    large_caption,
                    parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup(large_btns),
                )
                return

        size_str = f" | 📦 <b>ទំហំ:</b> {chosen_size / (1024 * 1024):.1f} MB" if chosen_size > 0 else ""
        quality_label = "TikTok (Standard គ្មាន Watermark)" if is_standard_fallback else "TikTok (គ្មាន Watermark HD)"

        caption_lines = [
            f"📹 <b>{quality_label}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"👤 <b>អ្នកបង្កើត:</b> {author_display}",
            f"📝 <b>ខ្លឹមសារ:</b> {html.escape(title[:260])}",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"⏱️ <b>រយៈពេល:</b> {duration}s{size_str}",
            f"❤️ <b>Likes:</b> {likes:,} | 👁️ <b>Views:</b> {views:,}",
        ]
        if comments > 0 or shares > 0:
            caption_lines.append(f"💬 <b>Comments:</b> {comments:,} | 🔁 <b>Shares:</b> {shares:,}")
        if music_title and music_title != "Original Sound":
            caption_lines.append(f"🎼 <b>សំឡេង:</b> <i>{html.escape(music_title[:50])}</i>")
        if is_standard_fallback and hd_url:
            caption_lines.append(
                f"💡 <i>សម្រាប់ Full HD 1080p ({hd_size / (1024 * 1024):.1f} MB) សូមចុចប៊ូតុង Direct Link</i>"
            )
        caption = "\n".join(caption_lines)[:1020]

        if is_standard_fallback and hd_url:
            custom_rows = [
                [InlineKeyboardButton(f"🌐 ទាញយក Full HD 1080p ({hd_size / (1024 * 1024):.1f} MB)", url=hd_url)]
            ]
            custom_rows.extend(kb.inline_keyboard)
            kb = InlineKeyboardMarkup(custom_rows)

        sent_msg = None

        # 1. 0ms Fast delivery from cached Telegram file_id
        if cached_vid_fid:
            try:
                sent_msg = await msg.reply_video(
                    video=cached_vid_fid,
                    caption=caption,
                    parse_mode="HTML",
                    duration=duration if duration > 0 else None,
                    supports_streaming=True,
                    reply_markup=kb,
                )
            except Exception as exc:
                logger.debug("Failed sending cached video file_id: %s", exc)

        # 2. Local Disk Stream Download (Reliable against TikTok CDN Referer-blocking)
        if not sent_msg and target_play_url:
            await animator.push_state(
                stage="រៀបចំឯកសារ (Packaging File)",
                percent=85,
                detail="កំពុងទាញយកទិន្នន័យវីដេអូច្បាស់ដើម...",
                header_extra=f"👤 <b>អ្នកបង្កើត:</b> {author_display}",
            )

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
                    tmp_path = tmp_f.name

                downloaded = await download_media_to_file(
                    target_play_url,
                    tmp_path,
                    max_bytes=TELEGRAM_MAX_UPLOAD_BYTES,
                    timeout_s=180.0,
                )

                if not downloaded or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    b_data = await download_media_bytes(target_play_url, max_bytes=TELEGRAM_MAX_UPLOAD_BYTES)
                    if b_data:
                        with open(tmp_path, "wb") as f:
                            f.write(b_data)
                        downloaded = len(b_data)

                if downloaded and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    with open(tmp_path, "rb") as vf:
                        sent_msg = await msg.reply_video(
                            video=vf,
                            caption=caption,
                            parse_mode="HTML",
                            duration=duration if duration > 0 else None,
                            supports_streaming=True,
                            reply_markup=kb,
                        )
                elif target_play_url:
                    try:
                        sent_msg = await msg.reply_video(
                            video=target_play_url,
                            caption=caption,
                            parse_mode="HTML",
                            duration=duration if duration > 0 else None,
                            supports_streaming=True,
                            reply_markup=kb,
                        )
                    except Exception as url_err:
                        logger.debug("Direct URL video upload fallback failed: %s", url_err)
            except Exception as vid_err:
                logger.warning("Local video upload failed: %s", vid_err)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    with suppress(Exception):
                        os.unlink(tmp_path)

        if sent_msg and getattr(sent_msg, "video", None):
            cached["video_file_id"] = sent_msg.video.file_id
            cache_tiktok(video_id, cached)

        await animator.stop()
        if status_msg and hasattr(status_msg, "delete"):
            with suppress(Exception):
                await status_msg.delete()

        if not sent_msg:
            fail_btns = [
                [InlineKeyboardButton("🌐 ទាញយកវីដេអូតាម Direct Link", url=target_play_url or clean_url)],
            ]
            if data.get("music"):
                fail_btns.append([InlineKeyboardButton("🎵 ទាញយក MP3", callback_data=f"tt_mp3:{video_id}")])
            fail_btns.append([InlineKeyboardButton("❌ បិទ", callback_data="close_msg")])
            await msg.reply_text(
                "⚠️ <b>មិនអាចបញ្ជូនវីដេអូទៅកាន់ Telegram ដោយផ្ទាល់បានទេ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "ឯកសារវីដេអូអាចមានទំហំធំ ឬបណ្ដាញរវល់។ លោកអ្នកអាចចុចប៊ូតុងខាងក្រោមដើម្បីទាញយកតាម Direct Link ផ្ទាល់៖",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(fail_btns),
            )

    except Exception as exc:
        logger.error("handle_tiktok_download error: %s", exc, exc_info=True)
        await animator.stop()
        if status_msg and hasattr(status_msg, "edit_text"):
            with suppress(Exception):
                await status_msg.edit_text(f"❌ បរាជ័យក្នុងការទាញយក TikTok: {exc}")
    finally:
        with _IN_FLIGHT_LOCK:
            _IN_FLIGHT_TASKS.discard(task_key)


async def handle_tiktok_mp3_download(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Deliver extracted MP3 audio for TikTok video with in-flight deduplication."""
    msg = query.message
    clean_vid = str(video_id).strip()
    cached = get_cached_tiktok(clean_vid) or {}

    if not cached or not cached.get("music"):
        with suppress(Exception):
            fresh = await fetch_tiktok_data(f"https://www.tiktok.com/@tiktok/video/{clean_vid}")
            if fresh:
                cached.update(fresh)
                cache_tiktok(clean_vid, cached)

    audio_fid = cached.get("audio_file_id")
    title = cached.get("music_title") or "TikTok Audio"
    performer = cached.get("music_author") or cached.get("author_username") or "TikTok"

    user_id = getattr(getattr(query, "from_user", None), "id", None) or getattr(getattr(msg, "chat", None), "id", "anon")
    task_key = f"mp3:{user_id}:{clean_vid}"

    with _IN_FLIGHT_LOCK:
        if task_key in _IN_FLIGHT_TASKS:
            with suppress(Exception):
                await query.answer("⏳ សំឡេង MP3 កំពុងដំណើរការ សូមរង់ចាំបន្តិច...", show_alert=False)
            return
        _IN_FLIGHT_TASKS.add(task_key)

    try:
        with suppress(Exception):
            await query.answer("⏳ កំពុងទាញយកសំឡេង MP3...")

        # Fast delivery from Telegram file_id (0ms)
        if audio_fid and msg:
            try:
                await msg.reply_audio(
                    audio=audio_fid,
                    title=title[:60],
                    performer=performer[:30],
                    caption="🎵 <b>សំឡេង TikTok (MP3 Audio)</b>",
                    parse_mode="HTML",
                )
                return
            except Exception as exc:
                logger.debug("Failed sending cached audio file_id: %s", exc)

        music_url = cached.get("music")
        if not music_url:
            with suppress(Exception):
                await query.answer("❌ រកមិនឃើញតំណភ្ជាប់សំឡេងទេ។", show_alert=True)
            return

        status_msg = None
        if msg and hasattr(msg, "reply_text"):
            with suppress(Exception):
                status_msg = await msg.reply_text(
                    "🎵 <b>TIKTOK MP3 EXTRACTOR</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🎼 <b>ចំណងជើង:</b> {html.escape(title[:40])}\n"
                    "▰▰▰▰▰▰▰▱▱▱  <b>70%</b>\n"
                    "📍 <b>ដំណាក់កាល:</b> <code>⠋ កំពុងទាញយកសំឡេង MP3...</code>\n"
                    "💡 <i>កំពុងរៀបចំបទភ្លេងគុណភាពខ្ពស់...</i>",
                    parse_mode="HTML",
                )

        from app.services.telegram.formatters import StatusCardAnimator

        animator = StatusCardAnimator(
            status_msg,
            title="TIKTOK MP3 EXTRACTOR",
            icon="🎵",
            header_extra=f"🎼 <b>ចំណងជើង:</b> {html.escape(title[:40])}",
            stage="កំពុងទាញយកសំឡេង MP3...",
            percent=70,
            detail="កំពុងរៀបចំបទភ្លេងគុណភាពខ្ពស់...",
        )

        try:
            animator.start()
            audio_bytes = await download_media_bytes(music_url, max_bytes=25 * 1024 * 1024)
            if audio_bytes and msg:
                await animator.push_state(
                    stage="កំពុងផ្ញើសំឡេង MP3...",
                    percent=90,
                    detail="កំពុងបញ្ជូនបទភ្លេង MP3...",
                )
                safe_name = re.sub(r'[\\/*?:"<>|]', "", title)[:40].strip() or "tiktok_audio"
                bio = io.BytesIO(audio_bytes)
                bio.name = f"{safe_name}.mp3"
                sent = await msg.reply_audio(
                    audio=bio,
                    title=title[:60],
                    performer=performer[:30],
                    caption="🎵 <b>សំឡេង TikTok (MP3 Audio)</b>",
                    parse_mode="HTML",
                )
                if sent and getattr(sent, "audio", None):
                    cached["audio_file_id"] = sent.audio.file_id
                    cache_tiktok(clean_vid, cached)
            else:
                with suppress(Exception):
                    await query.answer("❌ មិនអាចទាញយកសំឡេង MP3 បានទេ។", show_alert=True)
        finally:
            await animator.stop()
            if status_msg and hasattr(status_msg, "delete"):
                with suppress(Exception):
                    await status_msg.delete()
    finally:
        with _IN_FLIGHT_LOCK:
            _IN_FLIGHT_TASKS.discard(task_key)


async def handle_tiktok_file_download(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Deliver TikTok media as an uncompressed document file."""
    msg = query.message
    clean_vid = str(video_id).strip()
    cached = get_cached_tiktok(clean_vid) or {}

    if not cached or (not cached.get("hdplay") and not cached.get("play") and not cached.get("images")):
        with suppress(Exception):
            fresh = await fetch_tiktok_data(f"https://www.tiktok.com/@tiktok/video/{clean_vid}")
            if fresh:
                cached.update(fresh)
                cache_tiktok(clean_vid, cached)

    doc_fid = cached.get("doc_file_id")
    title = cached.get("title") or "TikTok Media"
    author_user = cached.get("author_username") or cached.get("author_name") or "creator"

    user_id = getattr(getattr(query, "from_user", None), "id", None) or getattr(getattr(msg, "chat", None), "id", "anon")
    task_key = f"file:{user_id}:{clean_vid}"

    with _IN_FLIGHT_LOCK:
        if task_key in _IN_FLIGHT_TASKS:
            with suppress(Exception):
                await query.answer("⏳ ឯកសារកំពុងដំណើរការ សូមរង់ចាំបន្តិច...", show_alert=False)
            return
        _IN_FLIGHT_TASKS.add(task_key)

    try:
        with suppress(Exception):
            await query.answer("⏳ កំពុងរៀបចំឯកសារ File...")

        caption = (
            f"📁 <b>ឯកសារ TikTok (Document File)</b>\n"
            f"👤 <b>អ្នកបង្កើត:</b> @{html.escape(author_user)}\n"
            f"📝 {html.escape(title[:250])}"
        )[:1020]

        # 1. 0ms Fast delivery from Telegram file_id
        if doc_fid and msg:
            try:
                await msg.reply_document(
                    document=doc_fid,
                    caption=caption,
                    parse_mode="HTML",
                    disable_content_type_detection=True,
                )
                return
            except Exception as exc:
                logger.debug("Failed sending cached document file_id: %s", exc)

        images = cached.get("images") or []
        play_url = cached.get("hdplay") or cached.get("play")
        file_size = int(cached.get("size") or cached.get("hd_size") or 0)

        if not images and play_url and file_size > TELEGRAM_MAX_UPLOAD_BYTES:
            if msg and hasattr(msg, "reply_text"):
                size_mb = file_size / (1024 * 1024)
                doc_kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton(f"🌐 ទាញយកឯកសារច្បាស់ដើម ({size_mb:.1f} MB)", url=play_url)],
                    [InlineKeyboardButton("❌ បិទ (Close)", callback_data="close_msg")],
                ])
                await msg.reply_text(
                    f"📁 <b>ឯកសារ TikTok (ទំហំលើសពី 50MB)</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"👤 <b>អ្នកបង្កើត:</b> @{html.escape(author_user)}\n"
                    f"📦 <b>ទំហំ:</b> {size_mb:.1f} MB (លើសពីកម្រិត Telegram Bot 50 MB)\n\n"
                    "💡 <i>សូមចុចប៊ូតុងខាងក្រោមដើម្បីទាញយកឯកសារច្បាស់ដើមតាម Browser ដោយផ្ទាល់៖</i>",
                    parse_mode="HTML",
                    reply_markup=doc_kb,
                )
            return

        status_msg = None
        if msg and hasattr(msg, "reply_text"):
            with suppress(Exception):
                status_msg = await msg.reply_text(
                    "📁 <b>TIKTOK FILE DOWNLOADER</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"👤 <b>អ្នកបង្កើត:</b> @{html.escape(author_user)}\n"
                    "▰▰▰▰▰▰▰▱▱▱  <b>70%</b>\n"
                    "📍 <b>ដំណាក់កាល:</b> <code>⠋ កំពុងរៀបចំឯកសារ File...</code>\n"
                    "💡 <i>កំពុងទាញយកទិន្នន័យច្បាស់ដើម (Full Quality)...</i>",
                    parse_mode="HTML",
                )

        from app.services.telegram.formatters import StatusCardAnimator

        animator = StatusCardAnimator(
            status_msg,
            title="TIKTOK FILE DOWNLOADER",
            icon="📁",
            header_extra=f"👤 <b>អ្នកបង្កើត:</b> @{html.escape(author_user)}",
            stage="កំពុងរៀបចំឯកសារ File...",
            percent=70,
            detail="កំពុងទាញយកទិន្នន័យច្បាស់ដើម (Full Quality)...",
        )

        try:
            animator.start()
            if images and msg:
                for idx, img_url in enumerate(images[:10]):
                    img_bytes = await download_media_bytes(img_url, max_bytes=10 * 1024 * 1024)
                    if img_bytes:
                        bio = io.BytesIO(img_bytes)
                        bio.name = f"tiktok_{clean_vid}_{idx + 1}.jpg"
                        await msg.reply_document(
                            document=bio,
                            filename=f"tiktok_{clean_vid}_{idx + 1}.jpg",
                            caption=caption if idx == 0 else None,
                            parse_mode="HTML",
                            disable_content_type_detection=True,
                        )
                return

            if play_url and msg:
                filename = f"tiktok_{clean_vid}.mp4"
                sent_doc = None
                tmp_doc_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
                        tmp_doc_path = tmp_f.name

                    b_doc = await download_media_bytes(play_url, max_bytes=TELEGRAM_MAX_UPLOAD_BYTES)
                    if b_doc:
                        with open(tmp_doc_path, "wb") as f:
                            f.write(b_doc)
                        downloaded_doc_bytes = len(b_doc)
                    else:
                        downloaded_doc_bytes = await download_media_to_file(
                            play_url,
                            tmp_doc_path,
                            max_bytes=TELEGRAM_MAX_UPLOAD_BYTES,
                            timeout_s=180.0,
                        )

                    if downloaded_doc_bytes and os.path.exists(tmp_doc_path) and os.path.getsize(tmp_doc_path) > 0:
                        size_mb = os.path.getsize(tmp_doc_path) / (1024 * 1024)
                        doc_caption = (
                            f"📁 <b>ឯកសារ TikTok (Document File)</b>\n"
                            f"👤 <b>អ្នកបង្កើត:</b> @{html.escape(author_user)}\n"
                            f"📝 {html.escape(title[:250])}\n"
                            f"📦 <b>ទំហំ:</b> {size_mb:.1f} MB | <b>ប្រភេទ:</b> MP4 HD (Full Quality)"
                        )[:1020]
                        with open(tmp_doc_path, "rb") as df:
                            sent_doc = await msg.reply_document(
                                document=df,
                                filename=filename,
                                caption=doc_caption,
                                parse_mode="HTML",
                                disable_content_type_detection=True,
                            )
                    elif play_url:
                        try:
                            sent_doc = await msg.reply_document(
                                document=play_url,
                                filename=filename,
                                caption=caption,
                                parse_mode="HTML",
                                disable_content_type_detection=True,
                            )
                        except Exception as url_doc_err:
                            logger.debug("Direct URL document fallback failed: %s", url_doc_err)
                finally:
                    if tmp_doc_path and os.path.exists(tmp_doc_path):
                        with suppress(Exception):
                            os.unlink(tmp_doc_path)

                if sent_doc:
                    doc_file_id = getattr(getattr(sent_doc, "document", None), "file_id", None)
                    if doc_file_id:
                        cached["doc_file_id"] = doc_file_id
                        cache_tiktok(clean_vid, cached)
                    return
        finally:
            await animator.stop()
            if status_msg and hasattr(status_msg, "delete"):
                with suppress(Exception):
                    await status_msg.delete()

        with suppress(Exception):
            await query.answer("❌ មិនអាចទាញយកឯកសារ File បានទេ។", show_alert=True)
    finally:
        with _IN_FLIGHT_LOCK:
            _IN_FLIGHT_TASKS.discard(task_key)


async def handle_tiktok_ai_summary(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Use Gemini AI to summarize or translate TikTok video topic into natural Khmer."""
    clean_vid = str(video_id).strip()
    cached = get_cached_tiktok(clean_vid) or {}
    if not cached or not cached.get("title"):
        with suppress(Exception):
            fresh = await fetch_tiktok_data(f"https://www.tiktok.com/@tiktok/video/{clean_vid}")
            if fresh:
                cached.update(fresh)
                cache_tiktok(clean_vid, cached)

    title = cached.get("title") or ""
    author = cached.get("author_username") or cached.get("author_name") or ""

    if not title.strip():
        with suppress(Exception):
            await query.answer("⚠️ វីដេអូនេះគ្មាន Caption សម្រាប់សង្ខេបទេ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer("🤖 AI កំពុងវិភាគខ្លឹមសារ...")

    gemini_client = None
    with suppress(Exception):
        from app.services.ai.gemini import get_gemini_client
        gemini_client = get_gemini_client()

    if not gemini_client:
        with suppress(Exception):
            from app import legacy
            gemini_client = getattr(legacy, "_gemini", None)

    if not gemini_client:
        with suppress(Exception):
            await query.answer("⚠️ សេវា AI មិនទាន់ត្រូវបានកំណត់ទេ។", show_alert=True)
        return

    from app.services.ai.gemini import extract_gemini_text, generate_content_with_fallback

    ai_prompt = (
        f"You are a helpful Cambodian AI assistant. Summarize and explain the key points of this TikTok video content "
        f"clearly, concisely, and naturally in natural Khmer:\n\n"
        f"Author: @{author}\n"
        f"Caption / Content: {title}\n\n"
        f"Keep the summary well-structured with 2-3 bullet points."
    )

    status_msg = None
    if query.message and hasattr(query.message, "reply_text"):
        with suppress(Exception):
            status_msg = await query.message.reply_text(
                "🤖 <b>AI TIKTOK SUMMARIZER</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 <b>អ្នកបង្កើត:</b> @{html.escape(author)}\n"
                "⚡ ម៉ូដែល: <code>Gemini 2.5 Flash</code>\n"
                "💡 <i>កំពុងស្តាប់ និងវិភាគខ្លឹមសារវីដេអូ ⠋</i>",
                parse_mode="HTML",
            )

    from app.services.telegram.formatters import StatusCardAnimator

    animator = StatusCardAnimator(
        status_msg,
        title="AI TIKTOK SUMMARIZER",
        icon="🤖",
        header_extra=f"👤 <b>អ្នកបង្កើត:</b> @{html.escape(author)}\n⚡ ម៉ូដែល: <code>Gemini 2.5 Flash</code>",
        detail="កំពុងស្តាប់ និងវិភាគខ្លឹមសារវីដេអូ",
    )

    summary = None
    try:
        animator.start()
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: generate_content_with_fallback(
                gemini_client,
                contents=ai_prompt,
                preferred_model="gemini-2.5-flash",
            ),
        )
        summary = extract_gemini_text(resp)
    except Exception as exc:
        logger.warning("TikTok AI summary error: %s", exc)
    finally:
        await animator.stop()
        if status_msg and hasattr(status_msg, "delete"):
            with suppress(Exception):
                await status_msg.delete()

    if summary and query.message:
        from app.services.telegram.formatters import markdown_to_telegram_html

        text = (
            f"🤖 <b>AI សង្ខេបខ្លឹមសារ TikTok (@{html.escape(author)}):</b>\n\n"
            f"{markdown_to_telegram_html(summary)}"
        )
        await query.message.reply_text(text, parse_mode="HTML")
    else:
        with suppress(Exception):
            await query.answer("❌ មិនអាចទាញយកការសង្ខេប AI បានទេ!", show_alert=True)


async def handle_tiktok_stats(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Display rich engagement analytics modal for TikTok post."""
    clean_vid = str(video_id).strip()
    cached = get_cached_tiktok(clean_vid) or {}
    if not cached:
        with suppress(Exception):
            fresh = await fetch_tiktok_data(f"https://www.tiktok.com/@tiktok/video/{clean_vid}")
            if fresh:
                cached.update(fresh)
                cache_tiktok(clean_vid, cached)

    if not cached:
        with suppress(Exception):
            await query.answer("⚠️ មិនមានទិន្នន័យស្ថិតិសម្រាប់មេឌៀនេះទេ។", show_alert=True)
        return

    author_user = cached.get("author_username") or "creator"
    author_name = cached.get("author_name") or ""
    views = cached.get("views", 0)
    likes = cached.get("likes", 0)
    comments = cached.get("comments", 0)
    shares = cached.get("shares", 0)
    downloads = cached.get("downloads", 0)
    duration = cached.get("duration", 0)
    music = cached.get("music_title") or "Original Sound"
    size_bytes = cached.get("size", 0)
    size_str = f"{size_bytes / (1024 * 1024):.1f} MB" if size_bytes > 0 else "N/A"

    author_display = f"@{author_user}"
    if author_name and author_name != author_user:
        author_display += f" ({author_name})"

    stats_lines = [
        f"📊 ស្ថិតិមេឌៀ TikTok ({author_display})",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"👁️ ចំនួនទស្សនា: {views:,} ដង",
        f"❤️ ចំនួន Likes: {likes:,} នាក់",
        f"💬 ចំនួនមតិ: {comments:,} មតិ",
        f"🔁 ចំនួនចែករំលែក: {shares:,} ដង",
    ]
    if downloads > 0:
        stats_lines.append(f"📥 ចំនួនទាញយក: {downloads:,} ដង")
    if duration > 0:
        stats_lines.append(f"⏱️ រយៈពេលវីដេអូ: {duration} វិនាទី")
    if size_str != "N/A":
        stats_lines.append(f"📦 ទំហំឯកសារ: {size_str}")
    stats_lines.extend([
        f"🎼 បទភ្លេង: {music}",
        f"🆔 ID: {clean_vid}",
    ])

    stats_text = "\n".join(stats_lines)

    with suppress(Exception):
        await query.answer(stats_text, show_alert=True)


async def cmd_tiktok(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command handler for /tiktok and /tt."""
    msg = update.effective_message
    if not msg:
        return

    from app.core.features import is_tiktok_enabled
    if not is_tiktok_enabled():
        await msg.reply_text("⚠️ មុខងារទាញយក TikTok ត្រូវបានបិទដំណើរការ (TikTok Feature Disabled)។")
        return

    args = context.args or []
    if not args:
        reply_markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🏠 ម៉ឺនុយដើម", callback_data="welcome_menu"),
                InlineKeyboardButton("❌ បិទ", callback_data="close_msg"),
            ]
        ])
        await msg.reply_text(
            "📥 <b>របៀបប្រើប្រាស់ TikTok Downloader</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "ផ្ញើតំណភ្ជាប់វីដេអូ TikTok ចូលក្នុង Chat ឬវាយ:\n"
            "<code>/tiktok https://vt.tiktok.com/...</code>\n\n"
            "✨ <b>លក្ខណៈពិសេស:</b>\n"
            "• 📹 វីដេអូកម្រិតច្បាស់ HD គ្មាន Watermark\n"
            "• 📁 ទាញយកជាឯកសារច្បាស់ដើម (Document File)\n"
            "• 🎵 ទាញយកតែសំឡេងដើមជា MP3 Audio\n"
            "• 🖼️ ទាញយករូបភាព Slide ទាំងអស់\n"
            "• 📊 ស្ថិតិវីដេអូ & Engagement Metrics\n"
            "• 🤖 AI សង្ខេបខ្លឹមសារវីដេអូ\n"
            "━━━━━━━━━━━━━━━━━━━━━━",
            reply_markup=reply_markup,
            parse_mode="HTML",
        )
        return

    url = " ".join(args).strip()
    try:
        await handle_tiktok_download(update, context, url)
    except Exception as exc:
        logger.error("Error in cmd_tiktok: %s", exc, exc_info=True)
        with suppress(Exception):
            await msg.reply_text(f"❌ បរាជ័យក្នុងការទាញយក TikTok: {exc}")


async def tiktok_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Callback router for TikTok buttons (tt_mp3:<id>, tt_file:<id>, tt_ai:<id>, tt_stats:<id>)."""
    query = update.callback_query
    if not query or not query.data:
        return

    try:
        from app.core.features import is_tiktok_enabled
        if not is_tiktok_enabled():
            with suppress(Exception):
                await query.answer("⚠️ មុខងារទាញយក TikTok ត្រូវបានបិទដំណើរការ។", show_alert=True)
            return

        data = query.data

        if data.startswith("tt_mp3:"):
            video_id = data.split(":", 1)[1]
            await handle_tiktok_mp3_download(query, context, video_id)
            return

        if data.startswith("tt_file:"):
            video_id = data.split(":", 1)[1]
            await handle_tiktok_file_download(query, context, video_id)
            return

        if data.startswith("tt_ai:"):
            video_id = data.split(":", 1)[1]
            await handle_tiktok_ai_summary(query, context, video_id)
            return

        if data.startswith("tt_stats:"):
            video_id = data.split(":", 1)[1]
            await handle_tiktok_stats(query, context, video_id)
            return

        with suppress(Exception):
            await query.answer()
    except Exception as exc:
        logger.error("tiktok_callback error: %s", exc, exc_info=True)
        with suppress(Exception):
            await query.answer("⚠️ មានបញ្ហាក្នុងការដំណើរការ TikTok។ សូមព្យាយាមម្ដងទៀត។", show_alert=True)


__all__ = [
    "cache_tiktok",
    "clear_tiktok_cache",
    "cmd_tiktok",
    "download_media_bytes",
    "download_media_to_file",
    "extract_tiktok_url",
    "fetch_tiktok_data",
    "get_cached_tiktok",
    "get_tiktok_video_kb",
    "handle_tiktok_ai_summary",
    "handle_tiktok_download",
    "handle_tiktok_file_download",
    "handle_tiktok_mp3_download",
    "handle_tiktok_stats",
    "is_tiktok_url",
    "tiktok_callback",
]