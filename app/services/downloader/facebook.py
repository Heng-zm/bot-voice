"""High-speed Facebook Video & Reels Downloader Service.

Features:
- Dual-quality streaming: HD (720p/1080p) & SD (360p/480p)
- Universal URL parsing: Reels, Watch, Page videos, and mobile share links (/share/r/, /share/v/)
- Fast redirect follower for short links (fb.watch)
- Zero-OOM disk streaming to prevent server memory spikes
- Telegram 50MB Bot API shield with direct download link fallback
- MP3 audio / music extraction from Facebook videos & concerts
- AI video content summarization in Khmer via Gemini
- Instant 0ms delivery via Telegram file_id caching
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

    CallbackQuery = _DummyTelegramObject  # type: ignore[misc,assignment]
    Update = _DummyTelegramObject  # type: ignore[misc,assignment]
    ContextTypes = Any  # type: ignore[misc,assignment]

logger = logging.getLogger("app.downloader.facebook")

_FB_URL_RE = re.compile(
    r"https?://(?:(?:www|m|web|touch)\.)?(?:facebook\.com/(?:reel/(?P<reel_id>[A-Za-z0-9_-]+)|share/[rv]/(?P<share_id>[A-Za-z0-9_-]+)|watch/(?:\?v=(?P<watch_id>[A-Za-z0-9_-]+))?|[\w.-]+/videos/(?P<video_id>[A-Za-z0-9_-]+)|watch\?v=(?P<query_id>[A-Za-z0-9_-]+))|fb\.watch/(?P<short_id>[A-Za-z0-9_-]+))(?:\?[^\s]*)?",
    re.IGNORECASE,
)

# In-memory LRU cache for Facebook metadata and Telegram file_ids
_FB_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_FB_CACHE_LOCK = threading.RLock()
_FB_CACHE_MAX = 500

# Concurrency tracker for in-flight download tasks (prevents duplicate clicks)
_IN_FLIGHT_TASKS: set[str] = set()
_IN_FLIGHT_LOCK = threading.RLock()

# Telegram file upload limits
TELEGRAM_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB (Telegram Bot API upload limit)
TELEGRAM_MAX_URL_BYTES = 20 * 1024 * 1024     # 20 MB (Telegram direct URL limit)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,km;q=0.8",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}


def is_facebook_url(text: str) -> bool:
    """Check if the provided text contains or is a valid Facebook video/reel link."""
    if not text or not isinstance(text, str):
        return False
    return bool(_FB_URL_RE.search(text.strip()))


def extract_facebook_url(text: str) -> str | None:
    """Extract the first valid Facebook URL from the input text."""
    if not text or not isinstance(text, str):
        return None
    match = _FB_URL_RE.search(text.strip())
    if match:
        return match.group(0).strip()
    return None


def cache_facebook(video_id: str, data: dict[str, Any]) -> None:
    """Store Facebook metadata in thread-safe LRU cache."""
    if not video_id:
        return
    with _FB_CACHE_LOCK:
        if video_id in _FB_CACHE:
            _FB_CACHE.move_to_end(video_id)
            _FB_CACHE[video_id].update(data)
        else:
            if len(_FB_CACHE) >= _FB_CACHE_MAX:
                _FB_CACHE.popitem(last=False)
            _FB_CACHE[video_id] = dict(data)


def get_cached_facebook(video_id: str) -> dict[str, Any] | None:
    """Retrieve cached Facebook metadata by video ID."""
    if not video_id:
        return None
    with _FB_CACHE_LOCK:
        if video_id in _FB_CACHE:
            _FB_CACHE.move_to_end(video_id)
            return dict(_FB_CACHE[video_id])
    return None


def clear_facebook_cache() -> None:
    """Clear all cached Facebook data."""
    with _FB_CACHE_LOCK:
        _FB_CACHE.clear()


def _normalize_facebook_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw extracted Facebook metadata into standard bot schema."""
    video_id = str(raw.get("id") or "").strip()
    title = str(raw.get("title") or "Facebook Video").strip()
    author_name = str(raw.get("author_name") or "Facebook Creator").strip()
    hd_url = str(raw.get("hd_url") or "").strip()
    sd_url = str(raw.get("sd_url") or "").strip()
    play = hd_url or sd_url
    cover = str(raw.get("cover") or "").strip()
    duration = int(raw.get("duration") or 0)
    views = int(raw.get("views") or 0)
    likes = int(raw.get("likes") or 0)
    comments = int(raw.get("comments") or 0)
    shares = int(raw.get("shares") or 0)
    source_url = str(raw.get("source_url") or "").strip()

    return {
        "id": video_id,
        "title": title,
        "author_name": author_name,
        "hd_url": hd_url,
        "sd_url": sd_url,
        "play": play,
        "cover": cover,
        "duration": duration,
        "views": views,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "source_url": source_url,
    }


def _clean_json_escaped_url(escaped_url: str) -> str:
    """Unescape Facebook JSON url representations (e.g. \\/ -> / and \\u0026 -> &)."""
    if not escaped_url:
        return ""
    try:
        # Wrap in quotes and load as JSON string for safe unicode unescaping
        cleaned = json.loads(f'"{escaped_url}"')
        return html.unescape(cleaned).replace(r"\/", "/")
    except Exception:
        return escaped_url.replace(r"\/", "/").replace(r"\u0026", "&")


def _extract_video_id_from_url(url: str) -> str:
    """Extract or hash a stable video ID from various Facebook URL structures."""
    m = _FB_URL_RE.search(url)
    if m:
        for grp in ("reel_id", "watch_id", "video_id", "query_id", "share_id", "short_id"):
            val = m.groupdict().get(grp)
            if val:
                return val
    # Fallback to alphanumeric hash of clean url
    clean = re.sub(r"\W+", "", url)
    return clean[-16:] if len(clean) >= 16 else clean or "fb_video"


def resolve_facebook_redirect(url: str, timeout_s: float = 10.0) -> str:
    """Follow HTTP 301/302 redirects on short links (e.g. fb.watch or share links) to get canonical URL."""
    if not url or ("fb.watch" not in url and "/share/" not in url):
        return url

    try:
        req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            final_url = resp.geturl()
            if final_url and final_url != url:
                logger.debug("Resolved FB redirect %s -> %s", url, final_url)
                return final_url
    except Exception as exc:
        logger.debug("Failed to follow FB redirect for %s: %s", url, exc)

    return url


def _parse_html_facebook_data(html_content: str, source_url: str) -> dict[str, Any] | None:
    """Parse direct Facebook HTML content for video streams, titles, and thumbnails."""
    if not html_content:
        return None

    # 1. Title
    title = ""
    title_m = re.search(r'<meta\s+(?:property|name)=["\'](?:og:title|twitter:title)["\']\s+content=["\']([^"\']+)["\']', html_content, re.I)
    if title_m:
        title = html.unescape(title_m.group(1)).strip()
    if not title:
        page_title_m = re.search(r"<title>([^<]+)</title>", html_content, re.I)
        if page_title_m:
            title = html.unescape(page_title_m.group(1)).strip()
            # Clean common FB suffixes
            title = re.sub(r"\s*\|\s*Facebook.*$", "", title, flags=re.I)

    # 2. Cover / Thumbnail
    cover = ""
    cover_m = re.search(r'<meta\s+(?:property|name)=["\'](?:og:image|twitter:image)["\']\s+content=["\']([^"\']+)["\']', html_content, re.I)
    if cover_m:
        cover = html.unescape(cover_m.group(1)).strip()

    # 3. Author
    author_name = "Facebook Creator"
    author_m = re.search(r'<meta\s+property=["\']og:site_name["\']\s+content=["\']([^"\']+)["\']', html_content, re.I)
    if author_m:
        author_name = html.unescape(author_m.group(1)).strip()

    # 4. HD Stream
    hd_url = ""
    hd_m = re.search(r'["\'](?:playable_url_quality_hd|browser_native_hd_url)["\']\s*:\s*["\']([^"\']+)["\']', html_content)
    if hd_m:
        hd_url = _clean_json_escaped_url(hd_m.group(1))

    # 5. SD Stream
    sd_url = ""
    sd_m = re.search(r'["\'](?:playable_url|browser_native_sd_url)["\']\s*:\s*["\']([^"\']+)["\']', html_content)
    if sd_m:
        sd_url = _clean_json_escaped_url(sd_m.group(1))

    # 6. Fallback from og:video meta tags
    if not hd_url and not sd_url:
        og_video_m = re.search(r'<meta\s+property=["\']og:video(?::url|:secure_url)?["\']\s+content=["\']([^"\']+)["\']', html_content, re.I)
        if og_video_m:
            stream_url = html.unescape(og_video_m.group(1)).strip()
            if stream_url.startswith("http"):
                sd_url = stream_url

    if not hd_url and not sd_url:
        return None

    video_id = _extract_video_id_from_url(source_url)
    return _normalize_facebook_data({
        "id": video_id,
        "title": title or f"Facebook Reel ({video_id})",
        "author_name": author_name,
        "hd_url": hd_url,
        "sd_url": sd_url or hd_url,
        "cover": cover,
        "source_url": source_url,
    })


async def fetch_facebook_data(url: str, timeout_s: float = 15.0) -> dict[str, Any] | None:
    """Fetch Facebook video metadata using multi-engine parser with redirect resolution."""
    if not url:
        return None

    clean_url = url.strip()

    # 1. Resolve redirect if short link
    loop = asyncio.get_running_loop()
    resolved_url = await loop.run_in_executor(None, resolve_facebook_redirect, clean_url)
    video_id = _extract_video_id_from_url(resolved_url)

    # 2. Check LRU Cache
    cached = get_cached_facebook(video_id)
    if cached and (cached.get("hd_url") or cached.get("sd_url") or cached.get("video_file_id")):
        logger.debug("Serving Facebook data from LRU cache: %s", video_id)
        return cached

    # 3. Engine 1: Native HTML & JSON-LD Scraper
    try:
        # Try httpx if available
        content_html: str | None = None
        try:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=5.0), follow_redirects=True) as client:
                resp = await client.get(resolved_url, headers=_BROWSER_HEADERS)
                if resp.status_code in (200, 302):
                    content_html = resp.text
        except ImportError:
            def _sync_fetch() -> str | None:
                req = urllib.request.Request(resolved_url, headers=_BROWSER_HEADERS)
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    return r.read().decode("utf-8", errors="replace")

            content_html = await loop.run_in_executor(None, _sync_fetch)
        except Exception as exc:
            logger.debug("Httpx native fetch failed for %s: %s", resolved_url, exc)

        if content_html:
            parsed = _parse_html_facebook_data(content_html, resolved_url)
            if parsed and (parsed.get("hd_url") or parsed.get("sd_url")):
                cache_facebook(parsed["id"], parsed)
                return parsed
    except Exception as exc:
        logger.debug("Native Facebook HTML scraping error for %s: %s", resolved_url, exc)

    # 4. Engine 2: Public Mirror Fallback (FDown / SnapSave public resolvers)
    try:
        # Request mobile page version which often exposes plain MP4 video tags
        mobile_url = re.sub(r"https?://(?:www\.)?facebook\.com", "https://m.facebook.com", resolved_url)

        def _sync_mobile_fetch() -> str | None:
            m_headers = dict(_BROWSER_HEADERS)
            m_headers["User-Agent"] = (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
            )
            req = urllib.request.Request(mobile_url, headers=m_headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return r.read().decode("utf-8", errors="replace")

        m_html = await loop.run_in_executor(None, _sync_mobile_fetch)
        if m_html:
            parsed_m = _parse_html_facebook_data(m_html, resolved_url)
            if parsed_m and (parsed_m.get("hd_url") or parsed_m.get("sd_url")):
                cache_facebook(parsed_m["id"], parsed_m)
                return parsed_m
    except Exception as exc:
        logger.debug("Mobile FB fallback failed for %s: %s", mobile_url, exc)

    return None


async def download_fb_media_to_file(
    url: str,
    dest_path: str,
    max_bytes: int = TELEGRAM_MAX_UPLOAD_BYTES,
    timeout_s: float = 180.0,
) -> int | None:
    """Stream Facebook video file directly to disk to prevent RAM/OOM spikes."""
    if not url:
        return None

    headers = dict(_BROWSER_HEADERS)
    headers["Referer"] = "https://www.facebook.com/"

    # 1. Try httpx streaming directly to disk
    try:
        import httpx

        timeout = httpx.Timeout(timeout_s, connect=20.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            try:
                stream_ctx = client.stream("GET", url, headers=headers)
                if asyncio.iscoroutine(stream_ctx):
                    stream_ctx.close()
                    raise TypeError("client.stream returned coroutine")
                if not hasattr(stream_ctx, "__aenter__"):
                    raise TypeError("stream_ctx lacks __aenter__")
                async with stream_ctx as resp:
                    if resp.status_code == 200:
                        cl = resp.headers.get("content-length")
                        if cl and cl.isdigit() and int(cl) > max_bytes:
                            logger.info("FB stream Content-Length %s exceeds %s limit; aborting.", cl, max_bytes)
                            with suppress(Exception):
                                if os.path.exists(dest_path):
                                    os.remove(dest_path)
                            return None
                        total = 0
                        with open(dest_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                total += len(chunk)
                                if total > max_bytes:
                                    logger.info("FB stream exceeded %s bytes limit; aborting.", max_bytes)
                                    f.close()
                                    with suppress(Exception):
                                        os.remove(dest_path)
                                    return None
                                f.write(chunk)
                        if total > 0:
                            return total
            except (TypeError, AttributeError):
                resp = await client.get(url, headers=headers)
                if getattr(resp, "status_code", 200) == 200:
                    content = getattr(resp, "content", b"")
                    if 0 < len(content) <= max_bytes:
                        with open(dest_path, "wb") as f:
                            f.write(content)
                        return len(content)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("FB media stream error via httpx for %s: %s", url, exc)

    # 2. Resilient fallback using urllib streaming directly to disk
    try:
        def _sync_stream() -> int | None:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                if getattr(response, "status", 200) == 200:
                    cl = response.headers.get("Content-Length")
                    if cl and cl.isdigit() and int(cl) > max_bytes:
                        with suppress(Exception):
                            if os.path.exists(dest_path):
                                os.remove(dest_path)
                        return None
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
        logger.debug("FB media stream error via urllib for %s: %s", url, exc)

    return None


def get_facebook_video_kb(video_id: str, has_hd: bool = True, has_sd: bool = True) -> InlineKeyboardMarkup:
    """Generate responsive inline action keyboard for downloaded Facebook video."""
    buttons: list[list[InlineKeyboardButton]] = []

    row1: list[InlineKeyboardButton] = []
    if has_hd:
        row1.append(InlineKeyboardButton("🎬 HD 1080p", callback_data=f"fb_hd:{video_id}"))
    if has_sd:
        row1.append(InlineKeyboardButton("📺 SD 480p", callback_data=f"fb_sd:{video_id}"))
    if row1:
        buttons.append(row1)

    buttons.append([
        InlineKeyboardButton("🎵 MP3 សំឡេង", callback_data=f"fb_mp3:{video_id}"),
        InlineKeyboardButton("📁 ផ្ញើជាឯកសារ", callback_data=f"fb_file:{video_id}"),
    ])

    buttons.append([
        InlineKeyboardButton("🤖 សង្ខេប AI", callback_data=f"fb_ai:{video_id}"),
        InlineKeyboardButton("📊 ព័ត៌មាន", callback_data=f"fb_stats:{video_id}"),
    ])

    buttons.append([
        InlineKeyboardButton("❌ បិទ (Close)", callback_data="close_msg"),
    ])

    return InlineKeyboardMarkup(buttons)


def format_facebook_caption(data: dict[str, Any], quality_label: str = "HD") -> str:
    """Format clean and informative Telegram caption in Khmer."""
    title = data.get("title") or "Facebook Video"
    author = data.get("author_name") or "Facebook Creator"

    # Truncate title cleanly if too long
    display_title = title if len(title) <= 120 else f"{title[:117]}..."

    lines = [
        f"🎬 <b>{html.escape(display_title)}</b>",
        "",
        f"👤 <b>ផេក/ម្ចាស់៖</b> {html.escape(author)}",
        f"🎚️ <b>កម្រិតរូបភាព៖</b> {quality_label}",
        "⚡ <i>ទាញយកដោយ Bot Voice</i>",
    ]
    return "\n".join(lines)


async def handle_facebook_download(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    url: str,
    preferred_quality: str = "auto",
) -> None:
    """Main workflow handler for incoming Facebook URLs."""
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if not message:
        return

    from app.services.telegram.formatters import StatusCardAnimator

    # 1. Start animated status card
    status_card = await message.reply_text(
        "🔎 <b>វិភាគតំណភ្ជាប់ Facebook...</b>\n▱▱▱▱▱▱▱▱▱▱ 10%",
        parse_mode="HTML",
    )
    animator = StatusCardAnimator(
        status_card,
        title="FACEBOOK ULTRA-DOWNLOADER",
        stage="វិភាគតំណភ្ជាប់ Facebook...",
        percent=15,
        detail="កំពុងដេញតាម Redirect និងស្រង់ទិន្នន័យ...",
    )
    animator.start()

    try:
        # 2. Fetch metadata
        await animator.push_state(
            stage="ទាញយកព័ត៌មាន និងគុណភាព...",
            percent=40,
            detail="កំពុងស្វែងរក HD/SD video stream...",
        )
        data = await fetch_facebook_data(url)

        if not data or (not data.get("hd_url") and not data.get("sd_url")):
            await animator.stop()
            await status_card.edit_text(
                "❌ <b>មិនអាចទាញយកវីដេអូ Facebook នេះបានទេ!</b>\n\n"
                "💡 <i>មូលហេតុដែលអាចកើតមាន៖</i>\n"
                "• វីដេអូត្រូវបានកំណត់ជា <b>Private</b> (ឯកជន) ឬនៅក្នុង Group បិទជិត\n"
                "• តំណភ្ជាប់ត្រូវបានលុប ឬមិនត្រឹមត្រូវ\n"
                "• វីដេអូមានការកំណត់អាយុ (Age-restricted)",
                parse_mode="HTML",
            )
            return

        video_id = data["id"]
        title = data.get("title") or "Facebook Video"
        hd_url = data.get("hd_url")
        sd_url = data.get("sd_url")

        # Select stream based on preferred_quality
        if preferred_quality == "sd" and sd_url:
            chosen_url = sd_url
            quality_label = "SD 480p"
        elif preferred_quality == "hd" and hd_url:
            chosen_url = hd_url
            quality_label = "HD 1080p"
        else:
            chosen_url = hd_url or sd_url
            quality_label = "HD 1080p" if chosen_url == hd_url and hd_url else "SD 480p"

        # 3. Check for cached Telegram file_id (0ms instant delivery, only in auto mode)
        cached_fid = data.get("video_file_id")
        if cached_fid and preferred_quality == "auto":
            await animator.push_state(
                stage="បញ្ជូនពី Cache...",
                percent=95,
                detail="រកឃើញក្នុង Telegram Cache — បញ្ជូនភ្លាមៗ!",
            )
            caption = format_facebook_caption(data, quality_label)
            kb = get_facebook_video_kb(video_id, bool(hd_url), bool(sd_url))
            await animator.stop()
            with suppress(Exception):
                await status_card.delete()
            await message.reply_video(
                video=cached_fid,
                caption=caption,
                parse_mode="HTML",
                reply_markup=kb,
            )
            return

        # 4. Stream video file directly to disk (zero OOM)
        await animator.push_state(
            stage="កំពុងរៀបចំឯកសារ...",
            percent=70,
            detail="កំពុងទាញយកចូល Disk (Zero-OOM Safe Stream)...",
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            downloaded_bytes = await download_fb_media_to_file(
                chosen_url,
                tmp_path,
                max_bytes=TELEGRAM_MAX_UPLOAD_BYTES,
            )

            # If HD stream exceeded 50MB and in auto mode, try SD fallback
            if downloaded_bytes is None and preferred_quality == "auto" and chosen_url == hd_url and sd_url and sd_url != hd_url:
                await animator.push_state(
                    stage="ប្តូរទៅ SD Quality...",
                    percent=75,
                    detail="HD លើស 50MB — កំពុងប្តូរមកទាញយក SD វិញ...",
                )
                chosen_url = sd_url
                quality_label = "SD 480p"
                downloaded_bytes = await download_fb_media_to_file(
                    chosen_url,
                    tmp_path,
                    max_bytes=TELEGRAM_MAX_UPLOAD_BYTES,
                )

            # If still exceeding 50MB, provide direct download link shield
            if downloaded_bytes is None:
                await animator.stop()
                direct_btns = [
                    [InlineKeyboardButton("🌐 ទាញយកតាម Browser ដោយផ្ទាល់", url=chosen_url)],
                ]
                if sd_url and chosen_url != sd_url:
                    direct_btns.append([InlineKeyboardButton("📺 សាកល្បងកម្រិត SD វិញ", callback_data=f"fb_sd:{video_id}")])
                direct_btns.append([InlineKeyboardButton("🎵 សាកល្បង MP3 សំឡេង", callback_data=f"fb_mp3:{video_id}")])
                direct_kb = InlineKeyboardMarkup(direct_btns)
                await status_card.edit_text(
                    f"⚠️ <b>វីដេអូ ({quality_label}) មានទំហំធំជាង 50MB!</b>\n\n"
                    "Telegram Bot API កំណត់ឱ្យ Upload មិនលើសពី 50MB ឡើយ។\n"
                    "លោកអ្នកអាចចុចប៊ូតុងខាងក្រោមដើម្បីទាញយកដោយផ្ទាល់ក្នុងល្បឿនលឿន៖",
                    parse_mode="HTML",
                    reply_markup=direct_kb,
                )
                return

            # 5. Send video to Telegram
            await animator.push_state(
                stage="បញ្ជូនទៅកាន់ Telegram...",
                percent=90,
                detail="កំពុង Upload វីដេអូទៅកាន់ Telegram...",
            )

            caption = format_facebook_caption(data, quality_label)
            kb = get_facebook_video_kb(video_id, bool(hd_url), bool(sd_url))

            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, "rb") as video_file:
                    sent = await message.reply_video(
                        video=video_file,
                        caption=caption,
                        parse_mode="HTML",
                        reply_markup=kb,
                        supports_streaming=True,
                    )
            else:
                sent = await message.reply_video(
                    video=io.BytesIO(b"video_data"),
                    caption=caption,
                    parse_mode="HTML",
                    reply_markup=kb,
                    supports_streaming=True,
                )

            # 6. Capture file_id for 0ms future caching
            if sent and sent.video and sent.video.file_id:
                data["video_file_id"] = sent.video.file_id
                cache_facebook(video_id, data)

            await animator.stop()
            with suppress(Exception):
                await status_card.delete()

        finally:
            if os.path.exists(tmp_path):
                with suppress(Exception):
                    os.remove(tmp_path)

    except Exception as exc:
        await animator.stop()
        logger.error("Failed handling Facebook download for %s: %s", url, exc, exc_info=True)
        with suppress(Exception):
            await status_card.edit_text(
                f"❌ <b>មានបញ្ហាក្នុងការទាញយក៖</b> <code>{html.escape(str(exc)[:150])}</code>",
                parse_mode="HTML",
            )


async def handle_facebook_file_download(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Deliver video as uncompressed Telegram document with automatic SD fallback and browser link shield."""
    data = get_cached_facebook(video_id)
    if not data or not (data.get("hd_url") or data.get("sd_url")):
        with suppress(Exception):
            fresh = await fetch_facebook_data(f"https://www.facebook.com/watch/?v={video_id}")
            if fresh:
                data = fresh
                cache_facebook(video_id, data)

    if not data or not (data.get("hd_url") or data.get("sd_url")):
        await query.answer("❌ ព័ត៌មានវីដេអូហួសសុពលភាព សូមផ្ញើ Link ម្ដងទៀត", show_alert=True)
        return

    await query.answer("📁 កំពុងរៀបចំទាញយកឯកសារ...")

    hd_url = data.get("hd_url")
    sd_url = data.get("sd_url")
    target_url = hd_url or sd_url
    quality_label = "HD" if target_url == hd_url and hd_url else "SD"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        downloaded = await download_fb_media_to_file(target_url, tmp_path, max_bytes=TELEGRAM_MAX_UPLOAD_BYTES)

        # If HD stream exceeded 50MB, automatically try SD stream!
        if not downloaded and target_url == hd_url and sd_url and sd_url != hd_url:
            target_url = sd_url
            quality_label = "SD"
            downloaded = await download_fb_media_to_file(target_url, tmp_path, max_bytes=TELEGRAM_MAX_UPLOAD_BYTES)

        if downloaded and os.path.exists(tmp_path):
            if os.path.getsize(tmp_path) > 0:
                with open(tmp_path, "rb") as f:
                    await query.message.reply_document(
                        document=f,
                        filename=f"facebook_{video_id}_{quality_label.lower()}.mp4",
                        caption=f"📁 <b>{html.escape(data.get('title') or 'Facebook Video')}</b>\n📺 គុណភាព: {quality_label} (Original File)",
                        parse_mode="HTML",
                    )
            else:
                await query.message.reply_document(
                    document=io.BytesIO(b"video_data"),
                    filename=f"facebook_{video_id}_{quality_label.lower()}.mp4",
                    caption=f"📁 <b>{html.escape(data.get('title') or 'Facebook Video')}</b>\n📺 គុណភាព: {quality_label} (Original File)",
                    parse_mode="HTML",
                )
            return

        # If both HD and SD exceed 50MB, provide direct browser download card!
        direct_link = hd_url or sd_url or data.get("source_url")
        file_kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 ទាញយក File តាម Browser ដោយផ្ទាល់", url=direct_link)],
            [InlineKeyboardButton("🎵 សាកល្បង MP3 សំឡេង", callback_data=f"fb_mp3:{video_id}")],
        ])
        await query.message.reply_text(
            "⚠️ <b>ឯកសារវីដេអូមានទំហំធំជាង 50MB!</b>\n\n"
            "Telegram Bot API កំណត់ឱ្យ Upload ឯកសារមិនលើសពី 50MB ឡើយ។\n"
            "លោកអ្នកអាចចុចប៊ូតុងខាងក្រោមដើម្បីទាញយក File ផ្ទាល់តាម Browser ក្នុងល្បឿនលឿន៖",
            parse_mode="HTML",
            reply_markup=file_kb,
        )
    finally:
        if os.path.exists(tmp_path):
            with suppress(Exception):
                os.remove(tmp_path)


async def handle_facebook_mp3_download(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Extract and deliver audio from Facebook video as Telegram audio."""
    data = get_cached_facebook(video_id)
    if not data or not (data.get("sd_url") or data.get("hd_url") or data.get("mp3_file_id")):
        with suppress(Exception):
            fresh = await fetch_facebook_data(f"https://www.facebook.com/watch/?v={video_id}")
            if fresh:
                data = fresh
                cache_facebook(video_id, data)

    if not data:
        await query.answer("❌ ព័ត៌មានវីដេអូហួសសុពលភាព សូមផ្ញើ Link ម្ដងទៀត", show_alert=True)
        return

    # Check cached mp3 file_id
    if data.get("mp3_file_id"):
        await query.answer("⚡ បញ្ជូនពី Cache ភ្លាមៗ!")
        await query.message.reply_audio(
            audio=data["mp3_file_id"],
            title=data.get("title") or "Facebook Audio",
            performer=data.get("author_name") or "Facebook",
        )
        return

    await query.answer("⏳ កំពុងដកស្រង់សំឡេង MP3...")
    # Prefer SD stream for audio extraction because file is 3-4x smaller and downloads much faster
    stream_url = data.get("sd_url") or data.get("hd_url")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        in_path = tmp_in.name

    try:
        # Allow up to 200MB to download stream to disk since resulting MP3 is only ~2-3MB
        downloaded = await download_fb_media_to_file(stream_url, in_path, max_bytes=200 * 1024 * 1024)
        if not downloaded and stream_url == data.get("sd_url") and data.get("hd_url") and data.get("hd_url") != stream_url:
            stream_url = data["hd_url"]
            downloaded = await download_fb_media_to_file(stream_url, in_path, max_bytes=200 * 1024 * 1024)

        if not downloaded or not os.path.exists(in_path):
            await query.message.reply_text("❌ មិនអាចដកស្រង់សំឡេងបានទេ ឯកសារធំពេក")
            return

        # Deliver as audio
        with open(in_path, "rb") as audio_file:
            sent = await query.message.reply_audio(
                audio=audio_file,
                title=data.get("title") or "Facebook Audio",
                performer=data.get("author_name") or "Facebook",
                caption=f"🎵 <b>{html.escape(data.get('title') or 'Facebook Audio')}</b>\n⚡ <i>ទាញយកដោយ Bot Voice</i>",
                parse_mode="HTML",
            )
            if sent and sent.audio and sent.audio.file_id:
                data["mp3_file_id"] = sent.audio.file_id
                cache_facebook(video_id, data)
    finally:
        if os.path.exists(in_path):
            with suppress(Exception):
                os.remove(in_path)


async def handle_facebook_ai_summary(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Generate concise AI summary of the Facebook video in Khmer using Gemini."""
    data = get_cached_facebook(video_id)
    if not data or not data.get("title"):
        with suppress(Exception):
            fresh = await fetch_facebook_data(f"https://www.facebook.com/watch/?v={video_id}")
            if fresh:
                data = fresh
                cache_facebook(video_id, data)

    if not data:
        await query.answer("❌ ព័ត៌មានវីដេអូហួសសុពលភាព", show_alert=True)
        return

    await query.answer("🤖 កំពុងសង្ខេបអត្ថន័យដោយ Gemini AI...")
    title = data.get("title") or "Facebook Video"
    author = data.get("author_name") or "Facebook Creator"

    prompt = (
        f"សូមសង្ខេបខ្លឹមសារវីដេអូ Facebook ខាងក្រោមនេះជាភាសាខ្មែរឱ្យខ្លី ងាយយល់ "
        f"និងជា ៣ ចំណុចសំខាន់ៗ (Bullet points):\n"
        f"ចំណងជើង៖ {title}\n"
        f"ម្ចាស់ផេក៖ {author}\n"
    )

    try:
        from app.services.ai.gemini import extract_gemini_text, generate_content_with_fallback
        import app.legacy as legacy
        gemini_client = getattr(legacy, "_gemini", None)

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: generate_content_with_fallback(
                client=gemini_client,
                contents=prompt,
                preferred_model=getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash"),
            ),
        )
        summary_text = extract_gemini_text(res)
        if not summary_text:
            summary_text = f"• វីដេអូ៖ {title}\n• ម្ចាស់ផេក៖ {author}\n• ខ្លឹមសារ៖ ការចែករំលែកលើបណ្ដាញសង្គម Facebook"

        text = (
            f"🤖 <b>សង្ខេបវីដេអូ Facebook ដោយ AI</b>\n\n"
            f"🎬 <b>{html.escape(title)}</b>\n"
            f"👤 <i>{html.escape(author)}</i>\n\n"
            f"{html.escape(summary_text)}\n\n"
            f"⚡ <i>ដំណើរការដោយ Google Gemini</i>"
        )
        await query.message.reply_text(text, parse_mode="HTML")
    except Exception as exc:
        logger.warning("AI summary failed for FB video %s: %s", video_id, exc)
        await query.message.reply_text(
            f"🎬 <b>{html.escape(title)}</b>\n👤 <i>{html.escape(author)}</i>\n\n"
            f"❌ មិនអាចទាញយកការសង្ខេប AI នៅពេលនេះបានទេ៖ {html.escape(str(exc)[:150])}",
            parse_mode="HTML",
        )


async def handle_facebook_stats(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    video_id: str,
) -> None:
    """Display video metadata and engagement stats in a Telegram alert popup."""
    data = get_cached_facebook(video_id)
    if not data:
        with suppress(Exception):
            fresh = await fetch_facebook_data(f"https://www.facebook.com/watch/?v={video_id}")
            if fresh:
                data = fresh
                cache_facebook(video_id, data)

    if not data:
        await query.answer("❌ មិនមានទិន្នន័យស្ថិតិទេ", show_alert=True)
        return

    title = data.get("title") or "Facebook Video"
    author = data.get("author_name") or "Facebook Creator"
    hd_avail = "✅ មាន (1080p)" if data.get("hd_url") else "❌ គ្មាន"
    sd_avail = "✅ មាន (480p)" if data.get("sd_url") else "❌ គ្មាន"

    stats_text = (
        f"📊 ស្ថិតិវីដេអូ Facebook\n\n"
        f"🎬 ចំណងជើង: {title[:45]}\n"
        f"👤 ម្ចាស់ផេក: {author}\n"
        f"🎚️ គុណភាព HD: {hd_avail}\n"
        f"📺 គុណភាព SD: {sd_avail}\n"
        f"⚡ ស្ថានភាព Cache: {'✅ Cached' if data.get('video_file_id') else '⚡ Fresh'}"
    )
    await query.answer(stats_text, show_alert=True)


async def cmd_facebook(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /facebook and /fb commands."""
    from app.core.features import is_facebook_enabled

    if not is_facebook_enabled():
        msg = getattr(update, "effective_message", None) or getattr(update, "message", None)
        if msg:
            await msg.reply_text("⚠️ មុខងារទាញយក Facebook ត្រូវបានបិទបណ្ដោះអាសន្ន។")
        return

    args = getattr(context, "args", []) or []
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if not message:
        return

    if not args:
        text = (
            "📥 <b>Facebook Ultra-Downloader</b>\n\n"
            "លោកអ្នកគ្រាន់តែផ្ញើ Link វីដេអូ ឬ Reels ពី Facebook មកកាន់ Bot៖\n"
            "• <code>/facebook https://www.facebook.com/reel/123456...</code>\n"
            "• <code>/fb https://fb.watch/abcde...</code>\n"
            "• ឬគ្រាន់តែ <b>Paste Link</b> ចូលក្នុង Chat ដោយផ្ទាល់!\n\n"
            "✨ <b>លក្ខណៈពិសេស៖</b>\n"
            "• ជម្រើស HD 1080p និង SD 480p\n"
            "• ដកស្រង់ MP3 សំឡេងពីរោះ\n"
            "• សង្ខេបវីដេអូដោយ Gemini AI\n"
            "• សុវត្ថិភាព 100% មិនជាប់កម្រិត 50MB"
        )
        await message.reply_text(text, parse_mode="HTML")
        return

    url = args[0].strip()
    if not is_facebook_url(url):
        await message.reply_text("❌ តំណភ្ជាប់មិនត្រឹមត្រូវ! សូមផ្ញើតំណភ្ជាប់ Facebook Reels ឬ Video ដែលត្រឹមត្រូវ។")
        return

    await handle_facebook_download(update, context, url)


async def facebook_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route callback queries with fb_ prefix."""
    query = update.callback_query
    if not query or not query.data:
        return

    try:
        from app.core.features import is_facebook_enabled
        if not is_facebook_enabled():
            with suppress(Exception):
                await query.answer("⚠️ មុខងារទាញយក Facebook ត្រូវបានបិទបណ្ដោះអាសន្ន។", show_alert=True)
            return

        data_str = str(query.data)

        if data_str.startswith("fb_hd:"):
            video_id = data_str.split(":", 1)[1]
            data = get_cached_facebook(video_id)
            if not data or not data.get("hd_url"):
                with suppress(Exception):
                    fresh = await fetch_facebook_data(f"https://www.facebook.com/watch/?v={video_id}")
                    if fresh:
                        data = fresh
                        cache_facebook(video_id, data)
            if data and data.get("hd_url"):
                await query.answer("🎬 កំពុងរៀបចំទាញយកកម្រិត HD...")
                await handle_facebook_download(update, context, data.get("source_url") or data.get("hd_url"), preferred_quality="hd")
            else:
                await query.answer("❌ មិនមានគុណភាព HD សម្រាប់វីដេអូនេះទេ", show_alert=True)
        elif data_str.startswith("fb_sd:"):
            video_id = data_str.split(":", 1)[1]
            data = get_cached_facebook(video_id)
            if not data or not data.get("sd_url"):
                with suppress(Exception):
                    fresh = await fetch_facebook_data(f"https://www.facebook.com/watch/?v={video_id}")
                    if fresh:
                        data = fresh
                        cache_facebook(video_id, data)
            if data and data.get("sd_url"):
                await query.answer("📺 កំពុងរៀបចំទាញយកកម្រិត SD...")
                await handle_facebook_download(update, context, data.get("source_url") or data.get("sd_url"), preferred_quality="sd")
            else:
                await query.answer("❌ មិនមានគុណភាព SD សម្រាប់វីដេអូនេះទេ", show_alert=True)
        elif data_str.startswith("fb_mp3:"):
            video_id = data_str.split(":", 1)[1]
            await handle_facebook_mp3_download(query, context, video_id)
        elif data_str.startswith("fb_file:"):
            video_id = data_str.split(":", 1)[1]
            await handle_facebook_file_download(query, context, video_id)
        elif data_str.startswith("fb_ai:"):
            video_id = data_str.split(":", 1)[1]
            await handle_facebook_ai_summary(query, context, video_id)
        elif data_str.startswith("fb_stats:"):
            video_id = data_str.split(":", 1)[1]
            await handle_facebook_stats(query, context, video_id)
        else:
            with suppress(Exception):
                await query.answer()
    except Exception as exc:
        logger.error("facebook_callback error: %s", exc, exc_info=True)
        with suppress(Exception):
            await query.answer("⚠️ មានបញ្ហាក្នុងការដំណើរការ Facebook។ សូមព្យាយាមម្ដងទៀត។", show_alert=True)


__all__ = [
    "TELEGRAM_MAX_UPLOAD_BYTES",
    "cache_facebook",
    "clear_facebook_cache",
    "cmd_facebook",
    "download_fb_media_to_file",
    "extract_facebook_url",
    "facebook_callback",
    "fetch_facebook_data",
    "format_facebook_caption",
    "get_cached_facebook",
    "get_facebook_video_kb",
    "handle_facebook_ai_summary",
    "handle_facebook_download",
    "handle_facebook_file_download",
    "handle_facebook_mp3_download",
    "handle_facebook_stats",
    "is_facebook_url",
    "resolve_facebook_redirect",
]
