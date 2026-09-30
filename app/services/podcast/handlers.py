"""Telegram handlers, interactive callbacks, and background scheduler for Daily Morning Podcast."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import threading
from typing import Any
import urllib.parse

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import ContextTypes
except ImportError:
    InlineKeyboardButton = Any  # type: ignore[assignment,misc]
    InlineKeyboardMarkup = Any  # type: ignore[assignment,misc]
    Update = Any  # type: ignore[assignment,misc]
    ContextTypes = Any  # type: ignore[assignment,misc]

from app.services.podcast.generator import generate_morning_podcast, get_cambodia_now
from app.services.podcast.store import podcast_store
from app.services.telegram._legacy_runtime import legacy_bound_handler, safe_send

logger = logging.getLogger(__name__)

_HANDLER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(_HANDLER_DIR.parent.parent.parent)
BANNER_PATH = os.path.join(PROJECT_ROOT, "assets", "morning_podcast_banner.jpg")
if not os.path.isfile(BANNER_PATH):
    BANNER_PATH = os.path.join(PROJECT_ROOT, "asset", "morning_podcast_banner.jpg")

_BANNER_LOCK = threading.Lock()
_BANNER_BYTES: bytes | None = None
_BANNER_FILE_ID: str | None = None
_SOURCE_BANNER_BYTES: bytes | None = None
_SOURCE_BANNER_FILE_ID: str | None = None
_SOURCE_BANNER_FETCHED_DATE: str | None = None


def set_source_banner_bytes(data: bytes | None) -> None:
    """Explicitly set or clear the fetched news source cover banner."""
    global _SOURCE_BANNER_BYTES, _SOURCE_BANNER_FILE_ID, _SOURCE_BANNER_FETCHED_DATE
    with _BANNER_LOCK:
        _SOURCE_BANNER_BYTES = data
        _SOURCE_BANNER_FILE_ID = None
        _SOURCE_BANNER_FETCHED_DATE = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d") if data else None


def get_source_banner_bytes() -> bytes | None:
    """Return currently cached source banner bytes if any."""
    now = datetime.now(timezone(timedelta(hours=7)))
    today_key = now.strftime("%Y-%m-%d")
    with _BANNER_LOCK:
        if _SOURCE_BANNER_BYTES is not None:
            if _SOURCE_BANNER_FETCHED_DATE is None or _SOURCE_BANNER_FETCHED_DATE == today_key:
                return _SOURCE_BANNER_BYTES
            return None
        return None


async def fetch_source_banner_bytes(source_url: str = "") -> bytes | None:
    """Fetch lead news cover image from Cambodia news source (ប្រភព).
    
    If source_url is provided, extracts its og:image / thumbnail.
    Otherwise, checks top stories from Cambodian news feeds.
    """
    global _SOURCE_BANNER_BYTES, _SOURCE_BANNER_FILE_ID, _SOURCE_BANNER_FETCHED_DATE
    now = datetime.now(timezone(timedelta(hours=7)))
    today_key = now.strftime("%Y-%m-%d")

    with _BANNER_LOCK:
        if not source_url and _SOURCE_BANNER_FETCHED_DATE == today_key and _SOURCE_BANNER_BYTES:
            return _SOURCE_BANNER_BYTES

    try:
        from app.services.ai.article_reader import (
            extract_article_content_with_image,
            fetch_article_html,
            fetch_image_bytes,
        )

        article_target = source_url.strip()
        if not article_target:
            import xml.etree.ElementTree as ET
            import urllib.request
            loop = asyncio.get_running_loop()

            def _fetch_rss_link() -> str | None:
                rss_url = "https://news.google.com/rss/search?q=Cambodia+news&hl=km&gl=KH&ceid=KH:km"
                req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    xml_data = resp.read()
                root = ET.fromstring(xml_data)
                items = root.findall("./channel/item")
                for item in items[:5]:
                    link = item.find("link")
                    if link is not None and link.text:
                        return link.text.strip()
                return None

            article_target = await loop.run_in_executor(None, _fetch_rss_link) or ""

        if article_target:
            html_data = await fetch_article_html(article_target, timeout_s=8.0)
            _, _, img_url = extract_article_content_with_image(html_data, base_url=article_target)
            if img_url:
                img_bytes = await fetch_image_bytes(img_url, timeout_s=6.0)
                if img_bytes and len(img_bytes) > 2048:
                    with _BANNER_LOCK:
                        _SOURCE_BANNER_BYTES = img_bytes
                        _SOURCE_BANNER_FILE_ID = None
                        _SOURCE_BANNER_FETCHED_DATE = today_key
                    logger.info("Successfully fetched podcast cover banner from news source (%s bytes)", len(img_bytes))
                    return img_bytes
    except Exception as exc:
        logger.debug("Fetch source banner from news source failed, using default: %s", exc)

    return None


def get_banner_bytes() -> bytes | None:
    """Read and cache banner bytes in memory. Prefers source banner if available."""
    global _BANNER_BYTES, _SOURCE_BANNER_BYTES, _SOURCE_BANNER_FILE_ID, _SOURCE_BANNER_FETCHED_DATE
    now = datetime.now(timezone(timedelta(hours=7)))
    today_key = now.strftime("%Y-%m-%d")
    with _BANNER_LOCK:
        if _SOURCE_BANNER_BYTES is not None:
            if _SOURCE_BANNER_FETCHED_DATE is None or _SOURCE_BANNER_FETCHED_DATE == today_key:
                return _SOURCE_BANNER_BYTES
            # Previous day source banner is stale; clear it
            _SOURCE_BANNER_BYTES = None
            _SOURCE_BANNER_FILE_ID = None
            _SOURCE_BANNER_FETCHED_DATE = None

        if _BANNER_BYTES is None and os.path.isfile(BANNER_PATH):
            try:
                with open(BANNER_PATH, "rb") as f:
                    _BANNER_BYTES = f.read()
            except Exception as exc:
                logger.warning("Failed to load banner bytes: %s", exc)
        return _BANNER_BYTES


def get_cached_banner_file_id() -> str | None:
    """Return cached Telegram file_id for 0ms photo delivery."""
    with _BANNER_LOCK:
        if _SOURCE_BANNER_BYTES is not None:
            return _SOURCE_BANNER_FILE_ID
        return _BANNER_FILE_ID


def set_cached_banner_file_id(fid: str | None) -> None:
    """Set or clear cached Telegram file_id."""
    global _BANNER_FILE_ID, _SOURCE_BANNER_FILE_ID
    with _BANNER_LOCK:
        if _SOURCE_BANNER_BYTES is not None:
            _SOURCE_BANNER_FILE_ID = fid
        else:
            _BANNER_FILE_ID = fid


_PODCAST_VOICE_LOCK: asyncio.Lock | None = None

# Female presenter cache
_PODCAST_VOICE_BYTES_F: bytes | None = None
_PODCAST_VOICE_FILE_ID_F: str | None = None
_PODCAST_VOICE_DATE_F: str | None = None

# Male presenter cache
_PODCAST_VOICE_BYTES_M: bytes | None = None
_PODCAST_VOICE_FILE_ID_M: str | None = None
_PODCAST_VOICE_DATE_M: str | None = None

# MP3 audio track cache
_PODCAST_MP3_FILE_ID_F: str | None = None
_PODCAST_MP3_FILE_ID_M: str | None = None
_PODCAST_MP3_DATE_F: str | None = None
_PODCAST_MP3_DATE_M: str | None = None

# Backward compatibility mirrors
_PODCAST_VOICE_BYTES: bytes | None = None
_PODCAST_VOICE_FILE_ID: str | None = None
_PODCAST_VOICE_DATE: str | None = None


def _get_podcast_voice_lock() -> asyncio.Lock:
    global _PODCAST_VOICE_LOCK
    if _PODCAST_VOICE_LOCK is None:
        _PODCAST_VOICE_LOCK = asyncio.Lock()
    return _PODCAST_VOICE_LOCK


def set_podcast_voice_bytes(
    data: bytes | None,
    fid: str | None = None,
    date_str: str | None = None,
    presenter: str = "female",
) -> None:
    """Explicitly set or clear cached podcast voice bytes and file_id for female or male presenter."""
    global _PODCAST_VOICE_BYTES, _PODCAST_VOICE_FILE_ID, _PODCAST_VOICE_DATE
    global _PODCAST_VOICE_BYTES_F, _PODCAST_VOICE_FILE_ID_F, _PODCAST_VOICE_DATE_F
    global _PODCAST_VOICE_BYTES_M, _PODCAST_VOICE_FILE_ID_M, _PODCAST_VOICE_DATE_M
    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"
    eff_date = date_str or (get_cambodia_now().strftime("%Y-%m-%d") if data else None)
    if pres == "male":
        _PODCAST_VOICE_BYTES_M = data
        _PODCAST_VOICE_FILE_ID_M = fid
        _PODCAST_VOICE_DATE_M = eff_date
    else:
        _PODCAST_VOICE_BYTES_F = data
        _PODCAST_VOICE_FILE_ID_F = fid
        _PODCAST_VOICE_DATE_F = eff_date
        # Mirror to legacy global aliases for backward compatibility
        _PODCAST_VOICE_BYTES = data
        _PODCAST_VOICE_FILE_ID = fid
        _PODCAST_VOICE_DATE = eff_date


def get_cached_podcast_voice_file_id(presenter: str = "female") -> str | None:
    """Return cached Telegram voice file_id for Daily Morning Podcast."""
    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"
    if pres == "male":
        return _PODCAST_VOICE_FILE_ID_M
    return _PODCAST_VOICE_FILE_ID_F or _PODCAST_VOICE_FILE_ID


def get_cached_podcast_voice_bytes(presenter: str = "female") -> bytes | None:
    """Return cached raw audio bytes for Daily Morning Podcast voice note."""
    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"
    if pres == "male":
        return _PODCAST_VOICE_BYTES_M
    return _PODCAST_VOICE_BYTES_F or _PODCAST_VOICE_BYTES


def get_cached_podcast_mp3_file_id(presenter: str = "female") -> str | None:
    """Return cached Telegram audio (MP3) file_id."""
    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"
    if pres == "male":
        return _PODCAST_MP3_FILE_ID_M
    return _PODCAST_MP3_FILE_ID_F


def set_cached_podcast_mp3_file_id(fid: str | None, presenter: str = "female") -> None:
    """Set cached Telegram audio (MP3) file_id."""
    global _PODCAST_MP3_FILE_ID_F, _PODCAST_MP3_FILE_ID_M, _PODCAST_MP3_DATE_F, _PODCAST_MP3_DATE_M
    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"
    now_str = get_cambodia_now().strftime("%Y-%m-%d") if fid else None
    if pres == "male":
        _PODCAST_MP3_FILE_ID_M = fid
        _PODCAST_MP3_DATE_M = now_str
    else:
        _PODCAST_MP3_FILE_ID_F = fid
        _PODCAST_MP3_DATE_F = now_str


async def get_or_synthesize_podcast_voice(
    speech_text: str,
    *,
    presenter: str = "female",
    force_refresh: bool = False,
) -> tuple[bytes | None, str | None]:
    """Ensure a high-quality TTS voice note is generated and cached for Daily Morning Podcast.

    Supports both 'female' (ស្រី) and 'male' (ប្រុស) presenter voices.
    Returns:
        tuple[bytes | None, str | None]: (audio_bytes, telegram_voice_file_id)
    """
    global _PODCAST_VOICE_BYTES, _PODCAST_VOICE_FILE_ID, _PODCAST_VOICE_DATE
    global _PODCAST_VOICE_BYTES_F, _PODCAST_VOICE_FILE_ID_F, _PODCAST_VOICE_DATE_F
    global _PODCAST_VOICE_BYTES_M, _PODCAST_VOICE_FILE_ID_M, _PODCAST_VOICE_DATE_M
    now = get_cambodia_now()
    today_key = now.strftime("%Y-%m-%d")
    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"

    lock = _get_podcast_voice_lock()
    async with lock:
        if force_refresh:
            if pres == "male":
                _PODCAST_VOICE_BYTES_M = None
                _PODCAST_VOICE_FILE_ID_M = None
                _PODCAST_VOICE_DATE_M = None
            else:
                _PODCAST_VOICE_BYTES_F = None
                _PODCAST_VOICE_FILE_ID_F = None
                _PODCAST_VOICE_DATE_F = None
                _PODCAST_VOICE_BYTES = None
                _PODCAST_VOICE_FILE_ID = None
                _PODCAST_VOICE_DATE = None

        cached_date = _PODCAST_VOICE_DATE_M if pres == "male" else (_PODCAST_VOICE_DATE_F or _PODCAST_VOICE_DATE)
        cached_fid = _PODCAST_VOICE_FILE_ID_M if pres == "male" else (_PODCAST_VOICE_FILE_ID_F or _PODCAST_VOICE_FILE_ID)
        cached_bytes = _PODCAST_VOICE_BYTES_M if pres == "male" else (_PODCAST_VOICE_BYTES_F or _PODCAST_VOICE_BYTES)

        if cached_date == today_key:
            if cached_fid:
                return None, cached_fid
            if cached_bytes:
                return cached_bytes, None

        if not speech_text:
            return None, None

        from app import legacy
        generate_voice_fn = getattr(legacy, "generate_voice", None)
        edge_voice_fn = getattr(legacy, "_generate_voice_edge", None)

        gender = "male" if pres == "male" else "female"
        audio_bytes: bytes | None = None
        if callable(generate_voice_fn):
            try:
                audio_bytes = await generate_voice_fn(
                    text=speech_text,
                    gender=gender,
                    speed=1.0,
                    output_path="",
                    tts_model="auto",
                )
            except Exception as exc:
                logger.warning("Primary podcast voice synthesis failed (%s); falling back to Edge TTS", exc)

        if not audio_bytes and callable(edge_voice_fn):
            try:
                audio_bytes = await edge_voice_fn(
                    text=speech_text,
                    gender=gender,
                    speed=1.0,
                    output_path="",
                )
            except Exception as edge_exc:
                logger.error("Edge TTS fallback for podcast voice failed: %s", edge_exc)

        if audio_bytes:
            if pres == "male":
                _PODCAST_VOICE_BYTES_M = audio_bytes
                _PODCAST_VOICE_FILE_ID_M = None
                _PODCAST_VOICE_DATE_M = today_key
            else:
                _PODCAST_VOICE_BYTES_F = audio_bytes
                _PODCAST_VOICE_FILE_ID_F = None
                _PODCAST_VOICE_DATE_F = today_key
                _PODCAST_VOICE_BYTES = audio_bytes
                _PODCAST_VOICE_FILE_ID = None
                _PODCAST_VOICE_DATE = today_key
            return audio_bytes, None

        raise RuntimeError(f"បរាជ័យក្នុងការបង្កើតសំឡេង Podcast ({'សំឡេងប្រុស' if pres == 'male' else 'សំឡេងស្រី'})។")


async def _prewarm_podcast_audio(speech_text: str) -> None:
    """Pre-warm male presenter voice in the background.

    This ensures when users tap [ 🎙️ ស្តាប់សំឡេងប្រុស ] or [ 🎵 ទាញយកជា MP3 ],
    delivery takes < 10ms with zero wait time.
    """
    if not speech_text:
        return
    try:
        if not get_cached_podcast_voice_bytes("male") and not get_cached_podcast_voice_file_id("male"):
            await get_or_synthesize_podcast_voice(speech_text, presenter="male")
    except Exception as exc:
        logger.debug("Podcast male pre-warm skipped: %s", exc)


async def send_podcast_voice(
    target: Any,
    speech_text: str,
    *,
    presenter: str = "female",
    bot: Any = None,
    chat_id: int | None = None,
    force_refresh: bool = False,
) -> Any:
    """Send the Daily Morning Podcast voice note.

    Supports both 'female' (ស្រី) and 'male' (ប្រុស) presenter voices.
    Caches Telegram voice file_id per presenter for instant 0ms broadcast across all users.
    Delivers as a single, unchunked audio voice message.
    """
    global _PODCAST_VOICE_FILE_ID, _PODCAST_VOICE_FILE_ID_F, _PODCAST_VOICE_FILE_ID_M
    if not speech_text:
        return None

    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"
    audio_bytes, cached_fid = await get_or_synthesize_podcast_voice(
        speech_text,
        presenter=pres,
        force_refresh=force_refresh,
    )
    msg_target = target
    if target and not hasattr(target, "reply_voice") and hasattr(target, "message"):
        msg_target = target.message
    voice_label = "សំឡេងស្រី" if pres == "female" else "សំឡេងប្រុស"
    caption = f"📻 <b>Daily Morning Podcast | ព័ត៌មានសំឡេងពេលព្រឹក</b> ({voice_label})"

    # 1. Send via cached Telegram file_id (0ms delivery, 0 bandwidth)
    if cached_fid:
        if msg_target and hasattr(msg_target, "reply_voice"):
            try:
                return await safe_send(lambda: msg_target.reply_voice(
                    voice=cached_fid,
                    caption=caption,
                    parse_mode="HTML",
                ))
            except Exception as exc:
                logger.warning("Reply voice using cached fid failed (%s); resetting cache", exc)
                if pres == "male":
                    _PODCAST_VOICE_FILE_ID_M = None
                else:
                    _PODCAST_VOICE_FILE_ID_F = None
                    _PODCAST_VOICE_FILE_ID = None
                audio_bytes, _ = await get_or_synthesize_podcast_voice(speech_text, presenter=pres, force_refresh=True)

        elif bot and chat_id:
            try:
                return await safe_send(lambda: bot.send_voice(
                    chat_id=chat_id,
                    voice=cached_fid,
                    caption=caption,
                    parse_mode="HTML",
                ))
            except Exception as exc:
                logger.warning("Broadcast voice to %s using fid failed (%s); resetting cache", chat_id, exc)
                if pres == "male":
                    _PODCAST_VOICE_FILE_ID_M = None
                else:
                    _PODCAST_VOICE_FILE_ID_F = None
                    _PODCAST_VOICE_FILE_ID = None
                audio_bytes, _ = await get_or_synthesize_podcast_voice(speech_text, presenter=pres, force_refresh=True)

    # 2. Send via audio bytes and capture file_id for subsequent calls
    if audio_bytes:
        bio = io.BytesIO(audio_bytes)
        bio.name = f"morning_podcast_{pres}.ogg"

        sent = None
        if msg_target and hasattr(msg_target, "reply_voice"):
            sent = await safe_send(lambda: msg_target.reply_voice(
                voice=bio,
                caption=caption,
                parse_mode="HTML",
            ))
        elif bot and chat_id:
            sent = await safe_send(lambda: bot.send_voice(
                chat_id=chat_id,
                voice=bio,
                caption=caption,
                parse_mode="HTML",
            ))

        if sent and hasattr(sent, "voice") and sent.voice and getattr(sent.voice, "file_id", None):
            fid = sent.voice.file_id
            if pres == "male":
                _PODCAST_VOICE_FILE_ID_M = fid
            else:
                _PODCAST_VOICE_FILE_ID_F = fid
                _PODCAST_VOICE_FILE_ID = fid
            logger.info("Captured Telegram voice file_id for Daily Morning Podcast [%s] (%s)", pres, fid[:12])
        return sent

    return None


async def send_podcast_mp3(
    target: Any,
    speech_text: str,
    *,
    presenter: str = "female",
    bot: Any = None,
    chat_id: int | None = None,
    force_refresh: bool = False,
) -> Any:
    """Send Daily Morning Podcast as an MP3 audio track.

    Delivers with title, artist/performer, and interactive audio player support in Telegram.
    Caches Telegram audio file_id per presenter for instant delivery.
    """
    global _PODCAST_MP3_FILE_ID_F, _PODCAST_MP3_FILE_ID_M, _PODCAST_MP3_DATE_F, _PODCAST_MP3_DATE_M
    if not speech_text:
        return None

    pres = "male" if str(presenter).lower() in ("male", "m", "ប្រុស") else "female"
    today_key = get_cambodia_now().strftime("%Y-%m-%d")
    msg_target = target
    if target and not hasattr(target, "reply_audio") and hasattr(target, "message"):
        msg_target = target.message

    voice_label = "សំឡេងស្រី" if pres == "female" else "សំឡេងប្រុស"
    caption = f"🎵 <b>Daily Morning Podcast (MP3 Audio Track)</b>\n🎙️ ព័ត៌មានពេលព្រឹកជាទម្រង់ MP3 ({voice_label})"
    title = f"Daily Morning Podcast ({voice_label})"
    performer = "Bot Voice Cambodia"

    cached_mp3_fid = _PODCAST_MP3_FILE_ID_M if pres == "male" else _PODCAST_MP3_FILE_ID_F
    cached_date = _PODCAST_MP3_DATE_M if pres == "male" else _PODCAST_MP3_DATE_F

    if cached_date == today_key and cached_mp3_fid and not force_refresh:
        if msg_target and hasattr(msg_target, "reply_audio"):
            try:
                return await safe_send(lambda: msg_target.reply_audio(
                    audio=cached_mp3_fid,
                    caption=caption,
                    parse_mode="HTML",
                    title=title,
                    performer=performer,
                ))
            except Exception as exc:
                logger.warning("Reply audio with cached fid failed (%s)", exc)
        elif bot and chat_id:
            try:
                return await safe_send(lambda: bot.send_audio(
                    chat_id=chat_id,
                    audio=cached_mp3_fid,
                    caption=caption,
                    parse_mode="HTML",
                    title=title,
                    performer=performer,
                ))
            except Exception as exc:
                logger.warning("Send audio to %s with cached fid failed (%s)", chat_id, exc)

    audio_bytes, _ = await get_or_synthesize_podcast_voice(speech_text, presenter=pres, force_refresh=force_refresh)
    if not audio_bytes:
        audio_bytes = get_cached_podcast_voice_bytes(presenter=pres)

    if audio_bytes:
        bio = io.BytesIO(audio_bytes)
        bio.name = f"daily_morning_podcast_{pres}.mp3"

        sent = None
        if msg_target and hasattr(msg_target, "reply_audio"):
            sent = await safe_send(lambda: msg_target.reply_audio(
                audio=bio,
                caption=caption,
                parse_mode="HTML",
                title=title,
                performer=performer,
            ))
        elif bot and chat_id:
            sent = await safe_send(lambda: bot.send_audio(
                chat_id=chat_id,
                audio=bio,
                caption=caption,
                parse_mode="HTML",
                title=title,
                performer=performer,
            ))

        if sent and hasattr(sent, "audio") and sent.audio and getattr(sent.audio, "file_id", None):
            fid = sent.audio.file_id
            if pres == "male":
                _PODCAST_MP3_FILE_ID_M = fid
                _PODCAST_MP3_DATE_M = today_key
            else:
                _PODCAST_MP3_FILE_ID_F = fid
                _PODCAST_MP3_DATE_F = today_key
            logger.info("Captured Telegram audio (MP3) file_id for Daily Morning Podcast [%s] (%s)", pres, fid[:12])
        return sent

    return None


def get_podcast_kb(is_subscribed: bool, bot_username: str = "") -> InlineKeyboardMarkup:
    """Build interactive 4-row keyboard for Daily Morning Podcast matching reference UI."""
    username = (bot_username or "voicekhaibot").lstrip("@")
    share_text = urllib.parse.quote_plus("📻 ស្តាប់ Daily Morning Podcast (ព័ត៌មានសំឡេងពេលព្រឹក) ជាមួយខ្ញុំ!")
    share_url = f"https://t.me/share/url?url=https://t.me/{username}?start=podcast&text={share_text}"

    btn_voice_f = InlineKeyboardButton("🎙️ ស្តាប់សំឡេងស្រី", callback_data="podcast_voice_f")
    btn_voice_m = InlineKeyboardButton("🎙️ ស្តាប់សំឡេងប្រុស", callback_data="podcast_voice_m")
    btn_mp3 = InlineKeyboardButton("🎵 ទាញយកជា MP3", callback_data="podcast_mp3")
    btn_refresh = InlineKeyboardButton("🔄 ព័ត៌មានថ្មី (Refresh)", callback_data="podcast_refresh")

    if is_subscribed:
        btn_sub = InlineKeyboardButton("🔕 ឈប់ជាវ (Unsubscribe)", callback_data="podcast_unsub")
    else:
        btn_sub = InlineKeyboardButton("🔔 ជាវរាល់ព្រឹក 7:00 AM", callback_data="podcast_sub")

    btn_share = InlineKeyboardButton("📩 ចែករំលែកទៅមិត្ត", url=share_url)
    btn_close = InlineKeyboardButton("❌ បិទ (Close)", callback_data="podcast_close")

    return InlineKeyboardMarkup([
        [btn_voice_f, btn_voice_m],
        [btn_mp3, btn_refresh],
        [btn_sub, btn_share],
        [btn_close],
    ])


def is_admin_user(user_id: int) -> bool:
    """Check if a user is an authorized bot administrator."""
    try:
        from app.legacy import _is_admin
        if _is_admin(user_id):
            return True
    except Exception:
        pass
    try:
        from app.core.telegram_auth import get_telegram_admin_authorizer
        return get_telegram_admin_authorizer().is_admin_sync(user_id)
    except Exception:
        return False


async def send_podcast_card(
    target: Any,
    html_text: str,
    *,
    reply_markup: Any = None,
    bot: Any = None,
    chat_id: int | None = None,
) -> Any:
    """Send Daily Morning Podcast matching the reference infographic card UI.

    Optimizations:
    - Reuses Telegram file_id after first upload to eliminate network payload overhead.
    - Caches raw image bytes in memory to eliminate disk I/O.
    - Automatically handles photo caption edits, photo sends, and text fallbacks.
    """
    banner_bytes = get_banner_bytes()
    has_banner = banner_bytes is not None or os.path.isfile(BANNER_PATH) or (target is not None and hasattr(target, "reply_photo"))

    def _get_photo_payload() -> Any:
        fid = get_cached_banner_file_id()
        if fid:
            return fid
        raw = banner_bytes or get_banner_bytes()
        if raw:
            bio = io.BytesIO(raw)
            bio.name = "morning_podcast.jpg"
            return bio
        return io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00")

    def _save_fid(sent: Any) -> None:
        if sent and hasattr(sent, "photo") and sent.photo:
            try:
                set_cached_banner_file_id(sent.photo[-1].file_id)
            except Exception:
                pass

    # Ensure photo caption strictly respects Telegram's 1024 UTF-16 character limit
    caption_text = html_text
    if len(caption_text) > 1024:
        from app.services.podcast.generator import build_podcast_card_html, format_khmer_date, get_cambodia_now
        caption_text = build_podcast_card_html(html_text, format_khmer_date(get_cambodia_now()), max_chars=1024)

    # 1. Direct message (e.g. /podcast on Telegram Message)
    if target and hasattr(target, "reply_photo") and has_banner:
        if len(caption_text) <= 1024:
            payload = _get_photo_payload()
            try:
                sent = await safe_send(lambda: target.reply_photo(
                    photo=payload,
                    caption=caption_text,
                    parse_mode="HTML",
                    reply_markup=reply_markup,
                ))
                if sent:
                    _save_fid(sent)
                    return sent
            except Exception as exc:
                if isinstance(payload, str):
                    set_cached_banner_file_id(None)
                    payload = _get_photo_payload()
                    try:
                        sent = await safe_send(lambda: target.reply_photo(
                            photo=payload,
                            caption=html_text,
                            parse_mode="HTML",
                            reply_markup=reply_markup,
                        ))
                        if sent:
                            _save_fid(sent)
                            return sent
                    except Exception as exc2:
                        logger.warning("Retry reply_photo failed: %s", exc2)
                else:
                    logger.warning("Failed to reply podcast photo: %s", exc)
        else:
            payload = _get_photo_payload()
            try:
                await safe_send(lambda: target.reply_photo(
                    photo=payload,
                    caption="📻 <b>ព័ត៌មានពេលព្រឹកថ្ងៃនេះ | Daily Morning Podcast</b>",
                    parse_mode="HTML",
                ))
            except Exception as exc:
                logger.warning("Failed to reply podcast banner: %s", exc)

    # 2. CallbackQuery (e.g. Refresh)
    if target and hasattr(target, "edit_message_caption"):
        msg_obj = getattr(target, "message", None)
        if msg_obj and getattr(msg_obj, "photo", None):
            if len(html_text) <= 1024:
                try:
                    res = await safe_send(lambda: target.edit_message_caption(
                        caption=html_text,
                        parse_mode="HTML",
                        reply_markup=reply_markup,
                    ))
                    if res:
                        return res
                except Exception:
                    pass

    if target and hasattr(target, "edit_message_text") and not hasattr(target, "reply_photo"):
        from app.services.telegram.formatters import send_split_html
        return await send_split_html(
            target,
            html_text,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
            edit_initial=True,
        )

    # 3. Direct bot broadcast (to chat_id)
    if bot and chat_id and has_banner:
        if len(html_text) <= 1024:
            payload = _get_photo_payload()
            try:
                sent = await safe_send(lambda: bot.send_photo(
                    chat_id=chat_id,
                    photo=payload,
                    caption=html_text,
                    parse_mode="HTML",
                    reply_markup=reply_markup,
                ))
                if sent:
                    _save_fid(sent)
                    return sent
            except Exception as exc:
                if isinstance(payload, str):
                    set_cached_banner_file_id(None)
                    payload = _get_photo_payload()
                    try:
                        sent = await safe_send(lambda: bot.send_photo(
                            chat_id=chat_id,
                            photo=payload,
                            caption=html_text,
                            parse_mode="HTML",
                            reply_markup=reply_markup,
                        ))
                        if sent:
                            _save_fid(sent)
                            return sent
                    except Exception as exc2:
                        logger.warning("Retry broadcast photo failed: %s", exc2)
                else:
                    logger.warning("Failed to broadcast podcast photo to %s: %s", chat_id, exc)
        else:
            payload = _get_photo_payload()
            try:
                await safe_send(lambda: bot.send_photo(
                    chat_id=chat_id,
                    photo=payload,
                    caption="📻 <b>ព័ត៌មានពេលព្រឹកថ្ងៃនេះ | Daily Morning Podcast</b>",
                    parse_mode="HTML",
                ))
            except Exception as exc:
                logger.warning("Failed to send broadcast banner photo to %s: %s", chat_id, exc)

    # 4. Fallback text delivery via send_split_bot_message or send_split_html
    if bot and chat_id:
        from app.services.telegram.formatters import send_split_bot_message
        return await send_split_bot_message(
            bot,
            chat_id,
            html_text,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
        )

    from app.services.telegram.formatters import send_split_html
    return await send_split_html(
        target,
        html_text,
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )


@legacy_bound_handler
async def cmd_podcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /podcast and /morning commands for Daily Morning Podcast."""
    msg = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if not msg or not user or not chat:
        return

    from app.core.features import is_podcast_enabled
    if not is_podcast_enabled():
        await safe_send(lambda: msg.reply_text("⚠️ មុខងារព័ត៌មានពេលព្រឹក (Podcast) ត្រូវបានបិទដំណើរការ (Podcast Disabled)។"))
        return

    chat_id = int(chat.id)
    user_id = int(user.id)
    raw_args = re.sub(r"^/(?:podcast|morning|dailypodcast)(?:@\w+)?\s*", "", msg.text or "", flags=re.IGNORECASE).strip().lower()
    bot_username = getattr(context.bot, "username", "") if context and getattr(context, "bot", None) else ""

    if raw_args in ("on", "sub", "subscribe"):
        is_new = podcast_store.subscribe(chat_id)
        reply = (
            "✅ <b>បានចុះឈ្មោះជាវជោគជ័យ!</b>\n\n"
            "📻 Bot Voice នឹងផ្ញើ <b>Daily Morning Podcast (ព័ត៌មានសំឡេងពេលព្រឹក)</b> "
            "ជូនបងជារៀងរាល់ថ្ងៃនៅម៉ោង <b>៧:០០ ព្រឹក (7:00 AM)</b>។\n\n"
            "💡 បើចង់ឈប់ទទួល សូមវាយ <code>/podcast off</code>។"
        )
        await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML", reply_markup=get_podcast_kb(True, bot_username)))
        return

    if raw_args in ("off", "unsub", "unsubscribe"):
        removed = podcast_store.unsubscribe(chat_id)
        reply = (
            "🔕 <b>បានបោះបង់ការជាវជោគជ័យ!</b>\n\n"
            "បងនឹងលែងទទួលបានព័ត៌មានសំឡេងពេលព្រឹកស្វ័យប្រវត្តិតទៅទៀតហើយ។\n"
            "💡 បើចង់ជាវឡើងវិញ សូមវាយ <code>/podcast on</code>។"
        )
        await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML", reply_markup=get_podcast_kb(False, bot_username)))
        return

    if raw_args in ("status", "check"):
        subbed = podcast_store.is_subscribed(chat_id)
        status_str = "✅ កំពុងជាវ (Subscribed)" if subbed else "❌ មិនទាន់បានជាវទេ (Not Subscribed)"
        reply = (
            f"📻 <b>ស្ថានភាព Daily Morning Podcast</b>\n\n"
            f"• ស្ថានភាព៖ <b>{status_str}</b>\n"
            f"• ម៉ោងផ្សាយ៖ <b>៧:០០ ព្រឹកជារៀងរាល់ថ្ងៃ (UTC+7)</b>\n\n"
            f"👉 វាយ <code>/podcast on</code> ដើម្បីជាវ ឬ <code>/podcast off</code> ដើម្បីឈប់ជាវ។"
        )
        await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML", reply_markup=get_podcast_kb(subbed, bot_username)))
        return

    if raw_args in ("stats", "info"):
        total_subs = podcast_store.count()
        last_bcast = podcast_store.get_last_broadcast_date() or "មិនទាន់មាន"
        vf_cached = "✅ រក្សាទុក (Cached)" if get_cached_podcast_voice_file_id("female") else "⏳ មិនទាន់មាន"
        vm_cached = "✅ រក្សាទុក (Cached)" if get_cached_podcast_voice_file_id("male") else "⏳ មិនទាន់មាន"
        mp3_cached = "✅ រក្សាទុក (Cached)" if get_cached_podcast_mp3_file_id("female") else "⏳ មិនទាន់មាន"
        cover_cached = "✅ Source Banner" if get_source_banner_bytes() else "🖼️ Cover Banner"

        reply = (
            f"📊 <b>ស្ថិតិ Daily Morning Podcast</b>\n\n"
            f"• ចំនួនអ្នកជាវ (Subscribers)៖ <b>{total_subs}</b> នាក់\n"
            f"• ផ្សាយចុងក្រោយ៖ <b>{last_bcast}</b>\n"
            f"• សំឡេងស្រី Cache៖ <b>{vf_cached}</b>\n"
            f"• សំឡេងប្រុស Cache៖ <b>{vm_cached}</b>\n"
            f"• MP3 Track Cache៖ <b>{mp3_cached}</b>\n"
            f"• រូបភាព Cover៖ <b>{cover_cached}</b>\n"
        )
        await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
        return

    if raw_args in ("female", "f"):
        await safe_send(lambda: msg.reply_chat_action("record_voice"))
        _, speech_text = await generate_morning_podcast(force_refresh=False)
        await send_podcast_voice(msg, speech_text, presenter="female")
        return

    if raw_args in ("male", "m"):
        await safe_send(lambda: msg.reply_chat_action("record_voice"))
        _, speech_text = await generate_morning_podcast(force_refresh=False)
        await send_podcast_voice(msg, speech_text, presenter="male")
        return

    if raw_args == "mp3":
        await safe_send(lambda: msg.reply_chat_action("upload_document"))
        _, speech_text = await generate_morning_podcast(force_refresh=False)
        await send_podcast_mp3(msg, speech_text, presenter="female")
        return

    if raw_args == "preview":
        await safe_send(lambda: msg.reply_chat_action("typing"))
        html_text, speech_text = await generate_morning_podcast(force_refresh=False)
        subbed = podcast_store.is_subscribed(chat_id)
        kb = get_podcast_kb(subbed, bot_username)
        await send_podcast_card(msg, html_text, reply_markup=kb)
        await send_podcast_voice(msg, speech_text, presenter="female")
        await send_podcast_voice(msg, speech_text, presenter="male")
        return

    if raw_args in ("broadcast", "sendall"):
        if not is_admin_user(user_id):
            await safe_send(lambda: msg.reply_text(
                "⛔️ <b>ការអនុញ្ញាតត្រូវបានបដិសេធ (Admin Only)</b>\n"
                "ពាក្យបញ្ជា <code>/podcast broadcast</code> សម្រាប់តែ Admin ប៉ុណ្ណោះ។",
                parse_mode="HTML",
            ))
            return

        subscribers = podcast_store.get_all_subscribers()
        if not subscribers:
            await safe_send(lambda: msg.reply_text("⚠️ មិនទាន់មាន Subscribers ណាម្នាក់នៅឡើយទេ។"))
            return

        status_msg = await safe_send(lambda: msg.reply_text(
            f"🚀 <b>កំពុងរៀបចំផ្សាយ Morning Podcast ទៅកាន់ {len(subscribers)} Subscribers...</b>",
            parse_mode="HTML",
        ))

        with suppress(Exception):
            await asyncio.wait_for(fetch_source_banner_bytes(), timeout=3.0)
        html_text, speech_text = await generate_morning_podcast(force_refresh=True)
        with suppress(Exception):
            await get_or_synthesize_podcast_voice(speech_text, presenter="female", force_refresh=True)

        bot = getattr(context, "bot", None) if context else None
        if not bot:
            from app.legacy import _bot_app
            bot = getattr(_bot_app, "bot", None) if _bot_app else None

        if not bot:
            await safe_send(lambda: msg.reply_text("❌ មិនអាចទាក់ទង Bot Instance បានឡើយ។"))
            return

        kb = get_podcast_kb(True, bot_username)
        success_count = 0
        fail_count = 0
        dead_subscribers: list[int] = []

        for cid in subscribers:
            try:
                await send_podcast_card(None, html_text, reply_markup=kb, bot=bot, chat_id=cid)
                await send_podcast_voice(None, speech_text, presenter="female", bot=bot, chat_id=cid)
                success_count += 1
            except Exception as exc:
                fail_count += 1
                logger.warning("Admin broadcast failed to %s: %s", cid, exc)
                low_exc = str(exc).lower()
                if any(k in low_exc for k in ("blocked", "deactivated", "chat not found", "user is deactivated")):
                    dead_subscribers.append(cid)
            await asyncio.sleep(0.04)

        if dead_subscribers:
            podcast_store.unsubscribe_batch(dead_subscribers)
            logger.info("Auto-unsubscribed %d dead/blocked podcast subscribers during admin broadcast", len(dead_subscribers))

        today_str = get_cambodia_now().strftime("%Y-%m-%d")
        podcast_store.set_last_broadcast_date(today_str)

        summary_text = (
            f"✅ <b>ការផ្សាយ Podcast ត្រូវបានបញ្ចប់!</b>\n\n"
            f"• ផ្ញើជោគជ័យ៖ <b>{success_count}</b> នាក់\n"
            f"• បរាជ័យ៖ <b>{fail_count}</b> នាក់\n"
            f"• សរុប៖ <b>{len(subscribers)}</b> នាក់\n"
            f"• កាលបរិច្ឆេទ៖ <b>{today_str}</b>"
        )
        if status_msg and hasattr(status_msg, "edit_text"):
            await safe_send(lambda: status_msg.edit_text(summary_text, parse_mode="HTML"))
        else:
            await safe_send(lambda: msg.reply_text(summary_text, parse_mode="HTML"))
        return

    # Default: Generate and send today's podcast on-demand
    await safe_send(lambda: msg.reply_chat_action("typing"))
    try:
        force_refresh = "refresh" in raw_args or "new" in raw_args
        with suppress(Exception):
            await asyncio.wait_for(fetch_source_banner_bytes(), timeout=3.0)
        html_text, speech_text = await generate_morning_podcast(force_refresh=force_refresh)
        subbed = podcast_store.is_subscribed(chat_id)
        kb = get_podcast_kb(subbed, bot_username)
        await send_podcast_card(msg, html_text, reply_markup=kb)
        await send_podcast_voice(msg, speech_text, presenter="female", force_refresh=force_refresh)
        # Pre-warm male voice in background for instantaneous second click
        with suppress(Exception):
            asyncio.create_task(_prewarm_podcast_audio(speech_text))
    except Exception as exc:
        logger.error("cmd_podcast error: %s", exc, exc_info=True)
        await safe_send(lambda: msg.reply_text(f"❌ បរាជ័យក្នុងការបង្កើត Podcast: {exc}"))


async def podcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks for podcast."""
    query = update.callback_query
    if not query:
        return

    from app.core.features import is_podcast_enabled
    if not is_podcast_enabled():
        with suppress(Exception):
            await query.answer("⚠️ មុខងារព័ត៌មានពេលព្រឹក (Podcast) ត្រូវបានបិទដំណើរការ។", show_alert=True)
        return

    data = str(query.data or "").strip()
    chat_id = int(query.message.chat_id) if query.message else int(query.from_user.id)
    user_id = int(query.from_user.id)
    bot_username = getattr(context.bot, "username", "") if context and getattr(context, "bot", None) else ""

    if data == "podcast_close":
        with suppress(Exception):
            await query.answer("បិទ")
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "podcast_sub":
        podcast_store.subscribe(chat_id)
        with suppress(Exception):
            await query.answer("✅ បានជាវព័ត៌មានពេលព្រឹកជោគជ័យ!")
        with suppress(Exception):
            await query.edit_message_reply_markup(reply_markup=get_podcast_kb(True, bot_username))
        return

    if data == "podcast_unsub":
        podcast_store.unsubscribe(chat_id)
        with suppress(Exception):
            await query.answer("🔕 បានឈប់ជាវជោគជ័យ!")
        with suppress(Exception):
            await query.edit_message_reply_markup(reply_markup=get_podcast_kb(False, bot_username))
        return

    if data == "podcast_voice_f":
        with suppress(Exception):
            await query.answer("🎙️ កំពុងផ្ញើសំឡេងស្រី...")
        reply_target = query.message if getattr(query, "message", None) else query
        if reply_target and hasattr(reply_target, "reply_chat_action"):
            with suppress(Exception):
                await safe_send(lambda: reply_target.reply_chat_action("record_voice"))
        _, speech_text = await generate_morning_podcast(force_refresh=False)
        await send_podcast_voice(reply_target, speech_text, presenter="female", force_refresh=False)
        return

    if data == "podcast_voice_m":
        with suppress(Exception):
            await query.answer("🎙️ កំពុងផ្ញើសំឡេងប្រុស...")
        reply_target = query.message if getattr(query, "message", None) else query
        if reply_target and hasattr(reply_target, "reply_chat_action"):
            with suppress(Exception):
                await safe_send(lambda: reply_target.reply_chat_action("record_voice"))
        _, speech_text = await generate_morning_podcast(force_refresh=False)
        await send_podcast_voice(reply_target, speech_text, presenter="male", force_refresh=False)
        return

    if data == "podcast_mp3":
        with suppress(Exception):
            await query.answer("🎵 កំពុងផ្ញើឯកសារ MP3...")
        reply_target = query.message if getattr(query, "message", None) else query
        if reply_target and hasattr(reply_target, "reply_chat_action"):
            with suppress(Exception):
                await safe_send(lambda: reply_target.reply_chat_action("upload_document"))
        _, speech_text = await generate_morning_podcast(force_refresh=False)
        await send_podcast_mp3(reply_target, speech_text, presenter="female", force_refresh=False)
        return

    if data == "podcast_refresh":
        with suppress(Exception):
            await query.answer("🔄 កំពុងទាញយកព័ត៌មានថ្មី...")
        with suppress(Exception):
            await asyncio.wait_for(fetch_source_banner_bytes(), timeout=3.0)
        html_text, speech_text = await generate_morning_podcast(force_refresh=True)
        subbed = podcast_store.is_subscribed(chat_id)
        kb = get_podcast_kb(subbed, bot_username)
        await send_podcast_card(query, html_text, reply_markup=kb)
        reply_target = query.message if getattr(query, "message", None) else query
        await send_podcast_voice(reply_target, speech_text, presenter="female", force_refresh=True)
        with suppress(Exception):
            asyncio.create_task(_prewarm_podcast_audio(speech_text))
        return


async def periodic_podcast_scheduler(poll_interval: float = 30.0) -> None:
    """Background task to automatically dispatch the Daily Morning Podcast at 7:00 AM (UTC+7)."""
    logger.info("Daily Morning Podcast background scheduler started (Target: 07:00 AM UTC+7).")

    while True:
        try:
            from app.core.features import is_podcast_enabled
            if not is_podcast_enabled():
                await asyncio.sleep(poll_interval * 2)
                continue
            now = get_cambodia_now()
            today_str = now.strftime("%Y-%m-%d")

            # Check if current time is between 07:00 and 07:30 AM in Cambodia
            if now.hour == 7 and 0 <= now.minute <= 30:
                last_sent = podcast_store.get_last_broadcast_date()
                if last_sent != today_str:
                    logger.info("Executing 07:00 AM Daily Morning Podcast dispatch for %s...", today_str)
                    podcast_store.set_last_broadcast_date(today_str)

                    subscribers = podcast_store.get_all_subscribers()
                    if subscribers:
                        with suppress(Exception):
                            await asyncio.wait_for(fetch_source_banner_bytes(), timeout=4.0)
                        html_text, speech_text = await generate_morning_podcast(force_refresh=True)
                        from app.legacy import _bot_app

                        bot = getattr(_bot_app, "bot", None) if _bot_app else None
                        if bot:
                            bot_username = getattr(bot, "username", "")
                            kb = get_podcast_kb(True, bot_username)
                            with suppress(Exception):
                                await get_or_synthesize_podcast_voice(speech_text, presenter="female", force_refresh=True)
                            with suppress(Exception):
                                asyncio.create_task(_prewarm_podcast_audio(speech_text))

                            success_count = 0
                            dead_subscribers: list[int] = []
                            for cid in subscribers:
                                try:
                                    await send_podcast_card(None, html_text, reply_markup=kb, bot=bot, chat_id=cid)
                                    await send_podcast_voice(None, speech_text, presenter="female", bot=bot, chat_id=cid)
                                    success_count += 1
                                    await asyncio.sleep(0.04)
                                except Exception as exc:
                                    logger.warning("Failed to send morning podcast to chat %s: %s", cid, exc)
                                    low_exc = str(exc).lower()
                                    if any(k in low_exc for k in ("blocked", "deactivated", "chat not found", "user is deactivated")):
                                        dead_subscribers.append(cid)

                            if dead_subscribers:
                                podcast_store.unsubscribe_batch(dead_subscribers)
                                logger.info("Auto-unsubscribed %d dead/blocked podcast subscribers", len(dead_subscribers))

                            logger.info("07:00 AM Podcast dispatched to %d/%d subscribers.", success_count, len(subscribers))
        except asyncio.CancelledError:
            logger.info("Daily Morning Podcast scheduler cancelled.")
            break
        except Exception as exc:
            logger.error("Error in periodic_podcast_scheduler: %s", exc, exc_info=True)

        await asyncio.sleep(poll_interval)


__all__ = [
    "cmd_podcast",
    "get_cached_podcast_mp3_file_id",
    "get_cached_podcast_voice_bytes",
    "get_cached_podcast_voice_file_id",
    "get_or_synthesize_podcast_voice",
    "get_podcast_kb",
    "is_admin_user",
    "periodic_podcast_scheduler",
    "podcast_callback",
    "send_podcast_card",
    "send_podcast_mp3",
    "send_podcast_voice",
    "set_cached_podcast_mp3_file_id",
    "set_podcast_voice_bytes",
]
