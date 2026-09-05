"""Telegram Channel Auto-Voice Narrator module.

Listens to channel posts (text and media captions) in channels where the bot is
an administrator, cleans the content for natural speech synthesis, and posts
high-quality audio voice notes directly under the post.
"""

from __future__ import annotations

# ruff: noqa: F821
import asyncio
import io
import logging
import os
import re
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

from app.services.telegram._legacy_runtime import legacy_bound_handler

logger = logging.getLogger(__name__)

# Opt-out hashtags that channel editors can place in a post to silence narration
OPT_OUT_TAGS: frozenset[str] = frozenset({
    "#notts",
    "#no_tts",
    "#novice",
    "#novoice",
    "#no_voice",
    "#nonarrate",
    "#no_narrate",
    "#noaudio",
    "#no_audio",
    "#silent",
    "#mute",
})

# Per-channel concurrency lock to prevent overlapping voice generation
_channel_locks: dict[int, asyncio.Lock] = {}
_channel_locks_guard = asyncio.Lock()


def is_narration_opted_out(text: str) -> bool:
    """Check if the text contains any opt-out hashtags silencing narration."""
    if not text:
        return False
    lower = text.lower()
    return any(tag in lower for tag in OPT_OUT_TAGS)


def clean_channel_text(raw_text: str, max_chars: int = 2000) -> str:
    """Prepare channel post text for studio-quality voice note synthesis.

    1. Removes raw HTTP/HTTPS URLs (so TTS doesn't spell them out letter by letter).
    2. Strips repeated decorative delimiter lines (e.g. ➖➖➖➖➖, ------).
    3. Removes trailing hashtags and signature channels.
    4. Truncates cleanly at the nearest sentence boundary if exceeding ``max_chars``.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1. Remove URLs (http, https, www, t.me)
    text = re.sub(r"https?://\S+|www\.\S+|t\.me/\S+", "", text, flags=re.IGNORECASE)

    # 2. Remove decorative divider lines (e.g. ➖➖➖, ----, ====, ****)
    text = re.sub(r"[\u2500-\u257f\u2580-\u259f\u25ac-\u25b0\u25ac\u25ad\u2796\u2014\-=_~*]{3,}", " ", text)

    # 3. Strip trailing combinations of hashtags and channel signatures (@channel)
    text = re.sub(r"(?:[\s\n]*(?:#[\w\u1780-\u17ff]+|@[\w_]+))+$", "", text)

    # 5. Collapse excessive whitespace and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 6. Minimum substantive text length check
    if len(text) < 15 or not re.search(r"[\w\u1780-\u17ff]", text):
        return ""

    # 7. Truncate at nearest sentence boundary if exceeding max_chars
    if len(text) > max_chars:
        candidate = text[:max_chars]
        # Khmer full stops are \u17d4 (។) and \u17d5 (៕); Latin punctuation . ? ! \n
        last_punct = max(
            candidate.rfind("។"),
            candidate.rfind("៕"),
            candidate.rfind(".\n"),
            candidate.rfind(". "),
            candidate.rfind("?\n"),
            candidate.rfind("! "),
            candidate.rfind("\n"),
        )
        if last_punct > int(max_chars * 0.6):
            text = candidate[: last_punct + 1].strip() + "…"
        else:
            text = candidate.strip() + "…"

    return text.strip()


async def _get_channel_lock(chat_id: int) -> asyncio.Lock:
    """Retrieve or create an async lock for the given channel chat ID."""
    if chat_id in _channel_locks:
        return _channel_locks[chat_id]
    async with _channel_locks_guard:
        if chat_id not in _channel_locks:
            _channel_locks[chat_id] = asyncio.Lock()
        return _channel_locks[chat_id]


def _is_channel_allowed(chat_id: int, username: str | None) -> bool:
    """Check if the channel is allowed based on ALLOWED_CHANNEL_IDS configuration."""
    raw_allowed = ""
    try:
        from app import legacy
        get_setting = getattr(legacy, "bot_setting_raw_cached", None)
        if callable(get_setting):
            raw_allowed = str(get_setting("allowed_channel_ids", "") or "").strip()
    except Exception:
        raw_allowed = ""

    if not raw_allowed:
        raw_allowed = os.environ.get("ALLOWED_CHANNEL_IDS", "").strip()
    if not raw_allowed:
        return True  # If not explicitly restricted, allow any channel where bot is admin

    allowed_set = {part.strip().lower() for part in raw_allowed.split(",") if part.strip()}
    if str(chat_id) in allowed_set:
        return True
    if username and f"@{username.lower().lstrip('@')}" in allowed_set:
        return True
    return bool(username and username.lower().lstrip("@") in allowed_set)


@legacy_bound_handler
async def on_channel_post(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Listen for new channel posts and automatically narrate to voice note."""
    post = update.channel_post
    if not post:
        return

    chat = update.effective_chat
    if not chat or chat.type != "channel":
        return

    # 1. Check global feature settings
    narrator_enabled = bot_setting_bool_cached("channel_narrator_enabled", True)
    if not narrator_enabled:
        return

    if bot_setting_bool_cached("maintenance_mode", False):
        logger.debug("Channel narrator skipped for %s: maintenance mode active", chat.id)
        return

    # 2. Check channel authorization / whitelist
    if not _is_channel_allowed(chat.id, chat.username):
        logger.debug("Channel %s (%s) is not in ALLOWED_CHANNEL_IDS; skipping", chat.title, chat.id)
        return

    # 3. Extract raw text from text post or media caption
    raw_text = (post.text or post.caption or "").strip()
    if not raw_text:
        return

    # 4. Check opt-out hashtags
    if is_narration_opted_out(raw_text):
        logger.info("Channel post %s in %s contains opt-out tag; skipping narration.", post.message_id, chat.title)
        return

    # 5. Clean text for natural speech
    max_chars_str = (
        bot_setting_raw_cached("channel_narrator_max_chars", os.environ.get("CHANNEL_NARRATOR_MAX_CHARS", "2000"))
        if "bot_setting_raw_cached" in globals()
        else os.environ.get("CHANNEL_NARRATOR_MAX_CHARS", "2000")
    )
    try:
        max_chars = int(max_chars_str)
    except Exception:
        max_chars = 2000
    tts_text = clean_channel_text(raw_text, max_chars=max_chars)
    if not tts_text:
        return

    # 6. Acquire per-channel lock to prevent overlapping voice jobs
    lock = await _get_channel_lock(chat.id)
    if lock.locked():
        logger.info("Channel %s has active narration in progress; queuing...", chat.title)

    async with lock:
        file_path: str | None = None
        try:
            gender = (
                bot_setting_raw_cached("channel_narrator_gender", os.environ.get("CHANNEL_NARRATOR_GENDER", "female"))
                if "bot_setting_raw_cached" in globals()
                else os.environ.get("CHANNEL_NARRATOR_GENDER", "female")
            ).strip().lower()
            speed_str = (
                bot_setting_raw_cached("channel_narrator_speed", os.environ.get("CHANNEL_NARRATOR_SPEED", "1.0"))
                if "bot_setting_raw_cached" in globals()
                else os.environ.get("CHANNEL_NARRATOR_SPEED", "1.0")
            )
            try:
                speed = float(speed_str)
            except Exception:
                speed = 1.0
            tts_model = (
                bot_setting_raw_cached("channel_narrator_model", os.environ.get("CHANNEL_NARRATOR_MODEL", "auto"))
                if "bot_setting_raw_cached" in globals()
                else os.environ.get("CHANNEL_NARRATOR_MODEL", "auto")
            ).strip().lower()
            file_path = _make_temp_ogg()

            logger.info(
                "Generating channel voice narration for '%s' (post_id=%s, len=%d chars)",
                chat.title,
                post.message_id,
                len(tts_text),
            )

            # Generate speech with fallback (Edge TTS / HF / Gemini)
            audio_bytes = await asyncio.wait_for(
                generate_voice_limited(
                    tts_text,
                    gender=gender,
                    speed=speed,
                    output_path=file_path,
                    tts_model=tts_model,
                ),
                timeout=90.0,
            )

            if not audio_bytes:
                logger.warning("Empty audio generated for channel post %s in %s", post.message_id, chat.title)
                return

            caption = (
                f"🗣️ {BOT_TAG}"
                if "BOT_TAG" in globals() and BOT_TAG
                else "🗣️ សំឡេងអានអត្ថបទ (Audio Narration)"
            )

            show_buttons = (
                bot_setting_bool_cached("channel_narrator_show_buttons", False)
                if "bot_setting_bool_cached" in globals()
                else os.environ.get("CHANNEL_NARRATOR_SHOW_BUTTONS", "0").lower() in ("1", "true", "yes")
            )
            reply_markup = None
            if show_buttons and "get_audio_action_kb" in globals() and "_tts_text_cache_set" in globals():
                text_cache_id = _tts_text_cache_set(tts_text)
                reply_markup = get_audio_action_kb(text_cache_id, tts_text, gender=gender, speed=speed, model=tts_model)

            # Try replying to the post first so it anchors directly in discussion threads
            sent = False
            try:
                await safe_send(lambda ab=audio_bytes, rm=reply_markup: context.bot.send_voice(
                    chat_id=chat.id,
                    voice=io.BytesIO(ab),
                    caption=caption,
                    reply_markup=rm,
                    reply_to_message_id=post.message_id,
                ))
                sent = True
            except Exception as reply_err:
                logger.debug(
                    "reply_to_message_id failed in channel %s (%s); trying direct post: %s",
                    chat.title,
                    chat.id,
                    reply_err,
                )

            if not sent:
                # Direct post without reply_to_message_id if replying is disabled in channel
                await safe_send(lambda ab=audio_bytes, rm=reply_markup: context.bot.send_voice(
                    chat_id=chat.id,
                    voice=io.BytesIO(ab),
                    caption=caption,
                    reply_markup=rm,
                ))

            _metric_inc("channel_narrations")
            logger.info("Successfully published channel narration for '%s' (post_id=%s)", chat.title, post.message_id)

        except TimeoutError:
            logger.warning("TTS timeout generating channel narration for %s (post_id=%s)", chat.title, post.message_id)
        except Exception as exc:
            logger.error("Channel voice narration error in %s: %s", chat.title, exc, exc_info=True)
        finally:
            if file_path:
                with suppress(FileNotFoundError, Exception):
                    os.remove(file_path)


__all__ = [
    "OPT_OUT_TAGS",
    "clean_channel_text",
    "is_narration_opted_out",
    "on_channel_post",
]
