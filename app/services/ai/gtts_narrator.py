"""Khmer Voice Narration engine using gTTS and pydub with resilient audio processing."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import io
import logging
import re
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# Check if ffmpeg is available on system
_FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg"))


def _speed_change(sound, speed: float = 1.0):
    """Alter speed without changing pitch using pydub or sample rate override."""
    if abs(speed - 1.0) < 0.05:
        return sound
    try:
        from pydub.effects import speedup
        if speed > 1.0:
            return speedup(sound, playback_speed=speed)
        # Slower speed via sample rate modification
        altered = sound._spawn(
            sound.raw_data,
            overrides={"frame_rate": int(sound.frame_rate * speed)},
        )
        return altered.set_frame_rate(sound.frame_rate)
    except Exception as exc:
        logger.debug("Failed applying speed change with pydub: %s", exc)
        return sound


def generate_khmer_gtts_audio(
    text: str,
    speed: float = 1.0,
    to_voice_note: bool = False,
    lang: str = "km",
) -> bytes:
    """Generate audio bytes for Khmer text using gTTS and pydub.
    
    Args:
        text: Text to narrate.
        speed: Playback speed multiplier (e.g. 1.0, 1.15, 1.25).
        to_voice_note: If True and ffmpeg is installed, converts to OGG Opus for native voice notes.
        lang: Language code ('km' for Khmer, or 'en').
        
    Returns:
        Raw audio bytes (OGG Opus if voice note conversion succeeded, else MP3).
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return b""

    try:
        from gtts import gTTS
    except ImportError:
        logger.error("gtts is not installed.")
        return b""

    # Truncate or chunk if overly long (gTTS handles up to ~5000 chars)
    if len(clean_text) > 4000:
        clean_text = clean_text[:4000].rsplit(" ", 1)[0]

    # Generate audio via gTTS
    try:
        tts = gTTS(text=clean_text, lang=lang, slow=(speed < 0.85))
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        mp3_bytes = mp3_io.getvalue()
    except Exception as exc:
        logger.warning("gTTS generation failed: %s", exc)
        return b""

    # If no pydub post-processing is needed or ffmpeg is missing, return MP3
    if not _FFMPEG_AVAILABLE and not shutil.which("ffmpeg"):
        return mp3_bytes

    if abs(speed - 1.0) < 0.05 and not to_voice_note:
        return mp3_bytes

    # Attempt post-processing via pydub
    try:
        from pydub import AudioSegment

        sound = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")

        # Apply speed adjustment if requested
        if abs(speed - 1.0) >= 0.05:
            sound = _speed_change(sound, speed)

        out_io = io.BytesIO()
        if to_voice_note:
            try:
                sound.export(out_io, format="ogg", codec="libopus", parameters=["-b:a", "32k"])
                return out_io.getvalue()
            except Exception:
                out_io = io.BytesIO()
                sound.export(out_io, format="mp3", bitrate="64k")
                return out_io.getvalue()
        else:
            sound.export(out_io, format="mp3", bitrate="64k")
            return out_io.getvalue()

    except Exception as err:
        logger.debug("pydub post-processing skipped (%s); returning direct gTTS MP3", err)
        return mp3_bytes


async def generate_khmer_gtts_audio_async(
    text: str,
    speed: float = 1.0,
    to_voice_note: bool = False,
    lang: str = "km",
) -> bytes:
    """Asynchronous wrapper for generate_khmer_gtts_audio running in a worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        generate_khmer_gtts_audio,
        text,
        speed,
        to_voice_note,
        lang,
    )


__all__ = [
    "generate_khmer_gtts_audio",
    "generate_khmer_gtts_audio_async",
]
