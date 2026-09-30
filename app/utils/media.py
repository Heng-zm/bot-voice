"""Audio/video media utilities and FFmpeg format conversion helpers."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any

logger = logging.getLogger("app.utils.media")


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg binary is installed and executable in PATH."""
    return shutil.which("ffmpeg") is not None


async def convert_audio_to_ogg_opus(input_path: str, output_path: str) -> bool:
    """Convert audio file to Telegram-compatible voice note format (OGG Opus)."""
    if not is_ffmpeg_available():
        logger.warning("ffmpeg is not installed. Audio conversion skipped.")
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-vbr",
        "on",
        "-compression_level",
        "10",
        output_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0 and os.path.exists(output_path):
            return True
        logger.warning("ffmpeg failed (%d): %s", proc.returncode, stderr.decode(errors="ignore"))
        return False
    except Exception as exc:
        logger.error("ffmpeg conversion error: %s", exc)
        return False


__all__ = ["convert_audio_to_ogg_opus", "is_ffmpeg_available"]
