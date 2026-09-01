"""Atomic synchronous and asynchronous binary file and temporary file utilities."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

_TMP_PREFIX = "tgbot_"
_TEMP_DIR_CACHE: str | None = None
_TEMP_DIR_CACHE_LOCK = threading.Lock()
_STALE_TEMP_EXTENSIONS = frozenset({
    ".ogg", ".jpg", ".jpeg", ".png", ".webp",
    ".mp3", ".wav", ".mp4", ".m4a", ".flac", ".aac", ".opus", ".webm",
})


def _read_file_bytes_sync(path: str, *, max_bytes: int | None = None) -> bytes:
    """Read a file with deterministic cleanup and an optional hard limit."""
    limit = None if max_bytes is None else max(0, int(max_bytes))
    with open(path, "rb") as handle:
        if limit is None:
            return handle.read()
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"File too large. Max {limit} bytes.")
    return data


def _write_file_bytes_sync(path: str, data: bytes) -> None:
    """Atomically replace a binary file using a temporary sibling file."""
    if not path:
        raise ValueError("Output path is required.")
    payload = bytes(data or b"")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path) or 'output'}.",
        suffix=".tmp",
        dir=parent or None,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        with suppress(OSError):
            os.unlink(temporary_path)
        raise


async def _read_file_bytes_async(
    path: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Move blocking disk reads off the event loop."""
    return await asyncio.to_thread(
        _read_file_bytes_sync,
        path,
        max_bytes=max_bytes,
    )


def get_temp_dir() -> str:
    """Return the bot temp directory, creating it once per process."""
    global _TEMP_DIR_CACHE
    configured = os.environ.get("BOT_TMP_DIR") or tempfile.gettempdir()
    temp_dir = os.path.abspath(configured)
    cached = _TEMP_DIR_CACHE
    if cached == temp_dir:
        return cached
    with _TEMP_DIR_CACHE_LOCK:
        if temp_dir != _TEMP_DIR_CACHE:
            os.makedirs(temp_dir, exist_ok=True)
            _TEMP_DIR_CACHE = temp_dir
        return _TEMP_DIR_CACHE


def make_temp_file(suffix: str) -> str:
    """Create a temporary file in the dedicated bot temp directory."""
    temp_dir = get_temp_dir()
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=_TMP_PREFIX, dir=temp_dir)
    os.close(fd)
    return path


def make_temp_ogg() -> str:
    """Create a temporary OGG file."""
    return make_temp_file(".ogg")


def make_temp_audio(suffix: str = ".mp3") -> str:
    """Create a temporary audio file."""
    return make_temp_file(suffix if suffix.startswith(".") else f".{suffix}")


def make_temp_img(suffix: str = ".jpg") -> str:
    """Create a temporary image file."""
    return make_temp_file(suffix if suffix.startswith(".") else f".{suffix}")


def cleanup_files(*paths: Any) -> None:
    """Remove one or more files safely without raising errors."""
    for p in paths:
        if not p or not isinstance(p, str):
            continue
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError as exc:
            logger.debug("Temp file cleanup skipped for %s: %s", p, exc)


def sweep_stale_temp_files(max_age_seconds: float = 7200.0) -> int:
    """Delete old bot temporary files matching prefix and known media extensions."""
    temp_dir = get_temp_dir()
    cutoff = time.time() - max(60.0, float(max_age_seconds))
    removed = 0

    try:
        with os.scandir(temp_dir) as entries:
            for entry in entries:
                try:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not name.startswith(_TMP_PREFIX):
                        continue
                    _root, ext = os.path.splitext(name)
                    if ext.lower() not in _STALE_TEMP_EXTENSIONS:
                        continue
                    stat = entry.stat()
                    if stat.st_mtime < cutoff:
                        os.remove(entry.path)
                        removed += 1
                except OSError:
                    continue
    except OSError as exc:
        logger.warning("Failed scanning temp directory %s: %s", temp_dir, exc)

    return removed


__all__ = [
    "_read_file_bytes_async",
    "_read_file_bytes_sync",
    "_write_file_bytes_sync",
    "cleanup_files",
    "get_temp_dir",
    "make_temp_audio",
    "make_temp_file",
    "make_temp_img",
    "make_temp_ogg",
    "sweep_stale_temp_files",
]
