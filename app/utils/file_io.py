"""Atomic synchronous and asynchronous binary file and temporary file utilities."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

_TMP_PREFIX: Final[str] = "tgbot_"
_TEMP_DIR_CACHE: str | None = None
_TEMP_DIR_CACHE_LOCK = threading.Lock()

_STALE_TEMP_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".ogg", ".jpg", ".jpeg", ".png", ".webp",
    ".mp3", ".wav", ".mp4", ".m4a", ".flac", ".aac", ".opus", ".webm",
})

FilePath = str | os.PathLike[str]


def _to_path_str(p: FilePath) -> str:
    """Normalize PathLike or str to an absolute string path."""
    return os.path.abspath(os.fspath(p))


def read_file_bytes_sync(path: FilePath, *, max_bytes: int | None = None) -> bytes:
    """Read file content deterministically with an optional size limit."""
    target_path = _to_path_str(path)
    limit = None if max_bytes is None else max(0, int(max_bytes))

    with open(target_path, "rb") as handle:
        if limit is None:
            return handle.read()

        chunk_size = 64 * 1024  # 64 KB chunks
        buffer = bytearray()
        while True:
            read_len = min(chunk_size, (limit + 1) - len(buffer))
            if read_len <= 0:
                raise ValueError(f"File too large. Max {limit} bytes.")
            chunk = handle.read(read_len)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise ValueError(f"File too large. Max {limit} bytes.")

    return bytes(buffer)


def write_file_bytes_sync(path: FilePath, data: bytes) -> None:
    """Atomically write binary data to a file using an fsynced temporary sibling file."""
    target_path = _to_path_str(path)
    if not target_path:
        raise ValueError("Output path is required.")

    payload = bytes(data or b"")
    parent = os.path.dirname(target_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    prefix = f".{os.path.basename(target_path) or 'output'}."
    fd, temporary_path = tempfile.mkstemp(
        prefix=prefix,
        suffix=".tmp",
        dir=parent or None,
    )

    fd_owned = False
    try:
        with os.fdopen(fd, "wb") as handle:
            fd_owned = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        # Windows-resilient atomic replace
        _replace_atomic(temporary_path, target_path)
    except BaseException:
        if not fd_owned:
            with suppress(OSError):
                os.close(fd)
        with suppress(OSError):
            os.unlink(temporary_path)
        raise


def _replace_atomic(src: str, dst: str, max_retries: int = 3) -> None:
    """Attempt atomic file replacement with short-interval retries for file locks."""
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            # Common on Windows if target was recently accessed or indexed
            if attempt < max_retries - 1 and sys.platform == "win32":
                time.sleep(0.02 * (attempt + 1))
                continue
            raise


async def read_file_bytes_async(
    path: FilePath,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Read binary file bytes without blocking the asyncio event loop."""
    return await asyncio.to_thread(
        read_file_bytes_sync,
        path,
        max_bytes=max_bytes,
    )


async def write_file_bytes_async(path: FilePath, data: bytes) -> None:
    """Write binary file bytes atomically without blocking the asyncio event loop."""
    await asyncio.to_thread(write_file_bytes_sync, path, data)


def get_temp_dir() -> str:
    """Return the dedicated bot temp directory, initializing it once per process."""
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


def make_temp_file(suffix: str = "") -> str:
    """Create an empty temporary file with the bot prefix in the temp directory."""
    temp_dir = get_temp_dir()
    normalized_suffix = suffix if (not suffix or suffix.startswith(".")) else f".{suffix}"
    fd, path = tempfile.mkstemp(suffix=normalized_suffix, prefix=_TMP_PREFIX, dir=temp_dir)
    os.close(fd)
    return path


def make_temp_ogg() -> str:
    """Create a temporary OGG voice/audio container."""
    return make_temp_file(".ogg")


def make_temp_audio(suffix: str = ".mp3") -> str:
    """Create a temporary audio container (default .mp3)."""
    return make_temp_file(suffix)


def make_temp_img(suffix: str = ".jpg") -> str:
    """Create a temporary image container (default .jpg)."""
    return make_temp_file(suffix)


def cleanup_files(*paths: Any) -> None:
    """Safely remove one or more files or PathLike objects without raising errors."""
    for p in paths:
        if not p:
            continue
        try:
            if isinstance(p, (str, bytes, os.PathLike)):
                path_str = _to_path_str(p)
                if os.path.isfile(path_str):
                    os.remove(path_str)
        except OSError as exc:
            logger.debug("Temp file cleanup skipped for %s: %s", p, exc)


def sweep_stale_temp_files(max_age_seconds: float = 7200.0) -> int:
    """Delete bot temporary files exceeding max_age_seconds matching known media formats."""
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


# Backward-compatible aliases
_read_file_bytes_sync = read_file_bytes_sync
_write_file_bytes_sync = write_file_bytes_sync
_read_file_bytes_async = read_file_bytes_async
_write_file_bytes_async = write_file_bytes_async

__all__ = [
    "_read_file_bytes_async",
    "_read_file_bytes_sync",
    "_write_file_bytes_async",
    "_write_file_bytes_sync",
    "cleanup_files",
    "get_temp_dir",
    "make_temp_audio",
    "make_temp_file",
    "make_temp_img",
    "make_temp_ogg",
    "read_file_bytes_async",
    "read_file_bytes_sync",
    "sweep_stale_temp_files",
    "write_file_bytes_async",
    "write_file_bytes_sync",
]