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
    """Safely write data to a temporary file in the same directory and atomic-replace."""
    target_path = _to_path_str(path)
    directory = os.path.dirname(target_path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="atomic_", dir=directory)
    try:
        with open(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        with suppress(OSError):
            os.remove(tmp_path)
        raise


async def read_file_bytes_async(path: FilePath, *, max_bytes: int | None = None) -> bytes:
    """Asynchronously read file content without blocking event loop."""
    return await asyncio.to_thread(read_file_bytes_sync, path, max_bytes=max_bytes)


async def write_file_bytes_async(path: FilePath, data: bytes) -> None:
    """Asynchronously write file content using atomic file replacement."""
    await asyncio.to_thread(write_file_bytes_sync, path, data)


def get_temp_dir() -> str:
    """Return standard writable temporary directory for Bot Voice operations."""
    global _TEMP_DIR_CACHE
    with _TEMP_DIR_CACHE_LOCK:
        if _TEMP_DIR_CACHE is not None:
            return _TEMP_DIR_CACHE

        system_tmp = tempfile.gettempdir()
        bot_tmp = os.path.join(system_tmp, "bot_voice")
        with suppress(OSError):
            os.makedirs(bot_tmp, exist_ok=True)
            _TEMP_DIR_CACHE = bot_tmp
            return bot_tmp

        _TEMP_DIR_CACHE = system_tmp
        return system_tmp


def make_temp_file(suffix: str = ".tmp") -> str:
    """Create an empty temporary file and return its absolute path."""
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    fd, path = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=clean_suffix, dir=get_temp_dir())
    os.close(fd)
    return path


def make_temp_ogg() -> str:
    return make_temp_file(".ogg")


def make_temp_audio() -> str:
    return make_temp_file(".mp3")


def make_temp_img() -> str:
    return make_temp_file(".jpg")


def cleanup_files(*paths: FilePath | None) -> None:
    """Safely remove files if they exist."""
    for p in paths:
        if not p:
            continue
        try:
            path_str = _to_path_str(p)
            if os.path.exists(path_str):
                os.remove(path_str)
        except OSError as exc:
            logger.debug("Temp file cleanup skipped for %s: %s", p, exc)


def safe_delete_file(path: FilePath | None) -> None:
    """Safely remove a file if it exists."""
    cleanup_files(path)


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


__all__ = [
    "cleanup_files",
    "get_temp_dir",
    "make_temp_audio",
    "make_temp_file",
    "make_temp_img",
    "make_temp_ogg",
    "read_file_bytes_async",
    "read_file_bytes_sync",
    "safe_delete_file",
    "sweep_stale_temp_files",
    "write_file_bytes_async",
    "write_file_bytes_sync",
]