"""Atomic synchronous and asynchronous binary file utilities."""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import suppress


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


__all__ = [
    "_read_file_bytes_async",
    "_read_file_bytes_sync",
    "_write_file_bytes_sync",
]
