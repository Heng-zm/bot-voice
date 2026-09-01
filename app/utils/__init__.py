"""Utilities package."""

from __future__ import annotations

from app.utils.file_io import _read_file_bytes_async, _write_file_bytes_sync
from app.utils.text import (
    TELEGRAM_MSG_LIMIT,
    html_safe_cut,
    paginate_html,
    paginate_pre_html,
    take_escaped_prefix,
    truncate_text,
)
from app.utils.time import _to_local_time

__all__ = [
    "TELEGRAM_MSG_LIMIT",
    "_read_file_bytes_async",
    "_to_local_time",
    "_write_file_bytes_sync",
    "html_safe_cut",
    "paginate_html",
    "paginate_pre_html",
    "take_escaped_prefix",
    "truncate_text",
]
