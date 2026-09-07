"""Utilities package."""

from __future__ import annotations

from app.utils.file_io import _read_file_bytes_async, _write_file_bytes_sync
from app.utils.logging import (
    TelegramPollingNetworkFilter,
    install_telegram_polling_filter,
    is_transient_network_error,
)
from app.utils.text import (
    TELEGRAM_CAPTION_LIMIT,
    TELEGRAM_MSG_LIMIT,
    format_safe_media_caption,
    html_safe_cut,
    paginate_html,
    paginate_pre_html,
    take_escaped_prefix,
    truncate_text,
)
from app.utils.time import _to_local_time

__all__ = [
    "TELEGRAM_CAPTION_LIMIT",
    "TELEGRAM_MSG_LIMIT",
    "TelegramPollingNetworkFilter",
    "_read_file_bytes_async",
    "_to_local_time",
    "_write_file_bytes_sync",
    "format_safe_media_caption",
    "html_safe_cut",
    "install_telegram_polling_filter",
    "is_transient_network_error",
    "paginate_html",
    "paginate_pre_html",
    "take_escaped_prefix",
    "truncate_text",
]
