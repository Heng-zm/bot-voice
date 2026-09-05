"""Logging utilities and filters for production stability."""

from __future__ import annotations

import logging
from typing import Any

TRANSIENT_NETWORK_ERRORS = (
    "bad gateway",
    "gateway timeout",
    "timed out",
    "timedout",
    "readtimeout",
    "connect timeout",
    "server disconnected",
    "connection reset",
    "network is unreachable",
    "502",
    "504",
)


def is_transient_network_error(exc_or_msg: Any) -> bool:
    """Check if an error or exception string corresponds to a transient upstream network glitch."""
    msg = str(exc_or_msg or "").lower()
    return any(term in msg for term in TRANSIENT_NETWORK_ERRORS)


class TelegramPollingNetworkFilter(logging.Filter):
    """Demote transient Telegram upstream polling network errors to clean warnings without tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg_str = str(record.msg or "")
        if "Exception happened while polling for updates" in msg_str or "Exception happened in polling action" in msg_str:
            exc = record.exc_info[1] if record.exc_info else None
            exc_str = str(exc or "")
            if is_transient_network_error(exc_str):
                record.levelno = logging.WARNING
                record.levelname = "WARNING"
                record.msg = f"Telegram polling transient network hiccup ({exc_str or 'NetworkError'}); retrying automatically in background..."
                record.exc_info = None
        return True


def install_telegram_polling_filter() -> None:
    """Install the polling filter on relevant Telegram Updater loggers."""
    filt = TelegramPollingNetworkFilter()
    for name in ("telegram.ext.Updater", "telegram.ext._utils.networkloop", "telegram.ext"):
        log_instance = logging.getLogger(name)
        if not any(isinstance(f, TelegramPollingNetworkFilter) for f in log_instance.filters):
            log_instance.addFilter(filt)


__all__ = [
    "TRANSIENT_NETWORK_ERRORS",
    "TelegramPollingNetworkFilter",
    "install_telegram_polling_filter",
    "is_transient_network_error",
]
