"""Centralized logging configuration and filters for production stability."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Final

TRANSIENT_NETWORK_ERRORS: Final[tuple[str, ...]] = (
    "bad gateway",
    "gateway timeout",
    "timed out",
    "timedout",
    "readtimeout",
    "connect timeout",
    "server disconnected",
    "connection reset",
    "network is unreachable",
    "remote protocol error",
    "broken pipe",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "ssl error",
    "handshake failed",
    "tls handshake",
    "eof occurred in violation of protocol",
    "502",
    "503",
    "504",
)

_POLLING_TRIGGER_PHRASES: Final[tuple[str, ...]] = (
    "exception happened while polling",
    "exception happened in polling action",
    "error while checking for updates",
    "failed to fetch updates",
    "conflict: terminated by other getupdates",
)

_DEFAULT_POLLING_LOGGERS: Final[tuple[str, ...]] = (
    "telegram.ext.Updater",
    "telegram.ext.Application",
    "telegram.ext._application",
    "telegram.ext._utils.networkloop",
    "telegram.ext",
    "aiogram.dispatcher",
)


def is_transient_network_error(exc_or_msg: Any) -> bool:
    """Check if an error or exception string corresponds to a transient upstream network glitch."""
    if exc_or_msg is None:
        return False
    msg = str(exc_or_msg).lower()
    return any(term in msg for term in TRANSIENT_NETWORK_ERRORS)


class TelegramPollingNetworkFilter(logging.Filter):
    """Filter out noisy, repetitive transient network errors from Telegram polling loggers."""

    def __init__(self, throttle_interval: float = 30.0) -> None:
        super().__init__()
        self.throttle_interval = throttle_interval
        self._last_logged_at: dict[str, float] = {}
        self._suppressed_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage() or "").lower()

        is_polling_issue = any(phrase in msg for phrase in _POLLING_TRIGGER_PHRASES)
        is_transient = is_transient_network_error(msg)
        if record.exc_info and record.exc_info[1]:
            is_transient = is_transient or is_transient_network_error(record.exc_info[1])

        if not (is_polling_issue or is_transient):
            return True

        key = record.name
        now = time.monotonic()

        with self._lock:
            last = self._last_logged_at.get(key, 0.0)
            if now - last < self.throttle_interval:
                self._suppressed_counts[key] = self._suppressed_counts.get(key, 0) + 1
                return False

            suppressed = self._suppressed_counts.pop(key, 0)
            self._last_logged_at[key] = now

        if suppressed > 0:
            record.msg = f"{record.msg} [Suppressed {suppressed} duplicate transient network logs]"
        return True


def install_telegram_polling_filter(
    logger_names: tuple[str, ...] = _DEFAULT_POLLING_LOGGERS,
    throttle_interval: float = 30.0,
) -> TelegramPollingNetworkFilter:
    """Attach the TelegramPollingNetworkFilter to all relevant polling loggers."""
    flt = TelegramPollingNetworkFilter(throttle_interval=throttle_interval)
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(f, TelegramPollingNetworkFilter) for f in logger.filters):
            logger.addFilter(flt)
    return flt


def configure_server_logging(level: int = logging.INFO) -> None:
    """Initialize structured stdout logging for the application and API server."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


__all__ = [
    "TRANSIENT_NETWORK_ERRORS",
    "TelegramPollingNetworkFilter",
    "configure_server_logging",
    "install_telegram_polling_filter",
    "is_transient_network_error",
]
