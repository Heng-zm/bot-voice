"""Logging utilities and filters for production stability."""

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
    """Demote transient Telegram upstream polling network errors to clean warnings without tracebacks."""

    def __init__(self, suppression_window: float = 20.0) -> None:
        super().__init__()
        self._suppression_window = float(suppression_window)
        self._last_logged: float = 0.0
        self._suppressed_count: int = 0
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered_msg = record.getMessage().lower()
        except Exception:
            rendered_msg = str(record.msg or "").lower()

        # Check if record originates from Telegram update polling
        if not any(phrase in rendered_msg for phrase in _POLLING_TRIGGER_PHRASES):
            return True

        exc = record.exc_info[1] if record.exc_info else None
        exc_str = str(exc or "") if exc is not None else rendered_msg

        if not is_transient_network_error(exc_str):
            return True

        with self._lock:
            now = time.monotonic()
            if (now - self._last_logged) < self._suppression_window:
                self._suppressed_count += 1
                return False

            suppressed_note = (
                f" (+{self._suppressed_count} consecutive retries suppressed)"
                if self._suppressed_count > 0
                else ""
            )
            self._suppressed_count = 0
            self._last_logged = now

        # Demote level to WARNING and strip traceback
        record.levelno = logging.WARNING
        record.levelname = "WARNING"
        record.msg = (
            f"Telegram polling transient network hiccup ({exc or 'NetworkError'})"
            f"{suppressed_note}; retrying automatically in background..."
        )
        record.args = None
        record.exc_info = None
        record.stack_info = None
        return True


def install_telegram_polling_filter(logger_names: tuple[str, ...] = _DEFAULT_POLLING_LOGGERS) -> None:
    """Install the polling filter on relevant Telegram Updater/Application loggers."""
    filt = TelegramPollingNetworkFilter()
    for name in logger_names:
        log_instance = logging.getLogger(name)
        if not any(isinstance(f, TelegramPollingNetworkFilter) for f in log_instance.filters):
            log_instance.addFilter(filt)


class FlushStreamHandler(logging.StreamHandler):
    """Stream handler that ensures non-blocking encoding fallbacks and immediate flushes."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            enc = getattr(stream, "encoding", None) or "utf-8"
            terminator = self.terminator or "\n"
            try:
                stream.write(msg + terminator)
            except UnicodeEncodeError:
                safe_msg = msg.encode(enc, errors="replace").decode(enc)
                stream.write(safe_msg + terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


_LOGGING_LOCK = threading.Lock()
_SERVER_LOGGING_CONFIGURED = False


def configure_server_logging(
    level: int = logging.INFO,
    *,
    force: bool = False,
) -> None:
    """Configure unbuffered real-time stdout console logging across server and bot components."""
    global _SERVER_LOGGING_CONFIGURED

    with _LOGGING_LOCK:
        if _SERVER_LOGGING_CONFIGURED and not force:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        has_flush_handler = any(isinstance(h, FlushStreamHandler) for h in root_logger.handlers)
        if not has_flush_handler:
            formatter = logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler = FlushStreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            handler.setLevel(level)
            root_logger.addHandler(handler)

        # Application modules set to active level
        app_loggers = (
            "app",
            "app.server",
            "app.webhook",
            "app.bot",
            "app.dispatcher",
            "app.telemetry",
            "core",
        )
        for app_name in app_loggers:
            logging.getLogger(app_name).setLevel(level)

        # Demote noisy third-party libraries to WARNING
        noisy_libs = (
            "httpx",
            "httpcore",
            "urllib3",
            "telegram.ext",
            "google",
            "asyncio",
            "aiogram.event",
        )
        for noisy in noisy_libs:
            logging.getLogger(noisy).setLevel(logging.WARNING)

        install_telegram_polling_filter()
        _SERVER_LOGGING_CONFIGURED = True


__all__ = [
    "FlushStreamHandler",
    "TRANSIENT_NETWORK_ERRORS",
    "TelegramPollingNetworkFilter",
    "configure_server_logging",
    "install_telegram_polling_filter",
    "is_transient_network_error",
]