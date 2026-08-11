"""Bounded, redacted runtime telemetry for the admin Mini App."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

_MONITOR_STARTED_AT = time.time()
_RUNTIME_LOGS: deque[dict[str, Any]] = deque(maxlen=400)
_RUNTIME_LOGS_LOCK = threading.RLock()
_RUNTIME_LOG_HANDLER_INSTALLED = False
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"bot\d{5,}:[A-Za-z0-9_-]{20,}"), "<telegram-token>"),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "<api-key>"),
    (
        re.compile(
            r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
        ),
        "<jwt>",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer <hidden>"),
    (
        re.compile(
            r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)"
            r"(\s*[:=]\s*)([^\s,;&]+)"
        ),
        r"\1\2<hidden>",
    ),
    (
        re.compile(r"(?i)\b(redis|rediss|postgres|postgresql)://([^\s:/]+):([^\s@]+)@"),
        r"\1://\2:<hidden>@",
    ),
)


def sanitize_monitor_text(value: Any, *, limit: int = 1200) -> str:
    """Return a single-line, secret-redacted value safe for the admin browser."""

    clean_limit = max(1, min(5000, int(limit)))
    text = re.sub(r"\s+", " ", str(value or "").strip())
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:clean_limit]


class _RuntimeMonitorHandler(logging.Handler):
    """Capture INFO+ records without changing normal logging output."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.INFO:
                return
            entry = {
                "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "level": record.levelname if record.levelname in _LOG_LEVELS else "INFO",
                "source": sanitize_monitor_text(record.name or "runtime", limit=100),
                "message": sanitize_monitor_text(self.format(record)),
            }
            with _RUNTIME_LOGS_LOCK:
                _RUNTIME_LOGS.appendleft(entry)
        except Exception:
            # Observability must never interrupt application logging.
            return


def install_runtime_log_handler() -> None:
    """Install the process-wide collector exactly once."""

    global _RUNTIME_LOG_HANDLER_INSTALLED
    if _RUNTIME_LOG_HANDLER_INSTALLED:
        return
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_bot_runtime_monitor", False):
            _RUNTIME_LOG_HANDLER_INSTALLED = True
            return
    handler = _RuntimeMonitorHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._bot_runtime_monitor = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    _RUNTIME_LOG_HANDLER_INSTALLED = True


def runtime_log_snapshot(
    *,
    limit: int = 100,
    level: str = "",
    query: str = "",
) -> dict[str, Any]:
    """Return a filtered copy of recent runtime log records."""

    clean_limit = max(1, min(200, int(limit)))
    clean_level = str(level or "").strip().upper()
    if clean_level and clean_level not in _LOG_LEVELS:
        raise ValueError("Unsupported log level.")
    clean_query = sanitize_monitor_text(query, limit=128).lower()
    with _RUNTIME_LOGS_LOCK:
        captured = len(_RUNTIME_LOGS)
        entries = list(_RUNTIME_LOGS)
    if clean_level:
        minimum = logging._nameToLevel[clean_level]  # noqa: SLF001
        entries = [
            entry
            for entry in entries
            if logging._nameToLevel.get(str(entry.get("level")), 0) >= minimum  # noqa: SLF001
        ]
    if clean_query:
        entries = [
            entry
            for entry in entries
            if clean_query
            in f"{entry.get('source', '')} {entry.get('message', '')}".lower()
        ]
    return {
        "entries": entries[:clean_limit],
        "count": min(len(entries), clean_limit),
        "matched": len(entries),
        "captured": captured,
    }


def process_snapshot() -> dict[str, Any]:
    """Return safe process-local facts using only the Python standard library."""

    now = time.time()
    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "started_at": _MONITOR_STARTED_AT,
        "uptime_seconds": max(0, int(now - _MONITOR_STARTED_AT)),
        "threads": threading.active_count(),
    }
    try:
        import resource

        snapshot["max_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        snapshot["max_rss_kb"] = None
    try:
        snapshot["load_average"] = [round(float(value), 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        snapshot["load_average"] = []
    return snapshot


install_runtime_log_handler()


__all__ = [
    "install_runtime_log_handler",
    "process_snapshot",
    "runtime_log_snapshot",
    "sanitize_monitor_text",
]
