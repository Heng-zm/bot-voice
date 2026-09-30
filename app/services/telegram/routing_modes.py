"""User UI routing states and interactive mode session management."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# Standard mode identifiers used across menus and handlers
MODE_AI_CHAT = "ai_chat"
MODE_REMOVE_BG = "remove_bg"
MODE_OCR = "ocr"
MODE_UPSCALE = "upscale"
MODE_PDF = "pdf"
MODE_EXCHANGE = "exchange"
MODE_FOOD = "food"
MODE_HOMEWORK = "homework"
MODE_HOMEWORK_EXPLAIN = "homework_explain"
MODE_DEPOSIT = "deposit"
MODE_SERVICES = "services"

VALID_MODES: frozenset[str] = frozenset({
    MODE_AI_CHAT,
    MODE_REMOVE_BG,
    MODE_OCR,
    MODE_UPSCALE,
    MODE_PDF,
    MODE_EXCHANGE,
    MODE_FOOD,
    MODE_HOMEWORK,
    MODE_HOMEWORK_EXPLAIN,
    MODE_DEPOSIT,
    MODE_SERVICES,
})

# Inactive mode time-to-live: 30 minutes
DEFAULT_MODE_TTL_SECONDS = 1800.0
MAX_STORED_SESSIONS = 10_000

# Thread-safe in-memory store: user_id -> (mode_name, last_active_monotonic)
_USER_MODES: dict[int, tuple[str, float]] = {}
_MODES_LOCK = threading.RLock()


def _clean_user_id(user_id: Any) -> int | None:
    """Normalize and validate incoming user ID as a positive integer."""
    try:
        val = int(str(user_id).strip())
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


def _clean_mode(mode: Any) -> str | None:
    """Normalize mode string."""
    if not mode:
        return None
    clean = str(mode).strip().lower()
    return clean if clean else None


def set_user_mode(
    user_id: int | str,
    mode: str | None,
    *,
    ttl_seconds: float = DEFAULT_MODE_TTL_SECONDS,
) -> None:
    """Set or clear a user's active UI interaction mode."""
    uid = _clean_user_id(user_id)
    if uid is None:
        return

    clean_mode = _clean_mode(mode)
    now = time.monotonic()

    with _MODES_LOCK:
        if clean_mode is not None:
            # Enforce max storage capacity to prevent memory bloat
            if len(_USER_MODES) >= MAX_STORED_SESSIONS:
                prune_expired_user_modes(ttl_seconds=ttl_seconds)

            _USER_MODES[uid] = (clean_mode, now)
        else:
            _USER_MODES.pop(uid, None)


def get_user_mode(
    user_id: int | str,
    *,
    ttl_seconds: float = DEFAULT_MODE_TTL_SECONDS,
    touch: bool = True,
) -> str | None:
    """Retrieve active user mode, returning None if expired or unset.

    Parameters:
        user_id: Telegram user ID.
        ttl_seconds: Maximum inactivity duration before mode expires.
        touch: If True, extends the active session window on access.
    """
    uid = _clean_user_id(user_id)
    if uid is None:
        return None

    now = time.monotonic()
    with _MODES_LOCK:
        entry = _USER_MODES.get(uid)
        if entry is None:
            return None

        current_mode, last_active = entry
        if (now - last_active) > ttl_seconds:
            # Mode expired due to inactivity
            _USER_MODES.pop(uid, None)
            return None

        if touch:
            _USER_MODES[uid] = (current_mode, now)

        return current_mode


def clear_user_mode(user_id: int | str) -> None:
    """Explicit alias to clear a user's current mode."""
    set_user_mode(user_id, None)


def is_in_mode(user_id: int | str, expected_mode: str) -> bool:
    """Check whether a user is currently in a specific active mode."""
    clean_target = _clean_mode(expected_mode)
    if not clean_target:
        return False
    return get_user_mode(user_id) == clean_target


# Alias for backward compatibility
is_user_in_mode = is_in_mode


def get_user_mode_with_elapsed(user_id: int | str) -> tuple[str | None, float]:
    """Retrieve the active mode alongside the seconds elapsed since activation."""
    uid = _clean_user_id(user_id)
    if uid is None:
        return None, 0.0

    now = time.monotonic()
    with _MODES_LOCK:
        entry = _USER_MODES.get(uid)
        if entry is None:
            return None, 0.0

        current_mode, last_active = entry
        elapsed = max(0.0, now - last_active)
        if elapsed > DEFAULT_MODE_TTL_SECONDS:
            _USER_MODES.pop(uid, None)
            return None, 0.0

        return current_mode, elapsed


def prune_expired_user_modes(ttl_seconds: float = DEFAULT_MODE_TTL_SECONDS) -> int:
    """Sweep and remove inactive user mode sessions. Returns count of purged sessions."""
    now = time.monotonic()
    evicted = 0

    with _MODES_LOCK:
        expired_ids = [
            uid
            for uid, (_, last_active) in _USER_MODES.items()
            if (now - last_active) > ttl_seconds
        ]
        for uid in expired_ids:
            _USER_MODES.pop(uid, None)
            evicted += 1

    if evicted > 0:
        logger.debug("Pruned %d expired user mode sessions.", evicted)
    return evicted


def reset_all_user_modes() -> None:
    """Clear all active user modes (used during shutdown, migrations, or testing)."""
    with _MODES_LOCK:
        _USER_MODES.clear()


def get_active_modes_count() -> int:
    """Return the total number of currently active, non-expired sessions."""
    prune_expired_user_modes()
    with _MODES_LOCK:
        return len(_USER_MODES)


def snapshot_user_modes() -> dict[int, str]:
    """Return a thread-safe snapshot mapping user_id -> active mode."""
    prune_expired_user_modes()
    with _MODES_LOCK:
        return {uid: mode for uid, (mode, _) in _USER_MODES.items()}


# Backward compatibility dictionary facade
class _UserModesDictProxy(Mapping[int, str]):
    def __getitem__(self, key: int) -> str:
        mode = get_user_mode(key, touch=False)
        if mode is None:
            raise KeyError(key)
        return mode

    def __iter__(self):
        return iter(snapshot_user_modes())

    def __len__(self) -> int:
        return get_active_modes_count()

    def get(self, key: int, default: Any = None) -> Any:  # type: ignore[override]
        res = get_user_mode(key, touch=False)
        return res if res is not None else default

    def pop(self, key: int, default: Any = None) -> Any:
        with _MODES_LOCK:
            val = get_user_mode(key, touch=False)
            clear_user_mode(key)
            return val if val is not None else default

    def __setitem__(self, key: int, value: str | None) -> None:
        set_user_mode(key, value)


user_modes = _UserModesDictProxy()


__all__ = [
    "DEFAULT_MODE_TTL_SECONDS",
    "MODE_AI_CHAT",
    "MODE_DEPOSIT",
    "MODE_EXCHANGE",
    "MODE_FOOD",
    "MODE_HOMEWORK",
    "MODE_HOMEWORK_EXPLAIN",
    "MODE_OCR",
    "MODE_PDF",
    "MODE_REMOVE_BG",
    "MODE_SERVICES",
    "MODE_UPSCALE",
    "VALID_MODES",
    "clear_user_mode",
    "get_active_modes_count",
    "get_user_mode",
    "get_user_mode_with_elapsed",
    "is_in_mode",
    "is_user_in_mode",
    "prune_expired_user_modes",
    "reset_all_user_modes",
    "set_user_mode",
    "snapshot_user_modes",
    "user_modes",
]