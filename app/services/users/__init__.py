"""Users and preferences service package."""

from __future__ import annotations

from app.services.users.prefs import (
    DEFAULT_GENDER,
    DEFAULT_USER_PREFS,
    SPEED_MAX,
    SPEED_MIN,
    UserPrefsCache,
    get_global_user_prefs_cache,
    normalize_user_prefs,
)

__all__ = [
    "DEFAULT_GENDER",
    "DEFAULT_USER_PREFS",
    "SPEED_MAX",
    "SPEED_MIN",
    "UserPrefsCache",
    "get_global_user_prefs_cache",
    "normalize_user_prefs",
]
