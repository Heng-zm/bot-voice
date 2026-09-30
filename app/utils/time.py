"""Local timezone conversion, dynamic offset calculations, and display helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

APP_TIMEZONE_NAME: str = (
    os.environ.get("APP_TIMEZONE")
    or os.environ.get("WEB_ADMIN_TIMEZONE")
    or "Asia/Phnom_Penh"
).strip() or "Asia/Phnom_Penh"

# Optional overrides; if empty or unset, labels are derived dynamically from the timezone.
_CONFIG_TZ_ALIAS: str = (os.environ.get("APP_TIMEZONE_ALIAS") or "").strip()
_CONFIG_UTC_LABEL: str = (os.environ.get("APP_TIMEZONE_UTC_LABEL") or "").strip()


def load_app_timezone(tz_name: str = APP_TIMEZONE_NAME) -> datetime.tzinfo:
    """Load ZoneInfo timezone with graceful fallback to standard UTC offsets."""
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        # Fallback to standard offset if known commonly configured default fails
        if tz_name in {"Asia/Phnom_Penh", "Asia/Bangkok", "Asia/Jakarta"}:
            return timezone(timedelta(hours=7), name="ICT")
        return UTC


APP_TIMEZONE: datetime.tzinfo = load_app_timezone()


def get_tz_offset_label(dt: datetime) -> str:
    """Return ISO-style UTC offset label (e.g., 'UTC+07:00', 'UTC-05:00', or 'UTC')."""
    if _CONFIG_UTC_LABEL:
        return _CONFIG_UTC_LABEL

    offset = dt.utcoffset()
    if offset is None or offset == timedelta(0):
        return "UTC"

    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60

    if minutes == 0:
        return f"UTC{sign}{hours}"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def get_tz_alias(dt: datetime) -> str:
    """Return timezone abbreviation (e.g., 'ICT', 'EDT', 'EST'), or fallback to offset label."""
    if _CONFIG_TZ_ALIAS:
        return _CONFIG_TZ_ALIAS
    return dt.tzname() or get_tz_offset_label(dt)


def local_now() -> datetime:
    """Return the current local datetime with tzinfo set to APP_TIMEZONE."""
    return datetime.now(APP_TIMEZONE)


def to_local_time(dt: datetime | None = None) -> datetime:
    """Convert an existing datetime (or current UTC time) to APP_TIMEZONE.

    Naive datetimes are assumed to be UTC before conversion.
    """
    if dt is None:
        return datetime.now(APP_TIMEZONE)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(APP_TIMEZONE)


def local_to_utc(dt: datetime) -> datetime:
    """Convert a local datetime to UTC.

    Naive datetimes are assumed to already be in APP_TIMEZONE.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=APP_TIMEZONE)
    return dt.astimezone(UTC)


def fmt_local_dt(dt: datetime | None = None, fmt: str = "%Y-%m-%d %I:%M %p") -> str:
    """Format datetime into a readable local string with dynamic alias and UTC offset."""
    loc_dt = to_local_time(dt)
    alias = get_tz_alias(loc_dt)
    utc_label = get_tz_offset_label(loc_dt)

    formatted_time = loc_dt.strftime(fmt)
    if alias and alias != utc_label:
        return f"{formatted_time} {alias} ({utc_label})"
    return f"{formatted_time} {utc_label}"


def fmt_local_time_hint() -> str:
    """Generate a dynamic contextual label describing the active timezone and offset."""
    now = local_now()
    alias = get_tz_alias(now)
    utc_label = get_tz_offset_label(now)
    region = APP_TIMEZONE_NAME.split("/")[-1].replace("_", " ")

    if alias and alias != utc_label:
        return f"{region} local time — AM/PM, {alias} ({utc_label})"
    return f"{region} local time — AM/PM, {utc_label}"


# Backward-compatible aliases for private-named imports
_load_app_timezone = load_app_timezone
_local_now = local_now
_to_local_time = to_local_time
_local_to_utc = local_to_utc
_fmt_local_dt = fmt_local_dt
_fmt_local_time_hint = fmt_local_time_hint

APP_TIMEZONE_ALIAS: str = get_tz_alias(local_now())
APP_TIMEZONE_UTC_LABEL: str = get_tz_offset_label(local_now())

__all__ = [
    "APP_TIMEZONE",
    "APP_TIMEZONE_ALIAS",
    "APP_TIMEZONE_NAME",
    "APP_TIMEZONE_UTC_LABEL",
    "_fmt_local_dt",
    "_fmt_local_time_hint",
    "_load_app_timezone",
    "_local_now",
    "_local_to_utc",
    "_to_local_time",
    "fmt_local_dt",
    "fmt_local_time_hint",
    "get_tz_alias",
    "get_tz_offset_label",
    "load_app_timezone",
    "local_now",
    "local_to_utc",
    "to_local_time",
]