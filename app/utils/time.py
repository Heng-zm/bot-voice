"""Local timezone conversion and display helpers."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

APP_TIMEZONE_NAME = (
    os.environ.get("APP_TIMEZONE")
    or os.environ.get("WEB_ADMIN_TIMEZONE")
    or "Asia/Phnom_Penh"
).strip() or "Asia/Phnom_Penh"
APP_TIMEZONE_ALIAS = (
    os.environ.get("APP_TIMEZONE_ALIAS") or "ICT"
).strip() or "ICT"
APP_TIMEZONE_UTC_LABEL = (
    os.environ.get("APP_TIMEZONE_UTC_LABEL") or "UTC+7"
).strip() or "UTC+7"


def _load_app_timezone():
    try:
        return ZoneInfo(APP_TIMEZONE_NAME)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone(timedelta(hours=7), APP_TIMEZONE_ALIAS)


APP_TIMEZONE = _load_app_timezone()


def _local_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def _to_local_time(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(APP_TIMEZONE)


def _local_to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=APP_TIMEZONE)
    return dt.astimezone(timezone.utc)


def _fmt_local_dt(dt: datetime | None = None) -> str:
    local_dt = _to_local_time(dt or datetime.now(timezone.utc))
    return (
        f"{local_dt.strftime('%Y-%m-%d %I:%M %p')} "
        f"{APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})"
    )


def _fmt_local_time_hint() -> str:
    return (
        "Phnom Penh local time — AM/PM, "
        f"{APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})"
    )


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
]
