"""Typed database models and data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class UserPrefDict(TypedDict, total=False):
    user_id: int
    gender: str
    speed: float
    tts_model: str
    bot_mode: str
    username: str | None
    first_name: str | None
    last_name: str | None
    language_code: str | None
    updated_at: str


class BotSettingDict(TypedDict, total=False):
    setting_key: str
    setting_val: str
    updated_at: str


@dataclass
class BroadcastSchedule:
    id: int
    creator_id: int
    message_text: str
    scheduled_time_utc: str
    target_group: str
    status: str
    created_at: str


__all__ = ["BotSettingDict", "BroadcastSchedule", "UserPrefDict"]
