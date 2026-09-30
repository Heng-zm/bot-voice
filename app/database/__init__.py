"""Database clients, models, batcher, and repositories package."""

from __future__ import annotations

from app.database.batcher import DatabaseBatcher, get_database_batcher
from app.database.client import get_sqlite_connection, get_supabase_client
from app.database.models import BotSettingDict, BroadcastSchedule, UserPrefDict
from app.database.repositories.donations import DonationStore, donation_store, get_donation_store
from app.database.repositories.requests import RequestLogRepository, get_request_repository
from app.database.repositories.settings import SettingsStore, get_settings_store
from app.database.repositories.users import (
    UserPrefsCache,
    get_global_user_prefs_cache,
    get_user_prefs_async,
    set_user_pref_async,
)

__all__ = [
    "BotSettingDict",
    "BroadcastSchedule",
    "DatabaseBatcher",
    "DonationStore",
    "RequestLogRepository",
    "SettingsStore",
    "UserPrefDict",
    "UserPrefsCache",
    "donation_store",
    "get_database_batcher",
    "get_donation_store",
    "get_global_user_prefs_cache",
    "get_request_repository",
    "get_settings_store",
    "get_sqlite_connection",
    "get_supabase_client",
    "get_user_prefs_async",
    "set_user_pref_async",
]
