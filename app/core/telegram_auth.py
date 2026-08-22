"""Telegram Mini App authentication and Supabase-backed administrator policy."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl

from fastapi import Request

from app.services.settings.store import SettingsStore, get_settings_store

_ADMIN_KEY = "security:admin_user_ids:v2"


class TelegramInitDataError(ValueError):
    """Telegram Mini App initData is malformed, stale or has an invalid hash."""


class TelegramAdminStoreError(RuntimeError):
    """Administrator policy could not be loaded."""


@dataclass(frozen=True)
class TelegramMiniAppUser:
    id: int
    first_name: str = ""
    last_name: str = ""
    username: str = ""
    language_code: str = ""
    photo_url: str = ""
    is_premium: bool = False

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "username": self.username,
            "language_code": self.language_code,
            "photo_url": self.photo_url,
            "is_premium": self.is_premium,
        }


@dataclass(frozen=True)
class TelegramInitData:
    user: TelegramMiniAppUser
    auth_date: int
    query_id: str = ""


@dataclass(frozen=True)
class TelegramAdminSession:
    user: TelegramMiniAppUser
    auth_date: int
    query_id: str = ""


def _parse_fields(raw: str) -> dict[str, str]:
    pairs = parse_qsl(str(raw or ""), keep_blank_values=True, strict_parsing=False)
    fields: dict[str, str] = {}
    for key, value in pairs:
        if key in fields:
            raise TelegramInitDataError(f"Telegram initData contains duplicate field {key!r}.")
        fields[key] = value
    return fields


def validate_telegram_init_data(
    raw: str,
    bot_token: str,
    *,
    now: int | None = None,
    max_age_seconds: int = 3600,
) -> TelegramInitData:
    fields = _parse_fields(raw)
    received_hash = str(fields.pop("hash", "") or "").strip().lower()
    if not received_hash or len(received_hash) != 64:
        raise TelegramInitDataError("Telegram initData signature is missing or invalid.")
    token = str(bot_token or "").strip()
    if not token:
        raise TelegramInitDataError("Telegram bot token is not configured.")

    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", token.encode("utf-8"), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(received_hash, expected_hash):
        raise TelegramInitDataError("Telegram initData signature is invalid.")

    try:
        auth_date = int(fields.get("auth_date") or 0)
    except (TypeError, ValueError) as exc:
        raise TelegramInitDataError("Telegram initData auth_date is invalid.") from exc
    current = int(time.time() if now is None else now)
    if auth_date > current + 30:
        raise TelegramInitDataError("Telegram initData auth_date is in the future.")
    if auth_date <= 0 or current - auth_date > max(60, int(max_age_seconds)):
        raise TelegramInitDataError("Telegram initData has expired.")

    try:
        user_payload = json.loads(fields.get("user") or "{}")
        if not isinstance(user_payload, dict):
            raise TypeError("Telegram user payload must be an object.")
        user_id = int(user_payload.get("id") or 0)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TelegramInitDataError("Telegram initData user is invalid.") from exc
    if user_id <= 0:
        raise TelegramInitDataError("Telegram initData user ID is invalid.")

    user = TelegramMiniAppUser(
        id=user_id,
        first_name=str(user_payload.get("first_name") or ""),
        last_name=str(user_payload.get("last_name") or ""),
        username=str(user_payload.get("username") or ""),
        language_code=str(user_payload.get("language_code") or ""),
        photo_url=str(user_payload.get("photo_url") or ""),
        is_premium=bool(user_payload.get("is_premium")),
    )
    return TelegramInitData(
        user=user,
        auth_date=auth_date,
        query_id=str(fields.get("query_id") or ""),
    )


def telegram_init_data_from_request(request: Request) -> tuple[str, bool]:
    header = str(request.headers.get("x-telegram-init-data") or "").strip()
    if header:
        return header, True
    authorization = str(request.headers.get("authorization") or "").strip()
    for prefix in ("tma ", "bearer "):
        if authorization.lower().startswith(prefix):
            credential = authorization[len(prefix):].strip()
            if credential:
                return credential, True
    return "", False


class TelegramAdminAuthorizer:
    def __init__(self, *, cache_ttl_seconds: float = 5.0) -> None:
        self.store: SettingsStore = get_settings_store()
        self.fallback_admin_ids: frozenset[int] = frozenset()
        self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))
        self._cache: frozenset[int] | None = None
        self._cache_at = 0.0
        self._lock = asyncio.Lock()

    def configure(
        self,
        *,
        settings_store: SettingsStore | None = None,
        fallback_admin_ids: set[int] | frozenset[int] | list[int] | tuple[int, ...] = (),
        redis_client: Any | None = None,
        **_ignored: Any,
    ) -> TelegramAdminAuthorizer:
        # redis_client is accepted temporarily so older call sites do not crash;
        # it is intentionally unused after the single-process migration.
        del redis_client
        self.store = settings_store or get_settings_store()
        self.fallback_admin_ids = frozenset(
            int(value) for value in fallback_admin_ids if int(value) > 0
        )
        self.invalidate()
        return self

    def invalidate(self) -> None:
        self._cache = None
        self._cache_at = 0.0

    def is_admin_sync(self, user_id: int) -> bool:
        """Check the already-loaded admin snapshot without blocking the event loop.

        Telegram command guards are synchronous callbacks. Runtime startup and
        Mini App mutations keep this snapshot current, so these guards must not
        start a second event loop or perform synchronous Supabase I/O.
        """
        try:
            candidate = int(user_id)
        except (TypeError, ValueError):
            return False
        if candidate <= 0:
            return False
        snapshot = self._cache
        if snapshot is not None:
            return candidate in snapshot
        return candidate in self.fallback_admin_ids

    async def load_ids(self, *, force: bool = False) -> frozenset[int]:
        now = time.monotonic()
        if not force and self._cache is not None and now - self._cache_at < self.cache_ttl_seconds:
            return self._cache
        async with self._lock:
            now = time.monotonic()
            if not force and self._cache is not None and now - self._cache_at < self.cache_ttl_seconds:
                return self._cache
            payload = await self.store.get_json(_ADMIN_KEY, [])
            ids: set[int] = set()
            if isinstance(payload, list):
                for value in payload:
                    try:
                        admin_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if admin_id > 0:
                        ids.add(admin_id)
            if not ids and self.fallback_admin_ids:
                ids.update(self.fallback_admin_ids)
                await self.store.set_json(_ADMIN_KEY, sorted(ids))
            self._cache = frozenset(ids)
            self._cache_at = now
            return self._cache

    async def save_ids(self, ids: set[int] | frozenset[int], *, updated_by: int | None = None) -> bool:
        clean = sorted({int(value) for value in ids if int(value) > 0})
        persistent = await self.store.set_json(_ADMIN_KEY, clean, updated_by=updated_by)
        self._cache = frozenset(clean)
        self._cache_at = time.monotonic()
        return persistent

    async def authorize(self, raw_init_data: str, bot_token: str) -> TelegramAdminSession:
        data = validate_telegram_init_data(raw_init_data, bot_token)
        admin_ids = await self.load_ids()
        if data.user.id not in admin_ids:
            raise PermissionError("This Telegram account is not an administrator.")
        return TelegramAdminSession(
            user=data.user,
            auth_date=data.auth_date,
            query_id=data.query_id,
        )


_AUTHORIZER = TelegramAdminAuthorizer()


def configure_telegram_admin_authorizer(**kwargs: Any) -> TelegramAdminAuthorizer:
    return _AUTHORIZER.configure(**kwargs)


def get_telegram_admin_authorizer() -> TelegramAdminAuthorizer:
    return _AUTHORIZER


__all__ = [
    "TelegramAdminAuthorizer",
    "TelegramAdminSession",
    "TelegramAdminStoreError",
    "TelegramInitData",
    "TelegramInitDataError",
    "TelegramMiniAppUser",
    "configure_telegram_admin_authorizer",
    "get_telegram_admin_authorizer",
    "telegram_init_data_from_request",
    "validate_telegram_init_data",
]
