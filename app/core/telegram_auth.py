"""Telegram Mini App init-data validation and administrator authorization."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl

from fastapi import Request

logger = logging.getLogger(__name__)

_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_INIT_DATA_BYTES = 16_384
_MAX_FIELDS = 32
_MAX_USER_JSON_BYTES = 8_192
_DEFAULT_MAX_AGE_SECONDS = 3_600
_DEFAULT_FUTURE_SKEW_SECONDS = 30


class TelegramInitDataError(ValueError):
    """Raised when Telegram Mini App data is malformed or untrusted."""


class TelegramAdminStoreError(RuntimeError):
    """Raised when the Redis-backed administrator allowlist is unavailable."""


@dataclass(frozen=True, slots=True)
class TelegramMiniAppUser:
    id: int
    first_name: str
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


@dataclass(frozen=True, slots=True)
class ValidatedTelegramInitData:
    user: TelegramMiniAppUser
    auth_date: int
    query_id: str
    fields: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TelegramAdminSession:
    user: TelegramMiniAppUser
    auth_date: int
    query_id: str


def _required_text(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise TelegramInitDataError(f"Telegram user field {field!r} must be text.")
    value = value.strip()
    if not value:
        raise TelegramInitDataError(f"Telegram user field {field!r} is required.")
    if len(value) > max_length:
        raise TelegramInitDataError(f"Telegram user field {field!r} is too long.")
    return value


def _optional_text(value: Any, field: str, *, max_length: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TelegramInitDataError(f"Telegram user field {field!r} must be text.")
    value = value.strip()
    if len(value) > max_length:
        raise TelegramInitDataError(f"Telegram user field {field!r} is too long.")
    return value


def _parse_user(raw_user: str) -> TelegramMiniAppUser:
    if not raw_user or len(raw_user.encode("utf-8")) > _MAX_USER_JSON_BYTES:
        raise TelegramInitDataError("Telegram Mini App user data is missing or too large.")
    try:
        value = json.loads(raw_user)
    except (TypeError, ValueError) as exc:
        raise TelegramInitDataError("Telegram Mini App user data is invalid JSON.") from exc
    if not isinstance(value, dict):
        raise TelegramInitDataError("Telegram Mini App user data must be a JSON object.")
    try:
        user_id = int(value.get("id"))
    except (TypeError, ValueError) as exc:
        raise TelegramInitDataError("Telegram Mini App user id is invalid.") from exc
    if user_id <= 0 or user_id >= 2**63:
        raise TelegramInitDataError("Telegram Mini App user id is outside the valid range.")
    if value.get("is_bot") is True:
        raise TelegramInitDataError("Bot accounts cannot authorize the admin Mini App.")
    return TelegramMiniAppUser(
        id=user_id,
        first_name=_required_text(value.get("first_name"), "first_name", max_length=256),
        last_name=_optional_text(value.get("last_name"), "last_name", max_length=256),
        username=_optional_text(value.get("username"), "username", max_length=64),
        language_code=_optional_text(
            value.get("language_code"),
            "language_code",
            max_length=35,
        ),
        photo_url=_optional_text(value.get("photo_url"), "photo_url", max_length=2_048),
        is_premium=value.get("is_premium") is True,
    )


def validate_telegram_init_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS,
    future_skew_seconds: int = _DEFAULT_FUTURE_SKEW_SECONDS,
    now: float | None = None,
) -> ValidatedTelegramInitData:
    """Validate Telegram ``WebApp.initData`` using Telegram's bot-token HMAC.

    The URL-decoded fields are sorted into the documented data-check string.
    The supplied hash is excluded, and every other received field is covered.
    """

    if not isinstance(init_data, str) or not init_data:
        raise TelegramInitDataError("Telegram Mini App init data is required.")
    if len(init_data.encode("utf-8")) > _MAX_INIT_DATA_BYTES:
        raise TelegramInitDataError("Telegram Mini App init data is too large.")
    token = str(bot_token or "").strip()
    # Render and some dotenv loaders preserve literal surrounding quotes when
    # a secret is entered as `"123:ABC"`; Telegram tokens never contain quote
    # characters, so normalize that common deployment mistake safely.
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        token = token[1:-1].strip()
    if not token:
        raise TelegramInitDataError("Telegram bot token is not configured.")

    try:
        pairs = parse_qsl(
            init_data,
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=_MAX_FIELDS,
        )
    except (UnicodeError, ValueError) as exc:
        raise TelegramInitDataError("Telegram Mini App init data is malformed.") from exc

    fields: dict[str, str] = {}
    for key, value in pairs:
        if not key or key in fields:
            raise TelegramInitDataError(
                "Telegram Mini App init data contains a missing or duplicate field."
            )
        fields[key] = value

    received_hash = fields.get("hash", "")
    if not _HASH_RE.fullmatch(received_hash):
        raise TelegramInitDataError("Telegram Mini App hash is missing or invalid.")

    data_check_string = "\n".join(
        f"{key}={value}"
        for key, value in sorted(fields.items())
        if key != "hash"
    )
    secret_key = hmac.new(
        b"WebAppData",
        token.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    calculated_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(calculated_hash, received_hash.lower()):
        raise TelegramInitDataError("Telegram Mini App signature is invalid.")

    try:
        auth_date = int(fields.get("auth_date", ""))
    except (TypeError, ValueError) as exc:
        raise TelegramInitDataError("Telegram Mini App auth_date is invalid.") from exc
    current_time = int(time.time() if now is None else now)
    if auth_date > current_time + max(0, int(future_skew_seconds)):
        raise TelegramInitDataError("Telegram Mini App init data is from the future.")
    if max_age_seconds > 0 and current_time - auth_date > int(max_age_seconds):
        raise TelegramInitDataError("Telegram Mini App init data has expired.")

    user = _parse_user(fields.get("user", ""))
    query_id = str(fields.get("query_id") or "")
    if len(query_id) > 256:
        raise TelegramInitDataError("Telegram Mini App query id is too long.")
    return ValidatedTelegramInitData(
        user=user,
        auth_date=auth_date,
        query_id=query_id,
        fields=fields,
    )


def telegram_init_data_from_request(request: Request) -> tuple[str, bool]:
    """Return init data and whether a Telegram credential was explicitly sent."""

    header_value = str(request.headers.get("x-telegram-init-data") or "").strip()
    if header_value:
        return header_value, True

    authorization = str(request.headers.get("authorization") or "").strip()
    if not authorization.lower().startswith("bearer "):
        return "", False
    bearer = authorization.split(None, 1)[1].strip()
    # Preserve compatibility with the existing opaque admin bearer tokens.
    looks_like_init_data = (
        "auth_date=" in bearer
        and "hash=" in bearer
        and "user=" in bearer
        and "&" in bearer
    )
    return (bearer, True) if looks_like_init_data else ("", False)


class TelegramAdminAuthorizer:
    """Resolve authorized Telegram user IDs from a persistent Redis set."""

    def __init__(
        self,
        *,
        redis_prefix: str = "tgbot",
        cache_ttl_seconds: float = 5.0,
    ) -> None:
        prefix = str(redis_prefix or "tgbot").strip().strip(":")
        self.redis_key = f"{prefix or 'tgbot'}:security:admin_user_ids:v1"
        self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))
        self._redis: Any | None = None
        self._fallback_ids: frozenset[int] = frozenset()
        self._cached_ids: frozenset[int] = frozenset()
        self._cache_deadline = 0.0
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()

    def configure(
        self,
        *,
        redis_client: Any | None,
        fallback_admin_ids: Iterable[int] = (),
    ) -> TelegramAdminAuthorizer:
        clean_ids: set[int] = set()
        for value in fallback_admin_ids:
            try:
                user_id = int(value)
            except (TypeError, ValueError):
                continue
            if user_id > 0:
                clean_ids.add(user_id)
        with self._load_lock, self._lock:
            self._redis = redis_client
            self._fallback_ids = frozenset(clean_ids)
            self._cached_ids = frozenset()
            self._cache_deadline = 0.0
        return self

    @staticmethod
    def _decode_id(value: Any) -> int | None:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="strict")
        try:
            user_id = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return user_id if user_id > 0 else None

    def _load_ids_sync(self, *, force: bool = False) -> frozenset[int]:
        with self._load_lock:
            now = time.monotonic()
            with self._lock:
                if not force and now < self._cache_deadline:
                    return self._cached_ids
                client = self._redis
                fallback = self._fallback_ids

            if client is None:
                allowed = fallback
            else:
                try:
                    members = client.smembers(self.redis_key)
                    allowed = frozenset(
                        user_id
                        for user_id in (
                            self._decode_id(value)
                            for value in members or ()
                        )
                        if user_id is not None
                    )
                    if not allowed and fallback:
                        client.sadd(
                            self.redis_key,
                            *[str(value) for value in sorted(fallback)],
                        )
                        allowed = fallback
                        logger.warning(
                            "Migrated %s configured Telegram admin id(s) to Redis key=%s.",
                            len(fallback),
                            self.redis_key,
                        )
                except Exception as exc:
                    if fallback:
                        logger.warning(
                            "Redis admin allowlist unavailable; using configured fallback IDs: %s",
                            exc,
                        )
                        allowed = fallback
                    else:
                        raise TelegramAdminStoreError(
                            "The Telegram administrator allowlist is unavailable."
                        ) from exc

            with self._lock:
                self._cached_ids = allowed
                self._cache_deadline = (
                    time.monotonic() + self.cache_ttl_seconds
                )
            return allowed

    async def load_ids(self, *, force: bool = False) -> frozenset[int]:
        return await asyncio.to_thread(self._load_ids_sync, force=force)

    def is_admin_sync(self, user_id: int) -> bool:
        try:
            candidate = int(user_id)
        except (TypeError, ValueError):
            return False
        return candidate > 0 and candidate in self._load_ids_sync()

    async def authorize(
        self,
        init_data: str,
        bot_token: str,
        *,
        max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS,
    ) -> TelegramAdminSession:
        validated = validate_telegram_init_data(
            init_data,
            bot_token,
            max_age_seconds=max_age_seconds,
        )
        allowed_ids = await self.load_ids()
        if validated.user.id not in allowed_ids:
            raise PermissionError("This Telegram account is not an authorized administrator.")
        return TelegramAdminSession(
            user=validated.user,
            auth_date=validated.auth_date,
            query_id=validated.query_id,
        )


_TELEGRAM_ADMIN_AUTHORIZER = TelegramAdminAuthorizer()


def configure_telegram_admin_authorizer(
    *,
    redis_client: Any | None,
    fallback_admin_ids: Iterable[int] = (),
) -> TelegramAdminAuthorizer:
    return _TELEGRAM_ADMIN_AUTHORIZER.configure(
        redis_client=redis_client,
        fallback_admin_ids=fallback_admin_ids,
    )


def get_telegram_admin_authorizer() -> TelegramAdminAuthorizer:
    return _TELEGRAM_ADMIN_AUTHORIZER


__all__ = [
    "TelegramAdminAuthorizer",
    "TelegramAdminSession",
    "TelegramAdminStoreError",
    "TelegramInitDataError",
    "TelegramMiniAppUser",
    "ValidatedTelegramInitData",
    "configure_telegram_admin_authorizer",
    "get_telegram_admin_authorizer",
    "telegram_init_data_from_request",
    "validate_telegram_init_data",
]
