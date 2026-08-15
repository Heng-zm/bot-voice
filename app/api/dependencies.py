"""Shared HTTP dependency accessors.

These functions deliberately resolve clients at call time because the combined
runtime initializes external clients after importing the ASGI application.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from app._legacy_bridge import legacy_module
from app.core.telegram_auth import (
    TelegramAdminStoreError,
    TelegramInitDataError,
    TelegramMiniAppUser,
    get_telegram_admin_authorizer,
    telegram_init_data_from_request,
)


def get_settings() -> Any:
    return legacy_module().SETTINGS


def get_supabase() -> Any:
    return legacy_module().supabase


def get_redis() -> Any:
    return legacy_module().redis_client


async def authorize_ai_request() -> tuple[bool, Any | None]:
    return await legacy_module()._authorize_ai_api_request()


@dataclass(frozen=True)
class AdminPrincipal:
    admin_id: int
    auth_method: str
    telegram_user: TelegramMiniAppUser | None = None


async def require_admin(request: Request) -> AdminPrincipal:
    """Authenticate a native FastAPI admin route.

    This intentionally reuses the established signed-session and short-lived
    bearer-token implementation so native routes and compatibility routes have
    one authorization boundary.
    """

    legacy = legacy_module()
    telegram_init_data, telegram_credential_sent = telegram_init_data_from_request(request)
    if telegram_credential_sent:
        # A separate launcher bot may own the Mini App while the primary bot
        # handles normal messages. Keep the admin token optional for backward
        # compatibility; otherwise fall back to the primary bot token.
        bot_token = str(
            os.getenv("TELEGRAM_ADMIN_BOT_TOKEN", "")
            or getattr(legacy, "TELEGRAM_ADMIN_BOT_TOKEN", "")
            or getattr(legacy, "TELEGRAM_BOT_TOKEN", "")
            or getattr(legacy.SETTINGS, "TELEGRAM_BOT_TOKEN", "")
            or ""
        ).strip()
        if not bot_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Telegram administrator authorization is not configured.",
            )
        try:
            session = await get_telegram_admin_authorizer().authorize(
                telegram_init_data,
                bot_token,
            )
        except TelegramInitDataError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc
        except TelegramAdminStoreError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return AdminPrincipal(
            session.user.id,
            "telegram_init_data",
            session.user,
        )

    if not bool(legacy._web_admin_enabled()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin backend is disabled.",
        )

    authorization = str(request.headers.get("authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        token = authorization.split(None, 1)[1].strip()
        raw_admin_id = legacy._admin_verify_api_token(token)
        try:
            admin_id = int(raw_admin_id or 0)
        except (TypeError, ValueError, OverflowError):
            admin_id = 0
        if admin_id > 0 and legacy._web_valid_admin_id(admin_id):
            return AdminPrincipal(admin_id, "bearer")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired administrator bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    session = request.scope.get("session") or {}
    try:
        admin_id = int(session.get("web_admin_id") or 0)
    except (TypeError, ValueError):
        admin_id = 0
    if bool(session.get("web_admin_ok")) and legacy._web_valid_admin_id(admin_id):
        return AdminPrincipal(admin_id, "cookie")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Administrator authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_admin_write(
    request: Request,
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> AdminPrincipal:
    """Require CSRF for cookie-authenticated writes.

    Bearer tokens are not attached automatically by browsers, so they do not
    require the cookie CSRF token.
    """

    if principal.auth_method == "cookie":
        session = request.scope.get("session") or {}
        expected = str(session.get("web_csrf_token") or "")
        received = str(request.headers.get("x-csrf-token") or "")
        if not expected or not received or not hmac.compare_digest(expected, received):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing or invalid CSRF token.",
            )
    return principal


__all__ = [
    "AdminPrincipal",
    "authorize_ai_request",
    "get_redis",
    "get_settings",
    "get_supabase",
    "require_admin",
    "require_admin_write",
]
