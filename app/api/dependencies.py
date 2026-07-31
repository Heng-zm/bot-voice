"""Shared HTTP dependency accessors.

These functions deliberately resolve clients at call time because the combined
runtime initializes external clients after importing the ASGI application.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from app._legacy_bridge import legacy_module


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


async def require_admin(request: Request) -> AdminPrincipal:
    """Authenticate a native FastAPI admin route.

    This intentionally reuses the established signed-session and short-lived
    bearer-token implementation so native routes and compatibility routes have
    one authorization boundary.
    """

    legacy = legacy_module()
    if not bool(legacy._web_admin_enabled()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin backend is disabled.",
        )

    authorization = str(request.headers.get("authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        token = authorization.split(None, 1)[1].strip()
        admin_id = legacy._admin_verify_api_token(token)
        if admin_id and legacy._web_valid_admin_id(int(admin_id)):
            return AdminPrincipal(int(admin_id), "bearer")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired administrator bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    session = request.session
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
        expected = str(request.session.get("web_csrf_token") or "")
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
