"""Authenticated administrator allowlist management API."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app._legacy_bridge import legacy_module
from app.api.dependencies import (
    AdminPrincipal,
    get_redis,
    require_admin,
    require_admin_write,
)
from app.core.admin_management import (
    AdminConfirmationError,
    AdminManagementError,
    LastAdministratorError,
    RedisAdminManager,
)
from app.core.telegram_auth import get_telegram_admin_authorizer

router = APIRouter(prefix="/api/admin/administrators", tags=["admin-users"])


class AdminConfirmationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add", "remove"]
    user_id: int = Field(gt=0, lt=2**63)


class AdminMutationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(gt=0, lt=2**63)
    confirmation_token: str = Field(min_length=32, max_length=256)


def _manager() -> RedisAdminManager:
    redis_client = get_redis()
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is required for administrator management.",
        )
    return RedisAdminManager(redis_client)


def _translate_error(exc: AdminManagementError) -> HTTPException:
    if isinstance(exc, AdminConfirmationError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    if isinstance(exc, LastAdministratorError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


@router.get("")
async def list_administrators(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    try:
        admin_ids = await _manager().list_ids()
    except AdminManagementError as exc:
        raise _translate_error(exc) from exc
    return {"ok": True, "administrators": list(admin_ids), "count": len(admin_ids)}


@router.post("/confirmations")
async def create_administrator_confirmation(
    payload: AdminConfirmationPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    try:
        token, expires_in = await _manager().create_confirmation(
            action=payload.action,
            actor_id=principal.admin_id,
            target_id=payload.user_id,
        )
    except AdminManagementError as exc:
        raise _translate_error(exc) from exc
    return {
        "ok": True,
        "action": payload.action,
        "user_id": payload.user_id,
        "confirmation_token": token,
        "expires_in": expires_in,
    }


@router.post("")
async def add_administrator(
    payload: AdminMutationPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    try:
        result = await _manager().add(
            actor_id=principal.admin_id,
            target_id=payload.user_id,
            confirmation_token=payload.confirmation_token,
        )
    except AdminManagementError as exc:
        raise _translate_error(exc) from exc
    get_telegram_admin_authorizer().invalidate()
    if result.changed:
        legacy_module().ADMIN_IDS.add(result.target_id)
    return {
        "ok": True,
        "action": result.action,
        "user_id": result.target_id,
        "changed": result.changed,
    }


@router.delete("")
async def remove_administrator(
    payload: AdminMutationPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    try:
        result = await _manager().remove(
            actor_id=principal.admin_id,
            target_id=payload.user_id,
            confirmation_token=payload.confirmation_token,
        )
    except AdminManagementError as exc:
        raise _translate_error(exc) from exc
    get_telegram_admin_authorizer().invalidate()
    if result.changed:
        legacy_module().ADMIN_IDS.discard(result.target_id)
    return {
        "ok": True,
        "action": result.action,
        "user_id": result.target_id,
        "changed": result.changed,
    }


@router.get("/audit")
async def administrator_audit(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict:
    del principal
    try:
        entries = await _manager().audit(limit=limit)
    except AdminManagementError as exc:
        raise _translate_error(exc) from exc
    return {"ok": True, "entries": entries, "count": len(entries)}


__all__ = [
    "AdminConfirmationPayload",
    "AdminMutationPayload",
    "router",
]
