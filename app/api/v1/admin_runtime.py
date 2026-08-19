"""Administrative visibility for the single-process runtime and AI providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import AdminPrincipal, require_admin, require_admin_write
from app.runtime import get_runtime_context
from app.services.ai.providers import get_provider_manager

router = APIRouter(prefix="/api/admin/runtime", tags=["admin-runtime"])


class ProviderResetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(min_length=1, max_length=64)


@router.get("/status")
async def runtime_status(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    snapshot = get_runtime_context().snapshot()
    return {"ok": True, **snapshot}


@router.get("/providers")
async def provider_health(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    manager = get_provider_manager()
    providers = manager.snapshot()
    return {
        "ok": True,
        "providers": providers,
        "count": len(providers),
        **manager.metadata(),
    }


@router.post("/providers/reset")
async def reset_provider(
    payload: ProviderResetPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    del principal
    try:
        get_provider_manager().reset(payload.provider)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {
        "ok": True,
        "provider": payload.provider.strip().lower(),
        **get_provider_manager().metadata(),
    }


__all__ = ["ProviderResetPayload", "router", "runtime_status"]
