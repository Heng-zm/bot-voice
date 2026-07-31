"""Authenticated administration API for the dynamic CORS allowlist."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app._legacy_bridge import legacy_module
from app.api.dependencies import AdminPrincipal, require_admin, require_admin_write
from app.core.cors import (
    DynamicCorsError,
    DynamicCorsUnavailable,
    InvalidOriginError,
    get_dynamic_cors_store,
    normalize_origin,
)

router = APIRouter(prefix="/api/admin/cors", tags=["admin-cors"])
logger = logging.getLogger(__name__)


class CorsOriginPayload(BaseModel):
    origin: str = Field(min_length=1, max_length=2048)


def _audit(action: str, detail: str) -> None:
    try:
        legacy_module()._web_admin_audit(action, detail)
    except Exception as exc:  # noqa: BLE001 - legacy audit must stay best-effort
        # Audit capture must not turn a successfully persisted policy change
        # into a failed admin response.
        logger.debug("Could not record dynamic CORS audit event: %s", exc)


def _service_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


@router.get("")
async def get_cors_origins(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    try:
        snapshot = await get_dynamic_cors_store().load(force=True)
    except DynamicCorsUnavailable as exc:
        raise _service_unavailable(exc) from exc
    return {
        "ok": True,
        **snapshot.as_dict(),
        "credentials": True,
        "wildcards_allowed": False,
    }


@router.post("")
async def add_cors_origin(
    payload: CorsOriginPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    try:
        normalized = normalize_origin(payload.origin)
        snapshot, changed = await get_dynamic_cors_store().add(
            normalized,
            admin_id=principal.admin_id,
        )
    except InvalidOriginError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except DynamicCorsUnavailable as exc:
        raise _service_unavailable(exc) from exc
    except DynamicCorsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    _audit("cors_origin_add", normalized)
    return {
        "ok": True,
        "changed": changed,
        "origin": normalized,
        **snapshot.as_dict(),
    }


@router.delete("")
async def delete_cors_origin(
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
    payload: Annotated[CorsOriginPayload | None, Body()] = None,
    origin: Annotated[str | None, Query(min_length=1, max_length=2048)] = None,
) -> dict:
    requested_origin = payload.origin if payload is not None else origin
    if not requested_origin:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide the origin in the JSON body or origin query parameter.",
        )
    try:
        normalized = normalize_origin(requested_origin)
        snapshot, changed = await get_dynamic_cors_store().delete(
            normalized,
            admin_id=principal.admin_id,
        )
    except InvalidOriginError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except DynamicCorsUnavailable as exc:
        raise _service_unavailable(exc) from exc
    except DynamicCorsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    _audit("cors_origin_delete", normalized)
    return {
        "ok": True,
        "changed": changed,
        "origin": normalized,
        **snapshot.as_dict(),
    }


__all__ = ["CorsOriginPayload", "router"]
