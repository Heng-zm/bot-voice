"""Administrative visibility and controls for jobs and providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import AdminPrincipal, require_admin, require_admin_write
from app.services.ai.providers import get_provider_manager
from app.services.jobs.queue import JobNotFound, JobQueueError
from app.services.jobs.runtime import get_job_queue

router = APIRouter(prefix="/api/admin/runtime", tags=["admin-runtime"])


class ProviderResetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)


def _safe_job(job) -> dict:
    return {
        "id": job.id,
        "type": job.type,
        "state": job.state,
        "priority": job.priority,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "timeout_seconds": job.timeout_seconds,
        "created_at": job.created_at,
        "available_at": job.available_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "worker_id": job.worker_id,
        "last_error": job.last_error,
        "cancel_requested": job.cancel_requested,
        "result": job.result,
    }


@router.get("/providers")
async def provider_health(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    providers = get_provider_manager().snapshot()
    return {"ok": True, "providers": providers, "count": len(providers)}


@router.post("/providers/reset")
async def reset_provider(
    payload: ProviderResetPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    del principal
    try:
        get_provider_manager().reset(payload.provider)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return {"ok": True, "provider": payload.provider.strip().lower()}


@router.get("/jobs")
async def job_stats(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    try:
        counts = await get_job_queue().stats()
    except JobQueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return {"ok": True, "jobs": counts}


@router.get("/jobs/{job_id}")
async def job_detail(
    job_id: str,
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    try:
        job = await get_job_queue().get(job_id)
    except JobNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except JobQueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return {"ok": True, "job": _safe_job(job)}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    del principal
    try:
        state = await get_job_queue().cancel(job_id)
    except JobQueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if state == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} was not found.",
        )
    return {"ok": True, "job_id": job_id, "state": state}


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    del principal
    try:
        changed = await get_job_queue().retry(job_id)
    except JobNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except JobQueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if not changed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only dead or cancelled jobs can be retried.",
        )
    return {"ok": True, "job_id": job_id, "state": "queued"}


__all__ = ["ProviderResetPayload", "router"]
