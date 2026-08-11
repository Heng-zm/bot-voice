"""Administrative visibility and controls for jobs, workers, and providers."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import AdminPrincipal, require_admin, require_admin_write
from app.services.ai.providers import get_provider_manager
from app.services.jobs.queue import JobNotFound, JobQueueError
from app.services.jobs.runtime import (
    get_job_queue,
    job_worker_snapshot,
    set_job_workers_accepting,
)

router = APIRouter(prefix="/api/admin/runtime", tags=["admin-runtime"])


class ProviderResetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=64)


class JobRetrySelectedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_ids: list[str] = Field(min_length=1, max_length=100)


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
        "progress_percent": job.progress_percent,
        "progress_stage": job.progress_stage,
        "progress_detail": job.progress_detail,
        "updated_at": job.updated_at,
        "result": job.result,
    }


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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return {
        "ok": True,
        "provider": payload.provider.strip().lower(),
        **get_provider_manager().metadata(),
    }


@router.get("/workers")
async def worker_health(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
) -> dict:
    del principal
    return {"ok": True, **job_worker_snapshot()}


@router.post("/workers/drain")
async def drain_workers(
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    del principal
    set_job_workers_accepting(False)
    return {"ok": True, **job_worker_snapshot()}


@router.post("/workers/resume")
async def resume_workers(
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    del principal
    set_job_workers_accepting(True)
    return {"ok": True, **job_worker_snapshot()}


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
    return {"ok": True, "jobs": counts, "workers": job_worker_snapshot()}


@router.get("/jobs/list")
async def job_list(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    state: Annotated[
        Literal["queued", "running", "dead", "succeeded", "cancelled"],
        Query(),
    ] = "dead",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str, Query(max_length=32)] = "",
    job_type: Annotated[str, Query(max_length=64)] = "",
    query: Annotated[str, Query(max_length=128)] = "",
) -> dict:
    del principal
    try:
        jobs, next_cursor = await get_job_queue().list_jobs(
            state=state,
            limit=limit,
            cursor=cursor,
            job_type=job_type,
            query=query,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except JobQueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return {
        "ok": True,
        "state": state,
        "job_type": job_type.strip().lower(),
        "query": query.strip(),
        "jobs": [_safe_job(job) for job in jobs],
        "count": len(jobs),
        "next_cursor": next_cursor,
    }


@router.post("/jobs/retry-selected")
async def retry_selected_jobs(
    payload: JobRetrySelectedPayload,
    principal: Annotated[AdminPrincipal, Depends(require_admin_write)],
) -> dict:
    del principal
    unique_ids = list(dict.fromkeys(job_id.strip() for job_id in payload.job_ids))
    if any(not job_id or len(job_id) > 128 for job_id in unique_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Every job ID must contain 1-128 characters.",
        )

    async def retry_one(job_id: str) -> tuple[str, str]:
        try:
            changed = await get_job_queue().retry(job_id)
            return job_id, "queued" if changed else "unchanged"
        except JobNotFound:
            return job_id, "not_found"
        except JobQueueError:
            return job_id, "unavailable"

    results = await asyncio.gather(*(retry_one(job_id) for job_id in unique_ids))
    return {
        "ok": True,
        "results": [
            {"job_id": job_id, "state": result_state}
            for job_id, result_state in results
        ],
        "retried": sum(1 for _job_id, result_state in results if result_state == "queued"),
    }


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


__all__ = [
    "JobRetrySelectedPayload",
    "drain_workers",
    "resume_workers",
    "ProviderResetPayload",
    "router",
]
