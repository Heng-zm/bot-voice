"""Administrative visibility and controls for jobs, workers, and providers."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from app._legacy_bridge import legacy_module
from app.api.dependencies import AdminPrincipal, require_admin, require_admin_write
from app.services.ai.providers import get_provider_manager
from app.services.incidents import incident_snapshot
from app.services.jobs.queue import JobNotFound, JobQueueError
from app.services.jobs.runtime import (
    get_job_queue,
    job_worker_snapshot,
    set_job_workers_accepting,
)
from app.services.monitoring import (
    discover_public_url,
    process_snapshot,
    runtime_log_snapshot,
    sanitize_monitor_text,
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


def _safe_monitor_job(job) -> dict:
    """Expose only fields needed by the live monitor, never payloads/results."""

    return {
        "id": sanitize_monitor_text(job.id, limit=128),
        "type": sanitize_monitor_text(job.type, limit=64),
        "state": job.state,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "worker_id": sanitize_monitor_text(job.worker_id, limit=128),
        "progress_percent": job.progress_percent,
        "progress_stage": sanitize_monitor_text(job.progress_stage, limit=100),
        "progress_detail": sanitize_monitor_text(job.progress_detail, limit=300),
        "last_error": sanitize_monitor_text(job.last_error, limit=500),
        "updated_at": job.updated_at,
    }


def _safe_worker_snapshot(snapshot: dict) -> dict:
    """Redact worker failures and IDs before returning them to the browser."""

    safe = {
        key: snapshot.get(key)
        for key in (
            "configured",
            "accepting",
            "count",
            "alive",
            "healthy",
            "restart_total",
        )
    }
    safe["workers"] = [
        {
            **{
                key: worker.get(key)
                for key in (
                    "alive",
                    "started_at",
                    "last_heartbeat_at",
                    "restart_count",
                    "last_restart_at",
                )
            },
            "worker_id": sanitize_monitor_text(worker.get("worker_id"), limit=128),
            "last_error": sanitize_monitor_text(worker.get("last_error"), limit=500),
        }
        for worker in list(snapshot.get("workers") or [])[:50]
    ]
    safe["restart_history"] = [
        {
            "timestamp": item.get("timestamp"),
            "restart_count": item.get("restart_count"),
            "restart_streak": item.get("restart_streak"),
            "delay_seconds": item.get("delay_seconds"),
            "worker_id": sanitize_monitor_text(item.get("worker_id"), limit=128),
            "error": sanitize_monitor_text(item.get("error"), limit=500),
        }
        for item in list(snapshot.get("restart_history") or [])[:50]
    ]
    return safe


def _legacy_monitor_snapshot() -> dict:
    legacy = legacy_module()
    performance = legacy._runtime_performance_snapshot(light=True)
    with legacy._tts_request_reservations_guard:
        reserved_requests = sum(legacy._tts_request_reservations.values())
    return {
        "uptime": str(performance.get("uptime") or "starting"),
        "active_requests": int(performance.get("web", {}).get("active_requests") or 0),
        "db_queue_size": int(legacy._db_executor_queue_size()),
        "metrics": dict(performance.get("metrics") or {}),
        "tts_slots": dict(performance.get("semaphores", {}).get("tts") or {}),
        "reserved_requests": reserved_requests,
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
    queue = get_job_queue()
    try:
        counts = await queue.stats()
    except JobQueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return {
        "ok": True,
        "jobs": counts,
        "workers": job_worker_snapshot(),
        "queue": {
            "backend": getattr(queue, "backend", "unknown"),
            "durable": bool(getattr(queue, "durable", False)),
        },
    }


@router.get("/monitor")
async def bot_monitor(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    log_limit: Annotated[int, Query(ge=1, le=200)] = 100,
    log_level: Annotated[
        Literal["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Query(),
    ] = "",
    log_query: Annotated[str, Query(max_length=128)] = "",
) -> dict:
    """Return a sanitized process, TTS, worker, queue, and log snapshot."""

    del principal
    queue = get_job_queue()
    try:
        counts, running_page, queued_page = await asyncio.gather(
            queue.stats(),
            # Active jobs are bounded by the worker count, so one unfiltered
            # page is cheaper than repeatedly scanning for a type match.
            queue.list_jobs(state="running", limit=100),
            # Keep live polling to one bounded queued page. The full Jobs view
            # remains available when an older TTS job is outside this window.
            queue.list_jobs(state="queued", limit=200),
        )
    except (JobQueueError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    running_page_jobs, running_cursor = running_page
    queued_page_jobs, queued_cursor = queued_page
    workload_types = {"tts", "ocr", "transcription"}
    running_workloads = [
        job for job in running_page_jobs if job.type in workload_types
    ][:100]
    queued_workloads = [
        job for job in queued_page_jobs if job.type in workload_types
    ][:100]
    running_jobs = [job for job in running_workloads if job.type == "tts"][:50]
    queued_jobs = [job for job in queued_workloads if job.type == "tts"][:50]
    legacy = _legacy_monitor_snapshot()
    process = process_snapshot()
    workers = _safe_worker_snapshot(job_worker_snapshot())
    queue_limit = max(1, int(counts.get("queue_limit") or 1))
    queue_pressure = round(
        min(100.0, (int(counts.get("queued") or 0) / queue_limit) * 100.0),
        1,
    )
    failure_rate = float(counts.get("failure_rate_percent") or 0.0)
    incidents = incident_snapshot(limit=50)
    if incidents.get("open_circuits") or (
        workers.get("count") and not workers.get("healthy")
    ):
        health_state = "critical"
    elif queue_pressure >= 80.0 or failure_rate >= 20.0:
        health_state = "warning"
    else:
        health_state = "healthy"
    process.update(
        {
            "instance_id": get_provider_manager().metadata().get("instance_id", ""),
            "uptime": legacy["uptime"],
            "active_requests": legacy["active_requests"],
            "db_queue_size": legacy["db_queue_size"],
            "metrics": legacy["metrics"],
        }
    )
    tts_slots = legacy["tts_slots"]
    return {
        "ok": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "health": {
            "state": health_state,
            "queue_pressure_percent": queue_pressure,
            "failure_rate_percent": failure_rate,
        },
        "process": process,
        "workers": workers,
        "public_url": discover_public_url(),
        "incidents": incidents,
        "queue": counts,
        "queue_mode": {
            "backend": getattr(queue, "backend", "unknown"),
            "durable": bool(getattr(queue, "durable", False)),
        },
        "workloads": {
            "running": [_safe_monitor_job(job) for job in running_workloads],
            "queued": [_safe_monitor_job(job) for job in queued_workloads],
            "running_count": len(running_workloads),
            "queued_count": len(queued_workloads),
            "counts_by_type": {
                job_type: {
                    "running": sum(job.type == job_type for job in running_workloads),
                    "queued": sum(job.type == job_type for job in queued_workloads),
                }
                for job_type in sorted(workload_types)
            },
            "running_truncated": bool(running_cursor),
            "queued_truncated": bool(queued_cursor),
        },
        "tts": {
            "configured": int(tts_slots.get("configured") or 0),
            "available": tts_slots.get("available"),
            "in_use": tts_slots.get("in_use"),
            "reserved_requests": legacy["reserved_requests"],
            "running": [_safe_monitor_job(job) for job in running_jobs],
            "queued": [_safe_monitor_job(job) for job in queued_jobs],
            "running_count": len(running_jobs),
            "queued_count": len(queued_jobs),
            "running_truncated": bool(running_cursor) or sum(
                job.type == "tts" for job in running_page_jobs
            ) > len(running_jobs),
            "queued_truncated": bool(queued_cursor) or sum(
                job.type == "tts" for job in queued_page_jobs
            ) > len(queued_jobs),
        },
        "logs": runtime_log_snapshot(
            limit=log_limit,
            level=log_level,
            query=log_query,
        ),
    }


@router.get("/monitor/logs/download", response_class=PlainTextResponse)
async def download_monitor_logs(
    principal: Annotated[AdminPrincipal, Depends(require_admin)],
    log_limit: Annotated[int, Query(ge=1, le=400)] = 400,
    log_level: Annotated[
        Literal["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Query(),
    ] = "",
    log_query: Annotated[str, Query(max_length=128)] = "",
) -> PlainTextResponse:
    """Download the same bounded, redacted log view shown in the monitor."""

    del principal
    snapshot = runtime_log_snapshot(
        limit=log_limit,
        level=log_level,
        query=log_query,
    )
    lines = [
        f"[{entry.get('ts') or ''}] {entry.get('level') or 'INFO'} "
        f"{entry.get('source') or 'runtime'} — {entry.get('message') or ''}"
        for entry in snapshot["entries"]
    ]
    filename = f"bot-runtime-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.log"
    return PlainTextResponse(
        "\n".join(lines) + ("\n" if lines else ""),
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


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
