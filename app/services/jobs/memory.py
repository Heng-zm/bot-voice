"""Process-local job queue used when Redis is explicitly disabled."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import secrets
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from app.services.jobs.queue import Job, JobNotFound, QueueFull


class MemoryJobQueue:
    """Single-process queue with the same worker-facing API as RedisJobQueue.

    Jobs are lost on restart and cannot be shared between processes.  This is
    deliberately available only through the explicit Redis-disabled mode.
    """

    backend = "memory"
    durable = False

    def __init__(
        self,
        *,
        lease_seconds: float = 60.0,
        retention_seconds: int = 86_400,
        max_payload_bytes: int = 1_048_576,
        max_result_bytes: int = 262_144,
        max_queued_jobs: int = 1_000,
    ) -> None:
        self.lease_seconds = max(5.0, min(3_600.0, float(lease_seconds)))
        self.retention_seconds = max(300, min(2_592_000, int(retention_seconds)))
        self.max_payload_bytes = max(1_024, int(max_payload_bytes))
        self.max_result_bytes = max(1_024, int(max_result_bytes))
        self.max_queued_jobs = max(1, min(1_000_000, int(max_queued_jobs)))
        self._jobs: dict[str, Job] = {}
        self._idempotency: dict[str, str] = {}
        self._idempotency_deadlines: dict[str, float] = {}
        self._lease_deadlines: dict[str, float] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _json_copy(value: Any, *, max_bytes: int, label: str) -> Any:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be JSON serializable.") from exc
        if len(encoded.encode("utf-8")) > max_bytes:
            raise ValueError(f"{label} exceeds the {max_bytes}-byte limit.")
        return json.loads(encoded)

    def _sweep(self, now: float) -> None:
        for job_id, deadline in list(self._lease_deadlines.items()):
            if deadline > now:
                continue
            job = self._jobs.get(job_id)
            self._lease_deadlines.pop(job_id, None)
            if job is None or job.state != "running":
                continue
            if job.cancel_requested:
                state = "cancelled"
                completed_at = now
                last_error = "cancelled while worker lease was unavailable"
                progress_stage = "cancelled"
            elif job.attempts >= job.max_attempts:
                state = "dead"
                completed_at = now
                last_error = "worker lease expired"
                progress_stage = "dead"
            else:
                state = "queued"
                completed_at = None
                last_error = "worker lease expired"
                progress_stage = "retrying"
            self._jobs[job_id] = replace(
                job,
                state=state,
                available_at=now,
                completed_at=completed_at,
                worker_id="",
                lease_token="",
                last_error=last_error,
                progress_stage=progress_stage,
                updated_at=now,
            )

        cutoff = now - self.retention_seconds
        for job_id, job in list(self._jobs.items()):
            if (
                job.state in {"dead", "succeeded", "cancelled"}
                and (job.completed_at or job.updated_at or now) < cutoff
            ):
                self._jobs.pop(job_id, None)
                self._lease_deadlines.pop(job_id, None)
        for key, deadline in list(self._idempotency_deadlines.items()):
            if deadline <= now:
                self._idempotency_deadlines.pop(key, None)
                self._idempotency.pop(key, None)
        live_ids = set(self._jobs)
        for key, job_id in list(self._idempotency.items()):
            if job_id not in live_ids:
                self._idempotency.pop(key, None)
                self._idempotency_deadlines.pop(key, None)

    @staticmethod
    def _clone_job(job: Job) -> Job:
        return replace(
            job,
            payload=copy.deepcopy(job.payload),
            result=copy.deepcopy(job.result),
        )

    async def enqueue(
        self,
        job_type: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str = "",
        priority: int = 0,
        delay_seconds: float = 0.0,
        max_attempts: int = 3,
        timeout_seconds: float = 300.0,
        idempotency_ttl_seconds: int = 86_400,
    ) -> tuple[Job, bool]:
        clean_type = str(job_type or "").strip().lower()
        if not clean_type or len(clean_type) > 64:
            raise ValueError("Job type is missing or too long.")
        clean_payload = self._json_copy(
            dict(payload),
            max_bytes=self.max_payload_bytes,
            label="Job payload",
        )
        clean_idempotency = str(idempotency_key or "").strip()
        idempotency_digest = (
            hashlib.sha256(clean_idempotency.encode("utf-8")).hexdigest()
            if clean_idempotency
            else ""
        )
        now = time.time()
        async with self._lock:
            self._sweep(now)
            existing_id = self._idempotency.get(idempotency_digest)
            if idempotency_digest and existing_id in self._jobs:
                return self._clone_job(self._jobs[existing_id]), False
            queued = sum(job.state == "queued" for job in self._jobs.values())
            if queued >= self.max_queued_jobs:
                raise QueueFull(
                    f"The in-memory job queue reached its {self.max_queued_jobs}-job limit."
                )
            job_id = uuid.uuid4().hex
            job = Job(
                id=job_id,
                type=clean_type,
                payload=clean_payload,
                state="queued",
                priority=max(-100, min(100, int(priority))),
                attempts=0,
                max_attempts=max(1, min(100, int(max_attempts))),
                timeout_seconds=max(0.1, min(86_400.0, float(timeout_seconds))),
                created_at=now,
                available_at=now + max(0.0, float(delay_seconds)),
                progress_stage="queued",
                updated_at=now,
            )
            self._jobs[job_id] = job
            if idempotency_digest:
                self._idempotency[idempotency_digest] = job_id
                self._idempotency_deadlines[idempotency_digest] = now + max(
                    60,
                    min(2_592_000, int(idempotency_ttl_seconds)),
                )
            return self._clone_job(job), True

    async def claim(self, worker_id: str) -> Job | None:
        worker = str(worker_id or "").strip()
        if not worker or len(worker) > 128:
            raise ValueError("Worker ID is missing or too long.")
        now = time.time()
        async with self._lock:
            self._sweep(now)
            ready = [
                job
                for job in self._jobs.values()
                if job.state == "queued" and job.available_at <= now
            ]
            if not ready:
                return None
            current = min(
                ready,
                # Match the Redis queue: priority wins once a job is due, then
                # original creation order provides FIFO within that priority.
                key=lambda job: (-job.priority, job.created_at),
            )
            token = secrets.token_urlsafe(24)
            claimed = replace(
                current,
                state="running",
                attempts=current.attempts + 1,
                started_at=now,
                worker_id=worker,
                lease_token=token,
                progress_stage="running",
                updated_at=now,
            )
            self._jobs[current.id] = claimed
            self._lease_deadlines[current.id] = now + self.lease_seconds
            return self._clone_job(claimed)

    async def renew(self, job_id: str, lease_token: str) -> int:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != "running" or job.lease_token != lease_token:
                return 0
            if job.cancel_requested:
                return -1
            self._lease_deadlines[job_id] = time.time() + self.lease_seconds
            return 1

    async def complete(self, job_id: str, lease_token: str, result: Any) -> bool:
        clean_result = self._json_copy(
            result,
            max_bytes=self.max_result_bytes,
            label="Job result",
        )
        now = time.time()
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != "running" or job.lease_token != lease_token:
                return False
            state = "cancelled" if job.cancel_requested else "succeeded"
            self._jobs[job_id] = replace(
                job,
                state=state,
                result=clean_result if state == "succeeded" else None,
                completed_at=now,
                worker_id="",
                lease_token="",
                last_error="cancelled" if state == "cancelled" else "",
                progress_percent=100 if state == "succeeded" else job.progress_percent,
                progress_stage=state,
                updated_at=now,
            )
            self._lease_deadlines.pop(job_id, None)
            return True

    async def fail(
        self,
        job_id: str,
        lease_token: str,
        error: BaseException | str,
        *,
        retryable: bool = True,
        retry_delay_seconds: float = 0.0,
    ) -> str:
        now = time.time()
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != "running" or job.lease_token != lease_token:
                return "lease_lost"
            if job.cancel_requested:
                state = "cancelled"
            elif retryable and job.attempts < job.max_attempts:
                state = "queued"
            else:
                state = "dead"
            self._jobs[job_id] = replace(
                job,
                state=state,
                available_at=now + max(0.0, float(retry_delay_seconds)),
                completed_at=now if state in {"dead", "cancelled"} else None,
                worker_id="",
                lease_token="",
                last_error="cancelled" if state == "cancelled" else str(error)[:1_000],
                progress_stage="retrying" if state == "queued" else state,
                updated_at=now,
            )
            self._lease_deadlines.pop(job_id, None)
            return state

    async def cancel(self, job_id: str) -> str:
        now = time.time()
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "not_found"
            if job.state == "queued":
                self._jobs[job_id] = replace(
                    job,
                    state="cancelled",
                    cancel_requested=True,
                    completed_at=now,
                    progress_stage="cancelled",
                    updated_at=now,
                )
                return "cancelled"
            if job.state == "running":
                self._jobs[job_id] = replace(
                    job,
                    cancel_requested=True,
                    progress_stage="cancelling",
                    updated_at=now,
                )
                return "requested"
            return "unchanged"

    async def retry(self, job_id: str) -> bool:
        now = time.time()
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(f"Job {job_id!r} was not found.")
            if job.state not in {"dead", "cancelled"}:
                return False
            self._jobs[job_id] = replace(
                job,
                state="queued",
                attempts=0,
                available_at=now,
                started_at=None,
                completed_at=None,
                worker_id="",
                lease_token="",
                last_error="",
                result=None,
                cancel_requested=False,
                progress_percent=0,
                progress_stage="queued",
                progress_detail="",
                updated_at=now,
            )
            return True

    async def get(self, job_id: str) -> Job:
        async with self._lock:
            self._sweep(time.time())
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(f"Job {job_id!r} was not found.")
            return self._clone_job(job)

    async def update_progress(
        self,
        job_id: str,
        lease_token: str,
        *,
        percent: int,
        stage: str,
        detail: str = "",
    ) -> bool:
        clean_stage = str(stage or "").strip()[:120]
        if not clean_stage:
            raise ValueError("Progress stage is required.")
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != "running" or job.lease_token != lease_token:
                return False
            self._jobs[job_id] = replace(
                job,
                progress_percent=max(0, min(100, int(percent))),
                progress_stage=clean_stage,
                progress_detail=str(detail or "").strip()[:500],
                updated_at=time.time(),
            )
            return True

    async def list_jobs(
        self,
        *,
        state: str = "dead",
        limit: int = 50,
        cursor: str = "",
        job_type: str = "",
        query: str = "",
    ) -> tuple[list[Job], str | None]:
        clean_state = str(state or "").strip().lower()
        if clean_state not in {"queued", "running", "dead", "succeeded", "cancelled"}:
            raise ValueError("state must be queued, running, dead, succeeded, or cancelled.")
        try:
            offset = max(0, int(str(cursor or "0")))
        except ValueError as exc:
            raise ValueError("Invalid job-list cursor.") from exc
        page_size = max(1, min(200, int(limit)))
        clean_type = str(job_type or "").strip().lower()
        clean_query = str(query or "").strip().lower()
        if len(clean_type) > 64 or len(clean_query) > 128:
            raise ValueError("Job filter is too long.")
        async with self._lock:
            self._sweep(time.time())
            jobs = sorted(
                (job for job in self._jobs.values() if job.state == clean_state),
                key=lambda job: job.updated_at or job.created_at,
                reverse=True,
            )
            jobs = [
                job
                for job in jobs
                if (not clean_type or job.type == clean_type)
                and (
                    not clean_query
                    or clean_query
                    in "\n".join(
                        (
                            job.id,
                            job.type,
                            job.state,
                            job.worker_id,
                            job.last_error,
                            job.progress_stage,
                            job.progress_detail,
                        )
                    ).lower()
                )
            ]
            page = [
                self._clone_job(job)
                for job in jobs[offset : offset + page_size]
            ]
            next_offset = offset + len(page)
            return page, str(next_offset) if next_offset < len(jobs) else None

    async def stats(self) -> dict[str, int | float]:
        now = time.time()
        async with self._lock:
            self._sweep(now)
            counts = {
                state: sum(job.state == state for job in self._jobs.values())
                for state in {"queued", "running", "dead", "succeeded", "cancelled"}
            }
            queued = [job for job in self._jobs.values() if job.state == "queued"]
            succeeded_hour = sum(
                job.state == "succeeded" and (job.completed_at or 0) >= now - 3_600
                for job in self._jobs.values()
            )
            failed_hour = sum(
                job.state == "dead" and (job.completed_at or 0) >= now - 3_600
                for job in self._jobs.values()
            )
            cancelled_hour = sum(
                job.state == "cancelled" and (job.completed_at or 0) >= now - 3_600
                for job in self._jobs.values()
            )
            terminal = succeeded_hour + failed_hour
            oldest = min((job.available_at for job in queued), default=now)
            return {
                **counts,
                "queue_limit": self.max_queued_jobs,
                "queue_available": max(0, self.max_queued_jobs - counts["queued"]),
                "oldest_queued_age_seconds": round(max(0.0, now - oldest), 1),
                "succeeded_last_hour": succeeded_hour,
                "failed_last_hour": failed_hour,
                "cancelled_last_hour": cancelled_hour,
                "throughput_per_minute": round(succeeded_hour / 60.0, 2),
                "failure_rate_percent": round((failed_hour / terminal) * 100.0, 2)
                if terminal
                else 0.0,
            }


__all__ = ["MemoryJobQueue"]
