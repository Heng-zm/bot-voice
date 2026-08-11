"""Durable Redis job queue with leases, retries, cancellation, and a DLQ."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import random
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Consecutive failed lease renewals tolerated before the running job is
# cancelled. Renewal runs at lease_seconds/3, so two failures still leave the
# lease valid while a third means it is about to expire anyway.
_MAX_HEARTBEAT_ERRORS = 3

_ENQUEUE_SCRIPT = """
-- bot_voice:enqueue_v1
if ARGV[9] == '1' then
  local existing = redis.call('GET', KEYS[3])
  if existing then
    return {existing, 0}
  end
end
if redis.call('EXISTS', KEYS[2]) == 1 then
  return {ARGV[1], 0}
end
if tonumber(ARGV[11]) > 0 and redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[11]) then
  return {'', -1}
end
redis.call(
  'HSET', KEYS[2],
  'id', ARGV[1],
  'type', ARGV[2],
  'payload', ARGV[3],
  'state', 'queued',
  'priority', ARGV[4],
  'created_at', ARGV[5],
  'available_at', ARGV[6],
  'attempts', '0',
  'max_attempts', ARGV[7],
  'timeout_seconds', ARGV[8],
  'cancel_requested', '0',
  'progress_percent', '0',
  'progress_stage', 'queued',
  'progress_detail', '',
  'updated_at', ARGV[5]
)
redis.call('ZADD', KEYS[1], ARGV[6], ARGV[1])
if ARGV[9] == '1' then
  redis.call('SET', KEYS[3], ARGV[1], 'EX', ARGV[10])
end
return {ARGV[1], 1}
""".strip()

_CLAIM_SCRIPT = """
-- bot_voice:claim_v1
local expired = redis.call(
  'ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[1], 'LIMIT', 0, 100
)
for _, job_id in ipairs(expired) do
  local job_key = ARGV[6] .. job_id
  if redis.call('HGET', job_key, 'state') == 'running' then
    local cancelled = redis.call('HGET', job_key, 'cancel_requested')
    local attempts = tonumber(redis.call('HGET', job_key, 'attempts') or '0')
    local max_attempts = tonumber(
      redis.call('HGET', job_key, 'max_attempts') or '1'
    )
    if cancelled == '1' then
      redis.call(
        'HSET', job_key,
        'state', 'cancelled',
        'completed_at', ARGV[1],
        'last_error', 'cancelled while worker lease was unavailable',
        'progress_stage', 'cancelled',
        'updated_at', ARGV[1]
      )
      redis.call('ZADD', KEYS[4], ARGV[1], job_id)
      redis.call(
        'HDEL', job_key, 'lease_token', 'lease_deadline', 'worker_id'
      )
      redis.call('EXPIRE', job_key, ARGV[5])
    elseif attempts >= max_attempts then
      redis.call(
        'HSET', job_key,
        'state', 'dead',
        'completed_at', ARGV[1],
        'last_error', 'worker lease expired',
        'progress_stage', 'dead',
        'updated_at', ARGV[1]
      )
      redis.call('ZADD', KEYS[3], ARGV[1], job_id)
      redis.call(
        'HDEL', job_key, 'lease_token', 'lease_deadline', 'worker_id'
      )
      redis.call('EXPIRE', job_key, ARGV[5])
    else
      redis.call(
        'HSET', job_key,
        'state', 'queued',
        'available_at', ARGV[1],
        'last_error', 'worker lease expired',
        'progress_stage', 'retrying',
        'updated_at', ARGV[1]
      )
      redis.call('HDEL', job_key, 'lease_token', 'lease_deadline', 'worker_id')
      redis.call('ZADD', KEYS[1], ARGV[1], job_id)
    end
  end
  redis.call('ZREM', KEYS[2], job_id)
end

local candidates = redis.call(
  'ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 100
)
local selected = nil
local selected_priority = -1
local selected_created = nil
for _, job_id in ipairs(candidates) do
  local job_key = ARGV[6] .. job_id
  if redis.call('HGET', job_key, 'state') == 'queued' then
    local priority = tonumber(redis.call('HGET', job_key, 'priority') or '0')
    local created = tonumber(redis.call('HGET', job_key, 'created_at') or '0')
    if (
      not selected
      or priority > selected_priority
      or (priority == selected_priority and created < selected_created)
    ) then
      selected = job_id
      selected_priority = priority
      selected_created = created
    end
  else
    redis.call('ZREM', KEYS[1], job_id)
  end
end
if not selected then
  return nil
end

local selected_key = ARGV[6] .. selected
redis.call('ZREM', KEYS[1], selected)
redis.call('HINCRBY', selected_key, 'attempts', 1)
redis.call(
  'HSET', selected_key,
  'state', 'running',
  'worker_id', ARGV[2],
  'lease_token', ARGV[3],
  'lease_deadline', ARGV[4],
  'started_at', ARGV[1],
  'cancel_requested', '0',
  'progress_stage', 'running',
  'updated_at', ARGV[1]
)
redis.call('ZADD', KEYS[2], ARGV[4], selected)
return selected
""".strip()

_RENEW_SCRIPT = """
-- bot_voice:renew_v1
if redis.call('HGET', KEYS[2], 'state') ~= 'running' then
  return 0
end
if redis.call('HGET', KEYS[2], 'lease_token') ~= ARGV[1] then
  return 0
end
if redis.call('HGET', KEYS[2], 'cancel_requested') == '1' then
  return -1
end
redis.call('HSET', KEYS[2], 'lease_deadline', ARGV[2], 'updated_at', ARGV[4])
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3])
return 1
""".strip()

_UPDATE_PROGRESS_SCRIPT = """
-- bot_voice:update_progress_v1
if redis.call('HGET', KEYS[1], 'state') ~= 'running' then
  return 0
end
if redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call(
  'HSET', KEYS[1],
  'progress_percent', ARGV[2],
  'progress_stage', ARGV[3],
  'progress_detail', ARGV[4],
  'updated_at', ARGV[5]
)
return 1
""".strip()

_COMPLETE_SCRIPT = """
-- bot_voice:complete_v1
if redis.call('HGET', KEYS[2], 'state') ~= 'running' then
  return 0
end
if redis.call('HGET', KEYS[2], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call('ZREM', KEYS[1], ARGV[2])
if redis.call('HGET', KEYS[2], 'cancel_requested') == '1' then
  redis.call(
    'HSET', KEYS[2],
    'state', 'cancelled',
    'completed_at', ARGV[3],
    'last_error', 'cancelled',
    'progress_stage', 'cancelled',
    'updated_at', ARGV[3]
  )
  redis.call('ZADD', KEYS[4], ARGV[3], ARGV[2])
else
  redis.call(
    'HSET', KEYS[2],
    'state', 'succeeded',
    'completed_at', ARGV[3],
    'result', ARGV[4],
    'last_error', '',
    'progress_percent', '100',
    'progress_stage', 'succeeded',
    'updated_at', ARGV[3]
  )
  redis.call('ZADD', KEYS[3], ARGV[3], ARGV[2])
end
redis.call(
  'HDEL', KEYS[2], 'lease_token', 'lease_deadline', 'worker_id'
)
redis.call('EXPIRE', KEYS[2], ARGV[5])
return 1
""".strip()

_FAIL_SCRIPT = """
-- bot_voice:fail_v1
if redis.call('HGET', KEYS[3], 'state') ~= 'running' then
  return 0
end
if redis.call('HGET', KEYS[3], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call('ZREM', KEYS[2], ARGV[2])
local cancelled = redis.call('HGET', KEYS[3], 'cancel_requested')
local attempts = tonumber(redis.call('HGET', KEYS[3], 'attempts') or '0')
local max_attempts = tonumber(
  redis.call('HGET', KEYS[3], 'max_attempts') or '1'
)
if cancelled == '1' then
  redis.call(
    'HSET', KEYS[3],
    'state', 'cancelled',
    'completed_at', ARGV[3],
    'last_error', 'cancelled',
    'progress_stage', 'cancelled',
    'updated_at', ARGV[3]
  )
  redis.call('ZADD', KEYS[5], ARGV[3], ARGV[2])
  redis.call(
    'HDEL', KEYS[3], 'lease_token', 'lease_deadline', 'worker_id'
  )
  redis.call('EXPIRE', KEYS[3], ARGV[7])
  return 2
end
if ARGV[6] == '1' and attempts < max_attempts then
  redis.call(
    'HSET', KEYS[3],
    'state', 'queued',
    'available_at', ARGV[5],
    'last_error', ARGV[4],
    'progress_stage', 'retrying',
    'updated_at', ARGV[3]
  )
  redis.call(
    'HDEL', KEYS[3], 'lease_token', 'lease_deadline', 'worker_id'
  )
  redis.call('ZADD', KEYS[1], ARGV[5], ARGV[2])
  return 1
end
redis.call(
  'HSET', KEYS[3],
  'state', 'dead',
  'completed_at', ARGV[3],
  'last_error', ARGV[4],
  'progress_stage', 'dead',
  'updated_at', ARGV[3]
)
redis.call(
  'HDEL', KEYS[3], 'lease_token', 'lease_deadline', 'worker_id'
)
redis.call('ZADD', KEYS[4], ARGV[3], ARGV[2])
redis.call('EXPIRE', KEYS[3], ARGV[7])
return 3
""".strip()

_CANCEL_SCRIPT = """
-- bot_voice:cancel_v1
local state = redis.call('HGET', KEYS[3], 'state')
if not state then
  return -1
end
if state == 'queued' then
  redis.call('ZREM', KEYS[1], ARGV[1])
  redis.call(
    'HSET', KEYS[3],
    'state', 'cancelled',
    'cancel_requested', '1',
    'completed_at', ARGV[2],
    'last_error', 'cancelled',
    'progress_stage', 'cancelled',
    'updated_at', ARGV[2]
  )
  redis.call('ZADD', KEYS[4], ARGV[2], ARGV[1])
  redis.call('EXPIRE', KEYS[3], ARGV[3])
  return 1
end
if state == 'running' then
  redis.call('HSET', KEYS[3], 'cancel_requested', '1', 'progress_stage', 'cancelling', 'updated_at', ARGV[2])
  return 2
end
return 0
""".strip()

_RETRY_SCRIPT = """
-- bot_voice:retry_v1
local state = redis.call('HGET', KEYS[4], 'state')
if not state then
  return -1
end
if state ~= 'dead' and state ~= 'cancelled' then
  return 0
end
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
redis.call(
  'HSET', KEYS[4],
  'state', 'queued',
  'available_at', ARGV[2],
  'attempts', '0',
  'cancel_requested', '0',
  'last_error', '',
  'progress_percent', '0',
  'progress_stage', 'queued',
  'progress_detail', '',
  'updated_at', ARGV[2]
)
redis.call(
  'HDEL', KEYS[4],
  'completed_at', 'result', 'started_at',
  'lease_token', 'lease_deadline', 'worker_id'
)
redis.call('PERSIST', KEYS[4])
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
return 1
""".strip()


class JobQueueError(RuntimeError):
    """Base durable queue error."""


class JobNotFound(JobQueueError):
    """Raised when a requested job does not exist."""


class QueueFull(JobQueueError):
    """Raised when queue backpressure rejects a new job."""


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    type: str
    payload: Mapping[str, Any]
    state: str
    priority: int
    attempts: int
    max_attempts: int
    timeout_seconds: float
    created_at: float
    available_at: float
    started_at: float | None = None
    completed_at: float | None = None
    worker_id: str = ""
    lease_token: str = ""
    last_error: str = ""
    result: Any = None
    cancel_requested: bool = False
    progress_percent: int = 0
    progress_stage: str = ""
    progress_detail: str = ""
    updated_at: float | None = None


@dataclass(frozen=True, slots=True)
class JobContext:
    queue: RedisJobQueue
    job: Job
    worker_id: str
    lease_token: str

    async def cancelled(self) -> bool:
        current = await self.queue.get(self.job.id)
        return current.cancel_requested or current.state == "cancelled"

    async def progress(
        self,
        percent: int,
        stage: str,
        detail: str = "",
    ) -> bool:
        return await self.queue.update_progress(
            self.job.id,
            self.lease_token,
            percent=percent,
            stage=stage,
            detail=detail,
        )


JobHandler = Callable[[Mapping[str, Any], JobContext], Any | Awaitable[Any]]


class RedisJobQueue:
    """Redis-backed queue safe for multiple workers and process restarts."""

    def __init__(
        self,
        redis_client: Any,
        *,
        redis_prefix: str = "tgbot",
        lease_seconds: float = 60.0,
        retention_seconds: int = 86_400,
        max_payload_bytes: int = 1_048_576,
        max_result_bytes: int = 262_144,
        max_queued_jobs: int = 1_000,
    ) -> None:
        if redis_client is None:
            raise JobQueueError("Redis is required for the durable job queue.")
        prefix = str(redis_prefix or "tgbot").strip().strip(":") or "tgbot"
        self.redis = redis_client
        self.key_prefix = f"{prefix}:jobs:v1"
        self.ready_key = f"{self.key_prefix}:ready"
        self.leased_key = f"{self.key_prefix}:leased"
        self.dead_key = f"{self.key_prefix}:dead"
        self.succeeded_key = f"{self.key_prefix}:succeeded"
        self.cancelled_key = f"{self.key_prefix}:cancelled"
        self.job_prefix = f"{self.key_prefix}:data:"
        self.idempotency_prefix = f"{self.key_prefix}:idempotency:"
        self.lease_seconds = max(5.0, min(3_600.0, float(lease_seconds)))
        self.retention_seconds = max(300, min(2_592_000, int(retention_seconds)))
        self.max_payload_bytes = max(1_024, int(max_payload_bytes))
        self.max_result_bytes = max(1_024, int(max_result_bytes))
        self.max_queued_jobs = max(1, min(1_000_000, int(max_queued_jobs)))

    def _job_key(self, job_id: str) -> str:
        return f"{self.job_prefix}{job_id}"

    def _idempotency_key(self, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"{self.idempotency_prefix}{digest}"

    @staticmethod
    def _dumps(value: Any, *, max_bytes: int, label: str) -> str:
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
        return encoded

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        return str(value or "")

    async def _redis_call(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            operation = getattr(self.redis, method)
            return await asyncio.to_thread(operation, *args, **kwargs)
        except Exception as exc:
            raise JobQueueError(
                f"Redis job queue operation {method} failed."
            ) from exc

    async def enqueue(
        self,
        job_type: str,
        payload: Mapping[str, Any],
        *,
        priority: int = 0,
        max_attempts: int = 3,
        timeout_seconds: float = 300.0,
        delay_seconds: float = 0.0,
        idempotency_key: str = "",
        idempotency_ttl_seconds: int = 86_400,
    ) -> tuple[Job, bool]:
        clean_type = str(job_type or "").strip().lower()
        if not clean_type or len(clean_type) > 64:
            raise ValueError("Job type is missing or too long.")
        payload_json = self._dumps(
            dict(payload),
            max_bytes=self.max_payload_bytes,
            label="Job payload",
        )
        job_id = uuid.uuid4().hex
        now = time.time()
        available_at = now + max(0.0, float(delay_seconds))
        clean_idempotency = str(idempotency_key or "").strip()
        use_idempotency = bool(clean_idempotency)
        idempotency_redis_key = (
            self._idempotency_key(clean_idempotency)
            if use_idempotency
            else f"{self.idempotency_prefix}unused:{job_id}"
        )
        raw = await self._redis_call(
            "eval",
            _ENQUEUE_SCRIPT,
            3,
            self.ready_key,
            self._job_key(job_id),
            idempotency_redis_key,
            job_id,
            clean_type,
            payload_json,
            str(max(-100, min(100, int(priority)))),
            str(now),
            str(available_at),
            str(max(1, min(100, int(max_attempts)))),
            str(max(0.1, min(86_400.0, float(timeout_seconds)))),
            "1" if use_idempotency else "0",
            str(max(60, min(2_592_000, int(idempotency_ttl_seconds)))),
            str(self.max_queued_jobs),
        )
        values = list(raw or ())
        if len(values) != 2:
            raise JobQueueError("Redis returned an invalid enqueue result.")
        resolved_id = self._decode(values[0])
        status_code = int(values[1])
        if status_code == -1:
            raise QueueFull(
                f"The durable job queue reached its {self.max_queued_jobs}-job limit."
            )
        created = bool(status_code)
        return await self.get(resolved_id), created

    async def claim(self, worker_id: str) -> Job | None:
        worker = str(worker_id or "").strip()
        if not worker or len(worker) > 128:
            raise ValueError("Worker ID is missing or too long.")
        now = time.time()
        token = secrets.token_urlsafe(24)
        deadline = now + self.lease_seconds
        raw = await self._redis_call(
            "eval",
            _CLAIM_SCRIPT,
            4,
            self.ready_key,
            self.leased_key,
            self.dead_key,
            self.cancelled_key,
            str(now),
            worker,
            token,
            str(deadline),
            str(self.retention_seconds),
            self.job_prefix,
        )
        job_id = self._decode(raw)
        if not job_id:
            return None
        return await self.get(job_id)

    async def renew(self, job_id: str, lease_token: str) -> int:
        deadline = time.time() + self.lease_seconds
        return int(
            await self._redis_call(
                "eval",
                _RENEW_SCRIPT,
                2,
                self.leased_key,
                self._job_key(job_id),
                lease_token,
                str(deadline),
                job_id,
                str(time.time()),
            )
        )

    async def complete(
        self,
        job_id: str,
        lease_token: str,
        result: Any,
    ) -> bool:
        result_json = self._dumps(
            result,
            max_bytes=self.max_result_bytes,
            label="Job result",
        )
        changed = await self._redis_call(
            "eval",
            _COMPLETE_SCRIPT,
            4,
            self.leased_key,
            self._job_key(job_id),
            self.succeeded_key,
            self.cancelled_key,
            lease_token,
            job_id,
            str(time.time()),
            result_json,
            str(self.retention_seconds),
        )
        return bool(changed)

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
        available_at = now + max(0.0, float(retry_delay_seconds))
        result = int(
            await self._redis_call(
                "eval",
                _FAIL_SCRIPT,
                5,
                self.ready_key,
                self.leased_key,
                self._job_key(job_id),
                self.dead_key,
                self.cancelled_key,
                lease_token,
                job_id,
                str(now),
                str(error)[:1_000],
                str(available_at),
                "1" if retryable else "0",
                str(self.retention_seconds),
            )
        )
        return {
            0: "lease_lost",
            1: "queued",
            2: "cancelled",
            3: "dead",
        }.get(result, "unknown")

    async def cancel(self, job_id: str) -> str:
        result = int(
            await self._redis_call(
                "eval",
                _CANCEL_SCRIPT,
                4,
                self.ready_key,
                self.leased_key,
                self._job_key(job_id),
                self.cancelled_key,
                job_id,
                str(time.time()),
                str(self.retention_seconds),
            )
        )
        states = {-1: "not_found", 0: "unchanged", 1: "cancelled", 2: "requested"}
        return states.get(result, "unknown")

    async def retry(self, job_id: str) -> bool:
        result = int(
            await self._redis_call(
                "eval",
                _RETRY_SCRIPT,
                4,
                self.ready_key,
                self.dead_key,
                self.cancelled_key,
                self._job_key(job_id),
                job_id,
                str(time.time()),
            )
        )
        if result == -1:
            raise JobNotFound(f"Job {job_id!r} was not found.")
        return result == 1

    async def get(self, job_id: str) -> Job:
        raw = await self._redis_call("hgetall", self._job_key(job_id))
        if not raw:
            raise JobNotFound(f"Job {job_id!r} was not found.")
        return self._job_from_raw(job_id, raw)

    def _job_from_raw(self, job_id: str, raw: Any) -> Job:
        """Decode one Redis hash into a validated job record."""

        values = {
            self._decode(key): self._decode(value)
            for key, value in dict(raw).items()
        }

        def optional_float(name: str) -> float | None:
            value = values.get(name, "")
            return float(value) if value else None

        try:
            payload = json.loads(values.get("payload") or "{}")
            result_raw = values.get("result", "")
            result = json.loads(result_raw) if result_raw else None
            return Job(
                id=values["id"],
                type=values["type"],
                payload=payload,
                state=values["state"],
                priority=int(values.get("priority") or 0),
                attempts=int(values.get("attempts") or 0),
                max_attempts=int(values.get("max_attempts") or 1),
                timeout_seconds=float(values.get("timeout_seconds") or 300.0),
                created_at=float(values.get("created_at") or 0.0),
                available_at=float(values.get("available_at") or 0.0),
                started_at=optional_float("started_at"),
                completed_at=optional_float("completed_at"),
                worker_id=values.get("worker_id", ""),
                lease_token=values.get("lease_token", ""),
                last_error=values.get("last_error", ""),
                result=result,
                cancel_requested=values.get("cancel_requested") == "1",
                progress_percent=int(values.get("progress_percent") or 0),
                progress_stage=values.get("progress_stage", ""),
                progress_detail=values.get("progress_detail", ""),
                updated_at=optional_float("updated_at"),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JobQueueError(f"Job {job_id!r} contains invalid Redis data.") from exc

    async def _get_many(self, job_ids: list[str]) -> list[Job]:
        """Load one job page with a single Redis pipeline round trip."""

        if not job_ids:
            return []

        def load() -> list[Any]:
            pipeline_factory = getattr(self.redis, "pipeline", None)
            if not callable(pipeline_factory):
                return [self.redis.hgetall(self._job_key(job_id)) for job_id in job_ids]

            pipeline = pipeline_factory(transaction=False)
            try:
                for job_id in job_ids:
                    pipeline.hgetall(self._job_key(job_id))
                return list(pipeline.execute())
            finally:
                reset = getattr(pipeline, "reset", None)
                if callable(reset):
                    reset()

        try:
            raw_jobs = await asyncio.to_thread(load)
        except Exception as exc:
            raise JobQueueError("Redis job queue bulk read failed.") from exc
        if len(raw_jobs) != len(job_ids):
            raise JobQueueError("Redis returned an invalid bulk job result.")

        jobs: list[Job] = []
        for job_id, raw in zip(job_ids, raw_jobs, strict=True):
            if raw:
                jobs.append(self._job_from_raw(job_id, raw))
        return jobs

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
        clean_detail = str(detail or "").strip()[:500]
        if not clean_stage:
            raise ValueError("Progress stage is required.")
        changed = await self._redis_call(
            "eval",
            _UPDATE_PROGRESS_SCRIPT,
            1,
            self._job_key(job_id),
            lease_token,
            str(max(0, min(100, int(percent)))),
            clean_stage,
            clean_detail,
            str(time.time()),
        )
        return bool(changed)

    async def list_jobs(
        self,
        *,
        state: str = "dead",
        limit: int = 50,
        cursor: str = "",
        job_type: str = "",
        query: str = "",
    ) -> tuple[list[Job], str | None]:
        """Return one filtered cursor page from a durable state index.

        The cursor is an opaque decimal offset.  It is intentionally simple: the
        admin surface is operational visibility, not a transactional export.
        Missing/expired job hashes are skipped safely. Filtered scans are bounded
        to 1,000 index entries per request to protect Redis from broad searches.
        """

        clean_state = str(state or "").strip().lower()
        keys = {
            "queued": self.ready_key,
            "running": self.leased_key,
            "dead": self.dead_key,
            "succeeded": self.succeeded_key,
            "cancelled": self.cancelled_key,
        }
        redis_key = keys.get(clean_state)
        if redis_key is None:
            raise ValueError(
                "state must be queued, running, dead, succeeded, or cancelled."
            )
        page_size = max(1, min(200, int(limit)))
        try:
            offset = max(0, int(str(cursor or "0")))
        except ValueError as exc:
            raise ValueError("Invalid job-list cursor.") from exc
        clean_type = str(job_type or "").strip().lower()
        clean_query = str(query or "").strip().lower()
        if len(clean_type) > 64:
            raise ValueError("Job type filter is too long.")
        if len(clean_query) > 128:
            raise ValueError("Job search query is too long.")

        def matches(job: Job) -> bool:
            if clean_type and job.type != clean_type:
                return False
            if not clean_query:
                return True
            searchable = "\n".join(
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
            return clean_query in searchable

<<<<<<< Updated upstream
        jobs: list[Job] = []
        scan_offset = offset
        scanned = 0
        max_scan = 1_000
        while len(jobs) < page_size and scanned < max_scan:
            fetch_count = min(201, max(51, page_size + 1), max_scan - scanned)
            raw_ids = await self._redis_call(
                "zrevrange",
                redis_key,
                scan_offset,
                scan_offset + fetch_count - 1,
            )
            ids = [self._decode(value) for value in list(raw_ids or ())]
            if not ids:
                return jobs, None
            loaded = {job.id: job for job in await self._get_many(ids)}
            for index, job_id in enumerate(ids):
                scan_offset += 1
                scanned += 1
                job = loaded.get(job_id)
                if job is not None and matches(job):
                    jobs.append(job)
                    if len(jobs) >= page_size:
                        has_more = index + 1 < len(ids) or len(ids) == fetch_count
                        return jobs, str(scan_offset) if has_more else None
            if len(ids) < fetch_count:
                return jobs, None
        return jobs, str(scan_offset) if scanned >= max_scan else None
=======
        raw_ids = await self._redis_call(
            "zrevrange",
            redis_key,
            offset,
            offset + page_size,
        )
        ids = [self._decode(value) for value in list(raw_ids or ())]
        has_more = len(ids) > page_size
        ids = ids[:page_size]
        jobs = await self._get_many(ids)
        next_cursor = str(offset + page_size) if has_more else None
        return jobs, next_cursor
>>>>>>> Stashed changes

    async def stats(self) -> dict[str, int | float]:
        now = time.time()
        cutoff = now - self.retention_seconds
        hour_ago = now - 3_600.0

        def load() -> list[Any]:
            operations = (
                ("zremrangebyscore", (self.dead_key, "-inf", cutoff), {}),
                ("zremrangebyscore", (self.succeeded_key, "-inf", cutoff), {}),
                ("zremrangebyscore", (self.cancelled_key, "-inf", cutoff), {}),
                ("zcard", (self.ready_key,), {}),
                ("zcard", (self.leased_key,), {}),
                ("zcard", (self.dead_key,), {}),
                ("zcard", (self.succeeded_key,), {}),
                ("zcard", (self.cancelled_key,), {}),
                ("zrange", (self.ready_key, 0, 0), {"withscores": True}),
                ("zcount", (self.succeeded_key, hour_ago, "+inf"), {}),
                ("zcount", (self.dead_key, hour_ago, "+inf"), {}),
                ("zcount", (self.cancelled_key, hour_ago, "+inf"), {}),
            )

            pipeline_factory = getattr(self.redis, "pipeline", None)
            if not callable(pipeline_factory):
                return [
                    getattr(self.redis, method)(*args, **kwargs)
                    for method, args, kwargs in operations
                ]

            pipeline = pipeline_factory(transaction=False)
            reset = getattr(pipeline, "reset", None)
            try:
                if not all(
                    callable(getattr(pipeline, method, None))
                    for method, _args, _kwargs in operations
                ):
                    return [
                        getattr(self.redis, method)(*args, **kwargs)
                        for method, args, kwargs in operations
                    ]
                for method, args, kwargs in operations:
                    getattr(pipeline, method)(*args, **kwargs)
                return list(pipeline.execute())
            finally:
                if callable(reset):
                    reset()

        try:
            values = await asyncio.to_thread(load)
        except Exception as exc:
            raise JobQueueError("Redis job queue stats read failed.") from exc
        if len(values) != 12:
            raise JobQueueError("Redis returned an invalid job stats result.")
        (
            _removed_dead,
            _removed_succeeded,
            _removed_cancelled,
            ready,
            running,
            dead,
            succeeded,
            cancelled,
            oldest_ready,
            succeeded_hour,
            failed_hour,
            cancelled_hour,
        ) = values
        queued = int(ready or 0)
        oldest_values = list(oldest_ready or ())
        oldest_score = float(oldest_values[0][1]) if oldest_values else now
        completed_recently = int(succeeded_hour or 0)
        failed_recently = int(failed_hour or 0)
        terminal_recently = completed_recently + failed_recently
        return {
            "queued": queued,
            "running": int(running or 0),
            "dead": int(dead or 0),
            "succeeded": int(succeeded or 0),
            "cancelled": int(cancelled or 0),
            "queue_limit": self.max_queued_jobs,
            "queue_available": max(0, self.max_queued_jobs - queued),
            "oldest_queued_age_seconds": round(max(0.0, now - oldest_score), 1),
            "succeeded_last_hour": completed_recently,
            "failed_last_hour": failed_recently,
            "cancelled_last_hour": int(cancelled_hour or 0),
            "throughput_per_minute": round(completed_recently / 60.0, 2),
            "failure_rate_percent": round(
                (failed_recently / terminal_recently) * 100.0,
                2,
            )
            if terminal_recently
            else 0.0,
        }


class RedisJobWorker:
    """Execute registered handlers while renewing each Redis lease."""

    def __init__(
        self,
        queue: RedisJobQueue,
        handlers: Mapping[str, JobHandler],
        *,
        worker_id: str = "",
        poll_interval_seconds: float = 0.5,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 300.0,
        can_claim: Callable[[], bool] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.queue = queue
        self.handlers = {
            str(name).strip().lower(): handler
            for name, handler in handlers.items()
        }
        self.worker_id = (
            str(worker_id or "").strip()
            or f"worker-{uuid.uuid4().hex[:12]}"
        )
        self.poll_interval_seconds = max(
            0.05,
            min(30.0, float(poll_interval_seconds)),
        )
        self.retry_base_seconds = max(0.1, float(retry_base_seconds))
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(retry_max_seconds),
        )
        self.can_claim = can_claim or (lambda: True)
        self.on_error = on_error

    def _record_error(self, exc: BaseException) -> None:
        """Publish a loop failure to the process-wide worker status table."""

        if self.on_error is None:
            return
        with suppress(Exception):
            self.on_error(f"{type(exc).__name__}: {exc}"[:500])

    async def _invoke(self, handler: JobHandler, context: JobContext) -> Any:
        if inspect.iscoroutinefunction(handler):
            return await handler(context.job.payload, context)
        value = await asyncio.to_thread(handler, context.job.payload, context)
        if inspect.isawaitable(value):
            return await value
        return value

    async def _heartbeat(
        self,
        job: Job,
        task: asyncio.Task[Any],
    ) -> None:
        interval = max(1.0, self.queue.lease_seconds / 3.0)
        consecutive_errors = 0
        while not task.done():
            await asyncio.sleep(interval)
            if task.done():
                return
            try:
                renewed = await self.queue.renew(job.id, job.lease_token)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient Redis boundary
                # A failed renew is retryable: the lease still has roughly two
                # thirds of its life left. Only give up once the lease can no
                # longer be saved, so a single Redis blip cannot orphan a job.
                consecutive_errors += 1
                logger.warning(
                    "Lease renewal failed job=%s attempt=%s: %s",
                    job.id,
                    consecutive_errors,
                    exc,
                )
                if consecutive_errors >= _MAX_HEARTBEAT_ERRORS:
                    task.cancel()
                    return
                continue
            consecutive_errors = 0
            if renewed != 1:
                task.cancel()
                return

    def _retry_delay(self, attempt: int) -> float:
        base = min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2 ** max(0, attempt - 1)),
        )
        return base * random.uniform(0.8, 1.2)

    async def process_one(self) -> bool:
        job = await self.queue.claim(self.worker_id)
        if job is None:
            return False
        handler = self.handlers.get(job.type)
        if handler is None:
            await self.queue.fail(
                job.id,
                job.lease_token,
                f"No handler is registered for job type {job.type!r}.",
                retryable=False,
            )
            return True

        context = JobContext(self.queue, job, self.worker_id, job.lease_token)
        execution = asyncio.create_task(
            self._invoke(handler, context),
            name=f"job-{job.id}",
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(job, execution),
            name=f"job-heartbeat-{job.id}",
        )
        try:
            result = await asyncio.wait_for(
                execution,
                timeout=job.timeout_seconds,
            )
            completed = await self.queue.complete(job.id, job.lease_token, result)
            if not completed:
                # The lease was lost (expired and swept, or the job was
                # cancelled), so this result is discarded. Surface it instead
                # of failing silently: another worker may redo the work.
                logger.warning(
                    "Job result discarded because the lease was no longer held "
                    "job=%s worker=%s",
                    job.id,
                    self.worker_id,
                )
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                execution.cancel()
                raise
            if execution.cancelled():
                await self.queue.fail(
                    job.id,
                    job.lease_token,
                    "Job was cancelled.",
                    retryable=False,
                )
                return True
            execution.cancel()
            raise
        except TimeoutError:
            await self.queue.fail(
                job.id,
                job.lease_token,
                f"Job timed out after {job.timeout_seconds:g} seconds.",
                retryable=True,
                retry_delay_seconds=self._retry_delay(job.attempts),
            )
        except Exception as exc:  # noqa: BLE001 - job handler boundary
            await self.queue.fail(
                job.id,
                job.lease_token,
                f"{type(exc).__name__}: {exc}",
                retryable=True,
                retry_delay_seconds=self._retry_delay(job.attempts),
            )
        finally:
            heartbeat.cancel()
            # The heartbeat may already have finished with an exception, in
            # which case cancel() is a no-op and awaiting it re-raises from a
            # finally block — replacing the real job outcome. Swallow anything
            # it raises; renewal failures are already logged in _heartbeat.
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat
        return True

    async def run(self, stop_event: asyncio.Event) -> None:
        consecutive_errors = 0
        while not stop_event.is_set():
            if not self.can_claim():
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=self.poll_interval_seconds,
                    )
                continue
            try:
                processed = await self.process_one()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - worker loop boundary
                # Claiming or bookkeeping failed (usually a Redis blip). The
                # worker must survive: letting this escape retires the task
                # permanently and silently drains the fleet.
                consecutive_errors += 1
                self._record_error(exc)
                logger.warning(
                    "Job worker iteration failed worker=%s streak=%s: %s",
                    self.worker_id,
                    consecutive_errors,
                    exc,
                    exc_info=True,
                )
                backoff = min(
                    self.retry_max_seconds,
                    self.poll_interval_seconds
                    * (2 ** min(consecutive_errors - 1, 6)),
                )
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                continue
            consecutive_errors = 0
            if processed:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.poll_interval_seconds,
                )


__all__ = [
    "Job",
    "JobContext",
    "JobHandler",
    "JobNotFound",
    "JobQueueError",
    "QueueFull",
    "RedisJobQueue",
    "RedisJobWorker",
]
