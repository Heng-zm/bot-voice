from __future__ import annotations

import unittest

from app.services.jobs.queue import RedisJobQueue


class TransitionRedis:
    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    def _zadd(self, key: str, score: float, member: str) -> None:
        self.zsets.setdefault(key, {})[member] = float(score)

    def _zrem(self, key: str, member: str) -> None:
        self.zsets.setdefault(key, {}).pop(member, None)

    def get(self, key: str):
        return self.strings.get(key)

    def hgetall(self, key: str):
        return dict(self.hashes.get(key, {}))

    def hset(self, key: str, *, mapping: dict[str, str]):
        self.hashes.setdefault(key, {}).update(
            {str(name): str(value) for name, value in mapping.items()}
        )
        return len(mapping)

    def zcard(self, key: str):
        return len(self.zsets.get(key, {}))

    def zrevrange(self, key: str, start: int, end: int):
        ordered = [
            member
            for member, _score in sorted(
                self.zsets.get(key, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        return ordered[start : end + 1]

    def zremrangebyscore(self, key: str, minimum, maximum):
        del minimum
        cutoff = float(maximum)
        members = self.zsets.setdefault(key, {})
        removed = [member for member, score in members.items() if score <= cutoff]
        for member in removed:
            members.pop(member, None)
        return len(removed)

    def eval(self, script: str, key_count: int, *values):
        keys = list(values[:key_count])
        args = [str(value) for value in values[key_count:]]
        if "bot_voice:enqueue_v1" in script:
            return self._enqueue(keys, args)
        if "bot_voice:claim_v1" in script:
            return self._claim(keys, args)
        if "bot_voice:renew_v1" in script:
            return self._renew(keys, args)
        if "bot_voice:update_progress_v1" in script:
            return self._update_progress(keys, args)
        if "bot_voice:complete_v1" in script:
            return self._complete(keys, args)
        if "bot_voice:fail_v1" in script:
            return self._fail(keys, args)
        if "bot_voice:cancel_v1" in script:
            return self._cancel(keys, args)
        if "bot_voice:retry_v1" in script:
            return self._retry(keys, args)
        raise AssertionError("Unexpected script")

    def _enqueue(self, keys: list[str], args: list[str]):
        ready, job_key, idempotency_key = keys
        (
            job_id,
            job_type,
            payload,
            priority,
            created_at,
            available_at,
            max_attempts,
            timeout_seconds,
            use_idempotency,
            _idempotency_ttl,
            max_queued,
            job_prefix,
        ) = args
        if use_idempotency == "1" and idempotency_key in self.strings:
            existing = self.strings[idempotency_key]
            if f"{job_prefix}{existing}" in self.hashes:
                return [existing, 0]
            self.strings.pop(idempotency_key, None)
        if len(self.zsets.get(ready, {})) >= int(max_queued):
            return ["", -1]
        self.hashes[job_key] = {
            "id": job_id,
            "type": job_type,
            "payload": payload,
            "state": "queued",
            "priority": priority,
            "created_at": created_at,
            "available_at": available_at,
            "attempts": "0",
            "max_attempts": max_attempts,
            "timeout_seconds": timeout_seconds,
            "cancel_requested": "0",
            "progress_percent": "0",
            "progress_stage": "queued",
            "progress_detail": "",
            "updated_at": created_at,
        }
        self._zadd(ready, float(available_at), job_id)
        if use_idempotency == "1":
            self.strings[idempotency_key] = job_id
        return [job_id, 1]

    def _claim(self, keys: list[str], args: list[str]):
        ready, leased, dead, cancelled = keys
        now, worker, token, deadline, _retention, job_prefix = args
        del dead, cancelled
        candidates = [
            job_id
            for job_id, score in self.zsets.get(ready, {}).items()
            if score <= float(now)
        ]
        if not candidates:
            return None
        job_id = min(
            candidates,
            key=lambda value: float(
                self.hashes[f"{job_prefix}{value}"]["created_at"]
            ),
        )
        data = self.hashes[f"{job_prefix}{job_id}"]
        data.update(
            state="running",
            attempts=str(int(data["attempts"]) + 1),
            worker_id=worker,
            lease_token=token,
            lease_deadline=deadline,
            started_at=now,
            cancel_requested="0",
            progress_stage="running",
            updated_at=now,
        )
        self._zrem(ready, job_id)
        self._zadd(leased, float(deadline), job_id)
        return job_id

    def _renew(self, keys: list[str], args: list[str]):
        leased, job_key = keys
        token, deadline, job_id, updated_at = args
        data = self.hashes[job_key]
        if data.get("lease_token") != token or data.get("state") != "running":
            return 0
        data.update(lease_deadline=deadline, updated_at=updated_at)
        self._zadd(leased, float(deadline), job_id)
        return 1

    def _update_progress(self, keys: list[str], args: list[str]):
        (job_key,) = keys
        token, percent, stage, detail, updated_at = args
        data = self.hashes[job_key]
        if data.get("lease_token") != token or data.get("state") != "running":
            return 0
        data.update(
            progress_percent=percent,
            progress_stage=stage,
            progress_detail=detail,
            updated_at=updated_at,
        )
        return 1

    def _complete(self, keys: list[str], args: list[str]):
        leased, job_key, succeeded, cancelled = keys
        token, job_id, completed_at, result, _retention = args
        data = self.hashes[job_key]
        if data.get("lease_token") != token or data.get("state") != "running":
            return 0
        self._zrem(leased, job_id)
        if data.get("cancel_requested") == "1":
            data.update(
                state="cancelled",
                completed_at=completed_at,
                last_error="cancelled",
                progress_stage="cancelled",
                updated_at=completed_at,
            )
            self._zadd(cancelled, float(completed_at), job_id)
        else:
            data.update(
                state="succeeded",
                completed_at=completed_at,
                result=result,
                last_error="",
                progress_percent="100",
                progress_stage="succeeded",
                updated_at=completed_at,
            )
            self._zadd(succeeded, float(completed_at), job_id)
        for field in ("lease_token", "lease_deadline", "worker_id"):
            data.pop(field, None)
        return 1

    def _fail(self, keys: list[str], args: list[str]):
        ready, leased, job_key, dead, cancelled = keys
        token, job_id, now, error, available_at, retryable, _retention = args
        data = self.hashes[job_key]
        if data.get("lease_token") != token or data.get("state") != "running":
            return 0
        self._zrem(leased, job_id)
        if data.get("cancel_requested") == "1":
            data.update(
                state="cancelled",
                completed_at=now,
                last_error="cancelled",
                progress_stage="cancelled",
                updated_at=now,
            )
            self._zadd(cancelled, float(now), job_id)
            for field in ("lease_token", "lease_deadline", "worker_id"):
                data.pop(field, None)
            return 2
        if retryable == "1" and int(data["attempts"]) < int(data["max_attempts"]):
            data.update(
                state="queued",
                available_at=available_at,
                last_error=error,
                progress_stage="retrying",
                updated_at=now,
            )
            for field in ("lease_token", "lease_deadline", "worker_id"):
                data.pop(field, None)
            self._zadd(ready, float(available_at), job_id)
            return 1
        data.update(
            state="dead",
            completed_at=now,
            last_error=error,
            progress_stage="dead",
            updated_at=now,
        )
        for field in ("lease_token", "lease_deadline", "worker_id"):
            data.pop(field, None)
        self._zadd(dead, float(now), job_id)
        return 3

    def _cancel(self, keys: list[str], args: list[str]):
        ready, _leased, job_key, cancelled = keys
        job_id, now, _retention = args
        data = self.hashes.get(job_key)
        if data is None:
            return -1
        if data["state"] == "queued":
            self._zrem(ready, job_id)
            data.update(
                state="cancelled",
                cancel_requested="1",
                completed_at=now,
                last_error="cancelled",
                progress_stage="cancelled",
                updated_at=now,
            )
            self._zadd(cancelled, float(now), job_id)
            return 1
        if data["state"] == "running":
            data.update(
                cancel_requested="1",
                progress_stage="cancelling",
                updated_at=now,
            )
            return 2
        return 0

    def _retry(self, keys: list[str], args: list[str]):
        ready, dead, cancelled, job_key = keys
        job_id, now = args
        data = self.hashes.get(job_key)
        if data is None:
            return -1
        if data["state"] not in {"dead", "cancelled"}:
            return 0
        self._zrem(dead, job_id)
        self._zrem(cancelled, job_id)
        data.update(
            state="queued",
            available_at=now,
            attempts="0",
            cancel_requested="0",
            last_error="",
            progress_percent="0",
            progress_stage="queued",
            progress_detail="",
            updated_at=now,
        )
        data.pop("completed_at", None)
        data.pop("result", None)
        data.pop("started_at", None)
        for field in ("lease_token", "lease_deadline", "worker_id"):
            data.pop(field, None)
        self._zadd(ready, float(now), job_id)
        return 1


class JobQueueTransitionV2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_and_success_are_visible(self) -> None:
        queue = RedisJobQueue(TransitionRedis(), redis_prefix="tests")
        job, _ = await queue.enqueue("tts", {"text": "hello"})
        running = await queue.claim("worker-1")
        self.assertEqual(job.id, running.id)
        self.assertTrue(
            await queue.update_progress(
                job.id,
                running.lease_token,
                percent=60,
                stage="generating_voice",
            )
        )
        self.assertTrue(
            await queue.complete(job.id, running.lease_token, {"ok": True})
        )
        completed = await queue.get(job.id)
        self.assertEqual("succeeded", completed.state)
        self.assertEqual(100, completed.progress_percent)
        listed, _ = await queue.list_jobs(state="succeeded")
        self.assertEqual([job.id], [item.id for item in listed])

    async def test_cancelled_job_can_be_retried(self) -> None:
        queue = RedisJobQueue(TransitionRedis(), redis_prefix="tests")
        job, _ = await queue.enqueue("ocr", {"file_id": "telegram-file"})
        self.assertEqual("cancelled", await queue.cancel(job.id))
        cancelled = await queue.get(job.id)
        self.assertEqual("cancelled", cancelled.state)
        self.assertTrue(await queue.retry(job.id))
        retried = await queue.get(job.id)
        self.assertEqual("queued", retried.state)
        self.assertEqual(0, retried.progress_percent)

    async def test_non_retryable_failure_enters_dead_index(self) -> None:
        queue = RedisJobQueue(TransitionRedis(), redis_prefix="tests")
        job, _ = await queue.enqueue(
            "transcription",
            {"file_id": "telegram-file"},
            max_attempts=1,
        )
        running = await queue.claim("worker-1")
        state = await queue.fail(
            job.id,
            running.lease_token,
            RuntimeError("provider unavailable"),
            retryable=False,
        )
        self.assertEqual("dead", state)
        dead, _ = await queue.list_jobs(state="dead")
        self.assertEqual([job.id], [item.id for item in dead])
        self.assertIn("provider unavailable", dead[0].last_error)

    async def test_running_cancellation_clears_lease_metadata(self) -> None:
        queue = RedisJobQueue(TransitionRedis(), redis_prefix="tests")
        job, _ = await queue.enqueue("tts", {"text": "hello"})
        running = await queue.claim("worker-1")
        self.assertEqual("requested", await queue.cancel(job.id))

        self.assertEqual(
            "cancelled",
            await queue.fail(job.id, running.lease_token, "cancelled"),
        )
        cancelled = await queue.get(job.id)
        self.assertEqual("", cancelled.lease_token)
        self.assertEqual("", cancelled.worker_id)

        self.assertTrue(await queue.retry(job.id))
        retried = await queue.get(job.id)
        self.assertIsNone(retried.started_at)
        self.assertIsNone(retried.completed_at)


if __name__ == "__main__":
    unittest.main()
