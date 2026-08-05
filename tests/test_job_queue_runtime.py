from __future__ import annotations

import json
import time
import unittest

from app.services.jobs.queue import QueueFull, RedisJobQueue


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.strings: dict[str, str] = {}

    def eval(self, script: str, key_count: int, *values):
        keys = list(values[:key_count])
        args = [str(value) for value in values[key_count:]]
        if "bot_voice:enqueue_v1" not in script:
            raise AssertionError("Unexpected Lua script")
        ready_key, job_key, idempotency_key = keys
        (
            job_id, job_type, payload, priority, created_at, available_at,
            max_attempts, timeout_seconds, use_idempotency, _ttl, max_queued,
        ) = args
        if use_idempotency == "1" and idempotency_key in self.strings:
            return [self.strings[idempotency_key], 0]
        if job_key in self.hashes:
            return [job_id, 0]
        if len(self.zsets.get(ready_key, {})) >= int(max_queued):
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
        }
        self.zsets.setdefault(ready_key, {})[job_id] = float(available_at)
        if use_idempotency == "1":
            self.strings[idempotency_key] = job_id
        return [job_id, 1]

    def hgetall(self, key: str):
        return dict(self.hashes.get(key, {}))

    def zcard(self, key: str) -> int:
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


class JobQueueRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_limit_applies_backpressure(self) -> None:
        queue = RedisJobQueue(FakeRedis(), redis_prefix="tests", max_queued_jobs=1)
        _job, created = await queue.enqueue("tts", {"chat_id": 1})
        self.assertTrue(created)
        with self.assertRaises(QueueFull):
            await queue.enqueue("ocr", {"chat_id": 1})

    async def test_dead_jobs_can_be_cursor_listed(self) -> None:
        redis = FakeRedis()
        queue = RedisJobQueue(redis, redis_prefix="tests")
        now = time.time()
        for index in range(3):
            job_id = f"dead-{index}"
            redis.hashes[queue._job_key(job_id)] = {
                "id": job_id,
                "type": "tts",
                "payload": json.dumps({"chat_id": index + 1}),
                "state": "dead",
                "priority": "0",
                "created_at": str(now + index),
                "available_at": str(now + index),
                "attempts": "3",
                "max_attempts": "3",
                "timeout_seconds": "10",
                "cancel_requested": "0",
                "last_error": "failed",
            }
            redis.zsets.setdefault(queue.dead_key, {})[job_id] = now + index

        first, cursor = await queue.list_jobs(state="dead", limit=2)
        second, next_cursor = await queue.list_jobs(
            state="dead",
            limit=2,
            cursor=cursor or "",
        )
        self.assertEqual(2, len(first))
        self.assertIsNotNone(cursor)
        self.assertEqual(1, len(second))
        self.assertIsNone(next_cursor)


if __name__ == "__main__":
    unittest.main()
