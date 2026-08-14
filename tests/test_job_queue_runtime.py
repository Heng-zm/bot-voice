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
        self.pipeline_execute_calls = 0

    def eval(self, script: str, key_count: int, *values):
        keys = list(values[:key_count])
        args = [str(value) for value in values[key_count:]]
        if "bot_voice:enqueue_v1" not in script:
            raise AssertionError("Unexpected Lua script")
        ready_key, job_key, idempotency_key = keys
        (
            job_id, job_type, payload, priority, created_at, available_at,
            max_attempts, timeout_seconds, use_idempotency, _ttl, max_queued,
            job_prefix,
        ) = args
        if use_idempotency == "1" and idempotency_key in self.strings:
            existing = self.strings[idempotency_key]
            if f"{job_prefix}{existing}" in self.hashes:
                return [existing, 0]
            self.strings.pop(idempotency_key, None)
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

    def pipeline(self, *, transaction: bool = False):
        assert not transaction
        redis = self

        class Pipeline:
            def __init__(self) -> None:
                self.operations: list[tuple[str, tuple, dict]] = []

            def hgetall(self, key: str):
                self.operations.append(("hgetall", (key,), {}))
                return self

            def zremrangebyscore(self, key: str, minimum, maximum):
                self.operations.append(
                    ("zremrangebyscore", (key, minimum, maximum), {})
                )
                return self

            def zcard(self, key: str):
                self.operations.append(("zcard", (key,), {}))
                return self

            def zrange(self, key: str, start: int, end: int, **kwargs):
                self.operations.append(("zrange", (key, start, end), kwargs))
                return self

            def zcount(self, key: str, minimum, maximum):
                self.operations.append(("zcount", (key, minimum, maximum), {}))
                return self

            def execute(self):
                redis.pipeline_execute_calls += 1
                return [
                    getattr(redis, method)(*args, **kwargs)
                    for method, args, kwargs in self.operations
                ]

            def reset(self) -> None:
                self.operations.clear()

        return Pipeline()

    def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, {}))

    def zcount(self, key: str, minimum, maximum) -> int:
        low = float(minimum)
        high = float("inf") if maximum == "+inf" else float(maximum)
        return sum(low <= score <= high for score in self.zsets.get(key, {}).values())

    def zrange(self, key: str, start: int, end: int, *, withscores: bool = False):
        ordered = sorted(self.zsets.get(key, {}).items(), key=lambda item: item[1])
        values = ordered[start : end + 1]
        return values if withscores else [member for member, _score in values]

    def zremrangebyscore(self, key: str, minimum, maximum) -> int:
        del minimum
        cutoff = float(maximum)
        values = self.zsets.setdefault(key, {})
        expired = [member for member, score in values.items() if score <= cutoff]
        for member in expired:
            values.pop(member, None)
        return len(expired)

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
    async def test_stale_idempotency_pointer_does_not_hide_new_job(self) -> None:
        redis = FakeRedis()
        queue = RedisJobQueue(redis, redis_prefix="tests")
        redis.strings[queue._idempotency_key("request-1")] = "expired-job"

        job, created = await queue.enqueue(
            "tts",
            {"chat_id": 1},
            idempotency_key="request-1",
        )

        self.assertTrue(created)
        self.assertNotEqual("expired-job", job.id)

    async def test_async_redis_pipeline_supports_listing_and_stats(self) -> None:
        class AsyncPipelineRedis(FakeRedis):
            def pipeline(self, *, transaction: bool = False):
                inner = super().pipeline(transaction=transaction)

                class AsyncPipeline:
                    def __getattr__(self, name):
                        operation = getattr(inner, name)

                        def queue(*args, **kwargs):
                            operation(*args, **kwargs)
                            return self

                        return queue

                    async def execute(self):
                        return inner.execute()

                    async def reset(self) -> None:
                        inner.reset()

                return AsyncPipeline()

        redis = AsyncPipelineRedis()
        queue = RedisJobQueue(redis, redis_prefix="tests")
        job, _created = await queue.enqueue("tts", {"chat_id": 1})

        jobs, cursor = await queue.list_jobs(state="queued")
        stats = await queue.stats()

        self.assertEqual([job.id], [item.id for item in jobs])
        self.assertIsNone(cursor)
        self.assertEqual(1, stats["queued"])
        self.assertEqual(2, redis.pipeline_execute_calls)

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
        self.assertEqual(2, redis.pipeline_execute_calls)
    async def test_job_filters_and_queue_metrics(self) -> None:
        redis = FakeRedis()
        queue = RedisJobQueue(redis, redis_prefix="tests")
        now = time.time()
        for index, job_type in enumerate(("tts", "ocr", "tts")):
            job_id = f"filter-{index}"
            redis.hashes[queue._job_key(job_id)] = {
                "id": job_id,
                "type": job_type,
                "payload": "{}",
                "state": "dead",
                "priority": "0",
                "created_at": str(now - index),
                "available_at": str(now - index),
                "attempts": "3",
                "max_attempts": "3",
                "timeout_seconds": "10",
                "cancel_requested": "0",
                "last_error": "provider timeout" if index == 2 else "failed",
            }
            redis.zsets.setdefault(queue.dead_key, {})[job_id] = now - index

        filtered, cursor = await queue.list_jobs(
            state="dead",
            job_type="tts",
            query="timeout",
        )
        self.assertEqual(["filter-2"], [job.id for job in filtered])
        self.assertIsNone(cursor)

        redis.zsets.setdefault(queue.ready_key, {})["queued"] = now - 45
        redis.zsets.setdefault(queue.succeeded_key, {})["success"] = now - 60
        pipeline_calls = redis.pipeline_execute_calls
        metrics = await queue.stats()
        self.assertEqual(pipeline_calls + 1, redis.pipeline_execute_calls)
        self.assertGreaterEqual(metrics["oldest_queued_age_seconds"], 45)
        self.assertEqual(1, metrics["succeeded_last_hour"])
        self.assertEqual(3, metrics["failed_last_hour"])
        self.assertEqual(75.0, metrics["failure_rate_percent"])


if __name__ == "__main__":
    unittest.main()
