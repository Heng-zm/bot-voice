from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from app.runtime import RuntimeContext, redis_runtime_enabled
from app.services.jobs.memory import MemoryJobQueue
from app.services.jobs.queue import JobQueueError, RedisJobWorker
from app.services.jobs.runtime import configure_job_queue, enqueue_bot_job


class RedisSwitchTests(unittest.TestCase):
    def test_missing_redis_url_defaults_to_disabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(redis_runtime_enabled())

    def test_existing_redis_url_preserves_legacy_enabled_default(self) -> None:
        with patch.dict(
            "os.environ",
            {"REDIS_URL": "redis://localhost:6379"},
            clear=True,
        ):
            self.assertTrue(redis_runtime_enabled())

    def test_redis_enabled_false_disables_redis(self) -> None:
        with patch.dict("os.environ", {"REDIS_ENABLED": "false"}, clear=False):
            self.assertFalse(redis_runtime_enabled())

    def test_disable_redis_alias_takes_precedence(self) -> None:
        with patch.dict(
            "os.environ",
            {"REDIS_ENABLED": "true", "DISABLE_REDIS": "true"},
            clear=False,
        ):
            self.assertFalse(redis_runtime_enabled())

    def test_runtime_snapshot_identifies_process_local_queue(self) -> None:
        runtime = RuntimeContext()
        runtime.redis_enabled = False
        runtime.job_queue = MemoryJobQueue()

        snapshot = runtime.snapshot()

        self.assertEqual("memory", snapshot["job_queue_backend"])
        self.assertFalse(snapshot["job_queue_durable"])


class MemoryJobQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_rejects_memory_queue_without_healthy_worker(
        self,
    ) -> None:
        configure_job_queue(None, memory_fallback=True)
        try:
            with patch(
                "app.services.jobs.runtime.job_worker_snapshot",
                return_value={"accepting": True, "healthy": False},
            ), self.assertRaises(JobQueueError):
                await enqueue_bot_job("tts", {"chat_id": 1})
        finally:
            configure_job_queue(None, memory_fallback=False)

    async def test_invalid_queue_limit_environment_uses_safe_default(self) -> None:
        with patch.dict("os.environ", {"BOT_JOB_QUEUE_MAX": "invalid"}):
            queue = configure_job_queue(None, memory_fallback=True)
        try:
            self.assertIsInstance(queue, MemoryJobQueue)
            self.assertEqual(1_000, queue.max_queued_jobs)
        finally:
            configure_job_queue(None, memory_fallback=False)

    async def test_concurrent_memory_enqueues_and_claims_are_atomic(self) -> None:
        queue = MemoryJobQueue()
        duplicate_results = await asyncio.gather(
            *(
                queue.enqueue(
                    "tts",
                    {"request": index},
                    idempotency_key="one-request",
                )
                for index in range(20)
            )
        )
        self.assertEqual(1, sum(created for _job, created in duplicate_results))
        self.assertEqual(1, len({job.id for job, _created in duplicate_results}))

        queue = MemoryJobQueue()
        await asyncio.gather(
            *(queue.enqueue("tts", {"request": index}) for index in range(20))
        )
        claimed = await asyncio.gather(
            *(queue.claim(f"worker-{index}") for index in range(20))
        )
        self.assertNotIn(None, claimed)
        self.assertEqual(20, len({job.id for job in claimed if job is not None}))

    async def test_memory_job_payload_is_isolated(self) -> None:
        queue = MemoryJobQueue()
        payload = {"nested": {"value": 1}}
        job, _created = await queue.enqueue("tts", payload)
        payload["nested"]["value"] = 99
        job.payload["nested"]["value"] = 88

        stored = await queue.get(job.id)
        self.assertEqual(1, stored.payload["nested"]["value"])
        stored.payload["nested"]["value"] = 77
        self.assertEqual(1, (await queue.get(job.id)).payload["nested"]["value"])

    async def test_memory_idempotency_key_expires(self) -> None:
        queue = MemoryJobQueue()
        with patch(
            "app.services.jobs.memory.time.time",
            side_effect=(100.0, 161.0),
        ):
            first, first_created = await queue.enqueue(
                "tts",
                {"text": "first"},
                idempotency_key="request-1",
                idempotency_ttl_seconds=60,
            )
            second, second_created = await queue.enqueue(
                "tts",
                {"text": "second"},
                idempotency_key="request-1",
                idempotency_ttl_seconds=60,
            )

        self.assertTrue(first_created)
        self.assertTrue(second_created)
        self.assertNotEqual(first.id, second.id)

    async def test_running_cancellation_interrupts_memory_worker_heartbeat(self) -> None:
        queue = MemoryJobQueue()
        job, _created = await queue.enqueue("tts", {"text": "hello"})
        running = await queue.claim("worker-1")

        self.assertEqual("requested", await queue.cancel(job.id))
        self.assertEqual(-1, await queue.renew(job.id, running.lease_token))
        cancelling = await queue.get(job.id)
        self.assertEqual("cancelling", cancelling.progress_stage)

    async def test_expired_memory_lease_records_retry_reason(self) -> None:
        queue = MemoryJobQueue()
        job, _created = await queue.enqueue("tts", {"text": "hello"})
        await queue.claim("worker-1")
        queue._lease_deadlines[job.id] = 0.0

        retried = await queue.get(job.id)

        self.assertEqual("queued", retried.state)
        self.assertEqual("retrying", retried.progress_stage)
        self.assertEqual("worker lease expired", retried.last_error)

    async def test_worker_processes_memory_job_without_redis(self) -> None:
        queue = MemoryJobQueue()
        received: list[dict] = []

        async def handle(payload, context):
            received.append(dict(payload))
            await context.progress(50, "working")
            return {"ok": True}

        job, created = await queue.enqueue(
            "tts",
            {"text": "hello"},
            idempotency_key="memory-job",
        )
        duplicate, duplicate_created = await queue.enqueue(
            "tts",
            {"text": "hello"},
            idempotency_key="memory-job",
        )
        worker = RedisJobWorker(queue, {"tts": handle}, worker_id="memory-worker")

        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(job.id, duplicate.id)
        self.assertTrue(await worker.process_one())
        completed = await queue.get(job.id)
        self.assertEqual("succeeded", completed.state)
        self.assertEqual({"ok": True}, completed.result)
        self.assertEqual([{"text": "hello"}], received)


if __name__ == "__main__":
    unittest.main()
