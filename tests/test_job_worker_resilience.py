"""Regression tests for durable worker survival across transient failures.

A worker task that raises is never recreated, so a single Redis blip used to
retire it permanently and silently drain the fleet. These tests pin the
recovery behaviour instead of the exact log output.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from app.services.jobs.queue import Job, JobQueueError, RedisJobWorker
from app.services.jobs.runtime import (
    _WORKER_RESTART_HISTORY,
    _WORKER_STATUS,
    _worker_runner,
    job_worker_snapshot,
)


def _job(job_id: str = "job-1", job_type: str = "tts") -> Job:
    now = 1_000.0
    return Job(
        id=job_id,
        type=job_type,
        payload={"chat_id": 1},
        state="running",
        priority=0,
        attempts=1,
        max_attempts=3,
        timeout_seconds=30.0,
        created_at=now,
        available_at=now,
        worker_id="worker-1",
        lease_token="token-1",
        cancel_requested=False,
        updated_at=now,
    )


class RecordingQueue:
    """Minimal queue double that scripts claim/renew/complete outcomes."""

    lease_seconds = 30.0

    def __init__(self, claim_results: list[object]) -> None:
        self._claim_results = list(claim_results)
        self.claim_calls = 0
        self.completed: list[str] = []
        self.failures: list[tuple[str, str]] = []
        self.renew_calls = 0
        self.renew_results: list[object] = []

    async def claim(self, worker_id: str):
        del worker_id
        self.claim_calls += 1
        if not self._claim_results:
            return None
        outcome = self._claim_results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def renew(self, job_id: str, lease_token: str) -> int:
        del job_id, lease_token
        self.renew_calls += 1
        if not self.renew_results:
            return 1
        outcome = self.renew_results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return int(outcome)

    async def complete(self, job_id: str, lease_token: str, result) -> bool:
        del lease_token, result
        self.completed.append(job_id)
        return True

    async def fail(self, job_id, lease_token, error, **kwargs) -> str:
        del lease_token, kwargs
        self.failures.append((job_id, str(error)))
        return "queued"

    async def get(self, job_id: str):
        return _job(job_id)


class WorkerResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_runner_restarts_an_unexpectedly_stopped_loop(self) -> None:
        _WORKER_RESTART_HISTORY.clear()
        recovered = asyncio.Event()

        class RestartingWorker:
            worker_id = "worker-restart"

            def __init__(self) -> None:
                self.calls = 0

            async def run(self, stop_event: asyncio.Event) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("unexpected exit")
                recovered.set()
                await stop_event.wait()

        worker = RestartingWorker()
        stop = asyncio.Event()
        with patch(
            "app.services.jobs.runtime._WORKER_RESTART_BASE_SECONDS",
            0.01,
        ):
            task = asyncio.create_task(_worker_runner(worker, stop))
            try:
                await asyncio.wait_for(recovered.wait(), timeout=1.0)
            finally:
                stop.set()
                await asyncio.wait_for(task, timeout=1.0)
                _WORKER_STATUS.pop(worker.worker_id, None)

        self.assertEqual(2, worker.calls)
        snapshot = job_worker_snapshot()
        self.assertEqual(1, snapshot["restart_total"])
        self.assertEqual(
            worker.worker_id,
            snapshot["restart_history"][0]["worker_id"],
        )
        self.assertIn("unexpected exit", snapshot["restart_history"][0]["error"])
        _WORKER_RESTART_HISTORY.clear()

    async def test_restart_backoff_resets_after_a_stable_worker_run(self) -> None:
        recovered = asyncio.Event()

        class RestartingWorker:
            worker_id = "worker-stable-restart"

            def __init__(self) -> None:
                self.calls = 0

            async def run(self, stop_event: asyncio.Event) -> None:
                self.calls += 1
                if self.calls <= 2:
                    raise RuntimeError(f"unexpected exit {self.calls}")
                recovered.set()
                await stop_event.wait()

        worker = RestartingWorker()
        stop = asyncio.Event()
        with (
            patch(
                "app.services.jobs.runtime._WORKER_RESTART_BASE_SECONDS",
                0.01,
            ),
            patch(
                "app.services.jobs.runtime._WORKER_STABLE_RUN_SECONDS",
                5.0,
            ),
            patch(
                "app.services.jobs.runtime._monotonic",
                side_effect=(0.0, 1.0, 10.0, 20.0, 30.0),
            ),
        ):
            task = asyncio.create_task(_worker_runner(worker, stop))
            try:
                await asyncio.wait_for(recovered.wait(), timeout=1.0)
                status = _WORKER_STATUS[worker.worker_id]
                self.assertEqual(2, status["restart_count"])
                self.assertEqual(1, status["restart_streak"])
            finally:
                stop.set()
                await asyncio.wait_for(task, timeout=1.0)
                _WORKER_STATUS.pop(worker.worker_id, None)

        self.assertEqual(3, worker.calls)

    async def test_transient_claim_error_does_not_kill_the_worker(self) -> None:
        """A Redis error during claim must be absorbed, not retire the task."""

        queue = RecordingQueue([JobQueueError("connection reset"), _job()])
        handled = asyncio.Event()

        async def handler(payload, context):
            del payload, context
            handled.set()
            return {"ok": True}

        worker = RedisJobWorker(
            queue,
            {"tts": handler},
            worker_id="worker-1",
            poll_interval_seconds=0.05,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        try:
            await asyncio.wait_for(handled.wait(), timeout=5.0)
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)

        # The worker survived the first failure and went on to run the job.
        self.assertFalse(task.cancelled())
        self.assertIsNone(task.exception())
        self.assertEqual(["job-1"], queue.completed)

    async def test_loop_errors_are_reported_for_health_reporting(self) -> None:
        queue = RecordingQueue([JobQueueError("connection reset"), _job()])
        errors: list[str] = []
        handled = asyncio.Event()

        async def handler(payload, context):
            del payload, context
            handled.set()
            return {"ok": True}

        worker = RedisJobWorker(
            queue,
            {"tts": handler},
            worker_id="worker-1",
            poll_interval_seconds=0.05,
            on_error=errors.append,
        )
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        try:
            await asyncio.wait_for(handled.wait(), timeout=5.0)
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)

        self.assertTrue(errors, "the loop failure should be reported")
        self.assertIn("connection reset", errors[0])

    async def test_failing_heartbeat_does_not_mask_a_successful_job(self) -> None:
        """A renew error must not replace the job outcome from the finally block."""

        queue = RecordingQueue([_job()])
        # Renew is scheduled at lease_seconds/3 but floored at one second, so
        # this yields the minimum 1s interval.
        queue.lease_seconds = 1.0
        # Force the heartbeat to fail while the handler is still running.
        queue.renew_results = [JobQueueError("renew failed")]

        async def handler(payload, context):
            del payload, context
            # The renew interval is floored at one second, so the job must
            # outlive that for the failing heartbeat to land mid-flight.
            await asyncio.sleep(1.4)
            return {"ok": True}

        worker = RedisJobWorker(
            queue,
            {"tts": handler},
            worker_id="worker-1",
            poll_interval_seconds=0.05,
        )

        processed = await worker.process_one()

        self.assertTrue(processed)
        self.assertGreaterEqual(queue.renew_calls, 1, "heartbeat must have run")
        # The successful result survived despite the heartbeat exception.
        self.assertEqual(["job-1"], queue.completed)
        self.assertEqual([], queue.failures)

    async def test_missing_handler_fails_the_job_without_retry(self) -> None:
        queue = RecordingQueue([_job(job_type="unregistered")])
        worker = RedisJobWorker(queue, {}, worker_id="worker-1")

        processed = await worker.process_one()

        self.assertTrue(processed)
        self.assertEqual(1, len(queue.failures))
        self.assertIn("No handler", queue.failures[0][1])


if __name__ == "__main__":
    unittest.main()
