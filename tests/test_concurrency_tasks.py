"""Comprehensive tests for high-concurrency scaling and background task engine."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

from app.services.tasks.batcher import DatabaseBatcher, get_database_batcher
from app.services.tasks.queue import BackgroundTaskManager, get_task_manager
from app.services.telegram.dispatcher import TelegramDispatcher
from app.services.telegram.telemetry import get_scaling_telemetry_snapshot
from app.services.telegram.workloads import get_telegram_workload_limiter


class TestBackgroundTaskManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.manager = BackgroundTaskManager()

    async def test_submit_and_complete_coroutine(self) -> None:
        executed = []

        async def _sample_job(val: int) -> int:
            await asyncio.sleep(0.01)
            executed.append(val)
            return val * 2

        task = self.manager.submit(_sample_job, 5, name="test-coro")
        result = await task
        self.assertEqual(result, 10)
        self.assertIn(5, executed)

        metrics = self.manager.get_metrics()
        self.assertEqual(metrics["submitted"], 1)
        self.assertEqual(metrics["completed"], 1)
        self.assertEqual(metrics["failed"], 0)

    async def test_submit_with_error_and_handler(self) -> None:
        errors = []

        async def _failing_job() -> None:
            await asyncio.sleep(0.01)
            raise ValueError("Intentional error")

        def _on_error(exc: Exception) -> None:
            errors.append(str(exc))

        task = self.manager.submit(_failing_job, on_error=_on_error, name="failing-job")
        with self.assertRaises(ValueError):
            await task

        self.assertEqual(len(errors), 1)
        self.assertIn("Intentional error", errors[0])

        metrics = self.manager.get_metrics()
        self.assertEqual(metrics["failed"], 1)

    async def test_category_concurrency_limiting(self) -> None:
        max_concurrent_seen = 0
        current_running = 0

        async def _tracked_job() -> None:
            nonlocal max_concurrent_seen, current_running
            current_running += 1
            max_concurrent_seen = max(max_concurrent_seen, current_running)
            await asyncio.sleep(0.05)
            current_running -= 1

        # Submit 10 jobs to media category with cap 2
        self.manager._category_semaphores["media"] = asyncio.Semaphore(2)
        tasks = [self.manager.submit(_tracked_job, category="media") for _ in range(8)]
        await asyncio.gather(*tasks)

        self.assertLessEqual(max_concurrent_seen, 2)

    async def test_drain_cancels_pending_cleanly(self) -> None:
        async def _long_job() -> None:
            await asyncio.sleep(10.0)

        task = self.manager.submit(_long_job, name="long-job")
        t0 = time.monotonic()
        await self.manager.drain(timeout=0.1)
        dur = time.monotonic() - t0

        self.assertLess(dur, 1.0)
        self.assertTrue(task.done())


class TestDatabaseBatcher(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.batcher = DatabaseBatcher()

    async def test_user_activity_deduplication(self) -> None:
        # User sending 5 rapid messages should only result in 1 buffered entry
        self.batcher.record_user_activity(1001, username="alice", first_name="Alice", last_active="2026-09-22T10:00:00Z")
        self.batcher.record_user_activity(1001, username="alice", first_name="Alice", last_active="2026-09-22T10:00:05Z")
        self.batcher.record_user_activity(1001, username="alice_updated", first_name="Alice A", last_active="2026-09-22T10:00:10Z")

        with self.batcher._lock:
            self.assertEqual(len(self.batcher._user_buffer), 1)
            entry = self.batcher._user_buffer[1001]
            self.assertEqual(entry["username"], "alice_updated")
            self.assertEqual(entry["first_name"], "Alice A")
            self.assertEqual(entry["last_active"], "2026-09-22T10:00:10Z")

    async def test_batch_flush_sync_with_mock_client(self) -> None:
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[])

        self.batcher.record_user_activity(1001, username="alice")
        self.batcher.record_user_activity(1002, username="bob")
        self.batcher.record_text_cache({"message_id": 501, "chat_id": 999, "original_text": "hello"})

        with patch.object(self.batcher, "_get_supabase_client", return_value=mock_client):
            users_flushed, text_flushed = self.batcher.flush_sync()

        self.assertEqual(users_flushed, 2)
        self.assertEqual(text_flushed, 1)
        self.assertEqual(mock_table.upsert.call_count, 2)

        # Buffer should now be empty
        metrics = self.batcher.get_metrics()
        self.assertEqual(metrics["users_buffered"], 0)
        self.assertEqual(metrics["text_cache_buffered"], 0)
        self.assertEqual(metrics["users_flushed_total"], 2)
        self.assertEqual(metrics["text_cache_flushed_total"], 1)

    async def test_worker_loop_and_drain(self) -> None:
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[])

        self.batcher._flush_interval_s = 0.05
        with patch.object(self.batcher, "_get_supabase_client", return_value=mock_client):
            task = self.batcher.start_worker()
            self.assertIsNotNone(task)

            self.batcher.record_user_activity(2001, username="charlie")
            await asyncio.sleep(0.1)

            await self.batcher.drain(timeout=1.0)

        self.assertFalse(self.batcher._running)
        self.assertEqual(self.batcher.get_metrics()["users_flushed_total"], 1)


class TestConcurrencyAndWorkloadScaling(unittest.TestCase):
    def test_dispatcher_concurrency_defaults(self) -> None:
        dispatcher = TelegramDispatcher()
        metrics = dispatcher.get_metrics()
        # Default concurrency must be at least 64 and queue depth at least 500
        self.assertGreaterEqual(metrics["concurrency_limit"], 64)
        self.assertGreaterEqual(metrics["max_queue_depth"], 500)

    def test_high_priority_update_detection(self) -> None:
        mock_update_callback = MagicMock()
        mock_update_callback.callback_query = MagicMock()
        mock_update_callback.message = None
        self.assertTrue(TelegramDispatcher._is_high_priority_update(mock_update_callback))

        mock_update_cancel = MagicMock()
        mock_update_cancel.callback_query = None
        mock_msg = MagicMock()
        mock_msg.text = "/cancel"
        mock_update_cancel.message = mock_msg
        self.assertTrue(TelegramDispatcher._is_high_priority_update(mock_update_cancel))

        mock_update_text = MagicMock()
        mock_update_text.callback_query = None
        mock_msg_text = MagicMock()
        mock_msg_text.text = "Hello please convert to audio"
        mock_update_text.message = mock_msg_text
        self.assertFalse(TelegramDispatcher._is_high_priority_update(mock_update_text))

    def test_workload_capacity_scaled(self) -> None:
        limiter = get_telegram_workload_limiter()
        # Capacity for OCR, transcribe, audio must be at least 8
        self.assertGreaterEqual(limiter._capacity("ocr"), 8)
        self.assertGreaterEqual(limiter._capacity("transcribe"), 8)
        self.assertGreaterEqual(limiter._capacity("audio"), 8)
        self.assertGreaterEqual(limiter.queue_timeout_s(), 20.0)

    def test_telemetry_scaling_snapshot(self) -> None:
        snapshot = get_scaling_telemetry_snapshot()
        self.assertIn("dispatcher", snapshot)
        self.assertIn("tasks", snapshot)
        self.assertIn("batcher", snapshot)
        self.assertIn("workloads", snapshot)
