from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from app.services.telegram.workloads import TelegramWorkloadLimiter, WorkloadBusy


class TelegramWorkloadLimiterTests(unittest.IsolatedAsyncioTestCase):
    def test_default_snapshot_is_bounded(self) -> None:
        limiter = TelegramWorkloadLimiter()
        snapshot = limiter.snapshot()
        self.assertGreaterEqual(snapshot["ocr"]["capacity"], 1)
        self.assertGreaterEqual(snapshot["transcribe"]["capacity"], 1)
        self.assertGreaterEqual(snapshot["audio"]["capacity"], 1)

    async def test_busy_workload_is_rejected_after_queue_timeout(self) -> None:
        limiter = TelegramWorkloadLimiter()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold_slot() -> None:
            async with limiter.slot("ocr"):
                entered.set()
                await release.wait()

        with patch.dict(
            "os.environ",
            {
                "TELEGRAM_OCR_MAX_CONCURRENT": "1",
                "TELEGRAM_WORKLOAD_QUEUE_TIMEOUT_S": "0.1",
            },
        ):
            task = asyncio.create_task(hold_slot())
            await entered.wait()
            with self.assertRaises(WorkloadBusy):
                async with limiter.slot("ocr"):
                    pass
            release.set()
            await task

            snapshot = limiter.snapshot()["ocr"]
            self.assertEqual(1, snapshot["accepted"])
            self.assertEqual(1, snapshot["rejected"])
            self.assertEqual(0, snapshot["in_use"])
            self.assertEqual(0, snapshot["waiting"])

    async def test_capacity_change_applies_when_bucket_is_idle(self) -> None:
        limiter = TelegramWorkloadLimiter()
        with patch.dict("os.environ", {"TELEGRAM_AUDIO_MAX_CONCURRENT": "1"}):
            async with limiter.slot("audio"):
                self.assertEqual(1, limiter.snapshot()["audio"]["capacity"])
        with patch.dict("os.environ", {"TELEGRAM_AUDIO_MAX_CONCURRENT": "3"}):
            async with limiter.slot("audio"):
                self.assertEqual(3, limiter.snapshot()["audio"]["capacity"])


if __name__ == "__main__":
    unittest.main()
