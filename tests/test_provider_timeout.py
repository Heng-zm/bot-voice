from __future__ import annotations

import asyncio
import os
import threading
import time
import unittest
from unittest.mock import patch

from app.services.ai.providers import (
    NoProviderAvailable,
    ProviderManager,
)
from app.utils.env import bounded_env_int


class ProviderTimeoutTests(unittest.TestCase):
    def test_execute_sync_applies_provider_timeout(self) -> None:
        manager = ProviderManager(sync_max_workers=1, sync_max_inflight=1)
        manager.register(
            "slow",
            capabilities={"ai"},
            timeout_seconds=0.05,
            failure_threshold=1,
            cooldown_seconds=10,
        )

        def slow(_provider: str) -> str:
            time.sleep(0.2)
            return "late"

        started = time.perf_counter()
        with self.assertRaises(NoProviderAvailable) as caught:
            manager.execute_sync("ai", slow)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.15)
        self.assertIn("ProviderTimeout", caught.exception.errors["slow"])
        snapshot = manager.snapshot()["slow"]
        self.assertFalse(snapshot["available"])
        self.assertEqual("process", snapshot["scope"])
        manager.close()

    def test_invalid_provider_pool_environment_uses_safe_default(self) -> None:
        with patch.dict(os.environ, {"PROVIDER_SYNC_MAX_WORKERS": "invalid"}):
            value = bounded_env_int(
                "PROVIDER_SYNC_MAX_WORKERS",
                4,
                minimum=1,
                maximum=32,
            )

        self.assertEqual(4, value)


class AsyncProviderBoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_operations_use_bounded_pool_from_async_api(self) -> None:
        manager = ProviderManager(sync_max_workers=1, sync_max_inflight=1)
        manager.register(
            "slow",
            capabilities={"ai"},
            timeout_seconds=0.05,
        )
        started = threading.Event()
        release = threading.Event()

        def slow(_provider: str) -> str:
            started.set()
            release.wait(0.2)
            return "late"

        first = asyncio.create_task(manager.execute("ai", slow))
        await asyncio.to_thread(started.wait, 0.1)
        try:
            with self.assertRaises(NoProviderAvailable) as busy:
                await manager.execute("ai", slow)
            with self.assertRaises(NoProviderAvailable) as timed_out:
                await first
        finally:
            release.set()
            manager.close()

        self.assertIn("ProviderBusy", busy.exception.errors["slow"])
        self.assertIn("ProviderTimeout", timed_out.exception.errors["slow"])


if __name__ == "__main__":
    unittest.main()
