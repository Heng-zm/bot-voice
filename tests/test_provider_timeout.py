from __future__ import annotations

import time
import unittest

from app.services.ai.providers import NoProviderAvailable, ProviderManager


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


if __name__ == "__main__":
    unittest.main()
