"""Unit tests for app.core modules."""

from __future__ import annotations

import unittest
import time

from app.core.security.auth import is_admin, get_admin_ids
from app.core.security.rate_limit import check_rate_limit, is_rate_limited
from app.core.telemetry.ring_buffer import TelemetryRingBuffer
from app.core.concurrency.locks import get_keyed_lock


class CoreModulesUnitTests(unittest.TestCase):
    def test_auth(self):
        admin_ids = get_admin_ids()
        self.assertIsInstance(admin_ids, (set, list, tuple))

    def test_rate_limiting(self):
        # Fresh key should not be rate limited
        key = f"test_key_{time.time()}"
        self.assertFalse(is_rate_limited(key, max_requests=10, window_seconds=60))

    def test_telemetry_ring_buffer(self):
        buf = TelemetryRingBuffer(capacity=10)
        buf.append({"message": "test 1"})
        buf.append({"message": "test 2"})
        self.assertEqual(len(buf.get_entries()), 2)

    def test_keyed_lock(self):
        lock1 = get_keyed_lock("user_1")
        lock2 = get_keyed_lock("user_1")
        self.assertIs(lock1, lock2)


if __name__ == "__main__":
    unittest.main()
