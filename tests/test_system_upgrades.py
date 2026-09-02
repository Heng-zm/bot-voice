from __future__ import annotations

import base64
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app import legacy
from app.main import app


class TestRedisAudioCache(unittest.TestCase):
    def setUp(self):
        legacy._TTS_AUDIO_CACHE.clear()
        legacy._TTS_AUDIO_CACHE_BYTES = 0

    def test_audio_cache_key_generation(self):
        key1 = legacy._tts_audio_cache_key("សួស្តី", "female", 1.0, "auto")
        key2 = legacy._tts_audio_cache_key("សួស្តី", "female", 1.0, "auto")
        key3 = legacy._tts_audio_cache_key("Hello", "male", 1.25, "edge")
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertEqual(64, len(key1)) # SHA256 hex length

    def test_l1_l2_cache_lifecycle(self):
        fake_audio = b"OggS\x00\x02\x00\x00\x00\x00FAKEAUDIOBYTES"
        cache_key = "test_cache_key_123"

        # Initially empty
        self.assertIsNone(legacy._tts_audio_cache_get(cache_key))

        # Store in cache
        legacy._tts_audio_cache_set(cache_key, fake_audio)

        # L1 Memory lookup
        cached = legacy._tts_audio_cache_get(cache_key)
        self.assertEqual(fake_audio, cached)

        # Mock Redis get when L1 is evicted
        legacy._TTS_AUDIO_CACHE.clear()
        
        mock_redis = MagicMock()
        b64_encoded = base64.b64encode(fake_audio).decode("ascii")
        mock_redis.get.return_value = b64_encoded

        with patch.object(legacy, "redis_client", mock_redis):
            redis_cached = legacy._tts_audio_cache_get(cache_key)
            self.assertEqual(fake_audio, redis_cached)
            # Should have re-populated L1
            self.assertIn(cache_key, legacy._TTS_AUDIO_CACHE)


class TestSystemMetricsSnapshot(unittest.TestCase):
    def test_metrics_snapshot_structure(self):
        snapshot = legacy._system_metrics_snapshot()
        
        self.assertTrue(snapshot.get("ok"))
        self.assertIn("status", snapshot)
        self.assertIn("version", snapshot)
        self.assertIn("uptime_seconds", snapshot)
        self.assertIn("storage", snapshot)
        self.assertIn("tts_audio_cache", snapshot)
        self.assertIn("anti_spam", snapshot)
        self.assertIn("providers", snapshot)

        # Validate storage structure
        storage = snapshot["storage"]
        self.assertIn("redis_connected", storage)
        self.assertIn("supabase_connected", storage)


class TestDatabasePruning(unittest.TestCase):
    def test_pruning_without_supabase_returns_gracefully(self):
        with patch.object(legacy, "supabase", None):
            res = legacy.db_run_periodic_pruning()
            self.assertFalse(res["ok"])
            self.assertEqual("no_supabase", res["reason"])

    def test_pruning_with_mock_supabase(self):
        mock_sb = MagicMock()
        mock_table = MagicMock()
        mock_delete = MagicMock()
        mock_lt = MagicMock()
        
        mock_sb.table.return_value = mock_table
        mock_table.delete.return_value = mock_delete
        mock_delete.lt.return_value = mock_lt
        mock_lt.execute.return_value = MagicMock(data=[{"id": 1}, {"id": 2}])

        with patch.object(legacy, "supabase", mock_sb):
            res = legacy.db_run_periodic_pruning()
            self.assertTrue(res["ok"])
            self.assertIn("pruned_history", res)
            self.assertIn("pruned_text_cache", res)


class TestFastAPISystemEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthz_endpoint(self):
        resp = self.client.get("/healthz")
        self.assertEqual(200, resp.status_code)
        data = resp.json()
        self.assertEqual("ok", data["status"])

    def test_system_metrics_endpoint(self):
        resp = self.client.get("/system")
        self.assertEqual(200, resp.status_code)
        data = resp.json()
        self.assertTrue(data.get("ok"))
        self.assertIn("uptime_seconds", data)

    def test_api_metrics_alias_endpoint(self):
        resp = self.client.get("/api/metrics")
        self.assertEqual(200, resp.status_code)
        data = resp.json()
        self.assertTrue(data.get("ok"))


class TestAntiSpamAndUnlock(unittest.TestCase):
    def setUp(self):
        legacy._tts_request_reservations.clear()

    def test_tts_reservation_auto_expiry(self):
        user_id = 998877
        
        # 1. First reservation succeeds
        self.assertTrue(legacy._reserve_tts_request(user_id))
        
        # 2. Immediate second reservation fails (already reserved)
        self.assertFalse(legacy._reserve_tts_request(user_id))
        
        # 3. Simulate passage of time (> 120s TTL)
        legacy._tts_request_reservations[user_id] = time.time() - 150.0
        
        # 4. Now reservation should auto-expire and succeed!
        self.assertTrue(legacy._reserve_tts_request(user_id))

    def test_tts_release_request(self):
        user_id = 112233
        self.assertTrue(legacy._reserve_tts_request(user_id))
        legacy._release_tts_request(user_id)
        # Should be able to reserve again immediately
        self.assertTrue(legacy._reserve_tts_request(user_id))


if __name__ == "__main__":
    unittest.main()
