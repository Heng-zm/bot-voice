from __future__ import annotations

import base64
import time
import unittest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient

    from app import legacy
    from app.main import app
    HAS_SERVER_DEPS = True
except (ImportError, ModuleNotFoundError):
    TestClient = None  # type: ignore[assignment,misc]
    legacy = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]
    HAS_SERVER_DEPS = False


@unittest.skipUnless(HAS_SERVER_DEPS, "Requires full server dependencies")
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


@unittest.skipUnless(HAS_SERVER_DEPS, "Requires full server dependencies")
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


@unittest.skipUnless(HAS_SERVER_DEPS, "Requires full server dependencies")
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


@unittest.skipUnless(HAS_SERVER_DEPS, "Requires full server dependencies")
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


@unittest.skipUnless(HAS_SERVER_DEPS, "Requires full server dependencies")
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


@unittest.skipUnless(HAS_SERVER_DEPS, "Requires full server dependencies")
class TestBotConfigPanel(unittest.TestCase):
    def test_mask_secret(self):
        self.assertEqual("Not configured", legacy._mask_secret(""))
        self.assertEqual("Not configured", legacy._mask_secret(None))
        self.assertEqual("******", legacy._mask_secret("123"))
        self.assertEqual("123******cba", legacy._mask_secret("1234567890cba"))

    def test_bot_setting_defaults_contains_all_domains(self):
        # AI & Speech
        self.assertIn("DEFAULT_TTS_MODEL", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("DEFAULT_GENDER", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("DEFAULT_SPEED", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("GEMINI_MODEL", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("OCR_PROVIDER", legacy.BOT_SETTING_DEFAULTS)
        # Channel Narrator
        self.assertIn("channel_narrator_enabled", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("channel_narrator_gender", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("channel_narrator_speed", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("channel_narrator_model", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("channel_narrator_max_chars", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("channel_narrator_show_buttons", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("allowed_channel_ids", legacy.BOT_SETTING_DEFAULTS)
        # Feature toggles & system
        self.assertIn("voice_reply_mode", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("anti_spam_window", legacy.BOT_SETTING_DEFAULTS)
        self.assertIn("maintenance_mode", legacy.BOT_SETTING_DEFAULTS)

    def test_cached_setting_lookup(self):
        legacy._bot_settings_cache["test_key_xyz"] = "hello_world"
        self.assertEqual("hello_world", legacy.bot_setting_raw_cached("test_key_xyz"))
        self.assertEqual("default_fallback", legacy.bot_setting_raw_cached("non_existent_key_abc", "default_fallback"))

    def test_keyboards_generation(self):
        # Home keyboard
        home_kb = legacy.get_bot_config_home_kb()
        self.assertIsNotNone(home_kb)
        flattened_home = [btn.callback_data for row in home_kb.inline_keyboard for btn in row]
        self.assertIn("cfg_cat:ai", flattened_home)
        self.assertIn("cfg_cat:channel", flattened_home)
        self.assertIn("cfg_cat:features", flattened_home)
        self.assertIn("cfg_cat:performance", flattened_home)
        self.assertIn("cfg_cat:all", flattened_home)

        # AI keyboard
        ai_kb = legacy.get_bot_config_ai_kb({"DEFAULT_TTS_MODEL": "auto", "DEFAULT_GENDER": "female", "DEFAULT_SPEED": 1.0})
        self.assertIsNotNone(ai_kb)
        flattened_ai = [btn.callback_data for row in ai_kb.inline_keyboard for btn in row]
        self.assertIn("cfg_set:DEFAULT_TTS_MODEL:auto", flattened_ai)
        self.assertIn("cfg_set:DEFAULT_GENDER:female", flattened_ai)

        # Channel keyboard
        ch_kb = legacy.get_bot_config_channel_kb({"channel_narrator_enabled": True, "channel_narrator_gender": "female"})
        self.assertIsNotNone(ch_kb)
        flattened_ch = [btn.callback_data for row in ch_kb.inline_keyboard for btn in row]
        self.assertIn("cfg_set:channel_narrator_enabled:toggle", flattened_ch)
        self.assertIn("cfg_set:channel_narrator_gender:female", flattened_ch)

        # All config keyboard
        all_kb = legacy.get_bot_config_all_kb()
        self.assertIsNotNone(all_kb)
        flattened_all = [btn.callback_data for row in all_kb.inline_keyboard for btn in row]
        self.assertIn("cfg_cat:home", flattened_all)

    def test_admin_config_text_renderers(self):
        settings = dict(legacy.BOT_SETTING_DEFAULTS)
        status = {"status": "ok", "uptime_seconds": 120, "version": "4.2.0"}

        home_text = legacy._admin_bot_config_home_text(settings, status)
        self.assertIn("Bot Configuration Hub", home_text)
        self.assertIn("Speech & AI Models", home_text)
        self.assertIn("Channel Narrator", home_text)

        ai_text = legacy._admin_bot_config_ai_text(settings)
        self.assertIn("Speech & AI Models", ai_text)

        channel_text = legacy._admin_bot_config_channel_text(settings)
        self.assertIn("Channel Auto-Voice Narrator", channel_text)

        all_text = legacy._admin_bot_config_all_text(settings, status)
        self.assertIn("Complete Configuration Dump", all_text)
        self.assertIn("Environment & Secrets", all_text)

    def test_admin_dashboard_cleaned_layout(self):
        kb = legacy.get_admin_dashboard_kb()
        self.assertIsNotNone(kb)
        # Verify exactly 6 clean rows
        self.assertEqual(6, len(kb.inline_keyboard))
        # Total buttons is 12 (2 per row)
        flattened = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertEqual(12, len(flattened))
        self.assertEqual(
            [
                "admin_bot_config", "admin_broadcast",
                "admin_users", "admin_schedules",
                "admin_user_needs", "admin_report",
                "admin_health", "admin_errors",
                "admin_optimize", "admin_stats",
                "admin_home", "admin_close",
            ],
            flattened,
        )

    def test_admin_health_and_stats_kbs(self):
        health_kb = legacy.get_admin_health_kb()
        stats_kb = legacy.get_admin_stats_kb()

        flat_health = [btn.callback_data for row in health_kb.inline_keyboard for btn in row]
        self.assertIn("admin_health", flat_health)
        self.assertIn("admin_home", flat_health)

        flat_stats = [btn.callback_data for row in stats_kb.inline_keyboard for btn in row]
        self.assertIn("admin_report", flat_stats)
        self.assertIn("admin_home", flat_stats)

    def test_admin_home_text_clean_format(self):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            text = loop.run_until_complete(legacy._admin_home_text(0))
            self.assertIn("Admin Control Center", text)
            self.assertIn("AI & Speech Stack", text)
            self.assertIn("Audience & Activity", text)
            self.assertNotIn("2.5-flash", text)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()


