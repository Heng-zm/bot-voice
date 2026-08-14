from __future__ import annotations

import unittest

from app.core.resources import resource_default, resource_profile, resource_value


class ResourceProfileTests(unittest.TestCase):
    def test_efficient_is_the_safe_default(self) -> None:
        env: dict[str, str] = {}

        self.assertEqual(resource_profile(env), "efficient")
        self.assertEqual(resource_default("HTTP_MAX_CONNECTIONS", 100, env), 50)
        self.assertEqual(resource_default("TTS_AUDIO_CACHE_MAX_MB", 64, env), 32)

    def test_efficient_profile_caps_persisted_large_values(self) -> None:
        env = {"BOT_RESOURCE_PROFILE": "efficient"}

        self.assertEqual(resource_value("HTTP_MAX_CONNECTIONS", 180, env), 50)
        self.assertEqual(resource_value("DB_EXECUTOR_MAX_WORKERS", 10, env), 3)
        self.assertEqual(resource_value("MAX_CONCURRENT_AI", 20, env), 2)
        self.assertEqual(resource_value("MAX_CONCURRENT_TTS_USERS", 20, env), 2)

    def test_cleanup_interval_can_be_increased_to_reduce_background_work(self) -> None:
        env = {"BOT_RESOURCE_PROFILE": "efficient"}

        self.assertEqual(
            resource_value("BOT_ARTIFACT_CLEANUP_SECONDS", 3_600, env), 3_600
        )

    def test_balanced_profile_preserves_configured_values(self) -> None:
        env = {"BOT_RESOURCE_PROFILE": "balanced"}

        self.assertEqual(resource_default("HTTP_MAX_CONNECTIONS", 100, env), 100)
        self.assertEqual(resource_value("HTTP_MAX_CONNECTIONS", 180, env), 180)

    def test_aliases_and_unknown_values_are_safe(self) -> None:
        self.assertEqual(
            resource_profile({"BOT_RESOURCE_PROFILE": "small"}), "efficient"
        )
        self.assertEqual(
            resource_profile({"BOT_RESOURCE_PROFILE": "fast"}), "performance"
        )
        self.assertEqual(
            resource_profile({"BOT_RESOURCE_PROFILE": "typo"}), "efficient"
        )
