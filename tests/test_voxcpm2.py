from __future__ import annotations

import time
import unittest
from unittest.mock import AsyncMock, patch

from app import legacy


class VoxCPM2ProfileTests(unittest.TestCase):
    def test_legacy_profile_defaults_to_controllable_mode(self) -> None:
        profile = legacy._voxcpm2_normalize_profile(
            {
                "file_id": "telegram-file-id",
                "control": "Warm and calm",
            }
        )

        self.assertEqual(profile["mode"], legacy.VOXCPM2_MODE_CONTROLLABLE)
        self.assertEqual(profile["control"], "Warm and calm")
        self.assertEqual(profile["prompt_text"], "")

    def test_ultimate_mode_requires_reference_and_prompt_text(self) -> None:
        missing_both = {
            "mode": legacy.VOXCPM2_MODE_ULTIMATE,
            "file_id": "",
            "prompt_text": "",
        }
        missing_prompt = {
            "mode": legacy.VOXCPM2_MODE_ULTIMATE,
            "file_id": "telegram-file-id",
            "prompt_text": "",
        }
        ready = {
            "mode": legacy.VOXCPM2_MODE_ULTIMATE,
            "file_id": "telegram-file-id",
            "prompt_text": "Exact words spoken in the reference.",
        }

        self.assertEqual(
            legacy._voxcpm2_profile_missing(missing_both),
            ["reference", "prompt_text"],
        )
        self.assertEqual(
            legacy._voxcpm2_profile_missing(missing_prompt),
            ["prompt_text"],
        )
        self.assertFalse(legacy._voxcpm2_profile_ready(missing_prompt))
        self.assertTrue(legacy._voxcpm2_profile_ready(ready))

    def test_setup_panel_switches_actions_with_clone_mode(self) -> None:
        controllable = legacy._voxcpm2_panel_kb(
            {"mode": legacy.VOXCPM2_MODE_CONTROLLABLE}
        )
        ultimate = legacy._voxcpm2_panel_kb(
            {"mode": legacy.VOXCPM2_MODE_ULTIMATE}
        )

        self.assertEqual(
            controllable.inline_keyboard[2][0].callback_data,
            "voxcpm2:set_control",
        )
        self.assertEqual(
            ultimate.inline_keyboard[2][0].callback_data,
            "voxcpm2:set_prompt",
        )


class VoxCPM2ApiTests(unittest.TestCase):
    def test_controllable_mode_uses_current_nine_input_api(self) -> None:
        reference = object()

        inputs = legacy._voxcpm2_api_inputs(
            "Text to synthesize",
            "A warm, measured delivery",
            reference,
            legacy.VOXCPM2_MODE_CONTROLLABLE,
            "",
        )

        self.assertEqual(len(inputs), 9)
        self.assertEqual(inputs[0], "Text to synthesize")
        self.assertEqual(inputs[1], "A warm, measured delivery")
        self.assertIs(inputs[2], reference)
        self.assertFalse(inputs[3])
        self.assertEqual(inputs[4], "")
        self.assertEqual(inputs[5], legacy.VOXCPM2_CFG_VALUE)
        self.assertEqual(inputs[6], legacy.VOXCPM2_NORMALIZE_TEXT)
        self.assertEqual(inputs[7], legacy.VOXCPM2_DENOISE_REFERENCE)
        self.assertEqual(inputs[8], legacy.VOXCPM2_INFERENCE_TIMESTEPS)

    def test_ultimate_mode_uses_transcript_and_ignores_style_control(self) -> None:
        reference = object()

        inputs = legacy._voxcpm2_api_inputs(
            "Text to synthesize",
            "This must not be sent in ultimate mode",
            reference,
            legacy.VOXCPM2_MODE_ULTIMATE,
            "Exact reference transcript",
        )

        self.assertEqual(len(inputs), 9)
        self.assertEqual(inputs[1], "")
        self.assertIs(inputs[2], reference)
        self.assertTrue(inputs[3])
        self.assertEqual(inputs[4], "Exact reference transcript")

    def test_ultimate_mode_rejects_missing_transcript(self) -> None:
        with self.assertRaisesRegex(ValueError, "transcript"):
            legacy._voxcpm2_api_inputs(
                "Text to synthesize",
                "",
                object(),
                legacy.VOXCPM2_MODE_ULTIMATE,
                "",
            )

    def test_provider_context_changes_with_clone_mode_and_transcript(self) -> None:
        base_profile = {
            "file_id": "telegram-file-id",
            "file_unique_id": "unique-id",
            "mode": legacy.VOXCPM2_MODE_CONTROLLABLE,
            "control": "Warm",
            "prompt_text": "",
        }
        ultimate_profile = {
            **base_profile,
            "mode": legacy.VOXCPM2_MODE_ULTIMATE,
            "prompt_text": "Reference transcript",
        }

        controllable_context = legacy._voxcpm2_provider_context(base_profile)
        ultimate_context = legacy._voxcpm2_provider_context(ultimate_profile)

        self.assertNotEqual(controllable_context, ultimate_context)
        with patch.object(
            legacy,
            "VOXCPM2_INFERENCE_TIMESTEPS",
            legacy.VOXCPM2_INFERENCE_TIMESTEPS + 1,
        ):
            changed_steps_context = legacy._voxcpm2_provider_context(ultimate_profile)
        self.assertNotEqual(ultimate_context, changed_steps_context)


class VoxCPM2ProfileCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        legacy._VOXCPM2_PROFILE_MEMORY.clear()

    def tearDown(self) -> None:
        legacy._VOXCPM2_PROFILE_MEMORY.clear()

    def test_memory_profile_does_not_expire_when_redis_is_unavailable(self) -> None:
        user_id = 771
        profile = {"file_id": "telegram-file-id"}
        stale_timestamp = time.monotonic() - legacy.VOXCPM2_PROFILE_MEMORY_TTL_S - 10
        legacy._VOXCPM2_PROFILE_MEMORY[user_id] = (profile, stale_timestamp)

        with patch.object(legacy, "redis_client", None):
            self.assertEqual(
                legacy._voxcpm2_profile_memory_get(user_id),
                profile,
            )

    def test_memory_profile_expires_when_redis_is_available(self) -> None:
        user_id = 772
        profile = {"file_id": "telegram-file-id"}
        stale_timestamp = time.monotonic() - legacy.VOXCPM2_PROFILE_MEMORY_TTL_S - 10
        legacy._VOXCPM2_PROFILE_MEMORY[user_id] = (profile, stale_timestamp)

        with patch.object(legacy, "redis_client", object()):
            self.assertIsNone(legacy._voxcpm2_profile_memory_get(user_id))
        self.assertNotIn(user_id, legacy._VOXCPM2_PROFILE_MEMORY)


class VoxCPM2CircuitBreakerTests(unittest.TestCase):
    def test_admin_reset_clears_failures_and_cooldown(self) -> None:
        with legacy._VOXCPM2_STATE_LOCK:
            original_failures = legacy._VOXCPM2_FAILURES
            original_disabled_until = legacy._VOXCPM2_DISABLED_UNTIL
            legacy._VOXCPM2_FAILURES = 3
            legacy._VOXCPM2_DISABLED_UNTIL = time.monotonic() + 300
        try:
            legacy._reset_voxcpm2_cooldown()

            self.assertEqual(0, legacy._VOXCPM2_FAILURES)
            self.assertEqual(0.0, legacy._VOXCPM2_DISABLED_UNTIL)
        finally:
            with legacy._VOXCPM2_STATE_LOCK:
                legacy._VOXCPM2_FAILURES = original_failures
                legacy._VOXCPM2_DISABLED_UNTIL = original_disabled_until


class VoxCPM2SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_carries_ultimate_mode_and_transcript_to_generation(self) -> None:
        profile = {
            "file_id": "telegram-file-id",
            "file_unique_id": "unique-id",
            "filename": "reference.ogg",
            "suffix": ".ogg",
            "duration": 8.0,
            "mode": legacy.VOXCPM2_MODE_ULTIMATE,
            "prompt_text": "Exact reference transcript",
            "control": "Ignored in Ultimate mode",
        }

        with (
            patch.object(
                legacy,
                "_voxcpm2_profile_get",
                AsyncMock(return_value=profile),
            ),
            patch.object(
                legacy,
                "safe_send",
                AsyncMock(return_value=object()),
            ),
            patch.object(
                legacy,
                "_download_telegram_file_to_temp_path",
                AsyncMock(return_value="reference.ogg"),
            ),
            patch.object(
                legacy,
                "_validate_voxcpm2_reference_path",
                AsyncMock(return_value=8.0),
            ),
        ):
            session = await legacy._prepare_voxcpm2_session(123, object())

        self.assertEqual(session["mode"], legacy.VOXCPM2_MODE_ULTIMATE)
        self.assertEqual(session["prompt_text"], "Exact reference transcript")
        self.assertEqual(session["reference_path"], "reference.ogg")


if __name__ == "__main__":
    unittest.main()
