from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.ai.ocr import (
    detect_image_mime,
    normalize_media_suffix,
    normalize_ocr_provider,
    normalize_ocr_result,
)
from app.services.ai.providers import get_provider_manager
from app.services.ai.tts import clean_tts_text, normalize_tts_request
from app.services.jobs.runtime import BOT_JOB_TYPES
from app.services.settings.runtime import (
    coerce_runtime_updates,
    coerce_runtime_value,
)
from app.services.telegram.broadcast import BroadcastRequest


class ExtractedTTSServiceTests(unittest.TestCase):
    def test_tts_request_is_validated_without_legacy_state(self) -> None:
        request = normalize_tts_request(
            {"text": " hello\u200b ", "gender": "MALE", "speed": 9, "tts_model": "edge"}
        )
        self.assertEqual("hello\u200b", request.text)
        self.assertEqual("male", request.gender)
        self.assertEqual(2.0, request.speed)
        self.assertEqual("edge", request.model)
        self.assertEqual("hello world", clean_tts_text("hello\u200b\x00   world"))

    def test_removed_tts_model_falls_back_to_auto(self) -> None:
        request = normalize_tts_request({"text": "hello", "tts_model": "voxcpm2"})
        self.assertEqual("auto", request.model)

    def test_removed_provider_and_job_type_are_not_registered(self) -> None:
        self.assertNotIn("voxcpm2", BOT_JOB_TYPES)
        self.assertNotIn("voxcpm2", get_provider_manager().snapshot())


class ExtractedOCRServiceTests(unittest.TestCase):
    def test_media_helpers_are_framework_independent(self) -> None:
        self.assertEqual(".jpg", normalize_media_suffix("../bad", default=".jpg"))
        self.assertEqual("nothing", normalize_ocr_result("NOTEXT", no_text_message="nothing"))
        self.assertEqual("hf", normalize_ocr_provider("huggingface"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.bin"
            path.write_bytes(b"\x89PNG\r\n\x1a\nrest")
            self.assertEqual("image/png", detect_image_mime(str(path)))


class ExtractedBroadcastServiceTests(unittest.TestCase):
    def test_broadcast_request_deduplicates_recipients(self) -> None:
        request = BroadcastRequest.from_payload(
            {
                "recipient_ids": [1, "1", 2, 0],
                "text": "hello",
                "concurrency": 99,
            }
        )
        self.assertEqual((1, 2), request.recipients)
        self.assertEqual(10, request.concurrency)


class ExtractedRuntimeSettingsTests(unittest.TestCase):
    def test_runtime_values_are_bounded_and_errors_name_the_key(self) -> None:
        specs = {"WORKERS": {"kind": "int", "min": 1, "max": 8}}
        self.assertEqual(8, coerce_runtime_value("WORKERS", "20", specs["WORKERS"]))
        with self.assertRaisesRegex(ValueError, "WORKERS"):
            coerce_runtime_updates({"WORKERS": "invalid"}, specs)

    def test_runtime_urls_and_non_finite_numbers_are_rejected(self) -> None:
        url_spec = {"kind": "url"}
        self.assertEqual(
            "https://example.com/base",
            coerce_runtime_value(
                "TELEGRAM_WEBHOOK_URL",
                "https://example.com/base/",
                url_spec,
            ),
        )
        for invalid in (
            "http://example.com",
            "javascript:alert(1)",
            "https://user:password@example.com",
            "https://example.com/tg-webhook-secret",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                coerce_runtime_value("TELEGRAM_WEBHOOK_URL", invalid, url_spec)
        with self.assertRaisesRegex(ValueError, "finite"):
            coerce_runtime_value(
                "RATE_WINDOW",
                "NaN",
                {"kind": "float", "min": 0.25, "max": 60.0},
            )


if __name__ == "__main__":
    unittest.main()
