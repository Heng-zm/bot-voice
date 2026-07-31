from __future__ import annotations

import unittest

from app.services.ai.language import (
    _detect_lang,
)


class LanguageDetectionTests(unittest.TestCase):
    def test_basic_script_detection(self) -> None:
        self.assertEqual("en", _detect_lang("Hello from Telegram"))
        self.assertEqual("km", _detect_lang("\u179f\u17bd\u179f\u17d2\u178f\u17b8"))
        self.assertEqual("zh", _detect_lang("你好，欢迎使用"))

    def test_manual_prefix_does_not_override_detected_script(self) -> None:
        self.assertEqual("km", _detect_lang("en: សួស្តី"))


if __name__ == "__main__":
    unittest.main()
