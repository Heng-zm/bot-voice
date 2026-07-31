from __future__ import annotations

import unittest

from app.services.ai.language import (
    _detect_lang,
    _extract_leading_lang_hint,
    _normalise_lang_hint,
)


class LanguageDetectionTests(unittest.TestCase):
    def test_explicit_language_hint_is_extracted(self) -> None:
        hint, text = _extract_leading_lang_hint("en: hello world")
        self.assertEqual("en", hint)
        self.assertEqual("hello world", text)

    def test_language_hint_is_normalized(self) -> None:
        self.assertEqual("en", _normalise_lang_hint("English"))
        self.assertEqual("", _normalise_lang_hint("not-a-language"))

    def test_basic_script_detection(self) -> None:
        self.assertEqual("en", _detect_lang("Hello from Telegram"))
        self.assertEqual("km", _detect_lang("\u179f\u17bd\u179f\u17d2\u178f\u17b8"))


if __name__ == "__main__":
    unittest.main()
