"""Unit tests for the Channel Auto-Voice Narrator module."""

import unittest

from app.services.telegram.channel import (
    _is_channel_allowed,
    clean_channel_text,
    is_narration_opted_out,
)


class TestChannelNarrator(unittest.TestCase):
    def test_clean_channel_text_strips_urls(self):
        sample = (
            "ព័ត៌មានទាន់ហេតុការណ៍៖ ក្រសួងអប់រំយុវជននិងកីឡាបានប្រកាសលទ្ធផលប្រឡង។\n"
            "ព័ត៌មានលម្អិតសូមចូលទៅកាន់៖ https://t.me/channel/12345 ឬ www.moeys.gov.kh"
        )
        cleaned = clean_channel_text(sample)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("www.moeys.gov.kh", cleaned)
        self.assertIn("ព័ត៌មានទាន់ហេតុការណ៍", cleaned)

    def test_clean_channel_text_strips_delimiters_and_trailing_tags(self):
        sample = (
            "➖➖➖➖➖➖➖➖➖➖\n"
            "សេចក្តីជូនដំណឹងស្តីពីស្ថានភាពធាតុអាកាសនៅកម្ពុជាសម្រាប់សប្តាហ៍នេះ។\n"
            "-------------------\n"
            "#Cambodia #Weather #News @cambodia_daily"
        )
        cleaned = clean_channel_text(sample)
        self.assertNotIn("➖➖➖", cleaned)
        self.assertNotIn("---", cleaned)
        self.assertNotIn("#Cambodia", cleaned)
        self.assertNotIn("@cambodia_daily", cleaned)
        self.assertIn("សេចក្តីជូនដំណឹងស្តីពីស្ថានភាពធាតុអាកាស", cleaned)

    def test_clean_channel_text_rejects_empty_or_short(self):
        self.assertEqual(clean_channel_text(""), "")
        self.assertEqual(clean_channel_text("   "), "")
        self.assertEqual(clean_channel_text("Hi"), "")
        self.assertEqual(clean_channel_text("https://google.com"), "")
        self.assertEqual(clean_channel_text("#news #today"), "")

    def test_clean_channel_text_truncates_at_sentence(self):
        base_sentence = "កម្ពុជាជាប្រទេសដែលមានវប្បធម៌សម្បូរបែប។ "
        long_text = base_sentence * 80  # ~3200 chars
        cleaned = clean_channel_text(long_text, max_chars=500)
        self.assertLessEqual(len(cleaned), 505)
        self.assertTrue(cleaned.endswith("…"))
        # Verify it truncated at a clean Khmer full stop
        self.assertTrue("។" in cleaned)

    def test_clean_channel_text_handles_media_caption(self):
        caption = "រូបភាពស្តីពីសកម្មភាពចុះជួយប្រជាពលរដ្ឋរងគ្រោះដោយទឹកជំនន់នៅខេត្តកំពង់ធំ។ ➖➖➖ #Aid #Flood"
        cleaned = clean_channel_text(caption)
        self.assertIn("រូបភាពស្តីពីសកម្មភាពចុះជួយប្រជាពលរដ្ឋ", cleaned)
        self.assertNotIn("➖➖➖", cleaned)
        self.assertNotIn("#Aid", cleaned)

    def test_is_narration_opted_out(self):
        self.assertTrue(is_narration_opted_out("Announcement for tomorrow #notts"))
        self.assertTrue(is_narration_opted_out("Private test #NoVoice please"))
        self.assertTrue(is_narration_opted_out("Silent update #silent"))
        self.assertTrue(is_narration_opted_out("No narration here #nonarrate"))
        self.assertFalse(is_narration_opted_out("Standard daily news post with regular text"))
        self.assertFalse(is_narration_opted_out("News update #breaking #khmer"))
        self.assertFalse(is_narration_opted_out(""))

    def test_is_channel_allowed(self):
        import os
        # When ALLOWED_CHANNEL_IDS is empty, all channels are allowed
        old_env = os.environ.get("ALLOWED_CHANNEL_IDS")
        try:
            os.environ["ALLOWED_CHANNEL_IDS"] = ""
            self.assertTrue(_is_channel_allowed(-100123456789, "my_channel"))

            # When restricted, only whitelisted channels match
            os.environ["ALLOWED_CHANNEL_IDS"] = "-100111222333, @allowed_channel"
            self.assertTrue(_is_channel_allowed(-100111222333, "random"))
            self.assertTrue(_is_channel_allowed(-100999888777, "allowed_channel"))
            self.assertFalse(_is_channel_allowed(-100999888777, "blocked_channel"))
        finally:
            if old_env is not None:
                os.environ["ALLOWED_CHANNEL_IDS"] = old_env
            else:
                os.environ.pop("ALLOWED_CHANNEL_IDS", None)

    def test_is_channel_allowed_with_bot_settings(self):
        import sys
        from types import ModuleType

        fake_legacy = ModuleType("app.legacy")
        fake_legacy.bot_setting_raw_cached = lambda key, default="": "-100555666, @khmer_news" if key == "allowed_channel_ids" else default

        old_legacy = sys.modules.get("app.legacy")
        try:
            sys.modules["app.legacy"] = fake_legacy
            self.assertTrue(_is_channel_allowed(-100555666, None))
            self.assertTrue(_is_channel_allowed(-100999999, "khmer_news"))
            self.assertFalse(_is_channel_allowed(-100999999, "other_channel"))
        finally:
            if old_legacy is not None:
                sys.modules["app.legacy"] = old_legacy
            else:
                sys.modules.pop("app.legacy", None)

    def test_clean_channel_text_truncates_at_latin_sentence(self):
        base_sentence = "This is a breaking news report from the capital city. "
        long_text = base_sentence * 50
        cleaned = clean_channel_text(long_text, max_chars=400)
        self.assertLessEqual(len(cleaned), 405)
        self.assertTrue(cleaned.endswith("…"))
        self.assertTrue("." in cleaned)

    def test_channel_narrator_buttons_disabled_by_default(self):
        import os

        from app.core.config import SETTINGS
        self.assertFalse(SETTINGS.CHANNEL_NARRATOR_SHOW_BUTTONS)

        old_val = os.environ.get("CHANNEL_NARRATOR_SHOW_BUTTONS")
        try:
            os.environ["CHANNEL_NARRATOR_SHOW_BUTTONS"] = "false"
            show_buttons = os.environ.get("CHANNEL_NARRATOR_SHOW_BUTTONS", "false").lower() in ("1", "true", "yes")
            self.assertFalse(show_buttons)

            os.environ["CHANNEL_NARRATOR_SHOW_BUTTONS"] = "true"
            show_buttons = os.environ.get("CHANNEL_NARRATOR_SHOW_BUTTONS", "false").lower() in ("1", "true", "yes")
            self.assertTrue(show_buttons)
        finally:
            if old_val is not None:
                os.environ["CHANNEL_NARRATOR_SHOW_BUTTONS"] = old_val
            else:
                os.environ.pop("CHANNEL_NARRATOR_SHOW_BUTTONS", None)


if __name__ == "__main__":
    unittest.main()
