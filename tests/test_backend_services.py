from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.services.ai.language import _detect_lang, _language_display
from app.services.ai.providers import NoProviderAvailable, ProviderManager
from app.services.settings.store import SettingsStore
from app.services.telegram.deduplication import WebhookReplayStore
from app.services.telegram.flow import callback_requires_tts_access, classify_callback


class LanguageDetectionTests(unittest.TestCase):
    def test_detects_khmer(self) -> None:
        self.assertEqual("km", _detect_lang("សួស្តីបងប្អូនទាំងអស់គ្នា"))

    def test_detects_english(self) -> None:
        self.assertEqual("en", _detect_lang("Hello world! How are you?"))

    def test_detects_korean(self) -> None:
        self.assertEqual("ko", _detect_lang("안녕하세요 반갑습니다"))

    def test_detects_japanese(self) -> None:
        self.assertEqual("ja", _detect_lang("こんにちは世界"))
        self.assertEqual("ja", _detect_lang("日本語のテスト"))

    def test_detects_chinese(self) -> None:
        self.assertEqual("zh", _detect_lang("你好世界，今天天气很好"))

    def test_detects_hindi(self) -> None:
        self.assertEqual("hi", _detect_lang("नमस्ते दुनिया"))

    def test_detects_arabic(self) -> None:
        self.assertEqual("ar", _detect_lang("مرحبا بالعالم كيف حالك"))

    def test_detects_indonesian_latin(self) -> None:
        self.assertEqual("id", _detect_lang("saya sedang belajar karena ingin bisa"))

    def test_empty_string_defaults_to_english(self) -> None:
        self.assertEqual("en", _detect_lang(""))

    def test_language_display(self) -> None:
        flag, name = _language_display("km")
        self.assertEqual("🇰🇭", flag)
        self.assertEqual("Khmer", name)


class WebhookReplayStoreTests(unittest.TestCase):
    def test_claim_and_complete_lifecycle(self) -> None:
        store = WebhookReplayStore()
        claim_state, token = store.claim(12345, include_token=True)
        self.assertEqual("claimed", claim_state)
        self.assertIsNotNone(token)

        # Duplicate incoming claim while processing
        dup_state = store.claim(12345)
        self.assertEqual("processing", dup_state)

        # Complete claim
        completed = store.complete(12345, claim_token=token)
        self.assertTrue(completed)

        # Duplicate claim after completion
        done_state = store.claim(12345)
        self.assertEqual("completed", done_state)

    def test_release_processing_lease(self) -> None:
        store = WebhookReplayStore()
        _, token = store.claim(999, include_token=True)
        released = store.release(999, claim_token=token)
        self.assertTrue(released)

        # Can reclaim after release
        reclaim_state = store.claim(999)
        self.assertEqual("claimed", reclaim_state)


class TelegramFlowTests(unittest.TestCase):
    def test_classify_exact_actions(self) -> None:
        self.assertEqual("show_speed", classify_callback("show_speed"))
        self.assertEqual("gender", classify_callback("tg_female"))
        self.assertEqual("welcome_profile", classify_callback("welcome_profile"))

    def test_classify_prefix_actions(self) -> None:
        self.assertEqual("tts_model", classify_callback("ttsmodel_edge"))
        self.assertEqual("delete", classify_callback("doc_del:123"))
        self.assertEqual("doc_read", classify_callback("doc_read:456"))
        self.assertEqual("admin", classify_callback("admin_home"))

    def test_unknown_callback(self) -> None:
        self.assertIsNone(classify_callback("unrecognized_action_123"))

    def test_callback_requires_tts_access(self) -> None:
        self.assertTrue(callback_requires_tts_access("speed", "spd_1.0"))
        self.assertTrue(callback_requires_tts_access("gender", "tg_female"))
        self.assertFalse(callback_requires_tts_access("admin", "admin_home"))


class SettingsStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_read_uses_one_supabase_request(self) -> None:
        class Query:
            def __init__(self) -> None:
                self.execute_calls = 0

            def select(self, _columns: str):
                return self

            def in_(self, _column: str, _keys: list[str]):
                return self

            def execute(self):
                self.execute_calls += 1
                return SimpleNamespace(data=[{"key": "runtime:one", "value": "1"}])

        query = Query()
        store = SettingsStore(SimpleNamespace(table=lambda _name: query))

        values = await store.get_many_text(["runtime:one", "runtime:missing"], "fallback")

        self.assertEqual(1, query.execute_calls)
        self.assertEqual({"runtime:one": "1", "runtime:missing": "fallback"}, values)


class ProviderManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_falls_back_to_next_provider(self) -> None:
        manager = ProviderManager()
        manager.register("slow", capabilities={"tts"}, priority=1, timeout_seconds=0.02)
        manager.register("fast", capabilities={"tts"}, priority=2, timeout_seconds=1)

        async def operation(provider: str) -> str:
            if provider == "slow":
                await asyncio.sleep(0.08)
            return provider

        result, provider = await manager.execute("tts", operation)
        self.assertEqual(("fast", "fast"), (result, provider))

    async def test_exhaustion_reports_each_provider(self) -> None:
        manager = ProviderManager()
        manager.register("one", capabilities={"ocr"})
        manager.register("two", capabilities={"ocr"})

        async def fail(provider: str) -> str:
            raise RuntimeError(provider)

        with self.assertRaises(NoProviderAvailable) as raised:
            await manager.execute("ocr", fail)
        self.assertEqual({"one", "two"}, set(raised.exception.errors))


class TTSServicesTests(unittest.TestCase):
    def test_clean_tts_text(self) -> None:
        from app.services.tts.voices import clean_tts_text

        raw = "Hello\u200b world!\ufeff\n\n\nHow are you?\x00"
        cleaned = clean_tts_text(raw)
        self.assertEqual("Hello world!\n\nHow are you?", cleaned)

    def test_split_text_chunks(self) -> None:
        from app.services.tts.voices import split_text_chunks

        short_text = "This is a short sentence."
        chunks = split_text_chunks(short_text, max_chars=100)
        self.assertEqual(["This is a short sentence."], chunks)

    def test_resolve_tts_voice_candidates(self) -> None:
        from app.services.tts.voices import resolve_tts_voice_candidates

        # Khmer female
        km_candidates = resolve_tts_voice_candidates("សួស្តី", "female")
        self.assertTrue(km_candidates[0].startswith("km-KH"))

        # English male
        en_candidates = resolve_tts_voice_candidates("Hello there", "male")
        self.assertTrue(en_candidates[0].startswith("en-US-Guy"))

    def test_normalize_gemini_tts_model(self) -> None:
        from app.services.tts.voices import normalize_tts_model, tts_model_label

        self.assertEqual("gemini", normalize_tts_model("gemini"))
        self.assertEqual("gemini", normalize_tts_model("gemini_tts"))
        self.assertEqual("gemini", normalize_tts_model("google_tts"))
        self.assertIn("Gemini", tts_model_label("gemini"))


    def test_audio_cache_lifecycle(self) -> None:
        from app.services.tts.cache import TTSAudioCache, make_tts_audio_cache_key

        cache = TTSAudioCache(max_bytes=1000, ttl_seconds=10.0)
        key = make_tts_audio_cache_key("hello", "female", 1.0, "auto")

        self.assertIsNone(cache.get(key))
        cache.set(key, b"fake_audio_bytes")
        self.assertEqual(b"fake_audio_bytes", cache.get(key))

        # Clear
        self.assertEqual(1, cache.clear())
        self.assertIsNone(cache.get(key))

    def test_user_history_tracker(self) -> None:
        from app.services.tts.cache import (
            TTSUserHistoryTracker,
            clear_user_tts_history,
            get_last_tts_text,
            set_last_tts_text,
        )

        tracker = TTSUserHistoryTracker(max_users=10)
        tracker.set_last_tts(1001)
        self.assertGreater(tracker.get_last_tts(1001), 0.0)

        tracker.set_last_tts_text(1001, "Hello from voice test")
        self.assertEqual("Hello from voice test", tracker.get_last_tts_text(1001))

        tracker.clear_user(1001)
        self.assertEqual(0.0, tracker.get_last_tts(1001))
        self.assertIsNone(tracker.get_last_tts_text(1001))

        # Module-level convenience functions with and without user_id
        set_last_tts_text(2002, "Sample global text")
        self.assertEqual("Sample global text", get_last_tts_text(2002))
        clear_user_tts_history(2002)
        self.assertIsNone(get_last_tts_text(2002))

        set_last_tts_text(3003, "Sample text 2")
        clear_user_tts_history()  # Clear all
        self.assertIsNone(get_last_tts_text(3003))



class GeminiServicesTests(unittest.TestCase):
    def test_detect_image_mime_from_bytes(self) -> None:
        from app.services.ai.gemini import detect_image_mime_from_bytes

        png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\r"
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        webp_header = b"RIFF\x00\x00\x00\x00WEBP"
        gif_header = b"GIF89a\x01\x00\x01\x00"

        self.assertEqual("image/png", detect_image_mime_from_bytes(png_header))
        self.assertEqual("image/jpeg", detect_image_mime_from_bytes(jpeg_header))
        self.assertEqual("image/webp", detect_image_mime_from_bytes(webp_header))
        self.assertEqual("image/gif", detect_image_mime_from_bytes(gif_header))
        self.assertEqual("image/jpeg", detect_image_mime_from_bytes(b"unknown"))

    def test_is_retryable_gemini_error(self) -> None:
        from app.services.ai.gemini import is_retryable_gemini_error

        self.assertTrue(is_retryable_gemini_error("429 Resource has been exhausted"))
        self.assertTrue(is_retryable_gemini_error("503 Service Unavailable"))
        self.assertTrue(is_retryable_gemini_error("temporarily overloaded"))
        self.assertFalse(is_retryable_gemini_error("Invalid API Key"))


class VectorStoreServicesTests(unittest.TestCase):
    def test_vector_store_initialization(self) -> None:
        from app.services.ai.vector_store import UpstashVectorStore

        store = UpstashVectorStore(
            url="https://test-endpoint.upstash.io",
            token="test-token-123",
        )
        self.assertTrue(store.is_configured)
        self.assertEqual("https://test-endpoint.upstash.io", store.url)
        self.assertEqual("Bearer test-token-123", store._headers()["Authorization"])

        unconfigured = UpstashVectorStore(url="", token="")
        self.assertFalse(unconfigured.is_configured)



class ButtonCustomizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_button_label_fallback_and_mutation(self) -> None:
        from app.services.telegram.buttons import (
            DEFAULT_BUTTON_LABELS,
            get_all_button_labels,
            get_button_label,
            reset_button_label,
            set_button_label,
        )

        self.assertEqual(DEFAULT_BUTTON_LABELS["btn_female"], get_button_label("btn_female"))
        all_labels = get_all_button_labels()
        self.assertIn("btn_female", all_labels)
        self.assertIn("btn_male", all_labels)

        # Custom override
        await set_button_label("btn_female", "👩 Custom Female Voice")
        self.assertEqual("👩 Custom Female Voice", get_button_label("btn_female"))

        # Reset
        await reset_button_label("btn_female")
        self.assertEqual(DEFAULT_BUTTON_LABELS["btn_female"], get_button_label("btn_female"))


class SecurityServicesTests(unittest.IsolatedAsyncioTestCase):

    def test_telegram_command_name_extraction(self) -> None:
        from unittest.mock import MagicMock

        from app.services.telegram.security import (
            ADMIN_ONLY_COMMANDS,
            telegram_command_name,
        )

        mock_update = MagicMock()
        mock_update.effective_message.text = "/admin@MyVoiceBot status"
        self.assertEqual("admin", telegram_command_name(mock_update))
        self.assertIn("admin", ADMIN_ONLY_COMMANDS)

        mock_update.effective_message.text = "/stats"
        self.assertEqual("stats", telegram_command_name(mock_update))
        self.assertIn("stats", ADMIN_ONLY_COMMANDS)

        mock_update.effective_message.text = "Hello world"
        self.assertEqual("", telegram_command_name(mock_update))

    async def test_security_notice_once_rate_limits(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.services.telegram.security import security_notice_once

        mock_update = MagicMock()
        mock_update.callback_query = None
        mock_update.effective_message.reply_text = AsyncMock()

        key = "test_user_guard_unique"
        # First send: should succeed
        await security_notice_once(mock_update, key, "Notice text", cooldown_seconds=10.0)
        self.assertEqual(1, mock_update.effective_message.reply_text.await_count)

        # Second send within cooldown: should be suppressed
        await security_notice_once(mock_update, key, "Notice text", cooldown_seconds=10.0)
        self.assertEqual(1, mock_update.effective_message.reply_text.await_count)


class BroadcastServicesTests(unittest.TestCase):
    def test_template_safe_id_and_int(self) -> None:
        from app.services.broadcast.templates import (
            broadcast_template_clean_preview,
            broadcast_template_fingerprint,
            broadcast_template_safe_id,
            broadcast_template_safe_int,
        )

        self.assertEqual("1234abcd", broadcast_template_safe_id("1234abcd"))
        self.assertEqual("", broadcast_template_safe_id("invalid-id-!@#$"))
        self.assertEqual(42, broadcast_template_safe_int("42"))
        self.assertEqual(0, broadcast_template_safe_int("invalid", default=0))

        preview = broadcast_template_clean_preview("<b>Hello</b> <i>World</i>", max_len=10)
        self.assertTrue(preview.startswith("Hello"))

        payload = {"text": "Broadcast message", "photo_file_id": None}
        fp = broadcast_template_fingerprint(payload)
        self.assertEqual(16, len(fp))


class UserPrefsServicesTests(unittest.TestCase):
    def test_normalize_user_prefs(self) -> None:
        from app.services.users.prefs import DEFAULT_USER_PREFS, normalize_user_prefs

        # Empty / None
        self.assertEqual(DEFAULT_USER_PREFS, normalize_user_prefs(None))
        self.assertEqual(DEFAULT_USER_PREFS, normalize_user_prefs({}))

        # Valid custom prefs
        custom = {"gender": "male", "speed": "1.5", "tts_model": "edge"}
        norm = normalize_user_prefs(custom)
        self.assertEqual("male", norm["gender"])
        self.assertEqual(1.5, norm["speed"])
        self.assertEqual("edge", norm["tts_model"])

        # Out-of-bounds speed clamped
        clamped = normalize_user_prefs({"gender": "invalid", "speed": 10.0})
        self.assertEqual("female", clamped["gender"])
        self.assertEqual(2.0, clamped["speed"])

    def test_user_prefs_cache(self) -> None:
        from app.services.users.prefs import UserPrefsCache

        cache = UserPrefsCache(max_size=5, ttl_seconds=10.0)
        self.assertIsNone(cache.get(101))

        cache.set(101, {"gender": "male", "speed": 1.2})
        saved = cache.get(101)
        self.assertIsNotNone(saved)
        self.assertEqual("male", saved["gender"])

        cache.invalidate(101)
        self.assertIsNone(cache.get(101))


class TextUtilityServicesTests(unittest.TestCase):
    def test_truncate_text(self) -> None:
        from app.utils.text import truncate_text

        self.assertEqual("Hello", truncate_text("Hello", 10))
        self.assertEqual("Hello…", truncate_text("Hello World", 6))

    def test_take_escaped_prefix(self) -> None:
        from app.utils.text import take_escaped_prefix

        prefix, rest = take_escaped_prefix("Hello <b>World</b>", 10)
        self.assertTrue(len(prefix) > 0)

    def test_html_safe_cut(self) -> None:
        from app.utils.text import html_safe_cut

        text = "<b>Hello World</b>"
        cut = html_safe_cut(text, 5)
        self.assertLessEqual(cut, len(text))

    def test_paginate_html(self) -> None:
        from app.utils.text import paginate_html

        pages = paginate_html("First paragraph\n\nSecond paragraph", limit=20)
        self.assertGreaterEqual(len(pages), 1)


class FileIOServicesTests(unittest.TestCase):
    def test_temp_file_lifecycle(self) -> None:
        import os

        from app.utils.file_io import cleanup_files, make_temp_img, make_temp_ogg

        ogg_path = make_temp_ogg()
        img_path = make_temp_img()

        self.assertTrue(ogg_path.endswith(".ogg"))
        self.assertTrue(img_path.endswith(".jpg"))
        self.assertTrue(os.path.isfile(ogg_path))
        self.assertTrue(os.path.isfile(img_path))

        cleanup_files(ogg_path, img_path)
        self.assertFalse(os.path.exists(ogg_path))
        self.assertFalse(os.path.exists(img_path))


if __name__ == "__main__":
    unittest.main()
