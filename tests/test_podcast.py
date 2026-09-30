"""Unit tests for Daily Morning Podcast service, generator, store, and handlers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_venv_site = Path(r"F:\ai project\bot-voice\.venv\Lib\site-packages")
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.append(str(_venv_site))

import types

if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except ImportError:
        sys.modules["httpx"] = MagicMock()

if "fastapi" not in sys.modules:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        import importlib.machinery
        fastapi_mod = types.ModuleType("fastapi")
        class MockHTTPException(Exception):
            def __init__(self, status_code: int = 400, detail: str = "", *args, **kwargs):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        fastapi_mod.HTTPException = MockHTTPException
        fastapi_responses = types.ModuleType("fastapi.responses")
        fastapi_responses.JSONResponse = type("JSONResponse", (), {"media_type": "application/json"})
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.services.podcast.generator import (
    build_podcast_card_html,
    clean_podcast_speech_text,
    format_khmer_date,
    generate_morning_podcast,
    get_cambodia_now,
    strip_podcast_unwanted_sections,
    to_khmer_numeral,
)
from app.services.podcast.handlers import (
    cmd_podcast,
    get_podcast_kb,
    podcast_callback,
    send_podcast_card,
)
from app.services.podcast.store import PodcastSubscriberStore


class PodcastStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.store_file = os.path.join(self.temp_dir, "test_subscribers.json")
        self.store = PodcastSubscriberStore(file_path=self.store_file)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_subscribe_and_unsubscribe(self) -> None:
        self.assertFalse(self.store.is_subscribed(12345))
        self.assertTrue(self.store.subscribe(12345))
        self.assertTrue(self.store.is_subscribed(12345))
        self.assertEqual(self.store.count(), 1)

        # Duplicate subscribe returns False
        self.assertFalse(self.store.subscribe(12345))
        self.assertEqual(self.store.count(), 1)

        # Unsubscribe
        self.assertTrue(self.store.unsubscribe(12345))
        self.assertFalse(self.store.is_subscribed(12345))
        self.assertEqual(self.store.count(), 0)

        # Unsubscribe non-existent returns False
        self.assertFalse(self.store.unsubscribe(12345))

    def test_unsubscribe_batch(self) -> None:
        self.store.subscribe(100)
        self.store.subscribe(200)
        self.store.subscribe(300)
        self.assertEqual(self.store.count(), 3)

        # Remove 100, 300, and non-existent 999
        removed = self.store.unsubscribe_batch([100, 300, 999])
        self.assertEqual(removed, 2)
        self.assertEqual(self.store.count(), 1)
        self.assertTrue(self.store.is_subscribed(200))
        self.assertFalse(self.store.is_subscribed(100))
        self.assertFalse(self.store.is_subscribed(300))

        # Empty batch returns 0
        self.assertEqual(self.store.unsubscribe_batch([]), 0)

    def test_persistence(self) -> None:
        self.store.subscribe(111)
        self.store.subscribe(222)
        self.store.set_last_broadcast_date("2026-09-14")

        # Reload store from same file
        reloaded = PodcastSubscriberStore(file_path=self.store_file)
        self.assertTrue(reloaded.is_subscribed(111))
        self.assertTrue(reloaded.is_subscribed(222))
        self.assertEqual(reloaded.get_last_broadcast_date(), "2026-09-14")
        self.assertEqual(reloaded.count(), 2)


class PodcastGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_morning_podcast_offline_fallback(self) -> None:
        # Test generation when Gemini is offline or returns fallback
        html_text, speech_text = await generate_morning_podcast(force_refresh=True)

        self.assertIn("Daily Morning Podcast", html_text)
        self.assertIn("<b>", html_text)
        self.assertIn("ព័ត៌មាន", html_text)
        self.assertIn("ប្រភពដើម", html_text)
        self.assertNotIn("ចំណេះដឹង", html_text)
        self.assertNotIn("ការលើកទឹកចិត្ត", html_text)
        self.assertNotIn("ហើយមុននឹងបញ្ចប់", html_text)
        self.assertTrue(len(speech_text) > 20)
        self.assertIn("ប្រភពដើម", speech_text)
        self.assertNotIn("<b>", speech_text)

    async def test_generate_morning_podcast_caching(self) -> None:
        html1, speech1 = await generate_morning_podcast(force_refresh=False)
        html2, speech2 = await generate_morning_podcast(force_refresh=False)

        self.assertEqual(html1, html2)
        self.assertEqual(speech1, speech2)

    def test_strip_podcast_unwanted_sections(self) -> None:
        dirty = (
            "🇰🇭 **ព័ត៌មានជាតិ:**\n• ព័ត៌មាន A (ប្រភពដើម៖ NBC)\n\n"
            "ហើយមុននឹងបញ្ចប់ សូមស្តាប់នូវគន្លឹះចំណេះដឹង និងការលើកទឹកចិត្តសម្រាប់ថ្ងៃថ្មី៖\n\n"
            "• 💡 ចំណេះដឹង & ការលើកទឹកចិត្ត:\n"
            "    • រៀនសូត្រថ្មីរាល់ថ្ងៃ: ចាប់ផ្តើមថ្ងៃថ្មីរបស់អ្នកដោយរៀនសូត្រអ្វីដែលថ្មីមួយ...\n\n"
            "ទាំងនេះគឺជាព័ត៌មានសំខាន់ៗ និងគន្លឹះលើកទឹកចិត្តប្រចាំថ្ងៃពី Bot Voice Cambodia។ "
            "សូមជូនពរលោកអ្នកទទួលបានថាមពល និងភាពជោគជ័យពេញមួយថ្ងៃ។ ជួបគ្នាសារជាថ្មីនៅថ្ងៃស្អែក! សូមអរគុណ!\n\n"
            "📰 **ប្រភពដើម (News Sources):**\n• NBC"
        )
        cleaned = strip_podcast_unwanted_sections(dirty)
        self.assertNotIn("ហើយមុននឹងបញ្ចប់", cleaned)
        self.assertNotIn("ចំណេះដឹង", cleaned)
        self.assertNotIn("ការលើកទឹកចិត្ត", cleaned)
        self.assertNotIn("ទាំងនេះគឺជាព័ត៌មានសំខាន់ៗ", cleaned)
        self.assertNotIn("សូមជូនពរលោកអ្នក", cleaned)
        self.assertNotIn("ជួបគ្នាសារជាថ្មី", cleaned)
        self.assertIn("ព័ត៌មានជាតិ", cleaned)
        self.assertIn("ប្រភពដើម", cleaned)

    def test_clean_podcast_speech_text(self) -> None:
        dirty_script = (
            "🇰🇭 <b>ព័ត៌មានជាតិ</b> [pause]\n"
            "• សម្តេចធិបតីអញ្ជើញជាអធិបតី (ប្រភពដើម៖ ក្រសួងព័ត៌មាន)\n"
            "• 🌤️ អាកាសធាតុថ្ងៃនេះល្អប្រសើរ 💡"
        )
        cleaned = clean_podcast_speech_text(dirty_script, khmer_date="ថ្ងៃពុធ")
        # Emojis must be completely stripped to prevent Edge TTS stuttering
        self.assertNotIn("🇰🇭", cleaned)
        self.assertNotIn("🌤️", cleaned)
        self.assertNotIn("💡", cleaned)
        # Bracket pause stripped
        self.assertNotIn("[pause]", cleaned)
        # HTML stripped
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("</b>", cleaned)
        # (ប្រភពដើម៖ ...) transformed into natural spoken Khmer
        self.assertIn("ប្រភពដើមពី ក្រសួងព័ត៌មាន", cleaned)
        self.assertNotIn("(ប្រភពដើម៖", cleaned)

    def test_build_podcast_card_html_limit(self) -> None:
        huge_body = "• ព័ត៌មានសំខាន់ប្រចាំថ្ងៃ៖ " + ("កម្ពុជាមានការអភិវឌ្ឍរីកចម្រើនយ៉ាងឆាប់រហ័ស។ " * 50)
        card = build_podcast_card_html(huge_body, khmer_date="ថ្ងៃពុធ ទី១៦ ខែកញ្ញា ឆ្នាំ២០២៦", max_chars=1024)
        # Must fit strictly within Telegram photo caption limit (1024 UTF-16 code units)
        utf16_units = len(card.encode("utf-16-le")) // 2
        self.assertLessEqual(utf16_units, 1024)
        self.assertIn("Daily Morning Podcast", card)
        self.assertIn("ថ្ងៃពុធ", card)


class PodcastHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.store_file = os.path.join(self.temp_dir, "test_subscribers.json")
        self.test_store = PodcastSubscriberStore(file_path=self.store_file)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_cmd_podcast_subscribe(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.text = "/podcast on"
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.effective_user.id = 123
        mock_update.effective_chat.id = 123

        with patch("app.services.podcast.handlers.podcast_store", self.test_store):
            await cmd_podcast(mock_update, MagicMock())
            self.assertTrue(self.test_store.is_subscribed(123))
            mock_msg.reply_text.assert_called_once()
            call_text = mock_msg.reply_text.call_args[0][0]
            self.assertIn("បានចុះឈ្មោះជាវជោគជ័យ", call_text)

    async def test_cmd_podcast_unsubscribe(self) -> None:
        self.test_store.subscribe(123)
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.text = "/podcast off"
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.effective_user.id = 123
        mock_update.effective_chat.id = 123

        with patch("app.services.podcast.handlers.podcast_store", self.test_store):
            await cmd_podcast(mock_update, MagicMock())
            self.assertFalse(self.test_store.is_subscribed(123))
            mock_msg.reply_text.assert_called_once()
            call_text = mock_msg.reply_text.call_args[0][0]
            self.assertIn("បានបោះបង់ការជាវជោគជ័យ", call_text)

    async def test_cmd_podcast_on_demand_delivery(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.text = "/podcast"
        mock_msg.reply_text = AsyncMock()
        mock_msg.reply_photo = AsyncMock(return_value="photo_sent")
        mock_msg.reply_chat_action = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.effective_user.id = 456
        mock_update.effective_chat.id = 456
        mock_context = MagicMock()

        with patch("app.services.podcast.handlers.podcast_store", self.test_store):
            with patch("app.services.podcast.handlers.send_podcast_voice", new_callable=AsyncMock) as mock_voice:
                await cmd_podcast(mock_update, mock_context)

                if mock_msg.reply_photo.called:
                    call_text = mock_msg.reply_photo.call_args[1].get("caption", "")
                else:
                    mock_msg.reply_text.assert_called_once()
                    call_text = mock_msg.reply_text.call_args[0][0]
                self.assertIn("Daily Morning Podcast", call_text)
                mock_voice.assert_awaited_once()

    async def test_podcast_callback_actions(self) -> None:
        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_query.data = "podcast_sub"
        mock_query.from_user.id = 789
        mock_query.message.chat_id = 789
        mock_query.answer = AsyncMock()
        mock_query.edit_message_reply_markup = AsyncMock()
        mock_update.callback_query = mock_query

        with patch("app.services.podcast.handlers.podcast_store", self.test_store):
            # Test subscribe callback
            await podcast_callback(mock_update, MagicMock())
            self.assertTrue(self.test_store.is_subscribed(789))
            mock_query.answer.assert_awaited_once()

            # Test unsubscribe callback
            mock_query.data = "podcast_unsub"
            mock_query.answer.reset_mock()
            await podcast_callback(mock_update, MagicMock())
            self.assertFalse(self.test_store.is_subscribed(789))

    def test_khmer_date_formatting(self) -> None:
        from datetime import datetime
        dt = datetime(2026, 9, 14, 7, 0)
        khmer_str = format_khmer_date(dt)
        self.assertEqual(khmer_str, "១៤ កញ្ញា ឆ្នាំ ២០២៦")
        self.assertEqual(to_khmer_numeral(2026), "២០២៦")
        self.assertEqual(to_khmer_numeral("100"), "១០០")

    def test_get_podcast_kb_has_share_button(self) -> None:
        kb = get_podcast_kb(is_subscribed=False, bot_username="voicekhaibot")
        self.assertEqual(len(kb.inline_keyboard), 4)
        # Row 1: Dual presenters
        self.assertEqual(kb.inline_keyboard[0][0].callback_data, "podcast_voice_f")
        self.assertEqual(kb.inline_keyboard[0][1].callback_data, "podcast_voice_m")
        # Row 2: MP3 and Refresh
        self.assertEqual(kb.inline_keyboard[1][0].callback_data, "podcast_mp3")
        self.assertEqual(kb.inline_keyboard[1][1].callback_data, "podcast_refresh")
        # Row 3: Subscribe and Share
        self.assertEqual(kb.inline_keyboard[2][0].callback_data, "podcast_sub")
        share_btn = kb.inline_keyboard[2][1]
        self.assertIn("ចែករំលែក", share_btn.text)
        self.assertIn("share/url", share_btn.url)
        self.assertIn("voicekhaibot", share_btn.url)
        # Row 4: Close
        self.assertEqual(kb.inline_keyboard[3][0].callback_data, "podcast_close")

    async def test_send_podcast_card_photo(self) -> None:
        mock_msg = MagicMock()
        mock_msg.reply_photo = AsyncMock(return_value="photo_sent")
        res = await send_podcast_card(mock_msg, "Sample <b>Daily Morning Podcast</b> HTML")
        mock_msg.reply_photo.assert_called_once()
        self.assertEqual(res, "photo_sent")

    def test_source_banner_bytes_priority(self) -> None:
        from app.services.podcast.handlers import (
            get_banner_bytes,
            get_cached_banner_file_id,
            get_source_banner_bytes,
            set_cached_banner_file_id,
            set_source_banner_bytes,
        )

        test_data = b"\xff\xd8\xff\xe0test_source_jpg_data"
        set_source_banner_bytes(test_data)
        self.assertEqual(get_source_banner_bytes(), test_data)
        self.assertEqual(get_banner_bytes(), test_data)

        set_cached_banner_file_id("file_id_source_123")
        self.assertEqual(get_cached_banner_file_id(), "file_id_source_123")

        # Clear and check fallback
        set_source_banner_bytes(None)
        self.assertIsNone(get_source_banner_bytes())

        # Test date invalidation: if fetched date is yesterday, stale banner is cleared
        set_source_banner_bytes(test_data)
        from app.services.podcast import handlers
        handlers._SOURCE_BANNER_FETCHED_DATE = "2020-01-01"
        self.assertIsNone(get_source_banner_bytes())
        # Calling get_banner_bytes should clear the stale source banner
        banner_result = get_banner_bytes()
        self.assertNotEqual(banner_result, test_data)
        self.assertIsNone(handlers._SOURCE_BANNER_BYTES)

    async def test_podcast_callback_close_deletes_message(self) -> None:
        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_query.data = "podcast_close"
        mock_query.from_user.id = 789
        mock_query.message.chat_id = 789
        mock_query.message.delete = AsyncMock()
        mock_query.answer = AsyncMock()
        mock_update.callback_query = mock_query

        await podcast_callback(mock_update, MagicMock())
        mock_query.answer.assert_awaited_once_with("បិទ")
        mock_query.message.delete.assert_awaited_once()

    async def test_podcast_content_has_source_and_no_social_clutter(self) -> None:
        html_out, speech_out = await generate_morning_podcast(force_refresh=True)
        self.assertIn("ប្រភពដើម", html_out)
        self.assertNotIn("Facebook", html_out)
        self.assertNotIn("TikTok", html_out)
        self.assertNotIn("Binance", html_out)
        self.assertNotIn("Group Support", html_out)
        self.assertIn("Daily Morning Podcast", html_out)

    async def test_get_or_synthesize_podcast_voice_caching(self) -> None:
        from app.services.podcast.handlers import (
            get_cached_podcast_voice_bytes,
            get_cached_podcast_voice_file_id,
            get_or_synthesize_podcast_voice,
            set_podcast_voice_bytes,
        )

        test_bytes = b"fake_ogg_opus_audio_bytes"
        set_podcast_voice_bytes(test_bytes, fid="fid_morning_voice_001")
        self.assertEqual(get_cached_podcast_voice_bytes(), test_bytes)
        self.assertEqual(get_cached_podcast_voice_file_id(), "fid_morning_voice_001")

        # Calling get_or_synthesize_podcast_voice should return cached file_id directly
        raw, fid = await get_or_synthesize_podcast_voice("Sample text", force_refresh=False)
        self.assertIsNone(raw)
        self.assertEqual(fid, "fid_morning_voice_001")

        # Test force_refresh resets cache and triggers fresh synthesis
        with patch("app.legacy.generate_voice", new_callable=AsyncMock, return_value=b"fresh_voice_bytes") as mock_gen:
            raw_fresh, fid_fresh = await get_or_synthesize_podcast_voice("New text", force_refresh=True)
            mock_gen.assert_awaited_once()
            self.assertEqual(raw_fresh, b"fresh_voice_bytes")
            self.assertIsNone(fid_fresh)
            self.assertEqual(get_cached_podcast_voice_bytes(), b"fresh_voice_bytes")

        # Reset
        set_podcast_voice_bytes(None)

    async def test_dual_presenter_voice_caching(self) -> None:
        from app.services.podcast.handlers import (
            get_cached_podcast_voice_bytes,
            get_cached_podcast_voice_file_id,
            get_or_synthesize_podcast_voice,
            set_podcast_voice_bytes,
        )

        # Set distinct caches for female and male
        set_podcast_voice_bytes(b"female_audio", fid="fid_female_01", presenter="female")
        set_podcast_voice_bytes(b"male_audio", fid="fid_male_02", presenter="male")

        self.assertEqual(get_cached_podcast_voice_bytes(presenter="female"), b"female_audio")
        self.assertEqual(get_cached_podcast_voice_file_id(presenter="female"), "fid_female_01")
        self.assertEqual(get_cached_podcast_voice_bytes(presenter="male"), b"male_audio")
        self.assertEqual(get_cached_podcast_voice_file_id(presenter="male"), "fid_male_02")

        # Calling get_or_synthesize_podcast_voice respects presenter
        _, fid_f = await get_or_synthesize_podcast_voice("test", presenter="female")
        self.assertEqual(fid_f, "fid_female_01")

        _, fid_m = await get_or_synthesize_podcast_voice("test", presenter="male")
        self.assertEqual(fid_m, "fid_male_02")

        # Reset
        set_podcast_voice_bytes(None, presenter="female")
        set_podcast_voice_bytes(None, presenter="male")

    async def test_send_podcast_voice_delivers_single_voice_note(self) -> None:
        from app.services.podcast.handlers import (
            send_podcast_voice,
            set_podcast_voice_bytes,
        )

        mock_msg = MagicMock()
        mock_msg.reply_voice = AsyncMock(return_value=MagicMock(voice=MagicMock(file_id="new_fid_123")))

        # 1. Send via raw audio bytes (first time)
        set_podcast_voice_bytes(b"initial_voice_bytes")
        await send_podcast_voice(mock_msg, "Spoken text")
        mock_msg.reply_voice.assert_awaited_once()
        call_kwargs = mock_msg.reply_voice.call_args[1]
        self.assertIn("Daily Morning Podcast", call_kwargs.get("caption", ""))

        # 2. Subsequent call reuses captured file_id
        mock_msg.reply_voice.reset_mock()
        await send_podcast_voice(mock_msg, "Spoken text")
        mock_msg.reply_voice.assert_awaited_once()
        self.assertEqual(mock_msg.reply_voice.call_args[1].get("voice"), "new_fid_123")

        # Reset
        set_podcast_voice_bytes(None)

    async def test_send_podcast_mp3_delivery_and_caching(self) -> None:
        from app.services.podcast.handlers import (
            get_cached_podcast_mp3_file_id,
            send_podcast_mp3,
            set_cached_podcast_mp3_file_id,
            set_podcast_voice_bytes,
        )

        mock_msg = MagicMock()
        mock_msg.reply_audio = AsyncMock(return_value=MagicMock(audio=MagicMock(file_id="fid_mp3_999")))

        set_podcast_voice_bytes(b"sample_mp3_bytes", presenter="female")
        await send_podcast_mp3(mock_msg, "Spoken text", presenter="female")
        mock_msg.reply_audio.assert_called_once()
        call_kwargs = mock_msg.reply_audio.call_args[1]
        self.assertIn("MP3", call_kwargs.get("caption", ""))
        self.assertEqual(get_cached_podcast_mp3_file_id("female"), "fid_mp3_999")

        # Subsequent call uses cached audio file_id
        mock_msg.reply_audio.reset_mock()
        await send_podcast_mp3(mock_msg, "Spoken text", presenter="female")
        mock_msg.reply_audio.assert_called_once()
        self.assertEqual(mock_msg.reply_audio.call_args[1].get("audio"), "fid_mp3_999")

        # Reset
        set_cached_podcast_mp3_file_id(None, presenter="female")
        set_podcast_voice_bytes(None, presenter="female")

    async def test_podcast_callback_actions_and_presenters(self) -> None:
        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_query.from_user.id = 999
        mock_query.message.chat_id = 999
        mock_query.answer = AsyncMock()
        mock_update.callback_query = mock_query

        with patch("app.services.podcast.handlers.send_podcast_voice", new_callable=AsyncMock) as mock_voice, \
             patch("app.services.podcast.handlers.send_podcast_mp3", new_callable=AsyncMock) as mock_mp3:

            # 1. podcast_voice_f
            mock_query.data = "podcast_voice_f"
            await podcast_callback(mock_update, MagicMock())
            mock_voice.assert_awaited_once()
            self.assertEqual(mock_voice.call_args[1].get("presenter"), "female")

            # 2. podcast_voice_m
            mock_voice.reset_mock()
            mock_query.data = "podcast_voice_m"
            await podcast_callback(mock_update, MagicMock())
            mock_voice.assert_awaited_once()
            self.assertEqual(mock_voice.call_args[1].get("presenter"), "male")

            # 3. podcast_mp3
            mock_query.data = "podcast_mp3"
            await podcast_callback(mock_update, MagicMock())
            mock_mp3.assert_awaited_once()

    async def test_cmd_podcast_stats_and_subcommands(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_msg.reply_chat_action = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.effective_user.id = 456
        mock_update.effective_chat.id = 456

        with patch("app.services.podcast.handlers.podcast_store", self.test_store):
            # Test /podcast stats
            mock_msg.text = "/podcast stats"
            await cmd_podcast(mock_update, MagicMock())
            mock_msg.reply_text.assert_called_once()
            stats_reply = mock_msg.reply_text.call_args[0][0]
            self.assertIn("ស្ថិតិ Daily Morning Podcast", stats_reply)

            # Test /podcast male
            with patch("app.services.podcast.handlers.send_podcast_voice", new_callable=AsyncMock) as mock_voice:
                mock_msg.text = "/podcast male"
                await cmd_podcast(mock_update, MagicMock())
                mock_voice.assert_awaited_once()
                self.assertEqual(mock_voice.call_args[1].get("presenter"), "male")

    async def test_cmd_podcast_broadcast_admin_check(self) -> None:
        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_msg.text = "/podcast broadcast"
        mock_update.effective_message = mock_msg
        mock_update.effective_user.id = 8888
        mock_update.effective_chat.id = 8888

        # Non-admin user gets rejected
        with patch("app.services.podcast.handlers.is_admin_user", return_value=False):
            await cmd_podcast(mock_update, MagicMock())
            mock_msg.reply_text.assert_called_once()
            call_text = mock_msg.reply_text.call_args[0][0]
            self.assertIn("Admin Only", call_text)

        # Admin user executes broadcast
        mock_msg.reply_text.reset_mock()
        self.test_store.subscribe(101)
        self.test_store.subscribe(102)

        mock_bot = MagicMock()
        mock_context = MagicMock()
        mock_context.bot = mock_bot

        with patch("app.services.podcast.handlers.is_admin_user", return_value=True), \
             patch("app.services.podcast.handlers.podcast_store", self.test_store), \
             patch("app.services.podcast.handlers.get_or_synthesize_podcast_voice", new_callable=AsyncMock, return_value=(b"voice", "fid")), \
             patch("app.services.podcast.handlers.send_podcast_card", new_callable=AsyncMock) as mock_send_card, \
             patch("app.services.podcast.handlers.send_podcast_voice", new_callable=AsyncMock) as mock_send_voice:
            await cmd_podcast(mock_update, mock_context)
            self.assertEqual(mock_send_card.call_count, 2)
            self.assertEqual(mock_send_voice.call_count, 2)

    async def test_podcast_callback_refresh_uses_send_podcast_voice(self) -> None:
        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_query.data = "podcast_refresh"
        mock_query.from_user.id = 999
        mock_query.message.chat_id = 999
        mock_query.answer = AsyncMock()
        mock_update.callback_query = mock_query

        with patch("app.services.podcast.handlers.podcast_store", self.test_store), \
             patch("app.services.podcast.handlers.send_podcast_card", new_callable=AsyncMock), \
             patch("app.services.podcast.handlers.send_podcast_voice", new_callable=AsyncMock) as mock_voice:
            await podcast_callback(mock_update, MagicMock())
            mock_voice.assert_awaited_once()
            self.assertTrue(mock_voice.call_args[1].get("force_refresh"))


class PodcastWeatherAndHeadlinesTests(unittest.TestCase):
    def test_fetch_cambodia_weather_structure(self) -> None:
        from app.services.podcast.generator import fetch_cambodia_weather
        weather = fetch_cambodia_weather(timeout_s=3.0)
        self.assertIsInstance(weather, dict)
        self.assertIn("phnom_penh", weather)
        self.assertIn("summary", weather)
        self.assertIn("សីតុណ្ហភាព", weather["phnom_penh"])

    def test_fetch_live_cambodia_headlines(self) -> None:
        from app.services.podcast.generator import fetch_live_cambodia_headlines
        headlines = fetch_live_cambodia_headlines(limit=3)
        self.assertIsInstance(headlines, list)
        self.assertTrue(len(headlines) > 0)
        self.assertIsInstance(headlines[0], str)
        self.assertTrue(len(headlines[0]) > 5)


class PodcastAdminIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_podcast_voice_and_mp3_callbacks(self) -> None:
        from app.services.admin.handlers import handle_admin_callback
        mock_query = MagicMock()
        mock_query.from_user.id = 12345
        mock_query.answer = AsyncMock()
        mock_context = MagicMock()

        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.podcast.generator.generate_morning_podcast", new_callable=AsyncMock, return_value=("html", "speech")), \
             patch("app.services.podcast.handlers.send_podcast_voice", new_callable=AsyncMock) as mock_voice, \
             patch("app.services.podcast.handlers.send_podcast_mp3", new_callable=AsyncMock) as mock_mp3:

            # Female voice
            res_f = await handle_admin_callback(mock_query, 12345, mock_context, "admin_podcast_voice_f")
            self.assertTrue(res_f)
            mock_voice.assert_awaited_with(None, "speech", presenter="female", bot=mock_context.bot, chat_id=12345)

            # Male voice
            mock_voice.reset_mock()
            res_m = await handle_admin_callback(mock_query, 12345, mock_context, "admin_podcast_voice_m")
            self.assertTrue(res_m)
            mock_voice.assert_awaited_with(None, "speech", presenter="male", bot=mock_context.bot, chat_id=12345)

            # MP3 track
            res_mp3 = await handle_admin_callback(mock_query, 12345, mock_context, "admin_podcast_mp3")
            self.assertTrue(res_mp3)
            mock_mp3.assert_awaited_with(None, "speech", presenter="female", bot=mock_context.bot, chat_id=12345)


if __name__ == "__main__":
    unittest.main()

