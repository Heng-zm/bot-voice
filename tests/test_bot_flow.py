"""Unit tests for Smart Text & AI Logic Flow."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_venv_site = Path(r"F:\ai project\bot-voice\.venv\Lib\site-packages")
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.append(str(_venv_site))

if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except ImportError:
        sys.modules["httpx"] = MagicMock()

if "fastapi" not in sys.modules or not hasattr(sys.modules["fastapi"], "FastAPI"):
    try:
        import fastapi  # noqa: F401
    except ImportError:
        import types
        fastapi_mod = types.ModuleType("fastapi")
        fastapi_responses = types.ModuleType("fastapi.responses")
        fastapi_responses.JSONResponse = type("JSONResponse", (), {"media_type": "application/json"})
        class HTTPException(Exception):
            def __init__(self, status_code: int = 500, detail: str = ""):
                self.status_code = status_code
                self.detail = detail
        class MockRoute:
            def __init__(self, path: str):
                self.path = path

        class MockFastAPI:
            def __init__(self, *args, **kwargs):
                self.routes: list[Any] = []
            def include_router(self, router, *args, **kwargs):
                if hasattr(router, "routes"):
                    self.routes.extend(router.routes)
            def middleware(self, *args, **kwargs):
                return lambda f: f
            def get(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(path))
                return lambda f: f
            def post(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(path))
                return lambda f: f
            def add_middleware(self, *args, **kwargs):
                pass

        class MockAPIRouter:
            def __init__(self, *args, prefix: str = "", **kwargs):
                self.prefix = prefix
                self.routes: list[Any] = []
            def get(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(f"{self.prefix}{path}"))
                return lambda f: f
            def post(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(f"{self.prefix}{path}"))
                return lambda f: f
            def head(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(f"{self.prefix}{path}"))
                return lambda f: f
            def put(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(f"{self.prefix}{path}"))
                return lambda f: f
            def delete(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(f"{self.prefix}{path}"))
                return lambda f: f
            def patch(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(f"{self.prefix}{path}"))
                return lambda f: f
            def options(self, path: str = "", *args, **kwargs):
                self.routes.append(MockRoute(f"{self.prefix}{path}"))
                return lambda f: f
            def include_router(self, router, *args, prefix: str = "", **kwargs):
                combined = f"{self.prefix}{prefix}"
                if hasattr(router, "routes"):
                    for r in router.routes:
                        self.routes.append(MockRoute(f"{combined}{r.path}"))

        fastapi_mod.FastAPI = MockFastAPI
        fastapi_mod.APIRouter = MockAPIRouter
        fastapi_mod.HTTPException = HTTPException
        fastapi_mod.Header = lambda default=None, **kw: default
        fastapi_mod.Request = type("Request", (), {})
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.services.telegram.flow import classify_callback, classify_text_intent
from app.services.users.prefs import DEFAULT_BOT_MODE, DEFAULT_USER_PREFS, normalize_user_prefs


class BotLogicFlowTests(unittest.IsolatedAsyncioTestCase):
    """Test suite for Smart Text Intent Classification and Bot Mode Logic Flow."""

    def test_classify_text_intent_auto_khmer_questions(self) -> None:
        # Question starters
        self.assertEqual(classify_text_intent("តើភ្នំពេញជារាជធានីនៃប្រទេសណា?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("ហេតុអ្វីបានជាមេឃពណ៌ខៀវ", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("របៀបធ្វើម្ហូបខ្មែរឆ្ងាញ់", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("អ្វីខ្លះជាអត្ថប្រយោជន៍នៃ AI?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("នៅឯណាជាកន្លែងស្អាតជាងគេ?", "auto"), "ai_chat")

        # Question endings
        self.assertEqual(classify_text_intent("ថ្ងៃនេះសុខសប្បាយទេ?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("បងញ៉ាំបាយហើយឬនៅ?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("រឿងនេះពិតមែនទេ?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("រឿងនេះពិតមែនទេ", "auto"), "ai_chat")

        # Greetings
        self.assertEqual(classify_text_intent("សួស្តីបង!", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("ជំរាបសួរលោកគ្រូ", "auto"), "ai_chat")

    def test_classify_text_intent_auto_english_questions(self) -> None:
        self.assertEqual(classify_text_intent("What is the capital of Cambodia?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("How can I learn Khmer fast?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("Why is artificial intelligence important?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("Tell me a story about Angkor Wat", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("Hello, how are you today?", "auto"), "ai_chat")
        self.assertEqual(classify_text_intent("Hi there!", "auto"), "ai_chat")

    def test_classify_text_intent_auto_narrative_to_tts(self) -> None:
        # Normal narrative, news, or statements should be read via TTS
        self.assertEqual(classify_text_intent("ព្រះរាជាណាចក្រកម្ពុជា ជាប្រទេសមួយដ៏ស្រស់បំព្រង។", "auto"), "tts")
        self.assertEqual(classify_text_intent("កិច្ចប្រជុំបានបញ្ចប់ដោយរលូនកាលពីម្សិលមិញ។", "auto"), "tts")
        self.assertEqual(classify_text_intent("Today is a great sunny day in Phnom Penh.", "auto"), "tts")

        # Negative statements ending in ទេ without ? should be read via TTS
        self.assertEqual(classify_text_intent("ខ្ញុំមិនចង់ទៅទេ", "auto"), "tts")
        self.assertEqual(classify_text_intent("រឿងនេះមិនពិតទេ", "auto"), "tts")
        self.assertEqual(classify_text_intent("កិច្ចប្រជុំមិនទាន់ចប់ទេ", "auto"), "tts")
        self.assertEqual(classify_text_intent("ខ្ញុំគ្មានលុយទេ", "auto"), "tts")

    def test_classify_text_intent_prefix_overrides(self) -> None:
        # '?' prefix always forces AI Chat, even in TTS mode or on non-questions
        self.assertEqual(classify_text_intent("? ប្រទេសកម្ពុជា", "tts"), "ai_chat")
        self.assertEqual(classify_text_intent("？ Angkor Wat", "auto"), "ai_chat")

        # '!' prefix always forces Direct TTS, even on questions
        self.assertEqual(classify_text_intent("! តើអ្នកឈ្មោះអ្វី?", "auto"), "tts")
        self.assertEqual(classify_text_intent("! What is this?", "ai_chat"), "tts")

    def test_classify_text_intent_explicit_modes(self) -> None:
        # In 'tts' mode, everything without '?' is TTS
        self.assertEqual(classify_text_intent("តើអ្នកឈ្មោះអ្វី?", "tts"), "tts")
        self.assertEqual(classify_text_intent("Hello how are you?", "tts"), "tts")

        # In 'ai_chat' mode, everything without '!' is AI
        self.assertEqual(classify_text_intent("ព្រះរាជាណាចក្រកម្ពុជា", "ai_chat"), "ai_chat")
        self.assertEqual(classify_text_intent("This is a simple sentence.", "ai_chat"), "ai_chat")

    def test_classify_callback_mode_actions(self) -> None:
        self.assertEqual(classify_callback("show_mode"), "show_mode")
        self.assertEqual(classify_callback("hide_mode"), "hide_mode")
        self.assertEqual(classify_callback("mode_auto"), "mode_change")
        self.assertEqual(classify_callback("mode_tts"), "mode_change")
        self.assertEqual(classify_callback("mode_ai"), "mode_change")

    def test_normalize_user_prefs_bot_mode(self) -> None:
        self.assertEqual(DEFAULT_USER_PREFS["bot_mode"], "auto")
        self.assertEqual(normalize_user_prefs(None)["bot_mode"], "auto")
        self.assertEqual(normalize_user_prefs({"bot_mode": "tts"})["bot_mode"], "tts")
        self.assertEqual(normalize_user_prefs({"bot_mode": "ai_chat"})["bot_mode"], "ai_chat")
        self.assertEqual(normalize_user_prefs({"bot_mode": "ai"})["bot_mode"], "ai_chat")
        self.assertEqual(normalize_user_prefs({"bot_mode": "invalid"})["bot_mode"], "auto")

    def test_get_bot_mode_kb_checkmarks(self) -> None:
        from app.legacy import get_bot_mode_kb

        # Auto checkmark
        kb_auto = get_bot_mode_kb("auto")
        all_texts_auto = [btn.text for row in kb_auto.inline_keyboard for btn in row]
        self.assertTrue(any("ស្វ័យប្រវត្តិ" in t and "✅" in t for t in all_texts_auto))
        self.assertFalse(any("អានសំឡេងផ្ទាល់" in t and "✅" in t for t in all_texts_auto))

        # TTS checkmark
        kb_tts = get_bot_mode_kb("tts")
        all_texts_tts = [btn.text for row in kb_tts.inline_keyboard for btn in row]
        self.assertTrue(any("អានសំឡេងផ្ទាល់" in t and "✅" in t for t in all_texts_tts))
        self.assertFalse(any("ស្វ័យប្រវត្តិ" in t and "✅" in t for t in all_texts_tts))

        # AI checkmark
        kb_ai = get_bot_mode_kb("ai_chat")
        all_texts_ai = [btn.text for row in kb_ai.inline_keyboard for btn in row]
        self.assertTrue(any("សួរឆ្លើយ AI" in t and "✅" in t for t in all_texts_ai))

    async def test_cmd_mode_inspect_and_switch(self) -> None:
        from app.services.telegram.commands import cmd_mode

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 12345
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg

        mock_context = MagicMock()
        mock_context.args = []

        with patch("app.legacy.get_user_prefs_async", new_callable=AsyncMock) as mock_prefs:
            mock_prefs.return_value = {"bot_mode": "auto", "gender": "female", "speed": 1.0}
            await cmd_mode(mock_update, mock_context)
            mock_msg.reply_text.assert_awaited_once()
            output = mock_msg.reply_text.call_args[0][0]
            self.assertIn("ជ្រើសរើសរបៀបដំណើរការរបស់ Bot", output)

        mock_msg.reply_text.reset_mock()
        mock_context.args = ["tts"]
        with patch("app.legacy.update_user_bot_mode") as mock_update_mode:
            mock_update_mode.return_value = "tts"
            await cmd_mode(mock_update, mock_context)
            mock_msg.reply_text.assert_awaited_once()
            output = mock_msg.reply_text.call_args[0][0]
            self.assertIn("បានកំណត់របៀបដំណើរការ", output)
            self.assertIn("អានសំឡេងផ្ទាល់", output)

    async def test_on_text_routes_to_ai_chat_for_questions(self) -> None:
        from app.services.telegram.media import on_text

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 55555
        mock_user.username = "test_user"
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.text = "តើ AI គឺជាអ្វី?"
        mock_update.message = mock_msg
        mock_context = MagicMock()
        mock_context.user_data = {}

        with patch("app.legacy._is_admin", return_value=False), \
             patch("app.legacy._get_admin_for_user", return_value=None), \
             patch("app.legacy._ensure_user_allowed", new_callable=AsyncMock, return_value=True), \
             patch("app.legacy._check_cooldown", new_callable=AsyncMock, return_value=False), \
             patch("app.legacy.get_user_prefs_async", new_callable=AsyncMock) as mock_prefs, \
             patch("app.services.telegram.media.process_ai_chat_for_text", new_callable=AsyncMock) as mock_ai, \
             patch("app.services.telegram.media.process_tts_for_text", new_callable=AsyncMock) as mock_tts:

            mock_prefs.return_value = {"bot_mode": "auto", "gender": "female", "speed": 1.0}
            await on_text(mock_update, mock_context)
            mock_ai.assert_awaited_once_with(mock_update, mock_context, "តើ AI គឺជាអ្វី?", 55555)
            mock_tts.assert_not_awaited()

    async def test_on_text_routes_to_tts_for_statement(self) -> None:
        from app.services.telegram.media import on_text

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 66666
        mock_user.username = "test_user2"
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.text = "ព្រះរាជាណាចក្រកម្ពុជា"
        mock_update.message = mock_msg
        mock_context = MagicMock()
        mock_context.user_data = {}

        with patch("app.legacy._is_admin", return_value=False), \
             patch("app.legacy._get_admin_for_user", return_value=None), \
             patch("app.legacy._ensure_user_allowed", new_callable=AsyncMock, return_value=True), \
             patch("app.legacy._check_cooldown", new_callable=AsyncMock, return_value=False), \
             patch("app.legacy.get_user_prefs_async", new_callable=AsyncMock) as mock_prefs, \
             patch("app.legacy._reserve_tts_request", return_value=True), \
             patch("app.services.telegram.media.process_ai_chat_for_text", new_callable=AsyncMock) as mock_ai, \
             patch("app.services.telegram.media.process_tts_for_text", new_callable=AsyncMock) as mock_tts:

            mock_prefs.return_value = {"bot_mode": "auto", "gender": "female", "speed": 1.0}
            await on_text(mock_update, mock_context)
            mock_tts.assert_awaited_once_with(mock_update, mock_context, "ព្រះរាជាណាចក្រកម្ពុជា", 66666)
            mock_ai.assert_not_awaited()

    async def test_process_ai_chat_for_text_is_text_only(self) -> None:
        from app.services.telegram.media import process_ai_chat_for_text

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 77777
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.reply_chat_action = AsyncMock()
        mock_status = MagicMock()
        mock_status.delete = AsyncMock()
        mock_msg.reply_text = AsyncMock(return_value=mock_status)
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        mock_ai_resp = MagicMock()

        with patch("app.legacy._gemini", MagicMock()), \
             patch("app.legacy._reserve_tts_request", return_value=True), \
             patch("app.legacy._release_tts_request") as mock_release, \
             patch("app.services.ai.gemini.generate_content_with_fallback", return_value=mock_ai_resp), \
             patch("app.services.ai.gemini.extract_gemini_text", return_value="AI គឺជាបញ្ញាសិប្បនិម្មិត"), \
             patch("app.services.telegram.formatters.send_split_html", new_callable=AsyncMock) as mock_send_html, \
             patch("app.services.telegram.media.process_tts_for_text", new_callable=AsyncMock) as mock_tts:

            await process_ai_chat_for_text(mock_update, mock_context, "តើ AI ជាអ្វី?", 77777)

            mock_send_html.assert_awaited_once()
            call_html = mock_send_html.call_args[0][1]
            self.assertIn("🤖 <b>AI:</b>", call_html)
            self.assertIn("AI គឺជាបញ្ញាសិប្បនិម្មិត", call_html)
            mock_tts.assert_not_awaited()
            mock_release.assert_called_once_with(77777)
            mock_context.bot.send_chat_action.assert_not_called()

    async def test_cmd_ask_is_text_only(self) -> None:
        from app.services.telegram.commands import cmd_ask

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 88888
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.text = "/ask តើអ្វីទៅជា AI?"
        mock_msg.reply_chat_action = AsyncMock()
        mock_status = MagicMock()
        mock_status.delete = AsyncMock()
        mock_msg.reply_text = AsyncMock(return_value=mock_status)
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        mock_ai_resp = MagicMock()

        with patch("app.legacy._gemini", MagicMock()), \
             patch("app.legacy._check_cooldown", new_callable=AsyncMock, return_value=False), \
             patch("app.legacy._reserve_tts_request", return_value=True), \
             patch("app.legacy._release_tts_request") as mock_release, \
             patch("app.services.ai.gemini.generate_content_with_fallback", return_value=mock_ai_resp), \
             patch("app.services.ai.gemini.extract_gemini_text", return_value="ចម្លើយ AI អំពីបច្ចេកវិទ្យា"), \
             patch("app.services.telegram.formatters.send_split_html", new_callable=AsyncMock) as mock_send_html, \
             patch("app.services.telegram.media.process_tts_for_text", new_callable=AsyncMock) as mock_tts:

            await cmd_ask(mock_update, mock_context)

            mock_send_html.assert_awaited_once()
            call_html = mock_send_html.call_args[0][1]
            self.assertIn("🤖 <b>AI:</b>", call_html)
            self.assertIn("ចម្លើយ AI អំពីបច្ចេកវិទ្យា", call_html)
            mock_tts.assert_not_awaited()
            mock_release.assert_called_once_with(88888)
            mock_context.bot.send_chat_action.assert_not_called()

    async def test_cmd_speed_view_and_set(self) -> None:
        from app.services.telegram.commands import cmd_speed

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 55555
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = []

        with patch("app.legacy.get_user_prefs_async", new_callable=AsyncMock, return_value={"speed": 1.0}):
            await cmd_speed(mock_update, mock_context)
            mock_msg.reply_text.assert_awaited_once()
            self.assertIn("1.0x", mock_msg.reply_text.call_args[0][0])

        mock_msg.reply_text.reset_mock()
        mock_context.args = ["1.25"]

        with patch("app.legacy.update_user_speed") as mock_set:
            await cmd_speed(mock_update, mock_context)
            mock_set.assert_called_once_with(55555, 1.25)
            mock_msg.reply_text.assert_awaited_once()
            self.assertIn("1.25x", mock_msg.reply_text.call_args[0][0])

    async def test_cmd_voice_toggle_and_set(self) -> None:
        from app.services.telegram.commands import cmd_voice

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 66666
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()
        mock_context.args = []

        with patch("app.legacy.get_user_prefs_async", new_callable=AsyncMock, return_value={"gender": "female"}), \
             patch("app.legacy.update_user_gender") as mock_set:
            await cmd_voice(mock_update, mock_context)
            mock_set.assert_called_once_with(66666, "male")
            mock_msg.reply_text.assert_awaited_once()
            self.assertIn("សំឡេងប្រុស", mock_msg.reply_text.call_args[0][0])

        mock_msg.reply_text.reset_mock()
        mock_context.args = ["female"]

        with patch("app.legacy.get_user_prefs_async", new_callable=AsyncMock, return_value={"gender": "male"}), \
             patch("app.legacy.update_user_gender") as mock_set:
            await cmd_voice(mock_update, mock_context)
            mock_set.assert_called_once_with(66666, "female")
            mock_msg.reply_text.assert_awaited_once()
            self.assertIn("សំឡេងស្រី", mock_msg.reply_text.call_args[0][0])

    async def test_on_photo_with_caption_does_not_raise_nameerror(self) -> None:
        from app.services.telegram.media import on_photo

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 99999
        mock_user.username = "testuser"
        mock_update.effective_user = mock_user

        mock_msg = MagicMock()
        mock_msg.chat_id = 99999
        mock_msg.message_id = 1234
        mock_msg.caption = "What is in this picture?"
        mock_photo_size = MagicMock()
        mock_file = MagicMock()
        mock_file.download_to_drive = AsyncMock()
        mock_photo_size.get_file = AsyncMock(return_value=mock_file)
        mock_msg.photo = [mock_photo_size]
        mock_msg.reply_chat_action = AsyncMock()

        mock_progress_msg = MagicMock()
        mock_progress_msg.message_id = 5678
        mock_progress_msg.edit_text = AsyncMock()
        mock_progress_msg.edit_reply_markup = AsyncMock()
        mock_msg.reply_text = AsyncMock(return_value=mock_progress_msg)
        mock_update.message = mock_msg
        mock_context = MagicMock()
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)
        mock_context.bot.send_chat_action = AsyncMock()

        with patch("app.legacy._ensure_user_allowed", new_callable=AsyncMock, return_value=True), \
             patch("app.legacy._ocr_configured", return_value=True), \
             patch("app.legacy._check_cooldown", new_callable=AsyncMock, return_value=False), \
             patch("app.legacy._make_temp_img", return_value="fake_temp.jpg"), \
             patch("app.legacy.sync_user_data", return_value=None), \
             patch("app.services.telegram.media.run_telegram_workload", new_callable=AsyncMock, return_value="This is a Cambodian landmark"), \
             patch("app.legacy._detect_image_mime", return_value="image/jpeg"), \
             patch("app.legacy._cleanup", return_value=None), \
             patch("app.legacy.save_text_cache", return_value=None), \
             patch("app.legacy.record_turn", return_value=None):
            await on_photo(mock_update, mock_context)
            mock_progress_msg.edit_reply_markup.assert_awaited()

    def test_get_quick_reply_keyboard(self) -> None:
        from app.services.telegram.menu import (
            MENU_BTN_AI,
            MENU_BTN_HELP,
            MENU_BTN_PODCAST,
            MENU_BTN_SETTINGS,
            MENU_BTN_TIKTOK,
            MENU_BTN_TTS,
            get_quick_reply_keyboard,
        )

        kb = get_quick_reply_keyboard()
        self.assertIsNotNone(kb)
        flat_buttons = [getattr(btn, "text", str(btn)) for row in kb.keyboard for btn in row]
        self.assertIn(MENU_BTN_TTS, flat_buttons)
        self.assertIn(MENU_BTN_AI, flat_buttons)
        self.assertIn(MENU_BTN_TIKTOK, flat_buttons)
        self.assertIn(MENU_BTN_PODCAST, flat_buttons)
        self.assertIn(MENU_BTN_SETTINGS, flat_buttons)
        self.assertIn(MENU_BTN_HELP, flat_buttons)

    async def test_menu_button_guides_and_cmd_menu(self) -> None:
        from app.services.telegram.menu import cmd_menu, send_ai_chat_quick_guide, send_tts_quick_guide

        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()

        with patch("app.legacy.get_user_prefs_async", new_callable=AsyncMock, return_value={"gender": "female", "speed": 1.0, "tts_model": "auto", "bot_mode": "auto"}):
            await send_tts_quick_guide(mock_msg, 12345)
            mock_msg.reply_text.assert_awaited_once()
            self.assertIn("បម្លែងអត្ថបទទៅជាសំឡេង", mock_msg.reply_text.call_args[0][0])

            mock_msg.reply_text.reset_mock()
            await send_ai_chat_quick_guide(mock_msg, 12345)
            mock_msg.reply_text.assert_awaited_once()
            self.assertIn("សួរឆ្លើយជាមួយ AI Assistant", mock_msg.reply_text.call_args[0][0])

        mock_update = MagicMock()
        mock_update.effective_message = mock_msg
        mock_msg.reply_text.reset_mock()
        await cmd_menu(mock_update, MagicMock())
        mock_msg.reply_text.assert_awaited_once()
        self.assertIn("ម៉ឺនុយរហ័ស", mock_msg.reply_text.call_args[0][0])

    async def test_on_text_dispatches_menu_buttons(self) -> None:
        from app.services.telegram.media import on_text

        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 55555
        mock_update.effective_user = mock_user
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.message = mock_msg
        mock_context = MagicMock()
        mock_context.user_data = {}

        with patch("app.legacy._is_admin", return_value=False), \
             patch("app.legacy._get_admin_for_user", return_value=None), \
             patch("app.legacy._ensure_user_allowed", new_callable=AsyncMock, return_value=True), \
             patch("app.services.telegram.menu.send_tts_quick_guide", new_callable=AsyncMock) as mock_tts_guide, \
             patch("app.services.telegram.menu.send_ai_chat_quick_guide", new_callable=AsyncMock) as mock_ai_guide:

            mock_msg.text = "🎙️ បម្លែងសំឡេង (TTS)"
            await on_text(mock_update, mock_context)
            mock_tts_guide.assert_awaited_once_with(mock_msg, 55555)

            mock_msg.text = "🤖 សួរឆ្លើយ AI"
            await on_text(mock_update, mock_context)
            mock_ai_guide.assert_awaited_once_with(mock_msg, 55555)


if __name__ == "__main__":
    unittest.main()
