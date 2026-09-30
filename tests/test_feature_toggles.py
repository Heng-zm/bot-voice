"""Unit tests for Feature Toggles architecture."""

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

if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "InlineKeyboardButton"):
    try:
        import telegram
    except (ImportError, ModuleNotFoundError):
        import types

        class _DynamicMockModule(types.ModuleType):
            def __getattr__(self, name: str):
                mock = MagicMock(name=name)
                setattr(self, name, mock)
                return mock

        telegram_mod = _DynamicMockModule("telegram")
        class InlineKeyboardButton:
            def __init__(self, text: str = "", callback_data: str | None = None, url: str | None = None, **kwargs):
                self.text = text
                self.callback_data = callback_data
                self.url = url

        class InlineKeyboardMarkup:
            def __init__(self, inline_keyboard: list[list] | None = None, **kwargs):
                self.inline_keyboard = inline_keyboard or []

        telegram_mod.InlineKeyboardButton = InlineKeyboardButton
        telegram_mod.InlineKeyboardMarkup = InlineKeyboardMarkup
        sys.modules["telegram"] = telegram_mod

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
        class MockFastAPI:
            def __init__(self, *args, **kwargs):
                pass
            def include_router(self, *args, **kwargs):
                pass
            def middleware(self, *args, **kwargs):
                return lambda f: f
            def get(self, *args, **kwargs):
                return lambda f: f
            def post(self, *args, **kwargs):
                return lambda f: f
            def add_middleware(self, *args, **kwargs):
                pass

        class MockAPIRouter:
            def __init__(self, *args, **kwargs):
                pass
            def get(self, *args, **kwargs):
                return lambda f: f
            def post(self, *args, **kwargs):
                return lambda f: f
            def head(self, *args, **kwargs):
                return lambda f: f
            def put(self, *args, **kwargs):
                return lambda f: f
            def delete(self, *args, **kwargs):
                return lambda f: f
            def patch(self, *args, **kwargs):
                return lambda f: f
            def options(self, *args, **kwargs):
                return lambda f: f
            def include_router(self, *args, **kwargs):
                pass

        fastapi_mod.FastAPI = MockFastAPI
        fastapi_mod.APIRouter = MockAPIRouter
        fastapi_mod.HTTPException = HTTPException
        fastapi_mod.Header = lambda default=None, **kw: default
        fastapi_mod.Query = lambda default=None, **kw: default
        fastapi_mod.Request = type("Request", (), {})
        fastapi_mod.Depends = lambda default=None, **kw: default
        fastapi_mod.Body = lambda default=None, **kw: default
        fastapi_mod.responses = fastapi_responses
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.core.features import (
    FEATURE_AI_CHAT,
    FEATURE_ARTICLE_READER,
    FEATURE_AUDIO_TRANSCRIPTION,
    FEATURE_CHANNEL_NARRATOR,
    FEATURE_DONATION,
    FEATURE_OCR,
    FEATURE_PODCAST,
    FEATURE_TIKTOK,
    FEATURE_TTS,
    get_feature_flags,
    is_ai_chat_enabled,
    is_article_reader_enabled,
    is_audio_transcription_enabled,
    is_channel_narrator_enabled,
    is_donation_enabled,
    is_feature_enabled,
    is_ocr_enabled,
    is_podcast_enabled,
    is_tiktok_enabled,
    is_tts_enabled,
    reset_feature_overrides,
    set_feature_override,
)


class TestFeatureToggles(unittest.IsolatedAsyncioTestCase):
    """Test suite for feature toggles behavior, dynamic UI, and handler guards."""

    def setUp(self) -> None:
        reset_feature_overrides()

    def tearDown(self) -> None:
        reset_feature_overrides()

    def test_default_features_all_enabled(self) -> None:
        """Verify that out of the box, all canonical features default to True."""
        flags = get_feature_flags()
        for name, state in flags.items():
            self.assertTrue(state, f"Feature {name} should default to True")

        self.assertTrue(is_tiktok_enabled())
        self.assertTrue(is_podcast_enabled())
        self.assertTrue(is_donation_enabled())
        self.assertTrue(is_channel_narrator_enabled())
        self.assertTrue(is_article_reader_enabled())
        self.assertTrue(is_ocr_enabled())
        self.assertTrue(is_audio_transcription_enabled())
        self.assertTrue(is_ai_chat_enabled())
        self.assertTrue(is_tts_enabled())

    def test_runtime_overrides(self) -> None:
        """Verify setting and resetting runtime feature overrides."""
        set_feature_override("tiktok", False)
        self.assertFalse(is_tiktok_enabled())

        set_feature_override("tiktok", True)
        self.assertTrue(is_tiktok_enabled())

        set_feature_override("tiktok", None)  # Clear override
        self.assertTrue(is_tiktok_enabled())

    def test_env_var_truthy_falsy_parsing(self) -> None:
        """Verify env var toggles with various string representations."""
        falsy_values = ("false", "0", "no", "off", "disable", "disabled", "False")
        for val in falsy_values:
            with patch.dict(os.environ, {"ENABLE_TIKTOK": val}):
                self.assertFalse(is_tiktok_enabled(), f"Value '{val}' should evaluate to False")

        truthy_values = ("true", "1", "yes", "on", "enable", "enabled", "True")
        for val in truthy_values:
            with patch.dict(os.environ, {"ENABLE_TIKTOK": val}):
                self.assertTrue(is_tiktok_enabled(), f"Value '{val}' should evaluate to True")

    def test_env_var_alias_resolution(self) -> None:
        """Verify that both ENABLE_<NAME> and <NAME>_ENABLED env vars work."""
        with patch.dict(os.environ, {"PODCAST_ENABLED": "false"}):
            self.assertFalse(is_podcast_enabled())

        with patch.dict(os.environ, {"ENABLE_DONATION": "false"}):
            self.assertFalse(is_donation_enabled())

    def test_dynamic_reply_keyboard_full(self) -> None:
        """Verify full 3-row keyboard when all features are enabled."""
        from app.services.telegram.menu import get_quick_reply_keyboard

        kb = get_quick_reply_keyboard()
        self.assertEqual(len(kb.keyboard), 3)
        # Row 1: TTS, AI
        self.assertEqual(len(kb.keyboard[0]), 2)
        # Row 2: TikTok, Podcast
        self.assertEqual(len(kb.keyboard[1]), 2)
        # Row 3: Settings, Help
        self.assertEqual(len(kb.keyboard[2]), 2)

    def test_dynamic_reply_keyboard_omits_disabled_features(self) -> None:
        """Verify that when TikTok and Podcast are disabled, Row 2 is omitted entirely."""
        from app.services.telegram.menu import (
            MENU_BTN_AI,
            MENU_BTN_HELP,
            MENU_BTN_PODCAST,
            MENU_BTN_SETTINGS,
            MENU_BTN_TIKTOK,
            MENU_BTN_TTS,
            get_quick_reply_keyboard,
        )

        set_feature_override("tiktok", False)
        set_feature_override("podcast", False)

        kb = get_quick_reply_keyboard()
        # Row 2 should be omitted, yielding 2 rows total!
        self.assertEqual(len(kb.keyboard), 2)

        flattened_buttons = [btn.text for row in kb.keyboard for btn in row]
        self.assertNotIn(MENU_BTN_TIKTOK, flattened_buttons)
        self.assertNotIn(MENU_BTN_PODCAST, flattened_buttons)
        self.assertIn(MENU_BTN_TTS, flattened_buttons)
        self.assertIn(MENU_BTN_AI, flattened_buttons)
        self.assertIn(MENU_BTN_SETTINGS, flattened_buttons)
        self.assertIn(MENU_BTN_HELP, flattened_buttons)

    async def test_cmd_tiktok_disabled_guard(self) -> None:
        """Verify that /tiktok politely informs user when feature is disabled."""
        from app.services.downloader.tiktok import cmd_tiktok

        set_feature_override("tiktok", False)

        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        await cmd_tiktok(mock_update, mock_context)
        mock_msg.reply_text.assert_awaited_once()
        text = mock_msg.reply_text.call_args[0][0]
        self.assertIn("បិទដំណើរការ", text)

    async def test_tiktok_callback_disabled_guard(self) -> None:
        """Verify that TikTok inline buttons answer query when feature is disabled."""
        from app.services.downloader.tiktok import tiktok_callback

        set_feature_override("tiktok", False)

        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_query.data = "tt_mp3:123"
        mock_update.callback_query = mock_query
        mock_context = MagicMock()

        await tiktok_callback(mock_update, mock_context)
        mock_query.answer.assert_awaited_once()
        alert_text = mock_query.answer.call_args[0][0]
        self.assertIn("បិទដំណើរការ", alert_text)

    async def test_cmd_podcast_disabled_guard(self) -> None:
        """Verify that /podcast informs user when feature is disabled."""
        from app.services.podcast.handlers import cmd_podcast

        set_feature_override("podcast", False)

        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_context = MagicMock()

        await cmd_podcast(mock_update, mock_context)
        mock_msg.reply_text.assert_awaited_once()
        text = mock_msg.reply_text.call_args[0][0]
        self.assertIn("បិទដំណើរការ", text)

    async def test_podcast_callback_disabled_guard(self) -> None:
        """Verify that podcast inline buttons answer query when feature is disabled."""
        from app.services.podcast.handlers import podcast_callback

        set_feature_override("podcast", False)

        mock_update = MagicMock()
        mock_query = MagicMock()
        mock_query.answer = AsyncMock()
        mock_query.data = "podcast_refresh"
        mock_update.callback_query = mock_query
        mock_context = MagicMock()

        await podcast_callback(mock_update, mock_context)
        mock_query.answer.assert_awaited_once()
        alert_text = mock_query.answer.call_args[0][0]
        self.assertIn("បិទដំណើរការ", alert_text)

    async def test_cmd_donate_disabled_guard(self) -> None:
        """Verify that /donate informs user when feature is disabled."""
        from app.services.donation.handlers import cmd_donate

        set_feature_override("donation", False)

        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.effective_user = MagicMock()
        mock_context = MagicMock()

        await cmd_donate(mock_update, mock_context)
        mock_msg.reply_text.assert_awaited_once()
        text = mock_msg.reply_text.call_args[0][0]
        self.assertIn("បិទដំណើរការ", text)

    async def test_cmd_ask_disabled_guard(self) -> None:
        """Verify that /ask informs user when AI Chat is disabled."""
        from app.services.telegram.commands import cmd_ask

        set_feature_override("ai_chat", False)

        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.text = "/ask What is AI?"
        mock_msg.reply_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.effective_user = MagicMock(id=12345)
        mock_context = MagicMock()

        await cmd_ask(mock_update, mock_context)
        mock_msg.reply_text.assert_awaited_once()
        text = mock_msg.reply_text.call_args[0][0]
        self.assertIn("បិទដំណើរការ", text)

    async def test_on_help_dynamic_keyboard_and_text(self) -> None:
        """Verify that on_help dynamically removes disabled features from text and inline keyboard."""
        from app.services.telegram.commands import on_help

        set_feature_override("tiktok", False)
        set_feature_override("podcast", False)
        set_feature_override("donation", False)

        mock_update = MagicMock()
        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock()
        mock_msg.edit_text = AsyncMock()
        mock_update.effective_message = mock_msg
        mock_update.callback_query = None
        mock_context = MagicMock()

        await on_help(mock_update, mock_context)
        mock_msg.reply_text.assert_awaited_once()

        help_text = mock_msg.reply_text.call_args[0][0]
        self.assertNotIn("ទាញយក TikTok", help_text)
        self.assertNotIn("/podcast", help_text)
        self.assertNotIn("Bakong KHQR", help_text)

        kb = mock_msg.reply_text.call_args[1]["reply_markup"]
        all_cb_data = [btn.callback_data for row in kb.inline_keyboard for btn in row if hasattr(btn, "callback_data")]
        self.assertNotIn("show_tiktok_guide", all_cb_data)
        self.assertNotIn("podcast_refresh", all_cb_data)
        self.assertNotIn("donate_menu", all_cb_data)


if __name__ == "__main__":
    unittest.main()
