"""Unit tests for Full Option Admin Bot Controller."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import types
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

if "fastapi" not in sys.modules:
    try:
        import fastapi  # noqa: F401
    except ImportError:
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
        fastapi_mod.Query = lambda default=None, **kw: default
        fastapi_mod.Request = type("Request", (), {})
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.services.admin.dashboard import (
    build_admin_bakong_text,
    build_admin_bot_mode_text,
    build_admin_home_full_text,
    build_admin_podcast_text,
    build_admin_quick_actions_text,
    build_admin_ui_hub_text,
    get_admin_bakong_kb,
    get_admin_bot_mode_kb,
    get_admin_dashboard_full_kb,
    get_admin_podcast_kb,
    get_admin_quick_actions_kb,
    get_admin_ui_hub_kb,
)
from app.services.admin.handlers import handle_admin_callback
from app.legacy import get_admin_dashboard_kb


class TestAdminDashboardKeyboards(unittest.TestCase):
    """Test inline keyboards for the Full Option Admin Bot Controller."""

    def test_admin_dashboard_full_keyboard_structure(self) -> None:
        kb = get_admin_dashboard_full_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]

        # Verify preserved legacy buttons
        self.assertIn("admin_bot_config", callbacks)
        self.assertIn("admin_broadcast", callbacks)
        self.assertIn("admin_users", callbacks)
        self.assertIn("admin_schedules", callbacks)
        self.assertIn("admin_user_needs", callbacks)
        self.assertIn("admin_report", callbacks)
        self.assertIn("admin_health", callbacks)
        self.assertIn("admin_errors", callbacks)
        self.assertIn("admin_db", callbacks)
        self.assertIn("admin_optimize", callbacks)
        self.assertIn("admin_stats", callbacks)
        self.assertIn("admin_home", callbacks)
        self.assertIn("admin_close", callbacks)

        # Verify newly added full-option buttons
        self.assertIn("admin_bot_mode", callbacks)
        self.assertIn("admin_podcast", callbacks)
        self.assertIn("admin_bakong", callbacks)
        self.assertIn("admin_ui_hub", callbacks)

    def test_legacy_get_admin_dashboard_kb_delegation(self) -> None:
        kb = get_admin_dashboard_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("admin_db", callbacks)
        self.assertIn("admin_podcast", callbacks)
        self.assertIn("admin_bakong", callbacks)
        self.assertIn("admin_bot_mode", callbacks)

    def test_admin_podcast_keyboard(self) -> None:
        kb = get_admin_podcast_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("admin_podcast_broadcast", callbacks)
        self.assertIn("admin_podcast_test", callbacks)
        self.assertIn("admin_podcast_refresh", callbacks)
        self.assertIn("admin_podcast_subs", callbacks)
        self.assertIn("admin_home", callbacks)
        self.assertIn("admin_close", callbacks)

    def test_admin_bakong_keyboard(self) -> None:
        kb = get_admin_bakong_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("admin_bakong_ping", callbacks)
        self.assertIn("admin_bakong_preview_qr", callbacks)
        self.assertIn("admin_bakong_pending", callbacks)
        self.assertIn("admin_bakong_donors", callbacks)
        self.assertIn("admin_bakong_config", callbacks)
        self.assertIn("admin_home", callbacks)
        self.assertIn("admin_close", callbacks)

    def test_admin_bot_mode_keyboard(self) -> None:
        kb_auto = get_admin_bot_mode_kb("auto")
        labels_auto = [btn.text for row in kb_auto.inline_keyboard for btn in row]
        self.assertTrue(any("✅" in l and "Auto" in l for l in labels_auto))

        kb_tts = get_admin_bot_mode_kb("tts")
        labels_tts = [btn.text for row in kb_tts.inline_keyboard for btn in row]
        self.assertTrue(any("✅" in l and "TTS" in l for l in labels_tts))

    def test_admin_ui_hub_keyboard(self) -> None:
        kb = get_admin_ui_hub_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("admin_welcome", callbacks)
        self.assertIn("admin_welcome_set_photo", callbacks)
        self.assertIn("admin_btn_editor", callbacks)
        self.assertIn("admin_tts_models", callbacks)

    def test_admin_quick_actions_keyboard(self) -> None:
        kb = get_admin_quick_actions_kb()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        self.assertIn("admin_quick_optimize", callbacks)
        self.assertIn("admin_quick_flush_cache", callbacks)
        self.assertIn("admin_quick_clear_errors", callbacks)


class TestAdminDashboardTexts(unittest.TestCase):
    """Test text rendering functions for administrative panels."""

    def test_build_admin_podcast_text(self) -> None:
        text = build_admin_podcast_text()
        self.assertIn("Daily Morning Podcast", text)
        self.assertIn("Subscribers", text)
        self.assertIn("07:00", text)

    def test_build_admin_bakong_text(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            text = loop.run_until_complete(build_admin_bakong_text())
            self.assertIn("Bakong KHQR", text)
            self.assertIn("Merchant Name", text)
        finally:
            loop.close()

    def test_build_admin_bot_mode_text(self) -> None:
        text_auto = build_admin_bot_mode_text("auto")
        self.assertIn("AUTO", text_auto)
        self.assertIn("Smart Auto-Detect", text_auto)

        text_tts = build_admin_bot_mode_text("tts")
        self.assertIn("TTS", text_tts)

        text_ai = build_admin_bot_mode_text("ai_chat")
        self.assertIn("AI_CHAT", text_ai)

    def test_build_admin_ui_hub_text(self) -> None:
        text = asyncio.run(build_admin_ui_hub_text())
        self.assertIn("UI", text)
        self.assertIn("Welcome", text)

    def test_build_admin_quick_actions_text(self) -> None:
        text = build_admin_quick_actions_text()
        self.assertIn("Quick Operations", text)
        self.assertIn("Optimize", text)


class TestAdminCallbackHandler(unittest.TestCase):
    """Test callback handling in handle_admin_callback."""

    def setUp(self) -> None:
        self.query = MagicMock()
        self.query.message = MagicMock()
        self.query.message.reply_text = AsyncMock()
        self.query.message.edit_text = AsyncMock()
        self.query.message.reply_photo = AsyncMock()
        self.query.answer = AsyncMock()
        self.context = MagicMock()
        self.context.bot = MagicMock()
        self.context.bot.send_message = AsyncMock()

    def test_handle_callback_denies_non_admin(self) -> None:
        with patch("app.legacy._is_admin", return_value=False):
            handled = asyncio.run(handle_admin_callback(self.query, 999999, self.context, "admin_podcast"))
            self.assertTrue(handled)
            self.query.message.reply_text.assert_called_once()
            self.assertIn("Admin only", self.query.message.reply_text.call_args[0][0])

    def test_handle_callback_podcast_menu(self) -> None:
        with patch("app.legacy._is_admin", return_value=True):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_podcast"))
            self.assertTrue(handled)
            self.query.message.edit_text.assert_called_once()
            args, kwargs = self.query.message.edit_text.call_args
            self.assertIn("Daily Morning Podcast", args[0])

    def test_handle_callback_bakong_menu(self) -> None:
        with patch("app.legacy._is_admin", return_value=True):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_bakong"))
            self.assertTrue(handled)
            self.query.message.edit_text.assert_called_once()
            args, kwargs = self.query.message.edit_text.call_args
            self.assertIn("Bakong KHQR", args[0])

    def test_handle_callback_bakong_preview_qr(self) -> None:
        mock_qr_bytes = b"\x89PNG\r\n\x1a\ntest_qr_image"
        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.donation.khqr.get_khqr_qr_image", new_callable=AsyncMock, return_value=mock_qr_bytes):
            self.query.message.reply_photo = AsyncMock()
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_bakong_preview_qr"))
            self.assertTrue(handled)
            self.query.message.reply_photo.assert_awaited_once()
            caption = self.query.message.reply_photo.call_args[1]["caption"]
            self.assertIn("Bakong KHQR Preview", caption)

    def test_handle_callback_bot_mode_menu(self) -> None:
        with patch("app.legacy._is_admin", return_value=True):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_bot_mode"))
            self.assertTrue(handled)
            self.query.message.edit_text.assert_called_once()
            args, kwargs = self.query.message.edit_text.call_args
            self.assertIn("Bot Mode", args[0])

    def test_handle_callback_mode_set(self) -> None:
        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.legacy.db_bot_setting_value_set", return_value=(True, "OK")):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_mode_set:tts"))
            self.assertTrue(handled)
            self.query.message.edit_text.assert_called_once()
            args, kwargs = self.query.message.edit_text.call_args
            self.assertIn("TTS", args[0])

    def test_handle_callback_quick_actions(self) -> None:
        with patch("app.legacy._is_admin", return_value=True):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_quick_actions"))
            self.assertTrue(handled)
            self.query.message.edit_text.assert_called_once()
            args, kwargs = self.query.message.edit_text.call_args
            self.assertIn("Quick Operations", args[0])

    def test_handle_callback_podcast_test(self) -> None:
        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.podcast.generator.generate_morning_podcast", new_callable=AsyncMock, return_value=("<b>News</b>", "Speech News")), \
             patch("app.services.podcast.handlers.send_podcast_card", new_callable=AsyncMock) as mock_send_card, \
             patch("app.services.podcast.handlers.send_podcast_voice", new_callable=AsyncMock) as mock_send_voice:
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_podcast_test"))
            self.assertTrue(handled)
            mock_send_card.assert_awaited_once()
            mock_send_voice.assert_awaited_once()
            self.assertIn("Admin Test Preview", mock_send_card.call_args[0][1])

    def test_handle_callback_podcast_refresh(self) -> None:
        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.podcast.handlers.fetch_source_banner_bytes", new_callable=AsyncMock, return_value=None), \
             patch("app.services.podcast.generator.generate_morning_podcast", new_callable=AsyncMock, return_value=("<b>News</b>", "Speech News")) as mock_gen, \
             patch("app.services.podcast.handlers.get_or_synthesize_podcast_voice", new_callable=AsyncMock, return_value=b"fake_voice") as mock_synth:
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_podcast_refresh"))
            self.assertTrue(handled)
            mock_gen.assert_awaited_once_with(force_refresh=True)
            mock_synth.assert_awaited_once_with("Speech News", force_refresh=True)
            self.query.message.edit_text.assert_called_once()
            self.assertIn("Refresh", self.query.message.edit_text.call_args[0][0])

    def test_handle_callback_quick_optimize(self) -> None:
        mock_stats = {
            "temp_files_swept": 5,
            "gc_objects_freed": 120,
            "audio_cache_trimmed": 2,
            "db_history_pruned": 10,
            "perf_knobs_applied": ["USER_RATE_LIMIT_PER_SECOND", "TTS_MAX_TEXT_LEN"],
        }
        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.admin.optimization.run_system_optimization_async", new_callable=AsyncMock, return_value=mock_stats):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_quick_optimize"))
            self.assertTrue(handled)
            self.query.message.edit_text.assert_called_once()
            text = self.query.message.edit_text.call_args[0][0]
            self.assertIn("Optimize", text)
            self.assertIn("Temp Files", text)

    def test_handle_callback_quick_flush_cache(self) -> None:
        with patch("app.legacy._is_admin", return_value=True), \
             patch("app.services.tts.cache.clear_all_tts_caches", return_value={"audio_items_cleared": 3, "file_ids_cleared": 4}), \
             patch("app.services.ai.gemini.clear_gemini_response_cache", return_value=5):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "admin_quick_flush_cache"))
            self.assertTrue(handled)
            self.query.message.edit_text.assert_called_once()
            text = self.query.message.edit_text.call_args[0][0]
            self.assertIn("សម្អាត Caches", text)

    def test_run_system_optimization_service(self) -> None:
        from app.services.admin.optimization import run_system_cleanup_sync, run_system_optimization_async

        with patch("app.services.admin.optimization.sweep_stale_temp_files", return_value=3), \
             patch("app.services.tts.cache.get_tts_cache") as mock_get_tts, \
             patch("app.services.ai.gemini.clear_gemini_response_cache", return_value=2), \
             patch("app.legacy.db_run_periodic_pruning", return_value={"pruned_history": 1, "pruned_text_cache": 2}), \
             patch("app.legacy._apply_all_bot_performance_settings", new_callable=AsyncMock, return_value=["knob1"]):
            mock_tts_inst = MagicMock()
            mock_tts_inst.trim_expired.return_value = 4
            mock_get_tts.return_value = mock_tts_inst

            sync_res = run_system_cleanup_sync(prune_db=True)
            self.assertEqual(3, sync_res["temp_files_swept"])
            self.assertEqual(4, sync_res["audio_cache_trimmed"])
            self.assertEqual(2, sync_res["gemini_cache_cleared"])
            self.assertEqual(1, sync_res["db_history_pruned"])

            async_res = asyncio.run(run_system_optimization_async(admin_id=123, prune_db=True))
            self.assertIn("knob1", async_res["perf_knobs_applied"])
            self.assertEqual(3, async_res["temp_files_swept"])

    def test_periodic_system_maintenance_loop(self) -> None:
        from app.main import periodic_system_maintenance

        call_count = 0

        async def mock_opt(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"temp_files_swept": 1}

        async def run_test():
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
                 patch("app.services.admin.optimization.run_system_optimization_async", side_effect=mock_opt):
                mock_sleep.side_effect = [None, None, asyncio.CancelledError()]
                await periodic_system_maintenance()

        asyncio.run(run_test())
        self.assertGreaterEqual(call_count, 1)

    def test_handle_callback_unknown_returns_false(self) -> None:
        with patch("app.legacy._is_admin", return_value=True):
            handled = asyncio.run(handle_admin_callback(self.query, 12345, self.context, "unknown_custom_action"))
            self.assertFalse(handled)


if __name__ == "__main__":
    unittest.main()
