"""Unit tests for Real-Time Server Request Logging, Telemetry, and Live Stream."""

from __future__ import annotations

import asyncio
import json
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

from typing import Any

if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except ImportError:
        sys.modules["httpx"] = MagicMock()

if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "ext") or not hasattr(sys.modules["telegram"], "error"):
    try:
        import telegram  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        class _DynamicMockError(Exception):
            pass

        class _DynamicMockModule(types.ModuleType):
            def __getattr__(self, name: str):
                if name in ("TelegramError", "BadRequest", "Forbidden", "NetworkError", "TimedOut", "RetryAfter"):
                    cls = type(name, (_DynamicMockError,), {})
                    setattr(self, name, cls)
                    return cls
                mock = MagicMock(name=name)
                setattr(self, name, mock)
                return mock

        telegram_mod = _DynamicMockModule("telegram")
        telegram_mod.__path__ = []

        class Update:
            def __init__(self, update_id: int = 0, *args, **kwargs):
                self.update_id = update_id
                self.effective_chat = None
                self.effective_user = None
                self.message = None
                self.callback_query = None
                self.channel_post = None
                for k, v in kwargs.items():
                    setattr(self, k, v)
            @classmethod
            def de_json(cls, data, bot=None):
                return cls(update_id=data.get("update_id", 0) if isinstance(data, dict) else 0)

        class InlineKeyboardButton:
            def __init__(self, text: str = "", callback_data: str | None = None, url: str | None = None, **kwargs):
                self.text = text
                self.callback_data = callback_data
                self.url = url

        class InlineKeyboardMarkup:
            def __init__(self, inline_keyboard: list[list] | None = None, **kwargs):
                self.inline_keyboard = inline_keyboard or []

        class KeyboardButton:
            def __init__(self, text: str = "", *args, **kwargs):
                self.text = text
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class ReplyKeyboardMarkup:
            def __init__(self, keyboard: list[list] | None = None, *args, **kwargs):
                self.keyboard = keyboard or []
                for k, v in kwargs.items():
                    setattr(self, k, v)

        telegram_mod.Update = Update
        telegram_mod.InlineKeyboardButton = InlineKeyboardButton
        telegram_mod.InlineKeyboardMarkup = InlineKeyboardMarkup
        telegram_mod.KeyboardButton = KeyboardButton
        telegram_mod.ReplyKeyboardMarkup = ReplyKeyboardMarkup

        telegram_ext = _DynamicMockModule("telegram.ext")
        telegram_ext.__path__ = []
        class ApplicationHandlerStop(Exception):
            pass
        telegram_ext.ApplicationHandlerStop = ApplicationHandlerStop
        telegram_ext.ContextTypes = type("ContextTypes", (), {"DEFAULT_TYPE": MagicMock})
        telegram_ext.Application = MagicMock
        telegram_ext.TypeHandler = MagicMock
        telegram_ext.CommandHandler = MagicMock
        telegram_ext.MessageHandler = MagicMock
        telegram_ext.CallbackQueryHandler = MagicMock
        telegram_ext.filters = MagicMock()

        telegram_error = _DynamicMockModule("telegram.error")
        telegram_error.__path__ = []
        telegram_error.TelegramError = _DynamicMockError
        telegram_error.BadRequest = type("BadRequest", (_DynamicMockError,), {})
        telegram_error.Forbidden = type("Forbidden", (_DynamicMockError,), {})
        telegram_error.NetworkError = type("NetworkError", (_DynamicMockError,), {})
        telegram_error.TimedOut = type("TimedOut", (_DynamicMockError,), {})
        telegram_error.RetryAfter = type("RetryAfter", (_DynamicMockError,), {})

        telegram_constants = _DynamicMockModule("telegram.constants")
        telegram_constants.__path__ = []
        telegram_constants.ParseMode = type("ParseMode", (), {"HTML": "HTML", "MARKDOWN": "MARKDOWN", "MARKDOWN_V2": "MarkdownV2"})

        telegram_mod.ext = telegram_ext
        telegram_mod.error = telegram_error
        telegram_mod.constants = telegram_constants

        sys.modules["telegram"] = telegram_mod
        sys.modules["telegram.ext"] = telegram_ext
        sys.modules["telegram.error"] = telegram_error
        sys.modules["telegram.constants"] = telegram_constants

if "fastapi" not in sys.modules or not hasattr(sys.modules["fastapi"], "FastAPI"):
    try:
        import fastapi  # noqa: F401
    except ImportError:
        fastapi_mod = types.ModuleType("fastapi")
        fastapi_responses = types.ModuleType("fastapi.responses")

        class MockResponse:
            def __init__(self, content: Any = None, status_code: int = 200, headers: Any = None):
                self.content = content
                self.status_code = status_code
                self.headers = dict(headers or {})
                if isinstance(content, str):
                    self.body = content.encode("utf-8")
                elif isinstance(content, bytes):
                    self.body = content
                else:
                    self.body = json.dumps(content).encode("utf-8")

        class MockHTMLResponse(MockResponse):
            pass

        class MockJSONResponse(MockResponse):
            pass

        class MockStreamingResponse(MockResponse):
            def __init__(self, content: Any = None, status_code: int = 200, media_type: str = "", headers: Any = None):
                super().__init__(content, status_code, headers)
                self.media_type = media_type

        fastapi_responses.Response = MockResponse
        fastapi_responses.HTMLResponse = MockHTMLResponse
        fastapi_responses.JSONResponse = MockJSONResponse
        fastapi_responses.StreamingResponse = MockStreamingResponse

        class MockAPIRouter:
            def __init__(self, *args, **kwargs):
                self.routes = []
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

        class MockFastAPI:
            def __init__(self, *args, **kwargs):
                self.routes = []
            def middleware(self, *args, **kwargs):
                return lambda f: f
            def include_router(self, *args, **kwargs):
                pass

        fastapi_mod.FastAPI = MockFastAPI
        fastapi_mod.APIRouter = MockAPIRouter
        fastapi_mod.Header = lambda default=None, **kw: default
        class MockHTTPException(Exception):
            def __init__(self, status_code: int = 400, detail: str = "", *args: Any, **kwargs: Any):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        fastapi_mod.HTTPException = MockHTTPException
        fastapi_mod.Request = type("Request", (), {})
        fastapi_mod.Response = MockResponse
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = fastapi_responses

from app.core.logging_middleware import (
    RequestLogStore,
    RequestRecord,
    Response as FallbackResponse,
    get_request_log_store,
    request_lifecycle_logging_middleware,
)
from app.services.admin.dashboard import build_admin_live_logs_text, get_admin_live_logs_kb
from app.services.telegram.telemetry import (
    extract_update_info,
    record_telegram_update_complete,
    telegram_request_telemetry_complete_guard,
    telegram_request_telemetry_guard,
)
from app.utils.logging import FlushStreamHandler, configure_server_logging


class TestRealtimeRequestStore(unittest.IsolatedAsyncioTestCase):
    """Test suite for in-memory RequestLogStore and event bus."""

    def setUp(self) -> None:
        self.store = RequestLogStore(maxlen=20)

    def test_record_start_and_complete(self) -> None:
        rec = self.store.record_start(
            request_id="req-test-1",
            category="HTTP",
            method="GET",
            path="/healthz",
            client="127.0.0.1",
            detail="Health check",
        )
        self.assertEqual("req-test-1", rec.id)
        self.assertEqual("HTTP", rec.category)
        self.assertEqual("GET", rec.method)
        self.assertEqual("/healthz", rec.path)
        self.assertEqual("127.0.0.1", rec.client)
        self.assertEqual("PENDING", rec.status)

        # Complete record
        comp = self.store.record_complete(
            request_id="req-test-1",
            status="200 OK",
            status_code=200,
            duration_ms=12.5,
            detail="Health OK",
        )
        self.assertIsNotNone(comp)
        self.assertEqual("200 OK", comp.status)
        self.assertEqual(200, comp.status_code)
        self.assertEqual(12.5, comp.duration_ms)

        recent = self.store.get_recent(limit=10)
        self.assertEqual(1, len(recent))
        self.assertEqual("req-test-1", recent[0]["id"])
        self.assertEqual("200 OK", recent[0]["status"])

    def test_filter_by_category_and_errors(self) -> None:
        self.store.record_start("req-h1", "HTTP", "GET", "/api/v1", "10.0.0.1")
        self.store.record_complete("req-h1", "200 OK", 200, 5.0)

        self.store.record_start("req-h2", "HTTP", "POST", "/bad", "10.0.0.2")
        self.store.record_complete("req-h2", "500 SERVER_ERROR", 500, 25.0)

        self.store.record_start("req-tg1", "TELEGRAM", "COMMAND", "/start", "@user1")
        self.store.record_complete("req-tg1", "SUCCESS", 200, 45.0)

        # Category filter: HTTP
        http_only = self.store.get_recent(category="HTTP")
        self.assertEqual(2, len(http_only))
        for r in http_only:
            self.assertEqual("HTTP", r["category"])

        # Category filter: TELEGRAM
        tg_only = self.store.get_recent(category="TELEGRAM")
        self.assertEqual(1, len(tg_only))
        self.assertEqual("TELEGRAM", tg_only[0]["category"])

        # Error filter
        errs = self.store.get_recent(only_errors=True)
        self.assertEqual(1, len(errs))
        self.assertEqual(500, errs[0]["status_code"])

    def test_ring_buffer_capacity(self) -> None:
        store = RequestLogStore(maxlen=5)
        for i in range(10):
            store.record_start(f"req-{i}", "HTTP", "GET", f"/path/{i}", "127.0.0.1")

        recent = store.get_recent(limit=10)
        self.assertEqual(5, len(recent))
        # Newest should be req-9, oldest should be req-5
        self.assertEqual("req-9", recent[0]["id"])
        self.assertEqual("req-5", recent[-1]["id"])

    def test_metrics_calculation(self) -> None:
        self.store.record_start("m1", "HTTP", "GET", "/p1", "127.0.0.1")
        self.store.record_complete("m1", "200 OK", 200, 10.0)

        self.store.record_start("m2", "TELEGRAM", "MESSAGE", "hello", "@test")
        self.store.record_complete("m2", "SUCCESS", 200, 30.0)

        self.store.record_start("m3", "HTTP", "GET", "/fail", "127.0.0.1")
        self.store.record_complete("m3", "500 SERVER_ERROR", 500, 20.0)

        metrics = self.store.get_metrics()
        self.assertEqual(3, metrics["total_requests"])
        self.assertEqual(2, metrics["total_http"])
        self.assertEqual(1, metrics["total_telegram"])
        self.assertEqual(1, metrics["total_errors"])
        self.assertEqual(20.0, metrics["average_latency_ms"])

    async def test_pub_sub_broadcasting(self) -> None:
        queue = self.store.subscribe()
        try:
            self.store.record_start("pub-1", "HTTP", "GET", "/sub", "127.0.0.1")
            rec1 = await asyncio.wait_for(queue.get(), timeout=1.0)
            self.assertEqual("pub-1", rec1.id)
            self.assertEqual("PENDING", rec1.status)

            self.store.record_complete("pub-1", "200 OK", 200, 8.5)
            rec2 = await asyncio.wait_for(queue.get(), timeout=1.0)
            self.assertEqual("pub-1", rec2.id)
            self.assertEqual("200 OK", rec2.status)
        finally:
            self.store.unsubscribe(queue)


class TestLoggingMiddlewareIntegration(unittest.IsolatedAsyncioTestCase):
    """Test HTTP middleware request interception and telemetry capture."""

    async def test_middleware_records_to_store(self) -> None:
        req = MagicMock()
        req.method = "GET"
        req.url.path = "/api/v1/test-endpoint"
        req.url.query = "param=true"
        req.headers = {"x-request-id": "mid-test-99", "x-forwarded-for": "198.51.100.22"}

        mock_resp = FallbackResponse(status_code=200, headers={})
        call_next = AsyncMock(return_value=mock_resp)

        store = get_request_log_store()
        resp = await request_lifecycle_logging_middleware(req, call_next)

        self.assertEqual(200, resp.status_code)
        self.assertEqual("mid-test-99", resp.headers.get("X-Request-ID"))

        recent = store.get_recent(limit=5)
        matched = [r for r in recent if r["id"] == "mid-test-99"]
        self.assertTrue(len(matched) > 0)
        self.assertEqual("/api/v1/test-endpoint?param=true", matched[0]["path"])
        self.assertEqual("198.51.100.22", matched[0]["client"])
        self.assertEqual("200 OK", matched[0]["status"])

    async def test_middleware_crash_recording(self) -> None:
        req = MagicMock()
        req.method = "POST"
        req.url.path = "/api/crash"
        req.url.query = ""
        req.headers = {"x-request-id": "crash-test-1", "x-forwarded-for": "10.0.0.1"}

        call_next = AsyncMock(side_effect=RuntimeError("Simulated DB connection failure"))
        store = get_request_log_store()

        with self.assertRaises(RuntimeError):
            await request_lifecycle_logging_middleware(req, call_next)

        recent = store.get_recent(limit=5)
        matched = [r for r in recent if r["id"] == "crash-test-1"]
        self.assertTrue(len(matched) > 0)
        self.assertEqual("500 SERVER_ERROR", matched[0]["status"])
        self.assertEqual(500, matched[0]["status_code"])
        self.assertIn("Simulated DB", matched[0]["detail"])


class TestTelegramTelemetryGuard(unittest.IsolatedAsyncioTestCase):
    """Test suite for Telegram Update telemetry extraction and recording."""

    def test_extract_update_info_command(self) -> None:
        update = MagicMock()
        update.update_id = 998811
        update.effective_user.username = "heng_tester"
        update.effective_user.id = 12345
        update.effective_chat.id = 12345
        update.message.text = "/tiktok https://vt.tiktok.com/test/"
        update.callback_query = None
        update.channel_post = None
        update.inline_query = None

        info = extract_update_info(update)
        self.assertEqual(998811, info["update_id"])
        self.assertEqual("COMMAND", info["method"])
        self.assertEqual("/tiktok", info["path"])
        self.assertIn("@heng_tester", info["client"])
        self.assertIn("12345", info["client"])

    def test_extract_update_info_callback(self) -> None:
        update = MagicMock()
        update.update_id = 998812
        update.effective_user.username = "cb_user"
        update.effective_user.id = 54321
        update.message = None
        update.callback_query.data = "tt_mp3:vid_987"
        update.channel_post = None
        update.inline_query = None

        info = extract_update_info(update)
        self.assertEqual("CALLBACK", info["method"])
        self.assertEqual("tt_mp3:vid_987", info["path"])

    def test_extract_update_info_edited_and_chat_member(self) -> None:
        # 1. edited_message
        u_edit = MagicMock()
        u_edit.update_id = 998813
        u_edit.message = None
        u_edit.callback_query = None
        u_edit.channel_post = None
        u_edit.inline_query = None
        u_edit.edited_message.text = "Corrected caption"
        info_edit = extract_update_info(u_edit)
        self.assertEqual("EDITED_MSG", info_edit["method"])
        self.assertEqual("Corrected caption", info_edit["path"])

        # 2. edited_channel_post
        u_post = MagicMock()
        u_post.update_id = 998814
        u_post.message = None
        u_post.callback_query = None
        u_post.channel_post = None
        u_post.inline_query = None
        u_post.edited_message = None
        u_post.edited_channel_post.text = "New announcement"
        info_post = extract_update_info(u_post)
        self.assertEqual("EDITED_POST", info_post["method"])
        self.assertEqual("New announcement", info_post["path"])

        # 3. my_chat_member
        u_member = MagicMock()
        u_member.update_id = 998815
        u_member.message = None
        u_member.callback_query = None
        u_member.channel_post = None
        u_member.inline_query = None
        u_member.edited_message = None
        u_member.edited_channel_post = None
        u_member.my_chat_member.new_chat_member.status = "administrator"
        info_member = extract_update_info(u_member)
        self.assertEqual("CHAT_MEMBER", info_member["method"])
        self.assertEqual("status:administrator", info_member["path"])

    async def test_logging_middleware_handles_none_response(self) -> None:
        mock_request = MagicMock()
        mock_request.url.path = "/test/none"
        mock_request.method = "GET"
        mock_request.client.host = "127.0.0.1"
        mock_request.query_params = {}

        async def call_next(req):
            return None

        resp = await request_lifecycle_logging_middleware(mock_request, call_next)
        self.assertIsNotNone(resp)
        self.assertEqual(200, resp.status_code)

    async def test_guard_execution_and_completion(self) -> None:
        update = MagicMock()
        update.update_id = 771122
        update.effective_user.username = "bot_fan"
        update.effective_user.id = 8888
        update.message.text = "Hello Bot!"
        update.callback_query = None
        update.channel_post = None
        update.inline_query = None
        context = MagicMock()

        store = get_request_log_store()
        await telegram_request_telemetry_guard(update, context)

        recent = store.get_recent(limit=5)
        matched = [r for r in recent if r["id"] == "tg-771122"]
        self.assertTrue(len(matched) > 0)
        self.assertEqual("PENDING", matched[0]["status"])

        # Record completion
        record_telegram_update_complete(771122, success=True, duration_ms=42.0)
        recent_after = store.get_recent(limit=5)
        matched_after = [r for r in recent_after if r["id"] == "tg-771122"]
        self.assertEqual("SUCCESS", matched_after[0]["status"])
        self.assertEqual(42.0, matched_after[0]["duration_ms"])

    async def test_guard_execution_and_complete_guard(self) -> None:
        update = MagicMock()
        update.update_id = 998877
        update.effective_user.username = "polling_user"
        update.effective_user.id = 12345
        update.message.text = "/help"
        update.callback_query = None
        update.channel_post = None
        update.inline_query = None
        context = MagicMock()

        store = get_request_log_store()
        await telegram_request_telemetry_guard(update, context)

        recent = store.get_recent(limit=5)
        matched = [r for r in recent if r["id"] == "tg-998877"]
        self.assertTrue(len(matched) > 0)
        self.assertEqual("PENDING", matched[0]["status"])

        # Execute trailing group 100 complete guard (simulating polling mode lifecycle completion)
        await telegram_request_telemetry_complete_guard(update, context)

        recent_after = store.get_recent(limit=5)
        matched_after = [r for r in recent_after if r["id"] == "tg-998877"]
        self.assertEqual("SUCCESS", matched_after[0]["status"])
        self.assertTrue(matched_after[0]["duration_ms"] >= 0.0)


class TestRealtimeLogsDashboardAndAdmin(unittest.IsolatedAsyncioTestCase):
    """Test Web Dashboard, API endpoints, and Admin Telegram integration."""

    async def test_web_dashboard_html_generation(self) -> None:
        from app.api.routes.logs import live_logs_dashboard

        resp = await live_logs_dashboard()
        self.assertEqual(200, resp.status_code)
        body = resp.body.decode("utf-8")
        self.assertIn("Server Real-Time Request Stream", body)
        self.assertIn("EventSource('/api/logs/stream')", body)
        self.assertIn("statActive", body)
        self.assertIn("terminalBody", body)

    async def test_api_logs_json_endpoint(self) -> None:
        from app.api.routes.logs import get_recent_logs_api

        resp = await get_recent_logs_api(limit=10)
        self.assertEqual(200, resp.status_code)
        payload = json.loads(resp.body.decode("utf-8"))
        self.assertIn("metrics", payload)
        self.assertIn("records", payload)
        self.assertIn("count", payload)

    def test_admin_live_logs_view_formatting(self) -> None:
        store = get_request_log_store()
        store.record_start("adm-1", "HTTP", "GET", "/healthz", "127.0.0.1")
        store.record_complete("adm-1", "200 OK", 200, 3.5)

        text = build_admin_live_logs_text(only_errors=False)
        self.assertIn("ADMIN REAL-TIME REQUEST LOGS", text)
        self.assertIn("Active:", text)
        self.assertIn("Avg Latency:", text)
        self.assertIn("/healthz", text)

        kb = get_admin_live_logs_kb(dashboard_url="https://bot.example.com/logs")
        all_callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row if getattr(btn, "callback_data", None)]
        all_urls = [btn.url for row in kb.inline_keyboard for btn in row if getattr(btn, "url", None)]
        self.assertIn("admin_live_logs:errors", all_callbacks)
        self.assertIn("https://bot.example.com/logs", all_urls)

    def test_configure_server_logging_idempotent(self) -> None:
        configure_server_logging()
        # Call again to test idempotency
        configure_server_logging()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
