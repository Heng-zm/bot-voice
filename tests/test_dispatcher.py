"""Tests for the modernized TelegramDispatcher."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from fastapi import HTTPException
    from starlette.requests import Request
    HAS_WEBHOOK_DEPS = True
except (ImportError, ModuleNotFoundError):
    HAS_WEBHOOK_DEPS = False
    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int = 400, detail: str = "", *args, **kwargs):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
    Request = object  # type: ignore[misc,assignment]

from pathlib import Path

_venv_site = Path(r"F:\ai project\bot-voice\.venv\Lib\site-packages")
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.append(str(_venv_site))

if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        sys.modules["httpx"] = MagicMock()

if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "InlineKeyboardButton"):
    try:
        import telegram
        from telegram import Update
    except (ImportError, ModuleNotFoundError):
        import sys
        import types

        class _DynamicMockModule(types.ModuleType):
            def __getattr__(self, name: str):
                mock = MagicMock(name=name)
                setattr(self, name, mock)
                return mock

        telegram_mod = _DynamicMockModule("telegram")
        class Update:  # type: ignore[no-redef]
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
        telegram_mod.__path__ = []
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

        class _DynamicMockError(Exception):
            pass

        class _DynamicErrorModule(types.ModuleType):
            def __getattr__(self, name: str):
                err_cls = type(name, (_DynamicMockError,), {})
                setattr(self, name, err_cls)
                return err_cls

        telegram_error = _DynamicErrorModule("telegram.error")
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

from app.services.telegram.deduplication import (
    _telegram_webhook_update_claim,
    reset_webhook_replay_store,
)
try:
    from app.services.telegram.dispatcher import TelegramDispatcher, get_telegram_dispatcher
except (ImportError, ModuleNotFoundError):
    TelegramDispatcher = object  # type: ignore[misc,assignment]
    get_telegram_dispatcher = lambda: None  # type: ignore[assignment]
    HAS_WEBHOOK_DEPS = False


def _make_mock_request(
    body_dict: dict,
    *,
    headers: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
) -> Request:
    if "message" in body_dict and isinstance(body_dict["message"], dict):
        body_dict["message"].setdefault("date", 1700000000)
        body_dict["message"].setdefault("chat", {"id": 12345, "type": "private"})
    raw_body = json.dumps(body_dict).encode("utf-8")

    async def receive() -> dict:
        return {"type": "http.request", "body": raw_body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (headers or {}).items()
        ],
        "client": (client_host, 12345),
    }
    return Request(scope, receive)


@unittest.skipUnless(HAS_WEBHOOK_DEPS, "Requires webhook server dependencies (fastapi, starlette, anyio)")
class TelegramDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        reset_webhook_replay_store()
        self.dispatcher = TelegramDispatcher()
        self.dispatcher.async_dispatch = False  # Synchronous default for deterministic testing

    async def test_singleton_accessor(self) -> None:
        d1 = get_telegram_dispatcher()
        d2 = get_telegram_dispatcher()
        self.assertIs(d1, d2)
        self.assertIsInstance(d1, TelegramDispatcher)

    async def test_rejected_on_invalid_secret_before_mode_check(self) -> None:
        # Secret check must happen before mode validation to prevent info leakage
        with patch.object(self.dispatcher, "_get_expected_secret", return_value="correct_secret"), \
             patch.object(self.dispatcher, "_is_webhook_mode", return_value=False):
            req = _make_mock_request({"update_id": 1001}, headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"})
            with self.assertRaises(HTTPException) as ctx:
                await self.dispatcher.dispatch_webhook_request(req)
            self.assertEqual(ctx.exception.status_code, 403)

    async def test_ignored_when_authenticated_but_not_webhook_mode(self) -> None:
        with patch.object(self.dispatcher, "_get_expected_secret", return_value="correct_secret"), \
             patch.object(self.dispatcher, "_is_webhook_mode", return_value=False):
            req = _make_mock_request({"update_id": 1002}, headers={"X-Telegram-Bot-Api-Secret-Token": "correct_secret"})
            resp = await self.dispatcher.dispatch_webhook_request(req)
            self.assertEqual(resp.status_code, 200)
            body = json.loads(resp.body)
            self.assertEqual(body.get("status"), "ignored")
            self.assertEqual(body.get("reason"), "not_webhook_mode")

    async def test_successful_dispatch_and_deduplication(self) -> None:
        mock_app = MagicMock()
        mock_app.bot = MagicMock()
        mock_app.bot.defaults = None
        mock_app.process_update = AsyncMock()

        with patch.object(self.dispatcher, "_get_expected_secret", return_value="my_secret_token"), \
             patch.object(self.dispatcher, "_is_webhook_mode", return_value=True), \
             patch.object(self.dispatcher, "_get_app_instance", return_value=mock_app), \
             patch.object(self.dispatcher, "_is_app_ready", return_value=True), \
             patch.object(self.dispatcher, "_should_process_update", return_value=True):

            headers = {"X-Telegram-Bot-Api-Secret-Token": "my_secret_token"}
            req1 = _make_mock_request({"update_id": 8888, "message": {"message_id": 1}}, headers=headers)

            resp1 = await self.dispatcher.dispatch_webhook_request(req1, path_secret_token="my_secret_token")
            self.assertEqual(resp1.status_code, 200)
            body1 = json.loads(resp1.body)
            self.assertEqual(body1.get("status"), "ok")
            mock_app.process_update.assert_awaited_once()

            # Second request with identical update_id must be dropped as duplicate
            req2 = _make_mock_request({"update_id": 8888, "message": {"message_id": 1}}, headers=headers)
            resp2 = await self.dispatcher.dispatch_webhook_request(req2, path_secret_token="my_secret_token")
            self.assertEqual(resp2.status_code, 200)
            body2 = json.loads(resp2.body)
            self.assertEqual(body2.get("status"), "ok")
            self.assertTrue(body2.get("duplicate"))
            self.assertEqual(mock_app.process_update.await_count, 1)

    async def test_async_background_dispatch_and_drain(self) -> None:
        self.dispatcher.async_dispatch = True
        mock_app = MagicMock()
        mock_app.bot = MagicMock()
        mock_app.bot.defaults = None
        started_event = asyncio.Event()

        async def slow_process(update: Update) -> None:
            started_event.set()
            await asyncio.sleep(0.05)

        mock_app.process_update = AsyncMock(side_effect=slow_process)

        with patch.object(self.dispatcher, "_get_expected_secret", return_value="token123"), \
             patch.object(self.dispatcher, "_is_webhook_mode", return_value=True), \
             patch.object(self.dispatcher, "_get_app_instance", return_value=mock_app), \
             patch.object(self.dispatcher, "_is_app_ready", return_value=True), \
             patch.object(self.dispatcher, "_should_process_update", return_value=True):

            headers = {"X-Telegram-Bot-Api-Secret-Token": "token123"}
            req = _make_mock_request({"update_id": 9999, "message": {"message_id": 2}}, headers=headers)

            resp = await self.dispatcher.dispatch_webhook_request(req)
            self.assertEqual(resp.status_code, 200)
            body = json.loads(resp.body)
            self.assertEqual(body.get("status"), "ok")
            self.assertTrue(body.get("dispatched"))

            # Wait for background worker to start
            await started_event.wait()
            metrics = self.dispatcher.get_metrics()
            self.assertEqual(metrics["updates_dispatched"], 1)

            # Drain gracefully finishes background worker
            await self.dispatcher.drain(timeout=2.0)
            self.assertEqual(len(self.dispatcher._active_tasks), 0)
            final_metrics = self.dispatcher.get_metrics()
            self.assertEqual(final_metrics["updates_completed"], 1)

    async def test_cancellation_releases_dedup_lease(self) -> None:
        """Verify that when a task is cancelled in drain(), its lease is released, preventing 503 retry deadlock."""
        self.dispatcher.async_dispatch = True
        mock_app = MagicMock()
        mock_app.bot = MagicMock()
        mock_app.bot.defaults = None
        started_event = asyncio.Event()

        async def hung_process(update: Update) -> None:
            started_event.set()
            await asyncio.sleep(100.0)  # Hung / very long task

        mock_app.process_update = AsyncMock(side_effect=hung_process)

        with patch.object(self.dispatcher, "_get_expected_secret", return_value="token123"), \
             patch.object(self.dispatcher, "_is_webhook_mode", return_value=True), \
             patch.object(self.dispatcher, "_get_app_instance", return_value=mock_app), \
             patch.object(self.dispatcher, "_is_app_ready", return_value=True), \
             patch.object(self.dispatcher, "_should_process_update", return_value=True):

            headers = {"X-Telegram-Bot-Api-Secret-Token": "token123"}
            req = _make_mock_request({"update_id": 7777, "message": {"message_id": 3}}, headers=headers)

            resp = await self.dispatcher.dispatch_webhook_request(req)
            self.assertEqual(resp.status_code, 200)

            await started_event.wait()
            # Cancel tasks via drain with 0.05s timeout
            await self.dispatcher.drain(timeout=0.05)

            # Lease MUST NOT be stuck in "processing". It was released, so claiming update 7777 again succeeds!
            claim_state = await _telegram_webhook_update_claim(7777)
            self.assertEqual(claim_state, "claimed")

    async def test_queue_saturation_applies_backpressure(self) -> None:
        self.dispatcher.async_dispatch = True
        self.dispatcher._concurrency_limit = 2
        self.dispatcher._max_queue_depth = 1

        mock_app = MagicMock()
        mock_app.bot = MagicMock()
        async def _slow_process(u):
            await asyncio.sleep(1.0)

        mock_app.process_update = AsyncMock(side_effect=_slow_process)

        with patch.object(self.dispatcher, "_get_expected_secret", return_value="token123"), \
             patch.object(self.dispatcher, "_is_webhook_mode", return_value=True), \
             patch.object(self.dispatcher, "_get_app_instance", return_value=mock_app), \
             patch.object(self.dispatcher, "_is_app_ready", return_value=True), \
             patch.object(self.dispatcher, "_should_process_update", return_value=True):

            headers = {"X-Telegram-Bot-Api-Secret-Token": "token123"}

            # Fill up active tasks: 2 concurrency + 1 queue depth = 3 max allowed
            for uid in range(3):
                req = _make_mock_request({"update_id": 100 + uid, "message": {"message_id": 100 + uid}}, headers=headers)
                resp = await self.dispatcher.dispatch_webhook_request(req)
                self.assertEqual(resp.status_code, 200)

            # 4th request exceeds max allowed and must be rejected with 503 queue_saturated
            req_saturated = _make_mock_request({"update_id": 104, "message": {"message_id": 104}}, headers=headers)
            resp_sat = await self.dispatcher.dispatch_webhook_request(req_saturated)
            self.assertEqual(resp_sat.status_code, 503)
            body = json.loads(resp_sat.body)
            self.assertEqual(body.get("reason"), "queue_saturated")

            # Clean up active tasks
            await self.dispatcher.drain(timeout=0.05)

    async def test_path_secret_token_accepted_without_header(self) -> None:
        mock_app = MagicMock()
        mock_app.bot = MagicMock()
        mock_app.bot.defaults = None
        mock_app.process_update = AsyncMock()

        with patch.object(self.dispatcher, "_get_expected_secret", return_value="path_secret_xyz"), \
             patch.object(self.dispatcher, "_is_webhook_mode", return_value=True), \
             patch.object(self.dispatcher, "_get_app_instance", return_value=mock_app), \
             patch.object(self.dispatcher, "_is_app_ready", return_value=True), \
             patch.object(self.dispatcher, "_should_process_update", return_value=True):

            # No X-Telegram-Bot-Api-Secret-Token header, but valid path secret token
            req = _make_mock_request({"update_id": 9999, "message": {"message_id": 99}})
            resp = await self.dispatcher.dispatch_webhook_request(req, path_secret_token="path_secret_xyz")
            self.assertEqual(resp.status_code, 200)
            mock_app.process_update.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
