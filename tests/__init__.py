"""Test suite initialization and dependency bootstrapping.

Ensures that if optional dependencies (telegram, httpx, fastapi) are not installed
in the host environment, complete and consistent mock structures are registered so
no test module inherits a broken or partial mock.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from pathlib import Path

_venv_site = Path(r"F:\ai project\bot-voice\.venv\Lib\site-packages")
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.append(str(_venv_site))

# 1. Ensure httpx
if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        sys.modules["httpx"] = MagicMock()

# 2. Ensure telegram package and submodules
if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "ext") or not hasattr(sys.modules["telegram"], "InlineKeyboardButton"):
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
