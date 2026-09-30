"""Integration tests for Telegram bot dispatcher and routing."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from app.bot.dispatcher import BotDispatcher
from app.bot.keyboards import get_main_menu_keyboard, get_admin_menu_keyboard


class TelegramIntegrationTests(unittest.TestCase):
    def test_keyboards_structure(self):
        main_kb = get_main_menu_keyboard()
        self.assertIsNotNone(main_kb)
        admin_kb = get_admin_menu_keyboard()
        self.assertIsNotNone(admin_kb)

    def test_dispatcher_init(self):
        dispatcher = BotDispatcher()
        self.assertIsNotNone(dispatcher)


if __name__ == "__main__":
    unittest.main()
