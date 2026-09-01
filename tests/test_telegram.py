from __future__ import annotations

import asyncio
import inspect
import unittest

from app import legacy
from app import main as bot_entrypoint


class TelegramOnlyRuntimeTests(unittest.TestCase):
    def test_entrypoint_exports_main_and_app(self) -> None:
        self.assertTrue(callable(bot_entrypoint.main))
        self.assertTrue(hasattr(bot_entrypoint, "app"))


    def test_runtime_is_polling_only(self) -> None:
        self.assertEqual("POLLING", legacy._run_state_bot_mode())
        with self.assertRaisesRegex(RuntimeError, "removed"):
            asyncio.run(legacy._switch_telegram_runtime_mode("WEBHOOK"))

    def test_startup_has_no_web_server_task(self) -> None:
        source = inspect.getsource(legacy._async_main_once)
        self.assertNotIn("run_fastapi", source)
        self.assertNotIn("_start_web_broadcast_queue_workers", source)
        self.assertNotIn("uvicorn", source.lower())

    def test_legacy_has_no_server_launcher(self) -> None:
        self.assertFalse(hasattr(legacy, "run_fastapi"))
        self.assertFalse(hasattr(legacy, "run_flask"))

    def test_bot_builder_creates_application(self) -> None:
        from app.bot import build_telegram_application

        app = build_telegram_application("123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")
        self.assertIsNotNone(app)
        self.assertEqual(app.bot.token, "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")



if __name__ == "__main__":
    unittest.main()
