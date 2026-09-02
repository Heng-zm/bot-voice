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


    def test_runtime_mode_switch(self) -> None:
        mode = legacy._run_state_bot_mode()
        self.assertIn(mode, {"POLLING", "WEBHOOK"})
        with self.assertRaises(Exception):
            asyncio.run(legacy._switch_telegram_runtime_mode("INVALID_MODE"))


    def test_startup_has_no_web_server_task(self) -> None:
        source = inspect.getsource(legacy._async_main_once)
        self.assertNotIn("run_fastapi", source)
        self.assertNotIn("_start_web_broadcast_queue_workers", source)
        self.assertNotIn("uvicorn", source.lower())

    def test_legacy_has_no_server_launcher(self) -> None:
        self.assertFalse(hasattr(legacy, "run_fastapi"))
        self.assertFalse(hasattr(legacy, "run_flask"))

    def test_webhook_routes_registered(self) -> None:
        route_paths = [route.path for route in bot_entrypoint.app.routes]
        self.assertIn("/webhook", route_paths)
        self.assertIn("/telegram/webhook", route_paths)
        self.assertIn("/healthz", route_paths)




if __name__ == "__main__":
    unittest.main()
