from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app import legacy
from app.main import _wait_for_critical_tasks
from app.runtime import RuntimeContext


class RuntimeOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_role_upgrade_rolls_back_owner(self) -> None:
        runtime = RuntimeContext()
        runtime.started = True
        runtime.role = "web"
        runtime._owners["web"] = "web"
        runtime._ensure_workers = AsyncMock(side_effect=RuntimeError("worker boot failed"))

        with self.assertRaisesRegex(RuntimeError, "worker boot failed"):
            await runtime.start(owner="worker", role="worker")

        self.assertEqual({"web": "web"}, runtime._owners)
        self.assertEqual("web", runtime.role)

    async def test_existing_owner_can_finish_deferred_web_setup(self) -> None:
        runtime = RuntimeContext()
        runtime.started = True
        runtime.role = "web"
        runtime._owners["web"] = "web"
        runtime._ensure_web_services = AsyncMock()
        application = object()

        await runtime.start(application, owner="web", role="web")

        runtime._ensure_web_services.assert_awaited_once_with(application)


class CriticalTaskSupervisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_critical_task_exit_is_detected(self) -> None:
        blocker = asyncio.Event()
        finished = asyncio.create_task(asyncio.sleep(0), name="finished")
        running = asyncio.create_task(blocker.wait(), name="running")
        try:
            await asyncio.wait_for(
                _wait_for_critical_tasks([finished, running]),
                timeout=0.2,
            )
            self.assertFalse(running.done())
        finally:
            running.cancel()
            await asyncio.gather(finished, running, return_exceptions=True)

    async def test_critical_task_exception_is_propagated(self) -> None:
        async def fail() -> None:
            raise RuntimeError("critical service failed")

        failed = asyncio.create_task(fail(), name="failed")
        running = asyncio.create_task(asyncio.Event().wait(), name="running")
        try:
            with self.assertRaisesRegex(RuntimeError, "critical service failed"):
                await _wait_for_critical_tasks([failed, running])
        finally:
            running.cancel()
            await asyncio.gather(failed, running, return_exceptions=True)


class TelegramStartupRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def test_missing_webhook_url_falls_back_to_polling(self) -> None:
        original_mode = legacy.BOT_MODE
        with legacy.RUN_STATE_LOCK:
            original_state = dict(legacy.RUN_STATE)
            legacy.RUN_STATE["BOT_MODE"] = "WEBHOOK"
            legacy.BOT_MODE = "WEBHOOK"
        try:
            with (
                patch.object(legacy, "_runtime_webhook_base_url", return_value=""),
                patch.object(legacy.webhook_logger, "warning") as warning,
            ):
                mode = legacy._ensure_startup_telegram_mode()

            self.assertEqual("POLLING", mode)
            self.assertEqual("POLLING", legacy._run_state_bot_mode())
            warning.assert_called_once()
        finally:
            with legacy.RUN_STATE_LOCK:
                legacy.RUN_STATE.clear()
                legacy.RUN_STATE.update(original_state)
                legacy.BOT_MODE = original_mode

    async def test_activation_failure_stops_started_telegram_application(self) -> None:
        class FakeApplication:
            running = False
            stopped = False

            async def start(self) -> None:
                self.running = True

            async def stop(self) -> None:
                self.running = False
                self.stopped = True

        application = FakeApplication()
        activation = AsyncMock(side_effect=RuntimeError("webhook activation failed"))
        with (
            patch.object(legacy, "_activate_telegram_application", activation),
            self.assertRaisesRegex(RuntimeError, "webhook activation failed"),
        ):
            await legacy._start_telegram_application(application)

        self.assertTrue(application.stopped)
        self.assertFalse(application.running)
        self.assertFalse(legacy._TELEGRAM_APP_READY)


class UvicornShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_waits_for_server_socket_shutdown(self) -> None:
        server_created = asyncio.Event()

        class FakeServer:
            latest = None

            def __init__(self, _config) -> None:
                type(self).latest = self
                self._should_exit = False
                self._exit_requested = asyncio.Event()
                self.force_exit = False
                self.started = asyncio.Event()
                self.stopped = asyncio.Event()
                server_created.set()

            @property
            def should_exit(self) -> bool:
                return self._should_exit

            @should_exit.setter
            def should_exit(self, value: bool) -> None:
                self._should_exit = bool(value)
                if self._should_exit:
                    self._exit_requested.set()

            async def serve(self) -> None:
                self.started.set()
                await self._exit_requested.wait()
                self.stopped.set()

        with (
            patch("uvicorn.Config", return_value=object()),
            patch("uvicorn.Server", FakeServer),
        ):
            task = asyncio.create_task(legacy.run_fastapi())
            await server_created.wait()
            server = FakeServer.latest
            await server.started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(server.should_exit)
        self.assertTrue(server.stopped.is_set())
        self.assertFalse(server.force_exit)


if __name__ == "__main__":
    unittest.main()
