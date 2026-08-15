from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app import legacy
from app.main import _wait_for_critical_tasks
from app.runtime import RuntimeContext
from app.services.incidents import (
    configure_incident_alert_handler,
    incident_snapshot,
    reset_incident_state,
)


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
    def test_wispbyte_allocation_overrides_persisted_webhook_mode(self) -> None:
        original_mode = legacy.BOT_MODE
        with legacy.RUN_STATE_LOCK:
            original_state = dict(legacy.RUN_STATE)
            legacy.RUN_STATE["BOT_MODE"] = "WEBHOOK"
            legacy.BOT_MODE = "WEBHOOK"
        try:
            with (
                patch.dict("os.environ", {"SERVER_PORT": "13961"}, clear=True),
                patch.object(
                    legacy,
                    "_runtime_webhook_base_url",
                    return_value="https://old-render-service.onrender.com",
                ),
            ):
                mode = legacy._ensure_startup_telegram_mode()

            self.assertEqual("POLLING", mode)
            self.assertEqual("POLLING", legacy._run_state_bot_mode())
        finally:
            with legacy.RUN_STATE_LOCK:
                legacy.RUN_STATE.clear()
                legacy.RUN_STATE.update(original_state)
                legacy.BOT_MODE = original_mode

    def test_wispbyte_disables_implicit_render_leader_lock(self) -> None:
        with (
            patch.dict("os.environ", {"SERVER_PORT": "13961"}, clear=True),
            patch.object(legacy, "TELEGRAM_ACTIVE_LOCK_ENABLED", True),
            patch.object(legacy, "TELEGRAM_ACTIVE_LOCK_REQUIRED", True),
        ):
            self.assertFalse(legacy._telegram_leader_lock_enabled())
            self.assertFalse(legacy._telegram_leader_require_store())

    def test_wispbyte_respects_explicit_webhook_and_leader_lock(self) -> None:
        original_mode = legacy.BOT_MODE
        with legacy.RUN_STATE_LOCK:
            original_state = dict(legacy.RUN_STATE)
            legacy.RUN_STATE["BOT_MODE"] = "POLLING"
            legacy.BOT_MODE = "POLLING"
        try:
            with (
                patch.dict(
                    "os.environ",
                    {
                        "SERVER_PORT": "13961",
                        "BOT_MODE": "WEBHOOK",
                        "TELEGRAM_ACTIVE_LOCK_ENABLED": "true",
                        "TELEGRAM_ACTIVE_LOCK_REQUIRED": "true",
                    },
                    clear=True,
                ),
                patch.object(
                    legacy,
                    "_runtime_webhook_base_url",
                    return_value="https://bot.example.com",
                ),
                patch.object(legacy, "TELEGRAM_ACTIVE_LOCK_ENABLED", True),
                patch.object(legacy, "TELEGRAM_ACTIVE_LOCK_REQUIRED", True),
            ):
                self.assertEqual("WEBHOOK", legacy._ensure_startup_telegram_mode())
                self.assertTrue(legacy._telegram_leader_lock_enabled())
                self.assertTrue(legacy._telegram_leader_require_store())
        finally:
            with legacy.RUN_STATE_LOCK:
                legacy.RUN_STATE.clear()
                legacy.RUN_STATE.update(original_state)
                legacy.BOT_MODE = original_mode

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

    async def test_polling_does_not_start_when_webhook_deletion_fails(self) -> None:
        class FakeUpdater:
            started = False

            async def start_polling(self, **_kwargs) -> None:
                self.started = True

        application = type("FakeApplication", (), {"updater": FakeUpdater()})()
        original_active = legacy._TELEGRAM_POLLING_ACTIVE
        original_task = legacy.ACTIVE_POLLING_TASK
        original_lock = legacy._TELEGRAM_POLLING_LOCK
        legacy._TELEGRAM_POLLING_ACTIVE = False
        legacy.ACTIVE_POLLING_TASK = None
        legacy._TELEGRAM_POLLING_LOCK = None
        try:
            with (
                patch.object(legacy, "_run_state_bot_mode", return_value="POLLING"),
                patch.object(
                    legacy,
                    "_cancel_active_polling_task",
                    AsyncMock(),
                ),
                patch.object(
                    legacy,
                    "_delete_telegram_webhook_via_http",
                    AsyncMock(side_effect=RuntimeError("deletion not confirmed")),
                ),
                self.assertRaisesRegex(RuntimeError, "deletion not confirmed"),
            ):
                await legacy._telegram_start_polling_runtime(application)
        finally:
            legacy._TELEGRAM_POLLING_ACTIVE = original_active
            legacy.ACTIVE_POLLING_TASK = original_task
            legacy._TELEGRAM_POLLING_LOCK = original_lock

        self.assertFalse(application.updater.started)

    async def test_webhook_deletion_is_verified_and_retried(self) -> None:
        class FakeResponse:
            status_code = 200
            text = ""

            def __init__(self, content: bytes) -> None:
                self.content = content

        class FakeClient:
            def __init__(self) -> None:
                self.info_calls = 0
                self.delete_calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args) -> None:
                return None

            async def post(self, _url, **_kwargs):
                self.delete_calls += 1
                return FakeResponse(b'{"ok":true,"result":true}')

            async def get(self, _url):
                self.info_calls += 1
                url = "https://old.example/webhook" if self.info_calls == 1 else ""
                return FakeResponse(
                    ('{"ok":true,"result":{"url":"' + url + '"}}').encode()
                )

        client = FakeClient()
        with (
            patch.object(legacy, "TELEGRAM_BOT_TOKEN", "test-token"),
            patch.object(legacy.httpx, "AsyncClient", return_value=client),
            patch.object(legacy.asyncio, "sleep", AsyncMock()),
        ):
            await legacy._delete_telegram_webhook_via_http(drop_pending=False)

        self.assertEqual(2, client.delete_calls)
        self.assertEqual(2, client.info_calls)

    async def test_polling_conflict_runs_verified_cleanup(self) -> None:
        reset_incident_state()
        cleanup = AsyncMock()
        legacy._TELEGRAM_CONFLICT_RECOVERY_TASK = asyncio.current_task()
        with (
            patch.object(legacy, "_run_state_bot_mode", return_value="POLLING"),
            patch.object(
                legacy,
                "_delete_telegram_webhook_via_http",
                cleanup,
            ),
        ):
            await legacy._recover_polling_webhook_conflict()

        cleanup.assert_awaited_once_with(drop_pending=False)
        self.assertIsNone(legacy._TELEGRAM_CONFLICT_RECOVERY_TASK)
        event = incident_snapshot()["events"][0]
        self.assertEqual("telegram_webhook", event["component"])
        self.assertEqual("conflict_recovered", event["event"])

    async def test_failed_polling_conflict_recovery_alerts_admin(self) -> None:
        reset_incident_state()
        alerted = asyncio.Event()
        alerts: list[dict] = []

        async def alert_handler(event: dict) -> None:
            alerts.append(event)
            alerted.set()

        configure_incident_alert_handler(alert_handler)
        legacy._TELEGRAM_CONFLICT_RECOVERY_TASK = asyncio.current_task()
        try:
            with (
                patch.object(legacy, "_run_state_bot_mode", return_value="POLLING"),
                patch.object(
                    legacy,
                    "_delete_telegram_webhook_via_http",
                    AsyncMock(side_effect=RuntimeError("Telegram API unavailable")),
                ),
            ):
                await legacy._recover_polling_webhook_conflict()
            await asyncio.wait_for(alerted.wait(), timeout=0.2)
        finally:
            configure_incident_alert_handler(None)

        self.assertEqual(1, len(alerts))
        self.assertEqual("conflict_recovery_failed", alerts[0]["event"])
        self.assertEqual("critical", alerts[0]["severity"])


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
            patch.dict(
                "os.environ",
                {
                    "SERVER_PORT": "13961",
                    "PORT": "8080",
                    "UVICORN_ACCESS_LOG": "false",
                    "WEB_KEEP_ALIVE_SECONDS": "15",
                },
                clear=False,
            ),
            patch("uvicorn.Config", return_value=object()) as config,
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
        self.assertEqual(
            "0.0.0.0",  # noqa: S104 - public binding is the requirement
            config.call_args.kwargs["host"],
        )
        self.assertEqual(13_961, config.call_args.kwargs["port"])
        self.assertFalse(config.call_args.kwargs["access_log"])
        self.assertEqual(15, config.call_args.kwargs["timeout_keep_alive"])


if __name__ == "__main__":
    unittest.main()
