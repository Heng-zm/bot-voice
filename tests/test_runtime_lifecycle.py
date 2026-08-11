from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock

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


if __name__ == "__main__":
    unittest.main()
