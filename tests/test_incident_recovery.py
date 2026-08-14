from __future__ import annotations

import asyncio
import unittest

from app.services.incidents import (
    configure_incident_alert_handler,
    incident_snapshot,
    reset_incident_state,
)
from app.services.supervision import ComponentSupervisor, SupervisorPolicy


class ComponentSupervisorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_incident_state()
        configure_incident_alert_handler(None)

    def tearDown(self) -> None:
        configure_incident_alert_handler(None)
        reset_incident_state()

    async def test_transient_failure_restarts_only_failed_component(self) -> None:
        recovered = asyncio.Event()
        stop = asyncio.Event()
        calls = 0
        alerts: list[str] = []

        async def alert_handler(event: dict) -> None:
            alerts.append(str(event["event"]))
            if event["event"] == "recovered":
                recovered.set()

        async def component() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary network outage")
            await stop.wait()

        configure_incident_alert_handler(alert_handler)
        supervisor = ComponentSupervisor(
            "telegram",
            component,
            policy=SupervisorPolicy(
                base_backoff_seconds=0.05,
                max_backoff_seconds=0.1,
                stable_run_seconds=0.05,
            ),
        )
        task = asyncio.create_task(supervisor.run(stop))
        try:
            await asyncio.wait_for(recovered.wait(), timeout=1.0)
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)

        events = [item["event"] for item in incident_snapshot()["events"]]
        self.assertEqual(2, calls)
        self.assertIn("failed", events)
        self.assertIn("restart_scheduled", events)
        self.assertIn("recovered", events)
        self.assertEqual(["failed", "recovered"], alerts)

    async def test_repeated_configuration_failure_opens_circuit(self) -> None:
        circuit_open = asyncio.Event()
        stop = asyncio.Event()
        calls = 0

        async def alert_handler(event: dict) -> None:
            if event["event"] == "circuit_open":
                circuit_open.set()

        async def component() -> None:
            nonlocal calls
            calls += 1
            raise ValueError("invalid configuration: BOT_TOKEN is missing")

        configure_incident_alert_handler(alert_handler)
        supervisor = ComponentSupervisor(
            "telegram",
            component,
            policy=SupervisorPolicy(
                base_backoff_seconds=0.05,
                max_backoff_seconds=0.05,
                stable_run_seconds=0.05,
                max_configuration_failures=3,
            ),
        )
        task = asyncio.create_task(supervisor.run(stop))
        try:
            await asyncio.wait_for(circuit_open.wait(), timeout=1.0)
            await asyncio.sleep(0.1)
            self.assertEqual(3, calls)
            snapshot = incident_snapshot()
            self.assertEqual(1, snapshot["open_circuits"])
            self.assertEqual(
                "circuit_open",
                snapshot["components"]["telegram"]["state"],
            )
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=1.0)


if __name__ == "__main__":
    unittest.main()
