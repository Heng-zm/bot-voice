"""Independent async component recovery with bounded exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.services.incidents import record_component_event, send_incident_alert

ComponentFactory = Callable[[], Awaitable[None]]
logger = logging.getLogger(__name__)

_CONFIGURATION_MARKERS = (
    "address already in use",
    "configuration",
    "invalid configuration",
    "invalid token",
    "is missing",
    "must contain",
    "not configured",
    "not set",
    "permission denied",
    "requires ",
    "unsupported mode",
)


def is_configuration_failure(error: BaseException) -> bool:
    """Classify deterministic failures that should eventually open a circuit."""

    message = f"{type(error).__name__}: {error}".lower()
    return isinstance(error, (ValueError, TypeError, PermissionError)) or any(
        marker in message for marker in _CONFIGURATION_MARKERS
    )


@dataclass(frozen=True, slots=True)
class SupervisorPolicy:
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    stable_run_seconds: float = 60.0
    max_configuration_failures: int = 3

    def normalized(self) -> SupervisorPolicy:
        base = max(0.05, min(60.0, float(self.base_backoff_seconds)))
        maximum = max(base, min(3600.0, float(self.max_backoff_seconds)))
        stable = max(0.05, min(86_400.0, float(self.stable_run_seconds)))
        failures = max(1, min(20, int(self.max_configuration_failures)))
        return SupervisorPolicy(base, maximum, stable, failures)


class ComponentSupervisor:
    """Keep one component alive without restarting healthy sibling services."""

    def __init__(
        self,
        component: str,
        factory: ComponentFactory,
        *,
        policy: SupervisorPolicy | None = None,
    ) -> None:
        self.component = str(component or "runtime").strip()[:64] or "runtime"
        self.factory = factory
        self.policy = (policy or SupervisorPolicy()).normalized()
        self.restart_count = 0
        self.consecutive_failures = 0
        self.configuration_failures = 0
        self._generation = 0
        self._component_task: asyncio.Task[None] | None = None

    def _delay(self) -> float:
        exponent = min(max(0, self.consecutive_failures - 1), 10)
        return min(
            self.policy.max_backoff_seconds,
            self.policy.base_backoff_seconds * (2**exponent),
        )

    async def _mark_stable(self, generation: int, had_failures: bool) -> None:
        await asyncio.sleep(self.policy.stable_run_seconds)
        task = self._component_task
        if generation != self._generation or task is None or task.done():
            return
        self.consecutive_failures = 0
        self.configuration_failures = 0
        event = record_component_event(
            self.component,
            "recovered" if had_failures else "running",
            severity="info",
            message="Component is stable and healthy.",
            state="running",
            restart_count=self.restart_count,
        )
        if had_failures:
            logger.info(
                "Component recovered component=%s restarts=%s.",
                self.component,
                self.restart_count,
            )
            await send_incident_alert(event)

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run until cancelled/stopped; an open config circuit waits for deploy."""

        record_component_event(
            self.component,
            "starting",
            message="Component supervisor started.",
            state="starting",
        )
        while not stop_event.is_set():
            self._generation += 1
            generation = self._generation
            had_failures = self.restart_count > 0
            started = time.monotonic()
            self._component_task = asyncio.create_task(
                self.factory(),
                name=f"component-{self.component}",
            )
            stable_task = asyncio.create_task(
                self._mark_stable(generation, had_failures),
                name=f"component-stable-{self.component}",
            )
            try:
                await self._component_task
                if stop_event.is_set():
                    return
                error: BaseException = RuntimeError(
                    f"{self.component} component exited unexpectedly."
                )
            except asyncio.CancelledError:
                if self._component_task and not self._component_task.done():
                    self._component_task.cancel()
                await asyncio.gather(self._component_task, return_exceptions=True)
                raise
            except BaseException as exc:  # component process boundary
                error = exc
            finally:
                stable_task.cancel()
                await asyncio.gather(stable_task, return_exceptions=True)

            duration = max(0.0, time.monotonic() - started)
            if duration >= self.policy.stable_run_seconds:
                self.consecutive_failures = 0
                self.configuration_failures = 0
            self.restart_count += 1
            self.consecutive_failures += 1
            configuration_failure = is_configuration_failure(error)
            if configuration_failure:
                self.configuration_failures += 1
            else:
                self.configuration_failures = 0

            error_text = f"{type(error).__name__}: {error}"
            failed_event = record_component_event(
                self.component,
                "failed",
                severity="error",
                message=error_text,
                state="failed",
                restart_count=self.restart_count,
                consecutive_failures=self.consecutive_failures,
                configuration_failure=configuration_failure,
            )
            logger.error(
                "Component failed component=%s restart=%s configuration_failure=%s: %s",
                self.component,
                self.restart_count,
                configuration_failure,
                error_text,
            )
            if self.consecutive_failures == 1:
                await send_incident_alert(failed_event)

            if (
                configuration_failure
                and self.configuration_failures
                >= self.policy.max_configuration_failures
            ):
                circuit_event = record_component_event(
                    self.component,
                    "circuit_open",
                    severity="critical",
                    message=(
                        "Automatic restarts stopped after repeated configuration "
                        f"failures. Last error: {error_text}"
                    ),
                    state="circuit_open",
                    restart_count=self.restart_count,
                    consecutive_failures=self.consecutive_failures,
                    configuration_failure=True,
                )
                logger.critical(
                    "Component restart circuit opened component=%s after %s "
                    "configuration failures.",
                    self.component,
                    self.configuration_failures,
                )
                await send_incident_alert(circuit_event)
                await stop_event.wait()
                return

            delay = self._delay()
            record_component_event(
                self.component,
                "restart_scheduled",
                severity="warning",
                message=f"Restart scheduled in {delay:g} seconds.",
                state="backoff",
                restart_count=self.restart_count,
                consecutive_failures=self.consecutive_failures,
                next_retry_seconds=delay,
                configuration_failure=configuration_failure,
            )
            logger.warning(
                "Component restart scheduled component=%s delay_seconds=%s.",
                self.component,
                delay,
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except TimeoutError:
                continue
            return


__all__ = [
    "ComponentSupervisor",
    "SupervisorPolicy",
    "is_configuration_failure",
]
