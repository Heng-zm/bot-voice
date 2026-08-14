"""Bounded process-local incident history and administrator notifications."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.services.monitoring import sanitize_monitor_text

IncidentAlertHandler = Callable[[dict[str, Any]], Awaitable[None]]

_INCIDENT_EVENTS: deque[dict[str, Any]] = deque(maxlen=200)
_COMPONENT_STATUS: dict[str, dict[str, Any]] = {}
_INCIDENT_LOCK = threading.RLock()
_ALERT_HANDLER: IncidentAlertHandler | None = None


def configure_incident_alert_handler(
    handler: IncidentAlertHandler | None,
) -> None:
    """Set the process-local async handler used for failure/recovery alerts."""

    global _ALERT_HANDLER
    _ALERT_HANDLER = handler


def record_component_event(
    component: str,
    event: str,
    *,
    severity: str = "info",
    message: str = "",
    state: str | None = None,
    restart_count: int | None = None,
    consecutive_failures: int | None = None,
    next_retry_seconds: float | None = None,
    configuration_failure: bool = False,
) -> dict[str, Any]:
    """Record one sanitized event and update the component's current state."""

    clean_component = sanitize_monitor_text(component, limit=64) or "runtime"
    clean_event = sanitize_monitor_text(event, limit=64) or "status"
    clean_severity = str(severity or "info").strip().lower()
    if clean_severity not in {"info", "warning", "error", "critical"}:
        clean_severity = "info"
    timestamp = time.time()
    item = {
        "id": f"{int(timestamp * 1000)}-{clean_component}-{clean_event}",
        "ts": datetime.fromtimestamp(timestamp, tz=UTC).isoformat(),
        "timestamp": timestamp,
        "component": clean_component,
        "event": clean_event,
        "severity": clean_severity,
        "message": sanitize_monitor_text(message, limit=600),
        "state": sanitize_monitor_text(state or clean_event, limit=40),
        "restart_count": max(0, int(restart_count or 0)),
        "consecutive_failures": max(0, int(consecutive_failures or 0)),
        "next_retry_seconds": (
            round(max(0.0, float(next_retry_seconds)), 2)
            if next_retry_seconds is not None
            else None
        ),
        "configuration_failure": bool(configuration_failure),
    }
    with _INCIDENT_LOCK:
        _INCIDENT_EVENTS.appendleft(item)
        current = dict(_COMPONENT_STATUS.get(clean_component) or {})
        current.update(item)
        current["updated_at"] = item["ts"]
        _COMPONENT_STATUS[clean_component] = current
    return dict(item)


async def send_incident_alert(event: dict[str, Any]) -> bool:
    """Notify administrators without ever interrupting component recovery."""

    handler = _ALERT_HANDLER
    if handler is None:
        return False
    try:
        await asyncio.wait_for(handler(dict(event)), timeout=12.0)
        return True
    except Exception:
        return False


def incident_snapshot(*, limit: int = 50) -> dict[str, Any]:
    """Return a bounded sanitized copy for the admin monitor."""

    clean_limit = max(1, min(200, int(limit)))
    with _INCIDENT_LOCK:
        events = list(_INCIDENT_EVENTS)[:clean_limit]
        components = {
            key: dict(value) for key, value in sorted(_COMPONENT_STATUS.items())
        }
    return {
        "components": components,
        "events": events,
        "count": len(events),
        "captured": len(_INCIDENT_EVENTS),
        "open_circuits": sum(
            str(item.get("state")) == "circuit_open"
            for item in components.values()
        ),
    }


def reset_incident_state() -> None:
    """Clear process-local state for deterministic tests."""

    with _INCIDENT_LOCK:
        _INCIDENT_EVENTS.clear()
        _COMPONENT_STATUS.clear()


__all__ = [
    "IncidentAlertHandler",
    "configure_incident_alert_handler",
    "incident_snapshot",
    "record_component_event",
    "reset_incident_state",
    "send_incident_alert",
]
