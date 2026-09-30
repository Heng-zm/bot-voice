"""Server-Sent Events (SSE) formatting and async generators for live telemetry streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any


def format_sse(data: Any, event: str | None = None, event_id: str | None = None) -> str:
    """Format payload into standard SSE data string."""
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    if event_id:
        lines.append(f"id: {event_id}")

    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)

    for line in payload.splitlines():
        lines.append(f"data: {line}")
    lines.append("\n")
    return "\n".join(lines)


async def sse_event_stream(
    queue: asyncio.Queue[Any],
    ping_interval: float = 15.0,
) -> AsyncGenerator[str, None]:
    """Stream events from an asyncio.Queue with periodic keep-alive pings."""
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=ping_interval)
                yield format_sse(event)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
    except asyncio.CancelledError:
        pass


__all__ = ["format_sse", "sse_event_stream"]
