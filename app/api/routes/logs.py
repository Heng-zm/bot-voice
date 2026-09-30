"""Backward compatibility shim for app.api.routes.logs."""

from __future__ import annotations

from app.api.logs import (
    get_live_logs_dashboard_html,
    get_live_logs_stream,
    get_recent_logs_api,
    live_logs_dashboard,
    router,
    stream_logs_sse,
)

__all__ = [
    "get_live_logs_dashboard_html",
    "get_live_logs_stream",
    "get_recent_logs_api",
    "live_logs_dashboard",
    "router",
    "stream_logs_sse",
]
