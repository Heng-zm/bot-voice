"""Admin feature package for system telemetry, dashboard controls, and bot configuration."""

from __future__ import annotations

from app.features.admin.dashboard import (
    build_admin_bakong_text,
    build_admin_bot_mode_text,
    build_admin_home_full_text,
    build_admin_podcast_text,
    build_admin_quick_actions_text,
    build_admin_ui_hub_text,
    get_admin_bakong_kb,
    get_admin_bot_mode_kb,
    get_admin_dashboard_full_kb,
    get_admin_podcast_kb,
    get_admin_quick_actions_kb,
    get_admin_ui_hub_kb,
)
from app.features.admin.handlers import handle_admin_callback
from app.features.admin.service import (
    AdminService,
    admin_service,
    run_system_cleanup_sync,
    run_system_optimization_async,
)

__all__ = [
    "AdminService",
    "admin_service",
    "build_admin_bakong_text",
    "build_admin_bot_mode_text",
    "build_admin_home_full_text",
    "build_admin_podcast_text",
    "build_admin_quick_actions_text",
    "build_admin_ui_hub_text",
    "get_admin_bakong_kb",
    "get_admin_bot_mode_kb",
    "get_admin_dashboard_full_kb",
    "get_admin_podcast_kb",
    "get_admin_quick_actions_kb",
    "get_admin_ui_hub_kb",
    "handle_admin_callback",
    "run_system_cleanup_sync",
    "run_system_optimization_async",
]
