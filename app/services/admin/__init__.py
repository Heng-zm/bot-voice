"""Full Option Admin Bot Controller package."""

from app.services.admin.dashboard import (
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
from app.services.admin.handlers import handle_admin_callback
from app.services.admin.optimization import (
    run_system_cleanup_sync,
    run_system_optimization_async,
)

__all__ = [
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
