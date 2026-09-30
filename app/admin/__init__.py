"""Admin management, dashboard, optimization, and handlers package."""

from __future__ import annotations

from app.features.admin.service import AdminService, admin_service
from app.services.admin.dashboard import (
    build_admin_home_full_text,
    get_admin_dashboard_full_kb,
)
from app.services.admin.handlers import handle_admin_callback
from app.services.admin.optimization import run_system_optimization_async

__all__ = [
    "AdminService",
    "admin_service",
    "build_admin_home_full_text",
    "get_admin_dashboard_full_kb",
    "handle_admin_callback",
    "run_system_optimization_async",
]
