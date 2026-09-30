"""Admin Telegram command and callback handlers."""

from __future__ import annotations

from app.services.admin.handlers import handle_admin_callback

__all__ = [
    "handle_admin_callback",
]
