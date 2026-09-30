"""Admin feature service managing bot operations, system maintenance, and performance tuning."""

from __future__ import annotations

from typing import Any

from app.services.admin.optimization import (
    run_system_cleanup_sync,
    run_system_optimization_async,
)


class AdminService:
    """Consolidated admin business service."""

    def __init__(self) -> None:
        pass

    def run_cleanup(self) -> dict[str, Any]:
        """Execute synchronous system cleanup (temp file sweeping)."""
        return run_system_cleanup_sync()

    async def run_optimization(self) -> dict[str, Any]:
        """Execute full asynchronous system optimization (garbage collection, cache trimming, temp sweep)."""
        return await run_system_optimization_async()


admin_service = AdminService()

__all__ = [
    "AdminService",
    "admin_service",
    "run_system_cleanup_sync",
    "run_system_optimization_async",
]
