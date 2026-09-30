"""Backward compatibility shim for app.api.routes.system."""

from __future__ import annotations

from app.api.health import health_check, router, system_metrics_endpoint

__all__ = ["health_check", "router", "system_metrics_endpoint"]
