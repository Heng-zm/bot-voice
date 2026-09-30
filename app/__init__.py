"""Package marker."""

from __future__ import annotations

# Ensure app.legacy is accessible on app package for mocking and backward compatibility
from app import legacy  # noqa: F401

__all__ = ["legacy"]
