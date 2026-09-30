"""Shared common utilities, progress callbacks, and formatters for media downloaders."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from app.services.telegram.formatters import StatusCardAnimator, escape_html

logger = logging.getLogger("app.features.downloader.common")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal or invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()[:100]


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


__all__ = ["StatusCardAnimator", "escape_html", "format_file_size", "sanitize_filename"]
