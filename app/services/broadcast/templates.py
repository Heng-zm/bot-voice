"""Broadcast template management and serialization."""

from __future__ import annotations

import hashlib
import html
import json
import re
from typing import Any

BROADCAST_TEMPLATES_SETTING_KEY = "broadcast_templates_json"
BROADCAST_TEMPLATE_LIBRARY_MAX = 20
BROADCAST_TEMPLATE_TITLE_MAX = 48
BROADCAST_TEMPLATE_PREVIEW_MAX = 700
BROADCAST_TEMPLATE_BUTTON_TITLE_MAX = 34

_HEX_ID_RE = re.compile(r"[a-f0-9]{8,16}")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def broadcast_template_safe_id(value: Any) -> str:
    """Validate and sanitize hex template ID."""
    text = str(value or "").strip().lower()
    return text if _HEX_ID_RE.fullmatch(text) else ""


def broadcast_template_safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def broadcast_template_clean_preview(text: Any, *, max_len: int = BROADCAST_TEMPLATE_TITLE_MAX) -> str:
    """Strip markup and format clean preview snippet."""
    clean = str(text or "").strip()
    clean = _TAG_RE.sub(" ", clean)
    clean = html.unescape(clean)
    clean = _SPACE_RE.sub(" ", clean).strip()
    if not clean:
        clean = "គ្មាន Caption"
    if len(clean) > max_len:
        clean = clean[: max(1, max_len - 1)].rstrip() + "…"
    return clean


def broadcast_template_fingerprint(payload: dict[str, Any]) -> str:
    """Generate SHA-256 fingerprint for deduplication."""
    clean = {
        "photo_file_id": str(payload.get("photo_file_id") or ""),
        "caption": str(payload.get("caption") or ""),
        "text": str(payload.get("text") or ""),
        "parse_mode": str(payload.get("parse_mode") or "auto"),
        "link_preview": bool(payload.get("link_preview", True)),
    }
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "BROADCAST_TEMPLATES_SETTING_KEY",
    "BROADCAST_TEMPLATE_BUTTON_TITLE_MAX",
    "BROADCAST_TEMPLATE_LIBRARY_MAX",
    "BROADCAST_TEMPLATE_PREVIEW_MAX",
    "BROADCAST_TEMPLATE_TITLE_MAX",
    "broadcast_template_clean_preview",
    "broadcast_template_fingerprint",
    "broadcast_template_safe_id",
    "broadcast_template_safe_int",
]
