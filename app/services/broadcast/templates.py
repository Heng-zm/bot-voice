"""Broadcast template management, persistence, and serialization."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import html
import json
import logging
import re
import threading
import time
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)

BROADCAST_TEMPLATES_SETTING_KEY = "broadcast_templates_json"
BROADCAST_TEMPLATE_LIBRARY_MAX = 20
BROADCAST_TEMPLATE_TITLE_MAX = 48
BROADCAST_TEMPLATE_PREVIEW_MAX = 700
BROADCAST_TEMPLATE_BUTTON_TITLE_MAX = 34

_HEX_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

_KHMER_DIACRITICS = set(
    "\u17b4\u17b5\u17b6\u17b7\u17b8\u17b9\u17ba\u17bb\u17bc\u17bd\u17be\u17bf"
    "\u17c0\u17c1\u17c2\u17c3\u17c4\u17c5\u17c6\u17c7\u17c8\u17c9\u17ca\u17cb"
    "\u17cc\u17cd\u17ce\u17cf\u17d0\u17d1\u17d2\u17d3\u17dd"
)

# Local memory cache for templates
_TEMPLATES_CACHE: list[dict[str, Any]] | None = None
_TEMPLATES_CACHE_TS: float = 0.0
_CACHE_TTL_S = 15.0
_LOCK = threading.RLock()


# ============================================================================
# STRING & ID SANITIZATION HELPERS
# ============================================================================

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


def _clean_trailing_khmer(text: str) -> str:
    """Strip trailing incomplete Khmer diacritics and punctuation."""
    t = text.rstrip(" .,;:!?-–—\n\r")
    while t and t[-1] in _KHMER_DIACRITICS:
        t = t[:-1]
    return t.rstrip(" .,;:!?-–—\n\r")


def broadcast_template_clean_preview(
    text: Any,
    *,
    max_len: int = BROADCAST_TEMPLATE_TITLE_MAX,
) -> str:
    """Strip markup and format a clean preview snippet."""
    clean = str(text or "").strip()
    clean = _TAG_RE.sub(" ", clean)
    clean = html.unescape(clean)
    clean = _SPACE_RE.sub(" ", clean).strip()

    if not clean:
        clean = "គ្មានចំណងជើង (No Caption)"

    if len(clean) > max_len:
        cut = clean[: max(1, max_len - 1)]
        space_idx = cut.rfind(" ", int(max_len * 0.6), max_len)
        if space_idx != -1:
            cut = cut[:space_idx]
        clean = _clean_trailing_khmer(cut) + "…"

    return clean


def broadcast_template_fingerprint(payload: dict[str, Any]) -> str:
    """Generate SHA-256 fingerprint for deduplication."""
    clean = {
        "photo_file_id": str(payload.get("photo_file_id") or ""),
        "caption": str(payload.get("caption") or "").strip(),
        "text": str(payload.get("text") or "").strip(),
        "parse_mode": str(payload.get("parse_mode") or "auto").lower(),
        "link_preview": bool(payload.get("link_preview", True)),
        "reply_markup": str(payload.get("reply_markup") or ""),
    }
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# PERSISTENT STORAGE LAYER (SUPABASE + MEMORY CACHE)
# ============================================================================

def _read_templates_raw_sync() -> list[dict[str, Any]]:
    """Synchronously load template list from settings store or cache."""
    global _TEMPLATES_CACHE, _TEMPLATES_CACHE_TS
    now = time.monotonic()

    with _LOCK:
        if _TEMPLATES_CACHE is not None and (now - _TEMPLATES_CACHE_TS) < _CACHE_TTL_S:
            return [dict(t) for t in _TEMPLATES_CACHE]

    raw_json: Any = None
    with suppress(Exception):
        from app.services.settings.store import get_settings_store
        store = get_settings_store()
        raw = store.get_text_sync(BROADCAST_TEMPLATES_SETTING_KEY, "[]")
        if raw:
            raw_json = json.loads(raw)

    templates: list[dict[str, Any]] = []
    if isinstance(raw_json, list):
        for item in raw_json:
            if isinstance(item, dict) and item.get("id"):
                templates.append(dict(item))

    with _LOCK:
        _TEMPLATES_CACHE = [dict(t) for t in templates]
        _TEMPLATES_CACHE_TS = now

    return templates


def _write_templates_raw_sync(templates: list[dict[str, Any]], admin_id: int | None = None) -> bool:
    """Synchronously persist template list to settings store and update cache."""
    global _TEMPLATES_CACHE, _TEMPLATES_CACHE_TS
    bounded = templates[:BROADCAST_TEMPLATE_LIBRARY_MAX]
    serialized = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))

    success = False
    with suppress(Exception):
        from app.services.settings.store import get_settings_store
        store = get_settings_store()
        # Direct synchronous database write
        if hasattr(store, "_write_sync") and store._get_supabase() is not None:
            store._write_sync(BROADCAST_TEMPLATES_SETTING_KEY, serialized, admin_id)
            success = True
        elif hasattr(store, "set_text_sync"):
            store.set_text_sync(BROADCAST_TEMPLATES_SETTING_KEY, serialized)
            success = True

    with _LOCK:
        _TEMPLATES_CACHE = [dict(t) for t in bounded]
        _TEMPLATES_CACHE_TS = time.monotonic()

    return success


def db_broadcast_template_list(limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve all saved broadcast templates sorted by recent update."""
    templates = _read_templates_raw_sync()
    return templates[:max(1, limit)]


async def db_broadcast_template_list_async(limit: int = 20) -> list[dict[str, Any]]:
    """Asynchronously retrieve all saved broadcast templates."""
    return await asyncio.to_thread(db_broadcast_template_list, limit)


def db_broadcast_template_get(tpl_id: str) -> dict[str, Any] | None:
    """Retrieve single broadcast template by ID."""
    clean_id = broadcast_template_safe_id(tpl_id)
    if not clean_id:
        return None

    templates = _read_templates_raw_sync()
    for t in templates:
        if t.get("id") == clean_id:
            return dict(t)
    return None


async def db_broadcast_template_get_async(tpl_id: str) -> dict[str, Any] | None:
    """Asynchronously retrieve single broadcast template by ID."""
    return await asyncio.to_thread(db_broadcast_template_get, tpl_id)


def db_broadcast_template_save(
    payload: dict[str, Any],
    admin_id: int,
    title: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Save or update broadcast template in library.

    Returns:
        (success: bool, status_message: str, template: dict | None)
    """
    if not isinstance(payload, dict):
        return False, "Invalid payload object.", None

    has_content = bool(payload.get("text") or payload.get("caption") or payload.get("photo_file_id"))
    if not has_content:
        return False, "Cannot save empty template.", None

    tpl_id = broadcast_template_fingerprint(payload)
    preview_src = payload.get("caption") or payload.get("text") or ""
    clean_title = (
        broadcast_template_clean_preview(title, max_len=BROADCAST_TEMPLATE_TITLE_MAX)
        if title
        else broadcast_template_clean_preview(preview_src, max_len=BROADCAST_TEMPLATE_BUTTON_TITLE_MAX)
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    template_record: dict[str, Any] = {
        "id": tpl_id,
        "title": clean_title,
        "text": str(payload.get("text") or ""),
        "caption": str(payload.get("caption") or ""),
        "photo_file_id": str(payload.get("photo_file_id") or ""),
        "parse_mode": str(payload.get("parse_mode") or "auto"),
        "link_preview": bool(payload.get("link_preview", True)),
        "created_by": admin_id,
        "updated_at": now_iso,
    }

    templates = _read_templates_raw_sync()
    is_update = False

    # Check if template with identical fingerprint already exists
    new_list: list[dict[str, Any]] = [template_record]
    for existing in templates:
        if existing.get("id") == tpl_id:
            is_update = True
            # Keep original creation timestamp if present
            if existing.get("created_at"):
                template_record["created_at"] = existing["created_at"]
        else:
            new_list.append(existing)

    if not is_update and not template_record.get("created_at"):
        template_record["created_at"] = now_iso

    ok = _write_templates_raw_sync(new_list, admin_id=admin_id)
    msg = "updated existing template" if is_update else "template saved"
    return ok, msg, template_record


async def db_broadcast_template_save_async(
    payload: dict[str, Any],
    admin_id: int,
    title: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Asynchronously save or update broadcast template."""
    return await asyncio.to_thread(db_broadcast_template_save, payload, admin_id, title)


def db_broadcast_template_delete(tpl_id: str, admin_id: int) -> tuple[bool, str]:
    """Delete a broadcast template by ID."""
    clean_id = broadcast_template_safe_id(tpl_id)
    if not clean_id:
        return False, "Invalid template ID."

    templates = _read_templates_raw_sync()
    filtered = [t for t in templates if t.get("id") != clean_id]

    if len(filtered) == len(templates):
        return False, "Template not found."

    ok = _write_templates_raw_sync(filtered, admin_id=admin_id)
    return ok, "deleted successfully" if ok else "failed to write"


async def db_broadcast_template_delete_async(tpl_id: str, admin_id: int) -> tuple[bool, str]:
    """Asynchronously delete a broadcast template."""
    return await asyncio.to_thread(db_broadcast_template_delete, tpl_id, admin_id)


# ============================================================================
# TEMPLATE FORMATTERS & UI KEYBOARD GENERATORS
# ============================================================================

def template_payload_from_dict(tpl: dict[str, Any]) -> dict[str, Any]:
    """Convert stored template record into live broadcast payload."""
    return {
        "text": str(tpl.get("text") or ""),
        "caption": str(tpl.get("caption") or ""),
        "photo_file_id": str(tpl.get("photo_file_id") or ""),
        "parse_mode": str(tpl.get("parse_mode") or "auto"),
        "link_preview": bool(tpl.get("link_preview", True)),
    }


def template_button_title(tpl: dict[str, Any]) -> str:
    """Format an emoji-badged button title for the templates keyboard."""
    title = str(tpl.get("title") or "").strip()
    has_photo = bool(tpl.get("photo_file_id"))
    prefix = "🖼️ " if has_photo else "💬 "

    if not title:
        preview = tpl.get("caption") or tpl.get("text") or "គំរូគ្មានចំណងជើង"
        title = broadcast_template_clean_preview(preview, max_len=BROADCAST_TEMPLATE_BUTTON_TITLE_MAX - 3)

    if len(title) > (BROADCAST_TEMPLATE_BUTTON_TITLE_MAX - 3):
        title = title[: BROADCAST_TEMPLATE_BUTTON_TITLE_MAX - 4].rstrip() + "…"

    return f"{prefix}{title}"


def template_delete_confirm_text(tpl: dict[str, Any]) -> str:
    """Format confirmation prompt for deleting a template."""
    title = html.escape(str(tpl.get("title") or "គំរូនេះ"))
    tpl_id = html.escape(str(tpl.get("id") or ""))
    has_photo = "រូបភាព + ចំណងជើង" if tpl.get("photo_file_id") else "អត្ថបទសុទ្ធ"

    return (
        f"🗑️ <b>បញ្ជាក់ការលុបគំរូផ្សាយសារ (Delete Template)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>ឈ្មោះគំរូ:</b> <code>{title}</code>\n"
        f"• <b>ប្រភេទ:</b> {has_photo}\n"
        f"• <b>ID:</b> <code>{tpl_id}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"តើអ្នកពិតជាចង់លុបគំរូផ្សាយសារនេះចេញពីបណ្ណាល័យមែនទេ?"
    )


def get_broadcast_template_delete_confirm_kb(tpl_id: str) -> InlineKeyboardMarkup:
    """Generate confirmation inline keyboard for template deletion."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🗑️ លុបចេញ (Delete)", callback_data=f"bc_tpl_del:{tpl_id}"),
            InlineKeyboardButton("❌ មិនលុប (Keep)", callback_data="bc_templates"),
        ],
        [
            InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ (Back)", callback_data="bc_templates"),
        ],
    ])


def get_broadcast_templates_kb(templates: list[dict[str, Any]], page: int = 0) -> InlineKeyboardMarkup:
    """Generate paginated template selection keyboard."""
    per_page = 5
    total = len(templates)
    start_idx = max(0, page * per_page)
    paged_items = templates[start_idx : start_idx + per_page]

    buttons: list[list[InlineKeyboardButton]] = []

    for tpl in paged_items:
        tid = tpl.get("id", "")
        title = template_button_title(tpl)
        buttons.append([
            InlineKeyboardButton(title, callback_data=f"bc_tpl_use:{tid}"),
            InlineKeyboardButton("🗑️", callback_data=f"bc_tpl_delask:{tid}"),
        ])

    # Pagination navigation bar
    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("⬅️ មុន", callback_data=f"bc_templates_page:{page - 1}"))
    if (start_idx + per_page) < total:
        nav_row.append(InlineKeyboardButton("បន្ទាប់ ➡️", callback_data=f"bc_templates_page:{page + 1}"))

    if nav_row:
        buttons.append(nav_row)

    buttons.append([
        InlineKeyboardButton("💾 រក្សាទុកសារបច្ចុប្បន្នជា Template", callback_data="bc_save_template"),
    ])
    buttons.append([
        InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ (Back)", callback_data="bc_cancel"),
    ])

    return InlineKeyboardMarkup(buttons)


# ============================================================================
# TRANSITIONAL COMPATIBILITY ALIASES
# ============================================================================

_broadcast_template_button_title = template_button_title
_broadcast_template_delete_confirm_text = template_delete_confirm_text
_broadcast_template_payload_from_template = template_payload_from_dict


__all__ = [
    "BROADCAST_TEMPLATES_SETTING_KEY",
    "BROADCAST_TEMPLATE_BUTTON_TITLE_MAX",
    "BROADCAST_TEMPLATE_LIBRARY_MAX",
    "BROADCAST_TEMPLATE_PREVIEW_MAX",
    "BROADCAST_TEMPLATE_TITLE_MAX",
    "_broadcast_template_button_title",
    "_broadcast_template_delete_confirm_text",
    "_broadcast_template_payload_from_template",
    "broadcast_template_clean_preview",
    "broadcast_template_fingerprint",
    "broadcast_template_safe_id",
    "broadcast_template_safe_int",
    "db_broadcast_template_delete",
    "db_broadcast_template_delete_async",
    "db_broadcast_template_get",
    "db_broadcast_template_get_async",
    "db_broadcast_template_list",
    "db_broadcast_template_list_async",
    "db_broadcast_template_save",
    "db_broadcast_template_save_async",
    "get_broadcast_template_delete_confirm_kb",
    "get_broadcast_templates_kb",
    "template_button_title",
    "template_delete_confirm_text",
    "template_payload_from_dict",
]