"""Telegra.ph Instant View generation service for clean, ad-free article reading."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import html
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_TITLE_CHARS = 256
_MAX_AUTHOR_CHARS = 128
_MAX_URL_CHARS = 512
_MAX_CONTENT_CHARS = 50000
_ACCESS_TOKEN_ENV = "TELEGRAPH_ACCESS_TOKEN"

_account_lock: Optional[asyncio.Lock] = None
_account_lock_loop: Optional[asyncio.AbstractEventLoop] = None
_telegraph_client = None
_account_ready = False


def _get_account_lock() -> asyncio.Lock:
    """Lazily obtain an asyncio.Lock tied to the active event loop."""
    global _account_lock, _account_lock_loop
    current_loop = asyncio.get_running_loop()
    if _account_lock is None or _account_lock_loop != current_loop or current_loop.is_closed():
        _account_lock = asyncio.Lock()
        _account_lock_loop = current_loop
    return _account_lock


def _escape_text(value: str) -> str:
    return html.escape(str(value or ""), quote=False)


def _escape_attr(value: str) -> str:
    return html.escape(str(value or ""), quote=True)


def _build_html_content(
    content_text: str = "",
    image_url: Optional[str] = None,
    source_url: Optional[str] = None,
) -> str:
    """Build valid Telegra.ph HTML body elements."""
    html_blocks: list[str] = []

    # Lead Image
    if image_url and isinstance(image_url, str) and image_url.strip():
        clean_img = image_url.strip()
        if clean_img.startswith(("http://", "https://", "/file/")):
            html_blocks.append(f'<img src="{_escape_attr(clean_img)}"/>')

    # Body paragraphs
    raw_text = str(content_text or "")
    if len(raw_text) > _MAX_CONTENT_CHARS:
        raw_text = raw_text[:_MAX_CONTENT_CHARS].rstrip() + "\n...(ខ្លឹមសារត្រូវបានកាត់ខ្លី)"

    for paragraph in raw_text.split("\n"):
        p = paragraph.strip()
        if p:
            clean_p = _escape_text(html.unescape(p))
            html_blocks.append(f"<p>{clean_p}</p>")

    # Source Link
    if source_url and isinstance(source_url, str) and source_url.strip():
        clean_source = source_url.strip()
        if clean_source.startswith(("http://", "https://")):
            html_blocks.append("<hr/>")
            html_blocks.append(
                f'<p><em>🔗 <a href="{_escape_attr(clean_source)}">អានប្រភពដើម (Read Original Source)</a></em></p>'
            )

    return "".join(html_blocks)


def _get_telegraph():
    global _telegraph_client, _account_ready
    if _telegraph_client is None:
        try:
            from telegraph import Telegraph

            env_tok = os.environ.get(_ACCESS_TOKEN_ENV)
            _telegraph_client = Telegraph(access_token=env_tok or None)
            _account_ready = bool(_telegraph_client.get_access_token())
        except Exception as exc:
            logger.warning("Telegraph package not initialized: %s", exc)
            return None
    return _telegraph_client


async def _ensure_account(force_refresh: bool = False) -> bool:
    global _account_ready
    tg = _get_telegraph()
    if not tg:
        return False

    if _account_ready and not force_refresh:
        return True

    async with _get_account_lock():
        if _account_ready and not force_refresh:
            return True
        try:
            await asyncio.to_thread(
                tg.create_account,
                short_name="KhmerVoiceBot",
                author_name="Bot Voice",
            )
            _account_ready = True
            return True
        except Exception as exc:
            logger.warning("Failed creating Telegraph account: %s", exc)
            return False


async def get_telegraph_url(
    title: str,
    content_text: str = "",
    image_url: Optional[str] = None,
    source_url: Optional[str] = None,
    author_name: str = "Bot Voice",
    author_url: Optional[str] = None,
) -> str:
    """Generate a Telegraph Instant View article page."""
    if not title or not str(title).strip():
        return ""

    safe_title = " ".join(str(title).split())
    if len(safe_title) > _MAX_TITLE_CHARS:
        safe_title = safe_title[: _MAX_TITLE_CHARS - 1].rstrip() + "…"

    html_content = _build_html_content(content_text, image_url, source_url)
    if not html_content.strip():
        html_content = "<p>&nbsp;</p>"

    clean_author = " ".join(str(author_name).split()) if author_name else "Bot Voice"
    if len(clean_author) > _MAX_AUTHOR_CHARS:
        clean_author = clean_author[: _MAX_AUTHOR_CHARS - 1].rstrip() + "…"

    clean_author_url = None
    target_url = author_url or source_url
    if target_url and isinstance(target_url, str) and target_url.strip().startswith(("http://", "https://")):
        clean_author_url = target_url.strip()
        if len(clean_author_url) > _MAX_URL_CHARS:
            clean_author_url = None

    create_kwargs = {
        "title": safe_title,
        "html_content": html_content,
        "author_name": clean_author,
    }
    if clean_author_url:
        create_kwargs["author_url"] = clean_author_url

    try:
        ok = await _ensure_account()
        if not ok:
            return ""

        tg = _get_telegraph()
        if not tg:
            return ""

        response = await asyncio.to_thread(tg.create_page, **create_kwargs)
        if isinstance(response, dict) and response.get("url"):
            return response["url"]
    except Exception as exc:
        err_msg = str(exc).upper()
        if "ACCESS_TOKEN" in err_msg or "TOKEN_INVALID" in err_msg:
            with suppress(Exception):
                await _ensure_account(force_refresh=True)
                tg = _get_telegraph()
                if tg:
                    response = await asyncio.to_thread(tg.create_page, **create_kwargs)
                    if isinstance(response, dict):
                        return response.get("url", "")
        logger.warning("Telegraph page creation failed: %s", exc)

    return ""


__all__ = ["get_telegraph_url"]
