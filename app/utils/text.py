"""Telegram text escaping, HTML entity-safe cutting, and message pagination."""

from __future__ import annotations

import html
import re

TELEGRAM_MSG_LIMIT = 4096


def truncate_text(text: str, max_len: int, ellipsis: str = "…") -> str:
    """Truncate text to max_len with optional ellipsis."""
    clean = str(text or "").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max(1, max_len - len(ellipsis))].rstrip() + ellipsis


def take_escaped_prefix(text: str, escaped_limit: int) -> tuple[str, str]:
    """Return the largest raw prefix whose html.escape() fits escaped_limit."""
    if escaped_limit <= 0:
        return "", text
    if len(html.escape(text)) <= escaped_limit:
        return text, ""

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(html.escape(text[:mid])) <= escaped_limit:
            lo = mid
        else:
            hi = mid - 1
    if lo <= 0:
        lo = 1
    return text[:lo], text[lo:]


def html_safe_cut(text: str, limit: int) -> int:
    """Pick a cut position that does not land inside an HTML tag or entity."""
    if len(text) <= limit:
        return len(text)
    cut = max(1, min(limit, len(text)))

    # Avoid splitting a tag token such as <code> or </b>.
    last_lt = text.rfind("<", 0, cut)
    last_gt = text.rfind(">", 0, cut)
    if last_lt > last_gt:
        cut = max(1, last_lt)

    # Avoid splitting an HTML entity such as &amp; or &#123;.
    last_amp = text.rfind("&", 0, cut)
    last_semi = text.rfind(";", 0, cut)
    if last_amp > last_semi and cut - last_amp <= 12:
        cut = max(1, last_amp)

    # Prefer natural whitespace / newline boundaries
    for sep in ("\n\n", "\n", " "):
        boundary = text.rfind(sep, 0, cut)
        if boundary > 0:
            return boundary
    return cut


def paginate_pre_html(text: str, limit: int = TELEGRAM_MSG_LIMIT, header: str = "") -> list[str]:
    """Paginate plain text as HTML <pre> blocks without splitting entities/tags."""
    text = str(text or "").strip()
    header = str(header or "")
    wrapper_len = len("<pre></pre>")
    body_limit = max(1, int(limit or TELEGRAM_MSG_LIMIT) - len(header) - wrapper_len)
    pages: list[str] = []

    while text:
        raw, rest = take_escaped_prefix(text, body_limit)
        if rest:
            # Prefer a clean boundary if it still leaves useful content.
            for sep in ("\n", " "):
                cut = raw.rfind(sep)
                if cut > 0 and len(html.escape(raw[:cut].rstrip())) <= body_limit:
                    rest = raw[cut:] + rest
                    raw = raw[:cut].rstrip()
                    break
        raw = raw.rstrip()
        if raw:
            pages.append(f"{header}<pre>{html.escape(raw)}</pre>")
        text = rest.lstrip()

    if not pages and header:
        pages.append(header.rstrip())
    return pages


def paginate_html(text: str, limit: int = TELEGRAM_MSG_LIMIT, header: str = "") -> list[str]:
    """Split already-escaped Telegram HTML without cutting through tags."""
    text = str(text or "").strip()
    header = str(header or "")
    if not text and not header:
        return []

    limit = max(1, int(limit or TELEGRAM_MSG_LIMIT))
    body_limit = max(1, limit - len(header))
    pages: list[str] = []
    current = ""

    blocks = re.split(r"(\n{2,})", text)
    for block in blocks:
        if not block:
            continue
        candidate = current + block
        if len(candidate) <= body_limit:
            current = candidate
            continue
        if current.strip():
            pages.append(header + current.strip())
            current = ""

        block = block.lstrip()
        while len(block) > body_limit:
            cut = html_safe_cut(block, body_limit)
            piece = block[:cut].strip()
            if piece:
                pages.append(header + piece)
            block = block[cut:].lstrip()
        current = block

    if current.strip():
        pages.append(header + current.strip())
    if not pages and header:
        pages.append(header.rstrip())
    return [p for p in pages if p]


__all__ = [
    "TELEGRAM_MSG_LIMIT",
    "html_safe_cut",
    "paginate_html",
    "paginate_pre_html",
    "take_escaped_prefix",
    "truncate_text",
]
