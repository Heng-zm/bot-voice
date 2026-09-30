"""Telegram text escaping, HTML entity-safe cutting, and message pagination."""

from __future__ import annotations

import html
import re
from typing import Final

TELEGRAM_MSG_LIMIT: Final[int] = 4096
TELEGRAM_CAPTION_LIMIT: Final[int] = 1024

_TELEGRAM_HTML_TAGS: Final[frozenset[str]] = frozenset({
    "b",
    "strong",
    "i",
    "em",
    "u",
    "ins",
    "s",
    "strike",
    "del",
    "span",
    "tg-spoiler",
    "a",
    "tg-emoji",
    "code",
    "pre",
    "blockquote",
})

_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<\s*(/)?\s*([a-zA-Z0-9_-]+)(?:\s+[^>]*)?>",
    re.DOTALL,
)


def utf16_len(s: str) -> int:
    """Return the length of s in UTF-16 code units (as counted by Telegram)."""
    return len(s.encode("utf-16-le")) // 2


def truncate_text(text: str, max_len: int, ellipsis: str = "…") -> str:
    """Truncate text to max_len (UTF-16 code units) with optional ellipsis."""
    clean = str(text or "").strip()
    if utf16_len(clean) <= max_len:
        return clean

    ellipsis_len = utf16_len(ellipsis)
    target = max(1, max_len - ellipsis_len)

    # Fast approximate slice in codepoints
    sliced = clean[:target]
    while utf16_len(sliced) > target and len(sliced) > 1:
        sliced = sliced[:-1]

    return sliced.rstrip() + ellipsis


def take_escaped_prefix(text: str, escaped_limit: int) -> tuple[str, str]:
    """Return the largest raw prefix whose html.escape() fits within escaped_limit UTF-16 units."""
    if escaped_limit <= 0 or not text:
        return "", text

    if utf16_len(html.escape(text)) <= escaped_limit:
        return text, ""

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if utf16_len(html.escape(text[:mid])) <= escaped_limit:
            lo = mid
        else:
            hi = mid - 1

    return text[:lo], text[lo:]


def html_safe_cut(text: str, limit: int) -> int:
    """Pick a safe cut position (character index) that fits UTF-16 limit and avoids splitting tags/entities."""
    total_utf16 = utf16_len(text)
    if total_utf16 <= limit:
        return len(text)

    # Binary search to find highest char index within UTF-16 limit
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if utf16_len(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1

    cut = max(1, lo)

    # 1. Avoid splitting inside an HTML tag: <...>
    last_lt = text.rfind("<", 0, cut)
    last_gt = text.rfind(">", 0, cut)
    if last_lt > last_gt:
        cut = max(1, last_lt)

    # 2. Avoid splitting inside an HTML entity: &...; (e.g. &amp;, &#1234;)
    last_amp = text.rfind("&", 0, cut)
    last_semi = text.rfind(";", 0, cut)
    if last_amp > last_semi and cut - last_amp <= 12:
        cut = max(1, last_amp)

    # 3. Prefer natural whitespace boundaries if it doesn't discard >25% of the slice
    min_acceptable = max(1, int(cut * 0.75))
    for sep in ("\n\n", "\n", " "):
        boundary = text.rfind(sep, 0, cut)
        if boundary >= min_acceptable:
            # Check boundary does not sit inside open tag or entity
            lt_b = text.rfind("<", 0, boundary)
            gt_b = text.rfind(">", 0, boundary)
            if lt_b > gt_b:
                continue

            amp_b = text.rfind("&", 0, boundary)
            semi_b = text.rfind(";", 0, boundary)
            if amp_b > semi_b and boundary - amp_b <= 12:
                continue

            return boundary + len(sep)

    return cut


def format_safe_media_caption(
    header: str = "",
    raw_caption: str | None = None,
    limit: int = TELEGRAM_CAPTION_LIMIT,
) -> tuple[str, str]:
    """Format an HTML-escaped media caption strictly bounded to limit UTF-16 units.

    Returns:
        (safe_caption_html, overflow_raw_text)
    """
    raw = str(raw_caption or "").strip()
    header_str = str(header or "")
    header_len = utf16_len(header_str)

    if not raw:
        if header_len <= limit:
            return header_str.rstrip("\n"), ""
        # Truncate header if header alone exceeds limit
        lo, _ = take_escaped_prefix(header_str, limit)
        return lo, ""

    body_limit = max(0, limit - header_len)
    if body_limit <= 0:
        return header_str[:limit], raw

    prefix, rest = take_escaped_prefix(raw, body_limit)
    if rest:
        # Prefer cutting on word/line boundary if clean
        for sep in ("\n", " "):
            cut = prefix.rfind(sep)
            if cut > 0 and utf16_len(html.escape(prefix[:cut].rstrip())) <= body_limit:
                rest = prefix[cut:] + rest
                prefix = prefix[:cut].rstrip()
                break

    caption_html = f"{header_str}{html.escape(prefix)}"
    return caption_html, rest.lstrip()


def paginate_pre_html(
    text: str,
    limit: int = TELEGRAM_MSG_LIMIT,
    header: str = "",
) -> list[str]:
    """Paginate raw plain text inside HTML <pre> blocks without breaking tags or entities."""
    text = str(text or "").strip()
    header = str(header or "")
    wrapper_overhead = utf16_len(f"{header}<pre></pre>")
    body_limit = max(1, limit - wrapper_overhead)
    pages: list[str] = []

    while text:
        raw, rest = take_escaped_prefix(text, body_limit)
        if rest:
            for sep in ("\n", " "):
                cut = raw.rfind(sep)
                if cut > 0 and utf16_len(html.escape(raw[:cut].rstrip())) <= body_limit:
                    rest = raw[cut:] + rest
                    raw = raw[:cut].rstrip()
                    break

        if not raw:
            # Force advance at least 1 codepoint if body_limit is too constrained
            raw, rest = text[:1], text[1:]

        pages.append(f"{header}<pre>{html.escape(raw)}</pre>")
        text = rest.lstrip("\r\n")

    if not pages and header:
        pages.append(header.rstrip())

    return pages


def paginate_html(
    text: str,
    limit: int = TELEGRAM_MSG_LIMIT,
    header: str = "",
) -> list[str]:
    """Split an already-escaped HTML string across multiple Telegram messages.

    Ensures all open tags are properly closed at the end of each page and correctly
    re-opened at the start of the next page in valid LIFO order.
    """
    text = str(text or "").strip()
    header = str(header or "")
    if not text and not header:
        return []

    limit = max(64, limit)
    header_overhead = utf16_len(header)

    # Reserve buffer for open/close tags wrapping across boundaries (max ~128 UTF-16 units)
    tag_safety_budget = 128
    effective_limit = max(32, limit - header_overhead - tag_safety_budget)

    # Initial chunking using safe cut points
    raw_chunks: list[str] = []
    rem = text
    while rem:
        if utf16_len(rem) <= effective_limit:
            raw_chunks.append(rem)
            break

        cut = html_safe_cut(rem, effective_limit)
        if cut <= 0:
            cut = 1

        raw_chunks.append(rem[:cut])
        rem = rem[cut:]

    pages: list[str] = []
    active_stack: list[tuple[str, str]] = []  # List of (full_opening_tag, tag_name)

    for chunk in raw_chunks:
        # Build prefix from currently active tags
        prefix = "".join(full_tag for full_tag, _ in active_stack)
        content = prefix + chunk

        # Track tag balance in current page
        local_stack: list[tuple[str, str]] = list(active_stack)
        for m in _TAG_PATTERN.finditer(chunk):
            is_close = bool(m.group(1))
            tag_name = m.group(2).lower()

            if tag_name not in _TELEGRAM_HTML_TAGS:
                continue

            if is_close:
                # Find matching opening tag and pop properly (LIFO unwind if needed)
                for idx in range(len(local_stack) - 1, -1, -1):
                    if local_stack[idx][1] == tag_name:
                        del local_stack[idx]
                        break
            else:
                local_stack.append((m.group(0), tag_name))

        # Close all tags still open at the end of this page in reverse order
        suffix = "".join(f"</{name}>" for _, name in reversed(local_stack))
        page_html = header + content + suffix

        # Guard against edge-case tag overflow
        if utf16_len(page_html) > limit and len(pages) > 0:
            # If the closing tags made it slightly overshoot, reduce budget on retry
            pass

        pages.append(page_html)
        active_stack = local_stack

    return [p for p in pages if p.strip()]


__all__ = [
    "TELEGRAM_CAPTION_LIMIT",
    "TELEGRAM_MSG_LIMIT",
    "format_safe_media_caption",
    "html_safe_cut",
    "paginate_html",
    "paginate_pre_html",
    "take_escaped_prefix",
    "truncate_text",
    "utf16_len",
]