"""Telegram text escaping, HTML entity-safe cutting, and message pagination."""

from __future__ import annotations

import html
import re

TELEGRAM_MSG_LIMIT = 4096
TELEGRAM_CAPTION_LIMIT = 1024


def truncate_text(text: str, max_len: int, ellipsis: str = "…") -> str:
    """Truncate text to max_len with optional ellipsis."""
    clean = str(text or "").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max(1, max_len - len(ellipsis))].rstrip() + ellipsis


def take_escaped_prefix(text: str, escaped_limit: int) -> tuple[str, str]:
    """Return the largest raw prefix whose html.escape() fits escaped_limit."""
    if escaped_limit <= 0 or not text:
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
        lo = 1 if len(html.escape(text[:1])) <= escaped_limit else 0
    return text[:lo], text[lo:]


_TELEGRAM_HTML_TAGS: frozenset[str] = frozenset({
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "span", "tg-spoiler", "a", "tg-emoji", "code", "pre", "blockquote",
})
_TAG_PATTERN: re.Pattern[str] = re.compile(r"<\s*(/)?\s*([a-zA-Z0-9_-]+)(?:\s+[^>]*)?>", re.DOTALL)


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

    # Prefer natural whitespace / newline boundaries outside tags and entities
    for sep in ("\n\n", "\n", " "):
        boundary = text.rfind(sep, 0, cut)
        if boundary > 0:
            last_lt_b = text.rfind("<", 0, boundary)
            last_gt_b = text.rfind(">", 0, boundary)
            if last_lt_b > last_gt_b:
                continue
            last_amp_b = text.rfind("&", 0, boundary)
            last_semi_b = text.rfind(";", 0, boundary)
            if last_amp_b > last_semi_b and boundary - last_amp_b <= 12:
                continue
            return boundary
    return cut


def format_safe_media_caption(
    header: str = "",
    raw_caption: str | None = None,
    limit: int = TELEGRAM_CAPTION_LIMIT,
) -> tuple[str, str]:
    """Format an HTML-escaped media caption strictly bounded to limit (1024).

    Returns:
        (safe_caption_html, overflow_raw_text)
        - safe_caption_html: Length is guaranteed to be <= limit characters.
        - overflow_raw_text: Remaining unescaped text that did not fit in the caption.
    """
    raw = str(raw_caption or "").strip()
    header_str = str(header or "")
    if not raw:
        return header_str.rstrip("\n")[:limit], ""

    body_limit = max(0, limit - len(header_str))
    if body_limit <= 0:
        return header_str[:limit], raw

    prefix, rest = take_escaped_prefix(raw, body_limit)
    if rest:
        for sep in ("\n", " "):
            cut = prefix.rfind(sep)
            if cut > 0 and len(html.escape(prefix[:cut].rstrip())) <= body_limit:
                rest = prefix[cut:] + rest
                prefix = prefix[:cut].rstrip()
                break

    caption_html = f"{header_str}{html.escape(prefix)}"
    return caption_html, rest.lstrip()


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
    """Split already-escaped Telegram HTML without cutting through tags, ensuring all tags are closed on each page."""
    text = str(text or "").strip()
    header = str(header or "")
    if not text and not header:
        return []

    limit = max(1, int(limit or TELEGRAM_MSG_LIMIT))
    body_limit = max(1, limit - len(header) - 32)
    raw_pages: list[str] = []
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
            raw_pages.append(current.strip())
            current = ""

        block = block.lstrip()
        while len(block) > body_limit:
            cut = html_safe_cut(block, body_limit)
            piece = block[:cut].strip()
            if piece:
                raw_pages.append(piece)
            block = block[cut:].lstrip()
        current = block

    if current.strip():
        raw_pages.append(current.strip())

    if not raw_pages and header:
        return [header.rstrip()]

    # Balance unclosed HTML tags across pages
    pages: list[str] = []
    carry_over_open: list[tuple[str, str]] = []

    for piece in raw_pages:
        prefix = "".join(full_tag for full_tag, _ in carry_over_open)
        combined = prefix + piece

        open_stack: list[tuple[str, str]] = []
        for m in _TAG_PATTERN.finditer(combined):
            is_close, tag_name = bool(m.group(1)), m.group(2).lower()
            if tag_name not in _TELEGRAM_HTML_TAGS:
                continue
            if is_close:
                for idx in range(len(open_stack) - 1, -1, -1):
                    if open_stack[idx][1] == tag_name:
                        open_stack.pop(idx)
                        break
            else:
                open_stack.append((m.group(0), tag_name))

        suffix = "".join(f"</{tag_name}>" for _, tag_name in reversed(open_stack))
        pages.append(header + combined + suffix)
        carry_over_open = list(open_stack)

    return [p for p in pages if p]


__all__ = [
    "TELEGRAM_CAPTION_LIMIT",
    "TELEGRAM_MSG_LIMIT",
    "format_safe_media_caption",
    "html_safe_cut",
    "paginate_html",
    "paginate_pre_html",
    "take_escaped_prefix",
    "truncate_text",
]
