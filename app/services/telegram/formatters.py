"""Telegram HTML formatting, sanitization, and Markdown-to-HTML conversion utilities."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import html
import re
import time
from typing import Any, Callable, Sequence

# Allowed Telegram Bot API HTML tags
ALLOWED_TAGS = {
    "b",
    "strong",
    "i",
    "em",
    "u",
    "ins",
    "s",
    "strike",
    "del",
    "tg-spoiler",
    "a",
    "code",
    "pre",
    "blockquote",
    "tg-emoji",
}


def escape_html(text: str) -> str:
    """Escape special HTML characters (&, <, >) for Telegram."""
    if not text:
        return ""
    return html.escape(str(text), quote=False)


def unescape_html(text: str) -> str:
    """Unescape HTML entities back to plain text."""
    if not text:
        return ""
    return html.unescape(str(text))


def strip_html_tags(text: str) -> str:
    """Remove all HTML tags from text, leaving only the inner plain text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]*>", "", str(text))
    return html.unescape(clean)


def clean_speech_text(text: str) -> str:
    """Strip formatting, markdown, HTML, code blocks, and URLs for clean TTS reading."""
    if not text:
        return ""
    t = str(text)
    # 1. Remove code blocks entirely from voice speech
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = re.sub(r"`[^`]*`", " ", t)
    # 2. Remove Markdown links, keep anchor text
    t = re.sub(r"\[([^\]]+)\]\((?:https?://|tg://)[^)]+\)", r"\1", t)
    # 3. Remove raw URLs
    t = re.sub(r"https?://\S+", " ", t)
    # 4. Remove HTML tags
    t = re.sub(r"<[^>]*>", " ", t)
    # 5. Remove markdown symbols and separator lines
    t = re.sub(r"[*_~#|`>]", " ", t)
    t = re.sub(r"[-=]{3,}", " ", t)
    # 6. Unescape entities and collapse whitespace
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def _sanitize_tag_attributes(tag_name: str, full_tag: str) -> str:
    """Sanitize tag attributes down to only those allowed by Telegram Bot API."""
    if tag_name == "a":
        m = re.search(r'href=["\']([^"\']+)["\']', full_tag, flags=re.IGNORECASE)
        if m:
            clean_url = m.group(1).strip()
            return f'<a href="{html.escape(clean_url, quote=True)}">'
        return "<a>"

    if tag_name == "code":
        m = re.search(r'class=["\'](language-[a-zA-Z0-9_+-]+)["\']', full_tag, flags=re.IGNORECASE)
        if m:
            return f'<code class="{m.group(1)}">'
        return "<code>"

    if tag_name == "blockquote":
        if "expandable" in full_tag.lower():
            return "<blockquote expandable>"
        return "<blockquote>"

    if tag_name == "tg-emoji":
        m = re.search(r'emoji-id=["\'](\d+)["\']', full_tag, flags=re.IGNORECASE)
        if m:
            return f'<tg-emoji emoji-id="{m.group(1)}">'
        return ""

    # All other tags (b, i, u, s, pre, tg-spoiler, etc.) permit zero attributes in Telegram HTML
    return f"<{tag_name}>"


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard Markdown into Telegram-compliant HTML.

    Supported Markdown elements:
    - Code blocks (```lang ... ``` and ``` ... ```)
    - Inline code (`code`)
    - Headers (# H1 .. ###### H6) -> <b>Header</b>
    - Bold-Italic (***text*** or ___text___)
    - Bold (**text** or __text__)
    - Italic (*text* or _text_)
    - Strikethrough (~~text~~)
    - Spoilers (||text||)
    - Blockquotes (> quote)
    - Bullet lists (*, -, +) -> •
    - Links ([text](url))

    All standalone <, >, & in normal text and code are safely escaped.
    """
    if not text:
        return ""

    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # 1. Protect code blocks with collision-proof control tokens
    code_blocks: list[str] = []

    def _replace_code_block(match: re.Match) -> str:
        lang = (match.group(1) or "").strip().lower()
        code = match.group(2)
        code_escaped = html.escape(code.rstrip("\n"), quote=False)
        idx = len(code_blocks)
        if lang:
            tag = f'<pre><code class="language-{html.escape(lang, quote=True)}">{code_escaped}</code></pre>'
        else:
            tag = f"<pre><code>{code_escaped}</code></pre>"
        code_blocks.append(tag)
        return f"\x00TGCODEBLOCK{idx}\x00"

    raw = re.sub(r"```([a-zA-Z0-9_+-]+)?\n([\s\S]*?)```", _replace_code_block, raw)
    raw = re.sub(r"```([\s\S]*?)```", lambda m: _replace_code_block(re.match(r"()([\s\S]*)", m.group(1))), raw)

    # 2. Protect inline code with control tokens
    inline_codes: list[str] = []

    def _replace_inline_code(match: re.Match) -> str:
        code_content = match.group(1)
        code_escaped = html.escape(code_content, quote=False)
        idx = len(inline_codes)
        inline_codes.append(f"<code>{code_escaped}</code>")
        return f"\x00TGINLINECODE{idx}\x00"

    raw = re.sub(r"`([^`\n]+)`", _replace_inline_code, raw)

    # 3. Escape raw HTML special characters outside code
    raw = re.sub(r"&(?!amp;|lt;|gt;|quot;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", raw)
    raw = raw.replace("<", "&lt;").replace(">", "&gt;")

    # 4. Convert blockquotes: lines starting with &gt;
    lines = raw.split("\n")
    processed_lines: list[str] = []
    in_quote = False
    quote_acc: list[str] = []

    for line in lines:
        m_quote = re.match(r"^&gt;\s?(.*)$", line)
        if m_quote:
            in_quote = True
            quote_acc.append(m_quote.group(1))
        else:
            if in_quote:
                quote_text = "\n".join(quote_acc)
                processed_lines.append(f"<blockquote>{quote_text}</blockquote>")
                quote_acc = []
                in_quote = False
            processed_lines.append(line)
    if in_quote:
        quote_text = "\n".join(quote_acc)
        processed_lines.append(f"<blockquote>{quote_text}</blockquote>")

    raw = "\n".join(processed_lines)

    # 5. Convert Headers: '# Header' -> '<b>Header</b>'
    raw = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", raw, flags=re.MULTILINE)

    # 6. Convert Bullet lists: '* item', '- item', '+ item' -> '• item'
    raw = re.sub(r"^(\s*)[*\-+]\s+(.+)$", r"\1• \2", raw, flags=re.MULTILINE)

    # 7. Convert Markdown Links: [label](url) -> <a href="url">label</a>
    def _replace_link(match: re.Match) -> str:
        label = match.group(1)
        url = match.group(2).strip()
        url_clean = html.unescape(url)
        if url_clean.startswith(("http://", "https://", "tg://")):
            return f'<a href="{html.escape(url_clean, quote=True)}">{label}</a>'
        return f"{label} ({url})"

    raw = re.sub(r"\[([^\]\n]+)\]\(((?:https?://|tg://)[^\s)]+)\)", _replace_link, raw)

    # 8. Convert Spoilers: ||spoiler|| -> <tg-spoiler>spoiler</tg-spoiler>
    raw = re.sub(r"\|\|([\s\S]+?)\|\|", r"<tg-spoiler>\1</tg-spoiler>", raw)

    # 9. Convert Strikethrough: ~~text~~ -> <s>text</s>
    raw = re.sub(r"~~([^~\n]+)~~", r"<s>\1</s>", raw)

    # 10. Convert Bold + Italic: ***text*** or ___text___ -> <b><i>text</i></b>
    raw = re.sub(r"\*\*\*([^*\n]+)\*\*\*", r"<b><i>\1</i></b>", raw)
    raw = re.sub(r"___([^_\n]+)___", r"<b><i>\1</i></b>", raw)

    # 11. Convert Bold: **text** or __text__ -> <b>text</b>
    raw = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", raw)
    raw = re.sub(r"(?<!\w)__([^_\n]+)__(?!\w)", r"<b>\1</b>", raw)

    # 12. Convert Italic: *text* or _text_ -> <i>text</i>
    raw = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", raw)
    raw = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", raw)

    # 13. Restore protected inline code
    for i, code_tag in enumerate(inline_codes):
        raw = raw.replace(f"\x00TGINLINECODE{i}\x00", code_tag)

    # 14. Restore protected code blocks
    for i, code_block_tag in enumerate(code_blocks):
        raw = raw.replace(f"\x00TGCODEBLOCK{i}\x00", code_block_tag)

    # 15. Balance and sanitize HTML tags
    return balance_telegram_html(raw)


def balance_telegram_html(html_text: str) -> str:
    """Ensure all opened Telegram HTML tags are sanitized and closed in proper LIFO order."""
    if not html_text:
        return ""

    tag_regex = re.compile(r"(</?([a-zA-Z0-9_-]+)(?:\s+[^>]*)?>)")
    stack: list[str] = []
    output_tokens: list[str] = []
    last_idx = 0

    for match in tag_regex.finditer(html_text):
        # Escape any raw stray angle brackets in between tags
        segment = html_text[last_idx:match.start()]
        output_tokens.append(segment)

        full_tag = match.group(1)
        tag_name = match.group(2).lower()
        is_closing = full_tag.startswith("</")

        if tag_name not in ALLOWED_TAGS:
            output_tokens.append(html.escape(full_tag, quote=False))
        elif is_closing:
            if tag_name in stack:
                while stack:
                    top = stack.pop()
                    output_tokens.append(f"</{top}>")
                    if top == tag_name:
                        break
        else:
            sanitized = _sanitize_tag_attributes(tag_name, full_tag)
            if sanitized:
                stack.append(tag_name)
                output_tokens.append(sanitized)

        last_idx = match.end()

    output_tokens.append(html_text[last_idx:])

    while stack:
        top = stack.pop()
        output_tokens.append(f"</{top}>")

    return "".join(output_tokens)


def split_telegram_html(text: str, max_length: int = 3800) -> list[str]:
    """Split long Telegram HTML text into chunks strictly under max_length, preserving tag state across boundaries."""
    if not text:
        return []
    if len(text) <= max_length:
        return [balance_telegram_html(text)]

    def _split_safe(chunk_text: str, limit: int) -> list[str]:
        if len(chunk_text) <= limit:
            return [chunk_text]
        parts: list[str] = []
        rem = chunk_text
        while len(rem) > limit:
            cut = limit
            space_idx = rem.rfind(" ", 0, limit)
            if space_idx > limit // 3:
                cut = space_idx

            # Prevent cutting in the middle of a tag (<...>)
            open_tag = rem.rfind("<", 0, cut)
            close_tag = rem.rfind(">", 0, cut)
            if open_tag > close_tag:
                cut = open_tag

            # Prevent cutting in the middle of an HTML entity (&...;)
            open_entity = rem.rfind("&", 0, cut)
            close_entity = rem.rfind(";", 0, cut)
            if open_entity > close_entity and (cut - open_entity) < 10:
                cut = open_entity

            if cut <= 0:
                cut = limit

            parts.append(rem[:cut].strip())
            rem = rem[cut:].strip()
        if rem:
            parts.append(rem)
        return parts

    paragraphs = text.split("\n\n")
    raw_chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current_len + para_len > max_length:
            if len(para) > max_length:
                if current:
                    raw_chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                for piece in _split_safe(para, max_length - 50):
                    if current_len + len(piece) + 1 > max_length:
                        if current:
                            raw_chunks.append("\n".join(current))
                            current = []
                            current_len = 0
                    current.append(piece)
                    current_len += len(piece) + 1
            else:
                if current:
                    raw_chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                current.append(para)
                current_len += para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        raw_chunks.append("\n\n".join(current))

    # Inherit open tags across chunk boundaries
    final_chunks: list[str] = []
    tag_regex = re.compile(r"(</?([a-zA-Z0-9_-]+)(?:\s+[^>]*)?>)")
    carry_tags: list[str] = []

    for chunk in raw_chunks:
        prefix = "".join(f"<{t}>" for t in carry_tags)
        chunk_content = prefix + chunk

        # Track active tags in this chunk
        active_stack: list[str] = list(carry_tags)
        for m in tag_regex.finditer(chunk):
            t_name = m.group(2).lower()
            if t_name in ALLOWED_TAGS:
                if m.group(1).startswith("</"):
                    if t_name in active_stack:
                        while active_stack:
                            popped = active_stack.pop()
                            if popped == t_name:
                                break
                else:
                    active_stack.append(t_name)

        final_chunks.append(balance_telegram_html(chunk_content))
        carry_tags = active_stack

    return final_chunks


async def _execute_safe_send(coro_fn: Callable[[], Any], fallback_text_fn: Callable[[str], Any] | None = None) -> Any:
    """Execute Telegram send operation with automatic recovery from entity parse errors."""
    try:
        res = coro_fn()
        if asyncio.iscoroutine(res):
            return await res
        return res
    except Exception as exc:
        err = str(exc).lower()
        if ("can't parse entities" in err or "tag" in err or "entity" in err) and fallback_text_fn:
            try:
                res = fallback_text_fn()
                if asyncio.iscoroutine(res):
                    return await res
                return res
            except Exception:
                return None
        return None


async def send_split_html(
    target: Any,
    text: str,
    *,
    reply_markup: Any = None,
    disable_web_page_preview: bool = True,
    max_length: int = 3800,
    edit_initial: bool = False,
) -> list[Any]:
    """Send HTML text with automatic chunking and parse error protection."""
    chunks = split_telegram_html(text, max_length=max_length)
    if not chunks:
        chunks = [text[:max_length] if text else ""]

    sent_messages: list[Any] = []
    total = len(chunks)

    # 1. Edit existing message or callback query
    if edit_initial or (hasattr(target, "edit_message_text") and not hasattr(target, "reply_text")):
        first_km = reply_markup if (total == 1) else None
        edit_func = getattr(target, "edit_message_text", None) or getattr(target, "edit_text", None)
        if edit_func:
            sent = await _execute_safe_send(
                lambda: edit_func(
                    chunks[0],
                    parse_mode="HTML",
                    reply_markup=first_km,
                    disable_web_page_preview=disable_web_page_preview,
                ),
                lambda: edit_func(
                    strip_html_tags(chunks[0]),
                    reply_markup=first_km,
                    disable_web_page_preview=disable_web_page_preview,
                ),
            )
            if sent:
                sent_messages.append(sent)

        reply_target = getattr(target, "message", target)
        if total > 1 and hasattr(reply_target, "reply_text"):
            for idx, chunk in enumerate(chunks[1:], start=1):
                km = reply_markup if (idx == total - 1) else None
                sent_sub = await _execute_safe_send(
                    lambda c=chunk, k=km: reply_target.reply_text(
                        c,
                        parse_mode="HTML",
                        reply_markup=k,
                        disable_web_page_preview=disable_web_page_preview,
                    ),
                    lambda c=chunk, k=km: reply_target.reply_text(
                        strip_html_tags(c),
                        reply_markup=k,
                        disable_web_page_preview=disable_web_page_preview,
                    ),
                )
                if sent_sub:
                    sent_messages.append(sent_sub)
        return sent_messages

    # 2. Target has reply_text
    if hasattr(target, "reply_text"):
        for idx, chunk in enumerate(chunks):
            km = reply_markup if (idx == total - 1) else None
            sent = await _execute_safe_send(
                lambda c=chunk, k=km: target.reply_text(
                    c,
                    parse_mode="HTML",
                    reply_markup=k,
                    disable_web_page_preview=disable_web_page_preview,
                ),
                lambda c=chunk, k=km: target.reply_text(
                    strip_html_tags(c),
                    reply_markup=k,
                    disable_web_page_preview=disable_web_page_preview,
                ),
            )
            if sent:
                sent_messages.append(sent)
        return sent_messages

    return sent_messages


async def send_split_bot_message(
    bot: Any,
    chat_id: int,
    text: str,
    *,
    reply_markup: Any = None,
    disable_web_page_preview: bool = True,
    max_length: int = 3800,
) -> list[Any]:
    """Send HTML text to a chat_id using bot.send_message with automatic chunking."""
    chunks = split_telegram_html(text, max_length=max_length)
    if not chunks:
        chunks = [text[:max_length] if text else ""]

    sent_messages: list[Any] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        km = reply_markup if (idx == total - 1) else None
        sent = await _execute_safe_send(
            lambda c=chunk, k=km: bot.send_message(
                chat_id=chat_id,
                text=c,
                parse_mode="HTML",
                reply_markup=k,
                disable_web_page_preview=disable_web_page_preview,
            ),
            lambda c=chunk, k=km: bot.send_message(
                chat_id=chat_id,
                text=strip_html_tags(c),
                reply_markup=k,
                disable_web_page_preview=disable_web_page_preview,
            ),
        )
        if sent:
            sent_messages.append(sent)
    return sent_messages


def make_cool_progress_bar(percent: int | float | str | None, width: int = 10) -> str:
    """Generate a sleek, modern progress bar with filled and unfilled Unicode blocks."""
    w = max(5, min(20, int(width or 10)))
    try:
        if percent is None:
            pct = 0.0
        elif isinstance(percent, str):
            pct = float(percent.strip().rstrip("%"))
        else:
            pct = float(percent)
        if pct != pct or pct < 0:
            pct = 0.0
        pct = max(0.0, min(100.0, pct))
    except Exception:
        pct = 0.0
    filled = int(round((pct / 100.0) * w))
    return "▰" * filled + "▱" * max(0, w - filled)


def format_cool_status_card(
    title: str,
    stage: str,
    percent: int | float | None = None,
    detail: str | None = None,
    elapsed_s: float | None = None,
) -> str:
    """Format a stylish, high-tech HTML status card for long-running operations."""
    parts = [f"⚡ <b>{escape_html(title)}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
    if percent is not None:
        p = max(0, min(100, int(round(float(percent)))))
        parts.append(f"{make_cool_progress_bar(p, 10)}  <b>{p}%</b>")
    if stage:
        parts.append(f"📍 <b>ស្ថានភាព:</b> <code>{escape_html(stage)}</code>")
    if detail:
        parts.append(f"ℹ️ <i>{escape_html(detail)}</i>")
    if elapsed_s is not None and elapsed_s >= 0:
        parts.append(f"⏱️ <i>រយៈពេល: {elapsed_s:.1f}s</i>")
    parts.append("━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(parts)


PROGRESS_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _is_stale_message_error(exc: Exception) -> bool:
    """Detect if a Telegram error indicates the target message can no longer be edited."""
    with suppress(Exception):
        from app.services.telegram.flow import is_stale_telegram_message_error
        return is_stale_telegram_message_error(exc)

    err = str(exc).lower()
    return any(
        s in err
        for s in (
            "message to edit not found",
            "message can't be edited",
            "message is not modified",
            "message_id_invalid",
            "chat not found",
            "bot was blocked",
            "user is deactivated",
        )
    )


class StatusCardAnimator:
    """Async background animator that cycles high-tech spinner frames on a Telegram message."""

    def __init__(
        self,
        message: Any,
        *,
        title: str = "កំពុងដំណើរការ",
        stage: str = "",
        percent: int | float | None = None,
        detail: str | None = None,
        header_extra: str | None = None,
        icon: str = "⚡",
        interval: float = 1.2,
        spinners: Sequence[str] | None = None,
    ):
        self.message = message
        self.title = str(title or "កំពុងដំណើរការ")
        self.stage = str(stage or "")
        self.percent = percent
        self.detail = detail
        self.header_extra = header_extra
        self.icon = icon
        self.interval = max(0.9, float(interval))
        self.spinners = tuple(spinners or PROGRESS_SPINNER_FRAMES)
        self.frame_idx = 0
        self._task: asyncio.Task | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        self.last_text = ""
        self.started_at = time.monotonic()

    def update_state(
        self,
        *,
        stage: str | None = None,
        percent: int | float | None = None,
        detail: str | None = None,
        header_extra: str | None = None,
        icon: str | None = None,
    ) -> None:
        """Update stage/percent/detail in real time without interrupting frame cycling."""
        if stage is not None:
            self.stage = str(stage)
        if percent is not None:
            self.percent = percent
        if detail is not None:
            self.detail = str(detail)
        if header_extra is not None:
            self.header_extra = str(header_extra)
        if icon is not None:
            self.icon = str(icon)

    def render(self) -> str:
        """Render the animated status card with the active spinner frame."""
        spinner = self.spinners[self.frame_idx % len(self.spinners)]
        parts = [f"{self.icon} <b>{escape_html(self.title)}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
        if self.header_extra:
            parts.append(self.header_extra)
        if self.percent is not None:
            p = max(0, min(100, int(round(float(self.percent)))))
            bar = make_cool_progress_bar(p, 10)
            parts.append(f"{bar}  <b>{p}%</b>")
        if self.stage:
            parts.append(f"📍 <b>ដំណាក់កាល:</b> <code>{spinner} {escape_html(self.stage)}</code>")
        if self.detail:
            if not self.stage:
                parts.append(f"💡 <i>{escape_html(self.detail)} {spinner}</i>")
            else:
                parts.append(f"💡 <i>{escape_html(self.detail)}</i>")
        return "\n".join(parts)

    def start(self) -> None:
        """Start the background animation loop."""
        if self.message is None or not hasattr(self.message, "edit_text"):
            return
        try:
            loop = asyncio.get_running_loop()
            if self._task is None or self._task.done():
                self._task = loop.create_task(self._run_loop(), name="status-card-animator")
        except RuntimeError:
            pass

    async def _run_loop(self) -> None:
        try:
            consecutive_errors = 0
            while not self._closed:
                elapsed = time.monotonic() - self.started_at
                current_interval = max(self.interval, 2.5) if elapsed > 15.0 else self.interval
                await asyncio.sleep(current_interval)
                async with self._lock:
                    if self._closed or self.message is None:
                        return
                    self.frame_idx += 1
                    rendered = self.render()
                    if rendered == self.last_text:
                        continue
                    try:
                        await self.message.edit_text(rendered, parse_mode="HTML")
                        self.last_text = rendered
                        consecutive_errors = 0
                    except Exception as exc:
                        consecutive_errors += 1
                        exc_str = str(exc).lower()

                        if _is_stale_message_error(exc):
                            self._closed = True
                            return

                        retry_after = getattr(exc, "retry_after", None)
                        if retry_after and isinstance(retry_after, (int, float)):
                            await asyncio.sleep(float(retry_after) + 0.5)
                        elif "retry after" in exc_str or "flood" in exc_str:
                            await asyncio.sleep(5.0)
                        elif "message is not modified" in exc_str:
                            pass
                        elif consecutive_errors >= 5:
                            await asyncio.sleep(4.0)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def push_state(
        self,
        *,
        stage: str | None = None,
        percent: int | float | None = None,
        detail: str | None = None,
        header_extra: str | None = None,
        icon: str | None = None,
    ) -> None:
        """Update state and immediately render and edit the message."""
        self.update_state(stage=stage, percent=percent, detail=detail, header_extra=header_extra, icon=icon)
        async with self._lock:
            if not self._closed and self.message is not None and hasattr(self.message, "edit_text"):
                rendered = self.render()
                if rendered != self.last_text:
                    with suppress(Exception):
                        await self.message.edit_text(rendered, parse_mode="HTML")
                        self.last_text = rendered

    async def finish(self, final_text: str | None = None, *, parse_mode: str = "HTML") -> None:
        """Stop animation and update message with final text."""
        await self.stop()
        if final_text and self.message is not None and hasattr(self.message, "edit_text"):
            with suppress(Exception):
                await self.message.edit_text(final_text, parse_mode=parse_mode)

    async def stop(self) -> None:
        """Stop animation and cancel background task."""
        self._closed = True
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task

    async def __aenter__(self) -> "StatusCardAnimator":
        self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


__all__ = [
    "ALLOWED_TAGS",
    "PROGRESS_SPINNER_FRAMES",
    "StatusCardAnimator",
    "balance_telegram_html",
    "clean_speech_text",
    "escape_html",
    "format_cool_status_card",
    "make_cool_progress_bar",
    "markdown_to_telegram_html",
    "send_split_bot_message",
    "send_split_html",
    "split_telegram_html",
    "strip_html_tags",
    "unescape_html",
]