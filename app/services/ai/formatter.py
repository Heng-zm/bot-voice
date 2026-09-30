"""
formatter.py — Shared Telegram message formatter matching modern media standards.

Editorial Style:
  បច្ចេកវិទ្យាថ្ងៃនេះ (or Category header)
  
  🚨 [BREAKING]៖ <Khmer Title> 🔥  (or 🛡️ [ប្រយ័ត្នបោក]៖ ... ⚠️, 🤖 [AI in 60s]៖ ... ⚡, etc.)
  
  <1-2 sentences conversational hook>
  
  <Themed Sections with emojis & bold bullet lead-ins>
  • <b>Keyword៖</b> details
  
  💡 <Memorable practical takeaway or insight>
  
  🔒 ប្រភព: <Source Name>
  🟢 ស្ថានភាព: ✅ បានបញ្ជាក់
  🔗 អានដើម: <domain>
  📅 <Khmer Date>
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from urllib.parse import urlparse

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)

# Telegram sendPhoto caption hard limit is 1024 chars
_MAX_CAPTION = 1020
_MAX_BULLET_CHARS = 220

_KHMER_DIGITS = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
_KHMER_MONTHS = {
    1: "មករា", 2: "កុម្ភៈ", 3: "មីនា", 4: "មេសា",
    5: "ឧសភា", 6: "មិថុនា", 7: "កក្កដា", 8: "សីហា",
    9: "កញ្ញា", 10: "តុលា", 11: "វិច្ឆិកា", 12: "ធ្នូ",
}


def format_khmer_date(dt: datetime | None = None, include_time: bool = False) -> str:
    """Format datetime into standard Khmer calendar string (e.g. ២៤ កញ្ញា ២០២៦ ⏰ ០៧:៤៥)."""
    if dt is None:
        dt = datetime.now()
    day = str(dt.day).translate(_KHMER_DIGITS)
    month = _KHMER_MONTHS.get(dt.month, "")
    year = str(dt.year).translate(_KHMER_DIGITS)
    res = f"{day} {month} {year}"
    if include_time:
        hour = f"{dt.hour:02d}".translate(_KHMER_DIGITS)
        minute = f"{dt.minute:02d}".translate(_KHMER_DIGITS)
        res += f" ⏰ {hour}:{minute}"
    return res


def sanitize_telegram_html(text: str) -> str:
    """Sanitize HTML specifically for Telegram parse_mode='HTML'.
    Converts unsupported tags (<p>, <div>, <br>, <li>) into clean newlines/bullets,
    unescapes HTML entities, and strictly validates/balances allowed Telegram tags.
    Prevents 'Can't parse entities' crashes forever.
    """
    if not text:
        return ""

    s = html.unescape(text)

    # Convert common block tags to clean newlines
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p>\s*<p[^>]*>", "\n\n", s)
    s = re.sub(r"(?i)<p[^>]*>", "", s)
    s = re.sub(r"(?i)</p>", "\n\n", s)
    s = re.sub(r"(?i)<div[^>]*>", "", s)
    s = re.sub(r"(?i)</div>", "\n", s)
    s = re.sub(r"(?i)<li[^>]*>", "• ", s)
    s = re.sub(r"(?i)</li>", "\n", s)
    s = re.sub(r"(?i)</?[uo]l[^>]*>", "\n", s)
    s = re.sub(r"(?i)<h[1-6][^>]*>", "\n<b>", s)
    s = re.sub(r"(?i)</h[1-6]>", "</b>\n", s)

    # Allowed Telegram tags
    allowed_pattern = re.compile(
        r"(</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|tg-spoiler)>|<a\s+href=[\"'][^\"']+[\"']>|</a>)",
        re.IGNORECASE,
    )

    parts = allowed_pattern.split(s)
    out: list[str] = []
    stack: list[str] = []

    for part in parts:
        if not part:
            continue
        if allowed_pattern.fullmatch(part):
            m = re.match(r"</?([a-z-]+)", part, re.IGNORECASE)
            tag_name = m.group(1).lower() if m else ""
            if tag_name in ("strong", "b"):
                tag_canonical = "b"
            elif tag_name in ("em", "i"):
                tag_canonical = "i"
            elif tag_name in ("ins", "u"):
                tag_canonical = "u"
            elif tag_name in ("strike", "del", "s"):
                tag_canonical = "s"
            else:
                tag_canonical = tag_name

            if part.startswith("</"):
                if stack and stack[-1] == tag_canonical:
                    stack.pop()
                    out.append(f"</{tag_canonical}>")
            else:
                if tag_canonical == "a":
                    out.append(part)
                else:
                    out.append(f"<{tag_canonical}>")
                stack.append(tag_canonical)
        else:
            escaped = html.escape(part, quote=False)
            out.append(escaped)

    while stack:
        tag_canonical = stack.pop()
        out.append(f"</{tag_canonical}>")

    res = "".join(out)
    res = re.sub(r"\n{3,}", "\n\n", res)
    return res.strip()


def _format_bullets(km_text: str, budget: int) -> str:
    """Format structured summary, section headings, numbered steps, or bullet lines respecting budget."""
    if not km_text or budget <= 0:
        return ""

    lines = [l.strip() for l in km_text.replace("|||", "\n").split("\n") if l.strip()]
    if not lines:
        return ""

    body_lines: list[str] = []
    used = 0

    for line in lines:
        if not line:
            continue
        # Convert markdown bold **word** to Telegram <b>word</b>
        formatted_line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

        # Check if line is a section heading (ends with colon or starts with standard section emoji/words)
        is_heading = (
            formatted_line.endswith("៖")
            or formatted_line.endswith(":")
            or any(formatted_line.startswith(c) for c in (
                "📖", "🎯", "💡", "🔥", "🛡️", "🤖", "🛠️", "🚨", "📌", "✨",
                "វិធីប្រើ", "ល្អបំផុតសម្រាប់", "របៀបដែល", "របៀបការពារ", "ចំណុចសំខាន់", "និយមន័យ"
            ))
        )

        if is_heading:
            if not formatted_line.startswith("<b>") and "<b>" not in formatted_line:
                formatted_line = f"<b>{formatted_line}</b>"
        elif formatted_line.startswith(("- ", "* ")):
            formatted_line = "• " + formatted_line[2:]
        elif re.match(r"^([១២៣៤៥៦៧៨៩0-9]+[\.\)])\s*", formatted_line):
            # Preserves authentic numbered list items: ១. ..., ២. ..., etc.
            pass
        elif not formatted_line.startswith("•") and not formatted_line.startswith("<b"):
            # Short bullet points without bullet marker
            if len(formatted_line) < 180 and not is_heading:
                formatted_line = "• " + formatted_line

        # Bold any keyword before colon in bullet line: e.g. "• Keyword៖ text" -> "• <b>Keyword៖</b> text"
        if formatted_line.startswith("• ") and "<b>" not in formatted_line:
            kw_match = re.match(r"^(•\s*)([^៖:]+?)([៖:])(\s*.+)$", formatted_line)
            if kw_match and len(kw_match.group(2)) <= 40:
                formatted_line = f"{kw_match.group(1)}<b>{kw_match.group(2)}{kw_match.group(3)}</b>{kw_match.group(4)}"

        overhead = len(formatted_line) + 1  # newline
        if used + overhead > budget:
            break
        body_lines.append(formatted_line)
        used += overhead

    return "\n".join(body_lines).strip()


def format_article_message(
    *,
    km_title: str,
    km_text: str,
    url: str,
    verification: dict | None = None,
    analysis: dict | None = None,
    telegraph_url: str | None = None,
    date_str: str | None = None,
    source_name: str | None = None,
    category_name: str | None = None,
    hook: str | None = None,
    takeaway: str | None = None,
    is_breaking: bool = False,
) -> tuple[str, InlineKeyboardMarkup]:
    """Builds an executive, studio-grade Telegram news card and inline keyboard.
    Matches the user's reference screenshots with archetype badge, conversational hook,
    themed sections, highlight takeaway, and clean verified footer.
    """
    analysis = analysis or {}
    verification = verification or {}
    parsed_domain = urlparse(url).netloc.replace("www.", "")
    domain = source_name or parsed_domain or "Online"
    clean_date = date_str or format_khmer_date(datetime.now(), include_time=bool(is_breaking or analysis.get("is_hot")))

    # 1. Category Header (e.g. បច្ចេកវិទ្យាថ្ងៃនេះ, ព័ត៌មានជាតិ, សេដ្ឋកិច្ច & ហិរញ្ញវត្ថុ)
    cat = (category_name or "").strip()
    if not cat:
        cats = analysis.get("categories", [])
        if cats:
            cat = str(cats[0])
    if not cat:
        cat = "បច្ចេកវិទ្យាថ្ងៃនេះ" if any(w in domain.lower() for w in ("tech", "register", "openai", "github", "data", "ai", "zdnet")) else "ព័ត៌មានទាន់ហេតុការណ៍"

    # 2. Title Line
    raw_title = km_title.strip()
    is_tool = any(w in raw_title.lower() or w in km_text.lower() for w in ("tool", "app", "course", "សាលា", "រៀន", "ឧបករណ៍", "academy"))

    # Check if title already contains archetype badge e.g. [BREAKING] or [AI in 60s]
    if "[" in raw_title and "]" in raw_title:
        title_line = raw_title
    else:
        if is_breaking or analysis.get("is_hot"):
            title_line = f"🚨 [BREAKING]៖ {raw_title} 🔥"
        elif any(w in raw_title.lower() or w in km_text.lower() for w in ("scam", "លួច", "បោក", "ក្លែងក្លាយ", "hack", "មេរោគ", "គ្រោះថ្នាក់")):
            title_line = f"🛡️ [ប្រយ័ត្នបោក] | {raw_title} ⚠️"
        elif is_tool:
            title_line = f"🤖 [AI Tool of the Week]៖ {raw_title} ✨"
        elif any(w in raw_title.lower() for w in ("model", "តើជាអ្វី", "system", "ច្បាប់", "របៀប")):
            title_line = f"📺 [AI in 60s]៖ {raw_title} ⚡"
        else:
            title_line = f"🗞️ {raw_title}"

    # Make bold if not already tagged
    if not title_line.startswith("<b>") and "<b>" not in title_line:
        title_line = f"<b>{title_line}</b>"

    # 3. Footer Metadata Block
    if verification.get("verified"):
        status_text = "🟢 ស្ថានភាព: ✅ បានបញ្ជាក់"
    else:
        status_text = "🟡 ស្ថានភាព: ℹ️ ប្រភពបឋម"

    # Match screenshot 4 for tools/academies vs news
    link_label = "🔗 ប្រភព / កន្លែងសាក:" if is_tool else "🔗 អានដើម:"
    link_display = f"<a href=\"{html.escape(url)}\">{html.escape(parsed_domain)}</a>" if parsed_domain else html.escape(url)

    footer_lines = [
        f"🔒 ប្រភព: {html.escape(domain)}",
        status_text,
        f"{link_label} {link_display}",
        f"📅 {clean_date}",
    ]
    footer = "\n".join(footer_lines)

    # 4. Assemble Lead-in Hook and Takeaway
    hook_part = f"{hook.strip()}\n\n" if (hook and hook.strip()) else ""
    if takeaway and takeaway.strip():
        t_clean = takeaway.strip()
        if not t_clean.startswith("💡") and not any(t_clean.startswith(c) for c in ("🛡️", "⚠️", "🚨")):
            t_clean = f"💡 {t_clean}"
        takeaway_part = f"\n\n{t_clean}"
    else:
        takeaway_part = ""

    fixed_length = len(cat) + len(title_line) + len(hook_part) + len(takeaway_part) + len(footer) + 12
    bullet_budget = max(100, _MAX_CAPTION - fixed_length)

    body_part = _format_bullets(km_text, budget=bullet_budget)

    # Assemble full message text
    elements = []
    if cat:
        elements.append(cat)
    elements.append(title_line)
    if hook_part:
        elements.append(hook_part.strip())
    if body_part:
        elements.append(body_part)
    if takeaway_part:
        elements.append(takeaway_part.strip())
    elements.append(footer)

    raw_caption = "\n\n".join(elements)

    # Sanitize specifically for Telegram HTML
    caption = sanitize_telegram_html(raw_caption)

    # 5. Build Inline Keyboard
    keyboard: list[list[InlineKeyboardButton]] = []
    if telegraph_url:
        keyboard.append([
            InlineKeyboardButton("⚡ អានអត្ថបទពេញ (Instant View)", url=telegraph_url)
        ])

    article_cats = analysis.get("categories", [])
    if "យោធា" in article_cats or "conflict" in str(article_cats).lower():
        try:
            from app.services.ai.categorizer import get_conflict_map_info
            map_info = get_conflict_map_info(text=f"{km_title} {km_text}", url=url)
            if map_info:
                keyboard.append([InlineKeyboardButton(map_info["label"], url=map_info["url"])])
        except Exception:
            pass

    keyboard.append([
        InlineKeyboardButton("🔗 អានប្រភពដើម (Read Original)", url=url)
    ])

    return caption, InlineKeyboardMarkup(keyboard)


__all__ = [
    "format_article_message",
    "format_khmer_date",
    "sanitize_telegram_html",
]