"""Dynamic Visual Card & Interactive Banner Generation for News Bulletins.

Generates studio-grade 1200x630 executive cards in memory with Pillow:
- Dark cyber-grid aesthetic with ambient radial lighting
- Archetype-specific color palettes (Breaking, Cyber Scam Alert, AI in 60s, Tool of Week, Tech Dispatch)
- Glowing archetype badge pills and category ribbons
- Clean multi-line headline typography
- Integrated highlight container and source metadata
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Any
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Archetype Color Palettes
_ARCHETYPE_THEMES = {
    "BREAKING": {
        "accent": (239, 68, 68),
        "glow": (249, 115, 22),
        "badge": "BREAKING NEWS",
        "badge_bg": (153, 27, 27),
        "pill": "🚨 HOT UPDATE",
    },
    "SCAM_ALERT": {
        "accent": (245, 158, 11),
        "glow": (220, 38, 38),
        "badge": "CYBER ALERT",
        "badge_bg": (146, 64, 14),
        "pill": "🛡️ FRAUD WARNING",
    },
    "AI_EXPLAINER": {
        "accent": (6, 182, 212),
        "glow": (99, 102, 241),
        "badge": "AI IN 60s",
        "badge_bg": (21, 94, 117),
        "pill": "⚡ TECH ESSENCE",
    },
    "TOOL_REVIEW": {
        "accent": (16, 185, 129),
        "glow": (59, 130, 246),
        "badge": "TOOL OF THE WEEK",
        "badge_bg": (6, 95, 70),
        "pill": "✨ PRACTICAL RESOURCE",
    },
    "GENERAL_NEWS": {
        "accent": (59, 130, 246),
        "glow": (147, 51, 234),
        "badge": "TECH DISPATCH",
        "badge_bg": (30, 58, 138),
        "pill": "📰 DAILY INTELLIGENCE",
    },
}


def detect_card_archetype(title: str, text: str = "") -> str:
    """Detect appropriate visual card archetype from title and content keywords."""
    combined = f"{title} {text}".lower()
    if any(w in combined for w in ("scam", "phish", "malware", "hack", "fraud", "បោក", "លួច", "ក្លែងក្លាយ", "គ្រោះថ្នាក់")):
        return "SCAM_ALERT"
    if any(w in combined for w in ("breaking", "urgent", "ទាន់ហេតុការណ៍", "បន្ទាន់", "fire", "attack", "crisis")):
        return "BREAKING"
    if any(w in combined for w in ("tool", "app", "course", "academy", "ឧបករណ៍", "សាលា", "រៀន")):
        return "TOOL_REVIEW"
    if any(w in combined for w in ("ai", "model", "gpt", "gemini", "neural", "system 1", "llm", "claude", "agent")):
        return "AI_EXPLAINER"
    return "GENERAL_NEWS"


def generate_dynamic_card_banner(
    title: str,
    category: str = "TECH NEWS",
    source_name: str = "News Dispatch",
    archetype: str = "GENERAL_NEWS",
    date_str: str = "TODAY",
) -> bytes:
    """Generate high-resolution 1200x630 visual banner card in memory and return JPEG bytes."""
    W, H = 1200, 630

    theme = _ARCHETYPE_THEMES.get(archetype, _ARCHETYPE_THEMES["GENERAL_NEWS"])
    accent = theme["accent"]
    glow = theme["glow"]
    badge_text = theme["badge"]
    badge_bg = theme["badge_bg"]

    # 1. Base Dark Canvas
    base = Image.new("RGBA", (W, H), (11, 19, 43, 255))
    draw = ImageDraw.Draw(base)

    # 2. Modern Cyber Gridlines
    grid_col = (255, 255, 255, 10)
    for x in range(0, W, 60):
        draw.line([(x, 0), (x, H)], fill=grid_col, width=1)
    for y in range(0, H, 60):
        draw.line([(0, y), (W, y)], fill=grid_col, width=1)

    # 3. Ambient Radial Lighting (Top Right & Bottom Left)
    glow_overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow_overlay)
    for r in range(400, 0, -20):
        alpha = int(45 * (1 - r / 400))
        g_draw.ellipse([W - 220 - r, -100 - r, W - 220 + r, -100 + r], fill=(glow[0], glow[1], glow[2], alpha))
        g_draw.ellipse([-50 - r, H - 50 - r, -50 + r, H - 50 + r], fill=(accent[0], accent[1], accent[2], alpha // 2))

    base = Image.alpha_composite(base, glow_overlay)
    draw = ImageDraw.Draw(base)

    # 4. Outer Rounded Border
    draw.rounded_rectangle([(24, 24), (W - 24, H - 24)], radius=24, outline=(255, 255, 255, 30), width=2)

    # 5. Fonts setup
    font_large = None
    font_sub = None
    font_badge = None
    font_meta = None

    font_candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            font_large = ImageFont.truetype(fp, 44)
            font_sub = ImageFont.truetype(fp, 22)
            font_badge = ImageFont.truetype(fp, 18)
            font_meta = ImageFont.truetype(fp, 16)
            break

    if not font_large:
        font_large = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_badge = ImageFont.load_default()
        font_meta = ImageFont.load_default()

    # 6. Top Left Badge Pill
    badge_w = 230
    draw.rounded_rectangle([(56, 56), (56 + badge_w, 98)], radius=12, fill=badge_bg + (220,), outline=accent + (180,), width=1)
    draw.ellipse([(72, 73), (83, 84)], fill=accent)
    draw.text((94, 67), badge_text, fill=(255, 255, 255), font=font_badge)

    # 7. Top Right Category Indicator
    cat_display = re.sub(r"[^\w\s-]", "", category).strip().upper() or "NEWS BRIEF"
    draw.text((W - 320, 68), f"TOPIC: {cat_display[:24]}", fill=(148, 163, 184), font=font_meta)

    # 8. Main Headline (Clean typography, word-wrapped)
    clean_title = re.sub(r"[\U00010000-\U0010ffff]", "", title).strip()
    clean_title = re.sub(r"^\s*\[[^\]]+\]\s*[:៖]?\s*", "", clean_title).strip()

    words = clean_title.split()
    lines: list[str] = []
    curr: list[str] = []
    for w in words:
        curr.append(w)
        test_line = " ".join(curr)
        bbox = draw.textbbox((0, 0), test_line, font=font_large)
        if bbox[2] - bbox[0] > W - 140:
            curr.pop()
            if curr:
                lines.append(" ".join(curr))
            curr = [w]
        if len(lines) >= 3:
            break
    if curr and len(lines) < 3:
        lines.append(" ".join(curr))

    y_pos = 150
    for line in lines[:3]:
        draw.text((56, y_pos), line, fill=(255, 255, 255), font=font_large)
        y_pos += 62

    # 9. Decorative Accent Bar
    draw.rounded_rectangle([(56, y_pos + 12), (180, y_pos + 17)], radius=3, fill=accent)

    # 10. Executive Highlight Container Box
    box_top = max(y_pos + 38, 380)
    draw.rounded_rectangle([(56, box_top), (W - 56, box_top + 115)], radius=16, fill=(15, 23, 42, 190), outline=(255, 255, 255, 22), width=1)
    draw.text((80, box_top + 20), "EXECUTIVE INTELLIGENCE BRIEF", fill=accent, font=font_badge)
    draw.text((80, box_top + 54), "Verified multi-source reporting & AI-curated key insights.", fill=(203, 213, 225), font=font_sub)

    # 11. Bottom Metadata Line
    footer_y = H - 72
    draw.line([(56, footer_y - 14), (W - 56, footer_y - 14)], fill=(255, 255, 255, 18), width=1)
    clean_source = re.sub(r"[^\w\s\.-]", "", source_name).strip().upper() or "ONLINE"
    draw.text((56, footer_y), f"SOURCE: {clean_source}  •  BOT VOICE DISPATCH", fill=(148, 163, 184), font=font_meta)
    clean_date = re.sub(r"[^\w\s\.-]", "", date_str).strip().upper() or "LIVE"
    draw.text((W - 220, footer_y), clean_date, fill=(148, 163, 184), font=font_meta)

    buf = io.BytesIO()
    rgb = base.convert("RGB")
    try:
        rgb.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    finally:
        buf.close()
        base.close()
        glow_overlay.close()
        rgb.close()


def build_dynamic_card_keyboard(
    *,
    url: str,
    session_id: str = "",
    archetype: str = "GENERAL_NEWS",
    telegraph_url: str | None = None,
    is_khmer: bool = True,
    orig_lang_flag: str = "🌐",
) -> Any:
    """Build dynamic adaptive inline keyboard for Telegram news cards."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    keyboard: list[list[InlineKeyboardButton]] = []

    # Row 1: Primary reading links
    row1: list[InlineKeyboardButton] = []
    if telegraph_url:
        row1.append(InlineKeyboardButton("⚡ អានពេញ (Instant View)", url=telegraph_url))
    row1.append(InlineKeyboardButton("🔗 អានប្រភពដើម", url=url))
    keyboard.append(row1)

    # Row 2: Audio & Full Chat Text
    if session_id:
        keyboard.append([
            InlineKeyboardButton("🎙️ ស្តាប់ជាសំឡេង (Audio)", callback_data=f"art_voice:km_female:{session_id}"),
            InlineKeyboardButton("📄 អត្ថបទក្នុងឆាត", callback_data=f"art_full:{session_id}"),
        ])

    # Row 3: Archetype-specific interactive action + reactions
    row3: list[InlineKeyboardButton] = []
    if archetype == "SCAM_ALERT":
        row3.append(InlineKeyboardButton("🛡️ គន្លឹះសុវត្ថិភាព", callback_data="art_tip:scam"))
    elif archetype == "TOOL_REVIEW":
        row3.append(InlineKeyboardButton("💡 គន្លឹះប្រើប្រាស់", callback_data="art_tip:tool"))
    elif archetype == "AI_EXPLAINER":
        row3.append(InlineKeyboardButton("🤖 សួរ AI (Ask AI)", callback_data="art_tip:ai"))

    short_id = session_id[:16] if session_id else "card"
    row3.append(InlineKeyboardButton("❤️ ចូលចិត្ត", callback_data=f"art_like:{short_id}"))
    row3.append(InlineKeyboardButton("🔖 ចំណាំទុក", callback_data=f"art_save:{short_id}"))
    keyboard.append(row3)

    # Row 4: Read original language if translated from foreign source
    if not is_khmer and session_id:
        keyboard.append([
            InlineKeyboardButton(f"🌐 អានជាភាសាដើម ({orig_lang_flag})", callback_data=f"art_orig:{session_id}")
        ])

    return InlineKeyboardMarkup(keyboard)

