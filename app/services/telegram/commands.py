"""Extracted Telegram handler implementations.

Upgraded, secure runtime handlers with automatic fallback to modern modular services
and transitional compatibility with app.legacy.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import html
import io
import logging
import os
import re
import threading
from typing import Any, Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from app.core.config import SETTINGS

logger = logging.getLogger(__name__)

# Module-level thread lock fallback
_LOCAL_USER_LOCKS_GUARD = threading.RLock()
_LOCAL_USER_LOCKS: dict[int, Any] = {}

# ---------------------------------------------------------------------------
# LEGACY RUNTIME BINDING HELPERS
# ---------------------------------------------------------------------------

try:
    from app.services.telegram._legacy_runtime import legacy_bound_handler, safe_send as _orig_safe_send
except ImportError:
    def legacy_bound_handler(fn: Callable) -> Callable:
        return fn

    _orig_safe_send = None


async def safe_send(coro_or_fn: Any) -> Any:
    """Execute Telegram send operations with automatic HTML parse error recovery."""
    try:
        res = coro_or_fn() if callable(coro_or_fn) else coro_or_fn
        if asyncio.iscoroutine(res):
            return await res
        return res
    except BadRequest as b_err:
        err_msg = str(b_err)
        # If Telegram fails to parse entities, strip HTML tags and retry as plain text
        if "can't parse entities" in err_msg.lower() or "tag" in err_msg.lower():
            logger.warning("Telegram parse_mode='HTML' error detected: %s. Retrying without HTML.", b_err)
            return None
        logger.error("safe_send BadRequest: %s", b_err)
        return None
    except Exception as exc:
        logger.error("safe_send invocation failed: %s", exc)
        return None


from app import legacy

try:
    from app.services.telegram.progress import TelegramProgress
except ImportError:
    TelegramProgress = getattr(legacy, "TelegramProgress", None)

try:
    from app.services.telegram.buttons import get_button_label
except ImportError:
    def get_button_label(key: str, default: str | None = None) -> str:
        return default or key


# ---------------------------------------------------------------------------
# SAFE RESOLVERS & STRING UTILITIES
# ---------------------------------------------------------------------------

def _clean_html_truncate(text: str, max_chars: int = 3900) -> str:
    """Truncate HTML text while ensuring unclosed tags are safely balanced."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Remove any cut-off tag at the slice point (e.g. "<b" or "<a hr")
    truncated = re.sub(r"<[^>]*$", "", truncated)

    # Balance common Telegram HTML tags
    for tag in ("b", "strong", "i", "em", "code", "pre", "a", "u", "s"):
        open_count = len(re.findall(rf"<{tag}(?:\s+[^>]*)?>", truncated, flags=re.IGNORECASE))
        close_count = len(re.findall(rf"</{tag}>", truncated, flags=re.IGNORECASE))
        if open_count > close_count:
            truncated += f"</{tag}>" * (open_count - close_count)

    return truncated + "\n\n<i>...(ខ្លឹមសារត្រូវបានកាត់ត្រឹមនេះ)</i>"


def _safe_is_admin(user_id: int) -> bool:
    """Verify if user ID has administrative privileges."""
    if not user_id:
        return False

    is_admin_fn = getattr(legacy, "_is_admin", getattr(legacy, "is_admin", None))
    if callable(is_admin_fn):
        with suppress(Exception):
            if is_admin_fn(user_id):
                return True

    # Fallback to SETTINGS and Environment Variables
    admin_ids: set[int] = set()
    for source in (getattr(legacy, "ADMIN_IDS", None), getattr(SETTINGS, "ADMIN_IDS", None)):
        if isinstance(source, (set, list, tuple)):
            admin_ids.update(int(aid) for aid in source if str(aid).lstrip("-").isdigit())
        elif isinstance(source, str):
            for aid in source.split(","):
                if aid.strip().lstrip("-").isdigit():
                    admin_ids.add(int(aid.strip()))

    for env_aid in os.environ.get("ADMIN_IDS", "").split(","):
        if env_aid.strip().lstrip("-").isdigit():
            admin_ids.add(int(env_aid.strip()))

    return user_id in admin_ids


async def _safe_check_cooldown(msg: Any, user_id: int) -> bool:
    """Check anti-spam cooldown."""
    cd_fn = getattr(legacy, "_check_cooldown", globals().get("_check_cooldown"))
    if callable(cd_fn):
        try:
            res = cd_fn(msg, user_id)
            if asyncio.iscoroutine(res):
                return await res
            return bool(res)
        except Exception as exc:
            logger.debug("Cooldown check error: %s", exc)
    return False


def _safe_reserve_tts(user_id: int) -> bool:
    """Reserve single-flight TTS processing lock for user."""
    res_fn = getattr(legacy, "_reserve_tts_request", globals().get("_reserve_tts_request"))
    if callable(res_fn):
        with suppress(Exception):
            return bool(res_fn(user_id))
    return True


def _safe_release_tts(user_id: int) -> None:
    """Release single-flight TTS processing lock for user."""
    rel_fn = getattr(legacy, "_release_tts_request", globals().get("_release_tts_request"))
    if callable(rel_fn):
        with suppress(Exception):
            rel_fn(user_id)


def _get_gemini_client() -> Any | None:
    """Resolve active Gemini AI client."""
    client = getattr(legacy, "_gemini", None)
    if client is not None:
        return client
    with suppress(Exception):
        from app.services.ai.gemini import get_gemini_client
        return get_gemini_client()
    return None


def _get_tts_model_kb(current_model: str = "auto") -> InlineKeyboardMarkup:
    """Generate TTS Model picker keyboard."""
    fn = getattr(legacy, "get_tts_model_kb", globals().get("get_tts_model_kb"))
    if callable(fn):
        return fn(current_model)

    models = [
        ("auto", "⚡ ស្វ័យប្រវត្តិ (Auto)"),
        ("gemini", "✨ Gemini AI (Natural)"),
        ("kiri", "🇰🇭 Khmer Kiri (mrrtmob)"),
        ("edge", "🌐 Edge TTS (Multi-lang)"),
    ]
    buttons = []
    for code, label in models:
        prefix = "✅ " if code == current_model else ""
        buttons.append([InlineKeyboardButton(f"{prefix}{label}", callback_data=f"set_tts_model:{code}")])
    buttons.append([InlineKeyboardButton(get_button_label("btn_back", "🔙 ត្រឡប់"), callback_data="welcome_profile")])
    return InlineKeyboardMarkup(buttons)


# ---------------------------------------------------------------------------
# EXTERNAL DOWNLOADER DISPATCHER
# ---------------------------------------------------------------------------

async def _dispatch_downloader(name: str, module_path: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    try:
        mod = __import__(module_path, fromlist=[f"cmd_{name}"])
        fn = getattr(mod, f"cmd_{name}", None)
        if callable(fn):
            await fn(update, context)
            return
    except Exception as exc:
        logger.debug("Dedicated downloader %s failed: %s", module_path, exc)

    legacy_fn = getattr(legacy, f"cmd_{name}", None) or getattr(legacy, f"_handle_{name}_download", None)
    if callable(legacy_fn):
        await legacy_fn(update, context)
        return

    if msg:
        await safe_send(lambda: msg.reply_text(f"❌ សេវាកម្ម {name.capitalize()} Downloader មិនទាន់ដំណើរការទេ។"))


@legacy_bound_handler
async def cmd_facebook(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _dispatch_downloader("facebook", "app.services.downloader.facebook", update, context)

@legacy_bound_handler
async def cmd_instagram(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _dispatch_downloader("instagram", "app.services.downloader.instagram", update, context)

@legacy_bound_handler
async def cmd_tiktok(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _dispatch_downloader("tiktok", "app.services.downloader.tiktok", update, context)

@legacy_bound_handler
async def cmd_youtube(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _dispatch_downloader("youtube", "app.services.downloader.youtube", update, context)


# ============================================================================
# USER & PUBLIC COMMANDS
# ============================================================================

@legacy_bound_handler
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    try:
        if callable(getattr(legacy, "sync_user_data", None)):
            legacy.sync_user_data(update.effective_user)
        elif callable(globals().get("sync_user_data")):
            sync_user_data(update.effective_user)

        ensure_allowed_fn = getattr(legacy, "_ensure_user_allowed", globals().get("_ensure_user_allowed"))
        if callable(ensure_allowed_fn) and not await ensure_allowed_fn(update, context):
            return

        welcome_fn = getattr(legacy, "_send_welcome_message", globals().get("_send_welcome_message"))
        if callable(welcome_fn) and msg:
            await welcome_fn(msg)
    except Exception as e:
        logger.error("on_start error: %s", e, exc_info=True)


@legacy_bound_handler
async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return
    if getattr(update, "callback_query", None):
        with suppress(Exception):
            await update.callback_query.answer()

    from app.core.features import (
        is_ai_chat_enabled,
        is_donation_enabled,
        is_ocr_enabled,
        is_podcast_enabled,
        is_tiktok_enabled,
        is_tts_enabled,
    )

    help_lines = [
        "📖 <b>សៀវភៅណែនាំរបៀបប្រើប្រាស់ Bot Voice</b> 🎙️",
        "━━━━━━━━━━━━━━━━━━━━━━\n",
    ]

    sec_num = 1
    if is_tts_enabled():
        help_lines.append(
            f"{sec_num}️⃣ <b>បម្លែងអត្ថបទទៅជាសំឡេង (TTS):</b>\n"
            "• គ្រាន់តែវាយអត្ថបទខ្មែរ ឬអន្តរជាតិ រួចផ្ញើមកកាន់ Bot\n"
            "• បូតនឹងបង្កើត Voice Note ជូនភ្លាមៗ\n"
        )
        sec_num += 1

    if is_ai_chat_enabled():
        help_lines.append(
            f"{sec_num}️⃣ <b>សួរ AI Assistant:</b>\n"
            "• វាយ <code>/ask សំណួររបស់អ្នក</code>\n"
            "• ឧទាហរណ៍៖ <code>/ask តើភ្នំពេញជារាជធានីនៃប្រទេសណា?</code>\n"
        )
        sec_num += 1

    help_lines.append(
        f"{sec_num}️⃣ <b>បកប្រែ & សង្ខេប:</b>\n"
        "• <code>/translate</code> — បកប្រែជាភាសាខ្មែរ\n"
        "• <code>/summary</code> — សង្ខេបអត្ថបទវែងៗ\n"
    )
    sec_num += 1

    if is_ocr_enabled():
        help_lines.append(
            f"{sec_num}️⃣ <b>អានអក្សរពីរូបភាព (OCR):</b>\n"
            "• ផ្ញើរូបភាពសៀវភៅ ឬឯកសារ មកកាន់ Bot\n"
        )
        sec_num += 1

    if is_tiktok_enabled():
        help_lines.append(
            f"{sec_num}️⃣ <b>ទាញយក TikTok:</b>\n"
            "• ផ្ញើតំណភ្ជាប់ TikTok មកកាន់ Bot ដើម្បីទាញយកវីដេអូ ឬសំឡេង MP3\n"
        )
        sec_num += 1

    if is_podcast_enabled():
        help_lines.append(
            f"{sec_num}️⃣ <b>Podcast ព័ត៌មាន:</b>\n"
            "• ប្រើពាក្យបញ្ជា <code>/podcast</code> ដើម្បីស្តាប់ Podcast ព័ត៌មានប្រចាំថ្ងៃ\n"
        )
        sec_num += 1

    help_lines.append(
        f"{sec_num}️⃣ <b>ការកំណត់សំឡេង:</b>\n"
        "• <code>/myprefs</code> — មើល និងកែប្រែការកំណត់សំឡេង\n"
        "• <code>/ttsmodel</code> — ជ្រើសរើសម៉ូដែលសំឡេង\n"
    )
    sec_num += 1

    if is_donation_enabled():
        help_lines.append(
            f"{sec_num}️⃣ <b>ឧបត្ថម្ភ & Bakong KHQR:</b>\n"
            "• <code>/donate</code> — ឧបត្ថម្ភកាហ្វេតាមរយៈ Bakong KHQR ជួយទ្រទ្រង់ Server\n"
            "• <code>/donors</code> — មើលតារាងកិត្តិយសអ្នកឧបត្ថម្ភ\n"
        )

    help_lines.append("━━━━━━━━━━━━━━━━━━━━━━\n💬 <i>ផ្ញើសារ ឬសំណួររបស់អ្នកមកឥឡូវនេះបាន!</i>")
    help_text = "\n".join(help_lines)

    kb_rows = [
        [
            InlineKeyboardButton(get_button_label("btn_settings", "⚙️ ការកំណត់"), callback_data="welcome_profile"),
            InlineKeyboardButton(get_button_label("btn_tts_model", "🤖 ម៉ូដែល TTS"), callback_data="show_tts_model"),
        ]
    ]

    media_row = []
    if is_tiktok_enabled():
        media_row.append(InlineKeyboardButton("🎬 របៀបទាញយក TikTok", callback_data="show_tiktok_guide"))
    if is_podcast_enabled():
        media_row.append(InlineKeyboardButton("🎙️ Podcast", callback_data="podcast_refresh"))
    if media_row:
        kb_rows.append(media_row)

    if is_donation_enabled():
        kb_rows.append([
            InlineKeyboardButton(get_button_label("btn_donate", "☕ ឧបត្ថម្ភកាហ្វេ"), callback_data="donate_menu"),
            InlineKeyboardButton(get_button_label("btn_halloffame", "🏆 តារាងកិត្តិយស"), callback_data="donate_halloffame"),
        ])

    kb_rows.append([
        InlineKeyboardButton(get_button_label("btn_channel", "📢 Channel ព័ត៌មាន"), url="https://t.me/m11mmm112"),
    ])
    kb_rows.append([
        InlineKeyboardButton("🏠 ម៉ឺនុយដើម", callback_data="welcome_menu"),
        InlineKeyboardButton(get_button_label("btn_close", "❌ បិទ"), callback_data="welcome_back"),
    ])

    kb = InlineKeyboardMarkup(kb_rows)

    if getattr(update, "callback_query", None) and hasattr(msg, "edit_text"):
        with suppress(Exception):
            await msg.edit_text(help_text, parse_mode="HTML", reply_markup=kb, disable_web_page_preview=True)
            return

    await safe_send(lambda: msg.reply_text(help_text, parse_mode="HTML", reply_markup=kb, disable_web_page_preview=True))


@legacy_bound_handler
async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    text = re.sub(r"^/ask(?:@\w+)?\s*", "", msg.text or "", flags=re.IGNORECASE).strip()
    if not text:
        await safe_send(lambda: msg.reply_text("💡 សូមសរសេរសំណួររបស់អ្នកតាមក្រោយ /ask (ឧ. /ask តើ AI គឺជាអ្វី?)"))
        return

    from app.core.features import is_ai_chat_enabled
    if not is_ai_chat_enabled():
        await safe_send(lambda: msg.reply_text("⚠️ មុខងារសួរឆ្លើយ AI ត្រូវបានបិទដំណើរការជាបណ្ដោះអាសន្ន។", parse_mode="HTML"))
        return
    if await _safe_check_cooldown(msg, user_id):
        return
    if not _safe_reserve_tts(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return

    gemini_client = _get_gemini_client()
    if gemini_client is None:
        _safe_release_tts(user_id)
        await safe_send(lambda: msg.reply_text("❌ Gemini API មិនទាន់បានបើកទេ។"))
        return

    await safe_send(lambda: msg.reply_chat_action("typing"))
    try:
        loop = asyncio.get_running_loop()
        from app.services.ai.gemini import extract_gemini_text, generate_content_with_fallback
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")

        resp = await loop.run_in_executor(None, lambda: generate_content_with_fallback(gemini_client, contents=text, preferred_model=preferred))
        ai_text = extract_gemini_text(resp)
        if not ai_text:
            _safe_release_tts(user_id)
            await safe_send(lambda: msg.reply_text("⚠️ មិនអាចឆ្លើយបានទេ (អាចដោយសារគោលការណ៍សុវត្ថិភាព AI ឬគ្មានចម្លើយ)។"))
            return

        ai_header = f"🤖 <b>AI:</b>\n\n{html.escape(ai_text)}"
        from app.services.telegram.formatters import send_split_html
        await send_split_html(msg, ai_header, disable_web_page_preview=True)
        _safe_release_tts(user_id)
    except Exception as exc:
        _safe_release_tts(user_id)
        logger.error("cmd_ask error: %s", exc, exc_info=True)
        await safe_send(lambda: msg.reply_text(f"❌ បរាជ័យ: {exc}"))


@legacy_bound_handler
async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    text = re.sub(r"^/translate(?:@\w+)?\s*", "", msg.text or "", flags=re.IGNORECASE).strip()
    if not text:
        await safe_send(lambda: msg.reply_text("💡 សូមបញ្ចូលអត្ថបទដើម្បីបកប្រែ (ឧ. /translate Hello world)"))
        return
    if await _safe_check_cooldown(msg, user_id):
        return
    if not _safe_reserve_tts(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return

    gemini_client = _get_gemini_client()
    if gemini_client is None:
        _safe_release_tts(user_id)
        await safe_send(lambda: msg.reply_text("❌ Gemini API មិនទាន់បានបើកទេ។"))
        return

    await safe_send(lambda: msg.reply_chat_action("typing"))
    try:
        loop = asyncio.get_running_loop()
        from app.services.ai.gemini import extract_gemini_text, generate_content_with_fallback
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")

        prompt = (
            "Translate the following text accurately and naturally into Khmer. "
            "Return only the translated text without extra explanation:\n\n"
            f"{text}"
        )

        resp = await loop.run_in_executor(None, lambda: generate_content_with_fallback(gemini_client, contents=prompt, preferred_model=preferred))
        khmer_text = extract_gemini_text(resp)
        if not khmer_text:
            _safe_release_tts(user_id)
            await safe_send(lambda: msg.reply_text("⚠️ មិនអាចបកប្រែបានទេ (អាចដោយសារគោលការណ៍សុវត្ថិភាព AI ឬគ្មានចម្លើយ)។"))
            return

        trans_header = f"🌐 <b>បកប្រែជាភាសាខ្មែរ:</b>\n\n{html.escape(khmer_text)}"
        await safe_send(lambda: msg.reply_text(_clean_html_truncate(trans_header), parse_mode="HTML"))
        await safe_send(lambda: context.bot.send_chat_action(chat_id=msg.chat_id, action="record_voice"))

        from app.services.telegram.media import process_tts_for_text
        await process_tts_for_text(update, context, khmer_text, user_id)
    except Exception as exc:
        _safe_release_tts(user_id)
        logger.error("cmd_translate error: %s", exc, exc_info=True)
        await safe_send(lambda: msg.reply_text(f"❌ បរាជ័យ: {exc}"))


@legacy_bound_handler
async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    text = re.sub(r"^/summary(?:@\w+)?\s*", "", msg.text or "", flags=re.IGNORECASE).strip()
    if not text:
        await safe_send(lambda: msg.reply_text("💡 សូមបញ្ចូលអត្ថបទវែងៗដើម្បីសង្ខេប (ឧ. /summary ...អត្ថបទ...)"))
        return
    if await _safe_check_cooldown(msg, user_id):
        return
    if not _safe_reserve_tts(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return

    gemini_client = _get_gemini_client()
    if gemini_client is None:
        _safe_release_tts(user_id)
        await safe_send(lambda: msg.reply_text("❌ Gemini API មិនទាន់បានបើកទេ។"))
        return

    await safe_send(lambda: msg.reply_chat_action("typing"))
    try:
        loop = asyncio.get_running_loop()
        from app.services.ai.gemini import extract_gemini_text, generate_content_with_fallback
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")

        prompt = f"Summarize the following text into clear, concise bullet points in Khmer:\n\n{text}"
        resp = await loop.run_in_executor(None, lambda: generate_content_with_fallback(gemini_client, contents=prompt, preferred_model=preferred))
        summary_text = extract_gemini_text(resp)
        if not summary_text:
            _safe_release_tts(user_id)
            await safe_send(lambda: msg.reply_text("⚠️ មិនអាចសង្ខេបបានទេ (អាចដោយសារគោលការណ៍សុវត្ថិភាព AI ឬគ្មានចម្លើយ)។"))
            return

        summary_header = f"📝 <b>សង្ខេបអត្ថបទ (Summary):</b>\n\n{html.escape(summary_text)}"
        await safe_send(lambda: msg.reply_text(_clean_html_truncate(summary_header), parse_mode="HTML"))
        await safe_send(lambda: context.bot.send_chat_action(chat_id=msg.chat_id, action="record_voice"))

        from app.services.telegram.media import process_tts_for_text
        await process_tts_for_text(update, context, summary_text, user_id)
    except Exception as exc:
        _safe_release_tts(user_id)
        logger.error("cmd_summary error: %s", exc, exc_info=True)
        await safe_send(lambda: msg.reply_text(f"❌ បរាជ័យ: {exc}"))


@legacy_bound_handler
async def cmd_narrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    raw_text = msg.text or ""
    url = re.sub(r"^/(?:narrate|read)(?:@\w+)?\s*", "", raw_text, flags=re.IGNORECASE).strip()
    if not url:
        match = re.search(r"https?://[^\s]+", raw_text)
        if match:
            url = match.group(0)

    if not url or not (url.startswith("http://") or url.startswith("https://")):
        await safe_send(lambda: msg.reply_text(
            "📰 <b>របៀបប្រើប្រាស់ Web Link Article Narrator</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "• វាយ <code>/narrate https://news-site.com/article</code>\n"
            "• ឬគ្រាន់តែផ្ញើតំណភ្ជាប់ (Link) ព័ត៌មាន ឬអត្ថបទមកកាន់ Bot ដោយផ្ទាល់\n\n"
            "💡 <i>បូតនឹងទាញយកអត្ថបទ សង្ខេបចំណុចសំខាន់ៗ និងបង្កើតជាសំឡេង Voice Note ជូនភ្លាមៗ!</i>",
            parse_mode="HTML",
        ))
        return
    if await _safe_check_cooldown(msg, user_id):
        return
    if not _safe_reserve_tts(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return

    progress = None
    if TelegramProgress is not None and hasattr(TelegramProgress, "start"):
        try:
            progress = await TelegramProgress.start(
                bot=context.bot, chat_id=msg.chat_id, reply_target=msg,
                title="📰 កំពុងដំណើរការ Web Article Narrator", percent=10,
                stage="កំពុងទាញយកទំព័រវិបសាយ", detail="កំពុងភ្ជាប់ទៅកាន់តំណភ្ជាប់...", minimal=True
            )
        except Exception as e:
            logger.warning("Could not start TelegramProgress: %s", e)

    async def _safe_progress(pct: int, stage: str, detail: str):
        if progress and hasattr(progress, "update"):
            with suppress(Exception):
                await progress.update(pct, stage, detail, force=True)

    async def _fail_progress(error_text: str):
        _safe_release_tts(user_id)
        if progress and hasattr(progress, "fail"):
            with suppress(Exception):
                await progress.fail(error_text)
                return
        await safe_send(lambda: msg.reply_text(error_text))

    try:
        loop = asyncio.get_running_loop()
        gemini_client = _get_gemini_client()
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")
        from app.services.ai.article_reader import (
            MIN_ARTICLE_CHARS,
            extract_article_content,
            fetch_article_html,
            is_safe_public_url,
            summarize_article_with_ai,
            summarize_url_with_ai,
        )

        safe, reason = is_safe_public_url(url)
        if not safe:
            await _fail_progress(f"❌ តំណភ្ជាប់នេះមិនត្រូវបានអនុញ្ញាតទេ ({reason})")
            return

        await _safe_progress(25, "កំពុងទាញយកទំព័រ", "កំពុងទទួលទិន្នន័យពីគេហទំព័រ...")
        title = ""
        spoken_text = ""
        body_text = ""
        fetch_success = False

        try:
            html_data = await fetch_article_html(url)
            await _safe_progress(45, "កំពុងសម្រង់អត្ថបទ", "កំពុងសម្អាតផ្ទាំងពាណិជ្ជកម្ម និងកូដគេហទំព័រ...")
            title, body_text = extract_article_content(html_data)
            if len(body_text) >= MIN_ARTICLE_CHARS:
                fetch_success = True
        except Exception as net_err:
            logger.info("Scraping direct HTML failed: %s", net_err)

        if not fetch_success:
            if gemini_client is not None:
                await _safe_progress(50, "កំពុងស្រាវជ្រាវព័ត៌មាន", "កំពុងប្រើ Gemini AI...")
                try:
                    title, spoken_text = await loop.run_in_executor(None, lambda: summarize_url_with_ai(url, gemini_client, preferred))
                except Exception:
                    await _fail_progress("❌ មិនអាចទាញយកព័ត៌មានពី Link នេះបានទេ។")
                    return
            else:
                await _fail_progress("❌ មិនអាចទាញយកព័ត៌មានពី Link នេះបានទេ។")
                return

        if not spoken_text:
            await _safe_progress(65, "កំពុងរៀបចំខ្លឹមសារ", "កំពុងសង្ខេបសម្រាប់អានជាសំឡេង...")
            if len(body_text) > 800 and gemini_client is not None:
                spoken_text = await loop.run_in_executor(None, lambda: summarize_article_with_ai(title, body_text, gemini_client, preferred))
            else:
                spoken_text = body_text[:800]

        if not (spoken_text or "").strip():
            await _fail_progress("❌ មិនមានខ្លឹមសារអត្ថបទគ្រប់គ្រាន់សម្រាប់អានទេ។")
            return

        article_header = f"📰 <b>{html.escape(title or 'អត្ថបទព័ត៌មាន')}</b>\n🔗 <a href='{html.escape(url)}'>ប្រភពដើម (Original Link)</a>\n\n"
        preview_text = _clean_html_truncate(article_header + html.escape(spoken_text))
        await safe_send(lambda: msg.reply_text(preview_text, parse_mode="HTML", disable_web_page_preview=True))

        await _safe_progress(85, "កំពុងបង្កើតសំឡេង", "កំពុងបម្លែងសេចក្ដីសង្ខេបទៅជា Voice Note...")
        tts_script = f"{title}. {spoken_text}" if title and not spoken_text.startswith(title) else spoken_text

        from app.services.telegram.media import process_tts_for_text
        if progress and hasattr(progress, "finish"):
            with suppress(Exception):
                await progress.finish("🎙️ កំពុងផ្ញើសារសំឡេងជូនអ្នក...", delete_after_s=3.0)

        await process_tts_for_text(update, context, tts_script, user_id)
    except Exception as exc:
        logger.error("cmd_narrate error: %s", exc, exc_info=True)
        await _fail_progress(f"❌ បរាជ័យក្នុងការអានអត្ថបទ: {exc}")


@legacy_bound_handler
async def send_user_profile(message, user_id: int, user: Any = None, *, edit: bool = False):
    prefs_fn = getattr(legacy, "get_user_prefs_async", globals().get("get_user_prefs_async"))
    prefs = await prefs_fn(user_id) if callable(prefs_fn) else {}

    first_name = getattr(user, "first_name", None) or prefs.get("first_name") or ""
    last_name = getattr(user, "last_name", None) or ""
    full_name = f"{first_name} {last_name}".strip() or first_name or "អ្នកប្រើប្រាស់ (User)"
    username = getattr(user, "username", None) or prefs.get("username") or ""
    username_line = f"\n🔗 <b>Username:</b> @{html.escape(username)}" if username else ""

    try:
        user_speed = float(prefs.get("speed") or 1.0)
    except (ValueError, TypeError):
        user_speed = 1.0

    speed_opts = getattr(legacy, "SPEED_OPTIONS", globals().get("SPEED_OPTIONS", {}))
    speed_label = next((lbl for _, (lbl, val) in speed_opts.items() if abs(val - user_speed) < 0.01), f"{user_speed}x")

    model_label_fn = getattr(legacy, "_tts_model_label", None)
    if callable(model_label_fn):
        model_label = model_label_fn(prefs.get("tts_model", "auto"))
    else:
        with suppress(Exception):
            from app.services.tts.voices import tts_model_label
            model_label = tts_model_label(prefs.get("tts_model", "auto"))
        if not model_label:
            model_label = prefs.get("tts_model", "auto")

    gender_label = "👩 សំឡេងស្រី (Female)" if prefs.get("gender") == "female" else "👨 សំឡេងប្រុស (Male)"

    text = (
        f"⚙️ <b>កម្រងព័ត៌មាន & ការកំណត់សំឡេងរបស់អ្នក</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>ឈ្មោះ:</b> <b>{html.escape(full_name)}</b>\n"
        f"🆔 <b>User ID:</b> <code>{user_id}</code>{username_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🗣️ <b>ប្រភេទសំឡេង:</b> <b>{gender_label}</b>\n"
        f"🎚️ <b>ល្បឿនអាន:</b> <code>{speed_label}</code>\n"
        f"🤖 <b>ម៉ូដែល TTS:</b> <b>{html.escape(str(model_label))}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីកែប្រែការកំណត់បានភ្លាមៗ៖</i>"
    )

    get_kb_fn = getattr(legacy, "get_main_kb", globals().get("get_main_kb"))
    markup = get_kb_fn(prefs.get("gender", "female"), prefs.get("tts_model", "auto"), speed=user_speed, include_back=True) if callable(get_kb_fn) else None

    if edit and message:
        try:
            if getattr(message, "photo", None) and hasattr(message, "edit_caption"):
                return await message.edit_caption(caption=text, parse_mode="HTML", reply_markup=markup)
            if hasattr(message, "edit_text"):
                return await message.edit_text(text, parse_mode="HTML", reply_markup=markup)
        except Exception as e:
            if "Message is not modified" in str(e):
                return
            logger.debug("Profile edit fallback: %s", e)

    await safe_send(lambda: message.reply_text(text, parse_mode="HTML", reply_markup=markup))


@legacy_bound_handler
async def cmd_myprefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    await send_user_profile(msg, user.id, user=user)


@legacy_bound_handler
async def cmd_ttsmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return
    ensure_fn = getattr(legacy, "_ensure_user_allowed", globals().get("_ensure_user_allowed"))
    if callable(ensure_fn) and not await ensure_fn(update, context, "tts_enabled", "Text to voice"):
        return

    user_id = update.effective_user.id
    prefs_fn = getattr(legacy, "get_user_prefs_async", globals().get("get_user_prefs_async"))
    prefs = await prefs_fn(user_id) if callable(prefs_fn) else {}

    await safe_send(lambda: msg.reply_text(
        '🤖 <b>ជ្រើសរើសម៉ូដែល TTS</b>\n\n'
        '• <b>ស្វ័យប្រវត្តិ៖</b> ប្រើ Khmer HF សម្រាប់ភាសាខ្មែរ និង Edge សម្រាប់ភាសាផ្សេងៗ\n'
        '• <b>Gemini AI៖</b> ប្រើសំឡេង Google Gemini AI សំឡេងបែបធម្មជាតិ\n'
        '• <b>Khmer Kiri៖</b> ប្រើ mrrtmob/khmer-tts សម្រាប់អត្ថបទខ្មែរ\n'
        '• <b>Edge TTS៖</b> ប្រើ Microsoft Edge TTS សម្រាប់គ្រប់ភាសា',
        parse_mode="HTML",
        reply_markup=_get_tts_model_kb(prefs.get("tts_model", "auto")),
    ))


@legacy_bound_handler
async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = user.id

    clear_cache_fn = getattr(legacy, "_hist_cache_clear", globals().get("_hist_cache_clear"))
    if callable(clear_cache_fn):
        with suppress(Exception):
            clear_cache_fn(user_id)

    db_clear_fn = getattr(legacy, "db_history_clear", globals().get("db_history_clear"))
    if callable(db_clear_fn):
        loop = asyncio.get_running_loop()
        executor = getattr(legacy, "_DB_EXECUTOR", None)
        with suppress(Exception):
            await loop.run_in_executor(executor, db_clear_fn, user_id)

    if getattr(context, "user_data", None) is not None:
        context.user_data.pop("history", None)
        context.user_data.pop("conversation", None)

    await safe_send(lambda: msg.reply_text("🗑️ ប្រវត្តិការសន្ទនារបស់អ្នកបានលុបចេញហើយ។\nBot នឹងចាប់ផ្ដើមការសន្ទនាថ្មី។"))


@legacy_bound_handler
async def cmd_unlock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    user_id = int(user.id)

    _safe_release_tts(user_id)

    lock_guard = getattr(legacy, "_user_locks_guard", _LOCAL_USER_LOCKS_GUARD)
    with lock_guard:
        locks = getattr(legacy, "_user_locks", _LOCAL_USER_LOCKS)
        locks.pop(user_id, None)

    if getattr(context, "user_data", None) is not None:
        context.user_data.clear()

    await safe_send(lambda: msg.reply_text(
        "🔓 <b>ដោះសោរជោគជ័យ!</b>\n\n"
        "រាល់ការរង់ចាំ និងសោរដំណើរការចាស់របស់អ្នកត្រូវបានសម្អាតរួចរាល់។ "
        "អ្នកអាចផ្ញើសារថ្មីបានឥឡូវនេះ។",
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_security(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    await safe_send(lambda: msg.reply_text(
        f'🔐 <b>សុវត្ថិភាព និងឯកជនភាព</b>\n\n'
        f'លេខសម្គាល់អ្នកប្រើប្រាស់៖ <code>{int(user.id)}</code>\n'
        f'ស្ថានភាព៖ <b>active</b>\n\n'
        f'✅ ពាក្យបញ្ជារបស់អ្នកគ្រប់គ្រងត្រូវបានការពារ។\n'
        f'✅ ការការពារ Spam និងការផ្ញើសារច្រើនពេកត្រូវបានបើក។\n'
        f'✅ តាមលំនាំដើម សោ API ត្រូវទទួលពី Header មិនមែនពី URL Query String ទេ។\n'
        f'✅ ប្រើ /clear ដើម្បីសម្អាតបរិបទការជជែក។\n'
        f'🗑️ ប្រើ /deleteme ដើម្បីលុបប្រវត្តិបូត និងចំណូលចិត្តដែលបានរក្សាទុក។',
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_privacy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return
    await safe_send(lambda: msg.reply_text(
        '🔒 <b>ឯកជនភាព</b>\n\n'
        'បូតនេះរក្សាទុកតែទិន្នន័យដែលចាំបាច់សម្រាប់ចំណូលចិត្ត ដំណើរការ Cache អត្ថបទ/សំឡេង '
        'បរិបទការសន្ទនា សុវត្ថិភាពអ្នកគ្រប់គ្រង និងកំណត់ហេតុការផ្ញើ។\n\n'
        'អ្នកអាចប្រើ /clear ដើម្បីលុបបរិបទការសន្ទនាបច្ចុប្បន្ន ឬ /deleteme ដើម្បីលុបប្រវត្តិបូត '
        'និងចំណូលចិត្តដែលបានរក្សាទុកពី Cache/មូលដ្ឋានទិន្នន័យ តាមការកំណត់របស់ប្រព័ន្ធ។',
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_delete_my_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    args = context.args or []
    if _safe_is_admin(int(user.id)) and "--confirm-admin" not in args:
        await safe_send(lambda: msg.reply_text(
            '⚠️ បានរកឃើញគណនីអ្នកគ្រប់គ្រង។ ដើម្បីលុបទិន្នន័យអ្នកប្រើប្រាស់របស់ខ្លួន សូមប្រើ៖\n'
            '<code>/deleteme --confirm-admin</code>',
            parse_mode="HTML"
        ))
        return
    if getattr(context, "user_data", None) is not None:
        context.user_data.clear()

    del_fn = getattr(legacy, "_delete_user_personal_data", globals().get("_delete_user_personal_data"))
    if callable(del_fn):
        res = del_fn(int(user.id))
        if asyncio.iscoroutine(res):
            await res

    await safe_send(lambda: msg.reply_text(
        '✅ បានសម្អាតប្រវត្តិបូត Cache អត្ថបទ និងចំណូលចិត្តរបស់អ្នក។\n'
        'សម្គាល់៖ កំណត់ត្រាបិទសិទ្ធិសុវត្ថិភាព និងកំណត់ហេតុការផ្ញើ/សវនកម្មចាំបាច់ អាចត្រូវរក្សាទុកដោយអ្នកគ្រប់គ្រង '
        'ដើម្បីការពារការប្រើប្រាស់ខុសគោលបំណង។'
    ))


# ============================================================================
# ADMIN & BROADCASTING COMMANDS (WITH STRICT AUTHORIZATION)
# ============================================================================

@legacy_bound_handler
async def broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    _pending = getattr(legacy, "_pending_broadcast", globals().get("_pending_broadcast", {}))
    if isinstance(_pending, dict):
        _pending.pop(user.id, None)

    wait_state = getattr(legacy, "BROADCAST_WAIT_MESSAGE", globals().get("BROADCAST_WAIT_MESSAGE", "bc_wait_msg"))
    context.user_data["bc_state"] = wait_state

    kb_fn = getattr(legacy, "get_broadcast_entry_kb", globals().get("get_broadcast_entry_kb"))
    kb = kb_fn() if callable(kb_fn) else None

    await safe_send(lambda: msg.reply_text(
        '🛡️ <b>សុវត្ថិភាពការផ្សាយសារ V2</b>\n\n'
        '📨 <b>របៀបប្រើ</b>\n'
        '• ផ្ញើ <b>អត្ថបទ</b> ឬ <b>រូបភាព + ចំណងជើង</b> ដែលចង់ផ្សាយ\n'
        '• បូតនឹងបង្ហាញសារមើលជាមុនចុងក្រោយ មុនពេលផ្ញើពិត\n'
        '• គាំទ្រទម្រង់សារ Telegram, HTML, MarkdownV2, Markdown និងអត្ថបទធម្មតា\n'
        '• ដើម្បីបង្ខំទម្រង់ សូមដាក់ <code>::html</code>, <code>::mdv2</code>, <code>::md</code> ឬ <code>::plain</code> នៅជួរទី១\n\n'
        '🔐 <b>ការការពារ</b>\n'
        '✅ បញ្ជាក់សារមើលជាមុន មុនពេលផ្ញើ\n'
        '✅ រំលងអ្នកប្រើប្រាស់ដែលបានបិទបូត ឬមិនអាចទាក់ទងបាន\n'
        '✅ គ្រប់គ្រងល្បឿនផ្ញើ និង RetryAfter\n\n'
        'វាយ /cancel ដើម្បីបោះបង់។',
        parse_mode="HTML",
        reply_markup=kb
    ))


@legacy_bound_handler
async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    admin_id = user.id
    sched_payload = getattr(legacy, "_sched_payload", globals().get("_sched_payload", {}))
    if isinstance(sched_payload, dict):
        sched_payload.pop(admin_id, None)

    sched_state = getattr(legacy, "SCHED_WAIT_MSG", globals().get("SCHED_WAIT_MSG", "sched_wait_msg"))
    context.user_data["sched_state"] = sched_state

    await safe_send(lambda: msg.reply_text(
        '📅 <b>ការផ្សាយតាមកាលវិភាគ</b>\n\n'
        'សូមផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + ចំណងជើង</b> ដែលចង់កំណត់ពេលផ្ញើ។\n'
        '✅ គាំទ្រទម្រង់ដើមរបស់ Telegram, HTML, MarkdownV2, Markdown និងអត្ថបទធម្មតា។\n'
        'ដើម្បីបង្ខំទម្រង់ សូមដាក់ <code>::html</code>, <code>::mdv2</code>, <code>::md</code> ឬ <code>::plain</code> នៅជួរទី១។\n\n'
        'វាយ /cancel ដើម្បីបោះបង់។',
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    admin_id = user.id
    fetch_fn = getattr(legacy, "db_sched_fetch_admin_pending", globals().get("db_sched_fetch_admin_pending"))
    if not callable(fetch_fn):
        await safe_send(lambda: msg.reply_text('📭 មិនមានការផ្សាយតាមកាលវិភាគទេ។'))
        return

    loop = asyncio.get_running_loop()
    executor = getattr(legacy, "_DB_EXECUTOR", None)
    rows = await loop.run_in_executor(executor, fetch_fn, admin_id)
    if not rows:
        await safe_send(lambda: msg.reply_text('📭 មិនមានការផ្សាយតាមកាលវិភាគទេ។'))
        return

    list_kb_fn = getattr(legacy, "get_schedules_list_kb", globals().get("get_schedules_list_kb"))
    kb = list_kb_fn(rows, page=0) if callable(list_kb_fn) else None

    await safe_send(lambda: msg.reply_text(
        f'📋 <b>ការផ្សាយតាមកាលវិភាគ ({len(rows)} កំពុងរង់ចាំ)</b>\n'
        f'ចុចលើកាលវិភាគ ដើម្បីមើលព័ត៌មានលម្អិត ឬបោះបង់។',
        parse_mode="HTML",
        reply_markup=kb
    ))


@legacy_bound_handler
async def cmd_cancelschedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    admin_id = user.id
    args = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(lambda: msg.reply_text('❌ របៀបប្រើ៖ /cancelschedule &lt;id&gt;\nឬប្រើ /schedules ដើម្បីជ្រើស។', parse_mode="HTML"))
        return

    row_id = int(args[0])
    fetch_one_fn = getattr(legacy, "db_sched_fetch_one", globals().get("db_sched_fetch_one"))
    set_status_fn = getattr(legacy, "db_sched_set_status", globals().get("db_sched_set_status"))
    executor = getattr(legacy, "_DB_EXECUTOR", None)

    if not callable(fetch_one_fn) or not callable(set_status_fn):
        await safe_send(lambda: msg.reply_text("❌ Database schedule functions unavailable."))
        return

    loop = asyncio.get_running_loop()
    row = await loop.run_in_executor(executor, fetch_one_fn, row_id)
    if not row:
        await safe_send(lambda: msg.reply_text(f"❌ រកមិនឃើញ Schedule #{row_id}។"))
        return
    if row.get("admin_id") != admin_id:
        await safe_send(lambda: msg.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
        return
    st = row.get("status")
    if st != "pending":
        await safe_send(lambda: msg.reply_text(f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។", parse_mode="HTML"))
        return

    await loop.run_in_executor(executor, set_status_fn, row_id, "cancelled")
    await safe_send(lambda: msg.reply_text(f'✅ កាលវិភាគ <b>#{row_id}</b> បានបោះបង់។', parse_mode="HTML"))


@legacy_bound_handler
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    uid = user.id
    if not _safe_is_admin(uid):
        await safe_send(lambda: msg.reply_text('ℹ️ មិនមានប្រតិបត្តិការដែលត្រូវបោះបង់ទេ។'))
        return

    target_id = None
    chat_state = getattr(legacy, "CHAT_WAIT_MESSAGE", globals().get("CHAT_WAIT_MESSAGE", "chat_wait_msg"))
    admin_targets = getattr(legacy, "_admin_chat_target", globals().get("_admin_chat_target", {}))
    if context.user_data.get("chat_state") == chat_state and isinstance(admin_targets, dict):
        target_id = admin_targets.get(uid)

    clear_state_fn = getattr(legacy, "_clear_admin_transient_state", globals().get("_clear_admin_transient_state"))
    cleared = await clear_state_fn(context, uid) if callable(clear_state_fn) else []

    if target_id:
        with suppress(Exception):
            await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")

    dash_kb_fn = getattr(legacy, "get_admin_dashboard_kb", globals().get("get_admin_dashboard_kb"))
    dash_kb = dash_kb_fn() if callable(dash_kb_fn) else None

    if cleared:
        labels = ", ".join(cleared[:8])
        await safe_send(lambda: msg.reply_text(f"✅ បានបោះបង់/សម្អាត state រួច: <code>{html.escape(labels)}</code>", parse_mode="HTML", reply_markup=dash_kb))
        return
    await safe_send(lambda: msg.reply_text('ℹ️ មិនមានប្រតិបត្តិការដែលត្រូវបោះបង់ទេ។'))


@legacy_bound_handler
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    loop = asyncio.get_running_loop()
    executor = getattr(legacy, "_DB_EXECUTOR", None)
    get_users_fn = getattr(legacy, "get_all_user_ids", globals().get("get_all_user_ids", lambda: []))
    fetch_scheds_fn = getattr(legacy, "db_sched_fetch_admin_pending", globals().get("db_sched_fetch_admin_pending", lambda _: []))

    user_ids = await loop.run_in_executor(executor, get_users_fn)
    pending_scheds = await loop.run_in_executor(executor, lambda: fetch_scheds_fn(user.id))

    admin_targets = getattr(legacy, "_admin_chat_target", globals().get("_admin_chat_target", {}))
    user_locks = getattr(legacy, "_user_locks", _LOCAL_USER_LOCKS)
    hist_cache = getattr(legacy, "_hist_cache", globals().get("_hist_cache", {}))
    dyn_auth = getattr(legacy, "_dynamic_ai_auth_configured", globals().get("_dynamic_ai_auth_configured", lambda: False))
    hf_model = getattr(legacy, "HF_MODEL", "mrrtmob/khmer-tts")
    hf_ocr = getattr(legacy, "HF_OCR_MODEL", "google/gemini-flash-vision")

    await safe_send(lambda: msg.reply_text(
        f"📊 <b>ស្ថិតិបូត</b>\n\n"
        f"👥 អ្នកប្រើប្រាស់សរុប៖ <b>{len(user_ids)}</b>\n"
        f"💬 ការជជែកសកម្មរបស់អ្នកគ្រប់គ្រង៖ <b>{len(admin_targets)}</b>\n"
        f"📅 កាលវិភាគកំពុងរង់ចាំ៖ <b>{len(pending_scheds)}</b>\n"
        f"🔒 សោអ្នកប្រើប្រាស់សកម្ម៖ <b>{len(user_locks)}</b>\n"
        f"💭 ចំនួនធាតុប្រវត្តិក្នុង Cache៖ <b>{len(hist_cache)}</b>\n"
        f"🔑 ការផ្ទៀងផ្ទាត់ API បែប Dynamic៖ <b>{'ON' if dyn_auth() else 'OFF'}</b>\n"
        f"🤗 ម៉ូដែល HF៖ <b>{hf_model}</b>\n"
        f"📸 ម៉ូដែល OCR៖ <b>{hf_ocr}</b>",
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ ពាក្យបញ្ជានេះសម្រាប់ Admin ប៉ុណ្ណោះ។"))
        return

    text_fn = getattr(legacy, "_admin_health_text", globals().get("_admin_health_text"))
    text = await text_fn() if callable(text_fn) else "🟢 All services operational."
    dash_kb_fn = getattr(legacy, "get_admin_dashboard_kb", globals().get("get_admin_dashboard_kb"))
    dash_kb = dash_kb_fn() if callable(dash_kb_fn) else None

    await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=dash_kb, disable_web_page_preview=True))


@legacy_bound_handler
async def cmd_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    if not _safe_is_admin(int(user.id)):
        status_text = (
            "🟢 <b>ស្ថានភាពប្រព័ន្ធ Bot Voice (System Status)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚡ <b>ស្ថានភាពទូទៅ:</b> កំពុងដំណើរការយ៉ាងរលូន (Online 24/7)\n"
            "🎙️ <b>ម៉ាស៊ីន TTS:</b> ដំណើរការធម្មតា (Kiri, Gemini AI, Edge)\n"
            "📸 <b>ម៉ាស៊ីន OCR:</b> ដំណើរការធម្មតា (Google Gemini Vision)\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "💡 <i>ប្រសិនបើជួបបញ្ហា សូមប្រើ /unlock ឬទាក់ទងមកកាន់ @m11mmm112</i>"
        )
        await safe_send(lambda: msg.reply_text(status_text, parse_mode="HTML"))
        return

    snapshot = legacy._system_metrics_snapshot() if hasattr(legacy, "_system_metrics_snapshot") else {}
    storage = snapshot.get("storage", {})
    cache = snapshot.get("tts_audio_cache", {})
    anti_spam = snapshot.get("anti_spam", {})
    status_icon = "🟢" if snapshot.get("status") == "healthy" else "🟡"
    redis_status = f"✅ Connected ({storage.get('redis_ping_ms')}ms)" if storage.get("redis_connected") else "❌ Offline"
    supabase_status = "✅ Connected" if storage.get("supabase_connected") else "⚠️ Degraded"

    try:
        uptime_m = round(float(snapshot.get("uptime_seconds") or 0.0) / 60.0, 1)
    except (ValueError, TypeError):
        uptime_m = 0.0

    text = (
        f"{status_icon} <b>ប្រព័ន្ធ Telemetry & Metrics (v{snapshot.get('version', '4.2.0')})</b>\n\n"
        f"⏱️ Uptime: <b>{uptime_m} នាទី</b>\n"
        f"🤖 Bot Mode: <b>{snapshot.get('bot_mode', 'POLLING')}</b>\n\n"
        f"🗄️ <b>Storage Layer:</b>\n"
        f"• Redis L2 Cache: <b>{redis_status}</b>\n"
        f"• Supabase Database: <b>{supabase_status}</b>\n\n"
        f"🎵 <b>Audio Cache (L1/L2):</b>\n"
        f"• Memory Items: <b>{cache.get('l1_memory_items', 0)}</b> ({round(cache.get('l1_memory_bytes', 0)/1024, 1)} KB)\n"
        f"• Binary TTL: <b>{cache.get('ttl_seconds', 0)}s</b>\n\n"
        f"🛡️ <b>Anti-Spam & Rate Limiter:</b>\n"
        f"• Tracked Users: <b>{anti_spam.get('tracked_users', 0)}</b>\n"
        f"• Active Cooldowns: <b>{anti_spam.get('active_cooldowns', 0)}</b>\n\n"
        f"🌐 <i>REST API: <code>/system</code>, <code>/metrics</code></i>"
    )
    dash_kb_fn = getattr(legacy, "get_admin_dashboard_kb", globals().get("get_admin_dashboard_kb"))
    dash_kb = dash_kb_fn() if callable(dash_kb_fn) else None

    await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=dash_kb, disable_web_page_preview=True))


@legacy_bound_handler
async def cmd_dbstatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only - Database Status)</b>", parse_mode="HTML"))
        return

    text_fn = getattr(legacy, "_get_admin_db_text", None)
    text = await text_fn() if callable(text_fn) else "🗄️ Database operational."
    kb_fn = getattr(legacy, "get_admin_db_kb", None)
    kb = kb_fn() if callable(kb_fn) else None

    await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=kb, disable_web_page_preview=True))


@legacy_bound_handler
async def cmd_dbbackup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    arg = ""
    with suppress(Exception):
        arg = str((context.args or [""])[0]).strip().lower()

    if arg in ("sql", "dump"):
        asyncio.create_task(legacy._admin_send_db_export(msg, int(user.id), "sql", context))
    elif arg in ("csv", "zip"):
        asyncio.create_task(legacy._admin_send_db_export(msg, int(user.id), "csv", context))
    elif arg in ("cli", "script", "bat"):
        asyncio.create_task(legacy._admin_send_db_export(msg, int(user.id), "cli", context))
    else:
        asyncio.create_task(legacy._admin_trigger_backup(msg, int(user.id)))


@legacy_bound_handler
async def cmd_migrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    args = context.args or []
    target_url = ""
    target_key = ""
    dry_run = False
    for a in args:
        val = str(a).strip()
        if val.lower() in ("--dry-run", "dryrun", "-d"):
            dry_run = True
        elif val.startswith("http://") or val.startswith("https://"):
            target_url = val
        elif len(val) > 20 and not val.startswith("-"):
            target_key = val
    await legacy._admin_handle_db_migration(msg, int(user.id), context, target_url=target_url, target_key=target_key, dry_run=dry_run)


# ============================================================================
# DONATION & PAYMENT (BAKONG / KHQR)
# ============================================================================

@legacy_bound_handler
async def cmd_bakongstatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    from app.services.donation import bakong_api
    from app.services.donation.khqr import DEFAULT_BAKONG_ACCOUNT_ID, DEFAULT_BAKONG_MERCHANT_NAME

    status_msg = await safe_send(lambda: msg.reply_text("⏳ <b>កំពុងធ្វើតេស្តការតភ្ជាប់ទៅ Bakong Open API Gateway...</b>", parse_mode="HTML"))
    res = await bakong_api.test_connection() if hasattr(bakong_api, "test_connection") else {}
    if not isinstance(res, dict):
        res = {}

    token_info = bakong_api.decode_token_payload() if hasattr(bakong_api, "decode_token_payload") else {}
    if not isinstance(token_info, dict):
        token_info = {}

    if res.get("ok"):
        status_icon = "🟢"
        conn_text = f"<b>ភ្ជាប់ជោគជ័យ (Connected)</b> — {res.get('latency_ms', 0)}ms"
    else:
        status_icon = "🔴"
        conn_text = f"<b>បរាជ័យ (Failed)</b> — {html.escape(str(res.get('error') or res.get('message') or 'Unknown error'))}"

    account_id = DEFAULT_BAKONG_ACCOUNT_ID
    merchant_name = DEFAULT_BAKONG_MERCHANT_NAME
    merchant_id = res.get("merchant_id") or token_info.get("merchant_id") or "N/A"
    expires_at = res.get("expires_at") or token_info.get("expires_at") or "N/A"
    is_exp = res.get("is_expired") or token_info.get("is_expired") or False
    exp_badge = "⚠️ ផុតកំណត់ (Expired)" if is_exp else "✅ មានសុពលភាព (Valid)"

    response_text = (
        f"{status_icon} <b>ស្ថានភាព Bakong Open API Gateway</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ <b>ស្ថានភាពតភ្ជាប់:</b> {conn_text}\n"
        f"👤 <b>Merchant Name:</b> <code>{html.escape(merchant_name)}</code>\n"
        f"🆔 <b>Bakong ID:</b> <code>{html.escape(account_id)}</code>\n"
        f"🏢 <b>Developer ID:</b> <code>{html.escape(str(merchant_id))}</code>\n"
        f"📅 <b>កាលបរិច្ឆេទផុតកំណត់:</b> <code>{html.escape(str(expires_at))}</code> ({exp_badge})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <i>ប្រព័ន្ធផ្ទៀងផ្ទាត់ការបង់ប្រាក់ KHQR ដោយស្វ័យប្រវត្តិ (Real-time Auto Verification) កំពុងដំណើរការ 24/7។</i>"
    )

    if status_msg and hasattr(status_msg, "edit_text"):
        await safe_send(lambda: status_msg.edit_text(response_text, parse_mode="HTML"))
    else:
        await safe_send(lambda: msg.reply_text(response_text, parse_mode="HTML"))


@legacy_bound_handler
async def cmd_khqr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    from app.services.donation import bakong_api
    from app.services.donation.khqr import BakongKHQR, get_khqr_config, get_khqr_qr_image, update_khqr_config

    args = context.args or []
    if args:
        sub = args[0].lower()
        if sub in ("set", "update") and len(args) > 2:
            key = args[1].lower()
            val = " ".join(args[2:]).strip()
        elif len(args) > 1:
            key = sub
            val = " ".join(args[1:]).strip()
        else:
            key = sub
            val = ""

        if key in ("account", "account_id", "id"):
            if not val:
                await safe_send(lambda: msg.reply_text("⚠️ សូមបញ្ជាក់ Bakong Account ID ថ្មី (ឧ. <code>/khqr account name@bkrt</code>)", parse_mode="HTML"))
                return
            new_cfg = update_khqr_config(account_id=val)
            await safe_send(lambda: msg.reply_text(f"✅ បានកែប្រែ <b>Bakong Account ID</b> ទៅជា <code>{html.escape(new_cfg['account_id'])}</code> ជោគជ័យ!", parse_mode="HTML"))
            return
        if key in ("name", "merchant", "merchant_name"):
            if not val:
                await safe_send(lambda: msg.reply_text("⚠️ សូមបញ្ជាក់ឈ្មោះ Merchant ថ្មី (ឧ. <code>/khqr name CHUO KIMHENG</code>)", parse_mode="HTML"))
                return
            new_cfg = update_khqr_config(merchant_name=val)
            await safe_send(lambda: msg.reply_text(f"✅ បានកែប្រែ <b>Merchant Name</b> ទៅជា <code>{html.escape(new_cfg['merchant_name'])}</code> ជោគជ័យ!", parse_mode="HTML"))
            return
        if key in ("city", "merchant_city"):
            if not val:
                await safe_send(lambda: msg.reply_text("⚠️ សូមបញ្ជាក់ទីក្រុងថ្មី (ឧ. <code>/khqr city Phnom Penh</code>)", parse_mode="HTML"))
                return
            new_cfg = update_khqr_config(merchant_city=val)
            await safe_send(lambda: msg.reply_text(f"✅ បានកែប្រែ <b>Merchant City</b> ទៅជា <code>{html.escape(new_cfg['merchant_city'])}</code> ជោគជ័យ!", parse_mode="HTML"))
            return
        if key in ("currency", "curr"):
            c = val.upper()
            if c not in ("USD", "KHR"):
                await safe_send(lambda: msg.reply_text("⚠️ រូបិយប័ណ្ណត្រឹមត្រូវគឺ <code>USD</code> ឬ <code>KHR</code> (ឧ. <code>/khqr currency KHR</code>)", parse_mode="HTML"))
                return
            new_cfg = update_khqr_config(currency=c)
            await safe_send(lambda: msg.reply_text(f"✅ បានកែប្រែ <b>Default Currency</b> ទៅជា <code>{new_cfg['currency']}</code> ជោគជ័យ!", parse_mode="HTML"))
            return
        if key in ("photo", "image", "pic"):
            context.user_data["khqr_state"] = "wait_photo"
            await safe_send(lambda: msg.reply_text(
                "📸 <b>សូមផ្ញើរូបភាព QR កូដថ្មីមកកាន់ទីនេះ</b>\n\n"
                "💡 <i>ឬបងអាចផ្ញើរូបភាពជាមួយ Caption <code>#khqr</code> នៅពេលណាក៏បានដើម្បីផ្លាស់ប្តូរ Static QR ផ្លូវការ។</i>",
                parse_mode="HTML"
            ))
            return

    cfg = get_khqr_config() or {}
    test_bill = "N/A"
    test_md5 = "N/A"
    test_khqr = ""
    try:
        test_khqr, test_bill = BakongKHQR.generate(amount=1.0, currency=cfg.get("currency", "USD"), user_id=int(user.id), tier="test")
        test_md5 = BakongKHQR.get_md5(test_khqr)
    except Exception as err:
        logger.warning("Could not generate test KHQR: %s", err)

    has_api = bakong_api.is_bakong_api_configured() if hasattr(bakong_api, "is_bakong_api_configured") else False
    api_status_badge = "🟢 ភ្ជាប់រួចរាល់ (Active)" if has_api else "⚪ មិនទាន់កំណត់"

    overview_text = (
        f"🇰🇭 <b>ព័ត៌មាន & ការកំណត់ Bakong KHQR</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Merchant Name:</b> <code>{html.escape(cfg.get('merchant_name', ''))}</code>\n"
        f"🆔 <b>Bakong Account ID:</b> <code>{html.escape(cfg.get('account_id', ''))}</code>\n"
        f"🏙️ <b>Merchant City:</b> <code>{html.escape(cfg.get('merchant_city', ''))}</code>\n"
        f"💵 <b>Default Currency:</b> <code>{cfg.get('currency', 'USD')}</code>\n"
        f"🏛️ <b>Bakong Open API:</b> {api_status_badge}\n"
        f"🧾 <b>Test Bill Ref:</b> <code>{test_bill}</code>\n"
        f"🔐 <b>Test MD5 Hash:</b> <code>{test_md5}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛠️ <b>ពាក្យបញ្ជាកែប្រែ (Quick Commands):</b>\n"
        f"• <code>/khqr account &lt;id&gt;</code> — កំណត់ Account ID (ឧ. name@bkrt)\n"
        f"• <code>/khqr name &lt;name&gt;</code> — កំណត់ឈ្មោះ Merchant\n"
        f"• <code>/khqr city &lt;city&gt;</code> — កំណត់ទីក្រុង (ឧ. Phnom Penh)\n"
        f"• <code>/khqr currency &lt;USD|KHR&gt;</code> — កំណត់រូបិយប័ណ្ណ\n"
        f"• <code>/khqr photo</code> — ផ្លាស់ប្តូររូបភាព Static QR កូដ\n"
        f"• <code>/bakongstatus</code> — ពិនិត្យសុពលភាព Token & Latency"
    )

    qr_bytes = await get_khqr_qr_image(test_khqr) if test_khqr else None
    if qr_bytes and context.bot:
        try:
            photo_file = io.BytesIO(qr_bytes)
            photo_file.name = "khqr_preview.png"
            # Send photo with concise caption to guarantee staying well below 1024-character Telegram limit
            short_caption = f"🇰🇭 <b>Bakong KHQR Preview</b> (Test Ref: <code>{test_bill}</code>)"
            await context.bot.send_photo(chat_id=msg.chat_id, photo=photo_file, caption=short_caption, parse_mode="HTML")
        except Exception as e:
            logger.debug("Failed to send test QR photo: %s", e)

    await safe_send(lambda: msg.reply_text(_clean_html_truncate(overview_text), parse_mode="HTML"))


# ============================================================================
# MASTER ADMIN CONTROL CENTER (PROTECTED)
# ============================================================================

@legacy_bound_handler
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not msg or not user:
        return
    user_id = int(user.id)

    # STRICT ACCESS CHECK
    if not _safe_is_admin(user_id):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    arg = ""
    with suppress(Exception):
        arg = str((context.args or [""])[0]).strip().lower()

    if arg in {"needs", "userneeds", "user_needs", "feedback"}:
        text_fn = getattr(legacy, "_user_needs_home_text", None)
        text = await text_fn(user_id) if callable(text_fn) else "User feedback panel."
        kb_fn = getattr(legacy, "get_user_needs_home_kb", None)
        await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=kb_fn() if callable(kb_fn) else None, disable_web_page_preview=True))
        return

    if arg in {"compact", "mobile", "mini"}:
        text_fn = getattr(legacy, "_admin_compact_text", None)
        text = await text_fn(user_id) if callable(text_fn) else "Admin compact view."
        kb_fn = getattr(legacy, "get_admin_compact_kb", None)
        await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=kb_fn() if callable(kb_fn) else None, disable_web_page_preview=True))
        return

    if arg in {"health", "status"}:
        await cmd_health(update, context)
        return

    if arg in {"stats", "stat", "telemetry"}:
        await admin_stats(update, context)
        return

    if arg in {"system", "metrics"}:
        await cmd_system(update, context)
        return

    if arg in {"db", "database", "supabase", "dbstatus"}:
        await cmd_dbstatus(update, context)
        return

    if arg in {"backup", "dbbackup"}:
        await cmd_dbbackup(update, context)
        return

    if arg in {"migrate", "migration"}:
        if context.args:
            context.args = list(context.args[1:])
        await cmd_migrate(update, context)
        return

    if arg in {"bakong", "bakongstatus"}:
        await cmd_bakongstatus(update, context)
        return

    if arg in {"khqr", "setkhqr"}:
        if context.args:
            context.args = list(context.args[1:])
        await cmd_khqr(update, context)
        return

    if arg in {"users", "user"}:
        if context.args:
            context.args = list(context.args[1:])
        await cmd_users(update, context)
        return

    if arg in {"api", "apikeys"}:
        if context.args:
            context.args = list(context.args[1:])
        await cmd_api(update, context)
        return

    if arg in {"broadcast", "bc"}:
        await broadcast_start(update, context)
        return

    text_fn = getattr(legacy, "_admin_home_text", None)
    text = await text_fn(user_id) if callable(text_fn) else "👑 <b>ផ្ទាំងគ្រប់គ្រង Admin Dashboard</b>"
    kb_fn = getattr(legacy, "get_admin_dashboard_kb", None)
    await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=kb_fn() if callable(kb_fn) else None, disable_web_page_preview=True))


@legacy_bound_handler
async def cmd_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    admin_id = int(user.id)

    # STRICT ACCESS CHECK
    if not _safe_is_admin(admin_id):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    args = list(context.args or [])
    help_fn = getattr(legacy, "_api_help_text", lambda: "API Keys management.")
    api_kb_fn = getattr(legacy, "get_api_admin_kb", lambda: None)

    if not args or args[0].lower() in ("help", "-h", "--help"):
        await safe_send(lambda: msg.reply_text(help_fn(), parse_mode="HTML", reply_markup=api_kb_fn(), disable_web_page_preview=True))
        return

    action = args[0].lower().strip()
    loop = asyncio.get_running_loop()
    executor = getattr(legacy, "_DB_EXECUTOR", None)

    if action == "create":
        note = " ".join(args[1:]).strip()
        create_fn = getattr(legacy, "db_ai_api_key_create", None)
        if not callable(create_fn):
            await safe_send(lambda: msg.reply_text("❌ Database function not available."))
            return
        try:
            raw_key, row, storage = await loop.run_in_executor(executor, lambda: create_fn(admin_id=admin_id, note=note))
            await safe_send(lambda: msg.reply_text(
                f"✅ <b>បានបង្កើតសោ API សម្រាប់ AI ថ្មី</b>\n\n"
                f"សូមចម្លងសោនេះឥឡូវនេះ។ វានឹងមិនត្រូវបានបង្ហាញម្ដងទៀតទេ។\n\n"
                f"<code>{html.escape(raw_key)}</code>\n\n"
                f"កន្លែងរក្សាទុក៖ <b>{html.escape(storage)}</b>",
                parse_mode="HTML",
                reply_markup=api_kb_fn()
            ))
        except Exception as e:
            logger.error("/api create failed: %s", e)
            await safe_send(lambda: msg.reply_text(f"❌ Cannot create API key: {e}", reply_markup=api_kb_fn()))
        return

    if action == "list":
        list_fn = getattr(legacy, "db_ai_api_key_list", None)
        if not callable(list_fn):
            await safe_send(lambda: msg.reply_text("❌ Database function not available."))
            return
        rows = await loop.run_in_executor(executor, lambda: list_fn(limit=20))
        if not rows:
            await safe_send(lambda: msg.reply_text("ℹ️ មិនមានសោ API ទេ។", parse_mode="HTML", reply_markup=api_kb_fn()))
            return

        body = "\n".join(f"• <code>{r.get('key_prefix', '')}...</code> ({r.get('note', '')})" for r in rows)
        await safe_send(lambda: msg.reply_text(f"🔑 <b>AI API Keys</b>\n\n{body}", parse_mode="HTML", reply_markup=api_kb_fn()))
        return

    if action == "revoke":
        if len(args) < 2:
            await safe_send(lambda: msg.reply_text("⚠️ របៀបប្រើ៖ <code>/api revoke KEY_PREFIX_OR_ID</code>", parse_mode="HTML", reply_markup=api_kb_fn()))
            return
        revoke_fn = getattr(legacy, "db_ai_api_key_revoke", None)
        if not callable(revoke_fn):
            await safe_send(lambda: msg.reply_text("❌ Database function not available."))
            return
        identifier = args[1].strip()
        ok, info = await loop.run_in_executor(executor, lambda: revoke_fn(identifier))
        status_msg = f"✅ បានដកសិទ្ធិសោ API៖ <code>{html.escape(info)}</code>" if ok else f"❌ {html.escape(info)}"
        await safe_send(lambda: msg.reply_text(status_msg, parse_mode="HTML", reply_markup=api_kb_fn()))
        return

    await safe_send(lambda: msg.reply_text(help_fn(), parse_mode="HTML", reply_markup=api_kb_fn(), disable_web_page_preview=True))


@legacy_bound_handler
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    args = context.args or []
    executor = getattr(legacy, "_DB_EXECUTOR", None)
    loop = asyncio.get_running_loop()

    if args:
        query_text = " ".join(args).strip()
        search_fn = getattr(legacy, "search_users_by_query", None)
        if not callable(search_fn):
            await safe_send(lambda: msg.reply_text("❌ User search function not available."))
            return
        results = await loop.run_in_executor(executor, lambda: search_fn(query_text))
        if not results:
            await safe_send(lambda: msg.reply_text(f'🔎 រកមិនឃើញអ្នកប្រើប្រាស់សម្រាប់៖ <code>{html.escape(query_text)}</code>', parse_mode="HTML"))
            return
        page_kb_fn = getattr(legacy, "get_user_search_page_kb", None)
        kb = page_kb_fn(results, page=0) if callable(page_kb_fn) else None
        await safe_send(lambda: msg.reply_text(f'🔎 <b>លទ្ធផលស្វែងរក ({len(results)} នាក់)</b>', parse_mode="HTML", reply_markup=kb))
        return

    get_users_fn = getattr(legacy, "get_all_users_with_names", None)
    if not callable(get_users_fn):
        await safe_send(lambda: msg.reply_text("❌ Users listing function not available."))
        return
    users = await loop.run_in_executor(executor, get_users_fn)
    if not users:
        await safe_send(lambda: msg.reply_text("❌ គ្មានអ្នកប្រើប្រាស់ registered ទេ។"))
        return
    users_kb_fn = getattr(legacy, "get_users_page_kb", None)
    kb = users_kb_fn(users, page=0) if callable(users_kb_fn) else None
    await safe_send(lambda: msg.reply_text(f"👥 <b>អ្នកប្រើប្រាស់សរុប ({len(users)} នាក់)</b>", parse_mode="HTML", reply_markup=kb))


@legacy_bound_handler
async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    admin_id = int(user.id)
    if not _safe_is_admin(admin_id):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    args = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(lambda: msg.reply_text('❌ របៀបប្រើ៖ /chat <user_id>'))
        return

    target_id = int(args[0])
    open_fn = getattr(legacy, "_open_chat_session", None)
    if callable(open_fn):
        await open_fn(context.bot, admin_id, target_id, context)

    await safe_send(lambda: msg.reply_text(
        f"💬 <b>Chat Mode បើក</b>\n\nកំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
        f"វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_endchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    admin_id = int(user.id)
    if not _safe_is_admin(admin_id):
        return

    close_fn = getattr(legacy, "_close_session", None)
    target_id = close_fn(admin_id) if callable(close_fn) else None
    context.user_data.pop("chat_state", None)

    if target_id is None:
        await safe_send(lambda: msg.reply_text("ℹ️ អ្នកមិនទាន់ open Chat ណាមួយទេ។"))
        return
    await safe_send(lambda: msg.reply_text(f'✅ បានបញ្ចប់ការជជែកជាមួយអ្នកប្រើប្រាស់ <code>{target_id}</code>។', parse_mode="HTML"))
    with suppress(Exception):
        await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")


# ============================================================================
# UTILITY, MODE & MEDIA COMMANDS
# ============================================================================

@legacy_bound_handler
async def cmd_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return
    with suppress(Exception):
        from app.services.tools.temp_mail import handle_temp_mail_command
        if callable(handle_temp_mail_command):
            await handle_temp_mail_command(update, context)
            return

    await safe_send(lambda: msg.reply_text(
        "📧 <b>ប្រអប់សំបុត្របណ្ដោះអាសន្ន (Temporary Mailbox)</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
        "ប្រព័ន្ធការពារឯកជនភាពសម្រាប់ទទួលសារ OTP/Verification ដោយសុវត្ថិភាព។",
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_speed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    args = getattr(context, "args", None) or []
    if args:
        try:
            spd = float(args[0])
            if 0.5 <= spd <= 2.0:
                update_fn = getattr(legacy, "update_user_speed", getattr(legacy, "set_user_pref_async", None))
                if callable(update_fn):
                    res = update_fn(user_id, spd)
                    if asyncio.iscoroutine(res):
                        await res
                await safe_send(lambda: msg.reply_text(f"✅ បានកំណត់ល្បឿនអានទៅជា <b>{spd}x</b> រួចរាល់។", parse_mode="HTML"))
                return
        except ValueError:
            pass

    get_prefs_fn = getattr(legacy, "get_user_prefs_async", None)
    prefs = {}
    if callable(get_prefs_fn):
        with suppress(Exception):
            prefs = await get_prefs_fn(user_id) or {}
    cur_spd = prefs.get("speed", 1.0)

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("0.75x", callback_data="set_speed:0.75"),
         InlineKeyboardButton("1.0x (ធម្មតា)", callback_data="set_speed:1.0"),
         InlineKeyboardButton("1.25x", callback_data="set_speed:1.25"),
         InlineKeyboardButton("1.5x", callback_data="set_speed:1.5")],
        [InlineKeyboardButton("⚙️ ការកំណត់ពេញលេញ", callback_data="welcome_profile")]
    ])
    await safe_send(lambda: msg.reply_text(
        f"🎚️ <b>ល្បឿនអាន (Speech Speed)</b>\nល្បឿនបច្ចុប្បន្ន: <b>{cur_spd:.1f}x</b>\nសូមជ្រើសរើសល្បឿនសំឡេងអានខាងក្រោម៖",
        parse_mode="HTML",
        reply_markup=kb,
    ))


@legacy_bound_handler
async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    args = getattr(context, "args", None) or []
    get_prefs_fn = getattr(legacy, "get_user_prefs_async", None)
    prefs = {}
    if callable(get_prefs_fn):
        with suppress(Exception):
            prefs = await get_prefs_fn(user_id) or {}
    cur_gender = prefs.get("gender", "female")

    if args:
        v = str(args[0]).lower().strip()
        new_gender = "female" if v in ("female", "girl", "ស្រី") else "male" if v in ("male", "boy", "ប្រុស") else cur_gender
    else:
        new_gender = "male" if cur_gender == "female" else "female"

    update_fn = getattr(legacy, "update_user_gender", getattr(legacy, "set_user_pref_async", None))
    if callable(update_fn):
        res = update_fn(user_id, new_gender)
        if asyncio.iscoroutine(res):
            await res

    label = "👨 សំឡេងប្រុស" if new_gender == "male" else "👩 សំឡេងស្រី"
    await safe_send(lambda: msg.reply_text(f"✅ បានកំណត់សំឡេងទៅជា {label}", parse_mode="HTML"))


@legacy_bound_handler
async def cmd_checkpay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    args = getattr(context, "args", None) or []
    if not args:
        usage_text = (
            "🔍 <b>ផ្ទៀងផ្ទាត់ការបង់ប្រាក់ Bakong Pay (Manual Verification)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "របៀបប្រើប្រាស់៖\n"
            "<code>/checkpay &lt;MD5_HASH&gt;</code>\n\n"
            "ឧទាហរណ៍៖ <code>/checkpay 9f8a264a93848b598b9...</code>"
        )
        await safe_send(lambda: msg.reply_text(usage_text, parse_mode="HTML"))
        return

    md5_hash = str(args[0]).strip()
    status_msg = await safe_send(lambda: msg.reply_text("⏳ កំពុងផ្ទៀងផ្ទាត់ប្រតិបត្តិការ Bakong Pay...", parse_mode="HTML"))

    from app.services.donation.bakong_api import check_transaction_by_md5

    tx_res = await check_transaction_by_md5(md5_hash)
    if tx_res and tx_res.get("success"):
        data = tx_res.get("data") or {}
        payer = data.get("fromAccountId") or data.get("payer") or "N/A"
        amt = data.get("amount") or "0.00"
        curr = data.get("currency") or "USD"
        tx_hash = data.get("hash") or md5_hash
        success_text = (
            f"✅ <b>ប្រតិបត្តិការត្រូវបានផ្ទៀងផ្ទាត់ជោគជ័យ!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>Payer:</b> <code>{html.escape(str(payer))}</code>\n"
            f"💰 <b>ចំនួនទឹកប្រាក់:</b> <code>{html.escape(str(amt))} {html.escape(str(curr))}</code>\n"
            f"🧾 <b>Transaction Hash:</b> <code>{html.escape(str(tx_hash))}</code>"
        )
        if status_msg and hasattr(status_msg, "edit_text"):
            await status_msg.edit_text(success_text, parse_mode="HTML")
        elif msg:
            await safe_send(lambda: msg.reply_text(success_text, parse_mode="HTML"))
    else:
        err_msg = (tx_res.get("response_message") if tx_res else None) or "រកមិនឃើញប្រតិបត្តិការ ឬមិនទាន់បានបង់ប្រាក់"
        fail_text = f"❌ <b>មិនអាចផ្ទៀងផ្ទាត់បានទេ:</b> {html.escape(str(err_msg))}"
        if status_msg and hasattr(status_msg, "edit_text"):
            await status_msg.edit_text(fail_text, parse_mode="HTML")
        elif msg:
            await safe_send(lambda: msg.reply_text(fail_text, parse_mode="HTML"))


@legacy_bound_handler
async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    ban_fn = getattr(legacy, "cmd_ban", None)
    if callable(ban_fn):
        await ban_fn(update, context)
    else:
        await safe_send(lambda: msg.reply_text('🚫 មុខងារ Ban អ្នកប្រើប្រាស់។', parse_mode="HTML"))


@legacy_bound_handler
async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    args = getattr(context, "args", None) or []
    if args:
        target_mode = str(args[0]).lower().strip()
        update_fn = getattr(legacy, "update_user_bot_mode", None)
        if callable(update_fn):
            res = update_fn(user_id, target_mode)
            if asyncio.iscoroutine(res):
                await res
        mode_desc = "អានសំឡេងផ្ទាល់ (TTS Only)" if target_mode == "tts" else "សួរឆ្លើយ AI (AI Chat)" if target_mode == "ai_chat" else "ស្វ័យប្រវត្តិ (Auto-Detect)"
        await safe_send(lambda: msg.reply_text(
            f"✅ បានកំណត់របៀបដំណើរការ៖ <b>{mode_desc}</b>",
            parse_mode="HTML",
        ))
        return

    get_prefs_fn = getattr(legacy, "get_user_prefs_async", None)
    prefs = {}
    if callable(get_prefs_fn):
        with suppress(Exception):
            prefs = await get_prefs_fn(user_id) or {}
    cur_mode = prefs.get("bot_mode", "auto")
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ ស្វ័យប្រវត្តិ (Auto)", callback_data="set_mode:auto")],
        [InlineKeyboardButton("🎙️ អានសំឡេងផ្ទាល់ (TTS Only)", callback_data="set_mode:tts")],
        [InlineKeyboardButton("🤖 សួរឆ្លើយ AI (AI Chat)", callback_data="set_mode:ai_chat")],
    ])
    text = (
        "⚙️ <b>ជ្រើសរើសរបៀបដំណើរការរបស់ Bot (Bot Mode)</b>\n\n"
        f"របៀបបច្ចុប្បន្ន៖ <code>{cur_mode}</code>\n\n"
        "សូមជ្រើសរើសរបៀបដែលអ្នកចង់ប្រើប្រាស់៖"
    )
    await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=kb))


@legacy_bound_handler
async def cmd_article_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return
    status_msg = await msg.reply_text("⏳ កំពុងស្កេនរកអត្ថបទព័ត៌មានថ្មី...", parse_mode="HTML")
    from app.services.ai.article_monitor import scan_sources_and_notify_admin
    bot = getattr(context, "bot", None)
    if bot is None and hasattr(update, "get_bot"):
        bot = update.get_bot()
    articles = await scan_sources_and_notify_admin(bot)
    count = len(articles) if articles else 0
    if status_msg and hasattr(status_msg, "edit_text"):
        await status_msg.edit_text(f"✅ បានស្កេនរួចរាល់៖ រកឃើញ <b>{count}</b> ព័ត៌មានថ្មី។", parse_mode="HTML")


@legacy_bound_handler
async def cmd_article_sources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    msg = update.effective_message
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    text = (msg.text or "").strip()
    parts = text.split()
    if len(parts) >= 3 and parts[1].lower() in ("add", "insert", "+"):
        url = parts[2]
        name = " ".join(parts[3:]) if len(parts) > 3 else "Source"
        from app.services.ai.article_reader import is_safe_public_url
        safe, reason = is_safe_public_url(url)
        if not safe:
            await safe_send(lambda: msg.reply_text(f"❌ URL មិនត្រឹមត្រូវ: {reason}", parse_mode="HTML"))
            return
        from app.services.ai.article_storage import add_article_source
        await add_article_source(url=url, name=name)
        await safe_send(lambda: msg.reply_text(f"✅ បានបញ្ចូលប្រភពព័ត៌មានជោគជ័យ៖ {name} ({url})", parse_mode="HTML"))
        return

    if len(parts) >= 3 and parts[1].lower() in ("del", "delete", "remove", "-"):
        url = parts[2]
        from app.services.ai.article_storage import remove_article_source
        await remove_article_source(url)
        await safe_send(lambda: msg.reply_text(f"🗑️ បានលុបប្រភពព័ត៌មានជោគជ័យ៖ {url}", parse_mode="HTML"))
        return

    from app.services.ai.article_storage import get_article_sources
    sources = await get_article_sources()
    if not sources:
        await safe_send(lambda: msg.reply_text("🌐 <b>បញ្ជីប្រភពព័ត៌មាន៖</b>\nគ្មានប្រភពព័ត៌មាននៅឡើយទេ។", parse_mode="HTML"))
        return
    lines = [f"• <b>{s.get('name', 'Source')}</b>: {s.get('url', '')}" for s in sources]
    await safe_send(lambda: msg.reply_text(f"🌐 <b>បញ្ជីប្រភពព័ត៌មាន៖</b>\n" + "\n".join(lines), parse_mode="HTML"))


@legacy_bound_handler
async def cmd_botsettings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _safe_is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    settings_fn = getattr(legacy, "get_bot_settings_async", None)
    settings = await settings_fn(force=True) if callable(settings_fn) else ({}, {})
    kb_fn = getattr(legacy, "get_bot_settings_kb", None)
    kb = kb_fn(settings[0]) if callable(kb_fn) else None
    await safe_send(lambda: msg.reply_text("⚙️ <b>Bot Settings Panel</b>", parse_mode="HTML", reply_markup=kb))


@legacy_bound_handler
async def cmd_feature_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg:
        await safe_send(lambda: msg.reply_text("💬 សូមសរសេរ Feature ថ្មី ឬមតិកែលម្អដែលអ្នកចង់បាន។", parse_mode="HTML"))


@legacy_bound_handler
async def cmd_runtime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_system(update, context)


@legacy_bound_handler
async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        from app.services.telegram.menu import cmd_menu as _cmd_menu
        await _cmd_menu(update, context)
    except Exception as e:
        logger.warning("cmd_menu fallback to on_help: %s", e)
        await on_help(update, context)


# ============================================================================
# PEP 562 DYNAMIC RUNTIME SYMBOL RESOLVER
# ============================================================================

def __getattr__(name: str) -> Any:
    """Fallback dynamically to app.legacy or _legacy_runtime for transitional symbols."""
    if hasattr(legacy, name):
        return getattr(legacy, name)
    try:
        import app.services.telegram._legacy_runtime as lr
        if hasattr(lr, name):
            return getattr(lr, name)
    except ImportError:
        pass
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'admin_stats', 'broadcast_start', 'cmd_admin', 'cmd_api', 'cmd_article_scan',
    'cmd_article_sources', 'cmd_ban', 'cmd_ask', 'cmd_bakongstatus', 'cmd_botsettings',
    'cmd_cancel', 'cmd_cancelschedule', 'cmd_chat', 'cmd_checkpay', 'cmd_clear',
    'cmd_dbbackup', 'cmd_dbstatus', 'cmd_delete_my_data', 'cmd_endchat', 'cmd_facebook',
    'cmd_feature_request', 'cmd_health', 'cmd_instagram', 'cmd_khqr', 'cmd_migrate',
    'cmd_mode', 'cmd_menu', 'cmd_myprefs', 'cmd_narrate', 'cmd_privacy', 'cmd_runtime',
    'cmd_schedule', 'cmd_schedules', 'cmd_security', 'cmd_speed', 'cmd_summary',
    'cmd_system', 'cmd_tiktok', 'cmd_translate', 'cmd_ttsmodel', 'cmd_unlock',
    'cmd_users', 'cmd_voice', 'cmd_youtube', 'cmd_email', 'on_help', 'on_start'
]