"""Pure routing helpers for Telegram callback flows and text intent classification.

Keeping callback and intent classification independent from the legacy runtime
makes the dispatcher easy to test and ensures clean routing between TTS and AI.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection
from contextlib import suppress
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Khmer question & conversation patterns
_KHMER_QUESTION_STARTS = (
    "តើ", "ហេតុអ្វី", "ហេតុអី", "របៀប", "ម្តេច", "ម៉េច", "យ៉ាងម៉េច",
    "យ៉ាងណា", "អ្វីខ្លះ", "អ្វីទៅ", "នៅឯណា", "នៅកន្លែងណា", "កាលណា",
    "នរណា", "អ្នកណា", "ប៉ុន្មាន", "ស្អី", "មានន័យថា", "ជួយពន្យល់",
    "ជួយប្រាប់", "ជួយរក", "សុំសួរ", "សូមសួរ", "ចង់ដឹង", "ប្រាប់ខ្ញុំ",
    "ពន្យល់ពី", "ធ្វើម៉េច", "តើធ្វើដូចម្តេច", "មានវិធី",
)

_KHMER_QUESTION_ENDS = (
    "?", "？", "ទេ?", "មែនទេ?", "អត់?", "យ៉ាងណា?", "ដូចម្តេច?", "ដែរឬទេ?",
    "ឬទេ?", "មែនអត់?", "ឬនៅ?", "ហើយឬនៅ?", "មែនទេ", "ដែរទេ", "មែនអត់", "ឬនៅ", "ទេ",
)

_KHMER_GREETINGS = (
    "សួស្តី", "ជំរាបសួរ", "ជម្រាបសួរ", "សុខសប្បាយ", "អរគុណ", "អរគុណច្រើន",
)

# Negation words used in declarative sentences ending with 'ទេ' (e.g. មិន...ទេ, អត់...ទេ)
_KHMER_NEGATION_WORDS = (
    "មិន", "អត់", "ពុំ", "ឥត", "គ្មាន", "កុំ", "មិនដែល",
)

# English/Latin question & conversation patterns
_EN_QUESTION_STARTS = (
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "can you", "could you", "would you", "will you", "should i", "do you",
    "is it", "are you", "explain", "tell me", "define", "describe",
    "write", "generate", "summarize", "help me", "give me", "list",
    "create", "compare", "calculate", "how to", "what is", "what are",
)

_EN_GREETINGS = (
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "thanks", "thank you",
)

_AI_COMMAND_PREFIXES = ("/ask", "/ai", "/chat", "/question")
_TTS_COMMAND_PREFIXES = ("/tts", "/voice", "/say", "/speak", "/read", "/narrate")


def classify_text_intent(text: str, user_mode: str = "auto") -> Literal["ai_chat", "tts"]:
    """Classify user text as either an AI conversational query ('ai_chat') or direct speech ('tts')."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "tts"

    # Explicit symbol/prefix overrides
    if cleaned.startswith(("?", "？", "❓", "❔", "💬", "🤖")):
        return "ai_chat"
    if cleaned.startswith(("!", "🗣️", "📢", "🎙️", "🔊")):
        return "tts"

    # Explicit command overrides
    lower = cleaned.lower()
    if lower.startswith(_AI_COMMAND_PREFIXES):
        return "ai_chat"
    if lower.startswith(_TTS_COMMAND_PREFIXES):
        return "tts"

    # User-selected persistent mode overrides
    mode = (user_mode or "auto").strip().lower()
    if mode in ("ai_chat", "ai"):
        return "ai_chat"
    if mode == "tts":
        return "tts"

    # Mode is 'auto': smart multi-lingual intent detection
    if cleaned.endswith(("?", "？")):
        return "ai_chat"

    # Check Khmer Question Starts
    for prefix in _KHMER_QUESTION_STARTS:
        if cleaned.startswith(prefix):
            return "ai_chat"

    # Check Khmer Question Ends (Distinguishing question particles from declarative negations)
    has_negation = any(neg in cleaned for neg in _KHMER_NEGATION_WORDS)
    for suffix in _KHMER_QUESTION_ENDS:
        if cleaned.endswith(suffix):
            # If the sentence ends with 'ទេ' but contains a negation particle (e.g. ខ្ញុំមិនទៅទេ / អត់ដឹងទេ), it's a statement
            if suffix == "ទេ" and has_negation:
                continue
            return "ai_chat"

    # Check Khmer Greetings
    for greeting in _KHMER_GREETINGS:
        if cleaned.startswith(greeting):
            return "ai_chat"

    # Check English Question/Imperative Starts
    for q in _EN_QUESTION_STARTS:
        if lower.startswith(q + " ") or lower == q:
            return "ai_chat"

    # Check English Greetings
    for g in _EN_GREETINGS:
        if lower.startswith(g + " ") or lower == g or lower.startswith((g + "!", g + ".")):
            return "ai_chat"

    return "tts"


def classify_callback(
    data: str | None,
    *,
    speed_callbacks: Collection[str] = (),
) -> str | None:
    """Return the generic callback action for *data*, or ``None`` if unknown."""
    value = str(data or "").strip()
    if not value:
        return None

    exact_actions: dict[str, str] = {
        # Audio & Playback Controls
        "show_speed": "show_speed",
        "hide_speed": "hide_speed",
        "show_tts_model": "show_tts_model",
        "hide_tts_model": "hide_tts_model",
        "show_mode": "show_mode",
        "hide_mode": "hide_mode",
        "show_bot_mode": "show_mode",
        "hide_bot_mode": "hide_mode",
        "mode_auto": "mode_change",
        "mode_tts": "mode_change",
        "mode_ai": "mode_change",
        "tg_female": "gender",
        "tg_male": "gender",
        # Navigation & Help
        "welcome_profile": "welcome_profile",
        "welcome_back": "welcome_back",
        "welcome_menu": "welcome_menu",
        "welcome_close": "welcome_back",
        "welcome_help": "help",
        "close": "welcome_back",
        "close_msg": "delete",
        "help": "help",
        "btn_help": "help",
        # Guides & Hubs
        "show_tiktok_guide": "show_tiktok_guide",
        "show_facebook_guide": "show_facebook_guide",
        "show_instagram_guide": "show_instagram_guide",
        "show_youtube_guide": "show_youtube_guide",
        "show_media_hub": "show_media_hub",
        "system_status": "system_status",
        "btn_system_status": "system_status",
        "noop": "admin",
        # Donation exact callbacks
        "donate_menu": "donation",
        "donate_halloffame": "donation",
    }

    if value in exact_actions:
        return exact_actions[value]
    if value in speed_callbacks:
        return "speed"

    prefix_actions: tuple[tuple[str, str], ...] = (
        # Mode & Preferences
        ("mode_", "mode_change"),
        ("ttsmodel_", "tts_model"),
        ("set_tts_model:", "tts_model"),
        ("set_speed:", "speed"),
        ("set_gender:", "gender"),
        # Transcripts & Documents
        ("tts_transcript:", "tts_transcript"),
        ("del_transcript:", "delete"),
        ("doc_del:", "delete"),
        ("audio_del:", "delete"),
        ("doc_read:", "doc_read"),
        ("doc_trans:", "doc_trans"),
        ("audio_tts:", "audio_tts"),
        # Administration
        ("needs_", "needs_admin"),
        ("api_", "api_admin"),
        ("rtadmin_", "admin"),
        ("admin_", "admin"),
        ("cfg_cat:", "admin"),
        ("cfg_set:", "admin"),
        ("cfg_", "admin"),
        ("sched_", "sched"),
        ("bc_", "broadcast"),
        ("users_", "users"),
        ("user_", "users"),
        ("history_", "users"),
        # Donations & Bakong
        ("donate_", "donation"),
        ("khqr_", "donation"),
        ("checkpay", "donation"),
        # Articles & News
        ("art_", "article"),
        # Media & Downloaders
        ("tt_", "tiktok"),
        ("fb_", "facebook"),
        ("ig_", "instagram"),
        ("yt_", "youtube"),
        # Dynamic Menu Actions
        ("action_", "menu_action"),
        ("guide_", "menu_guide"),
    )

    for prefix, action in prefix_actions:
        if value.startswith(prefix):
            return action

    return None


def callback_requires_tts_access(action: str, data: str | None = None) -> bool:
    """Return whether a generic callback changes or generates TTS state."""
    if action in {
        "speed",
        "gender",
        "tts_model",
        "tts_transcript",
        "doc_read",
        "doc_trans",
        "audio_tts",
    }:
        return True

    # Check article audio narration requests
    if action == "article" and data and "art_voice" in data:
        return True

    return False


def is_message_not_modified_error(exc: Exception | str) -> bool:
    """Return True if Telegram error indicates message was already identical."""
    return "message is not modified" in str(exc).lower()


def is_stale_telegram_message_error(exc: Exception | str) -> bool:
    """Return True for harmless Telegram edit/delete races on stale messages."""
    msg = str(exc).lower()
    return any(token in msg for token in (
        "message to edit not found",
        "message to delete not found",
        "message can't be deleted",
        "message can't be edited",
        "message is not found",
        "message_id_invalid",
        "message not found",
        "message to be replied not found",
        "reply message not found",
        "there is no text in the message to edit",
        "message is too old",
        "message to be edited not found",
        "query is too old",
        "query_id_invalid",
    ))


def is_nonfatal_telegram_edit_error(exc: Exception | str) -> bool:
    """Return True if Telegram edit error is harmless and non-fatal."""
    return is_message_not_modified_error(exc) or is_stale_telegram_message_error(exc)


async def safe_delete_message(
    message: Any = None,
    *,
    bot: Any = None,
    chat_id: int | str | None = None,
    message_id: int | None = None,
) -> bool:
    """Safely delete a Telegram message or callback query, suppressing harmless stale errors."""
    try:
        # 1. Direct message.delete()
        if message is not None:
            if hasattr(message, "delete") and callable(message.delete):
                res = message.delete()
                if asyncio.iscoroutine(res):
                    await res
                return True

            # 2. CallbackQuery wrapper: query.message.delete()
            sub_msg = getattr(message, "message", None)
            if sub_msg is not None and hasattr(sub_msg, "delete") and callable(sub_msg.delete):
                res = sub_msg.delete()
                if asyncio.iscoroutine(res):
                    await res
                return True

        # 3. Direct bot.delete_message()
        if bot is not None and chat_id is not None and message_id is not None:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
            return True

    except Exception as exc:
        if not is_stale_telegram_message_error(exc):
            logger.debug("safe_delete_message suppressed error: %s", exc)

    return False


def safe_delete_later(message: Any, delay_s: float = 3.5) -> asyncio.Task | None:
    """Schedule non-blocking delayed deletion of a temporary Telegram message."""
    if not message:
        return None

    async def _runner():
        try:
            await asyncio.sleep(max(0.0, float(delay_s)))
            await safe_delete_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("safe_delete_later suppressed error: %s", exc)

    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(_runner(), name="safe-delete-later")
    except RuntimeError:
        return None


async def safe_edit_text(message: Any, text: str, **kwargs: Any) -> bool:
    """Safely edit Telegram message text, suppressing non-fatal stale or identical errors."""
    if not message:
        return False
    edit_fn = getattr(message, "edit_text", None) or getattr(message, "edit_message_text", None)
    if not callable(edit_fn):
        return False
    try:
        res = edit_fn(text, **kwargs)
        if asyncio.iscoroutine(res):
            await res
        return True
    except Exception as exc:
        if not is_nonfatal_telegram_edit_error(exc):
            logger.warning("safe_edit_text failed: %s", exc)
        return False


__all__ = [
    "callback_requires_tts_access",
    "classify_callback",
    "classify_text_intent",
    "is_message_not_modified_error",
    "is_nonfatal_telegram_edit_error",
    "is_stale_telegram_message_error",
    "safe_delete_later",
    "safe_delete_message",
    "safe_edit_text",
]