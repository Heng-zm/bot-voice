import logging
import os
import re
import asyncio
import threading
import imageio_ffmpeg as _iio_ffmpeg
from flask import Flask
from supabase import create_client, Client

# ── Flask Web Server for Render Health Checks ──────────────────────────────
app_flask = Flask(__name__)

@app_flask.route('/')
def health_check():
    return "Bot is running!", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host='0.0.0.0', port=port)

# ── FFmpeg Configuration ───────────────────────────────────────────────────
_FFMPEG_EXE = _iio_ffmpeg.get_ffmpeg_exe()

import edge_tts
from dotenv import load_dotenv
from telegram import (
    Update, constants,
    InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, CallbackQueryHandler,
)

# ---------------------------------------------------------------------------
# Configuration & Database Init
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SB_URL, SB_KEY)

# edge-tts voice map for both languages
VOICE_MAP = {
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    "en": {"female": "en-US-AriaNeural",     "male": "en-US-GuyNeural"},
}

# Speed options: callback_data -> (label, atempo_value)
SPEED_OPTIONS = {
    "spd_0.5":  ("x0.5",   0.5),
    "spd_1.0":  ("Normal", 1.0),
    "spd_1.5":  ("x1.5",   1.5),
    "spd_2.0":  ("x2.0",   2.0),
}

WELCOME_TEXT = (
    "🎵 សួស្តី! ខ្ញុំជា Bot បំលែងអក្សរទៅជាសំឡេង\n\n"
    "📌 វាយអក្សរភាសាណាមួយ ផ្ញើរមក Bot នឹងបំលែងដោយស្វ័យប្រវត្តិ!\n\n"
    "🌍 ភាសាដែល Support:\n"
    "🇰🇭 ភាសាខ្មែរ | 🇺🇸 English\n\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)

# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------
def sync_user_data(user):
    try:
        supabase.table("user_prefs").upsert(
            {"user_id": user.id, "username": user.username or user.first_name},
            on_conflict="user_id",
        ).execute()
    except Exception as e:
        logger.error(f"DB Sync Error: {e}")


def get_user_prefs(user_id: int) -> dict:
    """Fetch gender and speed in one DB call. Falls back if speed column missing."""
    defaults = {"gender": "female", "speed": 1.0}
    # Try fetching both columns; fall back to gender-only if speed col missing
    for cols in ("gender, speed", "gender"):
        try:
            res = (
                supabase.table("user_prefs")
                .select(cols)
                .eq("user_id", user_id)
                .execute()
            )
            if res.data:
                row = res.data[0]
                defaults["gender"] = row.get("gender") or "female"
                if "speed" in row and row["speed"] is not None:
                    defaults["speed"] = float(row["speed"])
            return defaults
        except Exception as e:
            err_msg = str(e)
            if "speed" in err_msg and "does not exist" in err_msg:
                # Column not yet created — retry with gender only
                logger.warning("speed column missing in user_prefs — using default 1.0")
                continue
            logger.error(f"Error fetching prefs: {e}")
            break
    return defaults


def ensure_speed_column():
    """Try to add speed column if it doesn't exist (runs once at startup)."""
    try:
        supabase.table("user_prefs").select("speed").limit(1).execute()
        logger.info("speed column exists ✓")
    except Exception as e:
        if "does not exist" in str(e):
            logger.warning(
                "⚠️  speed column missing. Run this in Supabase SQL editor:\n"
                "    ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;\n"
                "Bot will use speed=1.0 for all users until then."
            )


def update_user_gender(user_id: int, gender: str):
    try:
        supabase.table("user_prefs").update({"gender": gender}).eq("user_id", user_id).execute()
    except Exception as e:
        logger.error(f"Error updating gender: {e}")


def update_user_speed(user_id: int, speed: float):
    try:
        supabase.table("user_prefs").update({"speed": speed}).eq("user_id", user_id).execute()
    except Exception as e:
        if "does not exist" in str(e):
            logger.warning("Cannot save speed — column missing. Run ALTER TABLE first.")
        else:
            logger.error(f"Error updating speed: {e}")


# ---------------------------------------------------------------------------
# Status Timer  (animated dots while TTS is being generated)
# ---------------------------------------------------------------------------
_STATUS_FRAMES = [
    "⏳ កំពុងបង្កើតសំឡេង ·",
    "⏳ កំពុងបង្កើតសំឡេង ··",
    "⏳ កំពុងបង្កើតសំឡេង ···",
    "⏳ កំពុងបង្កើតសំឡេង ····",
]

async def send_status_timer(chat_id: int, bot, stop_event: asyncio.Event):
    """Send an animated status message that updates every second until stopped."""
    msg = await bot.send_message(chat_id=chat_id, text=_STATUS_FRAMES[0])
    frame = 1
    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
            if stop_event.is_set():
                break
            try:
                await msg.edit_text(_STATUS_FRAMES[frame % len(_STATUS_FRAMES)])
            except Exception:
                pass
            frame += 1
    finally:
        try:
            await msg.delete()
        except Exception:
            pass


def save_text_cache(msg_id: int, text: str):
    try:
        supabase.table("text_cache").upsert(
            {"message_id": msg_id, "original_text": text}
        ).execute()
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def get_text_cache(msg_id: int) -> str | None:
    try:
        res = (
            supabase.table("text_cache")
            .select("original_text")
            .eq("message_id", msg_id)
            .execute()
        )
        if res.data:
            return res.data[0]["original_text"]
    except Exception as e:
        logger.error(f"Error fetching cache: {e}")
    return None

# ---------------------------------------------------------------------------
# Audio Helpers
# ---------------------------------------------------------------------------
async def _run_ffmpeg(*args):
    """Run an ffmpeg command asynchronously and wait for it to finish."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()
    return process.returncode


async def convert_mp3_to_ogg(mp3_path: str, ogg_path: str):
    """Convert an MP3/any audio file to Opus OGG."""
    await _run_ffmpeg(
        _FFMPEG_EXE, "-y", "-i", mp3_path,
        "-c:a", "libopus", "-b:a", "32k", ogg_path,
    )


async def apply_speed_to_ogg(input_ogg: str, output_ogg: str, speed: float):
    """Apply atempo speed filter. atempo supports 0.5–2.0; chain for extremes."""
    if speed == 1.0:
        # No processing needed — just copy
        await _run_ffmpeg(
            _FFMPEG_EXE, "-y", "-i", input_ogg,
            "-c:a", "libopus", "-b:a", "32k", output_ogg,
        )
        return

    # Build chained atempo filters if needed (each stage: 0.5–2.0)
    filters_chain = _build_atempo_chain(speed)
    await _run_ffmpeg(
        _FFMPEG_EXE, "-y", "-i", input_ogg,
        "-filter:a", filters_chain,
        "-c:a", "libopus", "-b:a", "32k", output_ogg,
    )


def _build_atempo_chain(speed: float) -> str:
    """Return a comma-joined atempo filter string safe for ffmpeg."""
    # atempo is limited to [0.5, 2.0] per stage
    stages = []
    remaining = speed
    if remaining < 1.0:
        while remaining < 0.5:
            stages.append("atempo=0.5")
            remaining /= 0.5
        stages.append(f"atempo={remaining:.4f}")
    else:
        while remaining > 2.0:
            stages.append("atempo=2.0")
            remaining /= 2.0
        stages.append(f"atempo={remaining:.4f}")
    return ",".join(stages)


async def generate_voice(text: str, gender: str, speed: float, output_path: str) -> str:
    """Generate TTS audio using edge-tts, then apply speed adjustment."""
    is_khmer = bool(re.search(r"[\u1780-\u17FF]", text))
    lang_key = "km" if is_khmer else "en"
    voice = VOICE_MAP[lang_key][gender]
    label = "🇰🇭 ភាសាខ្មែរ" if is_khmer else "🇺🇸 English"

    tmp_mp3 = f"{output_path}.raw.mp3"
    tmp_ogg = f"{output_path}.base.ogg"
    try:
        # Step 1: edge-tts → MP3
        await edge_tts.Communicate(text, voice).save(tmp_mp3)
        # Step 2: MP3 → OGG (base, no speed)
        await convert_mp3_to_ogg(tmp_mp3, tmp_ogg)
        # Step 3: Apply speed → final OGG
        await apply_speed_to_ogg(tmp_ogg, output_path, speed)
    finally:
        for f in (tmp_mp3, tmp_ogg):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    return label

# ---------------------------------------------------------------------------
# Keyboard Builders
# ---------------------------------------------------------------------------
def get_main_kb(gender: str) -> InlineKeyboardMarkup:
    """Main keyboard: gender buttons + speed toggle button."""
    f_btn = "👩 សំឡេងស្រី" + (" ✅" if gender == "female" else "")
    m_btn = "👨 សំឡេងប្រុស" + (" ✅" if gender == "male" else "")
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f_btn, callback_data="tg_female"),
            InlineKeyboardButton(m_btn, callback_data="tg_male"),
        ],
        [InlineKeyboardButton("🎚️ ល្បឿនសំឡេង", callback_data="show_speed")],
    ])


def get_speed_kb(current_speed: float) -> InlineKeyboardMarkup:
    """Speed selection keyboard (replaces main kb after clicking ល្បឿនសំឡេង)."""
    rows = []
    speed_row = []
    for cb, (lbl, val) in SPEED_OPTIONS.items():
        mark = " ✅" if abs(val - current_speed) < 0.01 else ""
        speed_row.append(InlineKeyboardButton(lbl + mark, callback_data=cb))
    rows.append(speed_row)
    rows.append([InlineKeyboardButton("🔙 ត្រឡប់", callback_data="hide_speed")])
    return InlineKeyboardMarkup(rows)

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sync_user_data(update.effective_user)
    await update.message.reply_text(
        WELCOME_TEXT,
        reply_markup=ReplyKeyboardRemove(),
        disable_web_page_preview=True,
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    text = msg.text
    if not text:
        return
    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    sync_user_data(update.effective_user)
    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    gender = prefs["gender"]
    speed  = prefs["speed"]

    file_path = f"v_{user_id}_{msg.message_id}.ogg"
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(update.effective_chat.id, context.bot, stop_event)
    )
    try:
        label = await generate_voice(text, gender, speed, file_path)
        stop_event.set()
        await timer_task
        with open(file_path, "rb") as audio:
            sent_msg = await msg.reply_voice(
                voice=audio,
                caption=f"🗣️ {label}",
                reply_markup=get_main_kb(gender),
            )
        save_text_cache(sent_msg.message_id, text)
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        stop_event.set()
        await timer_task
        await msg.reply_text("❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។ សូមព្យាយាមម្តងទៀត។")
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    msg_id  = query.message.message_id
    data    = query.data

    # ── Show speed sub-menu ─────────────────────────────────────────────
    if data == "show_speed":
        await query.answer()
        prefs = get_user_prefs(user_id)
        try:
            await query.message.edit_reply_markup(
                reply_markup=get_speed_kb(prefs["speed"])
            )
        except Exception as e:
            logger.error(f"Edit markup error: {e}")
        return

    # ── Hide speed sub-menu (back button) ───────────────────────────────
    if data == "hide_speed":
        await query.answer()
        prefs = get_user_prefs(user_id)
        try:
            await query.message.edit_reply_markup(
                reply_markup=get_main_kb(prefs["gender"])
            )
        except Exception as e:
            logger.error(f"Edit markup error: {e}")
        return

    # ── Speed selected ──────────────────────────────────────────────────
    if data in SPEED_OPTIONS:
        _, new_speed = SPEED_OPTIONS[data]
        update_user_speed(user_id, new_speed)

        original_text = get_text_cache(msg_id)
        if not original_text:
            await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
            return

        await query.answer("🔄 កំពុងប្តូរល្បឿន...")

        prefs  = get_user_prefs(user_id)
        gender = prefs["gender"]
        file_path = f"spd_{user_id}_{msg_id}.ogg"
        stop_event = asyncio.Event()
        timer_task = asyncio.create_task(
            send_status_timer(query.message.chat.id, query.get_bot(), stop_event)
        )
        try:
            label = await generate_voice(original_text, gender, new_speed, file_path)
            stop_event.set()
            await timer_task
            try:
                await query.message.delete()
            except Exception:
                pass
            with open(file_path, "rb") as audio:
                new_msg = await query.message.chat.send_voice(
                    voice=audio,
                    caption=f"🗣️ {label}",
                    reply_markup=get_main_kb(gender),
                )
            save_text_cache(new_msg.message_id, original_text)
        except Exception as e:
            logger.error(f"Speed regen error: {e}")
            stop_event.set()
            await timer_task
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        return

    # ── Gender selected ─────────────────────────────────────────────────
    if data in ("tg_female", "tg_male"):
        new_gender = data.replace("tg_", "")
        original_text = get_text_cache(msg_id)
        if not original_text:
            await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
            return

        update_user_gender(user_id, new_gender)
        await query.answer("🔄 កំពុងប្តូរសំឡេង...")

        prefs = get_user_prefs(user_id)
        speed = prefs["speed"]
        file_path = f"rev_{user_id}_{msg_id}.ogg"
        stop_event = asyncio.Event()
        timer_task = asyncio.create_task(
            send_status_timer(query.message.chat.id, query.get_bot(), stop_event)
        )
        try:
            label = await generate_voice(original_text, new_gender, speed, file_path)
            stop_event.set()
            await timer_task
            try:
                await query.message.delete()
            except Exception:
                pass
            with open(file_path, "rb") as audio:
                new_msg = await query.message.chat.send_voice(
                    voice=audio,
                    caption=f"🗣️ {label}",
                    reply_markup=get_main_kb(new_gender),
                )
            save_text_cache(new_msg.message_id, original_text)
        except Exception as e:
            logger.error(f"Gender regen error: {e}")
            stop_event.set()
            await timer_task
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        return

    # Unknown callback
    await query.answer()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Start Flask health-check server in background
    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Health check server started.")

    # Check DB schema
    ensure_speed_column()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print("🚀 Bot is running and waiting for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()
