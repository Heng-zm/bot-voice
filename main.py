import logging
import os
import re
import asyncio
import threading
import time
import functools
import requests
import imageio_ffmpeg as _iio_ffmpeg
from collections import OrderedDict
from flask import Flask
from supabase import create_client, Client

# ── Flask (Render health check + keep-alive target) ────────────────────────
app_flask = Flask(__name__)

@app_flask.route('/')
def health_check():
    return "Bot is running!", 200

@app_flask.route('/ping')
def ping():
    return "pong", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host='0.0.0.0', port=port, threaded=True)

def keep_alive():
    """Self-ping every 4 min as secondary keep-alive layer."""
    logger_ka = logging.getLogger("keep_alive")
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not render_url:
        logger_ka.warning("RENDER_EXTERNAL_URL not set — self-ping disabled. "
                          "Use UptimeRobot to ping /ping every 5 min.")
        return
    time.sleep(20)
    headers = {"User-Agent": "Mozilla/5.0 (KeepAlive/1.0)"}
    while True:
        try:
            r = requests.get(f"{render_url}/ping", headers=headers, timeout=10)
            logger_ka.info(f"Keep-alive → {r.status_code}")
        except Exception as e:
            logger_ka.warning(f"Keep-alive failed: {e}")
        time.sleep(240)

# ── FFmpeg ─────────────────────────────────────────────────────────────────
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
# Config
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

VOICE_MAP = {
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    "en": {"female": "en-US-AriaNeural",     "male": "en-US-GuyNeural"},
}

SPEED_OPTIONS = {
    "spd_0.5": ("x0.5",   0.5),
    "spd_1.0": ("Normal", 1.0),
    "spd_1.5": ("x1.5",   1.5),
    "spd_2.0": ("x2.0",   2.0),
}

WELCOME_TEXT = (
    "🎵 សួស្តី! ខ្ញុំជា Bot បំលែងអក្សរទៅជាសំឡេង\n\n"
    "📌 វាយអក្សរភាសាណាមួយ ផ្ញើរមក Bot នឹងបំលែងដោយស្វ័យប្រវត្តិ!\n\n"
    "🌍 ភាសាដែល Support:\n"
    "🇰🇭 ភាសាខ្មែរ | 🇺🇸 English\n\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)

# ---------------------------------------------------------------------------
# In-memory LRU cache for user prefs (avoids a DB round-trip on every message)
# ---------------------------------------------------------------------------
_PREFS_CACHE: OrderedDict[int, dict] = OrderedDict()
_PREFS_CACHE_MAX = 500  # keep last 500 users in memory

def _cache_get(user_id: int) -> dict | None:
    if user_id in _PREFS_CACHE:
        _PREFS_CACHE.move_to_end(user_id)
        return _PREFS_CACHE[user_id].copy()
    return None

def _cache_set(user_id: int, prefs: dict):
    _PREFS_CACHE[user_id] = prefs.copy()
    _PREFS_CACHE.move_to_end(user_id)
    if len(_PREFS_CACHE) > _PREFS_CACHE_MAX:
        _PREFS_CACHE.popitem(last=False)

def _cache_invalidate(user_id: int):
    _PREFS_CACHE.pop(user_id, None)

# ---------------------------------------------------------------------------
# Per-user lock — prevents duplicate TTS jobs from double-taps
# ---------------------------------------------------------------------------
_USER_LOCKS: dict[int, asyncio.Lock] = {}

def _get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _USER_LOCKS:
        _USER_LOCKS[user_id] = asyncio.Lock()
    return _USER_LOCKS[user_id]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def sync_user_data(user):
    """Fire-and-forget upsert run in a thread so it never blocks handlers."""
    def _run():
        try:
            supabase.table("user_prefs").upsert(
                {"user_id": user.id, "username": user.username or user.first_name},
                on_conflict="user_id",
            ).execute()
        except Exception as e:
            logger.error(f"DB Sync Error: {e}")
    threading.Thread(target=_run, daemon=True).start()


def get_user_prefs(user_id: int) -> dict:
    """Return cached prefs, or fetch from DB (with speed-column fallback)."""
    cached = _cache_get(user_id)
    if cached:
        return cached

    defaults = {"gender": "female", "speed": 1.0}
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
            _cache_set(user_id, defaults)
            return defaults
        except Exception as e:
            if "speed" in str(e) and "does not exist" in str(e):
                logger.warning("speed column missing — using default 1.0")
                continue
            logger.error(f"Error fetching prefs: {e}")
            break

    _cache_set(user_id, defaults)
    return defaults


def update_user_gender(user_id: int, gender: str):
    _cache_invalidate(user_id)
    def _run():
        try:
            supabase.table("user_prefs").update({"gender": gender}).eq("user_id", user_id).execute()
        except Exception as e:
            logger.error(f"Error updating gender: {e}")
    threading.Thread(target=_run, daemon=True).start()


def update_user_speed(user_id: int, speed: float):
    _cache_invalidate(user_id)
    def _run():
        try:
            supabase.table("user_prefs").update({"speed": speed}).eq("user_id", user_id).execute()
        except Exception as e:
            if "does not exist" in str(e):
                logger.warning("Cannot save speed — run: ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;")
            else:
                logger.error(f"Error updating speed: {e}")
    threading.Thread(target=_run, daemon=True).start()


def ensure_speed_column():
    try:
        supabase.table("user_prefs").select("speed").limit(1).execute()
        logger.info("speed column exists ✓")
    except Exception as e:
        if "does not exist" in str(e):
            logger.warning(
                "⚠️  speed column missing. Run in Supabase SQL editor:\n"
                "    ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
            )


def save_text_cache(msg_id: int, text: str, user_id: int = None, username: str = None):
    """Async fire-and-forget cache save."""
    def _run():
        try:
            payload = {"message_id": msg_id, "original_text": text}
            if user_id  is not None: payload["user_id"]  = user_id
            if username is not None: payload["username"] = username
            supabase.table("text_cache").upsert(payload).execute()
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    threading.Thread(target=_run, daemon=True).start()


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
# Audio — single-pass FFmpeg pipeline (MP3 → OGG + speed in one command)
# ---------------------------------------------------------------------------
def _build_atempo_chain(speed: float) -> str:
    stages = []
    r = speed
    if r < 1.0:
        while r < 0.5:
            stages.append("atempo=0.5"); r /= 0.5
        stages.append(f"atempo={r:.4f}")
    else:
        while r > 2.0:
            stages.append("atempo=2.0"); r /= 2.0
        stages.append(f"atempo={r:.4f}")
    return ",".join(stages)


async def generate_voice(text: str, gender: str, speed: float, output_path: str) -> str:
    """
    edge-tts → raw MP3 bytes piped directly into FFmpeg → final OGG.
    One FFmpeg process, no intermediate temp files when speed == 1.0.
    """
    is_khmer = bool(re.search(r"[\u1780-\u17FF]", text))
    lang_key  = "km" if is_khmer else "en"
    voice     = VOICE_MAP[lang_key][gender]
    label     = "@voicekhaibot" if is_khmer else "@voicekhaibot"

    # 1. Collect all MP3 chunks from edge-tts into memory (avoids disk write)
    mp3_chunks: list[bytes] = []
    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])
    mp3_data = b"".join(mp3_chunks)

    if not mp3_data:
        raise RuntimeError("edge-tts returned empty audio")

    # 2. Single FFmpeg pass: stdin MP3 → stdout OGG (+ optional atempo)
    af_filter = _build_atempo_chain(speed) if speed != 1.0 else None
    cmd = [_FFMPEG_EXE, "-y", "-f", "mp3", "-i", "pipe:0"]
    if af_filter:
        cmd += ["-filter:a", af_filter]
    cmd += ["-c:a", "libopus", "-b:a", "32k", output_path]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate(input=mp3_data)

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with code {proc.returncode}")

    return label

# ---------------------------------------------------------------------------
# Status timer
# ---------------------------------------------------------------------------
_STATUS_FRAMES = [
    "⏳ កំពុងបង្កើតសំឡេង ·",
    "⏳ កំពុងបង្កើតសំឡេង ··",
    "⏳ កំពុងបង្កើតសំឡេង ···",
    "⏳ កំពុងបង្កើតសំឡេង ····",
]

async def send_status_timer(chat_id: int, bot, stop_event: asyncio.Event):
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

# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------
def get_main_kb(gender: str) -> InlineKeyboardMarkup:
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
    speed_row = []
    for cb, (lbl, val) in SPEED_OPTIONS.items():
        mark = " ✅" if abs(val - current_speed) < 0.01 else ""
        speed_row.append(InlineKeyboardButton(lbl + mark, callback_data=cb))
    return InlineKeyboardMarkup([
        speed_row,
        [InlineKeyboardButton("🔙 ត្រឡប់", callback_data="hide_speed")],
    ])

# ---------------------------------------------------------------------------
# Shared regen helper (used by both gender + speed callbacks)
# ---------------------------------------------------------------------------
async def _regen_voice(
    chat_id: int,
    bot,
    original_msg,           # the message whose voice is being replaced
    original_text: str,
    gender: str,
    speed: float,
    user_id: int,
    username: str,
    file_prefix: str,
    msg_id: int,
) -> None:
    file_path = f"{file_prefix}_{user_id}_{msg_id}.ogg"
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, bot, stop_event))
    try:
        label = await generate_voice(original_text, gender, speed, file_path)
        stop_event.set()
        await timer_task
        try:
            await original_msg.delete()
        except Exception:
            pass
        with open(file_path, "rb") as audio:
            new_msg = await bot.send_voice(
                chat_id=chat_id,
                voice=audio,
                caption=f"🗣️ {label}",
                reply_markup=get_main_kb(gender),
            )
        save_text_cache(new_msg.message_id, original_text, user_id=user_id, username=username)
    except Exception as e:
        logger.error(f"Regen error: {e}")
        stop_event.set()
        await timer_task
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass

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
    msg  = update.message
    text = msg.text
    if not text:
        return
    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    user   = update.effective_user
    user_id = user.id

    # Fire-and-forget user sync (non-blocking)
    sync_user_data(user)

    # Per-user lock: ignore if already generating for this user
    lock = _get_user_lock(user_id)
    if lock.locked():
        await msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ...")
        return

    async with lock:
        prefs  = get_user_prefs(user_id)
        gender = prefs["gender"]
        speed  = prefs["speed"]

        file_path  = f"v_{user_id}_{msg.message_id}.ogg"
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
            save_text_cache(
                sent_msg.message_id, text,
                user_id=user_id,
                username=user.username or user.first_name,
            )
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            stop_event.set()
            await timer_task
            await msg.reply_text("❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។ សូមព្យាយាមម្តងទៀត។")
        finally:
            try:
                os.remove(file_path)
            except OSError:
                pass


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    msg_id  = query.message.message_id
    data    = query.data
    bot     = context.bot

    # ── Show / hide speed sub-menu (no TTS, just keyboard swap) ────────
    if data == "show_speed":
        await query.answer()
        prefs = get_user_prefs(user_id)
        try:
            await query.message.edit_reply_markup(reply_markup=get_speed_kb(prefs["speed"]))
        except Exception as e:
            logger.error(f"Markup error: {e}")
        return

    if data == "hide_speed":
        await query.answer()
        prefs = get_user_prefs(user_id)
        try:
            await query.message.edit_reply_markup(reply_markup=get_main_kb(prefs["gender"]))
        except Exception as e:
            logger.error(f"Markup error: {e}")
        return

    # ── Speed or gender change — both use shared _regen_voice ───────────
    if data in SPEED_OPTIONS or data in ("tg_female", "tg_male"):
        original_text = get_text_cache(msg_id)
        if not original_text:
            await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
            return

        # Per-user lock: ignore double-taps
        lock = _get_user_lock(user_id)
        if lock.locked():
            await query.answer("⏳ សូមរង់ចាំ...")
            return

        if data in SPEED_OPTIONS:
            _, new_speed = SPEED_OPTIONS[data]
            update_user_speed(user_id, new_speed)   # async, cache already cleared
            prefs  = get_user_prefs(user_id)
            gender = prefs["gender"]
            speed  = new_speed
            await query.answer("🔄 កំពុងប្តូរល្បឿន...")
            prefix = "spd"
        else:
            new_gender = data.replace("tg_", "")
            update_user_gender(user_id, new_gender)  # async, cache already cleared
            prefs  = get_user_prefs(user_id)
            gender = new_gender
            speed  = prefs["speed"]
            await query.answer("🔄 កំពុងប្តូរសំឡេង...")
            prefix = "rev"

        async with lock:
            await _regen_voice(
                chat_id=query.message.chat.id,
                bot=bot,
                original_msg=query.message,
                original_text=original_text,
                gender=gender,
                speed=speed,
                user_id=user_id,
                username=query.from_user.username or query.from_user.first_name,
                file_prefix=prefix,
                msg_id=msg_id,
            )
        return

    await query.answer()

# ---------------------------------------------------------------------------
# Main — infinite restart loop so the process never dies
# ---------------------------------------------------------------------------
def main():
    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Flask health-check server started.")

    threading.Thread(target=keep_alive, daemon=True).start()
    print("✅ Keep-alive thread started.")

    ensure_speed_column()

    print("🚀 Bot is starting...")
    while True:
        try:
            app = (
                Application.builder()
                .token(TELEGRAM_BOT_TOKEN)
                .connect_timeout(30)
                .read_timeout(30)
                .write_timeout(30)
                .pool_timeout(30)
                .build()
            )
            app.add_handler(CommandHandler("start", on_start))
            app.add_handler(CallbackQueryHandler(on_callback))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

            print("🟢 Bot polling started.")
            app.run_polling(
                allowed_updates=["message", "callback_query"],
                drop_pending_updates=True,
                close_loop=False,
            )
        except Exception as e:
            logger.error(f"💥 Bot crashed: {e} — restarting in 5 s...")
            time.sleep(5)
        else:
            logger.warning("⚠️  Polling stopped — restarting in 5 s...")
            time.sleep(5)


if __name__ == "__main__":
    main()
