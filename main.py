import logging
import os
import re
import asyncio
import threading
import time
import base64
import requests
import imageio_ffmpeg as _iio_ffmpeg
from google import genai
from google.genai import types as genai_types
from flask import Flask
from supabase import create_client, Client

# ── Flask Web Server ───────────────────────────────────────────────────────
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
        logger_ka.warning("RENDER_EXTERNAL_URL not set — self-ping disabled.")
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
    Update,
    InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.error import (
    NetworkError, TimedOut, RetryAfter,
    BadRequest, TelegramError,
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
# Silence noisy telegram library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SB_URL  = os.getenv("SUPABASE_URL")
SB_KEY  = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Safe init — bot keeps running even if a service is misconfigured
try:
    supabase: Client = create_client(SB_URL, SB_KEY)
except Exception as e:
    logger.error(f"Supabase init failed: {e}")
    supabase = None

try:
    _gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception as e:
    logger.error(f"Gemini init failed: {e}")
    _gemini = None

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

BOT_TAG = "@voicekhaibot"

# ---------------------------------------------------------------------------
# Per-user lock — prevents duplicate TTS jobs from double-taps
# ---------------------------------------------------------------------------
_USER_LOCKS: dict[int, asyncio.Lock] = {}

def _get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _USER_LOCKS:
        _USER_LOCKS[user_id] = asyncio.Lock()
    return _USER_LOCKS[user_id]

# ---------------------------------------------------------------------------
# Safe Telegram send helpers — auto-retry on network errors
# ---------------------------------------------------------------------------
async def safe_send(coro, retries: int = 3, delay: float = 2.0):
    """Wrap any Telegram API call with retry logic."""
    for attempt in range(retries):
        try:
            return await coro
        except RetryAfter as e:
            wait = e.retry_after + 1
            logger.warning(f"Rate limited — waiting {wait}s")
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError) as e:
            if attempt < retries - 1:
                logger.warning(f"Network error (attempt {attempt+1}): {e} — retrying in {delay}s")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Network error after {retries} attempts: {e}")
                raise
        except BadRequest as e:
            logger.error(f"Bad request (not retrying): {e}")
            raise
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            raise
    return None

# ---------------------------------------------------------------------------
# Database Helpers — all wrapped, never crash the bot
# ---------------------------------------------------------------------------
def _db(fn_name: str):
    """Decorator: catch all DB exceptions and log them."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"DB [{fn_name}] error: {e}")
                return None
        return wrapper
    return decorator


def sync_user_data(user):
    """Fire-and-forget upsert in background thread."""
    if not supabase:
        return
    def _run():
        try:
            supabase.table("user_prefs").upsert(
                {"user_id": user.id, "username": user.username or user.first_name},
                on_conflict="user_id",
            ).execute()
        except Exception as e:
            logger.error(f"DB sync_user_data: {e}")
    threading.Thread(target=_run, daemon=True).start()


def get_user_prefs(user_id: int) -> dict:
    defaults = {"gender": "female", "speed": 1.0}
    if not supabase:
        return defaults
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
            if "speed" in str(e) and "does not exist" in str(e):
                logger.warning("speed column missing — using default 1.0")
                continue
            logger.error(f"DB get_user_prefs: {e}")
            break
    return defaults


def update_user_gender(user_id: int, gender: str):
    if not supabase:
        return
    def _run():
        try:
            supabase.table("user_prefs").update({"gender": gender}).eq("user_id", user_id).execute()
        except Exception as e:
            logger.error(f"DB update_user_gender: {e}")
    threading.Thread(target=_run, daemon=True).start()


def update_user_speed(user_id: int, speed: float):
    if not supabase:
        return
    def _run():
        try:
            supabase.table("user_prefs").update({"speed": speed}).eq("user_id", user_id).execute()
        except Exception as e:
            if "does not exist" in str(e):
                logger.warning("speed column missing — run: ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;")
            else:
                logger.error(f"DB update_user_speed: {e}")
    threading.Thread(target=_run, daemon=True).start()


def save_text_cache(msg_id: int, text: str, user_id: int = None, username: str = None):
    if not supabase:
        return
    def _run():
        try:
            payload = {"message_id": msg_id, "original_text": text}
            if user_id  is not None: payload["user_id"]  = user_id
            if username is not None: payload["username"] = username
            supabase.table("text_cache").upsert(payload).execute()
        except Exception as e:
            logger.error(f"DB save_text_cache: {e}")
    threading.Thread(target=_run, daemon=True).start()


def get_text_cache(msg_id: int) -> str | None:
    if not supabase:
        return None
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
        logger.error(f"DB get_text_cache: {e}")
    return None


def ensure_speed_column():
    if not supabase:
        return
    try:
        supabase.table("user_prefs").select("speed").limit(1).execute()
        logger.info("speed column exists ✓")
    except Exception as e:
        if "does not exist" in str(e):
            logger.warning(
                "⚠️  speed column missing. Run in Supabase SQL editor:\n"
                "    ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
            )

# ---------------------------------------------------------------------------
# Status Timer
# ---------------------------------------------------------------------------
_STATUS_FRAMES = [
    "⏳ កំពុងបង្កើតសំឡេង ·",
    "⏳ កំពុងបង្កើតសំឡេង ··",
    "⏳ កំពុងបង្កើតសំឡេង ···",
    "⏳ កំពុងបង្កើតសំឡេង ····",
]

_TRANSCRIBE_FRAMES = [
    "🎙️ កំពុង Transcribe ·",
    "🎙️ កំពុង Transcribe ··",
    "🎙️ កំពុង Transcribe ···",
    "🎙️ កំពុង Transcribe ····",
]

async def send_status_timer(chat_id: int, bot, stop_event: asyncio.Event, frames=None):
    frames = frames or _STATUS_FRAMES
    msg = None
    try:
        msg = await safe_send(bot.send_message(chat_id=chat_id, text=frames[0]))
        frame = 1
        while not stop_event.is_set():
            await asyncio.sleep(1)
            if stop_event.is_set():
                break
            try:
                await safe_send(msg.edit_text(frames[frame % len(frames)]))
            except Exception:
                pass
            frame += 1
    except Exception as e:
        logger.warning(f"Status timer error: {e}")
    finally:
        if msg:
            try:
                await msg.delete()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Audio — FFmpeg pipeline
# ---------------------------------------------------------------------------
def _build_atempo_chain(speed: float) -> str:
    stages, r = [], speed
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
    """edge-tts → MP3 chunks in memory → single FFmpeg pass → OGG."""
    is_khmer = bool(re.search(r"[\u1780-\u17FF]", text))
    lang_key = "km" if is_khmer else "en"
    voice    = VOICE_MAP[lang_key][gender]
    label    = BOT_TAG

    # Stream edge-tts into memory
    mp3_chunks: list[bytes] = []
    try:
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])
    except Exception as e:
        raise RuntimeError(f"edge-tts failed: {e}")

    mp3_data = b"".join(mp3_chunks)
    if not mp3_data:
        raise RuntimeError("edge-tts returned empty audio")

    # Single FFmpeg pass: stdin MP3 → OGG + optional speed
    af = _build_atempo_chain(speed) if speed != 1.0 else None
    cmd = [_FFMPEG_EXE, "-y", "-f", "mp3", "-i", "pipe:0"]
    if af:
        cmd += ["-filter:a", af]
    cmd += ["-c:a", "libopus", "-b:a", "32k", output_path]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate(input=mp3_data)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode})")

    return label

# ---------------------------------------------------------------------------
# Gemini transcription
# ---------------------------------------------------------------------------
async def transcribe_voice(ogg_path: str) -> str:
    """Send OGG bytes to Gemini Flash for transcription."""
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")

    with open(ogg_path, "rb") as f:
        audio_bytes = f.read()

    prompt = (
        "Transcribe this audio exactly as spoken. "
        "Output ONLY the transcribed text — no labels, no explanation. "
        "Support both Khmer (ភាសាខ្មែរ) and English."
    )

    def _call():
        return _gemini.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                prompt,
            ],
        )

    loop = asyncio.get_event_loop()
    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(None, _call),
            timeout=60,
        )
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini transcription timed out after 60s")

# ---------------------------------------------------------------------------
# Keyboard Builders
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

def get_transcription_kb(transcript_msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📢 AI អាន", callback_data=f"tts_transcript:{transcript_msg_id}"),
        InlineKeyboardButton("🗑️ លុប",    callback_data=f"del_transcript:{transcript_msg_id}"),
    ]])

# ---------------------------------------------------------------------------
# Safe file cleanup
# ---------------------------------------------------------------------------
def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sync_user_data(update.effective_user)
        await safe_send(update.message.reply_text(
            WELCOME_TEXT,
            reply_markup=ReplyKeyboardRemove(),
            disable_web_page_preview=True,
        ))
    except Exception as e:
        logger.error(f"on_start error: {e}")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg  = update.message
    text = msg.text
    if not text:
        return
    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    user    = update.effective_user
    user_id = user.id

    lock = _get_user_lock(user_id)
    if lock.locked():
        try:
            await safe_send(msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        except Exception:
            pass
        return

    sync_user_data(user)
    prefs  = get_user_prefs(user_id)
    gender = prefs["gender"]
    speed  = prefs["speed"]

    file_path  = f"v_{user_id}_{msg.message_id}.ogg"
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(update.effective_chat.id, context.bot, stop_event)
    )

    async with lock:
        try:
            label = await generate_voice(text, gender, speed, file_path)
            stop_event.set()
            await timer_task
            with open(file_path, "rb") as audio:
                sent_msg = await safe_send(msg.reply_voice(
                    voice=audio,
                    caption=f"🗣️ {label}",
                    reply_markup=get_main_kb(gender),
                ))
            if sent_msg:
                save_text_cache(
                    sent_msg.message_id, text,
                    user_id=user_id,
                    username=user.username or user.first_name,
                )
        except Exception as e:
            logger.error(f"on_text TTS error: {e}")
            stop_event.set()
            await timer_task
            try:
                await safe_send(msg.reply_text("❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។ សូមព្យាយាមម្តងទៀត។"))
            except Exception:
                pass
        finally:
            _cleanup(file_path)


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Transcribe incoming voice message using Gemini."""
    msg     = update.message
    user    = update.effective_user
    user_id = user.id

    if not _gemini:
        try:
            await safe_send(msg.reply_text(
                "❌ Gemini API មិន Activate ទេ។\n"
                "សូម Set GEMINI_API_KEY ក្នុង Environment Variables។"
            ))
        except Exception:
            pass
        return

    sync_user_data(user)

    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(
            msg.chat_id, context.bot, stop_event,
            frames=_TRANSCRIBE_FRAMES,
        )
    )

    ogg_path = f"voice_in_{user_id}_{msg.message_id}.ogg"
    try:
        voice_file = await safe_send(context.bot.get_file(msg.voice.file_id))
        if not voice_file:
            raise RuntimeError("Could not get voice file from Telegram")
        await voice_file.download_to_drive(ogg_path)

        transcript = await transcribe_voice(ogg_path)

        stop_event.set()
        await timer_task

        if not transcript:
            await safe_send(msg.reply_text("❌ រក Transcript មិនឃើញ។ សូមព្យាយាមម្តងទៀត។"))
            return

        is_khmer  = bool(re.search(r"[\u1780-\u17FF]", transcript))
        lang_flag = "🇰🇭" if is_khmer else "🇺🇸"

        safe_transcript = transcript.replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")

        reply = await safe_send(msg.reply_text(
            f"📝 *Transcript* {lang_flag}\n\n{safe_transcript}",
            parse_mode="Markdown",
            reply_markup=get_transcription_kb(0),
        ))

        if reply:
            await safe_send(reply.edit_reply_markup(
                reply_markup=get_transcription_kb(reply.message_id)
            ))
            save_text_cache(
                reply.message_id, transcript,
                user_id=user_id,
                username=user.username or user.first_name,
            )

    except Exception as e:
        logger.error(f"on_voice error: {e}")
        stop_event.set()
        await timer_task
        try:
            await safe_send(msg.reply_text("❌ មានបញ្ហាក្នុងការ Transcribe។ សូមព្យាយាមម្តងទៀត។"))
        except Exception:
            pass
    finally:
        _cleanup(ogg_path)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    msg_id  = query.message.message_id
    data    = query.data

    try:
        await query.answer()
    except Exception:
        pass

    try:
        # ── Show / hide speed keyboard ───────────────────────────────────
        if data == "show_speed":
            prefs = get_user_prefs(user_id)
            await safe_send(query.message.edit_reply_markup(
                reply_markup=get_speed_kb(prefs["speed"])
            ))
            return

        if data == "hide_speed":
            prefs = get_user_prefs(user_id)
            await safe_send(query.message.edit_reply_markup(
                reply_markup=get_main_kb(prefs["gender"])
            ))
            return

        # ── Speed change ─────────────────────────────────────────────────
        if data in SPEED_OPTIONS:
            _, new_speed = SPEED_OPTIONS[data]
            original_text = get_text_cache(msg_id)
            if not original_text:
                await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
                return

            lock = _get_user_lock(user_id)
            if lock.locked():
                await query.answer("⏳ សូមរង់ចាំ...")
                return

            update_user_speed(user_id, new_speed)
            prefs  = get_user_prefs(user_id)
            gender = prefs["gender"]

            file_path  = f"spd_{user_id}_{msg_id}.ogg"
            stop_event = asyncio.Event()
            timer_task = asyncio.create_task(
                send_status_timer(query.message.chat.id, context.bot, stop_event)
            )
            async with lock:
                try:
                    label = await generate_voice(original_text, gender, new_speed, file_path)
                    stop_event.set()
                    await timer_task
                    try:
                        await query.message.delete()
                    except Exception:
                        pass
                    with open(file_path, "rb") as audio:
                        new_msg = await safe_send(query.message.chat.send_voice(
                            voice=audio,
                            caption=f"🗣️ {label}",
                            reply_markup=get_main_kb(gender),
                        ))
                    if new_msg:
                        save_text_cache(
                            new_msg.message_id, original_text,
                            user_id=user_id,
                            username=query.from_user.username or query.from_user.first_name,
                        )
                except Exception as e:
                    logger.error(f"speed regen error: {e}")
                    stop_event.set()
                    await timer_task
                finally:
                    _cleanup(file_path)
            return

        # ── Gender change ─────────────────────────────────────────────────
        if data in ("tg_female", "tg_male"):
            new_gender    = data.replace("tg_", "")
            original_text = get_text_cache(msg_id)
            if not original_text:
                await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
                return

            lock = _get_user_lock(user_id)
            if lock.locked():
                await query.answer("⏳ សូមរង់ចាំ...")
                return

            update_user_gender(user_id, new_gender)
            prefs = get_user_prefs(user_id)
            speed = prefs["speed"]

            file_path  = f"rev_{user_id}_{msg_id}.ogg"
            stop_event = asyncio.Event()
            timer_task = asyncio.create_task(
                send_status_timer(query.message.chat.id, context.bot, stop_event)
            )
            async with lock:
                try:
                    label = await generate_voice(original_text, new_gender, speed, file_path)
                    stop_event.set()
                    await timer_task
                    try:
                        await query.message.delete()
                    except Exception:
                        pass
                    with open(file_path, "rb") as audio:
                        new_msg = await safe_send(query.message.chat.send_voice(
                            voice=audio,
                            caption=f"🗣️ {label}",
                            reply_markup=get_main_kb(new_gender),
                        ))
                    if new_msg:
                        save_text_cache(
                            new_msg.message_id, original_text,
                            user_id=user_id,
                            username=query.from_user.username or query.from_user.first_name,
                        )
                except Exception as e:
                    logger.error(f"gender regen error: {e}")
                    stop_event.set()
                    await timer_task
                finally:
                    _cleanup(file_path)
            return

        # ── Transcript → TTS ─────────────────────────────────────────────
        if data.startswith("tts_transcript:"):
            transcript_msg_id = int(data.split(":")[1])
            original_text     = get_text_cache(transcript_msg_id)
            if not original_text:
                await query.answer("❌ រកអត្ថបទមិនឃើញ។", show_alert=True)
                return

            lock = _get_user_lock(user_id)
            if lock.locked():
                await query.answer("⏳ សូមរង់ចាំ...")
                return

            prefs  = get_user_prefs(user_id)
            gender = prefs["gender"]
            speed  = prefs["speed"]

            file_path  = f"tts_tr_{user_id}_{transcript_msg_id}.ogg"
            stop_event = asyncio.Event()
            timer_task = asyncio.create_task(
                send_status_timer(query.message.chat.id, context.bot, stop_event)
            )
            async with lock:
                try:
                    label = await generate_voice(original_text, gender, speed, file_path)
                    stop_event.set()
                    await timer_task
                    with open(file_path, "rb") as audio:
                        new_msg = await safe_send(query.message.chat.send_voice(
                            voice=audio,
                            caption=f"🗣️ {label}",
                            reply_markup=get_main_kb(gender),
                        ))
                    if new_msg:
                        save_text_cache(
                            new_msg.message_id, original_text,
                            user_id=user_id,
                            username=query.from_user.username or query.from_user.first_name,
                        )
                except Exception as e:
                    logger.error(f"transcript TTS error: {e}")
                    stop_event.set()
                    await timer_task
                finally:
                    _cleanup(file_path)
            return

        # ── Transcript → delete ───────────────────────────────────────────
        if data.startswith("del_transcript:"):
            try:
                await query.message.delete()
            except Exception as e:
                logger.error(f"delete transcript error: {e}")
            return

    except Exception as e:
        logger.error(f"on_callback unhandled error [data={data}]: {e}")

# ---------------------------------------------------------------------------
# Global error handler — logs but never crashes the bot
# ---------------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Unhandled exception: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await safe_send(update.effective_message.reply_text(
                "⚠️ មានបញ្ហាបច្ចេកទេស។ Bot នៅដំណើរការ — សូមព្យាយាមម្តងទៀត។"
            ))
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Bot runner — isolated event loop per attempt, guarantees clean restart
# ---------------------------------------------------------------------------
async def _run_bot():
    """Build and run the bot. Raises on any fatal error so main() can retry."""
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
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_error_handler(error_handler)

    print("🟢 Bot polling started.")
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )
        # Park here forever — any exception propagates out and triggers restart
        await asyncio.Event().wait()

# ---------------------------------------------------------------------------
# Main — infinite restart loop, never dies
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
            asyncio.run(_run_bot())
        except Exception as e:
            logger.error(f"💥 Bot crashed: {e} — restarting in 5s...")
        else:
            logger.warning("⚠️  Polling stopped — restarting in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
