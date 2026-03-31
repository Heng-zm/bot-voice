import logging
import os
import re
import asyncio
import threading
import time
import html
import functools
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
            logger_ka.info(f"Keep-alive -> {r.status_code}")
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
    BadRequest, TelegramError, Forbidden,
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SB_URL             = os.getenv("SUPABASE_URL")
SB_KEY             = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")

# Admin user IDs — comma-separated in env: ADMIN_IDS=123456,789012
_raw_admin_ids = os.getenv("ADMIN_IDS", "")
ADMIN_IDS: set[int] = set()
for _aid in _raw_admin_ids.split(","):
    _aid = _aid.strip()
    if _aid.isdigit():
        ADMIN_IDS.add(int(_aid))

GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
MAX_VOICE_BYTES = 20 * 1024 * 1024
DEFAULT_SPEED   = 1.0

# ---------------------------------------------------------------------------
# State constants (stored in context.user_data)
# ---------------------------------------------------------------------------
BROADCAST_WAIT_MESSAGE = 1   # admin waiting to send broadcast payload
CHAT_WAIT_MESSAGE      = 2   # admin in per-user chat mode

# ---------------------------------------------------------------------------
# In-memory stores
# FIX: cleared at each _run_bot() start to avoid stale state across restarts.
# ---------------------------------------------------------------------------
_pending_broadcast: dict[int, dict] = {}   # admin_id -> broadcast payload
_admin_chat_target: dict[int, int]  = {}   # admin_id -> target_user_id
_user_to_admin:     dict[int, int]  = {}   # FIX: inverse map for O(1) lookup

# ---------------------------------------------------------------------------
# Supabase + Gemini init
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
# User prefs in-memory cache
# FIX: avoid hitting Supabase on every callback; TTL = 300s
# ---------------------------------------------------------------------------
_prefs_cache: dict[int, tuple[dict, float]] = {}   # user_id -> (prefs, timestamp)
_PREFS_TTL = 300.0  # seconds

def _cache_prefs(user_id: int, prefs: dict) -> None:
    _prefs_cache[user_id] = (prefs, time.monotonic())

def _get_cached_prefs(user_id: int) -> dict | None:
    entry = _prefs_cache.get(user_id)
    if entry and time.monotonic() - entry[1] < _PREFS_TTL:
        return entry[0]
    return None

def _invalidate_prefs(user_id: int) -> None:
    _prefs_cache.pop(user_id, None)

# ---------------------------------------------------------------------------
# Per-user async lock
# FIX: dict guarded by threading.Lock avoids "Future attached to a different
# loop" errors that lru_cache+asyncio.Lock causes across asyncio.run() restarts.
# ---------------------------------------------------------------------------
_user_locks: dict[int, asyncio.Lock] = {}
_user_locks_mutex = threading.Lock()

def _get_user_lock(user_id: int) -> asyncio.Lock:
    with _user_locks_mutex:
        if user_id not in _user_locks:
            _user_locks[user_id] = asyncio.Lock()
        return _user_locks[user_id]

# ---------------------------------------------------------------------------
# Safe Telegram send with retry
# ---------------------------------------------------------------------------
async def safe_send(coro, retries: int = 3, delay: float = 2.0):
    for attempt in range(retries):
        try:
            return await coro
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 1)
        except (TimedOut, NetworkError) as e:
            if attempt < retries - 1:
                logger.warning(f"Network error (attempt {attempt+1}): {e}")
                await asyncio.sleep(delay)
            else:
                raise
        except BadRequest as e:
            logger.error(f"Bad request: {e}")
            raise
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            raise
    return None

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def sync_user_data(user):
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
    """
    Returns user prefs from cache if fresh, otherwise fetches from Supabase.
    FIX: results are cached to avoid a DB call on every callback.
    """
    cached = _get_cached_prefs(user_id)
    if cached is not None:
        return cached

    defaults = {"gender": "female", "speed": DEFAULT_SPEED}
    if not supabase:
        return defaults
    try:
        res = (
            supabase.table("user_prefs")
            .select("gender, speed")
            .eq("user_id", user_id)
            .execute()
        )
        if res.data:
            row = res.data[0]
            defaults["gender"] = row.get("gender") or "female"
            defaults["speed"]  = float(row.get("speed") or DEFAULT_SPEED)
    except Exception as e:
        logger.error(f"DB get_user_prefs: {e}")

    _cache_prefs(user_id, defaults)
    return defaults


def get_all_user_ids() -> list[int]:
    """
    FIX: paginated fetch to avoid silent truncation at Supabase's 1000-row default limit.
    """
    if not supabase:
        return []
    try:
        all_ids, page, page_size = [], 0, 1000
        while True:
            res = (
                supabase.table("user_prefs")
                .select("user_id")
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            batch = res.data or []
            all_ids.extend(row["user_id"] for row in batch)
            if len(batch) < page_size:
                break
            page += 1
        return all_ids
    except Exception as e:
        logger.error(f"DB get_all_user_ids: {e}")
        return []


def get_all_users_with_names() -> list[dict]:
    """
    Returns list of {user_id, username}.
    FIX: paginated fetch.
    """
    if not supabase:
        return []
    try:
        all_users, page, page_size = [], 0, 1000
        while True:
            res = (
                supabase.table("user_prefs")
                .select("user_id, username")
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            batch = res.data or []
            all_users.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return all_users
    except Exception as e:
        logger.error(f"DB get_all_users_with_names: {e}")
        return []


def update_user_gender(user_id: int, gender: str):
    _invalidate_prefs(user_id)  # FIX: bust cache on write
    if not supabase:
        return
    def _run():
        try:
            supabase.table("user_prefs").update({"gender": gender}).eq("user_id", user_id).execute()
        except Exception as e:
            logger.error(f"DB update_user_gender: {e}")
    threading.Thread(target=_run, daemon=True).start()


def update_user_speed(user_id: int, speed: float):
    _invalidate_prefs(user_id)  # FIX: bust cache on write
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
        logger.info("speed column exists")
    except Exception as e:
        if "does not exist" in str(e):
            logger.warning(
                "speed column missing. Run: ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
            )
        else:
            logger.error(f"ensure_speed_column unexpected error: {e}")

# ---------------------------------------------------------------------------
# Status timer
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


async def _stop_timer(stop_event: asyncio.Event, timer_task: asyncio.Task):
    """
    FIX: guard against awaiting an already-finished task, which would re-raise
    its stored exception instead of CancelledError.
    """
    stop_event.set()
    if not timer_task.done():
        timer_task.cancel()
        try:
            await timer_task
        except (asyncio.CancelledError, Exception):
            pass

# ---------------------------------------------------------------------------
# Audio pipeline
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _build_atempo_chain(speed: float) -> str:
    """
    FFmpeg atempo filter only accepts values in [0.5, 2.0].
    Chain multiple stages to achieve speeds outside that range.
    FIX: decorated with lru_cache — result is deterministic and pure.
    """
    stages, r = [], round(speed, 6)
    if r < 1.0:
        while r < 0.5:
            stages.append("atempo=0.5")
            r = round(r / 0.5, 6)
        stages.append(f"atempo={r:.6f}")
    else:
        while r > 2.0:
            stages.append("atempo=2.0")
            r = round(r / 2.0, 6)
        stages.append(f"atempo={r:.6f}")
    return ",".join(stages)


async def generate_voice(text: str, gender: str, speed: float, output_path: str) -> str:
    is_khmer = bool(re.search(r"[\u1780-\u17FF]", text))
    voice    = VOICE_MAP["km" if is_khmer else "en"][gender]

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

    af  = _build_atempo_chain(speed) if abs(speed - DEFAULT_SPEED) > 1e-6 else None
    cmd = [_FFMPEG_EXE, "-y", "-f", "mp3", "-i", "pipe:0"]
    if af:
        cmd += ["-filter:a", af]
    cmd += ["-c:a", "libopus", "-b:a", "32k", output_path]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,   # FIX: capture stderr for diagnostics
    )
    # FIX: communicate() returns (stdout, stderr); include stderr in error message
    _, stderr_data = await proc.communicate(input=mp3_data)
    if proc.returncode != 0:
        stderr_snippet = (stderr_data or b"").decode(errors="replace")[-400:]
        raise RuntimeError(
            f"FFmpeg failed (code {proc.returncode}): {stderr_snippet}"
        )
    return BOT_TAG

# ---------------------------------------------------------------------------
# Gemini transcription
# ---------------------------------------------------------------------------
async def transcribe_voice(ogg_path: str) -> str:
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    with open(ogg_path, "rb") as f:
        audio_bytes = f.read()
    prompt = (
        "Transcribe this audio exactly as spoken. "
        "Output ONLY the transcribed text — no labels, no explanation. "
        "Support both Khmer and English."
    )
    def _call():
        return _gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                prompt,
            ],
        )
    loop = asyncio.get_running_loop()
    try:
        response = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=60)
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini transcription timed out after 60s")

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


def get_transcription_kb(transcript_msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📢 AI អាន", callback_data=f"tts_transcript:{transcript_msg_id}"),
        InlineKeyboardButton("🗑️ លុប",    callback_data=f"del_transcript:{transcript_msg_id}"),
    ]])


def get_broadcast_confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ បញ្ជាក់ Broadcast", callback_data="bc_confirm"),
            InlineKeyboardButton("❌ បោះបង់",            callback_data="bc_cancel"),
        ]
    ])


def get_users_page_kb(users: list[dict], page: int, page_size: int = 8) -> InlineKeyboardMarkup:
    """Paginated user list — each button opens a chat session with that user."""
    total_pages = max(1, (len(users) + page_size - 1) // page_size)
    chunk = users[page * page_size : page * page_size + page_size]

    rows = []
    for u in chunk:
        uid   = u["user_id"]
        uname = (u.get("username") or str(uid))[:20]
        rows.append([InlineKeyboardButton(
            f"👤 {uname}  ({uid})",
            callback_data=f"chat_open:{uid}",
        )])

    # Pagination row
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"users_page:{page - 1}"))
    nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"users_page:{page + 1}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton("❌ បិទ", callback_data="users_close")])
    return InlineKeyboardMarkup(rows)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# Admin guard
# FIX: use functools.wraps to preserve __doc__ and __name__ on decorated handlers.
# ---------------------------------------------------------------------------
def admin_only(handler):
    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        if uid not in ADMIN_IDS:
            if update.message:
                await safe_send(update.message.reply_text("⛔ អ្នកមិនមានសិទ្ធិប្រើពាក្យបញ្ជានេះទេ។"))
            return
        return await handler(update, context)
    return wrapper

# ---------------------------------------------------------------------------
# Chat session helpers
# FIX: all open/close operations go through these helpers to keep
# _admin_chat_target and _user_to_admin (inverse map) in sync.
# ---------------------------------------------------------------------------
def _open_session(admin_id: int, target_id: int) -> None:
    """Register an admin↔user chat session in both maps."""
    # If this admin was already chatting with someone else, close that first
    old_target = _admin_chat_target.get(admin_id)
    if old_target is not None:
        _user_to_admin.pop(old_target, None)
    _admin_chat_target[admin_id] = target_id
    _user_to_admin[target_id]    = admin_id


def _close_session(admin_id: int) -> int | None:
    """Remove the session and return the target_id that was removed, or None."""
    target_id = _admin_chat_target.pop(admin_id, None)
    if target_id is not None:
        _user_to_admin.pop(target_id, None)
    return target_id


def _get_admin_for_user(user_id: int) -> int | None:
    """
    FIX: O(1) inverse-map lookup instead of O(n) linear scan.
    """
    return _user_to_admin.get(user_id)

# ===========================================================================
# BROADCAST
# ===========================================================================

@admin_only
async def broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/broadcast  Ask admin to send message or photo+caption."""
    context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
    await safe_send(update.message.reply_text(
        "📡 <b>Admin Broadcast</b>\n\n"
        "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Broadcast ។\n"
        "👉 អាចផ្ញើរូបភាព + Caption រួមគ្នា ឬ តែ text ។\n\n"
        "វាយ /cancel ដើម្បីបោះបង់។",
        parse_mode="HTML",
    ))


# FIX: broadcast_receive is an internal helper — it is only ever called from
# on_text / on_photo after an explicit ADMIN_IDS check at the call site.
# Added a runtime guard here as defence-in-depth so a future code path cannot
# accidentally expose this to non-admins.
async def broadcast_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receive admin broadcast payload; called from on_text/on_photo."""
    user_id = update.effective_user.id
    # Defence-in-depth guard (call sites already check, but be safe)
    if user_id not in ADMIN_IDS:
        return

    msg = update.message

    photo_file_id: str | None = None
    caption_text:  str | None = None
    plain_text:    str | None = None

    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text  = msg.caption or ""
    elif msg.text:
        plain_text = msg.text
    else:
        await safe_send(msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return

    _pending_broadcast[user_id] = {
        "photo_file_id": photo_file_id,
        "caption":       caption_text,
        "text":          plain_text,
    }

    if photo_file_id:
        preview_cap = html.escape(caption_text) if caption_text else "<i>(គ្មាន Caption)</i>"
        await safe_send(msg.reply_photo(
            photo=photo_file_id,
            caption=(
                f"👁️ <b>Preview Broadcast</b>\n\n{preview_cap}\n\n"
                "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?"
            ),
            parse_mode="HTML",
            reply_markup=get_broadcast_confirm_kb(),
        ))
    else:
        await safe_send(msg.reply_text(
            f"👁️ <b>Preview Broadcast</b>\n\n{html.escape(plain_text)}\n\n"
            "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?",
            parse_mode="HTML",
            reply_markup=get_broadcast_confirm_kb(),
        ))


async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """bc_confirm / bc_cancel."""
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data

    if user_id not in ADMIN_IDS:
        try:
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        except Exception:
            pass
        return

    try:
        await query.answer()
    except Exception:
        pass

    if data == "bc_cancel":
        _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        try:
            await query.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        await safe_send(query.message.reply_text("❌ Broadcast បានបោះបង់។"))
        return

    if data == "bc_confirm":
        pending = _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        if not pending:
            await safe_send(query.message.reply_text("⚠️ រកទិន្នន័យ Broadcast មិនឃើញ។"))
            return
        try:
            await query.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        # FIX: use application.create_task so PTB tracks and awaits it on shutdown
        context.application.create_task(_do_broadcast(context.bot, user_id, pending))


async def _do_broadcast(bot, admin_id: int, pending: dict):
    """
    Deliver broadcast to all users with proper rate-limit handling.
    FIX:
      - Per-message 50ms sleep keeps send rate ~20/s, under Telegram's 30/s limit.
      - Retry loop retries RetryAfter twice before marking as failed.
      - Progress updates are best-effort (ignored if they fail).
    """
    user_ids = await asyncio.get_running_loop().run_in_executor(None, get_all_user_ids)
    total    = len(user_ids)
    sent = failed = blocked = 0

    photo_file_id = pending.get("photo_file_id")
    caption       = pending.get("caption") or ""
    plain_text    = pending.get("text") or ""

    progress_msg = await safe_send(
        bot.send_message(chat_id=admin_id, text=f"📡 កំពុង Broadcast ទៅ {total} នាក់...")
    )

    async def _send_one(uid: int) -> str:
        """Returns 'sent', 'blocked', or 'failed'."""
        for attempt in range(2):
            try:
                if photo_file_id:
                    await bot.send_photo(
                        chat_id=uid,
                        photo=photo_file_id,
                        caption=caption or None,
                        parse_mode="HTML" if caption else None,
                    )
                else:
                    await bot.send_message(chat_id=uid, text=plain_text, parse_mode="HTML")
                return "sent"
            except Forbidden:
                return "blocked"
            except RetryAfter as e:
                await asyncio.sleep(e.retry_after + 1)
                # retry once more after waiting
            except Exception as e:
                logger.error(f"Broadcast error uid={uid} attempt={attempt}: {e}")
                if attempt == 1:
                    return "failed"
        return "failed"

    for i, uid in enumerate(user_ids):
        result = await _send_one(uid)
        if result == "sent":
            sent += 1
        elif result == "blocked":
            blocked += 1
        else:
            failed += 1

        # FIX: throttle to ~20 messages/second to stay under Telegram's 30/s limit
        await asyncio.sleep(0.05)

        if (i + 1) % 25 == 0 and progress_msg:
            try:
                pct = int((i + 1) / total * 100)
                await progress_msg.edit_text(f"📡 Broadcast {pct}% ({i+1}/{total})...")
            except Exception:
                pass

    report = (
        f"✅ <b>Broadcast រួចរាល់!</b>\n\n"
        f"👥 សរុប: {total}\n"
        f"📨 បានផ្ញើ: {sent}\n"
        f"🚫 Blocked: {blocked}\n"
        f"❌ Failed: {failed}"
    )
    try:
        if progress_msg:
            await progress_msg.edit_text(report, parse_mode="HTML")
        else:
            await bot.send_message(chat_id=admin_id, text=report, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Broadcast report error: {e}")

# ===========================================================================
# PER-USER CHAT
# ===========================================================================

@admin_only
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/users  Show paginated list of all registered users with chat buttons."""
    users = await asyncio.get_running_loop().run_in_executor(None, get_all_users_with_names)
    if not users:
        await safe_send(update.message.reply_text("❌ គ្មានអ្នកប្រើប្រាស់ registered ទេ។"))
        return
    kb = get_users_page_kb(users, page=0)
    await safe_send(update.message.reply_text(
        f"👥 <b>អ្នកប្រើប្រាស់ ({len(users)} នាក់)</b>\n"
        "ចុចលើឈ្មោះ ដើម្បីចាប់ផ្ដើម Chat ។",
        parse_mode="HTML",
        reply_markup=kb,
    ))


@admin_only
async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/chat <user_id>  Open a direct chat session with a specific user."""
    admin_id = update.effective_user.id
    args     = context.args or []

    if not args or not args[0].isdigit():
        await safe_send(update.message.reply_text(
            "❌ Usage: /chat <user_id>\n"
            "ឬប្រើ /users ដើម្បីជ្រើស user ។"
        ))
        return

    target_id  = int(args[0])
    known_ids  = await asyncio.get_running_loop().run_in_executor(None, get_all_user_ids)
    if target_id not in known_ids:
        await safe_send(update.message.reply_text(
            f"❌ User <code>{target_id}</code> មិនមាននៅក្នុង Database ។",
            parse_mode="HTML",
        ))
        return

    await _open_chat_session(context.bot, admin_id, target_id, context)
    await safe_send(update.message.reply_text(
        f"💬 <b>Chat Mode បើក</b>\n\n"
        f"កំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
        f"សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
        f"វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
        parse_mode="HTML",
    ))


@admin_only
async def cmd_endchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/endchat  Close the current per-user chat session."""
    admin_id  = update.effective_user.id
    target_id = _close_session(admin_id)   # FIX: use helper to keep both maps in sync
    context.user_data.pop("chat_state", None)

    if target_id is None:
        await safe_send(update.message.reply_text("ℹ️ អ្នកមិនទាន់ open Chat ណាមួយទេ។"))
        return

    await safe_send(update.message.reply_text(
        f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។",
        parse_mode="HTML",
    ))
    try:
        await context.bot.send_message(
            chat_id=target_id,
            text="ℹ️ Admin បានបញ្ចប់ Session Chat ។",
        )
    except Exception:
        pass


async def _open_chat_session(bot, admin_id: int, target_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Internal helper — sets state and notifies both sides."""
    _open_session(admin_id, target_id)  # FIX: use helper to keep both maps in sync
    context.user_data["chat_state"] = CHAT_WAIT_MESSAGE
    try:
        await bot.send_message(
            chat_id=target_id,
            text="🔔 Admin ចង់ Chat ជាមួយអ្នក។ ផ្ញើសារតបមកបាន!",
        )
    except Exception as e:
        logger.warning(f"Could not notify user {target_id}: {e}")


async def users_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle users_page:<n>  users_close  chat_open:<uid>  noop callbacks."""
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data

    if user_id not in ADMIN_IDS:
        try:
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        except Exception:
            pass
        return

    try:
        await query.answer()
    except Exception:
        pass

    if data == "users_close":
        try:
            await query.message.delete()
        except Exception:
            pass
        return

    if data == "noop":
        return

    if data.startswith("users_page:"):
        page  = int(data.split(":")[1])
        users = await asyncio.get_running_loop().run_in_executor(None, get_all_users_with_names)
        try:
            await query.message.edit_reply_markup(reply_markup=get_users_page_kb(users, page=page))
        except Exception:
            pass
        return

    if data.startswith("chat_open:"):
        target_id = int(data.split(":")[1])
        admin_id  = user_id
        try:
            await query.message.delete()
        except Exception:
            pass
        await _open_chat_session(context.bot, admin_id, target_id, context)
        await safe_send(context.bot.send_message(
            chat_id=admin_id,
            text=(
                f"💬 <b>Chat Mode បើក</b>\n\n"
                f"កំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
                f"សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
                f"វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។"
            ),
            parse_mode="HTML",
        ))


async def _fwd_admin_to_user(bot, admin_id: int, target_id: int, msg) -> bool:
    """Forward any message type from admin to target user. Returns success."""
    try:
        if msg.text:
            await bot.send_message(
                chat_id=target_id,
                text=f"📩 <b>Admin:</b>\n{html.escape(msg.text)}",
                parse_mode="HTML",
            )
        elif msg.photo:
            cap = f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>"
            await bot.send_photo(chat_id=target_id, photo=msg.photo[-1].file_id,
                                 caption=cap, parse_mode="HTML")
        elif msg.voice:
            await bot.send_voice(chat_id=target_id, voice=msg.voice.file_id,
                                 caption="📩 Admin voice message")
        elif msg.sticker:
            await bot.send_sticker(chat_id=target_id, sticker=msg.sticker.file_id)
        elif msg.document:
            cap = f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>"
            await bot.send_document(chat_id=target_id, document=msg.document.file_id,
                                    caption=cap, parse_mode="HTML")
        elif msg.video:
            cap = f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>"
            await bot.send_video(chat_id=target_id, video=msg.video.file_id,
                                 caption=cap, parse_mode="HTML")
        elif msg.audio:
            await bot.send_audio(chat_id=target_id, audio=msg.audio.file_id)
        else:
            await bot.forward_message(chat_id=target_id, from_chat_id=admin_id,
                                      message_id=msg.message_id)
        return True
    except Forbidden:
        return False
    except Exception as e:
        logger.error(f"_fwd_admin_to_user error: {e}")
        return False


async def _fwd_user_to_admin(bot, admin_id: int, user_id: int, username: str, msg) -> bool:
    """Forward any message type from user to admin with a context banner. Returns success."""
    banner = f"💬 <b>{html.escape(username)} ({user_id}):</b>"
    try:
        if msg.text:
            await bot.send_message(
                chat_id=admin_id,
                text=f"{banner}\n{html.escape(msg.text)}",
                parse_mode="HTML",
            )
        elif msg.photo:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_photo(chat_id=admin_id, photo=msg.photo[-1].file_id,
                                 caption=cap, parse_mode="HTML")
        elif msg.voice:
            await bot.send_voice(chat_id=admin_id, voice=msg.voice.file_id,
                                 caption=banner, parse_mode="HTML")
        elif msg.sticker:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_sticker(chat_id=admin_id, sticker=msg.sticker.file_id)
        elif msg.document:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_document(chat_id=admin_id, document=msg.document.file_id,
                                    caption=cap, parse_mode="HTML")
        elif msg.video:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_video(chat_id=admin_id, video=msg.video.file_id,
                                 caption=cap, parse_mode="HTML")
        elif msg.audio:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_audio(chat_id=admin_id, audio=msg.audio.file_id)
        else:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.forward_message(chat_id=admin_id, from_chat_id=user_id,
                                      message_id=msg.message_id)
        return True
    except Exception as e:
        logger.error(f"_fwd_user_to_admin error: {e}")
        return False

# ===========================================================================
# STATS + CANCEL
# ===========================================================================

@admin_only
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_ids     = await asyncio.get_running_loop().run_in_executor(None, get_all_user_ids)
    active_chats = len(_admin_chat_target)
    await safe_send(update.message.reply_text(
        f"📊 <b>Bot Statistics</b>\n\n"
        f"👥 អ្នកប្រើប្រាស់សរុប: <b>{len(user_ids)}</b>\n"
        f"💬 Active Admin Chats: <b>{active_chats}</b>",
        parse_mode="HTML",
    ))


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /cancel  Clear whichever admin state is currently active
    (broadcast input or per-user chat).
    FIX: both states are checked independently so neither silently shadows the other.
    """
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        return

    cleared = False

    if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
        _pending_broadcast.pop(uid, None)
        context.user_data.pop("bc_state", None)
        await safe_send(update.message.reply_text("❌ Broadcast បានបោះបង់។"))
        cleared = True

    if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _close_session(uid)   # FIX: use helper to keep both maps in sync
        context.user_data.pop("chat_state", None)
        if target_id:
            await safe_send(update.message.reply_text(
                f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។",
                parse_mode="HTML",
            ))
            try:
                await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")
            except Exception:
                pass
        cleared = True

    if not cleared:
        await safe_send(update.message.reply_text("ℹ️ មិនមាន operation ត្រូវ cancel ទេ។"))

# ===========================================================================
# REGULAR HANDLERS
# ===========================================================================

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


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Photos from admin  → broadcast input OR chat forward
    Photos from user   → forward to admin if session is open
    """
    msg     = update.message
    user    = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    # Admin: broadcast input
    if user_id in ADMIN_IDS and context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
        await broadcast_receive(update, context)
        return

    # Admin: chat mode
    if user_id in ADMIN_IDS and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ Photo ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok else
                f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)   # FIX: use helper
                context.user_data.pop("chat_state", None)
        return

    # Regular user → forward to admin if chatting
    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(msg.reply_text("✅ រូបភាពបានផ្ញើដល់ Admin ។"))


async def on_any_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle sticker / document / video / audio.
    Admin in chat mode → forward to user.
    User being chatted → forward to admin.
    Otherwise → ignore.
    """
    msg     = update.message
    user    = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    # Admin chat mode
    if user_id in ADMIN_IDS and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok else
                f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)   # FIX: use helper
                context.user_data.pop("chat_state", None)
        return

    # User → admin
    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(msg.reply_text("✅ ផ្ញើដល់ Admin ។"))


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Voice messages:
    - Admin in chat mode  → forward to target user (skip transcription)
    - User being chatted  → forward to admin
    - Otherwise           → normal Gemini transcription
    """
    msg     = update.message
    user    = update.effective_user
    user_id = user.id

    # Admin chat mode
    if user_id in ADMIN_IDS and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ Voice ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok else
                f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)   # FIX: use helper
                context.user_data.pop("chat_state", None)
        return

    # User → admin
    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(msg.reply_text("✅ Voice ផ្ញើដល់ Admin ។"))
        return

    # Normal transcription
    if not _gemini:
        try:
            await safe_send(msg.reply_text(
                "❌ Gemini API មិន Activate ទេ។ សូម Set GEMINI_API_KEY ។"
            ))
        except Exception:
            pass
        return

    if msg.voice.file_size and msg.voice.file_size > MAX_VOICE_BYTES:
        await safe_send(msg.reply_text("❌ ឯកសារសំឡេងធំពេក (អតិបរមា 20MB)។"))
        return

    sync_user_data(user)

    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_TRANSCRIBE_FRAMES)
    )
    ogg_path = f"voice_in_{user_id}_{msg.message_id}.ogg"
    try:
        voice_file = await safe_send(context.bot.get_file(msg.voice.file_id))
        if not voice_file:
            raise RuntimeError("Could not get voice file")
        await voice_file.download_to_drive(ogg_path)

        transcript = await transcribe_voice(ogg_path)
        await _stop_timer(stop_event, timer_task)

        if not transcript:
            await safe_send(msg.reply_text("❌ រក Transcript មិនឃើញ។"))
            return

        is_khmer  = bool(re.search(r"[\u1780-\u17FF]", transcript))
        lang_flag = "🇰🇭" if is_khmer else "🇺🇸"

        reply = await safe_send(msg.reply_text(
            f"📝 <b>Transcript</b> {lang_flag}\n\n{html.escape(transcript)}",
            parse_mode="HTML",
            reply_markup=get_transcription_kb(0),
        ))
        if reply:
            await safe_send(reply.edit_reply_markup(
                reply_markup=get_transcription_kb(reply.message_id)
            ))
            save_text_cache(reply.message_id, transcript,
                            user_id=user_id, username=user.username or user.first_name)
    except Exception as e:
        logger.error(f"on_voice error: {e}")
        await _stop_timer(stop_event, timer_task)
        try:
            await safe_send(msg.reply_text("❌ មានបញ្ហាក្នុងការ Transcribe។"))
        except Exception:
            pass
    finally:
        _cleanup(ogg_path)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg  = update.message
    text = msg.text
    if not text:
        return

    user    = update.effective_user
    user_id = user.id

    # Admin: broadcast input
    if user_id in ADMIN_IDS and context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
        await broadcast_receive(update, context)
        return

    # Admin: chat mode → forward text to target user
    # FIX: check chat_state independently — previously bc_state could shadow it
    if user_id in ADMIN_IDS and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            if ok:
                await safe_send(msg.reply_text(
                    f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។",
                    parse_mode="HTML",
                ))
            else:
                await safe_send(msg.reply_text(
                    f"❌ User <code>{target_id}</code> blocked bot ។ Chat session បានបិទ។",
                    parse_mode="HTML",
                ))
                _close_session(user_id)   # FIX: use helper
                context.user_data.pop("chat_state", None)
        return

    # Regular user → forward to admin if session is open
    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(msg.reply_text("✅ សាររបស់អ្នកបានផ្ញើដល់ Admin ។"))
        return

    # Welcome shortcut
    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    # Normal TTS
    lock = _get_user_lock(user_id)
    if lock.locked():
        try:
            await safe_send(msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        except Exception:
            pass
        return

    sync_user_data(user)
    prefs  = await asyncio.get_running_loop().run_in_executor(None, get_user_prefs, user_id)
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
            await _stop_timer(stop_event, timer_task)
            with open(file_path, "rb") as audio:
                sent_msg = await safe_send(msg.reply_voice(
                    voice=audio,
                    caption=f"🗣️ {label}",
                    reply_markup=get_main_kb(gender),
                ))
            if sent_msg:
                save_text_cache(sent_msg.message_id, text,
                                user_id=user_id, username=user.username or user.first_name)
        except Exception as e:
            logger.error(f"on_text TTS error: {e}")
            await _stop_timer(stop_event, timer_task)
            try:
                await safe_send(msg.reply_text("❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"))
            except Exception:
                pass
        finally:
            _cleanup(file_path)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    msg_id  = query.message.message_id
    data    = query.data

    try:
        if data == "show_speed":
            try:
                await query.answer()
            except Exception:
                pass
            prefs = await asyncio.get_running_loop().run_in_executor(None, get_user_prefs, user_id)
            await safe_send(query.message.edit_reply_markup(reply_markup=get_speed_kb(prefs["speed"])))
            return

        if data == "hide_speed":
            try:
                await query.answer()
            except Exception:
                pass
            prefs = await asyncio.get_running_loop().run_in_executor(None, get_user_prefs, user_id)
            await safe_send(query.message.edit_reply_markup(reply_markup=get_main_kb(prefs["gender"])))
            return

        if data in SPEED_OPTIONS:
            _, new_speed  = SPEED_OPTIONS[data]
            original_text = await asyncio.get_running_loop().run_in_executor(
                None, get_text_cache, msg_id
            )
            if not original_text:
                await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
                return
            lock = _get_user_lock(user_id)
            if lock.locked():
                await query.answer("⏳ សូមរង់ចាំ...")
                return
            try:
                await query.answer()
            except Exception:
                pass
            prefs  = await asyncio.get_running_loop().run_in_executor(None, get_user_prefs, user_id)
            gender = prefs["gender"]
            update_user_speed(user_id, new_speed)
            file_path  = f"spd_{user_id}_{msg_id}.ogg"
            stop_event = asyncio.Event()
            timer_task = asyncio.create_task(
                send_status_timer(query.message.chat.id, context.bot, stop_event)
            )
            async with lock:
                try:
                    label = await generate_voice(original_text, gender, new_speed, file_path)
                    await _stop_timer(stop_event, timer_task)
                    try:
                        await query.message.delete()
                    except Exception:
                        pass
                    with open(file_path, "rb") as audio:
                        new_msg = await safe_send(query.message.chat.send_voice(
                            voice=audio, caption=f"🗣️ {label}", reply_markup=get_main_kb(gender),
                        ))
                    if new_msg:
                        save_text_cache(new_msg.message_id, original_text, user_id=user_id,
                                        username=query.from_user.username or query.from_user.first_name)
                except Exception as e:
                    logger.error(f"speed regen error: {e}")
                    await _stop_timer(stop_event, timer_task)
                finally:
                    _cleanup(file_path)
            return

        if data in ("tg_female", "tg_male"):
            new_gender    = data.replace("tg_", "")
            original_text = await asyncio.get_running_loop().run_in_executor(
                None, get_text_cache, msg_id
            )
            if not original_text:
                await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
                return
            lock = _get_user_lock(user_id)
            if lock.locked():
                await query.answer("⏳ សូមរង់ចាំ...")
                return
            try:
                await query.answer()
            except Exception:
                pass
            prefs = await asyncio.get_running_loop().run_in_executor(None, get_user_prefs, user_id)
            speed = prefs["speed"]
            update_user_gender(user_id, new_gender)
            file_path  = f"rev_{user_id}_{msg_id}.ogg"
            stop_event = asyncio.Event()
            timer_task = asyncio.create_task(
                send_status_timer(query.message.chat.id, context.bot, stop_event)
            )
            async with lock:
                try:
                    label = await generate_voice(original_text, new_gender, speed, file_path)
                    await _stop_timer(stop_event, timer_task)
                    try:
                        await query.message.delete()
                    except Exception:
                        pass
                    with open(file_path, "rb") as audio:
                        new_msg = await safe_send(query.message.chat.send_voice(
                            voice=audio, caption=f"🗣️ {label}", reply_markup=get_main_kb(new_gender),
                        ))
                    if new_msg:
                        save_text_cache(new_msg.message_id, original_text, user_id=user_id,
                                        username=query.from_user.username or query.from_user.first_name)
                except Exception as e:
                    logger.error(f"gender regen error: {e}")
                    await _stop_timer(stop_event, timer_task)
                finally:
                    _cleanup(file_path)
            return

        if data.startswith("tts_transcript:"):
            transcript_msg_id = int(data.split(":")[1])
            original_text     = await asyncio.get_running_loop().run_in_executor(
                None, get_text_cache, transcript_msg_id
            )
            if not original_text:
                await query.answer("❌ រកអត្ថបទមិនឃើញ។", show_alert=True)
                return
            lock = _get_user_lock(user_id)
            if lock.locked():
                await query.answer("⏳ សូមរង់ចាំ...")
                return
            try:
                await query.answer()
            except Exception:
                pass
            prefs  = await asyncio.get_running_loop().run_in_executor(None, get_user_prefs, user_id)
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
                    await _stop_timer(stop_event, timer_task)
                    with open(file_path, "rb") as audio:
                        new_msg = await safe_send(query.message.chat.send_voice(
                            voice=audio, caption=f"🗣️ {label}", reply_markup=get_main_kb(gender),
                        ))
                    if new_msg:
                        save_text_cache(new_msg.message_id, original_text, user_id=user_id,
                                        username=query.from_user.username or query.from_user.first_name)
                except Exception as e:
                    logger.error(f"transcript TTS error: {e}")
                    await _stop_timer(stop_event, timer_task)
                finally:
                    _cleanup(file_path)
            return

        if data.startswith("del_transcript:"):
            try:
                await query.answer()
            except Exception:
                pass
            try:
                await query.message.delete()
            except Exception as e:
                logger.error(f"del_transcript error: {e}")
            return

        # Unknown callback — still acknowledge it
        try:
            await query.answer()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"on_callback unhandled [data={data}]: {e}")

# ---------------------------------------------------------------------------
# Global error handler
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
# Bot runner
# ---------------------------------------------------------------------------
async def _run_bot():
    """
    FIX: clear ALL module-level state on each start so that a restart after
    a crash never inherits stale admin sessions or broadcast payloads.
    """
    with _user_locks_mutex:
        _user_locks.clear()
    _admin_chat_target.clear()
    _user_to_admin.clear()
    _pending_broadcast.clear()
    _prefs_cache.clear()

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    # ── Handler registration (priority: top = highest) ──────────────────────

    # 1. Commands
    app.add_handler(CommandHandler("start",     on_start))
    app.add_handler(CommandHandler("broadcast", broadcast_start))
    app.add_handler(CommandHandler("cancel",    cmd_cancel))
    app.add_handler(CommandHandler("stats",     admin_stats))
    app.add_handler(CommandHandler("users",     cmd_users))
    app.add_handler(CommandHandler("chat",      cmd_chat))
    app.add_handler(CommandHandler("endchat",   cmd_endchat))

    # 2. Broadcast confirm/cancel — must be before generic on_callback
    app.add_handler(CallbackQueryHandler(broadcast_callback,  pattern=r"^bc_(confirm|cancel)$"))

    # 3. Users list / chat-open — must be before generic on_callback
    app.add_handler(CallbackQueryHandler(
        users_page_callback,
        pattern=r"^(users_page:\d+|users_close|chat_open:\d+|noop)$",
    ))

    # 4. Generic TTS callbacks
    app.add_handler(CallbackQueryHandler(on_callback))

    # 5. Message handlers (photo first so broadcast/chat intercepts work)
    app.add_handler(MessageHandler(filters.PHOTO,   on_photo))
    app.add_handler(MessageHandler(filters.VOICE,   on_voice))
    app.add_handler(MessageHandler(
        filters.Sticker.ALL | filters.Document.ALL | filters.VIDEO | filters.AUDIO,
        on_any_media,
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.add_error_handler(error_handler)

    logger.info(f"Bot polling started. Admins: {ADMIN_IDS or 'none configured'}")
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )
        await asyncio.Event().wait()

# ---------------------------------------------------------------------------
# Main — infinite restart loop
# ---------------------------------------------------------------------------
def main():
    threading.Thread(target=run_flask, daemon=True).start()
    print("Flask health-check server started.")

    threading.Thread(target=keep_alive, daemon=True).start()
    print("Keep-alive thread started.")

    ensure_speed_column()

    if not ADMIN_IDS:
        logger.warning(
            "No ADMIN_IDS configured. "
            "Set ADMIN_IDS=123456,789012 in your environment."
        )

    print("Bot is starting...")
    while True:
        try:
            asyncio.run(_run_bot())
        except Exception as e:
            logger.error(f"Bot crashed: {e} — restarting in 5s...")
        else:
            logger.warning("Polling stopped — restarting in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
