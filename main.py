import logging
import os
import re
import asyncio
import inspect
import threading
import time
import html
import functools
import tempfile
import glob
import requests
import imageio_ffmpeg as _iio_ffmpeg
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone
from google import genai
from google.genai import types as genai_types
from flask import Flask
from supabase import create_client, Client

# ── Flask Web Server ───────────────────────────────────────────────────────
app_flask = Flask(__name__)

@app_flask.route("/")
def health_check():
    return "Bot is running!", 200

@app_flask.route("/ping")
def ping():
    return "pong", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host="0.0.0.0", port=port, threaded=True)

def keep_alive():
    """Self-ping every 4 min as secondary keep-alive layer."""
    logger_ka = logging.getLogger("keep_alive")
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not render_url:
        logger_ka.warning("RENDER_EXTERNAL_URL not set — self-ping disabled.")
        return
    time.sleep(30)
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
import io
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.error import (
    NetworkError,
    TimedOut,
    RetryAfter,
    BadRequest,
    TelegramError,
    Forbidden,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    TypeHandler,
    CallbackQueryHandler,
    ApplicationHandlerStop,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = ""
SB_URL: str = ""
SB_KEY: str = ""
GEMINI_API_KEY: str = ""
ADMIN_IDS: set[int] = set()
GEMINI_MODEL = "gemma-4-31b-it"
MAX_VOICE_BYTES = 20 * 1024 * 1024
MAX_INPUT_CHARS = 5_000
TTS_CHUNK_CHARS = 900
DEFAULT_SPEED = 1.0
TELE_MSG_LIMIT = 4000
USER_COOLDOWN_S = 3.0
# Grace window: valid messages arriving within 30s of restart are not dropped.
_STALE_GRACE_S = 30.0
_KHMER_RE = re.compile(r"[\u1780-\u17FF]")

# FIX: clamp speed to a safe range to avoid atempo infinite loops
_SPEED_MIN = 0.25
_SPEED_MAX = 4.0

# Separate executor pools: DB writes (blocking I/O) vs Gemini API (slow network).
_DB_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db_write")
_GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini")

# FIX: semaphore to cap concurrent Gemini API calls (avoids quota exhaustion)
_GEMINI_SEMAPHORE: asyncio.Semaphore | None = None  # init in _run_bot()

# FIX: module-level broadcast semaphore so simultaneous broadcasts
# (immediate + scheduled) don't together exceed 10 concurrent sends.
_BROADCAST_SEMAPHORE: asyncio.Semaphore | None = None  # init in _run_bot()

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
BROADCAST_WAIT_MESSAGE = 1
CHAT_WAIT_MESSAGE = 2
SCHED_WAIT_MSG = 3
SCHED_WAIT_TIME = 4
_SCHED_POLL_INTERVAL = 60
_DT_FORMATS = [
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M",
]

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
_pending_broadcast: dict[int, dict] = {}
_admin_chat_target: dict[int, int] = {}
_user_to_admin: dict[int, int] = {}
_sched_payload: dict[int, dict] = {}

# LRU-capped OrderedDict to prevent unbounded memory growth.
_USER_LAST_TTS_MAX = 10_000
_user_last_tts: OrderedDict[int, float] = OrderedDict()

def _set_last_tts(user_id: int) -> None:
    """Record TTS completion time with LRU eviction."""
    _user_last_tts.pop(user_id, None)
    _user_last_tts[user_id] = time.monotonic()
    while len(_user_last_tts) > _USER_LAST_TTS_MAX:
        _user_last_tts.popitem(last=False)

def _get_last_tts(user_id: int) -> float:
    return _user_last_tts.get(user_id, 0.0)

# ---------------------------------------------------------------------------
# Supabase + Gemini clients
# ---------------------------------------------------------------------------
supabase: Client | None = None
_gemini: genai.Client | None = None

def _init_clients() -> None:
    global supabase, _gemini, TELEGRAM_BOT_TOKEN, SB_URL, SB_KEY
    global GEMINI_API_KEY, ADMIN_IDS, GEMINI_MODEL

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    SB_URL = os.getenv("SUPABASE_URL", "")
    SB_KEY = os.getenv("SUPABASE_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

    for _aid in os.getenv("ADMIN_IDS", "").split(","):
        _aid = _aid.strip()
        if _aid.isdigit():
            ADMIN_IDS.add(int(_aid))

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    if SB_URL and SB_KEY:
        try:
            supabase = create_client(SB_URL, SB_KEY)
            logger.info("Supabase client initialised.")
        except Exception as e:
            logger.error(f"Supabase init failed: {e}")
            supabase = None
    else:
        logger.warning("Supabase env vars missing — DB features disabled.")

    if GEMINI_API_KEY:
        try:
            _gemini = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Gemini client initialised (model: {GEMINI_MODEL}).")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            _gemini = None
    else:
        logger.warning("GEMINI_API_KEY not set — OCR / transcription disabled.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOICE_MAP = {
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
}
SPEED_OPTIONS = {
    "spd_0.5": ("x0.5", 0.5),
    "spd_1.0": ("Normal", 1.0),
    "spd_1.5": ("x1.5", 1.5),
    "spd_2.0": ("x2.0", 2.0),
}
WELCOME_TEXT = (
    "🎵 សួស្តី! ខ្ញុំជា Bot បំលែងអក្សរទៅជាសំឡេង\n\n"
    "📌 វាយអក្សរភាសាណាមួយ ផ្ញើរមក Bot នឹងបំលែងដោយស្វ័យប្រវត្តិ!\n\n"
    "🌍 ភាសាដែល Support:\n"
    "🇰🇭 ភាសាខ្មែរ | 🇺🇸 English\n\n"
    "✨ មុខងារថ្មី:\n"
    "🖼️ ផ្ញើរូបភាព → Bot OCR អត្ថបទ ហើយអាន!\n"
    "🎙️ ផ្ញើ Voice → Bot Transcribe ហើយអាន!\n"
    "💬 Bot ចងចាំការសន្ទនា — អាចនិយាយ 'អានម្តងទៀត' ឬ 'បកប្រែ' បាន!\n\n"
    "⚙️ ប្រើ /myprefs ដើម្បីមើលការកំណត់របស់អ្នក\n"
    "🗑️ ប្រើ /clear ដើម្បីលុបប្រវត្តិការសន្ទនា\n\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)
BOT_TAG = "@voicekhaibot"

# ---------------------------------------------------------------------------
# FIX #11: Prefs cache — keep all mutations on the event loop to avoid
# non-atomic OrderedDict operations across threads (move_to_end + popitem).
# get_user_prefs() is called via run_in_executor for the Supabase I/O but
# the cache is now guarded by an asyncio.Lock accessed only from async code.
# ---------------------------------------------------------------------------
_PREFS_TTL = 300.0
_PREFS_MAX_SIZE = 10_000
_prefs_cache: OrderedDict[int, tuple[dict, float]] = OrderedDict()
_prefs_cache_lock: asyncio.Lock | None = None  # init in _run_bot()

def _cache_prefs_sync(user_id: int, prefs: dict) -> None:
    """Thread-safe-enough cache write (called only from the event loop via async wrapper)."""
    _prefs_cache.pop(user_id, None)
    _prefs_cache[user_id] = (prefs, time.monotonic())
    while len(_prefs_cache) > _PREFS_MAX_SIZE:
        _prefs_cache.popitem(last=False)

def _get_cached_prefs_sync(user_id: int) -> dict | None:
    entry = _prefs_cache.get(user_id)
    if entry and time.monotonic() - entry[1] < _PREFS_TTL:
        return entry[0]
    return None

def _invalidate_prefs(user_id: int) -> None:
    _prefs_cache.pop(user_id, None)

async def _async_cache_prefs(user_id: int, prefs: dict) -> None:
    """Cache prefs from async context, protected by the asyncio lock."""
    assert _prefs_cache_lock is not None
    async with _prefs_cache_lock:
        _cache_prefs_sync(user_id, prefs)

async def _async_get_cached_prefs(user_id: int) -> dict | None:
    """Read prefs cache from async context, protected by the asyncio lock."""
    assert _prefs_cache_lock is not None
    async with _prefs_cache_lock:
        return _get_cached_prefs_sync(user_id)

async def get_user_prefs_async(user_id: int) -> dict:
    """
    Async-safe version of get_user_prefs.
    Cache reads/writes happen on the event loop; Supabase I/O is off-loaded.
    """
    cached = await _async_get_cached_prefs(user_id)
    if cached is not None:
        return cached
    defaults = {"gender": "female", "speed": DEFAULT_SPEED}
    if supabase:
        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(
                _DB_EXECUTOR,
                lambda: supabase.table("user_prefs")
                    .select("gender, speed")
                    .eq("user_id", user_id)
                    .execute()
            )
            if res.data:
                row = res.data[0]
                defaults["gender"] = row.get("gender") or "female"
                raw_speed = row.get("speed")
                if raw_speed is not None:
                    defaults["speed"] = max(_SPEED_MIN, min(_SPEED_MAX, float(raw_speed)))
                else:
                    defaults["speed"] = DEFAULT_SPEED
        except Exception as e:
            logger.error(f"DB get_user_prefs_async: {e}")
    await _async_cache_prefs(user_id, defaults)
    return defaults

# Keep the sync version for any remaining executor calls (DB helpers).
def get_user_prefs(user_id: int) -> dict:
    cached = _get_cached_prefs_sync(user_id)
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
            raw_speed = row.get("speed")
            if raw_speed is not None:
                defaults["speed"] = max(_SPEED_MIN, min(_SPEED_MAX, float(raw_speed)))
            else:
                defaults["speed"] = DEFAULT_SPEED
    except Exception as e:
        logger.error(f"DB get_user_prefs: {e}")
    _cache_prefs_sync(user_id, defaults)
    return defaults

# ---------------------------------------------------------------------------
# Per-user async locks (LRU-capped, asyncio-safe)
# ---------------------------------------------------------------------------
_USER_LOCK_MAX = 5_000
_user_locks: OrderedDict[int, asyncio.Lock] = OrderedDict()

def _get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id in _user_locks:
        _user_locks.move_to_end(user_id)
        return _user_locks[user_id]
    lock = asyncio.Lock()
    _user_locks[user_id] = lock
    while len(_user_locks) > _USER_LOCK_MAX:
        _user_locks.popitem(last=False)
    return lock

# ---------------------------------------------------------------------------
# DB executor helpers
# ---------------------------------------------------------------------------
def _log_future_exception(future):
    with suppress(Exception):
        exc = future.exception()
        if exc:
            logger.error(f"DB executor task raised: {exc}", exc_info=exc)

def _submit_db(fn, *args, **kwargs):
    future = _DB_EXECUTOR.submit(fn, *args, **kwargs)
    future.add_done_callback(_log_future_exception)
    return future

# ---------------------------------------------------------------------------
# Safe Telegram send with retry
# ---------------------------------------------------------------------------
async def safe_send(coro_factory, retries: int = 3, delay: float = 2.0):
    if inspect.isawaitable(coro_factory):
        logger.warning(
            "safe_send received a raw coroutine — retries disabled. "
            "Pass a zero-arg lambda instead."
        )
        try:
            return await coro_factory
        except Exception as e:
            logger.error(f"safe_send (no-retry coroutine): {e}")
            raise

    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except RetryAfter as e:
            wait = e.retry_after + 1
            logger.warning(f"Rate-limited — sleeping {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
            last_exc = e
            if attempt == retries - 1:
                raise
        except (TimedOut, NetworkError) as e:
            last_exc = e
            if attempt < retries - 1:
                logger.warning(f"Network error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)
            else:
                raise
        except BadRequest as e:
            logger.warning(f"Telegram BadRequest (not retried): {e}")
            raise
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            raise

    if last_exc:
        logger.warning(f"safe_send giving up after {retries} attempts: {last_exc}")
    return None

# ---------------------------------------------------------------------------
# Telegram-safe text pagination
# ---------------------------------------------------------------------------
def _paginate_plain(text: str, limit: int = TELE_MSG_LIMIT) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    pages = []
    while text:
        if len(text) <= limit:
            pages.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        pages.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return pages

# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------
_TMP_PREFIX = "tgbot_"

def _make_temp_ogg() -> str:
    fd, path = tempfile.mkstemp(suffix=".ogg", prefix=_TMP_PREFIX, dir="/tmp")
    os.close(fd)
    return path

def _make_temp_img(suffix: str = ".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=_TMP_PREFIX, dir="/tmp")
    os.close(fd)
    return path

def _sweep_stale_temps() -> None:
    for pattern in [
        f"/tmp/{_TMP_PREFIX}*.ogg",
        f"/tmp/{_TMP_PREFIX}*.jpg",
        f"/tmp/{_TMP_PREFIX}*.png",
        f"/tmp/{_TMP_PREFIX}*.webp",
    ]:
        for f in glob.glob(pattern):
            with suppress(OSError):
                os.remove(f)
                logger.info(f"Swept stale temp file: {f}")

async def _periodic_temp_sweep(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=3600.0)
        except asyncio.TimeoutError:
            pass
        if not stop_event.is_set():
            _sweep_stale_temps()

# ---------------------------------------------------------------------------
# Database helpers — user prefs
# ---------------------------------------------------------------------------
def sync_user_data(user):
    """Upsert user record asynchronously (fire-and-forget)."""
    if not supabase:
        return
    def _run():
        with suppress(Exception):
            supabase.table("user_prefs").upsert(
                {"user_id": user.id, "username": user.username or user.first_name},
                on_conflict="user_id",
            ).execute()
    _submit_db(_run)

def _paginated_fetch(select_fields: str) -> list[dict]:
    if not supabase:
        return []
    try:
        all_rows, page, page_size = [], 0, 1000
        while True:
            res = (
                supabase.table("user_prefs")
                .select(select_fields)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            batch = res.data or []
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return all_rows
    except Exception as e:
        logger.error(f"DB _paginated_fetch({select_fields!r}): {e}")
        return []

def get_all_user_ids() -> list[int]:
    return [row["user_id"] for row in _paginated_fetch("user_id")]

def get_all_users_with_names() -> list[dict]:
    return _paginated_fetch("user_id, username")

def user_exists_in_db(user_id: int) -> bool:
    if not supabase:
        return False
    try:
        res = (
            supabase.table("user_prefs")
            .select("user_id")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        logger.error(f"DB user_exists_in_db: {e}")
        return False

def update_user_gender(user_id: int, gender: str):
    _invalidate_prefs(user_id)
    if not supabase:
        return
    def _run():
        with suppress(Exception):
            supabase.table("user_prefs").update({"gender": gender}).eq(
                "user_id", user_id
            ).execute()
    _submit_db(_run)

def update_user_speed(user_id: int, speed: float):
    # FIX: clamp speed before storing
    speed = round(max(_SPEED_MIN, min(_SPEED_MAX, speed)), 4)
    _invalidate_prefs(user_id)
    if not supabase:
        return
    def _run():
        try:
            supabase.table("user_prefs").update({"speed": speed}).eq(
                "user_id", user_id
            ).execute()
        except Exception as e:
            if "does not exist" in str(e):
                logger.error(
                    "speed column missing — run: "
                    "ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
                )
            else:
                logger.error(f"update_user_speed error: {e}")
    _submit_db(_run)

# Track whether we've already warned about RLS so we don't spam logs
_rls_warned = False

def save_text_cache(
    msg_id: int,
    text: str,
    chat_id: int = 0,
    user_id: int = None,
    username: str = None,
):
    if not supabase:
        return
    def _run():
        global _rls_warned
        try:
            payload = {"message_id": msg_id, "chat_id": chat_id, "original_text": text}
            if user_id is not None:
                payload["user_id"] = user_id
            if username is not None:
                payload["username"] = username
            supabase.table("text_cache").upsert(
                payload, on_conflict="chat_id,message_id"
            ).execute()
        except Exception as e:
            err_str = str(e)
            if "42501" in err_str or "row-level security" in err_str.lower():
                if not _rls_warned:
                    _rls_warned = True
                    logger.error(
                        "text_cache RLS policy blocking inserts. "
                        "Fix: run in Supabase SQL editor:\n"
                        "  ALTER TABLE text_cache DISABLE ROW LEVEL SECURITY;\n"
                        "  -- or --\n"
                        "  CREATE POLICY \"service_role_all\" ON text_cache "
                        "FOR ALL TO service_role USING (true) WITH CHECK (true);"
                    )
            else:
                logger.error(f"save_text_cache error: {e}")
    _submit_db(_run)

def get_text_cache(msg_id: int, chat_id: int = 0) -> str | None:
    if not supabase:
        return None
    try:
        res = (
            supabase.table("text_cache")
            .select("original_text")
            .eq("message_id", msg_id)
            .eq("chat_id", chat_id)
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
        logger.info("speed column present.")
    except Exception as e:
        err = str(e).lower()
        if "does not exist" in err or "column" in err:
            logger.warning(
                "speed column missing. Run in Supabase SQL editor:\n"
                "  ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
            )
        else:
            logger.error(f"ensure_speed_column unexpected error: {e}")

# ---------------------------------------------------------------------------
# Database helpers — conversation history
# ---------------------------------------------------------------------------
CONV_HISTORY_LIMIT = 10
CONV_CONTEXT_MAX_CHARS = 3000
CONV_RESOLVE_TIMEOUT_S = 15

# In-memory history cache
_HIST_CACHE_MAX_USERS = 5_000
_HIST_CACHE_TURNS = 10
_hist_cache: OrderedDict[int, deque] = OrderedDict()

def _hist_cache_append(user_id: int, role: str, content: str) -> None:
    """Append one turn to the in-memory history cache with LRU eviction."""
    if user_id not in _hist_cache:
        while len(_hist_cache) >= _HIST_CACHE_MAX_USERS:
            _hist_cache.popitem(last=False)
        _hist_cache[user_id] = deque(maxlen=_HIST_CACHE_TURNS)
    _hist_cache.move_to_end(user_id)
    _hist_cache[user_id].append({"role": role, "content": content})

def _hist_cache_get(user_id: int) -> list[dict] | None:
    """Return cached turns (oldest-first) or None if user is not warmed."""
    d = _hist_cache.get(user_id)
    if d is None:
        return None
    _hist_cache.move_to_end(user_id)
    return list(d)

def _hist_cache_clear(user_id: int) -> None:
    _hist_cache.pop(user_id, None)

def db_history_append(user_id: int, role: str, content: str) -> None:
    """Fire-and-forget: write one conversation turn to Supabase."""
    if not supabase:
        return
    def _run():
        with suppress(Exception):
            supabase.table("conversation_history").insert(
                {"user_id": user_id, "role": role, "content": content}
            ).execute()
    _submit_db(_run)

def db_history_fetch(user_id: int, limit: int = CONV_HISTORY_LIMIT) -> list[dict]:
    """Return the last `limit` turns in chronological order (oldest first)."""
    if not supabase:
        return []
    try:
        res = (
            supabase.table("conversation_history")
            .select("role, content")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = res.data or []
        rows.reverse()
        return rows
    except Exception as e:
        logger.error(f"db_history_fetch error: {e}")
        return []

def db_history_clear(user_id: int) -> None:
    """Delete all conversation history rows for this user."""
    if not supabase:
        return
    def _run():
        with suppress(Exception):
            supabase.table("conversation_history").delete().eq(
                "user_id", user_id
            ).execute()
    _submit_db(_run)

def _build_context_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for row in history:
        prefix = "User" if row["role"] == "user" else "Bot"
        lines.append(f"{prefix}: {row['content']}")
    block = "\n".join(lines)
    if len(block) > CONV_CONTEXT_MAX_CHARS:
        block = block[-CONV_CONTEXT_MAX_CHARS:]
        nl = block.find("\n")
        if nl != -1:
            block = block[nl + 1:]
    return block

def record_turn(user_id: int, role: str, content: str) -> None:
    """Record one conversation turn to both in-memory cache and Supabase."""
    _hist_cache_append(user_id, role, content)
    db_history_append(user_id, role, content)

# ---------------------------------------------------------------------------
# Conversation history: Gemini-powered text resolution
# ---------------------------------------------------------------------------
async def resolve_tts_text(
    user_id: int,
    raw_text: str,
    loop: asyncio.AbstractEventLoop,
) -> str:
    if not _gemini:
        return raw_text

    history = _hist_cache_get(user_id)
    if history is None:
        history = await loop.run_in_executor(_DB_EXECUTOR, db_history_fetch, user_id)
        for row in history:
            _hist_cache_append(user_id, row["role"], row["content"])

    if not history:
        return raw_text

    context_block = _build_context_block(history)
    system_prompt = (
        "You are a text pre-processor for a Khmer/English text-to-speech bot. "
        "You receive a conversation history and the user's latest message. "
        "Your ONLY job is to output the exact text that should be spoken aloud. "
        "Output nothing else — no labels, no explanation, no markdown.\n\n"
        "Rules:\n"
        "1. If the message is a normal sentence or paragraph to be read, output it verbatim.\n"
        "2. If the message is a follow-up like 'read that again' or 'អានម្តងទៀត', "
        "output the last Bot turn verbatim.\n"
        "3. If the message asks to translate something (e.g. 'translate that to Khmer', "
        "'បកប្រែជាភាសាខ្មែរ'), output the translated text in the target language.\n"
        "4. If the message references previous content using pronouns like 'that', 'it', "
        "'the last thing', resolve the reference and output the resolved text.\n"
        "5. If the message asks 'what did I say?' or similar, output the last User turn.\n"
        "6. Preserve the original language (Khmer or English) unless translation is requested.\n"
        "7. If you cannot determine what to speak, output the user's message verbatim."
    )
    combined_prompt = (
        f"{system_prompt}\n\n"
        f"Conversation history:\n{context_block}\n\n"
        f"User's new message: {raw_text}\n\n"
        "Output only the text to speak:"
    )

    async def _guarded_call():
        assert _GEMINI_SEMAPHORE is not None
        async with _GEMINI_SEMAPHORE:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=combined_prompt,
                )
            return await loop.run_in_executor(_GEMINI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(
            _guarded_call(),
            timeout=CONV_RESOLVE_TIMEOUT_S,
        )
        resolved = (response.text or "").strip()
        return resolved if resolved else raw_text
    except asyncio.TimeoutError:
        logger.warning(f"resolve_tts_text timed out for user {user_id} — using raw text")
        return raw_text
    except Exception as e:
        logger.warning(f"resolve_tts_text error for user {user_id}: {e} — using raw text")
        return raw_text

# ---------------------------------------------------------------------------
# Database helpers — scheduled broadcasts
# ---------------------------------------------------------------------------
def db_sched_insert(payload: dict, admin_id: int, broadcast_at: datetime) -> dict:
    if not supabase:
        raise RuntimeError("Supabase not configured.")
    row = {
        "admin_id": admin_id,
        "photo_file_id": payload.get("photo_file_id"),
        "caption": payload.get("caption"),
        "plain_text": payload.get("text"),
        "broadcast_at": broadcast_at.isoformat(),
        "status": "pending",
    }
    res = supabase.table("scheduled_broadcasts").insert(row).execute()
    return res.data[0]

def db_sched_fetch_due() -> list[dict]:
    if not supabase:
        return []
    try:
        now = datetime.now(timezone.utc).isoformat()
        res = (
            supabase.table("scheduled_broadcasts")
            .select("*")
            .eq("status", "pending")
            .lte("broadcast_at", now)
            .order("broadcast_at")
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"db_sched_fetch_due: {e}")
        return []

def db_sched_claim(row_id: int) -> bool:
    if not supabase:
        return False
    try:
        res = (
            supabase.table("scheduled_broadcasts")
            .update({"status": "sending"})
            .eq("id", row_id)
            .eq("status", "pending")
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        logger.error(f"db_sched_claim #{row_id}: {e}")
        return False

def db_sched_set_status(row_id: int, status: str, **extra) -> None:
    if not supabase:
        return
    with suppress(Exception):
        patch = {"status": status, **extra}
        supabase.table("scheduled_broadcasts").update(patch).eq("id", row_id).execute()

def db_sched_fetch_admin_pending(admin_id: int) -> list[dict]:
    if not supabase:
        return []
    try:
        res = (
            supabase.table("scheduled_broadcasts")
            .select("id, broadcast_at, plain_text, caption, photo_file_id, status")
            .eq("admin_id", admin_id)
            .eq("status", "pending")
            .order("broadcast_at")
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"db_sched_fetch_admin_pending: {e}")
        return []

def db_sched_fetch_one(row_id: int) -> dict | None:
    if not supabase:
        return None
    try:
        res = (
            supabase.table("scheduled_broadcasts")
            .select("*")
            .eq("id", row_id)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"db_sched_fetch_one: {e}")
        return None

# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------
def _parse_dt(text: str) -> datetime | None:
    text = text.strip()
    for fmt in _DT_FORMATS:
        try:
            naive = datetime.strptime(text, fmt)
            return naive.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")

# ---------------------------------------------------------------------------
# Keyboard helpers — scheduled broadcast
# ---------------------------------------------------------------------------
def get_sched_confirm_kb(row_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ បញ្ជាក់ Schedule", callback_data=f"sched_ok:{row_id}"),
            InlineKeyboardButton("❌ បោះបង់", callback_data=f"sched_no:{row_id}"),
        ]]
    )

def get_schedules_list_kb(
    rows: list[dict], page: int, page_size: int = 5
) -> InlineKeyboardMarkup:
    total = max(1, (len(rows) + page_size - 1) // page_size)
    chunk = rows[page * page_size : (page + 1) * page_size]
    kbd_rows = []
    for r in chunk:
        try:
            dt_str = _fmt_dt(datetime.fromisoformat(r["broadcast_at"]))
        except Exception:
            dt_str = str(r.get("broadcast_at", "?"))
        label = f"#{r['id']}  {dt_str}"
        kbd_rows.append([InlineKeyboardButton(label, callback_data=f"sched_view:{r['id']}")])
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"sched_page:{page - 1}"))
    nav.append(InlineKeyboardButton(f"{page + 1}/{total}", callback_data="sched_noop"))
    if page < total - 1:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"sched_page:{page + 1}"))
    if nav:
        kbd_rows.append(nav)
    kbd_rows.append([InlineKeyboardButton("❌ បិទ", callback_data="sched_close")])
    return InlineKeyboardMarkup(kbd_rows)

# ---------------------------------------------------------------------------
# Status timer (animated spinner messages)
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
_OCR_FRAMES = [
    "🔍 កំពុង OCR រូបភាព ·",
    "🔍 កំពុង OCR រូបភាព ··",
    "🔍 កំពុង OCR រូបភាព ···",
    "🔍 កំពុង OCR រូបភាព ····",
]

async def send_status_timer(
    chat_id: int, bot, stop_event: asyncio.Event, frames=None
):
    frames = frames or _STATUS_FRAMES
    msg = None
    try:
        msg = await safe_send(lambda: bot.send_message(chat_id=chat_id, text=frames[0]))
        if msg is None:
            return
        frame_idx = 1
        while not stop_event.is_set():
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            if stop_event.is_set():
                break
            current_frame = frames[frame_idx % len(frames)]
            with suppress(BadRequest, TelegramError):
                await msg.edit_text(current_frame)
            frame_idx += 1
    except (asyncio.CancelledError, Exception) as e:
        if not isinstance(e, asyncio.CancelledError):
            logger.warning(f"Status timer error: {e}")
    finally:
        if msg:
            with suppress(Exception):
                await msg.delete()

async def _stop_timer(stop_event: asyncio.Event, timer_task: asyncio.Task):
    stop_event.set()
    if not timer_task.done():
        timer_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await timer_task

# ---------------------------------------------------------------------------
# Audio pipeline
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _build_atempo_chain(speed: float) -> str:
    speed = round(max(_SPEED_MIN, min(_SPEED_MAX, speed)), 4)
    if abs(speed - 1.0) < 1e-6:
        return "atempo=1.0"
    stages: list[str] = []
    r = speed
    if r < 1.0:
        while r < 0.5 - 1e-9:
            stages.append("atempo=0.5")
            r = round(r / 0.5, 6)
            if r < 1e-6:
                break
        stages.append(f"atempo={max(0.5, r):.6f}")
    else:
        while r > 2.0 + 1e-9:
            stages.append("atempo=2.0")
            r = round(r / 2.0, 6)
        stages.append(f"atempo={min(2.0, r):.6f}")
    return ",".join(stages)

async def generate_voice(text: str, gender: str, speed: float, output_path: str) -> bytes:
    """Generate TTS audio and return the raw OGG/Opus bytes."""
    text = text.strip()
    if not text:
        raise ValueError("generate_voice: text must not be empty")

    khmer_chars = len(_KHMER_RE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())
    is_khmer = khmer_chars > (total_alpha * 0.3) if total_alpha else False
    voice = VOICE_MAP["km" if is_khmer else "en"][gender]

    mp3_chunks: list[bytes] = []
    try:
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])
    except Exception as e:
        raise RuntimeError(f"edge-tts failed: {e}") from e

    mp3_data = b"".join(mp3_chunks)
    if not mp3_data:
        raise RuntimeError("edge-tts returned empty audio")

    speed_key = round(max(_SPEED_MIN, min(_SPEED_MAX, speed)), 4)
    af = _build_atempo_chain(speed_key) if abs(speed_key - DEFAULT_SPEED) > 1e-4 else None

    cmd = [_FFMPEG_EXE, "-y", "-f", "mp3", "-i", "pipe:0"]
    if af:
        cmd += ["-filter:a", af]
    cmd += ["-c:a", "libopus", "-b:a", "32k", output_path]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr_data = await proc.communicate(input=mp3_data)
    if proc.returncode != 0:
        stderr_snippet = (stderr_data or b"").decode(errors="replace")[-400:]
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode}): {stderr_snippet}")

    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(
            None, lambda: open(output_path, "rb").read()
        )
    except OSError as e:
        raise RuntimeError(f"Failed to read output audio file: {e}") from e
    return audio_bytes

# ---------------------------------------------------------------------------
# Gemini transcription
# ---------------------------------------------------------------------------
async def transcribe_voice(ogg_path: str) -> str:
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(
            None, lambda: open(ogg_path, "rb").read()
        )
    except OSError as e:
        raise RuntimeError(f"Cannot read voice file: {e}") from e

    prompt = (
        "Transcribe this audio exactly as spoken. "
        "Output ONLY the transcribed text — no labels, no explanation. "
        "Support both Khmer and English."
    )

    assert _GEMINI_SEMAPHORE is not None

    async def _guarded_call():
        async with _GEMINI_SEMAPHORE:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                        prompt,
                    ],
                )
            return await loop.run_in_executor(_GEMINI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(_guarded_call(), timeout=60)
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini transcription timed out after 60s")

# ---------------------------------------------------------------------------
# Text chunking for TTS
# ---------------------------------------------------------------------------
def _split_text_chunks(text: str, max_chars: int = TTS_CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentence_re = re.compile(r"(?<=[។!\?\.。])\s*")
    sentences = [s for s in sentence_re.split(text) if s.strip()]
    chunks, current = [], ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            if " " in sent:
                for word in sent.split():
                    if not word:
                        continue
                    if len(current) + len(word) + (1 if current else 0) > max_chars:
                        if current:
                            chunks.append(current.strip())
                        current = word
                    else:
                        current = (current + " " + word).strip() if current else word
            else:
                pos = 0
                while pos < len(sent):
                    space = max_chars - len(current)
                    current += sent[pos : pos + space]
                    pos += space
                    if len(current) >= max_chars:
                        chunks.append(current)
                        current = ""
        elif len(current) + len(sent) + (1 if current else 0) > max_chars:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c and c.strip()]

# ---------------------------------------------------------------------------
# Detect image MIME type from magic bytes
# ---------------------------------------------------------------------------
def _detect_image_mime(path: str) -> str:
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
        if header[:2] == b"\xff\xd8":
            return "image/jpeg"
        if header[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
    except Exception:
        pass
    return "image/jpeg"

# ---------------------------------------------------------------------------
# Gemini OCR
# ---------------------------------------------------------------------------
async def ocr_image(image_path: str, mime_type: str = "image/jpeg") -> str:
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    loop = asyncio.get_running_loop()
    try:
        image_bytes = await loop.run_in_executor(
            None, lambda: open(image_path, "rb").read()
        )
    except OSError as e:
        raise RuntimeError(f"Cannot read image file: {e}") from e

    prompt = (
        "You are an OCR engine. Extract EVERY line of text visible in this image, "
        "from top to bottom, left to right. "
        "Do NOT skip, truncate, summarise, or omit any line — even if the text is long. "
        "Do NOT add any labels, headers, explanations, bullet points, or markdown. "
        "Output ONLY the raw extracted text, preserving original line breaks exactly. "
        "Support both Khmer (ខ្មែរ) and English with full Unicode accuracy. "
        "If there is truly no text, output only the single word: NOTEXT"
    )

    assert _GEMINI_SEMAPHORE is not None

    async def _guarded_call():
        async with _GEMINI_SEMAPHORE:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
            return await loop.run_in_executor(_GEMINI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(_guarded_call(), timeout=60)
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini OCR timed out after 60s")

# ---------------------------------------------------------------------------
# Cooldown check — shared by all TTS entry points
# ---------------------------------------------------------------------------
async def _check_cooldown(reply_target, user_id: int) -> bool:
    """Returns True if the request is within cooldown (warning sent)."""
    lock = _get_user_lock(user_id)
    if lock.locked():
        await safe_send(
            lambda: reply_target.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ...")
        )
        return True
    now = time.monotonic()
    last = _get_last_tts(user_id)
    if now - last < USER_COOLDOWN_S:
        rem = round(USER_COOLDOWN_S - (now - last), 1)
        await safe_send(
            lambda r=rem: reply_target.reply_text(
                f"⏳ សូមរង់ចាំ {r}s មុននឹងផ្ញើម្តងទៀត។"
            )
        )
        return True
    return False

# ---------------------------------------------------------------------------
# Paged TTS delivery
# ---------------------------------------------------------------------------
async def _deliver_paged_tts(
    chat_id: int,
    bot,
    text: str,
    gender: str,
    speed: float,
    user_id: int,
    username: str,
) -> None:
    chunks = _split_text_chunks(text)
    if not chunks:
        await safe_send(
            lambda: bot.send_message(chat_id=chat_id, text="❌ រកអត្ថបទមិនឃើញ។")
        )
        return

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        if not chunk or not chunk.strip():
            logger.warning(f"_deliver_paged_tts: skipping empty chunk {i}/{total}")
            continue
        file_path = _make_temp_ogg()
        try:
            audio_bytes = await generate_voice(chunk, gender, speed, file_path)
            caption = f"🗣️ {BOT_TAG}  [{i}/{total}]"
            sent = await safe_send(
                lambda ab=audio_bytes, cap=caption: bot.send_voice(
                    chat_id=chat_id,
                    voice=io.BytesIO(ab),
                    caption=cap,
                    reply_markup=get_main_kb(gender),
                )
            )
            if sent:
                save_text_cache(
                    sent.message_id,
                    chunk,
                    chat_id=chat_id,
                    user_id=user_id,
                    username=username,
                )
        except Exception as e:
            logger.error(f"paged TTS chunk {i}/{total} error: {e}")
            ci, ct = i, total
            await safe_send(
                lambda ci=ci, ct=ct: bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ មានបញ្ហាក្នុង chunk {ci}/{ct}។",
                )
            )
        finally:
            _cleanup(file_path)
        if i < total:
            await asyncio.sleep(0.3)

    # Cooldown set AFTER all chunks delivered
    _set_last_tts(user_id)

# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------
def get_main_kb(gender: str) -> InlineKeyboardMarkup:
    f_btn = "👩 សំឡេងស្រី" + (" ✅" if gender == "female" else "")
    m_btn = "👨 សំឡេងប្រុស" + (" ✅" if gender == "male" else "")
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(f_btn, callback_data="tg_female"),
                InlineKeyboardButton(m_btn, callback_data="tg_male"),
            ],
            [InlineKeyboardButton("🎚️ ល្បឿនសំឡេង", callback_data="show_speed")],
        ]
    )

def get_speed_kb(current_speed: float) -> InlineKeyboardMarkup:
    speed_row = []
    for cb, (lbl, val) in SPEED_OPTIONS.items():
        mark = " ✅" if abs(val - current_speed) < 0.01 else ""
        speed_row.append(InlineKeyboardButton(lbl + mark, callback_data=cb))
    return InlineKeyboardMarkup(
        [
            speed_row,
            [InlineKeyboardButton("🔙 ត្រឡប់", callback_data="hide_speed")],
        ]
    )

def get_transcription_kb(transcript_msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("📢 AI អាន", callback_data=f"tts_transcript:{transcript_msg_id}"),
            InlineKeyboardButton("🗑️ លុប", callback_data=f"del_transcript:{transcript_msg_id}"),
        ]]
    )

def get_ocr_confirm_kb(msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("▶️ អាន", callback_data=f"doc_read:{msg_id}"),
            InlineKeyboardButton("🗑️ លុប", callback_data=f"doc_del:{msg_id}"),
        ]]
    )

def get_broadcast_confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ បញ្ជាក់ Broadcast", callback_data="bc_confirm"),
            InlineKeyboardButton("❌ បោះបង់", callback_data="bc_cancel"),
        ]]
    )

def get_users_page_kb(
    users: list[dict], page: int, page_size: int = 8
) -> InlineKeyboardMarkup:
    total_pages = max(1, (len(users) + page_size - 1) // page_size)
    chunk = users[page * page_size : page * page_size + page_size]
    rows = []
    for u in chunk:
        uid = u["user_id"]
        uname = (u.get("username") or str(uid))[:20]
        rows.append(
            [InlineKeyboardButton(f"👤 {uname}  ({uid})", callback_data=f"chat_open:{uid}")]
        )
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
            with suppress(OSError):
                os.remove(p)

# ---------------------------------------------------------------------------
# Admin guard
# ---------------------------------------------------------------------------
def admin_only(handler):
    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        if uid not in ADMIN_IDS:
            if update.message:
                await safe_send(
                    lambda: update.message.reply_text(
                        "⛔ អ្នកមិនមានសិទ្ធិប្រើពាក្យបញ្ជានេះទេ។"
                    )
                )
            return
        return await handler(update, context)
    return wrapper

def _is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

# ---------------------------------------------------------------------------
# Chat session helpers
# ---------------------------------------------------------------------------
def _open_session(admin_id: int, target_id: int) -> None:
    old_target = _admin_chat_target.get(admin_id)
    if old_target is not None:
        _user_to_admin.pop(old_target, None)
    _admin_chat_target[admin_id] = target_id
    _user_to_admin[target_id] = admin_id

def _close_session(admin_id: int) -> int | None:
    target_id = _admin_chat_target.pop(admin_id, None)
    if target_id is not None:
        _user_to_admin.pop(target_id, None)
    return target_id

def _get_admin_for_user(user_id: int) -> int | None:
    return _user_to_admin.get(user_id)

# ===========================================================================
# FIX #7: Shared broadcast delivery helper (replaces duplicated code in
# _do_broadcast and _fire_scheduled_broadcast).
# ===========================================================================
async def _run_broadcast_to_all(
    bot,
    admin_id: int,
    pending: dict,
    label: str = "Broadcast",
) -> None:
    """
    Send `pending` to every registered user.
    `label` is used in progress/report messages (e.g. "Broadcast" or "Scheduled #42").
    """
    assert _BROADCAST_SEMAPHORE is not None

    loop = asyncio.get_running_loop()
    user_ids = await loop.run_in_executor(None, get_all_user_ids)
    total = len(user_ids)
    if total == 0:
        await safe_send(
            lambda: bot.send_message(
                chat_id=admin_id,
                text=f"⚠️ {label}: មិនមានអ្នកប្រើប្រាស់ registered ណាមួយទេ។",
            )
        )
        return

    sent = failed = blocked = 0
    photo_file_id = pending.get("photo_file_id")
    caption = pending.get("caption") or ""
    plain_text = pending.get("text") or ""

    progress_msg = await safe_send(
        lambda: bot.send_message(
            chat_id=admin_id, text=f"📡 {label} — កំពុង Broadcast ទៅ {total} នាក់..."
        )
    )

    async def _send_one(uid: int) -> str:
        async with _BROADCAST_SEMAPHORE:
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
                        await bot.send_message(
                            chat_id=uid, text=plain_text, parse_mode="HTML"
                        )
                    return "sent"
                except Forbidden:
                    return "blocked"
                except RetryAfter as e:
                    await asyncio.sleep(e.retry_after + 1)
                    if attempt == 1:
                        return "failed"
                except Exception as e:
                    logger.error(f"{label} error uid={uid} attempt={attempt}: {e}")
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
        await asyncio.sleep(0.034)
        if (i + 1) % 25 == 0 and progress_msg:
            with suppress(Exception):
                pct = int((i + 1) / total * 100) if total else 0
                await progress_msg.edit_text(
                    f"📡 {label}: {pct}% ({i + 1}/{total})\n"
                    f"✅ {sent}  🚫 {blocked}  ❌ {failed}"
                )

    report = (
        f"✅ <b>{label} រួចរាល់!</b>\n\n"
        f"👥 សរុប: {total}\n"
        f"📨 បានផ្ញើ: {sent}\n"
        f"🚫 Blocked: {blocked}\n"
        f"❌ Failed: {failed}"
    )
    try:
        if progress_msg:
            await safe_send(lambda: progress_msg.edit_text(report, parse_mode="HTML"))
        else:
            await safe_send(
                lambda: bot.send_message(chat_id=admin_id, text=report, parse_mode="HTML")
            )
    except Exception as e:
        logger.error(f"{label} report error: {e}")

    return sent, failed, blocked

# ===========================================================================
# BROADCAST (immediate)
# ===========================================================================
@admin_only
async def broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _pending_broadcast.pop(update.effective_user.id, None)
    context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
    await safe_send(
        lambda: update.message.reply_text(
            "📡 <b>Admin Broadcast</b>\n\n"
            "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Broadcast ។\n"
            "👉 អាចផ្ញើរូបភាព + Caption រួមគ្នា ឬ តែ text ។\n\n"
            "វាយ /cancel ដើម្បីបោះបង់។",
            parse_mode="HTML",
        )
    )

# FIX #6: Added @admin_only decorator (was missing, relied on manual _is_admin check).
@admin_only
async def broadcast_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if context.user_data.get("bc_state") != BROADCAST_WAIT_MESSAGE:
        return

    msg = update.message
    photo_file_id: str | None = None
    caption_text: str | None = None
    plain_text: str | None = None

    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text = msg.caption or ""
    elif msg.text:
        plain_text = msg.text.strip()
        if not plain_text:
            await safe_send(lambda: msg.reply_text("⚠️ អត្ថបទមិនអាចទទេបាន។ សូមវាយសារ។"))
            return
    else:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return

    _pending_broadcast[user_id] = {
        "photo_file_id": photo_file_id,
        "caption": caption_text,
        "text": plain_text,
    }

    if photo_file_id:
        preview_cap = html.escape(caption_text) if caption_text else "<i>(គ្មាន Caption)</i>"
        await safe_send(
            lambda: msg.reply_photo(
                photo=photo_file_id,
                caption=(
                    f"👁️ <b>Preview Broadcast</b>\n\n{preview_cap}\n\n"
                    "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?"
                ),
                parse_mode="HTML",
                reply_markup=get_broadcast_confirm_kb(),
            )
        )
    else:
        await safe_send(
            lambda: msg.reply_text(
                f"👁️ <b>Preview Broadcast</b>\n\n{html.escape(plain_text)}\n\n"
                "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?",
                parse_mode="HTML",
                reply_markup=get_broadcast_confirm_kb(),
            )
        )

async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    if data == "bc_cancel":
        _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text("❌ Broadcast បានបោះបង់។"))
        return

    if data == "bc_confirm":
        pending = _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        if not pending:
            await safe_send(
                lambda: query.message.reply_text(
                    "⚠️ រកទិន្នន័យ Broadcast មិនឃើញ។ សូមចាប់ផ្ដើមថ្មី។"
                )
            )
            return
        context.application.create_task(
            _run_broadcast_to_all(context.bot, user_id, pending, label="Broadcast")
        )

# ===========================================================================
# SCHEDULED BROADCAST
# ===========================================================================
@admin_only
async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    _sched_payload.pop(admin_id, None)
    context.user_data["sched_state"] = SCHED_WAIT_MSG
    await safe_send(
        lambda: update.message.reply_text(
            "📅 <b>Scheduled Broadcast</b>\n\n"
            "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Schedule ។\n\n"
            "វាយ /cancel ដើម្បីបោះបង់។",
            parse_mode="HTML",
        )
    )

@admin_only
async def cmd_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, db_sched_fetch_admin_pending, admin_id)
    if not rows:
        await safe_send(
            lambda: update.message.reply_text("📭 មិនមាន Scheduled Broadcast ណាមួយទេ។")
        )
        return
    await safe_send(
        lambda: update.message.reply_text(
            f"📋 <b>Scheduled Broadcasts ({len(rows)} pending)</b>\n"
            "ចុចលើ Schedule ដើម្បីមើលលម្អិត ឬ Cancel ។",
            parse_mode="HTML",
            reply_markup=get_schedules_list_kb(rows, page=0),
        )
    )

@admin_only
async def cmd_cancelschedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    args = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(
            lambda: update.message.reply_text(
                "❌ Usage: /cancelschedule &lt;id&gt;\nឬប្រើ /schedules ដើម្បីជ្រើស។",
                parse_mode="HTML",
            )
        )
        return
    row_id = int(args[0])
    loop = asyncio.get_running_loop()
    row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
    if not row:
        await safe_send(lambda: update.message.reply_text(f"❌ រកមិនឃើញ Schedule #{row_id}។"))
        return
    if row["admin_id"] != admin_id:
        await safe_send(lambda: update.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
        return
    if row["status"] != "pending":
        st = row["status"]
        await safe_send(
            lambda: update.message.reply_text(
                f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
                parse_mode="HTML",
            )
        )
        return
    await loop.run_in_executor(None, db_sched_set_status, row_id, "cancelled")
    await safe_send(
        lambda: update.message.reply_text(
            f"✅ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
        )
    )

async def _handle_sched_content(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        return False
    if context.user_data.get("sched_state") != SCHED_WAIT_MSG:
        return False

    msg = update.message
    photo_file_id: str | None = None
    caption_text: str | None = None
    plain_text: str | None = None

    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text = msg.caption or ""
    elif msg.text:
        plain_text = msg.text.strip()
        if not plain_text:
            await safe_send(
                lambda: msg.reply_text("⚠️ អត្ថបទមិនអាចទទេបាន។ សូមវាយសារ ឬ ផ្ញើរូបភាព។")
            )
            return True
    else:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return True

    _sched_payload[user_id] = {
        "photo_file_id": photo_file_id,
        "caption": caption_text,
        "text": plain_text,
    }
    context.user_data["sched_state"] = SCHED_WAIT_TIME
    await safe_send(
        lambda: msg.reply_text(
            "🕐 <b>ពេលវេលា Broadcast (UTC)</b>\n\n"
            "វាយកាលបរិច្ឆេទ និងម៉ោង ។\n"
            "ទម្រង់: <code>YYYY-MM-DD HH:MM</code>\n"
            "ឧទាហរណ៍: <code>2025-12-25 09:00</code>",
            parse_mode="HTML",
        )
    )
    return True

async def _handle_sched_datetime(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        return False
    if context.user_data.get("sched_state") != SCHED_WAIT_TIME:
        return False

    msg = update.message
    if not msg.text:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើ Text ពេលវេលា (UTC)។"))
        return True

    broadcast_at = _parse_dt(msg.text)
    if broadcast_at is None:
        await safe_send(
            lambda: msg.reply_text(
                "❌ ទម្រង់ពេលវេលាខុស។\nឧទាហរណ៍ត្រឹមត្រូវ: <code>2025-12-25 09:00</code>",
                parse_mode="HTML",
            )
        )
        return True

    now = datetime.now(timezone.utc)
    if broadcast_at <= now:
        await safe_send(
            lambda: msg.reply_text(
                "❌ ពេលវេលាត្រូវតែជាអនាគត (UTC) ។\n"
                f"ឥឡូវ: <code>{_fmt_dt(now)}</code>",
                parse_mode="HTML",
            )
        )
        return True

    payload = _sched_payload.pop(user_id, None)
    if not payload:
        context.user_data.pop("sched_state", None)
        await safe_send(
            lambda: msg.reply_text(
                "❌ រកទិន្នន័យ Schedule មិនឃើញ (session expired)។\n"
                "សូមចាប់ផ្ដើម /schedule ម្តងទៀត។"
            )
        )
        return True

    context.user_data.pop("sched_state", None)
    loop = asyncio.get_running_loop()
    try:
        row = await loop.run_in_executor(None, db_sched_insert, payload, user_id, broadcast_at)
    except Exception as e:
        logger.error(f"db_sched_insert failed: {e}")
        await safe_send(
            lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ Save Schedule ។ សូមព្យាយាមម្តងទៀត។")
        )
        return True

    row_id = row["id"]
    dt_str = _fmt_dt(broadcast_at)

    if payload.get("photo_file_id"):
        cap_preview = (
            html.escape(payload["caption"])
            if payload.get("caption")
            else "<i>(គ្មាន Caption)</i>"
        )
        await safe_send(
            lambda: msg.reply_photo(
                photo=payload["photo_file_id"],
                caption=(
                    f"📅 <b>Preview Schedule #{row_id}</b>\n"
                    f"⏰ {dt_str}\n\n{cap_preview}"
                ),
                parse_mode="HTML",
                reply_markup=get_sched_confirm_kb(row_id),
            )
        )
    else:
        await safe_send(
            lambda: msg.reply_text(
                f"📅 <b>Preview Schedule #{row_id}</b>\n"
                f"⏰ {dt_str}\n\n"
                f"{html.escape(payload.get('text') or '')}",
                parse_mode="HTML",
                reply_markup=get_sched_confirm_kb(row_id),
            )
        )
    return True

async def sched_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    loop = asyncio.get_running_loop()

    if data.startswith("sched_ok:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text("❌ រកមិនឃើញ Schedule ។"))
            return
        if row["status"] != "pending":
            st = row["status"]
            await safe_send(
                lambda: query.message.reply_text(
                    f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាចបញ្ជាក់ទៀតទេ។",
                    parse_mode="HTML",
                )
            )
            return
        dt_str = _fmt_dt(datetime.fromisoformat(row["broadcast_at"]))
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(
            lambda: query.message.reply_text(
                f"✅ <b>Schedule #{row_id} បានបញ្ជាក់!</b>\n⏰ នឹង Broadcast នៅ {dt_str}",
                parse_mode="HTML",
            )
        )
        return

    if data.startswith("sched_no:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if row and row["status"] == "pending":
            await loop.run_in_executor(None, db_sched_set_status, row_id, "cancelled")
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(
            lambda: query.message.reply_text(
                f"❌ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
            )
        )
        return

    if data == "sched_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "sched_noop":
        return

    if data.startswith("sched_page:"):
        page = int(data.split(":")[1])
        rows = await loop.run_in_executor(None, db_sched_fetch_admin_pending, user_id)
        with suppress(Exception):
            await query.message.edit_reply_markup(
                reply_markup=get_schedules_list_kb(rows, page=page)
            )
        return

    if data.startswith("sched_view:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text("❌ រកមិនឃើញ Schedule ។"))
            return
        try:
            dt_str = _fmt_dt(datetime.fromisoformat(row["broadcast_at"]))
        except Exception:
            dt_str = str(row.get("broadcast_at", "?"))
        content = (row.get("plain_text") or row.get("caption") or "(photo)")[:300]
        cancel_kb = (
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("🗑️ Cancel Schedule", callback_data=f"sched_cancel_confirm:{row_id}")]]
            )
            if row["status"] == "pending"
            else None
        )
        await safe_send(
            lambda: query.message.reply_text(
                f"📋 <b>Schedule #{row_id}</b>\n"
                f"⏰ {dt_str}\n"
                f"ស្ថានភាព: <b>{row['status']}</b>\n\n"
                f"{html.escape(content)}",
                parse_mode="HTML",
                reply_markup=cancel_kb,
            )
        )
        return

    if data.startswith("sched_cancel_confirm:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row or row.get("admin_id") != user_id:
            await safe_send(
                lambda: query.message.reply_text("⛔ អ្នកមិនមានសិទ្ធិ cancel Schedule នេះ។")
            )
            return
        if row["status"] != "pending":
            st = row["status"]
            await safe_send(
                lambda: query.message.reply_text(
                    f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
                    parse_mode="HTML",
                )
            )
            return
        await loop.run_in_executor(None, db_sched_set_status, row_id, "cancelled")
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(
            lambda: query.message.reply_text(
                f"✅ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
            )
        )

async def _fire_scheduled_broadcast(bot, row: dict) -> None:
    row_id = row["id"]
    admin_id = row["admin_id"]
    logger.info(f"Firing scheduled broadcast #{row_id} for admin {admin_id}")
    loop = asyncio.get_running_loop()

    claimed = await loop.run_in_executor(None, db_sched_claim, row_id)
    if not claimed:
        logger.warning(f"Scheduled broadcast #{row_id} already claimed — skipping.")
        return

    pending = {
        "photo_file_id": row.get("photo_file_id"),
        "caption": row.get("caption") or "",
        "text": row.get("plain_text") or "",
    }

    # FIX #7: use shared helper instead of duplicated loop
    result = await _run_broadcast_to_all(
        bot, admin_id, pending, label=f"Scheduled #{row_id}"
    )
    sent, failed, blocked = result if result else (0, 0, 0)

    try:
        if supabase:
            await loop.run_in_executor(
                None,
                lambda: supabase.table("scheduled_broadcasts")
                .update({
                    "status": "done",
                    "sent_count": sent,
                    "failed_count": failed,
                    "blocked_count": blocked,
                })
                .eq("id", row_id)
                .execute(),
            )
    except Exception as e:
        logger.error(f"Could not mark scheduled broadcast #{row_id} done: {e}")

_scheduler_tasks: set[asyncio.Task] = set()

async def _scheduler_loop(bot, stop_event: asyncio.Event) -> None:
    logger.info("Scheduled broadcast loop started.")
    while not stop_event.is_set():
        try:
            loop = asyncio.get_running_loop()
            due = await loop.run_in_executor(None, db_sched_fetch_due)
            for row in due:
                task = asyncio.create_task(_fire_scheduled_broadcast(bot, row))
                _scheduler_tasks.add(task)
                task.add_done_callback(_scheduler_tasks.discard)
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=float(_SCHED_POLL_INTERVAL))
        except asyncio.TimeoutError:
            pass
    logger.info("Scheduled broadcast loop stopped.")

# ===========================================================================
# PER-USER CHAT
# ===========================================================================
@admin_only
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users = await asyncio.get_running_loop().run_in_executor(None, get_all_users_with_names)
    if not users:
        await safe_send(
            lambda: update.message.reply_text("❌ គ្មានអ្នកប្រើប្រាស់ registered ទេ។")
        )
        return
    await safe_send(
        lambda: update.message.reply_text(
            f"👥 <b>អ្នកប្រើប្រាស់ ({len(users)} នាក់)</b>\nចុចលើឈ្មោះ ដើម្បីចាប់ផ្ដើម Chat ។",
            parse_mode="HTML",
            reply_markup=get_users_page_kb(users, page=0),
        )
    )

@admin_only
async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    args = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(
            lambda: update.message.reply_text(
                "❌ Usage: /chat <user_id>\nឬប្រើ /users ដើម្បីជ្រើស user ។"
            )
        )
        return
    target_id = int(args[0])
    exists = await asyncio.get_running_loop().run_in_executor(None, user_exists_in_db, target_id)
    if not exists:
        await safe_send(
            lambda: update.message.reply_text(
                f"❌ User <code>{target_id}</code> មិនមាននៅក្នុង Database ។",
                parse_mode="HTML",
            )
        )
        return
    await _open_chat_session(context.bot, admin_id, target_id, context)
    await safe_send(
        lambda: update.message.reply_text(
            f"💬 <b>Chat Mode បើក</b>\n\n"
            f"កំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
            "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
            "វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
            parse_mode="HTML",
        )
    )

@admin_only
async def cmd_endchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    target_id = _close_session(admin_id)
    context.user_data.pop("chat_state", None)
    if target_id is None:
        await safe_send(lambda: update.message.reply_text("ℹ️ អ្នកមិនទាន់ open Chat ណាមួយទេ។"))
        return
    await safe_send(
        lambda: update.message.reply_text(
            f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។",
            parse_mode="HTML",
        )
    )
    with suppress(Exception):
        await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")

async def _open_chat_session(bot, admin_id: int, target_id: int, context):
    _open_session(admin_id, target_id)
    context.user_data["chat_state"] = CHAT_WAIT_MESSAGE
    with suppress(Exception):
        await bot.send_message(
            chat_id=target_id, text="🔔 Admin ចង់ Chat ជាមួយអ្នក។ ផ្ញើសារតបមកបាន!"
        )

async def users_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    if data == "users_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "noop":
        return

    if data.startswith("users_page:"):
        page = int(data.split(":")[1])
        users = await asyncio.get_running_loop().run_in_executor(None, get_all_users_with_names)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=get_users_page_kb(users, page=page))
        return

    if data.startswith("chat_open:"):
        target_id = int(data.split(":")[1])
        admin_id = user_id
        with suppress(Exception):
            await query.message.delete()
        await _open_chat_session(context.bot, admin_id, target_id, context)
        await safe_send(
            lambda: context.bot.send_message(
                chat_id=admin_id,
                text=(
                    f"💬 <b>Chat Mode បើក</b>\n\n"
                    f"កំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
                    "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
                    "វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។"
                ),
                parse_mode="HTML",
            )
        )

async def _fwd_admin_to_user(bot, admin_id: int, target_id: int, msg) -> bool:
    async def _do():
        if msg.text:
            await bot.send_message(
                chat_id=target_id,
                text=f"📩 <b>Admin:</b>\n{html.escape(msg.text)}",
                parse_mode="HTML",
            )
        elif msg.photo:
            cap = (f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>")
            await bot.send_photo(chat_id=target_id, photo=msg.photo[-1].file_id, caption=cap, parse_mode="HTML")
        elif msg.voice:
            await bot.send_voice(chat_id=target_id, voice=msg.voice.file_id, caption="📩 Admin voice message")
        elif msg.video_note:
            await bot.send_video_note(chat_id=target_id, video_note=msg.video_note.file_id)
        elif msg.sticker:
            await bot.send_sticker(chat_id=target_id, sticker=msg.sticker.file_id)
        elif msg.document:
            cap = (f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>")
            await bot.send_document(chat_id=target_id, document=msg.document.file_id, caption=cap, parse_mode="HTML")
        elif msg.video:
            cap = (f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>")
            await bot.send_video(chat_id=target_id, video=msg.video.file_id, caption=cap, parse_mode="HTML")
        elif msg.audio:
            await bot.send_audio(chat_id=target_id, audio=msg.audio.file_id)
        else:
            await bot.forward_message(chat_id=target_id, from_chat_id=admin_id, message_id=msg.message_id)

    try:
        await safe_send(_do)
        return True
    except Forbidden:
        return False
    except Exception as e:
        logger.error(f"_fwd_admin_to_user error: {e}")
        return False

async def _fwd_user_to_admin(bot, admin_id: int, user_id: int, username: str, msg) -> bool:
    banner = f"💬 <b>{html.escape(username)} ({user_id}):</b>"

    async def _do():
        if msg.text:
            await bot.send_message(chat_id=admin_id, text=f"{banner}\n{html.escape(msg.text)}", parse_mode="HTML")
        elif msg.photo:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_photo(chat_id=admin_id, photo=msg.photo[-1].file_id, caption=cap, parse_mode="HTML")
        elif msg.voice:
            await bot.send_voice(chat_id=admin_id, voice=msg.voice.file_id, caption=banner, parse_mode="HTML")
        elif msg.video_note:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_video_note(chat_id=admin_id, video_note=msg.video_note.file_id)
        elif msg.sticker:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_sticker(chat_id=admin_id, sticker=msg.sticker.file_id)
        elif msg.document:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_document(chat_id=admin_id, document=msg.document.file_id, caption=cap, parse_mode="HTML")
        elif msg.video:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_video(chat_id=admin_id, video=msg.video.file_id, caption=cap, parse_mode="HTML")
        elif msg.audio:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_audio(chat_id=admin_id, audio=msg.audio.file_id)
        else:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.forward_message(chat_id=admin_id, from_chat_id=user_id, message_id=msg.message_id)

    try:
        await safe_send(_do)
        return True
    except Exception as e:
        logger.error(f"_fwd_user_to_admin error: {e}")
        return False

# ===========================================================================
# STATS + CANCEL
# ===========================================================================
@admin_only
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loop = asyncio.get_running_loop()
    user_ids = await loop.run_in_executor(None, get_all_user_ids)
    active_chats = len(_admin_chat_target)
    pending_scheds = await loop.run_in_executor(
        None, db_sched_fetch_admin_pending, update.effective_user.id
    )
    await safe_send(
        lambda: update.message.reply_text(
            f"📊 <b>Bot Statistics</b>\n\n"
            f"👥 អ្នកប្រើប្រាស់សរុប: <b>{len(user_ids)}</b>\n"
            f"💬 Active Admin Chats: <b>{active_chats}</b>\n"
            f"📅 Scheduled (pending): <b>{len(pending_scheds)}</b>\n"
            f"🔒 Active user locks: <b>{len(_user_locks)}</b>\n"
            f"💭 History cache entries: <b>{len(_hist_cache)}</b>\n"
            f"🤖 Gemini Model: <b>{GEMINI_MODEL}</b>",
            parse_mode="HTML",
        )
    )

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not _is_admin(uid):
        return
    cleared = False

    if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
        _pending_broadcast.pop(uid, None)
        context.user_data.pop("bc_state", None)
        await safe_send(lambda: update.message.reply_text("❌ Broadcast បានបោះបង់។"))
        cleared = True

    if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _close_session(uid)
        context.user_data.pop("chat_state", None)
        if target_id:
            await safe_send(
                lambda: update.message.reply_text(
                    f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។",
                    parse_mode="HTML",
                )
            )
            with suppress(Exception):
                await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")
        cleared = True

    if context.user_data.get("sched_state") in (SCHED_WAIT_MSG, SCHED_WAIT_TIME):
        _sched_payload.pop(uid, None)
        context.user_data.pop("sched_state", None)
        await safe_send(lambda: update.message.reply_text("❌ Schedule flow បានបោះបង់។"))
        cleared = True

    if not cleared:
        await safe_send(lambda: update.message.reply_text("ℹ️ មិនមាន operation ត្រូវ cancel ទេ។"))

# ===========================================================================
# REGULAR HANDLERS
# ===========================================================================
_BOT_START_TIME: float = 0.0

async def _drop_stale_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _BOT_START_TIME == 0.0:
        return
    msg = update.message or update.edited_message
    if msg and hasattr(msg, "date") and msg.date:
        update_ts = msg.date.timestamp()
        if update_ts < (_BOT_START_TIME - _STALE_GRACE_S):
            logger.debug(
                f"Dropping stale update (id={update.update_id}, "
                f"age={_BOT_START_TIME - update_ts:.1f}s)"
            )
            raise ApplicationHandlerStop

async def _drop_stale_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Drop stale callback queries from pre-restart inline buttons."""
    if not update.callback_query:
        return
    if _BOT_START_TIME == 0.0:
        return
    msg = update.callback_query.message
    if msg is None:
        return
    if msg.date and msg.date.timestamp() < (_BOT_START_TIME - _STALE_GRACE_S):
        logger.debug(f"Dropping stale callback (id={update.update_id})")
        raise ApplicationHandlerStop

async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sync_user_data(update.effective_user)
        await safe_send(
            lambda: update.message.reply_text(
                WELCOME_TEXT,
                reply_markup=ReplyKeyboardRemove(),
                disable_web_page_preview=True,
            )
        )
    except Exception as e:
        logger.error(f"on_start error: {e}")

async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await on_start(update, context)

async def cmd_myprefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    # FIX #11: use async-safe prefs fetch
    prefs = await get_user_prefs_async(user_id)
    gender_label = "👩 ស្រី (Female)" if prefs["gender"] == "female" else "👨 ប្រុស (Male)"
    speed_label = next(
        (lbl for _, (lbl, val) in SPEED_OPTIONS.items() if abs(val - prefs["speed"]) < 0.01),
        f"{prefs['speed']}x",
    )
    await safe_send(
        lambda: update.message.reply_text(
            f"⚙️ <b>ការកំណត់របស់អ្នក</b>\n\n"
            f"🗣️ សំឡេង: <b>{gender_label}</b>\n"
            f"🎚️ ល្បឿន: <b>{speed_label}</b>\n\n"
            "ផ្ញើ text ណាមួយ ហើយប្រើប៊ូតុងក្រោមសំឡេង ដើម្បីប្តូរ។",
            parse_mode="HTML",
            reply_markup=get_main_kb(prefs["gender"]),
        )
    )

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _hist_cache_clear(user_id)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_DB_EXECUTOR, db_history_clear, user_id)
    await safe_send(
        lambda: update.message.reply_text(
            "🗑️ ប្រវត្តិការសន្ទនារបស់អ្នកបានលុបចេញហើយ។\nBot នឹងចាប់ផ្ដើមការសន្ទនាថ្មី។"
        )
    )

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    if _is_admin(user_id):
        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        elif context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        elif context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                reply = (
                    f"✅ Photo ផ្ញើដល់ User <code>{target_id}</code> ។"
                    if ok
                    else f"❌ User <code>{target_id}</code> blocked bot ។"
                )
                await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
                if not ok:
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ រូបភាពបានផ្ញើដល់ Admin ។"))
        return

    if not _gemini:
        await safe_send(
            lambda: msg.reply_text("❌ Gemini API មិន Activate ទេ។ សូម Set GEMINI_API_KEY ។")
        )
        return

    sync_user_data(user)
    uname = user.username or user.first_name or str(user_id)
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_OCR_FRAMES)
    )
    best_photo = msg.photo[-1]
    img_path = None
    try:
        img_path = _make_temp_img(suffix=".jpg")
        tg_file = await safe_send(lambda: context.bot.get_file(best_photo.file_id))
        if not tg_file:
            raise RuntimeError("Could not download photo.")
        await tg_file.download_to_drive(img_path)
        mime_type = _detect_image_mime(img_path)
        ocr_text = await ocr_image(img_path, mime_type=mime_type)

        if not ocr_text or ocr_text.upper() == "NOTEXT":
            await safe_send(lambda: msg.reply_text("🖼️ រូបភាពនេះមិនមានអត្ថបទដែលអាចអានបាន។"))
            return

        record_turn(user_id, "user", f"[Image OCR]: {ocr_text[:500]}")

        is_khmer = bool(_KHMER_RE.search(ocr_text))
        lang_flag = "🇰🇭" if is_khmer else "🇺🇸"
        header = f"🔍 <b>OCR {lang_flag}</b>\n\n"
        plain_pages = _paginate_plain(ocr_text, limit=TELE_MSG_LIMIT - len(header))
        sent_pages = []
        for page_idx, plain_page in enumerate(plain_pages):
            page_body = (header if page_idx == 0 else "") + html.escape(plain_page)
            sent = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, ocr_text,
                chat_id=msg.chat_id, user_id=user_id, username=uname,
            )
            await safe_send(
                lambda: last_sent.edit_reply_markup(
                    reply_markup=get_ocr_confirm_kb(last_sent.message_id)
                )
            )
    except Exception as e:
        logger.error(f"on_photo OCR error: {e}")
        await safe_send(lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ OCR រូបភាព។"))
    finally:
        await _stop_timer(stop_event, timer_task)
        if img_path:
            _cleanup(img_path)

async def on_any_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok
                else f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)
                context.user_data.pop("chat_state", None)
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ ផ្ញើដល់ Admin ។"))

async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ Voice ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok
                else f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)
                context.user_data.pop("chat_state", None)
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ Voice ផ្ញើដល់ Admin ។"))
        return

    if not _gemini:
        await safe_send(
            lambda: msg.reply_text("❌ Gemini API មិន Activate ទេ។ សូម Set GEMINI_API_KEY ។")
        )
        return

    if msg.voice.file_size and msg.voice.file_size > MAX_VOICE_BYTES:
        await safe_send(lambda: msg.reply_text("❌ ឯកសារសំឡេងធំពេក (អតិបរមា 20MB)។"))
        return

    sync_user_data(user)
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_TRANSCRIBE_FRAMES)
    )
    ogg_path = _make_temp_ogg()
    try:
        voice_file = await safe_send(lambda: context.bot.get_file(msg.voice.file_id))
        if not voice_file:
            raise RuntimeError("Could not get voice file")
        await voice_file.download_to_drive(ogg_path)
        transcript = await transcribe_voice(ogg_path)

        if not transcript:
            await safe_send(lambda: msg.reply_text("❌ រក Transcript មិនឃើញ។"))
            return

        record_turn(user_id, "user", f"[Voice Transcript]: {transcript[:500]}")

        is_khmer = bool(_KHMER_RE.search(transcript))
        lang_flag = "🇰🇭" if is_khmer else "🇺🇸"
        header = f"📝 <b>Transcript</b> {lang_flag}\n\n"
        plain_pages = _paginate_plain(transcript, limit=TELE_MSG_LIMIT - len(header))
        sent_pages = []
        for page_idx, plain_page in enumerate(plain_pages):
            page_body = (header if page_idx == 0 else "") + html.escape(plain_page)
            sent = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, transcript,
                chat_id=msg.chat_id, user_id=user_id,
                username=user.username or user.first_name,
            )
            await safe_send(
                lambda: last_sent.edit_reply_markup(
                    reply_markup=get_transcription_kb(last_sent.message_id)
                )
            )
    except Exception as e:
        logger.error(f"on_voice error: {e}")
        with suppress(Exception):
            await safe_send(lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ Transcribe។"))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(ogg_path)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    text = msg.text
    if not text:
        return

    user = update.effective_user
    user_id = user.id

    # ── Admin: state machine flows ─────────────────────────────────────────
    if _is_admin(user_id):
        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        if sched_state == SCHED_WAIT_TIME:
            await _handle_sched_datetime(update, context)
            return
        if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                if ok:
                    await safe_send(
                        lambda: msg.reply_text(
                            f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។",
                            parse_mode="HTML",
                        )
                    )
                else:
                    await safe_send(
                        lambda: msg.reply_text(
                            f"❌ User <code>{target_id}</code> blocked bot ។ Chat session បានបិទ។",
                            parse_mode="HTML",
                        )
                    )
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    # Non-admin in chat session — forward to admin, do NOT run TTS
    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ សាររបស់អ្នកបានផ្ញើដល់ Admin ។"))
        return

    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    stripped = text.strip()
    if not stripped:
        return

    if len(stripped) > MAX_INPUT_CHARS:
        await safe_send(
            lambda: msg.reply_text(
                f"❌ អត្ថបទវែងពេក។ អតិបរមា {MAX_INPUT_CHARS} តួអក្សរ។\n"
                f"(អ្នកបានផ្ញើ {len(stripped)} តួ)"
            )
        )
        return

    if await _check_cooldown(msg, user_id):
        return

    sync_user_data(user)
    # FIX #11: use async-safe prefs fetch
    prefs = await get_user_prefs_async(user_id)
    gender = prefs["gender"]
    speed = prefs["speed"]

    # Resolve text through conversation context
    tts_text = await resolve_tts_text(user_id, stripped, asyncio.get_running_loop())

    # FIX: guard against empty resolved text
    if not tts_text or not tts_text.strip():
        tts_text = stripped

    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(update.effective_chat.id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)
    async with lock:
        try:
            audio_bytes = await generate_voice(tts_text, gender, speed, file_path)
            sent_msg = await safe_send(
                lambda: msg.reply_voice(
                    voice=io.BytesIO(audio_bytes),
                    caption=f"🗣️ {BOT_TAG}",
                    reply_markup=get_main_kb(gender),
                )
            )
            if sent_msg:
                save_text_cache(
                    sent_msg.message_id, tts_text,
                    chat_id=msg.chat_id, user_id=user_id,
                    username=user.username or user.first_name,
                )
                # FIX #2: record both turns only after successful TTS delivery
                record_turn(user_id, "user", stripped)
                record_turn(user_id, "assistant", tts_text)
            _set_last_tts(user_id)
        except Exception as e:
            logger.error(f"on_text TTS error: {e}")
            with suppress(Exception):
                await safe_send(lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"))
        finally:
            await _stop_timer(stop_event, timer_task)
            _cleanup(file_path)

# ---------------------------------------------------------------------------
# Cache key helper
# ---------------------------------------------------------------------------
def _cache_key_from_query(query) -> tuple[int, int]:
    return query.message.chat.id, query.message.message_id

# ---------------------------------------------------------------------------
# Callback dispatch helpers
# ---------------------------------------------------------------------------
async def _cb_show_speed(query, user_id: int, context):
    # FIX #11: use async-safe prefs fetch
    prefs = await get_user_prefs_async(user_id)
    await safe_send(
        lambda: query.message.edit_reply_markup(reply_markup=get_speed_kb(prefs["speed"]))
    )

async def _cb_hide_speed(query, user_id: int, context):
    # FIX #11: use async-safe prefs fetch
    prefs = await get_user_prefs_async(user_id)
    await safe_send(
        lambda: query.message.edit_reply_markup(reply_markup=get_main_kb(prefs["gender"]))
    )

async def _cb_speed(query, user_id: int, context, data: str):
    _, new_speed = SPEED_OPTIONS[data]
    chat_id, msg_id = _cache_key_from_query(query)

    # FIX #11: use async-safe prefs fetch; FIX: fetch cache and prefs concurrently
    loop = asyncio.get_running_loop()
    original_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទដើមមិនឃើញ។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    update_user_speed(user_id, new_speed)

    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(query.message.chat.id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)
    async with lock:
        try:
            audio_bytes = await generate_voice(original_text, gender, new_speed, file_path)
            with suppress(Exception):
                await query.message.delete()
            new_msg = await safe_send(
                lambda: query.message.chat.send_voice(
                    voice=io.BytesIO(audio_bytes),
                    caption=f"🗣️ {BOT_TAG}",
                    reply_markup=get_main_kb(gender),
                )
            )
            if new_msg:
                save_text_cache(
                    new_msg.message_id, original_text,
                    chat_id=chat_id, user_id=user_id,
                    username=query.from_user.username or query.from_user.first_name,
                )
                # FIX #3: record assistant turn after speed regen
                record_turn(user_id, "assistant", original_text)
            _set_last_tts(user_id)
        except Exception as e:
            logger.error(f"speed regen error: {e}")
        finally:
            await _stop_timer(stop_event, timer_task)
            _cleanup(file_path)

async def _cb_gender(query, user_id: int, context, data: str):
    new_gender = data.replace("tg_", "")
    chat_id, msg_id = _cache_key_from_query(query)

    # FIX #11: use async-safe prefs fetch; FIX: fetch cache and prefs concurrently
    loop = asyncio.get_running_loop()
    original_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទដើមមិនឃើញ។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    speed = prefs["speed"]
    update_user_gender(user_id, new_gender)

    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(query.message.chat.id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)
    async with lock:
        try:
            audio_bytes = await generate_voice(original_text, new_gender, speed, file_path)
            with suppress(Exception):
                await query.message.delete()
            new_msg = await safe_send(
                lambda: query.message.chat.send_voice(
                    voice=io.BytesIO(audio_bytes),
                    caption=f"🗣️ {BOT_TAG}",
                    reply_markup=get_main_kb(new_gender),
                )
            )
            if new_msg:
                save_text_cache(
                    new_msg.message_id, original_text,
                    chat_id=chat_id, user_id=user_id,
                    username=query.from_user.username or query.from_user.first_name,
                )
                # FIX #3: record assistant turn after gender regen
                record_turn(user_id, "assistant", original_text)
            _set_last_tts(user_id)
        except Exception as e:
            logger.error(f"gender regen error: {e}")
        finally:
            await _stop_timer(stop_event, timer_task)
            _cleanup(file_path)

async def _cb_tts_transcript(query, user_id: int, context, data: str):
    transcript_msg_id = int(data.split(":")[1])
    chat_id = query.message.chat.id

    # FIX #11: use async-safe prefs fetch; FIX: fetch cache and prefs concurrently
    loop = asyncio.get_running_loop()
    original_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, transcript_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    speed = prefs["speed"]
    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(query.message.chat.id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)
    async with lock:
        try:
            audio_bytes = await generate_voice(original_text, gender, speed, file_path)
            new_msg = await safe_send(
                lambda: query.message.chat.send_voice(
                    voice=io.BytesIO(audio_bytes),
                    caption=f"🗣️ {BOT_TAG}",
                    reply_markup=get_main_kb(gender),
                )
            )
            if new_msg:
                save_text_cache(
                    new_msg.message_id, original_text,
                    chat_id=chat_id, user_id=user_id,
                    username=query.from_user.username or query.from_user.first_name,
                )
                record_turn(user_id, "assistant", original_text)
            _set_last_tts(user_id)
        except Exception as e:
            logger.error(f"transcript TTS error: {e}")
        finally:
            await _stop_timer(stop_event, timer_task)
            _cleanup(file_path)

async def _cb_doc_read(query, user_id: int, context, data: str):
    src_msg_id = int(data.split(":")[1])
    chat_id = query.message.chat.id

    # FIX #11: use async-safe prefs fetch; FIX: fetch cache and prefs concurrently
    loop = asyncio.get_running_loop()
    full_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, src_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not full_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ (cache expired)។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    speed = prefs["speed"]
    uname = query.from_user.username or query.from_user.first_name or str(user_id)

    with suppress(Exception):
        await query.message.delete()

    tts_stop = asyncio.Event()
    tts_timer = asyncio.create_task(send_status_timer(chat_id, context.bot, tts_stop))
    lock = _get_user_lock(user_id)
    async with lock:
        try:
            await _deliver_paged_tts(
                chat_id=chat_id, bot=context.bot, text=full_text,
                gender=gender, speed=speed, user_id=user_id, username=uname,
            )
            record_turn(user_id, "assistant", full_text[:CONV_CONTEXT_MAX_CHARS])
        except Exception as e:
            logger.error(f"_cb_doc_read delivery error: {e}")
            with suppress(Exception):
                await context.bot.send_message(chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។")
        finally:
            await _stop_timer(tts_stop, tts_timer)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catch-all callback handler for TTS/voice callbacks."""
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    with suppress(Exception):
        await query.answer()

    # FIX: guard against callbacks without a message (inline mode)
    if query.message is None:
        logger.debug(f"on_callback: no message for data={data!r} (inline mode?)")
        return

    try:
        if data == "show_speed":
            await _cb_show_speed(query, user_id, context)
        elif data == "hide_speed":
            await _cb_hide_speed(query, user_id, context)
        elif data in SPEED_OPTIONS:
            await _cb_speed(query, user_id, context, data)
        elif data in ("tg_female", "tg_male"):
            await _cb_gender(query, user_id, context, data)
        elif data.startswith("tts_transcript:"):
            await _cb_tts_transcript(query, user_id, context, data)
        elif data.startswith("del_transcript:"):
            with suppress(Exception):
                await query.message.delete()
        elif data.startswith("doc_del:"):
            with suppress(Exception):
                await query.message.delete()
        elif data.startswith("doc_read:"):
            await _cb_doc_read(query, user_id, context, data)
        else:
            logger.debug(f"on_callback: unhandled data={data!r}")
    except Exception as e:
        logger.error(f"on_callback unhandled [data={data}]: {e}", exc_info=True)

# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Unhandled exception: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        with suppress(Exception):
            await safe_send(
                lambda: update.effective_message.reply_text(
                    "⚠️ មានបញ្ហាបច្ចេកទេស។ Bot នៅដំណើរការ — សូមព្យាយាមម្តងទៀត។"
                )
            )

# ---------------------------------------------------------------------------
# Bot runner
# ---------------------------------------------------------------------------
async def _run_bot():
    global _BOT_START_TIME, _GEMINI_SEMAPHORE, _BROADCAST_SEMAPHORE, _prefs_cache_lock
    _BOT_START_TIME = time.time()

    # Init all asyncio primitives on the running event loop
    _GEMINI_SEMAPHORE = asyncio.Semaphore(4)
    _BROADCAST_SEMAPHORE = asyncio.Semaphore(10)  # FIX #8: module-level broadcast semaphore
    _prefs_cache_lock = asyncio.Lock()             # FIX #11: prefs cache asyncio lock

    _admin_chat_target.clear()
    _user_to_admin.clear()
    _pending_broadcast.clear()
    _prefs_cache.clear()
    _user_last_tts.clear()
    _sched_payload.clear()
    _scheduler_tasks.clear()
    _hist_cache.clear()
    # NOTE: do NOT clear _user_locks — in-flight coroutines may hold references.

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    # Stale-update and stale-callback guards registered first (group=-1).
    app.add_handler(TypeHandler(Update, _drop_stale_updates), group=-1)
    app.add_handler(CallbackQueryHandler(_drop_stale_callback), group=-1)

    # ── Commands ──────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("myprefs", cmd_myprefs))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("broadcast", broadcast_start))
    app.add_handler(CommandHandler("schedule", cmd_schedule))
    app.add_handler(CommandHandler("schedules", cmd_schedules))
    app.add_handler(CommandHandler("cancelschedule", cmd_cancelschedule))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("stats", admin_stats))
    app.add_handler(CommandHandler("users", cmd_users))
    app.add_handler(CommandHandler("chat", cmd_chat))
    app.add_handler(CommandHandler("endchat", cmd_endchat))

    # ── Callback query handlers (most-specific first) ─────────────────────
    app.add_handler(
        CallbackQueryHandler(broadcast_callback, pattern=r"^bc_(confirm|cancel)$")
    )
    app.add_handler(
        CallbackQueryHandler(
            users_page_callback,
            pattern=r"^(users_page:\d+|users_close|chat_open:\d+|noop)$",
        )
    )
    app.add_handler(CallbackQueryHandler(sched_callback, pattern=r"^sched_"))
    app.add_handler(CallbackQueryHandler(on_callback))  # catch-all for TTS

    # ── Message handlers ──────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(
        MessageHandler(
            filters.Sticker.ALL
            | filters.Document.ALL
            | filters.VIDEO
            | filters.AUDIO
            | filters.VIDEO_NOTE,
            on_any_media,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    logger.info(f"Bot polling started. Admins: {ADMIN_IDS or 'none configured'}")
    logger.info(f"Gemini model: {GEMINI_MODEL}")

    # ── Background tasks ──────────────────────────────────────────────────
    sched_stop = asyncio.Event()
    sweep_stop = asyncio.Event()
    sched_task = asyncio.create_task(_scheduler_loop(app.bot, sched_stop))
    sweep_task = asyncio.create_task(_periodic_temp_sweep(sweep_stop))

    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )
        try:
            await asyncio.Event().wait()
        finally:
            sched_stop.set()
            sweep_stop.set()
            for task in (sched_task, sweep_task):
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    load_dotenv()
    _init_clients()

    threading.Thread(target=run_flask, daemon=True).start()
    print("Flask health-check server started.")

    threading.Thread(target=keep_alive, daemon=True).start()
    print("Keep-alive thread started.")

    _sweep_stale_temps()
    ensure_speed_column()

    if not ADMIN_IDS:
        logger.warning(
            "No ADMIN_IDS configured. "
            "Set ADMIN_IDS=123456,789012 in your environment."
        )

    print(f"Bot is starting... (Gemini model: {GEMINI_MODEL})")
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
