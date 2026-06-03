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
import hashlib
import hmac
import secrets
import socket
import requests
import imageio_ffmpeg as _iio_ffmpeg
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone, timedelta
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None
from flask import Flask
from supabase import create_client, Client
from typing import Any, Callable

try:
    import redis as redis_lib
except Exception:
    redis_lib = None


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
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if not render_url:
        logger_ka.warning("RENDER_EXTERNAL_URL not set — self-ping disabled.")
        return

    time.sleep(10)
    headers = {"User-Agent": "Mozilla/5.0 (KeepAlive/1.0)", "Accept": "text/plain,*/*"}

    while True:
        urls = [render_url]
        if not render_url.endswith("/ping"):
            urls.append(f"{render_url}/ping")

        ok = False
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if 200 <= r.status_code < 400:
                    logger_ka.info(f"Keep-alive OK -> {r.status_code} {url}")
                    ok = True
                    break
                logger_ka.warning(f"Keep-alive non-OK -> {r.status_code} {url}")
            except Exception as e:
                logger_ka.warning(f"Keep-alive failed for {url}: {e}")

        if not ok:
            logger_ka.warning("Keep-alive failed for all URLs. Check RENDER_EXTERNAL_URL.")

        time.sleep(240)

# ── AI Assistant REST API ──────────────────────────────────────────────────
import base64
import json as _json
from flask import request, jsonify, Response, stream_with_context

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

_AI_API_MAX_IMAGE_BYTES   = 10 * 1024 * 1024   # 10 MB
_AI_API_MAX_AUDIO_BYTES   = 50 * 1024 * 1024   # 50 MB
_AI_API_MAX_MESSAGE_CHARS = 32_000
_AI_API_MAX_HISTORY_TURNS = 20

_AI_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_AI_ALLOWED_AUDIO_MIME = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/flac",
    "audio/aac", "audio/opus", "audio/webm", "audio/mp4", "audio/x-m4a",
    "video/mp4", "video/webm",
}

_AI_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that supports both Khmer (ភាសាខ្មែរ) and English. "
    "Always reply in the same language the user uses. "
    "If the user sends an image, describe and analyse it fully. "
    "If the user sends audio, provide an accurate transcription then answer any follow-up. "
    "Be concise, accurate, and friendly. "
    "Never refuse reasonable requests. "
    "Format answers clearly using plain text (no markdown unless the user asks for it)."
)

# ---------------------------------------------------------------------------
# Concurrency limits
# ---------------------------------------------------------------------------
MAX_CONCURRENT_TTS_USERS   = int(os.environ.get("MAX_CONCURRENT_TTS_USERS", "6"))
MAX_CONCURRENT_AI          = int(os.environ.get("MAX_CONCURRENT_AI", os.environ.get("MAX_CONCURRENT_GEMINI", "3")))
MAX_CONCURRENT_BROADCAST   = int(os.environ.get("MAX_CONCURRENT_BROADCAST", "3"))
BROADCAST_BATCH_SIZE       = max(1, int(os.environ.get("BROADCAST_BATCH_SIZE", str(MAX_CONCURRENT_BROADCAST))))
BROADCAST_INTER_BATCH_DELAY = float(os.environ.get("BROADCAST_INTER_BATCH_DELAY", "0.20"))
TTS_RESOLVER_AI_ENABLED    = os.environ.get("TTS_RESOLVER_AI_ENABLED", "0") == "1"

# ---------------------------------------------------------------------------
# Subtitle/Text document support
# ---------------------------------------------------------------------------
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".txt"}
MAX_SUBTITLE_BYTES  = 2 * 1024 * 1024
MAX_SUBTITLE_CHARS  = 20_000


def _ai_error(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg, "code": code}), code


def _ai_cors(response_or_tuple):
    if isinstance(response_or_tuple, tuple):
        resp, code = response_or_tuple
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, code
    response_or_tuple.headers["Access-Control-Allow-Origin"] = "*"
    return response_or_tuple


def _detect_lang(text: str) -> str:
    khmer = sum(1 for c in text if "\u1780" <= c <= "\u17FF")
    alpha = sum(1 for c in text if c.isalpha())
    return "km" if alpha and khmer / alpha > 0.25 else "en"



# ---------------------------------------------------------------------------
# Dynamic AI API keys
# ---------------------------------------------------------------------------
# Admins can create AI access keys from Telegram with:
#   /api create optional-name
#
# The raw key is shown only once. The bot stores only SHA-256 hashes in
# Supabase table `ai_api_keys`, so a database leak does not expose usable keys.
# Static AI_API_KEY is still supported for backwards compatibility.
_AI_API_KEY_PREFIX = "sk-ai-"
_AI_API_KEY_RANDOM_BYTES = 32
_AI_API_KEY_DISPLAY_CHARS = 18
_AI_API_KEY_VALIDATION_CACHE_TTL_S = float(os.environ.get("AI_API_KEY_CACHE_TTL_S", "60"))
_AI_API_KEY_TOUCH_INTERVAL_S = float(os.environ.get("AI_API_KEY_TOUCH_INTERVAL_S", "60"))
_AI_API_KEY_CACHE_MAX = 10_000

_api_key_validation_cache: OrderedDict[str, tuple[bool, int | None, float]] = OrderedDict()
_api_key_validation_cache_lock = threading.Lock()
_api_key_touch_last: dict[int, float] = {}
_api_key_touch_lock = threading.Lock()

# Used only when Supabase is not configured. Keys created in this mode work
# until the process restarts. Production should use Supabase.
_api_keys_memory_by_hash: OrderedDict[str, dict] = OrderedDict()

AI_API_KEYS_TABLE_SQL = """-- Required table for /api generated AI access keys
-- Use your Supabase SQL editor. For production, set SUPABASE_SERVICE_ROLE_KEY
-- on the bot server instead of a publishable/anon key.
create table if not exists public.ai_api_keys (
  id bigint generated by default as identity primary key,
  key_prefix text not null,
  key_hash text not null unique,
  admin_id bigint not null,
  note text,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  revoked_at timestamptz,
  last_used_at timestamptz
);

create index if not exists ai_api_keys_key_hash_idx
  on public.ai_api_keys (key_hash);

create index if not exists ai_api_keys_active_idx
  on public.ai_api_keys (active);

alter table public.ai_api_keys enable row level security;

drop policy if exists "service_role_ai_api_keys_all" on public.ai_api_keys;
create policy "service_role_ai_api_keys_all"
on public.ai_api_keys
for all
to service_role
using (true)
with check (true);
"""



ADMIN_V2_TABLES_SQL = AI_API_KEYS_TABLE_SQL + """

-- Optional but recommended tables for Admin Dashboard V2
create table if not exists public.blocked_users (
  user_id bigint primary key,
  admin_id bigint,
  reason text,
  blocked_at timestamptz not null default now()
);

create index if not exists blocked_users_blocked_at_idx
  on public.blocked_users (blocked_at desc);

alter table public.blocked_users enable row level security;

drop policy if exists "service_role_blocked_users_all" on public.blocked_users;
create policy "service_role_blocked_users_all"
on public.blocked_users
for all
to service_role
using (true)
with check (true);

create table if not exists public.bot_settings (
  key text primary key,
  value text not null,
  updated_by bigint,
  updated_at timestamptz not null default now()
);

alter table public.bot_settings enable row level security;

drop policy if exists "service_role_bot_settings_all" on public.bot_settings;
create policy "service_role_bot_settings_all"
on public.bot_settings
for all
to service_role
using (true)
with check (true);

insert into public.bot_settings (key, value) values
  ('maintenance_mode', '0'),
  ('tts_enabled', '1'),
  ('ocr_enabled', '1'),
  ('voice_transcribe_enabled', '1'),
  ('audio_transcribe_enabled', '1'),
  ('ai_resolver_enabled', '1')
on conflict (key) do nothing;

-- Distributed scheduler lock. Required for safe scheduled broadcasts when
-- Render restarts quickly or when more than one worker/instance is running.
create table if not exists public.bot_locks (
  lock_key text primary key,
  owner text not null,
  locked_until timestamptz not null,
  updated_at timestamptz not null default now()
);

create index if not exists bot_locks_locked_until_idx
  on public.bot_locks (locked_until);

alter table public.bot_locks enable row level security;

drop policy if exists "service_role_bot_locks_all" on public.bot_locks;
create policy "service_role_bot_locks_all"
on public.bot_locks
for all
to service_role
using (true)
with check (true);
"""

def _extract_ai_request_key() -> str:
    auth_header = (request.headers.get("Authorization") or "").strip()
    bearer_key = ""
    if auth_header.lower().startswith("bearer "):
        bearer_key = auth_header[7:].strip()
    return (
        (request.headers.get("X-Api-Key") or "").strip()
        or bearer_key
        or (request.args.get("api_key") or "").strip()
    )


def _hash_ai_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _api_key_prefix(raw_key: str) -> str:
    return raw_key[:_AI_API_KEY_DISPLAY_CHARS]


def _generate_ai_api_key() -> str:
    return _AI_API_KEY_PREFIX + secrets.token_urlsafe(_AI_API_KEY_RANDOM_BYTES)


def _api_key_cache_get(key_hash: str) -> tuple[bool, int | None] | None:
    now = time.monotonic()
    with _api_key_validation_cache_lock:
        item = _api_key_validation_cache.get(key_hash)
        if not item:
            return None
        valid, row_id, created_at = item
        if now - created_at > _AI_API_KEY_VALIDATION_CACHE_TTL_S:
            _api_key_validation_cache.pop(key_hash, None)
            return None
        _api_key_validation_cache.move_to_end(key_hash)
        return valid, row_id


def _api_key_cache_set(key_hash: str, valid: bool, row_id: int | None) -> None:
    with _api_key_validation_cache_lock:
        _api_key_validation_cache.pop(key_hash, None)
        _api_key_validation_cache[key_hash] = (valid, row_id, time.monotonic())
        while len(_api_key_validation_cache) > _AI_API_KEY_CACHE_MAX:
            _api_key_validation_cache.popitem(last=False)


def _api_key_cache_clear() -> None:
    with _api_key_validation_cache_lock:
        _api_key_validation_cache.clear()


def _touch_ai_api_key(row_id: int | None) -> None:
    if not row_id or not supabase:
        return

    now = time.monotonic()
    with _api_key_touch_lock:
        last = _api_key_touch_last.get(row_id, 0.0)
        if now - last < _AI_API_KEY_TOUCH_INTERVAL_S:
            return
        _api_key_touch_last[row_id] = now

    def _run():
        try:
            supabase.table("ai_api_keys").update({
                "last_used_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", row_id).execute()
        except Exception as e:
            logger.warning(f"ai_api_keys touch failed id={row_id}: {e}")

    try:
        _submit_db(_run)
    except Exception:
        # During early startup or interpreter shutdown the DB executor may not
        # be available. Authentication should not fail just because telemetry
        # could not be updated.
        pass


def _validate_dynamic_ai_api_key(raw_key: str) -> bool:
    raw_key = (raw_key or "").strip()
    if not raw_key or len(raw_key) > 256 or not raw_key.startswith(_AI_API_KEY_PREFIX):
        return False

    key_hash = _hash_ai_api_key(raw_key)

    cached = _api_key_cache_get(key_hash)
    if cached is not None:
        valid, row_id = cached
        if valid:
            _touch_ai_api_key(row_id)
        return valid

    mem_row = _api_keys_memory_by_hash.get(key_hash)
    if mem_row:
        valid = bool(mem_row.get("active")) and not mem_row.get("revoked_at")
        _api_key_cache_set(key_hash, valid, mem_row.get("id"))
        return valid

    if not supabase:
        _api_key_cache_set(key_hash, False, None)
        return False

    try:
        res = (
            supabase.table("ai_api_keys")
            .select("id, active, revoked_at")
            .eq("key_hash", key_hash)
            .limit(1)
            .execute()
        )
        row = (res.data or [None])[0]
        valid = bool(row and row.get("active") and not row.get("revoked_at"))
        row_id = int(row["id"]) if row and row.get("id") is not None else None
        _api_key_cache_set(key_hash, valid, row_id)
        if valid:
            _touch_ai_api_key(row_id)
        return valid
    except Exception as e:
        # Missing table or RLS should not expose the API. We fail closed.
        logger.warning(f"Dynamic AI API key validation unavailable: {e}")
        _api_key_cache_set(key_hash, False, None)
        return False


def _dynamic_ai_auth_configured() -> bool:
    if _api_keys_memory_by_hash:
        return True
    if not supabase:
        return False
    try:
        res = supabase.table("ai_api_keys").select("id").eq("active", True).limit(1).execute()
        return bool(res.data)
    except Exception:
        return False


def db_ai_api_key_create(admin_id: int, note: str = "") -> tuple[str, dict, str]:
    raw_key = _generate_ai_api_key()
    row = {
        "key_prefix": _api_key_prefix(raw_key),
        "key_hash": _hash_ai_api_key(raw_key),
        "admin_id": int(admin_id),
        "note": (note or "").strip()[:120] or None,
        "active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revoked_at": None,
        "last_used_at": None,
    }

    if not supabase:
        # Safe fallback for development/demo. This is intentionally explicit
        # because it is not persistent across deploy restarts.
        row = {**row, "id": int(time.time() * 1000)}
        _api_keys_memory_by_hash[row["key_hash"]] = row
        _api_key_cache_clear()
        return raw_key, row, "memory"

    try:
        res = supabase.table("ai_api_keys").insert(row).execute()
        saved = (res.data or [row])[0]
        _api_key_cache_clear()
        return raw_key, saved, "supabase"
    except Exception as e:
        raise RuntimeError(
            "Could not create API key in Supabase. Create table `ai_api_keys` first "
            "and use SUPABASE_SERVICE_ROLE_KEY on the bot server.\n\nSQL:\n"
            + AI_API_KEYS_TABLE_SQL
        ) from e


def db_ai_api_key_list(limit: int = 20) -> list[dict]:
    limit = max(1, min(50, int(limit or 20)))
    if not supabase:
        rows = list(_api_keys_memory_by_hash.values())
        return sorted(rows, key=lambda r: str(r.get("created_at", "")), reverse=True)[:limit]

    try:
        res = (
            supabase.table("ai_api_keys")
            .select("id, key_prefix, admin_id, note, active, created_at, revoked_at, last_used_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        raise RuntimeError(
            "Could not list API keys. If this is the first setup, run:\n\n"
            + AI_API_KEYS_TABLE_SQL
        ) from e


def db_ai_api_key_revoke(identifier: str) -> tuple[bool, str]:
    ident = (identifier or "").strip()
    if not ident:
        return False, "Missing key id or prefix."

    now = datetime.now(timezone.utc).isoformat()

    if not supabase:
        for key_hash, row in list(_api_keys_memory_by_hash.items()):
            if str(row.get("id")) == ident or str(row.get("key_prefix")) == ident:
                row["active"] = False
                row["revoked_at"] = now
                _api_key_cache_clear()
                return True, str(row.get("key_prefix") or ident)
        return False, "API key not found."

    try:
        query = supabase.table("ai_api_keys").update({
            "active": False,
            "revoked_at": now,
        })
        if ident.isdigit():
            res = query.eq("id", int(ident)).execute()
        else:
            res = query.eq("key_prefix", ident).execute()

        if not res.data:
            return False, "API key not found."
        _api_key_cache_clear()
        row = res.data[0]
        return True, str(row.get("key_prefix") or ident)
    except Exception as e:
        raise RuntimeError(f"Could not revoke API key: {e}") from e


def db_ai_api_key_status() -> dict:
    """Return API auth/setup status for the Telegram admin UI."""
    status = {
        "static_key": bool(os.environ.get("AI_API_KEY", "").strip()),
        "service_role_key": bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()),
        "supabase": supabase is not None,
        "table_ok": False,
        "active_count": 0,
        "total_sampled": 0,
        "memory_count": len(_api_keys_memory_by_hash),
        "error": "",
    }

    if not supabase:
        status["table_ok"] = bool(_api_keys_memory_by_hash)
        status["active_count"] = sum(1 for r in _api_keys_memory_by_hash.values() if r.get("active"))
        status["total_sampled"] = len(_api_keys_memory_by_hash)
        return status

    try:
        res = (
            supabase.table("ai_api_keys")
            .select("id, active")
            .limit(1000)
            .execute()
        )
        rows = res.data or []
        status["table_ok"] = True
        status["total_sampled"] = len(rows)
        status["active_count"] = sum(1 for r in rows if r.get("active"))
    except Exception as e:
        status["error"] = str(e)[:500]
    return status


def _check_ai_api_key() -> bool:
    """Validate static AI_API_KEY or Telegram-admin generated /api keys.

    Accepted auth methods:
      - X-Api-Key: <key>
      - Authorization: Bearer <key>
      - ?api_key=<key> (kept for simple browser/testing tools)

    Security:
      - The API fails closed when no valid key is present.
      - Generated keys are stored only as SHA-256 hashes.
      - AI_API_KEY env remains supported for existing clients.
    """
    client_key = _extract_ai_request_key()
    if not client_key:
        return False

    static_key = os.environ.get("AI_API_KEY", "").strip()
    if static_key and hmac.compare_digest(client_key, static_key):
        return True

    return _validate_dynamic_ai_api_key(client_key)


def _build_gemini_contents(
    message: str,
    history: list,
    image_data: bytes | None,
    audio_data: bytes | None,
    image_mime: str,
    audio_mime: str,
) -> list:
    from google.genai import types as _gtypes
    contents = []
    for turn in history[-_AI_API_MAX_HISTORY_TURNS:]:
        role    = turn.get("role", "user")
        content = str(turn.get("content", ""))
        if role not in ("user", "model", "assistant"):
            continue
        gemini_role = "model" if role in ("assistant", "model") else "user"
        contents.append(_gtypes.Content(
            role=gemini_role,
            parts=[_gtypes.Part.from_text(text=content)],
        ))
    user_parts = []
    if image_data:
        user_parts.append(_gtypes.Part.from_bytes(data=image_data, mime_type=image_mime))
    if audio_data:
        user_parts.append(_gtypes.Part.from_bytes(data=audio_data, mime_type=audio_mime))
    if message:
        user_parts.append(_gtypes.Part.from_text(text=message))
    elif not image_data and not audio_data:
        user_parts.append(_gtypes.Part.from_text(text="Hello"))
    contents.append(_gtypes.Content(role="user", parts=user_parts))
    return contents


def _ai_gen_config():
    from google.genai import types as _gtypes
    return _gtypes.GenerateContentConfig(
        system_instruction=_AI_SYSTEM_PROMPT,
        temperature=0.7,
        max_output_tokens=8192,
        safety_settings=[
            _gtypes.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_ONLY_HIGH"),
            _gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_ONLY_HIGH"),
            _gtypes.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
            _gtypes.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        ],
    )


def _parse_multipart_ai_request():
    message   = (request.form.get("message") or "").strip()
    do_stream = request.form.get("stream", "").lower() == "true"
    history   = []
    raw_history = request.form.get("history", "")
    if raw_history:
        try:
            history = _json.loads(raw_history)
        except Exception:
            pass

    image_data = audio_data = None
    image_mime = audio_mime = ""

    img_file = request.files.get("image")
    if img_file:
        image_mime = img_file.mimetype or "image/jpeg"
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        image_data = img_file.read()
        if len(image_data) > _AI_API_MAX_IMAGE_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Image file too large (max 10 MB)."))

    aud_file = request.files.get("audio")
    if aud_file:
        audio_mime = aud_file.mimetype or "audio/mpeg"
        if audio_mime not in _AI_ALLOWED_AUDIO_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported audio type: {audio_mime}"))
        audio_data = aud_file.read()
        if len(audio_data) > _AI_API_MAX_AUDIO_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Audio file too large (max 50 MB)."))

    return message, history, image_data, audio_data, image_mime, audio_mime, do_stream, None


def _parse_json_ai_request():
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return None, None, None, None, None, None, None, \
            _ai_cors(_ai_error("Invalid JSON body."))

    message   = (body.get("message") or "").strip()
    do_stream = bool(body.get("stream", False))
    history   = body.get("history") or []
    image_data = audio_data = None
    image_mime = audio_mime = ""

    if body.get("image_base64"):
        image_mime = body.get("image_mime", "image/jpeg")
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        try:
            image_data = base64.b64decode(body["image_base64"])
        except Exception:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Invalid base64 image data."))
        if len(image_data) > _AI_API_MAX_IMAGE_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Image too large (max 10 MB)."))

    if body.get("audio_base64"):
        audio_mime = body.get("audio_mime", "audio/mpeg")
        if audio_mime not in _AI_ALLOWED_AUDIO_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported audio type: {audio_mime}"))
        try:
            audio_data = base64.b64decode(body["audio_base64"])
        except Exception:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Invalid base64 audio data."))
        if len(audio_data) > _AI_API_MAX_AUDIO_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Audio too large (max 50 MB)."))

    return message, history, image_data, audio_data, image_mime, audio_mime, do_stream, None


@app_flask.route("/ai-assistant", methods=["POST", "OPTIONS"])
def ai_assistant():
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key",
        })
        return resp

    if not _check_ai_api_key():
        return _ai_cors(_ai_error("Unauthorized — invalid or missing API key.", 401))

    content_type = request.content_type or ""
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        message, history, image_data, audio_data, image_mime, audio_mime, do_stream, err = \
            _parse_multipart_ai_request()
    elif "application/json" in content_type:
        message, history, image_data, audio_data, image_mime, audio_mime, do_stream, err = \
            _parse_json_ai_request()
    else:
        return _ai_cors(_ai_error("Content-Type must be multipart/form-data or application/json."))

    if err is not None:
        return err

    if not message and image_data is None and audio_data is None:
        return _ai_cors(_ai_error("Provide at least one of: message, image, or audio."))

    if message and len(message) > _AI_API_MAX_MESSAGE_CHARS:
        return _ai_cors(_ai_error(f"Message too long (max {_AI_API_MAX_MESSAGE_CHARS} chars)."))

    if do_stream:
        return _ai_cors(_ai_error("Streaming is disabled for Hugging Face provider. Send stream=false.", 400))

    try:
        if image_data is not None:
            ocr_text, ocr_provider, ocr_model = ask_ocr_image(image_data, image_mime or "image/jpeg")
            if message:
                prompt = (
                    f"User instruction: {message}\n\n"
                    f"OCR text extracted from image:\n{ocr_text}\n\n"
                    "Answer using the OCR text above."
                )
                reply_text, model_used, tokens_used = ai_text_reply(prompt, history)
            else:
                reply_text  = ocr_text
                model_used  = ocr_model
                tokens_used = None

            return _ai_cors(jsonify({
                "ok": True,
                "reply": reply_text,
                "ocr_text": ocr_text,
                "detected_language": _detect_lang(reply_text or ocr_text or ""),
                "tokens_used": tokens_used,
                "model": model_used,
                "ocr_model": ocr_model,
                "ocr_provider": ocr_provider,
                "provider": AI_PROVIDER,
            }))

        if audio_data is not None:
            if _gemini is None:
                return _ai_cors(_ai_error(
                    "Audio transcription requires Gemini. Set GEMINI_API_KEY and GEMINI_MODEL.", 503
                ))
            contents = _build_gemini_contents(
                message, history, image_data, audio_data, image_mime, audio_mime
            )
            response = _gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=_ai_gen_config(),
            )
            reply_text = (response.text or "").strip()
            return _ai_cors(jsonify({
                "ok": True,
                "reply": reply_text,
                "detected_language": _detect_lang(reply_text or message or ""),
                "tokens_used": None,
                "model": GEMINI_MODEL,
                "provider": "gemini-legacy",
            }))

        if _hf_client is None:
            return _ai_cors(_ai_error("Hugging Face is not configured. Set HF_TOKEN.", 503))

        reply_text, model_used, tokens_used = ai_text_reply(message, history)
        return _ai_cors(jsonify({
            "ok": True,
            "reply": reply_text,
            "detected_language": _detect_lang(reply_text or message or ""),
            "tokens_used": tokens_used,
            "model": model_used,
            "provider": "hf",
        }))

    except Exception as exc:
        logging.getLogger(__name__).error(f"/ai-assistant error: {exc}", exc_info=True)
        return _ai_cors(_ai_error(f"AI generation failed: {exc}", 500))


@app_flask.route("/ai-assistant/transcribe", methods=["POST", "OPTIONS"])
def ai_transcribe():
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key",
        })
        return resp

    if not _check_ai_api_key():
        return _ai_cors(_ai_error("Unauthorized.", 401))
    if _gemini is None:
        return _ai_cors(_ai_error("AI service not configured.", 503))

    aud_file = request.files.get("audio")
    if not aud_file:
        return _ai_cors(_ai_error("No audio file provided (field name: 'audio')."))

    audio_mime = aud_file.mimetype or "audio/mpeg"
    if audio_mime not in _AI_ALLOWED_AUDIO_MIME:
        return _ai_cors(_ai_error(f"Unsupported audio type: {audio_mime}"))

    audio_data = aud_file.read()
    if len(audio_data) > _AI_API_MAX_AUDIO_BYTES:
        return _ai_cors(_ai_error("Audio too large (max 50 MB)."))

    lang_hint = request.form.get("language", "auto")
    lang_instruction = {
        "km": " The audio is in Khmer (ភាសាខ្មែរ).",
        "en": " The audio is in English.",
    }.get(lang_hint, "")

    from google.genai import types as _gtypes
    try:
        response = _gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                _gtypes.Part.from_bytes(data=audio_data, mime_type=audio_mime),
                (
                    "Transcribe this audio exactly as spoken. "
                    "Output ONLY the transcribed text — no labels, no explanation, "
                    "no timestamps." + lang_instruction
                ),
            ],
        )
        transcript = (response.text or "").strip()
        return _ai_cors(jsonify({
            "ok": True,
            "transcript": transcript,
            "detected_language": _detect_lang(transcript),
            "model": GEMINI_MODEL,
        }))
    except Exception as exc:
        logging.getLogger(__name__).error(f"/ai-assistant/transcribe error: {exc}")
        return _ai_cors(_ai_error(f"Transcription failed: {exc}", 500))


@app_flask.route("/ai-assistant/vision", methods=["POST", "OPTIONS"])
def ai_vision():
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key",
        })
        return resp

    if not _check_ai_api_key():
        return _ai_cors(_ai_error("Unauthorized.", 401))
    if not _ocr_configured():
        return _ai_cors(_ai_error(_ocr_status_for_user(), 503))

    image_data = None
    image_mime  = "image/jpeg"
    content_type = request.content_type or ""

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        img_file = request.files.get("image")
        if not img_file:
            return _ai_cors(_ai_error("No image file provided (field name: 'image')."))
        image_mime = img_file.mimetype or "image/jpeg"
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        image_data = img_file.read()
    elif "application/json" in content_type:
        try:
            body = request.get_json(force=True) or {}
        except Exception:
            return _ai_cors(_ai_error("Invalid JSON."))
        image_mime = body.get("image_mime", "image/jpeg")
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        try:
            image_data = base64.b64decode(body.get("image_base64", ""))
        except Exception:
            return _ai_cors(_ai_error("Invalid base64 image."))
    else:
        return _ai_cors(_ai_error("Content-Type must be multipart/form-data or application/json."))

    if not image_data:
        return _ai_cors(_ai_error("Empty image data."))
    if len(image_data) > _AI_API_MAX_IMAGE_BYTES:
        return _ai_cors(_ai_error("Image too large (max 10 MB)."))

    try:
        result_text, ocr_provider, ocr_model = ask_ocr_image(image_data, image_mime)
        return _ai_cors(jsonify({
            "ok": True,
            "result": result_text,
            "detected_language": _detect_lang(result_text),
            "model": ocr_model,
            "provider": ocr_provider,
            "ocr_provider": ocr_provider,
        }))
    except Exception as exc:
        logging.getLogger(__name__).error(f"/ai-assistant/vision OCR error: {exc}")
        return _ai_cors(_ai_error(f"Vision OCR failed: {exc}", 500))


@app_flask.route("/ai-assistant/info", methods=["GET"])
def ai_info():
    resp = jsonify({
        "ok": True,
        "provider": AI_PROVIDER,
        "model": HF_MODEL,
        "ocr_model": HF_OCR_MODEL,
        "ocr_provider": OCR_PROVIDER,
        "features": ["chat", "multi-turn-history", "image-ocr", "khmer-language", "english-language", "admin-generated-api-keys"],
        "providers": {
            "active": AI_PROVIDER,
            "huggingface_available": _hf_client is not None,
            "huggingface_model": HF_MODEL,
            "huggingface_ocr_model": HF_OCR_MODEL,
            "ocr_provider": OCR_PROVIDER,
            "ocr_configured": _ocr_configured(),
            "hf_ocr_temporarily_disabled": _hf_ocr_is_temporarily_disabled() or _ocr_provider_is_temporarily_disabled("hf"),
            "gemini_ocr_temporarily_disabled": _ocr_provider_is_temporarily_disabled("gemini"),
            "gemini_legacy_available": _gemini is not None,
            "gemini_legacy_model": GEMINI_MODEL or None,
        },
        "limits": {
            "max_message_chars": _AI_API_MAX_MESSAGE_CHARS,
            "max_image_bytes":   _AI_API_MAX_IMAGE_BYTES,
            "max_audio_bytes":   _AI_API_MAX_AUDIO_BYTES,
            "max_history_turns": _AI_API_MAX_HISTORY_TURNS,
        },
        "supported_image_types": sorted(_AI_ALLOWED_IMAGE_MIME),
        "supported_audio_types": sorted(_AI_ALLOWED_AUDIO_MIME),
        "auth_required": True,
        "auth_configured": bool(os.environ.get("AI_API_KEY", "").strip()) or _dynamic_ai_auth_configured(),
        "static_auth_configured": bool(os.environ.get("AI_API_KEY", "").strip()),
        "dynamic_auth_configured": _dynamic_ai_auth_configured(),
    })
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


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
for _noisy in ("httpx", "httpcore", "urllib3", "telegram",
               "google_genai.models", "google.genai", "google"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config  (populated by _init_clients)
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = ""
SB_URL:             str = ""
SB_KEY:             str = ""
REDIS_URL:          str = ""
REDIS_CACHE_PREFIX       = os.environ.get("REDIS_CACHE_PREFIX", "tgbot")
REDIS_PREFS_TTL_S        = int(os.environ.get("REDIS_PREFS_TTL_S", "1800"))
REDIS_TEXT_CACHE_TTL_S   = int(os.environ.get("REDIS_TEXT_CACHE_TTL_S", "86400"))
REDIS_HISTORY_TTL_S      = int(os.environ.get("REDIS_HISTORY_TTL_S", "86400"))
REDIS_SOCKET_TIMEOUT_S   = float(os.environ.get("REDIS_SOCKET_TIMEOUT_S", "3"))
GEMINI_API_KEY:     str = ""
ADMIN_IDS:  set[int]    = set()
GEMINI_MODEL            = ""
HF_TOKEN                = ""
HF_MODEL                = "Qwen/Qwen2.5-7B-Instruct"
HF_OCR_MODEL            = "microsoft/trocr-base-printed"
AI_PROVIDER             = "hf"
OCR_PROVIDER            = "auto"   # auto | hf | gemini
OCR_TIMEOUT_SECONDS     = 90
HF_OCR_DNS_COOLDOWN_S   = 300.0
OCR_PROVIDER_FAILURE_LIMIT = max(1, int(os.environ.get("OCR_PROVIDER_FAILURE_LIMIT", "3")))
OCR_PROVIDER_COOLDOWN_S    = float(os.environ.get("OCR_PROVIDER_COOLDOWN_S", "300"))
MAX_VOICE_BYTES         = 20 * 1024 * 1024
MAX_AUDIO_FILE_BYTES    = 50 * 1024 * 1024
MAX_INPUT_CHARS         = 5_000
TTS_CHUNK_CHARS         = 900
# Edge TTS can intermittently return NoAudioReceived on Render when a voice
# is busy, text is too long, or websocket audio chunks stop arriving.
# These values keep each Edge request small and allow retry/fallback voices.
EDGE_TTS_CHUNK_CHARS      = max(120, min(700, int(os.environ.get("EDGE_TTS_CHUNK_CHARS", "600"))))
EDGE_TTS_RETRIES          = max(1, int(os.environ.get("EDGE_TTS_RETRIES", "3")))
EDGE_TTS_RETRY_DELAY_S    = max(0.2, float(os.environ.get("EDGE_TTS_RETRY_DELAY_S", "0.8")))
EDGE_TTS_STREAM_TIMEOUT_S = max(15.0, float(os.environ.get("EDGE_TTS_STREAM_TIMEOUT_S", "45")))
EDGE_TTS_CROSS_LANG_FALLBACK = os.environ.get("EDGE_TTS_CROSS_LANG_FALLBACK", "0") == "1"
DEFAULT_SPEED           = 1.0
TELE_MSG_LIMIT          = 4000
USER_COOLDOWN_S         = 3.0
_STALE_GRACE_S          = 30.0
_KHMER_RE               = re.compile(r"[\u1780-\u17FF]")
_SPEED_MIN              = 0.25
_SPEED_MAX              = 4.0
_SENTENCE_RE            = re.compile(r"(?<=[។!\?\.。])\s*")

# ---------------------------------------------------------------------------
# Supported audio file extensions
# ---------------------------------------------------------------------------
_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".oga",
    ".flac", ".aac", ".wma", ".opus", ".webm", ".aiff", ".aif",
}
_AUDIO_MIME_PREFIXES = ("audio/", "video/mp4", "video/webm")


def _is_audio_file(filename: str | None, mime_type: str | None) -> bool:
    if filename and os.path.splitext(filename.lower())[1] in _AUDIO_EXTENSIONS:
        return True
    if mime_type:
        for prefix in _AUDIO_MIME_PREFIXES:
            if mime_type.lower().startswith(prefix):
                return True
    return False


def _audio_mime_for_gemini(filename: str | None, mime_type: str | None) -> str:
    if mime_type:
        mt = mime_type.lower()
        if mt.startswith("audio/") or mt in ("video/mp4", "video/webm"):
            return mt
    ext_map = {
        ".wav":  "audio/wav",  ".mp3":  "audio/mpeg", ".mp4":  "audio/mp4",
        ".m4a":  "audio/mp4",  ".ogg":  "audio/ogg",  ".oga":  "audio/ogg",
        ".flac": "audio/flac", ".aac":  "audio/aac",  ".opus": "audio/opus",
        ".webm": "audio/webm", ".aiff": "audio/aiff", ".aif":  "audio/aiff",
        ".wma":  "audio/x-ms-wma",
    }
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        return ext_map.get(ext, "audio/mpeg")
    return "audio/mpeg"


_DB_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(4, MAX_CONCURRENT_TTS_USERS), thread_name_prefix="db_write"
)
_AI_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, MAX_CONCURRENT_AI), thread_name_prefix="ai"
)

_AI_SEMAPHORE:         asyncio.Semaphore | None = None
_BROADCAST_SEMAPHORE:  asyncio.Semaphore | None = None
_TTS_CHUNK_SEMAPHORE:  asyncio.Semaphore | None = None

# ---------------------------------------------------------------------------
# Redis/Supabase safe retry + cache helpers
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """Small circuit breaker to avoid hammering Redis/Supabase during outages."""
    def __init__(self, name: str, max_failures: int = 5, reset_after: float = 30.0):
        self.name = name
        self.max_failures = max(1, int(max_failures))
        self.reset_after = max(1.0, float(reset_after))
        self.failures = 0
        self.open_until = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self.open_until

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.open_until = 0.0

    def record_failure(self, exc: BaseException | str) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.max_failures:
                self.open_until = time.monotonic() + self.reset_after
                logger.warning(
                    "%s circuit breaker opened for %.0fs after %d failure(s): %s",
                    self.name,
                    self.reset_after,
                    self.failures,
                    str(exc)[:240],
                )


supabase_breaker = CircuitBreaker(
    "Supabase",
    max_failures=int(os.environ.get("SUPABASE_BREAKER_FAILURES", "5")),
    reset_after=float(os.environ.get("SUPABASE_BREAKER_RESET_S", "30")),
)
redis_breaker = CircuitBreaker(
    "Redis",
    max_failures=int(os.environ.get("REDIS_BREAKER_FAILURES", "5")),
    reset_after=float(os.environ.get("REDIS_BREAKER_RESET_S", "20")),
)

_LOG_ONCE_TTL_S = float(os.environ.get("LOG_ONCE_TTL_S", "60"))
_log_once_seen: dict[str, float] = {}


def _log_once(level: int, key: str, message: str, *args, exc_info=False) -> None:
    """Log repeated outage errors once per TTL so callbacks don't flood logs."""
    now = time.monotonic()
    last = _log_once_seen.get(key, 0.0)
    if now - last < _LOG_ONCE_TTL_S:
        return
    _log_once_seen[key] = now
    logger.log(level, message, *args, exc_info=exc_info)


def _is_retryable_store_error(exc: BaseException | str) -> bool:
    msg = str(exc).lower()
    retryable_words = (
        "server disconnected", "connection reset", "connection aborted",
        "connection refused", "temporarily unavailable", "temporary failure",
        "timeout", "timed out", "read timed out", "network", "dns",
        "name resolution", "too many requests", "rate limit", "429",
        "500", "502", "503", "504", "bad gateway", "service unavailable",
        "gateway timeout", "max retries exceeded", "connectionerror",
    )
    non_retryable_words = (
        "400", "401", "403", "unauthorized", "forbidden", "invalid api key",
        "invalid key", "permission denied", "violates row-level security",
        "row-level security", "rls", "relation does not exist",
        "column does not exist", "undefined column", "syntax error",
        "invalid input syntax",
    )
    if any(word in msg for word in non_retryable_words):
        return False
    if any(word in msg for word in retryable_words):
        return True
    name = exc.__class__.__name__.lower() if not isinstance(exc, str) else ""
    return any(word in name for word in ("timeout", "connection", "network"))


def retry_call_sync(
    name: str,
    factory: Callable[[], Any],
    *,
    default: Any = None,
    attempts: int = 3,
    base_delay: float = 0.35,
    max_delay: float = 2.0,
    breaker: CircuitBreaker | None = None,
    critical: bool = False,
) -> Any:
    """Synchronous retry wrapper for Supabase/Redis clients."""
    if breaker and breaker.is_open():
        _log_once(logging.WARNING, f"{name}:breaker_open", "%s skipped: circuit breaker is open", name)
        if critical:
            raise RuntimeError(f"{name} unavailable: circuit breaker open")
        return default

    last_exc: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            result = factory()
            if breaker:
                breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_store_error(exc)
            if not retryable:
                if breaker:
                    breaker.record_failure(exc)
                _log_once(
                    logging.ERROR,
                    f"{name}:non_retryable:{type(exc).__name__}:{str(exc)[:120]}",
                    "%s failed with non-retryable error: %s",
                    name,
                    exc,
                )
                if critical:
                    raise
                return default

            if attempt >= attempts:
                if breaker:
                    breaker.record_failure(exc)
                _log_once(
                    logging.WARNING,
                    f"{name}:failed:{type(exc).__name__}:{str(exc)[:120]}",
                    "%s failed after %d attempt(s): %s",
                    name,
                    attempts,
                    exc,
                )
                break

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning("%s temporary error attempt %d/%d: %s", name, attempt, attempts, exc)
            time.sleep(delay)

    if critical and last_exc:
        raise last_exc
    return default


async def retry_call(
    name: str,
    factory: Callable[[], Any],
    *,
    default: Any = None,
    attempts: int = 3,
    timeout: float = 8.0,
    base_delay: float = 0.35,
    max_delay: float = 2.0,
    breaker: CircuitBreaker | None = None,
    critical: bool = False,
) -> Any:
    """Async-safe retry wrapper. Blocking SDK calls run in a thread."""
    if breaker and breaker.is_open():
        _log_once(logging.WARNING, f"{name}:breaker_open", "%s skipped: circuit breaker is open", name)
        if critical:
            raise RuntimeError(f"{name} unavailable: circuit breaker open")
        return default

    last_exc: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            result = await asyncio.wait_for(asyncio.to_thread(factory), timeout=timeout)
            if breaker:
                breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_store_error(exc)
            if not retryable:
                if breaker:
                    breaker.record_failure(exc)
                _log_once(
                    logging.ERROR,
                    f"{name}:non_retryable:{type(exc).__name__}:{str(exc)[:120]}",
                    "%s failed with non-retryable error: %s",
                    name,
                    exc,
                )
                if critical:
                    raise
                return default

            if attempt >= attempts:
                if breaker:
                    breaker.record_failure(exc)
                _log_once(
                    logging.WARNING,
                    f"{name}:failed:{type(exc).__name__}:{str(exc)[:120]}",
                    "%s failed after %d attempt(s): %s",
                    name,
                    attempts,
                    exc,
                )
                break

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning("%s temporary error attempt %d/%d: %s", name, attempt, attempts, exc)
            await asyncio.sleep(delay)

    if critical and last_exc:
        raise last_exc
    return default


def db_call_sync(name: str, factory: Callable[[], Any], *, default: Any = None, attempts: int = 3, critical: bool = False) -> Any:
    return retry_call_sync(
        f"Supabase:{name}",
        factory,
        default=default,
        attempts=attempts,
        breaker=supabase_breaker,
        critical=critical,
    )


async def db_call(name: str, factory: Callable[[], Any], *, default: Any = None, attempts: int = 3, timeout: float = 8.0, critical: bool = False) -> Any:
    return await retry_call(
        f"Supabase:{name}",
        factory,
        default=default,
        attempts=attempts,
        timeout=timeout,
        breaker=supabase_breaker,
        critical=critical,
    )


def redis_call_sync(name: str, factory: Callable[[], Any], *, default: Any = None, attempts: int = 2, critical: bool = False) -> Any:
    if redis_client is None:
        return default
    return retry_call_sync(
        f"Redis:{name}",
        factory,
        default=default,
        attempts=attempts,
        breaker=redis_breaker,
        critical=critical,
    )


async def redis_call(name: str, factory: Callable[[], Any], *, default: Any = None, attempts: int = 2, timeout: float = 3.0, critical: bool = False) -> Any:
    if redis_client is None:
        return default
    return await retry_call(
        f"Redis:{name}",
        factory,
        default=default,
        attempts=attempts,
        timeout=timeout,
        breaker=redis_breaker,
        critical=critical,
    )


def _redis_key(*parts: Any) -> str:
    clean = [str(p).replace(" ", "_").replace("\n", "_") for p in parts]
    return f"{REDIS_CACHE_PREFIX}:" + ":".join(clean)


def _redis_get_json_sync(key: str, default: Any = None) -> Any:
    raw = redis_call_sync(f"get:{key}", lambda: redis_client.get(key), default=None)
    if not raw:
        return default
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return _json.loads(raw)
    except Exception as exc:
        _log_once(logging.WARNING, f"redis_bad_json:{key}", "Invalid Redis JSON for %s: %s", key, exc)
        return default


def _redis_set_json_sync(key: str, value: Any, ttl: int) -> bool:
    raw = _json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    ok = redis_call_sync(f"set:{key}", lambda: redis_client.setex(key, max(1, int(ttl)), raw), default=False)
    return bool(ok)


def _redis_delete_sync(key: str) -> None:
    redis_call_sync(f"delete:{key}", lambda: redis_client.delete(key), default=None)


def _redis_scan_keys_sync(pattern: str, limit: int = 200) -> list[str]:
    if redis_client is None:
        return []
    keys: list[str] = []
    try:
        for key in redis_client.scan_iter(match=pattern, count=100):
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="ignore")
            keys.append(str(key))
            if len(keys) >= max(1, int(limit or 200)):
                break
    except Exception as exc:
        _log_once(logging.WARNING, "redis_scan_failed", "Redis scan failed: %s", exc)
    return keys


async def _redis_get_json(key: str, default: Any = None) -> Any:
    raw = await redis_call(f"get:{key}", lambda: redis_client.get(key), default=None)
    if not raw:
        return default
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return _json.loads(raw)
    except Exception as exc:
        _log_once(logging.WARNING, f"redis_bad_json:{key}", "Invalid Redis JSON for %s: %s", key, exc)
        return default


async def _redis_set_json(key: str, value: Any, ttl: int) -> bool:
    raw = _json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    ok = await redis_call(f"set:{key}", lambda: redis_client.setex(key, max(1, int(ttl)), raw), default=False)
    return bool(ok)


async def _redis_delete(key: str) -> None:
    await redis_call(f"delete:{key}", lambda: redis_client.delete(key), default=None)


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
BROADCAST_WAIT_MESSAGE = 1
CHAT_WAIT_MESSAGE      = 2
SCHED_WAIT_MSG         = 3
SCHED_WAIT_TIME        = 4
USER_SEARCH_WAIT_QUERY = 5
SCHED_EDIT_WAIT_TIME   = 6
SCHED_EDIT_WAIT_TEXT   = 7
SCHED_EDIT_WAIT_PHOTO  = 8
_SCHED_POLL_INTERVAL   = int(os.environ.get("SCHED_POLL_INTERVAL", "60"))
_SCHED_SENDING_STALE_SECONDS = int(os.environ.get("SCHED_SENDING_STALE_SECONDS", "1800"))
_SCHED_DUE_LIMIT      = max(1, int(os.environ.get("SCHED_DUE_LIMIT", "5")))
_SCHED_SCAN_LIMIT     = max(25, int(os.environ.get("SCHED_SCAN_LIMIT", "250")))
_SCHED_LOCK_ENABLED   = os.environ.get("SCHED_LOCK_ENABLED", "1") == "1"
_SCHED_LOCK_REQUIRED  = os.environ.get("SCHED_LOCK_REQUIRED", "0") == "1"
_SCHED_LOCK_KEY       = os.environ.get("SCHED_LOCK_KEY", "scheduled_broadcast_runner").strip() or "scheduled_broadcast_runner"
_SCHED_LOCK_TTL_S     = max(30, int(os.environ.get("SCHED_LOCK_TTL_S", str(max(90, _SCHED_POLL_INTERVAL * 3)))))
_BOT_LOCK_OWNER       = os.environ.get("BOT_LOCK_OWNER", "").strip() or (
    f"{os.environ.get('RENDER_SERVICE_NAME', 'bot')}:{os.environ.get('RENDER_INSTANCE_ID', 'local')}:{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(4)}"
)
_DT_FORMATS = [
    "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M",
]

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
_pending_broadcast:  dict[int, dict] = {}
_admin_chat_target:  dict[int, int]  = {}
_user_to_admin:      dict[int, int]  = {}
_sched_payload:      dict[int, dict] = {}

_USER_LAST_TTS_MAX = 10_000
_user_last_tts: OrderedDict[int, float] = OrderedDict()


def _set_last_tts(user_id: int) -> None:
    _user_last_tts.pop(user_id, None)
    _user_last_tts[user_id] = time.monotonic()
    while len(_user_last_tts) > _USER_LAST_TTS_MAX:
        _user_last_tts.popitem(last=False)


def _get_last_tts(user_id: int) -> float:
    return _user_last_tts.get(user_id, 0.0)


# ---------------------------------------------------------------------------
# Last TTS text fallback cache
# ---------------------------------------------------------------------------
_LAST_TTS_TEXT_MAX = 10_000
_last_tts_text: OrderedDict[int, tuple[str, float]] = OrderedDict()


def set_last_tts_text(user_id: int, text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    _last_tts_text.pop(user_id, None)
    _last_tts_text[user_id] = (text, time.monotonic())
    while len(_last_tts_text) > _LAST_TTS_TEXT_MAX:
        _last_tts_text.popitem(last=False)


def get_last_tts_text(user_id: int) -> str | None:
    item = _last_tts_text.get(user_id)
    if not item:
        return None
    _last_tts_text.move_to_end(user_id)
    text, _ = item
    return text


# ---------------------------------------------------------------------------
# Supabase + AI clients
# ---------------------------------------------------------------------------
supabase: Client | None = None
redis_client = None
_gemini    = None
_hf_client = None

# OCR circuit-breaker state
_hf_ocr_disabled_until = 0.0
_hf_ocr_failures       = 0
_hf_ocr_state_lock     = threading.Lock()

# Generic OCR provider circuit breaker.
_ocr_provider_disabled_until: dict[str, float] = {"hf": 0.0, "gemini": 0.0}
_ocr_provider_failures: dict[str, int] = {"hf": 0, "gemini": 0}
_ocr_provider_state_lock = threading.Lock()


def _ocr_provider_is_temporarily_disabled(provider: str) -> bool:
    provider = (provider or "").lower().strip()
    with _ocr_provider_state_lock:
        return time.monotonic() < _ocr_provider_disabled_until.get(provider, 0.0)


def _mark_ocr_provider_success(provider: str) -> None:
    provider = (provider or "").lower().strip()
    with _ocr_provider_state_lock:
        _ocr_provider_failures[provider] = 0
        _ocr_provider_disabled_until[provider] = 0.0


def _mark_ocr_provider_failure(provider: str, exc: Exception | str) -> None:
    provider = (provider or "").lower().strip()
    with _ocr_provider_state_lock:
        failures = _ocr_provider_failures.get(provider, 0) + 1
        _ocr_provider_failures[provider] = failures
        if _is_dns_or_network_error(exc) or failures >= OCR_PROVIDER_FAILURE_LIMIT:
            cooldown = HF_OCR_DNS_COOLDOWN_S if provider == "hf" and _is_dns_or_network_error(exc) else OCR_PROVIDER_COOLDOWN_S
            _ocr_provider_disabled_until[provider] = time.monotonic() + cooldown
            logger.warning(
                "OCR provider %s temporarily disabled for %.0fs after %d failure(s): %s",
                provider, cooldown, failures, str(exc)[:300],
            )


def _init_hf_client() -> None:
    global _hf_client
    if InferenceClient is None:
        logger.error("huggingface_hub not installed. Run: pip install -U huggingface_hub")
        _hf_client = None
        return
    if not HF_TOKEN:
        logger.error("HF_TOKEN not set — Hugging Face disabled.")
        _hf_client = None
        return
    try:
        _hf_client = InferenceClient(token=HF_TOKEN)
        logger.info(f"Hugging Face client initialised (chat={HF_MODEL}, ocr={HF_OCR_MODEL}).")
    except Exception as e:
        logger.error(f"Hugging Face init failed: {e}")
        _hf_client = None


def _hf_history_messages(history: list[dict] | None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _AI_SYSTEM_PROMPT}]
    for turn in (history or [])[-_AI_API_MAX_HISTORY_TURNS:]:
        role    = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        role = "assistant" if role in ("assistant", "model", "bot") else "user"
        messages.append({"role": role, "content": content})
    return messages


def ask_huggingface(prompt: str, history: list[dict] | None = None) -> str:
    """Generate a text reply with Hugging Face Inference Providers (3 retries)."""
    if _hf_client is None:
        raise RuntimeError("Hugging Face client is not configured.")

    prompt   = (prompt or "").strip() or "Hello"
    messages = _hf_history_messages(history)
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = _hf_client.chat_completion(
                model=HF_MODEL,
                messages=messages,
                max_tokens=int(os.environ.get("HF_MAX_TOKENS", "1024")),
                temperature=float(os.environ.get("HF_TEMPERATURE", "0.7")),
            )
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "".join(
                    str(p.get("text", p)) if isinstance(p, dict) else str(p)
                    for p in content
                )
            reply = str(content or "").strip()
            if reply:
                return reply
            raise RuntimeError("Hugging Face returned an empty response.")
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))

    raise RuntimeError(f"Hugging Face generation failed: {last_exc}")


def _extract_hf_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    for attr in ("generated_text", "text"):
        value = getattr(result, attr, None)
        if value:
            return str(value).strip()
    if isinstance(result, dict):
        value = result.get("generated_text") or result.get("text")
        if value:
            return str(value).strip()
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return str(result).strip()
    if isinstance(result, list):
        texts = [_extract_hf_text(item) for item in result]
        return "\n".join(t for t in texts if t).strip()
    return str(result).strip()


def _is_dns_or_network_error(exc: Exception | str) -> bool:
    msg = str(exc).lower()
    return any(needle in msg for needle in (
        "getaddrinfo failed", "failed to resolve", "nameresolutionerror",
        "temporary failure in name resolution", "nodename nor servname",
        "connectionerror", "maxretryerror", "name or service not known",
        "no address associated with hostname", "dns",
    ))


def _friendly_ocr_error(errors: list[str]) -> RuntimeError:
    joined = " | ".join(errors)
    low    = joined.lower()

    if _is_dns_or_network_error(joined):
        return RuntimeError(
            "OCR network/DNS failed. Cannot reach api-inference.huggingface.co. "
            "Set OCR_PROVIDER=gemini with GEMINI_API_KEY/GEMINI_MODEL for fallback, "
            f"or fix DNS/firewall. Details: {joined[:900]}"
        )
    if "stopiteration" in low:
        return RuntimeError(
            "Hugging Face OCR model is not available for hosted inference. "
            "Try HF_OCR_MODEL=microsoft/trocr-base-printed, or set OCR_PROVIDER=gemini. "
            f"Details: {joined[:900]}"
        )
    if "401" in low or "unauthorized" in low:
        return RuntimeError("HF_TOKEN is invalid or expired. Create a new token and update env.")
    if "403" in low or "forbidden" in low:
        return RuntimeError("HF_TOKEN lacks permission for this Hugging Face model.")
    return RuntimeError("OCR failed. " + joined[:1200])


def _hf_ocr_is_temporarily_disabled() -> bool:
    with _hf_ocr_state_lock:
        return time.monotonic() < _hf_ocr_disabled_until


def _mark_hf_ocr_success() -> None:
    global _hf_ocr_failures, _hf_ocr_disabled_until
    with _hf_ocr_state_lock:
        _hf_ocr_failures       = 0
        _hf_ocr_disabled_until = 0.0


def _mark_hf_ocr_failure(exc: Exception | str) -> None:
    global _hf_ocr_failures, _hf_ocr_disabled_until
    with _hf_ocr_state_lock:
        _hf_ocr_failures += 1
        if _is_dns_or_network_error(exc):
            _hf_ocr_disabled_until = time.monotonic() + HF_OCR_DNS_COOLDOWN_S
        elif _hf_ocr_failures >= 3:
            _hf_ocr_disabled_until = time.monotonic() + 60.0


def _ocr_configured() -> bool:
    p  = (OCR_PROVIDER or "auto").lower().strip()
    hf = _hf_client is not None and not _hf_ocr_is_temporarily_disabled() and not _ocr_provider_is_temporarily_disabled("hf")
    gm = _gemini is not None and not _ocr_provider_is_temporarily_disabled("gemini")
    if p == "hf":
        return hf
    if p == "gemini":
        return gm
    return hf or gm


def _ocr_status_for_user() -> str:
    p = (OCR_PROVIDER or "auto").lower().strip()
    if p == "hf" and _hf_client is None:
        return "❌ OCR មិនទាន់ Activate ទេ។ សូម Set HF_TOKEN ឬប្ដូរ OCR_PROVIDER=gemini។"
    if p == "gemini" and _gemini is None:
        return "❌ OCR Gemini មិនទាន់ Activate ទេ។ សូម Set GEMINI_API_KEY និង GEMINI_MODEL។"
    if _hf_client is None and _gemini is None:
        return "❌ OCR មិនទាន់ Activate ទេ។ សូម Set HF_TOKEN ឬ GEMINI_API_KEY/GEMINI_MODEL។"
    if p == "hf" and (_hf_ocr_is_temporarily_disabled() or _ocr_provider_is_temporarily_disabled("hf")):
        return "❌ Hugging Face OCR ត្រូវបានបិទបណ្តោះអាសន្ន ព្រោះមាន network/API errors ជាប់គ្នា។"
    if p == "gemini" and _ocr_provider_is_temporarily_disabled("gemini"):
        return "❌ Gemini OCR ត្រូវបានបិទបណ្តោះអាសន្ន ព្រោះមាន network/API errors ជាប់គ្នា។"
    return "❌ OCR មិនអាចប្រើបាននៅពេលនេះ។ សូមពិនិត្យ API key និង network។"


def _hf_ocr_via_rest(image_data: bytes) -> str:
    """Fallback OCR via raw Hugging Face Inference API."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")

    model_id  = (HF_OCR_MODEL or "microsoft/trocr-base-printed").strip()
    url_model = requests.utils.quote(model_id, safe="")
    url       = f"https://api-inference.huggingface.co/models/{url_model}"
    headers   = {
        "Authorization":    f"Bearer {HF_TOKEN}",
        "Accept":           "application/json",
        "Content-Type":     "application/octet-stream",
        "X-Wait-For-Model": "true",
    }

    last_body = ""
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, data=image_data, timeout=OCR_TIMEOUT_SECONDS)
            last_body = resp.text[:700]

            if resp.status_code in (503, 504, 429) and attempt < 2:
                wait_s = 1.5 * (attempt + 1)
                try:
                    wait_s = max(wait_s, float(resp.json().get("estimated_time", wait_s)))
                except Exception:
                    pass
                time.sleep(min(wait_s, 10.0))
                continue

            if not (200 <= resp.status_code < 300):
                raise RuntimeError(f"HTTP {resp.status_code}: {last_body}")

            try:
                payload = resp.json()
            except Exception:
                payload = resp.text

            text = _extract_hf_text(payload)
            if text:
                return text
            raise RuntimeError(f"REST OCR returned empty result: {payload!r}")

        except Exception as e:
            if attempt >= 2:
                if _is_dns_or_network_error(e):
                    raise RuntimeError(
                        f"REST OCR network/DNS failed: {type(e).__name__}: {e}; body={last_body!r}"
                    ) from e
                raise RuntimeError(
                    f"REST OCR failed: {type(e).__name__}: {e!r}; body={last_body!r}"
                ) from e
            time.sleep(0.8 * (attempt + 1))

    raise RuntimeError("REST OCR failed after retries.")


def ask_huggingface_ocr(image_data: bytes) -> str:
    if _hf_client is None:
        raise RuntimeError("Hugging Face client is not configured.")
    if not image_data:
        raise RuntimeError("Empty image data.")
    if _hf_ocr_is_temporarily_disabled():
        raise RuntimeError("Hugging Face OCR temporarily disabled after recent failures.")

    errors: list[str] = []

    for attempt in range(3):
        try:
            result = _hf_client.image_to_text(image=image_data, model=HF_OCR_MODEL)
            text   = _extract_hf_text(result)
            if text:
                _mark_hf_ocr_success()
                return text
            raise RuntimeError(f"SDK OCR returned empty result: {result!r}")
        except Exception as e:
            errors.append(f"sdk_attempt_{attempt + 1}={type(e).__name__}: {e!r}")
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))

    try:
        text = _hf_ocr_via_rest(image_data)
        _mark_hf_ocr_success()
        return text
    except Exception as e:
        errors.append(f"rest={type(e).__name__}: {e!r}")
        _mark_hf_ocr_failure(e)

    raise _friendly_ocr_error(errors)


def ask_gemini_ocr(image_data: bytes, mime_type: str = "image/jpeg") -> str:
    if _gemini is None:
        raise RuntimeError("Gemini OCR is not configured. Set GEMINI_API_KEY and GEMINI_MODEL.")
    if not image_data:
        raise RuntimeError("Empty image data.")
    from google.genai import types as _gtypes
    prompt = (
        "Extract all readable text from this image. Preserve Khmer and English exactly. "
        "Keep useful line breaks. If there is no readable text, output only NOTEXT. "
        "Do not describe the image and do not add explanations."
    )
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            _gtypes.Part.from_bytes(data=image_data, mime_type=mime_type or "image/jpeg"),
            prompt,
        ],
    )
    text = (response.text or "").strip()
    return text or "NOTEXT"


def ask_ocr_image(image_data: bytes, mime_type: str = "image/jpeg") -> tuple[str, str, str]:
    """Unified OCR with provider fallback. Returns (text, provider, model)."""
    if not image_data:
        raise RuntimeError("Empty image data.")

    provider = (OCR_PROVIDER or "auto").lower().strip()
    if provider not in ("auto", "hf", "gemini"):
        provider = "auto"

    errors: list[str] = []

    if provider in ("auto", "hf") and _hf_client is not None:
        hf_disabled = _hf_ocr_is_temporarily_disabled() or _ocr_provider_is_temporarily_disabled("hf")
        if hf_disabled:
            errors.append("hf=temporarily disabled after recent OCR failures")
            if provider == "hf":
                raise _friendly_ocr_error(errors)
        else:
            try:
                text = ask_huggingface_ocr(image_data)
                _mark_ocr_provider_success("hf")
                return text, "hf", HF_OCR_MODEL
            except Exception as e:
                errors.append(f"hf={type(e).__name__}: {e!r}")
                _mark_ocr_provider_failure("hf", e)
                if provider == "hf":
                    raise _friendly_ocr_error(errors)
                logger.warning(f"HF OCR failed; trying Gemini fallback: {e}")

    if provider in ("auto", "gemini") and _gemini is not None:
        if _ocr_provider_is_temporarily_disabled("gemini"):
            errors.append("gemini=temporarily disabled after recent OCR failures")
            if provider == "gemini":
                raise _friendly_ocr_error(errors)
        else:
            try:
                text = ask_gemini_ocr(image_data, mime_type)
                _mark_ocr_provider_success("gemini")
                return text, "gemini", GEMINI_MODEL
            except Exception as e:
                errors.append(f"gemini={type(e).__name__}: {e!r}")
                _mark_ocr_provider_failure("gemini", e)
                if provider == "gemini":
                    raise _friendly_ocr_error(errors)
                logger.warning(f"Gemini OCR fallback failed: {e}")

    if not errors:
        errors.append(
            f"No OCR provider configured. OCR_PROVIDER={OCR_PROVIDER!r}, "
            f"hf_available={_hf_client is not None}, gemini_available={_gemini is not None}"
        )
    raise _friendly_ocr_error(errors)


def ai_text_reply(prompt: str, history: list[dict] | None = None) -> tuple[str, str, int | None]:
    """Return (reply, model_used, tokens_used) for text-only chat."""
    provider = (AI_PROVIDER or "hf").lower().strip()

    if provider == "hf":
        reply = ask_huggingface(prompt, history)
        return reply, HF_MODEL, None

    if _gemini is None:
        raise RuntimeError("Gemini client is not configured.")

    contents = _build_gemini_contents(
        message=prompt, history=history or [],
        image_data=None, audio_data=None,
        image_mime="", audio_mime="",
    )
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=_ai_gen_config(),
    )
    tokens_used = None
    with suppress(Exception):
        tokens_used = response.usage_metadata.total_token_count
    return (response.text or "").strip(), GEMINI_MODEL, tokens_used


def _init_clients() -> None:
    global supabase, redis_client, _gemini, TELEGRAM_BOT_TOKEN, SB_URL, SB_KEY, REDIS_URL
    global GEMINI_API_KEY, ADMIN_IDS, GEMINI_MODEL
    global HF_TOKEN, HF_MODEL, HF_OCR_MODEL, AI_PROVIDER, OCR_PROVIDER

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    SB_URL             = os.getenv("SUPABASE_URL", "")
    SB_KEY             = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY", "")
    REDIS_URL          = os.getenv("REDIS_URL", "").strip()
    GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "").strip()
    HF_TOKEN           = os.getenv("HF_TOKEN", "")
    HF_MODEL           = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    HF_OCR_MODEL       = os.getenv("HF_OCR_MODEL", "microsoft/trocr-base-printed")

    AI_PROVIDER = os.getenv("AI_PROVIDER", "hf").lower().strip()
    if AI_PROVIDER not in ("hf", "gemini"):
        logger.warning(f"Unknown AI_PROVIDER={AI_PROVIDER!r}; falling back to hf.")
        AI_PROVIDER = "hf"

    OCR_PROVIDER = os.getenv("OCR_PROVIDER", "auto").lower().strip()
    if OCR_PROVIDER not in ("auto", "hf", "gemini"):
        logger.warning(f"Unknown OCR_PROVIDER={OCR_PROVIDER!r}; falling back to auto.")
        OCR_PROVIDER = "auto"

    ADMIN_IDS.clear()
    for _aid in os.getenv("ADMIN_IDS", "").split(","):
        _aid = _aid.strip()
        if _aid.isdigit():
            ADMIN_IDS.add(int(_aid))

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    if SB_URL and SB_KEY:
        try:
            if SB_KEY.startswith("sb_publishable_") or "publishable" in SB_KEY.lower():
                logger.warning("SUPABASE_KEY looks like a publishable key. Admin tables such as ai_api_keys should use SUPABASE_SERVICE_ROLE_KEY on the server.")
            supabase = create_client(SB_URL, SB_KEY)
            logger.info("Supabase client initialised.")
        except Exception as e:
            logger.error(f"Supabase init failed: {e}")
            supabase = None
    else:
        logger.warning("Supabase env vars missing — DB features disabled.")

    redis_client = None
    if REDIS_URL:
        if redis_lib is None:
            logger.warning("REDIS_URL is set but package `redis` is not installed. Install: pip install redis")
        else:
            try:
                redis_client = redis_lib.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_timeout=REDIS_SOCKET_TIMEOUT_S,
                    socket_connect_timeout=REDIS_SOCKET_TIMEOUT_S,
                    health_check_interval=30,
                    retry_on_timeout=True,
                )
                redis_client.ping()
                logger.info("Redis cache initialised.")
            except Exception as e:
                logger.warning(f"Redis init failed — cache disabled: {e}")
                redis_client = None
    else:
        logger.info("REDIS_URL not set — Redis cache disabled; using memory + Supabase fallback.")

    if GEMINI_API_KEY and GEMINI_MODEL and genai is not None:
        try:
            _gemini = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Gemini legacy fallback initialised (model: {GEMINI_MODEL}).")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            _gemini = None
    else:
        _gemini = None
        logger.info("Gemini disabled. Using Hugging Face for chat/OCR.")

    _init_hf_client()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOICE_MAP = {
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    "en": {"female": "en-US-AriaNeural",     "male": "en-US-GuyNeural"},
}
SPEED_OPTIONS = {
    "spd_0.5": ("x0.5", 0.5),
    "spd_1.0": ("Normal", 1.0),
    "spd_1.5": ("x1.5", 1.5),
    "spd_2.0": ("x2.0", 2.0),
}
WELCOME_TEXT = (
    "🎵 សួស្តី! ខ្ញុំជា Bot បំលែងអក្សរទៅជាសំឡេង អេអាយ\n\n"
    "📌 វាយអក្សរភាសាណាមួយ ផ្ញើរមក Bot នឹងបំលែងដោយស្វ័យប្រវត្តិ!\n\n"
    "🌍 ភាសាដែល Support:\n"
    "🇰🇭 ភាសាខ្មែរ | 🇺🇸 English\n\n"
    "⚙️ ប្រើ /myprefs ដើម្បីមើលការកំណត់របស់អ្នក\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)
BOT_TAG = "@voicekhaibot"

# ---------------------------------------------------------------------------
# Prefs cache — memory -> Redis -> Supabase -> safe defaults
# ---------------------------------------------------------------------------
_PREFS_TTL      = 300.0
_PREFS_MAX_SIZE = 10_000
_prefs_cache: OrderedDict[int, tuple[dict, float]] = OrderedDict()
_prefs_cache_lock: asyncio.Lock | None = None

DEFAULT_USER_PREFS: dict = {"gender": "female", "speed": DEFAULT_SPEED}


def _prefs_redis_key(user_id: int) -> str:
    return _redis_key("prefs", int(user_id))


def _normalize_user_prefs(row: dict | None) -> dict:
    prefs = dict(DEFAULT_USER_PREFS)
    row = row or {}

    gender = row.get("gender") or prefs["gender"]
    if gender not in ("female", "male"):
        gender = "female"
    prefs["gender"] = gender

    raw_speed = row.get("speed", prefs["speed"])
    try:
        prefs["speed"] = max(_SPEED_MIN, min(_SPEED_MAX, float(raw_speed)))
    except Exception:
        prefs["speed"] = DEFAULT_SPEED

    return prefs


def _cache_prefs_sync(user_id: int, prefs: dict) -> None:
    _prefs_cache.pop(user_id, None)
    _prefs_cache[user_id] = (_normalize_user_prefs(prefs), time.monotonic())
    while len(_prefs_cache) > _PREFS_MAX_SIZE:
        _prefs_cache.popitem(last=False)


def _get_cached_prefs_sync(user_id: int) -> dict | None:
    entry = _prefs_cache.get(user_id)
    if entry and time.monotonic() - entry[1] < _PREFS_TTL:
        _prefs_cache.move_to_end(user_id)
        return dict(entry[0])
    if entry:
        _prefs_cache.pop(user_id, None)
    return None


def _invalidate_prefs(user_id: int) -> None:
    _prefs_cache.pop(user_id, None)
    if redis_client is not None:
        _submit_db(lambda: _redis_delete_sync(_prefs_redis_key(user_id)))


async def _async_cache_prefs(user_id: int, prefs: dict, *, write_redis: bool = False) -> None:
    assert _prefs_cache_lock is not None
    prefs = _normalize_user_prefs(prefs)
    async with _prefs_cache_lock:
        _cache_prefs_sync(user_id, prefs)
    if write_redis and redis_client is not None:
        await _redis_set_json(_prefs_redis_key(user_id), prefs, REDIS_PREFS_TTL_S)


async def _async_get_cached_prefs(user_id: int) -> dict | None:
    assert _prefs_cache_lock is not None
    async with _prefs_cache_lock:
        return _get_cached_prefs_sync(user_id)


async def get_user_prefs_async(user_id: int) -> dict:
    """
    Safe prefs flow:
      1) try memory cache
      2) if missing, try Redis
      3) if Redis missing, try Supabase with retry
      4) if Supabase works, refresh Redis
      5) if Supabase fails, return safe defaults
      6) never crash Telegram callbacks
    """
    defaults = dict(DEFAULT_USER_PREFS)

    cached = await _async_get_cached_prefs(user_id)
    if cached is not None:
        return cached

    redis_key = _prefs_redis_key(user_id)
    redis_prefs = await _redis_get_json(redis_key, default=None)
    if isinstance(redis_prefs, dict):
        prefs = _normalize_user_prefs(redis_prefs)
        await _async_cache_prefs(user_id, prefs, write_redis=False)
        return prefs

    if not supabase:
        return defaults

    try:
        res = await db_call(
            f"get_user_prefs:{user_id}",
            lambda: supabase.table("user_prefs")
                .select("gender, speed")
                .eq("user_id", user_id)
                .limit(1)
                .execute(),
            default=None,
            attempts=3,
            timeout=12,
            critical=False,
        )

        rows = getattr(res, "data", None) if res else None
        prefs = _normalize_user_prefs((rows or [None])[0])
        await _async_cache_prefs(user_id, prefs, write_redis=True)
        return prefs

    except Exception as exc:
        _log_once(
            logging.WARNING,
            f"prefs:fallback:{type(exc).__name__}:{str(exc)[:120]}",
            "get_user_prefs_async fallback user=%s: %s",
            user_id,
            exc,
        )
        return defaults


# ---------------------------------------------------------------------------
# Per-user async locks (LRU-capped)
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
def _is_message_not_modified_error(exc: Exception | str) -> bool:
    return "message is not modified" in str(exc).lower()


async def safe_send(coro_factory, retries: int = 3, delay: float = 2.0):
    """Run Telegram send/edit calls with retry.

    Important fix: Telegram raises BadRequest when an inline button edits a
    message to the same text + same markup. That is not a real failure, so this
    helper now returns None instead of logging an unhandled callback error.
    """
    if inspect.isawaitable(coro_factory):
        raise TypeError(
            "safe_send requires a zero-arg callable (lambda), not a raw coroutine."
        )
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
        except BadRequest as e:
            if _is_message_not_modified_error(e):
                logger.debug("Telegram edit skipped: message is not modified.")
                return None
            logger.warning(f"Telegram BadRequest (not retried): {e}")
            raise
        except (TimedOut, NetworkError) as e:
            if _is_message_not_modified_error(e):
                logger.debug("Telegram edit skipped: message is not modified.")
                return None
            last_exc = e
            if attempt < retries - 1:
                logger.warning(f"Network error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)
            else:
                raise
        except TelegramError as e:
            if _is_message_not_modified_error(e):
                logger.debug("Telegram edit skipped: message is not modified.")
                return None
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
        remainder = text[cut:].lstrip()
        if not remainder:
            break
        text = remainder
    return [p for p in pages if p]


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------
_TMP_PREFIX = "tgbot_"


def _get_temp_dir() -> str:
    temp_dir = os.environ.get("BOT_TMP_DIR") or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def _make_temp_file(suffix: str) -> str:
    temp_dir = _get_temp_dir()
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=_TMP_PREFIX, dir=temp_dir)
    os.close(fd)
    return path


def _make_temp_ogg()                          -> str: return _make_temp_file(".ogg")
def _make_temp_audio(suffix: str = ".mp3")    -> str: return _make_temp_file(suffix if suffix.startswith(".") else f".{suffix}")
def _make_temp_img(suffix: str = ".jpg")      -> str: return _make_temp_file(suffix if suffix.startswith(".") else f".{suffix}")


def _cleanup(*paths) -> None:
    """Remove one or more temp files, silencing errors."""
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError as e:
            logger.warning(f"Temp cleanup failed for {p}: {e}")


_STALE_TEMP_AGE_S = 7200


def _sweep_stale_temps() -> None:
    temp_dir = _get_temp_dir()
    extensions = [
        ".ogg", ".jpg", ".jpeg", ".png", ".webp",
        ".mp3", ".wav", ".mp4", ".m4a", ".flac", ".aac", ".opus", ".webm",
    ]
    now = time.time()
    for ext in extensions:
        for file_path in glob.glob(os.path.join(temp_dir, f"{_TMP_PREFIX}*{ext}")):
            try:
                if now - os.path.getmtime(file_path) > _STALE_TEMP_AGE_S:
                    os.remove(file_path)
                    logger.info(f"Swept stale temp file: {file_path}")
            except OSError:
                pass


async def _periodic_temp_sweep(stop_event: asyncio.Event) -> None:
    while True:
        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=3600.0)
        except asyncio.TimeoutError:
            pass
        if not stop_event.is_set():
            _sweep_stale_temps()


# ---------------------------------------------------------------------------
# Database helpers — user prefs
# ---------------------------------------------------------------------------
_USER_SYNC_TTL = 300.0
_USER_SYNC_MAX = 20_000
_user_sync_seen: OrderedDict[int, float] = OrderedDict()


def sync_user_data(user) -> None:
    if not supabase or not user:
        return
    now  = time.monotonic()
    last = _user_sync_seen.get(user.id, 0.0)
    if now - last < _USER_SYNC_TTL:
        return
    _user_sync_seen.pop(user.id, None)
    _user_sync_seen[user.id] = now
    while len(_user_sync_seen) > _USER_SYNC_MAX:
        _user_sync_seen.popitem(last=False)

    def _run():
        payload = {"user_id": user.id, "username": user.username or user.first_name}
        primary_payload = {**payload, "last_active": datetime.now(timezone.utc).isoformat()}

        res = db_call_sync(
            f"sync_user_data:{user.id}:with_last_active",
            lambda: supabase.table("user_prefs").upsert(
                primary_payload,
                on_conflict="user_id",
            ).execute(),
            default=None,
            attempts=3,
            critical=False,
        )

        if res is not None:
            return

        # Fallback for older user_prefs tables without last_active.
        db_call_sync(
            f"sync_user_data:{user.id}:basic",
            lambda: supabase.table("user_prefs").upsert(payload, on_conflict="user_id").execute(),
            default=None,
            attempts=2,
            critical=False,
        )

    _submit_db(_run)


def _paginated_fetch(select_fields: str) -> list[dict]:
    if not supabase:
        return []

    all_rows, page, page_size = [], 0, 1000
    while True:
        res = db_call_sync(
            f"paginated_fetch:{select_fields}:page{page}",
            lambda p=page: supabase.table("user_prefs")
                .select(select_fields)
                .range(p * page_size, (p + 1) * page_size - 1)
                .execute(),
            default=None,
            attempts=3,
            critical=False,
        )
        if res is None:
            return all_rows

        batch = getattr(res, "data", None) or []
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return all_rows


def get_all_user_ids()         -> list[int]:  return [row["user_id"] for row in _paginated_fetch("user_id")]
def get_all_users_with_names() -> list[dict]: return _paginated_fetch("user_id, username")


def _normalize_user_search_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lstrip("@")).lower()


def search_users_by_query(query: str, limit: int = 80) -> list[dict]:
    """Search users by Telegram user ID or username.

    This intentionally fetches the same safe fields as the user list, then
    filters in Python. It avoids fragile Supabase `.or_()` syntax and keeps
    working even when the table only has `user_id` and `username`.
    """
    q = _normalize_user_search_query(query)
    if not q:
        return []

    users = get_all_users_with_names()
    if not users:
        return []

    exact: list[dict] = []
    prefix: list[dict] = []
    contains: list[dict] = []

    for row in users:
        uid = str(row.get("user_id") or "").strip()
        username = str(row.get("username") or "").strip()
        username_l = username.lower().lstrip("@")

        if q == uid or q == username_l:
            exact.append(row)
        elif uid.startswith(q) or username_l.startswith(q):
            prefix.append(row)
        elif q in uid or q in username_l:
            contains.append(row)

    seen: set[int] = set()
    merged: list[dict] = []
    for row in exact + prefix + contains:
        try:
            uid_int = int(row.get("user_id") or 0)
        except Exception:
            uid_int = 0
        if not uid_int or uid_int in seen:
            continue
        seen.add(uid_int)
        merged.append(row)
        if len(merged) >= limit:
            break
    return merged


def user_exists_in_db(user_id: int) -> bool:
    if not supabase:
        return False
    res = db_call_sync(
        f"user_exists:{user_id}",
        lambda: supabase.table("user_prefs")
            .select("user_id")
            .eq("user_id", user_id)
            .limit(1)
            .execute(),
        default=None,
        attempts=3,
        critical=False,
    )
    return bool(getattr(res, "data", None)) if res else False


def update_user_gender(user_id: int, gender: str) -> None:
    gender = gender if gender in ("female", "male") else "female"
    _invalidate_prefs(user_id)
    if not supabase:
        return

    def _run():
        db_call_sync(
            f"update_user_gender:{user_id}",
            lambda: supabase.table("user_prefs").update({"gender": gender}).eq("user_id", user_id).execute(),
            default=None,
            attempts=3,
            critical=False,
        )

    _submit_db(_run)


def update_user_speed(user_id: int, speed: float) -> None:
    speed = round(max(_SPEED_MIN, min(_SPEED_MAX, speed)), 4)
    _invalidate_prefs(user_id)
    if not supabase:
        return

    def _run():
        res = db_call_sync(
            f"update_user_speed:{user_id}",
            lambda: supabase.table("user_prefs").update({"speed": speed}).eq("user_id", user_id).execute(),
            default=None,
            attempts=3,
            critical=False,
        )
        if res is None:
            _log_once(
                logging.ERROR,
                "speed_column_or_update_error",
                "update_user_speed failed. If the column is missing, run: ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;",
            )

    _submit_db(_run)


_rls_warned = False

# Fast cache for callback buttons: memory -> Redis -> Supabase.
_TEXT_CACHE_MEMORY_MAX = 20_000
_text_cache_memory: OrderedDict[tuple[int, int], tuple[str, float]] = OrderedDict()


def _text_cache_redis_key(msg_id: int, chat_id: int) -> str:
    return _redis_key("text_cache", int(chat_id or 0), int(msg_id))


def _remember_text_cache_sync(msg_id: int, chat_id: int, text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    key = (int(chat_id or 0), int(msg_id))
    _text_cache_memory.pop(key, None)
    _text_cache_memory[key] = (text, time.monotonic())
    while len(_text_cache_memory) > _TEXT_CACHE_MEMORY_MAX:
        _text_cache_memory.popitem(last=False)


def _get_text_cache_memory_sync(msg_id: int, chat_id: int) -> str | None:
    key = (int(chat_id or 0), int(msg_id))
    item = _text_cache_memory.get(key)
    if not item:
        return None
    _text_cache_memory.move_to_end(key)
    return item[0]


def _write_text_cache_redis_sync(msg_id: int, chat_id: int, text: str) -> None:
    if redis_client is not None and text:
        _redis_set_json_sync(_text_cache_redis_key(msg_id, chat_id), {"text": text}, REDIS_TEXT_CACHE_TTL_S)


def _text_cache_user_history_redis_key(user_id: int) -> str:
    return _redis_key("text_history", int(user_id))


def _text_cache_user_history_append_sync(
    user_id: int | None,
    username: str | None,
    msg_id: int,
    chat_id: int,
    text: str,
    created_at: str | None = None,
) -> None:
    """Keep a Redis per-user view of text_cache for fast admin history.

    This is only a cache. Supabase text_cache remains the source of truth.
    """
    if redis_client is None or user_id is None:
        return
    try:
        uid = int(user_id)
    except Exception:
        return
    text = (text or "").strip()
    if not text:
        return
    key = _text_cache_user_history_redis_key(uid)
    rows = _redis_get_json_sync(key, default=[])
    if not isinstance(rows, list):
        rows = []
    rows.append({
        "user_id": uid,
        "username": (username or "").strip(),
        "message_id": int(msg_id or 0),
        "chat_id": int(chat_id or 0),
        "original_text": text,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    })
    keep = max(50, int(os.environ.get("ADMIN_TEXT_CACHE_REDIS_TURNS", "120")))
    _redis_set_json_sync(key, rows[-keep:], REDIS_HISTORY_TTL_S)


def save_text_cache(
    msg_id: int,
    text: str,
    chat_id: int = 0,
    user_id: int = None,
    username: str = None,
) -> None:
    """
    Save callback text safely:
      - memory cache immediately
      - Redis cache in background if configured
      - Supabase in background with retry
      - never raises into Telegram callback flow
    """
    text = (text or "").strip()
    if not text:
        return

    _remember_text_cache_sync(msg_id, chat_id, text)

    def _run():
        global _rls_warned

        _write_text_cache_redis_sync(msg_id, chat_id, text)
        _text_cache_user_history_append_sync(
            user_id=user_id,
            username=username,
            msg_id=msg_id,
            chat_id=chat_id,
            text=text,
        )

        if not supabase:
            return

        payload = {"message_id": msg_id, "chat_id": chat_id, "original_text": text}
        if user_id is not None:
            payload["user_id"] = user_id
        if username is not None:
            payload["username"] = username

        res = db_call_sync(
            f"save_text_cache:{chat_id}:{msg_id}",
            lambda: supabase.table("text_cache").upsert(
                payload,
                on_conflict="chat_id,message_id",
            ).execute(),
            default=None,
            attempts=3,
            critical=False,
        )

        if res is None:
            err_msg = "text_cache Supabase write failed or returned no result."
            if not _rls_warned:
                _rls_warned = True
                logger.warning(
                    "%s If RLS blocks inserts, fix with:\\n"
                    "  ALTER TABLE text_cache DISABLE ROW LEVEL SECURITY;\\n"
                    "  -- or --\\n"
                    "  CREATE POLICY \"service_role_all\" ON text_cache "
                    "FOR ALL TO service_role USING (true) WITH CHECK (true);",
                    err_msg,
                )

    _submit_db(_run)


def get_text_cache(msg_id: int, chat_id: int = 0) -> str | None:
    """
    Read callback text safely:
      1) memory cache
      2) Redis cache
      3) Supabase with retry
      4) return None if unavailable
    """
    cached = _get_text_cache_memory_sync(msg_id, chat_id)
    if cached:
        return cached

    redis_payload = _redis_get_json_sync(_text_cache_redis_key(msg_id, chat_id), default=None)
    if isinstance(redis_payload, dict):
        redis_text = (redis_payload.get("text") or "").strip()
        if redis_text:
            _remember_text_cache_sync(msg_id, chat_id, redis_text)
            return redis_text

    if not supabase:
        return None

    res = db_call_sync(
        f"get_text_cache:{chat_id}:{msg_id}",
        lambda: supabase.table("text_cache")
            .select("original_text")
            .eq("message_id", msg_id)
            .eq("chat_id", chat_id)
            .limit(1)
            .execute(),
        default=None,
        attempts=3,
        critical=False,
    )

    rows = getattr(res, "data", None) if res else None
    if rows:
        text_value = (rows[0].get("original_text") or "").strip()
        if text_value:
            _remember_text_cache_sync(msg_id, chat_id, text_value)
            _write_text_cache_redis_sync(msg_id, chat_id, text_value)
            return text_value

    return None


def ensure_speed_column() -> None:
    if not supabase:
        logger.info("ensure_speed_column: Supabase not configured, skipping.")
        return

    res = db_call_sync(
        "ensure_speed_column",
        lambda: supabase.table("user_prefs").select("speed").limit(1).execute(),
        default=None,
        attempts=2,
        critical=False,
    )
    if res is not None:
        logger.info("speed column present.")
    else:
        logger.warning(
            "Could not verify user_prefs.speed. If missing, run:\n"
            "  ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
        )



# ---------------------------------------------------------------------------
# Admin Dashboard V2 — settings, blocks, metrics, user tools
# ---------------------------------------------------------------------------
BOT_SETTING_DEFAULTS: dict[str, str] = {
    "maintenance_mode": "0",
    "tts_enabled": "1",
    "ocr_enabled": "1",
    "voice_transcribe_enabled": "1",
    "audio_transcribe_enabled": "1",
    "ai_resolver_enabled": "1",
}
BOT_SETTING_LABELS: dict[str, str] = {
    "maintenance_mode": "🛠️ Maintenance Mode",
    "tts_enabled": "🗣️ Text → Voice",
    "ocr_enabled": "🔍 Image OCR",
    "voice_transcribe_enabled": "🎙️ Voice Transcribe",
    "audio_transcribe_enabled": "🎵 Audio File Transcribe",
    "ai_resolver_enabled": "🧠 AI Text Resolver",
}
BOT_SETTING_DESCRIPTIONS: dict[str, str] = {
    "maintenance_mode": "When ON, normal users cannot use the bot.",
    "tts_enabled": "Allow normal text messages to generate voice.",
    "ocr_enabled": "Allow photo OCR reading.",
    "voice_transcribe_enabled": "Allow Telegram voice transcription.",
    "audio_transcribe_enabled": "Allow uploaded audio-file transcription.",
    "ai_resolver_enabled": "Allow AI to rewrite/resolve text before TTS when enabled by env.",
}
_SETTINGS_CACHE_TTL_S = float(os.environ.get("BOT_SETTINGS_CACHE_TTL_S", "30"))
_bot_settings_memory: dict[str, str] = dict(BOT_SETTING_DEFAULTS)
_bot_settings_cache: dict = {
    "data": dict(BOT_SETTING_DEFAULTS),
    "status": {"db_ok": False, "error": "not loaded", "memory": True},
    "ts": 0.0,
}

_blocked_users_memory: set[int] = set()
_blocked_user_cache: OrderedDict[int, tuple[bool, float]] = OrderedDict()
_BLOCKED_USER_CACHE_TTL_S = 60.0
_BLOCKED_USER_CACHE_MAX = 20_000

_RUNTIME_METRICS: OrderedDict[str, int] = OrderedDict([
    ("tts", 0),
    ("ocr", 0),
    ("voice", 0),
    ("audio", 0),
    ("blocked_hits", 0),
    ("disabled_hits", 0),
    ("errors", 0),
])


def _metric_inc(name: str, amount: int = 1) -> None:
    _RUNTIME_METRICS[name] = int(_RUNTIME_METRICS.get(name, 0)) + int(amount)


def _format_uptime() -> str:
    if not _BOT_START_TIME:
        return "starting"
    seconds = max(0, int(time.time() - _BOT_START_TIME))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _bool_to_setting_value(enabled: bool) -> str:
    return "1" if bool(enabled) else "0"


def _setting_bool_from(settings: dict | None, key: str, default: bool = True) -> bool:
    if not settings:
        settings = BOT_SETTING_DEFAULTS
    raw = str(settings.get(key, BOT_SETTING_DEFAULTS.get(key, _bool_to_setting_value(default)))).strip().lower()
    return raw in ("1", "true", "yes", "on", "enabled")


def bot_setting_bool_cached(key: str, default: bool = True) -> bool:
    return _setting_bool_from(_bot_settings_cache.get("data") or BOT_SETTING_DEFAULTS, key, default)


def db_bot_settings_fetch_all() -> tuple[dict[str, str], dict]:
    data = dict(BOT_SETTING_DEFAULTS)
    data.update(_bot_settings_memory)
    status = {"db_ok": False, "error": "", "memory": not bool(supabase)}
    if not supabase:
        status["error"] = "Supabase not configured; settings are memory-only."
        return data, status
    try:
        res = supabase.table("bot_settings").select("key,value").execute()
        for row in res.data or []:
            key = str(row.get("key") or "").strip()
            if key in BOT_SETTING_DEFAULTS:
                data[key] = str(row.get("value") or BOT_SETTING_DEFAULTS[key])
        status.update({"db_ok": True, "error": "", "memory": False})
    except Exception as e:
        status.update({"db_ok": False, "error": str(e), "memory": True})
    return data, status


async def get_bot_settings_async(force: bool = False) -> tuple[dict[str, str], dict]:
    now = time.monotonic()
    if not force and now - float(_bot_settings_cache.get("ts") or 0.0) < _SETTINGS_CACHE_TTL_S:
        return dict(_bot_settings_cache["data"]), dict(_bot_settings_cache["status"])
    data, status = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, db_bot_settings_fetch_all)
    _bot_settings_cache["data"] = dict(data)
    _bot_settings_cache["status"] = dict(status)
    _bot_settings_cache["ts"] = now
    return data, status


def db_bot_setting_set(key: str, enabled: bool, admin_id: int) -> tuple[bool, str]:
    if key not in BOT_SETTING_DEFAULTS:
        return False, f"Unknown setting: {key}"
    value = _bool_to_setting_value(enabled)
    _bot_settings_memory[key] = value
    if not supabase:
        _bot_settings_cache["ts"] = 0.0
        return True, "saved in memory only"
    try:
        supabase.table("bot_settings").upsert({
            "key": key,
            "value": value,
            "updated_by": int(admin_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="key").execute()
        _bot_settings_cache["ts"] = 0.0
        return True, "saved"
    except Exception as e:
        _bot_settings_cache["ts"] = 0.0
        return False, str(e)


def _blocked_cache_set(user_id: int, blocked: bool) -> None:
    _blocked_user_cache.pop(int(user_id), None)
    _blocked_user_cache[int(user_id)] = (bool(blocked), time.monotonic())
    while len(_blocked_user_cache) > _BLOCKED_USER_CACHE_MAX:
        _blocked_user_cache.popitem(last=False)


def _blocked_cache_get(user_id: int) -> bool | None:
    item = _blocked_user_cache.get(int(user_id))
    if not item:
        return None
    blocked, ts = item
    if time.monotonic() - ts > _BLOCKED_USER_CACHE_TTL_S:
        _blocked_user_cache.pop(int(user_id), None)
        return None
    _blocked_user_cache.move_to_end(int(user_id))
    return blocked


def db_user_is_blocked(user_id: int) -> bool:
    user_id = int(user_id)
    cached = _blocked_cache_get(user_id)
    if cached is not None:
        return cached
    if user_id in _blocked_users_memory:
        _blocked_cache_set(user_id, True)
        return True
    if not supabase:
        _blocked_cache_set(user_id, False)
        return False
    try:
        res = supabase.table("blocked_users").select("user_id").eq("user_id", user_id).limit(1).execute()
        blocked = bool(res.data)
        _blocked_cache_set(user_id, blocked)
        return blocked
    except Exception as e:
        logger.warning(f"blocked_users check skipped user={user_id}: {e}")
        _blocked_cache_set(user_id, False)
        return False


def db_blocked_user_count() -> int:
    if not supabase:
        return len(_blocked_users_memory)
    try:
        res = supabase.table("blocked_users").select("user_id").execute()
        return len(res.data or [])
    except Exception:
        return len(_blocked_users_memory)


def db_user_set_blocked(user_id: int, admin_id: int, blocked: bool, reason: str = "") -> tuple[bool, str]:
    user_id = int(user_id)
    admin_id = int(admin_id)
    if blocked:
        _blocked_users_memory.add(user_id)
    else:
        _blocked_users_memory.discard(user_id)
    _blocked_cache_set(user_id, blocked)
    if not supabase:
        return True, "memory-only"
    try:
        if blocked:
            supabase.table("blocked_users").upsert({
                "user_id": user_id,
                "admin_id": admin_id,
                "reason": reason or "blocked from admin panel",
                "blocked_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="user_id").execute()
        else:
            supabase.table("blocked_users").delete().eq("user_id", user_id).execute()
        return True, "saved"
    except Exception as e:
        return False, str(e)


def db_user_reset_prefs(user_id: int) -> tuple[bool, str]:
    user_id = int(user_id)
    _invalidate_prefs(user_id)
    if not supabase:
        return True, "memory-only"
    try:
        supabase.table("user_prefs").upsert({
            "user_id": user_id,
            "gender": "female",
            "speed": DEFAULT_SPEED,
        }, on_conflict="user_id").execute()
        return True, "reset"
    except Exception as e:
        return False, str(e)


def db_user_detail(user_id: int) -> dict:
    user_id = int(user_id)
    row: dict = {"user_id": user_id}
    if supabase:
        res = db_call_sync(
            f"user_detail:{user_id}",
            lambda: supabase.table("user_prefs").select("*").eq("user_id", user_id).limit(1).execute(),
            default=None,
            attempts=3,
            critical=False,
        )
        if res is not None and getattr(res, "data", None):
            row.update(res.data[0])
        elif res is None:
            row["error"] = "User detail database read temporarily unavailable."
    row["blocked"] = db_user_is_blocked(user_id)
    # Admin Recent History/User Detail must read from text_cache, not conversation_history.
    row["history"] = db_user_history_fetch(user_id, limit=ADMIN_DETAIL_HISTORY_TURNS)
    return row


def _format_user_detail_text(row: dict) -> str:
    user_id = int(row.get("user_id") or 0)
    username = html.escape(str(row.get("username") or row.get("first_name") or "-"))
    gender = html.escape(str(row.get("gender") or "female"))
    speed = html.escape(str(row.get("speed") or DEFAULT_SPEED))
    last_active = html.escape(str(row.get("last_active") or "-")[:19].replace("T", " "))
    blocked = "🚫 BLOCKED" if row.get("blocked") else "✅ ACTIVE"
    history_rows = row.get("history") or []
    last_lines = []
    for h in history_rows[-ADMIN_DETAIL_HISTORY_TURNS:]:
        role = "👤" if _normalize_role(h.get("role")) == "user" else "🤖"
        content = html.escape(_history_compact_text(h.get("content"), 180))
        if content:
            last_lines.append(f"{role} {content}")
    history_text = "\n".join(last_lines) if last_lines else "-"
    extra_error = f"\n\n⚠️ <pre>{html.escape(str(row.get('error'))[:500])}</pre>" if row.get("error") else ""
    return (
        "👤 <b>User Detail</b>\n\n"
        f"ID: <code>{user_id}</code>\n"
        f"Username: <b>{username}</b>\n"
        f"Status: <b>{blocked}</b>\n"
        f"Voice: <b>{gender}</b>\n"
        f"Speed: <b>{speed}x</b>\n"
        f"Last active: <b>{last_active}</b>\n\n"
        "<b>Recent history</b>\n"
        f"{history_text}"
        f"{extra_error}"
    )


async def _ensure_user_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE, feature_key: str | None = None, feature_name: str = "feature") -> bool:
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return False
    if _is_admin(user.id):
        return True

    blocked = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_is_blocked(user.id))
    if blocked:
        _metric_inc("blocked_hits")
        await safe_send(lambda: msg.reply_text("⛔ អ្នកត្រូវបាន Block មិនអាចប្រើ Bot នេះបានទេ។"))
        return False

    settings, _status = await get_bot_settings_async()
    if _setting_bool_from(settings, "maintenance_mode", False):
        _metric_inc("disabled_hits")
        await safe_send(lambda: msg.reply_text("🛠️ Bot កំពុង Maintenance។ សូមព្យាយាមម្តងទៀតពេលក្រោយ។"))
        return False

    if feature_key and not _setting_bool_from(settings, feature_key, True):
        _metric_inc("disabled_hits")
        await safe_send(lambda: msg.reply_text(f"⚠️ {feature_name} ត្រូវបានបិទបណ្តោះអាសន្នដោយ Admin។"))
        return False

    return True


# ---------------------------------------------------------------------------
# Database helpers — conversation history
# ---------------------------------------------------------------------------
CONV_HISTORY_LIMIT    = 10
CONV_CONTEXT_MAX_CHARS = 3000
CONV_RESOLVE_TIMEOUT_S = 15

# Admin history display/cache limits.
# The old admin detail screen was hard-coded to show only 3 turns.
# These env values let you increase/decrease history without editing code again.
ADMIN_DETAIL_HISTORY_TURNS = max(3, min(20, int(os.environ.get("ADMIN_DETAIL_HISTORY_TURNS", "10"))))
ADMIN_FULL_HISTORY_TURNS   = max(10, min(100, int(os.environ.get("ADMIN_FULL_HISTORY_TURNS", "50"))))
ADMIN_HISTORY_PAGE_SIZE    = max(5, min(20, int(os.environ.get("ADMIN_HISTORY_PAGE_SIZE", "10"))))

_HIST_CACHE_MAX_USERS = 5_000
_HIST_CACHE_TURNS     = max(ADMIN_FULL_HISTORY_TURNS, 50)
_hist_cache: OrderedDict[int, deque] = OrderedDict()


def _hist_redis_key(user_id: int) -> str:
    return _redis_key("history", int(user_id))


def _normalize_role(role: str) -> str:
    role = (role or "").strip().lower()
    return "assistant" if role in ("assistant", "bot", "model") else "user"


def _hist_cache_append(user_id: int, role: str, content: str) -> None:
    role    = _normalize_role(role)
    content = (content or "").strip()
    if not content:
        return
    if user_id not in _hist_cache:
        while len(_hist_cache) >= _HIST_CACHE_MAX_USERS:
            _hist_cache.popitem(last=False)
        _hist_cache[user_id] = deque(maxlen=_HIST_CACHE_TURNS)
    _hist_cache.move_to_end(user_id)
    _hist_cache[user_id].append({
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def _hist_cache_get(user_id: int) -> list[dict] | None:
    d = _hist_cache.get(user_id)
    if d is None:
        return None
    _hist_cache.move_to_end(user_id)
    return list(d)


def _hist_cache_clear(user_id: int) -> None:
    _hist_cache.pop(user_id, None)


def _hist_rows_normalized(rows: list[dict] | None) -> list[dict]:
    clean: list[dict] = []
    for row in rows or []:
        content = (row.get("content") or "").strip()
        if content:
            item = {"role": _normalize_role(row.get("role")), "content": content}
            created_at = row.get("created_at") or row.get("created") or row.get("ts")
            if created_at:
                item["created_at"] = str(created_at)
            clean.append(item)
    return clean[-_HIST_CACHE_TURNS:]


def _hist_redis_save_sync(user_id: int, rows: list[dict]) -> None:
    if redis_client is not None:
        _redis_set_json_sync(_hist_redis_key(user_id), _hist_rows_normalized(rows), REDIS_HISTORY_TTL_S)


def db_history_append(user_id: int, role: str, content: str) -> None:
    role    = _normalize_role(role)
    content = (content or "").strip()
    if not content:
        return

    def _run():
        # Keep Redis warm even if Supabase is temporarily unavailable.
        if redis_client is not None:
            rows = _redis_get_json_sync(_hist_redis_key(user_id), default=[])
            if not isinstance(rows, list):
                rows = []
            rows = _hist_rows_normalized(rows + [{
                "role": role,
                "content": content,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }])
            _hist_redis_save_sync(user_id, rows)

        if supabase:
            db_call_sync(
                f"history_append:{user_id}",
                lambda: supabase.table("conversation_history").insert({
                    "user_id": user_id,
                    "role": role,
                    "content": content,
                }).execute(),
                default=None,
                attempts=3,
                critical=False,
            )

    _submit_db(_run)


def db_history_fetch(user_id: int, limit: int = CONV_HISTORY_LIMIT) -> list[dict]:
    redis_rows = _redis_get_json_sync(_hist_redis_key(user_id), default=None)
    if isinstance(redis_rows, list) and redis_rows:
        return _hist_rows_normalized(redis_rows)[-limit:]

    if not supabase:
        return []

    res = db_call_sync(
        f"history_fetch:{user_id}",
        lambda: supabase.table("conversation_history")
            .select("role, content, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute(),
        default=None,
        attempts=3,
        critical=False,
    )

    rows = list(reversed(getattr(res, "data", None) or [])) if res else []
    clean = _hist_rows_normalized(rows)[-limit:]
    if clean:
        _hist_redis_save_sync(user_id, clean)
    return clean


def db_history_clear(user_id: int) -> None:
    """Clear user history from both admin text_cache history and AI conversation history.

    Admin Recent History reads from text_cache, so Clear History must delete
    text_cache rows too. conversation_history is also cleared for the AI context.
    """
    user_id = int(user_id)
    _hist_cache_clear(user_id)

    def _run():
        if redis_client is not None:
            _redis_delete_sync(_hist_redis_key(user_id))
            _redis_delete_sync(_text_cache_user_history_redis_key(user_id))
        if supabase:
            db_call_sync(
                f"text_cache_history_clear:{user_id}",
                lambda: supabase.table("text_cache").delete().eq("user_id", user_id).execute(),
                default=None,
                attempts=3,
                critical=False,
            )
            db_call_sync(
                f"conversation_history_clear:{user_id}",
                lambda: supabase.table("conversation_history").delete().eq("user_id", user_id).execute(),
                default=None,
                attempts=3,
                critical=False,
            )

    _submit_db(_run)


def _admin_history_rows_normalized(rows: list[dict] | None, limit: int | None = None) -> list[dict]:
    """Normalize admin history rows from text_cache.

    Admin Recent History intentionally uses text_cache as the source of truth,
    because text_cache stores the actual text connected to Telegram callback
    buttons/audio/OCR/transcript messages. conversation_history is still used
    by the AI context system, but not by this admin history panel.
    """
    clean: list[dict] = []
    for row in rows or []:
        content = str(
            row.get("original_text")
            or row.get("content")
            or row.get("text")
            or ""
        ).strip()
        if not content:
            continue

        raw_uid = row.get("user_id")
        try:
            uid = int(raw_uid or 0)
        except Exception:
            uid = 0
        if not uid:
            continue

        item = {
            "user_id": uid,
            "username": str(row.get("username") or "").strip(),
            "message_id": row.get("message_id"),
            "chat_id": row.get("chat_id"),
            "role": "user",
            "content": content,
            "original_text": content,
            "created_at": str(row.get("created_at") or ""),
            "source": "text_cache",
        }
        clean.append(item)
    if limit is not None:
        return clean[-max(1, int(limit)):]
    return clean


def db_user_history_fetch(user_id: int, limit: int = 30) -> list[dict]:
    """Admin full history view from text_cache: Redis first, then Supabase.

    This function no longer reads conversation_history for the admin Recent
    History/User Detail screens. The user asked for Recent History to fetch
    data from text_cache.
    """
    user_id = int(user_id)
    limit = max(1, min(120, int(limit or 30)))

    redis_rows = _redis_get_json_sync(_text_cache_user_history_redis_key(user_id), default=None)
    if isinstance(redis_rows, list) and redis_rows:
        clean = _admin_history_rows_normalized(redis_rows, limit=limit)
        if clean:
            return clean[-limit:]

    if not supabase:
        return []

    res = db_call_sync(
        f"admin_text_cache_history_fetch:{user_id}",
        lambda: supabase.table("text_cache")
            .select("user_id, username, message_id, chat_id, original_text, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute(),
        default=None,
        attempts=3,
        critical=False,
    )
    rows = list(reversed(getattr(res, "data", None) or [])) if res else []
    clean = _admin_history_rows_normalized(rows, limit=limit)
    if clean and redis_client is not None:
        _redis_set_json_sync(_text_cache_user_history_redis_key(user_id), clean[-max(limit, 50):], REDIS_HISTORY_TTL_S)
    return clean


def db_recent_history_users(limit_users: int = 80, scan_limit: int = 500) -> list[dict]:
    """Return users ordered by latest text_cache activity.

    Source of truth: text_cache, not conversation_history.
    It scans recent text_cache rows and groups by user_id in Python so this
    works with normal Supabase REST queries and does not require SQL functions.
    """
    limit_users = max(1, min(200, int(limit_users or 80)))
    scan_limit = max(limit_users, min(1000, int(scan_limit or 500)))

    name_map: dict[int, str] = {}
    try:
        for u in get_all_users_with_names():
            uid = int(u.get("user_id") or 0)
            if uid:
                name_map[uid] = str(u.get("username") or "").strip()
    except Exception as exc:
        _log_once(logging.WARNING, "recent_text_cache_names_failed", f"recent text_cache username lookup failed: {exc}")

    rows: list[dict] = []
    if supabase:
        res = db_call_sync(
            "admin_recent_text_cache_scan",
            lambda: supabase.table("text_cache")
                .select("user_id, username, message_id, chat_id, original_text, created_at")
                .order("created_at", desc=True)
                .limit(scan_limit)
                .execute(),
            default=None,
            attempts=3,
            critical=False,
        )
        rows = getattr(res, "data", None) or [] if res else []

    # Fallback only to Redis per-user text_cache history if Supabase is down.
    # This is still text_cache-derived data, not conversation_history.
    if not rows and redis_client is not None:
        for key in list(_redis_scan_keys_sync(_redis_key("text_history", "*"), limit=limit_users * 2)):
            cached_rows = _redis_get_json_sync(key, default=[])
            if isinstance(cached_rows, list) and cached_rows:
                rows.append(cached_rows[-1])

    grouped: OrderedDict[int, dict] = OrderedDict()
    for row in rows:
        norm_rows = _admin_history_rows_normalized([row], limit=1)
        if not norm_rows:
            continue
        item = norm_rows[0]
        uid = int(item.get("user_id") or 0)
        if not uid:
            continue
        if uid not in grouped:
            grouped[uid] = {
                "user_id": uid,
                "username": item.get("username") or name_map.get(uid, ""),
                "role": "user",
                "content": item.get("content") or "",
                "original_text": item.get("content") or "",
                "created_at": str(item.get("created_at") or ""),
                "turns": 0,
                "source": "text_cache",
            }
        grouped[uid]["turns"] = int(grouped[uid].get("turns") or 0) + 1
        if len(grouped) >= limit_users and len(rows) >= scan_limit:
            break

    return list(grouped.values())[:limit_users]


def _format_recent_history_panel_text(rows: list[dict], page: int, page_size: int = 7) -> str:
    page = _clamp_users_page(rows, page, page_size)
    total_pages = max(1, (len(rows) + page_size - 1) // page_size)
    chunk = rows[page * page_size : page * page_size + page_size]

    if not rows:
        return (
            "🕘 <b>Recent User History</b>\n\n"
            "No recent text_cache history found.\n\n"
            "Possible reasons:\n"
            "• <code>text_cache</code> table has no rows with <code>user_id</code> yet\n"
            "• Supabase is temporarily unavailable\n"
            "• Redis cache was just restarted"
        )

    lines = [
        "🕘 <b>Recent User History</b>",
        "",
        f"Users with text_cache history: <b>{len(rows)}</b>",
        f"Page: <b>{page + 1}/{total_pages}</b>",
        "",
    ]
    for i, item in enumerate(chunk, start=page * page_size + 1):
        uid = int(item.get("user_id") or 0)
        username = html.escape(str(item.get("username") or "-")[:40])
        role = "👤" if _normalize_role(item.get("role")) == "user" else "🤖"
        preview = html.escape(_history_compact_text(item.get("content"), 120))
        created = html.escape(str(item.get("created_at") or "")[:19].replace("T", " ")) or "-"
        turns = int(item.get("turns") or 1)
        lines.append(
            f"{i}. <code>{uid}</code> · <b>{username}</b> · turns: <b>{turns}</b>\n"
            f"   {role} {preview}\n"
            f"   🕒 {created}"
        )
    lines.append("\nSource: <code>text_cache</code>. Tap a user below to view full cached text history.")
    return "\n".join(lines)[:3900]


def _format_user_full_history_text(user_id: int, rows: list[dict], page: int = 0, page_size: int = ADMIN_HISTORY_PAGE_SIZE) -> str:
    """Format one page of user history for Telegram.

    Telegram has a message length limit, so this shows history in pages instead
    of silently truncating to only the latest few turns.
    """
    user_id = int(user_id)
    rows = rows or []
    page_size = max(5, min(20, int(page_size or ADMIN_HISTORY_PAGE_SIZE)))
    total = len(rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(int(page or 0), total_pages - 1))

    if not rows:
        return (
            "📜 <b>User Recent Text Cache History</b>\n\n"
            f"User ID: <code>{user_id}</code>\n\n"
            "No text_cache history found for this user."
        )

    start_i = page * page_size
    end_i = min(total, start_i + page_size)
    chunk = rows[start_i:end_i]

    lines = [
        "📜 <b>User Recent Text Cache History</b>",
        "",
        f"User ID: <code>{user_id}</code>",
        f"Showing <b>{start_i + 1}-{end_i}</b> of <b>{total}</b> text_cache item(s)",
        f"Page: <b>{page + 1}/{total_pages}</b>",
        "",
    ]
    max_total = 3850
    for i, row in enumerate(chunk, start=start_i + 1):
        role = "👤 Text Cache"
        created = str(row.get("created_at") or "")[:19].replace("T", " ")
        content = html.escape(_history_compact_text(row.get("content"), 520))
        time_part = f" · 🕒 {html.escape(created)}" if created else ""
        block = f"{i}. <b>{role}</b>{time_part}\n{content}\n"
        if sum(len(x) + 1 for x in lines) + len(block) > max_total:
            lines.append("\n…this page is long, so Telegram message limit was reached. Use next page for more.")
            break
        lines.append(block)
    return "\n".join(lines)


def _build_context_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for row in history:
        role    = _normalize_role(row.get("role"))
        content = (row.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{'User' if role == 'user' else 'Bot'}: {content}")
    block = "\n".join(lines)
    if len(block) > CONV_CONTEXT_MAX_CHARS:
        block = block[-CONV_CONTEXT_MAX_CHARS:]
        nl    = block.find("\n")
        if nl != -1:
            block = block[nl + 1:]
    return block


def record_turn(user_id: int, role: str, content: str) -> None:
    role    = _normalize_role(role)
    content = (content or "").strip()
    if not content:
        return
    _hist_cache_append(user_id, role, content)
    db_history_append(user_id, role, content)


# ---------------------------------------------------------------------------
# TTS text resolver (history-aware)
# ---------------------------------------------------------------------------
async def resolve_tts_text(
    user_id: int,
    raw_text: str,
    loop: asyncio.AbstractEventLoop,
) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return raw_text

    # Load history (cache-first, DB fallback)
    history = _hist_cache_get(user_id)
    if history is None:
        try:
            db_rows = await loop.run_in_executor(_DB_EXECUTOR, db_history_fetch, user_id)
            for row in db_rows:
                _hist_cache_append(user_id, row.get("role", "user"), row.get("content", ""))
            history = db_rows
        except Exception as e:
            logger.error(f"resolve_tts_text history fetch failed for user {user_id}: {e}")
            history = []

    if not history:
        return raw_text

    def _last_by_role(role_name: str) -> str | None:
        for row in reversed(history):
            role    = _normalize_role(row.get("role", ""))
            content = (row.get("content") or "").strip()
            if role == role_name and content:
                return content
        return None

    lower   = raw_text.lower()
    compact = lower.replace(" ", "")

    repeat_phrases = (
        "read again", "read that again", "say again", "repeat", "again",
        "អានម្តងទៀត", "អានម្ដងទៀត", "ម្តងទៀត", "ម្ដងទៀត",
        "អានឡើងវិញ", "និយាយម្តងទៀត", "និយាយម្ដងទៀត",
    )
    if any(p in lower for p in repeat_phrases) or any(
        p in compact for p in ("អានម្តងទៀត", "អានម្ដងទៀត", "ម្តងទៀត", "ម្ដងទៀត")
    ):
        last_bot = _last_by_role("assistant")
        if last_bot:
            logger.info(f"resolve_tts_text local repeat for user {user_id}")
            return last_bot

    what_user_said_phrases = (
        "what did i say", "what i said", "my last message",
        "អ្វីដែលខ្ញុំនិយាយ", "ខ្ញុំនិយាយអ្វី", "សារចុងក្រោយរបស់ខ្ញុំ",
    )
    if any(p in lower for p in what_user_said_phrases) or any(
        p in compact for p in ("ខ្ញុំនិយាយអ្វី", "អ្វីដែលខ្ញុំនិយាយ")
    ):
        last_user = _last_by_role("user")
        if last_user:
            logger.info(f"resolve_tts_text local last-user for user {user_id}")
            return last_user

    if not TTS_RESOLVER_AI_ENABLED or not bot_setting_bool_cached("ai_resolver_enabled", True):
        return raw_text
    if not _gemini:
        return raw_text

    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        return raw_text

    context_block = _build_context_block(history)
    if not context_block.strip():
        return raw_text

    system_prompt = (
        "You are a text pre-processor for a Khmer/English TTS bot. "
        "Output ONLY the exact text to be spoken — no labels, no explanation, no markdown.\n\n"
        "Rules:\n"
        "1. Normal sentence → output verbatim.\n"
        "2. 'read again' / 'អានម្តងទៀត' → output last Bot turn verbatim.\n"
        "3. Translation request → output translated text in target language.\n"
        "4. Pronoun references ('that', 'it') → resolve and output resolved text.\n"
        "5. 'what did I say?' → output last User turn.\n"
        "6. Preserve original language unless translation is requested.\n"
        "7. If uncertain → output user's message verbatim."
    )
    combined = (
        f"{system_prompt}\n\n"
        f"Conversation history:\n{context_block}\n\n"
        f"User's new message: {raw_text}\n\n"
        "Output only the text to speak:"
    )

    async def _guarded_call():
        async with semaphore:
            def _call():
                return _gemini.models.generate_content(model=GEMINI_MODEL, contents=combined)
            return await loop.run_in_executor(_AI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(_guarded_call(), timeout=CONV_RESOLVE_TIMEOUT_S)
        resolved = (response.text or "").strip()
        if resolved:
            logger.info(f"resolve_tts_text Gemini resolved for user {user_id}")
            return resolved
        return raw_text
    except asyncio.TimeoutError:
        logger.warning(f"resolve_tts_text timed out for user {user_id} — using raw text")
        return raw_text
    except Exception as e:
        logger.warning(f"resolve_tts_text error for user {user_id}: {e} — using raw text")
        return raw_text



# ---------------------------------------------------------------------------
# Distributed lock helpers — scheduler ownership across bot instances
# ---------------------------------------------------------------------------
def _lock_until_iso(ttl_s: int | float) -> str:
    return _sched_iso(datetime.now(timezone.utc) + timedelta(seconds=max(1, int(ttl_s))))


def db_lock_acquire(lock_key: str = _SCHED_LOCK_KEY, owner: str = _BOT_LOCK_OWNER, ttl_s: int = _SCHED_LOCK_TTL_S) -> bool:
    """Acquire or renew a lightweight Supabase lock.

    This intentionally does NOT use db_call_sync, because a missing bot_locks
    table should not trip the global Supabase circuit breaker and break the bot.

    Flow:
    1) renew if this process already owns the lock
    2) steal only if locked_until is in the past
    3) insert if the lock row does not exist yet
    """
    if not _SCHED_LOCK_ENABLED:
        return True
    if not supabase:
        return not _SCHED_LOCK_REQUIRED

    lock_key = str(lock_key or _SCHED_LOCK_KEY)
    owner = str(owner or _BOT_LOCK_OWNER)[:240]
    now_iso = _sched_iso()
    until_iso = _lock_until_iso(ttl_s)
    update = {"owner": owner, "locked_until": until_iso, "updated_at": now_iso}

    try:
        # Fast path: renew our own lock.
        res = (
            supabase.table("bot_locks")
            .update(update)
            .eq("lock_key", lock_key)
            .eq("owner", owner)
            .execute()
        )
        if getattr(res, "data", None):
            return True

        # Safe takeover: only expired rows are updateable. Postgres re-checks
        # this predicate after row locking, so two instances should not both win.
        res = (
            supabase.table("bot_locks")
            .update(update)
            .eq("lock_key", lock_key)
            .lt("locked_until", now_iso)
            .execute()
        )
        if getattr(res, "data", None):
            return True

        # First boot path: create the lock row. If another instance inserts
        # first, the unique constraint fails and we simply do not own it.
        try:
            res = supabase.table("bot_locks").insert({"lock_key": lock_key, **update}).execute()
            return bool(getattr(res, "data", None))
        except Exception as insert_exc:
            # Unique violation means another instance already owns/created it.
            low = str(insert_exc).lower()
            if "duplicate" in low or "23505" in low or "unique" in low:
                return False
            raise
    except Exception as exc:
        _log_once(
            logging.WARNING,
            f"sched_lock_unavailable:{type(exc).__name__}:{str(exc)[:120]}",
            "Scheduler distributed lock unavailable. Run bot_locks SQL or set SCHED_LOCK_ENABLED=0. Error: %s",
            exc,
        )
        return not _SCHED_LOCK_REQUIRED


def db_lock_release(lock_key: str = _SCHED_LOCK_KEY, owner: str = _BOT_LOCK_OWNER) -> bool:
    if not _SCHED_LOCK_ENABLED or not supabase:
        return True
    try:
        res = (
            supabase.table("bot_locks")
            .delete()
            .eq("lock_key", str(lock_key))
            .eq("owner", str(owner)[:240])
            .execute()
        )
        return bool(getattr(res, "data", None))
    except Exception as exc:
        _log_once(logging.WARNING, "sched_lock_release_failed", "Scheduler lock release failed: %s", exc)
        return False


def db_lock_read(lock_key: str = _SCHED_LOCK_KEY) -> dict | None:
    if not supabase:
        return None
    try:
        res = (
            supabase.table("bot_locks")
            .select("lock_key, owner, locked_until, updated_at")
            .eq("lock_key", str(lock_key))
            .limit(1)
            .execute()
        )
        rows = list(getattr(res, "data", None) or [])
        return rows[0] if rows else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Database helpers — scheduled broadcasts
# ---------------------------------------------------------------------------
# IMPORTANT: keep these status values compatible with the existing
# Supabase scheduled_broadcasts_status_check constraint.  Many deployed DBs
# only allow: pending, sending, done, failed, cancelled.
#
# An unconfirmed preview is stored as status="pending" plus this error_msg
# marker. The scheduler only sends pending rows where error_msg IS NULL.
SCHED_STATUS_DRAFT     = "pending"    # DB-compatible unconfirmed preview marker state
SCHED_STATUS_PENDING   = "pending"    # confirmed; scheduler may send it
SCHED_STATUS_SENDING   = "sending"    # claimed by scheduler
SCHED_STATUS_DONE      = "done"
SCHED_STATUS_FAILED    = "failed"
SCHED_STATUS_CANCELLED = "cancelled"
SCHED_DRAFT_MARKER     = "__awaiting_admin_confirmation__"

_SCHED_CLAIM_RE = re.compile(r"sending_started_at=([^\s]+)")


def _sched_to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sched_iso(dt: datetime | None = None) -> str:
    return _sched_to_utc(dt or datetime.now(timezone.utc)).isoformat()


def _sched_parse_iso(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        # Supabase/Postgres may return a trailing Z.
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return _sched_to_utc(dt)
    except Exception:
        return None


def _sched_claim_marker(dt: datetime | None = None) -> str:
    return f"sending_started_at={_sched_iso(dt)}"


def _sched_claim_time(row: dict) -> datetime | None:
    msg = str(row.get("error_msg") or "")
    m = _SCHED_CLAIM_RE.search(msg)
    if m:
        return _sched_parse_iso(m.group(1))
    # Backward-compatible fallback for old rows that were already in "sending"
    # before this fix. Do not use this for newly claimed rows.
    return _sched_parse_iso(row.get("broadcast_at"))


def _sched_is_draft(row: dict | None) -> bool:
    if not row:
        return False
    return (
        str(row.get("status") or "").lower() == SCHED_STATUS_PENDING
        and str(row.get("error_msg") or "") == SCHED_DRAFT_MARKER
    )


def _sched_is_confirmed_pending(row: dict | None) -> bool:
    """Return True only for rows that are safe for the scheduler to send.

    Confirmed schedule = status pending + error_msg is NULL/empty.
    Draft preview      = status pending + SCHED_DRAFT_MARKER.

    This strict check prevents accidental sends for rows that are pending but
    contain an error/debug marker in error_msg.
    """
    if not row:
        return False
    status = str(row.get("status") or "").strip().lower()
    error_msg = row.get("error_msg")
    error_clean = "" if error_msg is None else str(error_msg).strip()
    return status == SCHED_STATUS_PENDING and error_clean == ""


def db_sched_insert(payload: dict, admin_id: int, broadcast_at: datetime) -> dict:
    """Create a schedule preview row.

    Important fix: new rows remain DB-compatible with status="pending", but
    are tagged with SCHED_DRAFT_MARKER in error_msg. The scheduler only sends
    pending rows where error_msg IS NULL, so the broadcast cannot fire until
    the admin taps ✅ Confirm Schedule.
    """
    if not supabase:
        raise RuntimeError("Supabase not configured.")

    row = {
        "admin_id":      int(admin_id),
        "photo_file_id": payload.get("photo_file_id"),
        "caption":       payload.get("caption"),
        "plain_text":    payload.get("text"),
        "broadcast_at":  _sched_iso(broadcast_at),
        "status":        SCHED_STATUS_PENDING,
        "error_msg":     SCHED_DRAFT_MARKER,
    }
    res = db_call_sync(
        "sched_insert",
        lambda: supabase.table("scheduled_broadcasts").insert(row).execute(),
        critical=True,
    )
    if not getattr(res, "data", None):
        raise RuntimeError("scheduled_broadcasts insert returned no row.")
    return res.data[0]


def db_sched_fetch_due(limit: int = _SCHED_DUE_LIMIT) -> list[dict]:
    if not supabase:
        return []
    now = _sched_iso()

    # Fetch DB-compatible pending rows and filter locally.
    # Confirmed schedule: status="pending" and error_msg is NULL/empty.
    # Unconfirmed preview: status="pending" and error_msg=SCHED_DRAFT_MARKER.
    # This avoids requiring a DB migration for a new "draft" status.
    res = db_call_sync(
        "sched_fetch_due",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .select("*")
            .eq("status", SCHED_STATUS_PENDING)
            .lte("broadcast_at", now)
            .order("broadcast_at")
            .limit(max(_SCHED_SCAN_LIMIT, int(limit or _SCHED_DUE_LIMIT) * 25))
            .execute()
        ),
        default=None,
    )
    rows = list(getattr(res, "data", None) or [])
    return [r for r in rows if _sched_is_confirmed_pending(r)][: max(1, int(limit or _SCHED_DUE_LIMIT))]

def db_sched_claim(row_id: int) -> bool:
    """Atomically claim one confirmed pending schedule.

    The scheduler never sends preview rows because `_sched_is_confirmed_pending`
    requires error_msg to be empty before this function updates the row.
    """
    if not supabase:
        return False

    current = db_sched_fetch_one(row_id)
    if not _sched_is_confirmed_pending(current):
        return False

    now_iso = _sched_iso()
    broadcast_at = _sched_parse_iso(current.get("broadcast_at"))
    if not broadcast_at or broadcast_at > datetime.now(timezone.utc):
        return False

    update = {
        "status": SCHED_STATUS_SENDING,
        "error_msg": _sched_claim_marker(),
    }
    res = db_call_sync(
        f"sched_claim:{row_id}",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .update(update)
            .eq("id", int(row_id))
            .eq("status", SCHED_STATUS_PENDING)
            .is_("error_msg", "null")
            .lte("broadcast_at", now_iso)
            .execute()
        ),
        default=None,
    )
    return bool(getattr(res, "data", None))

def db_sched_confirm(row_id: int, admin_id: int) -> tuple[bool, str, dict | None]:
    """Confirm one schedule preview.

    DB-compatible design:
    - Before confirmation: status="pending" and error_msg=SCHED_DRAFT_MARKER
    - After confirmation:  status="pending" and error_msg=NULL
    """
    row = db_sched_fetch_one(row_id)
    if not row:
        return False, "not_found", None
    if int(row.get("admin_id") or 0) != int(admin_id):
        return False, "not_owner", row

    status = str(row.get("status") or "").lower()
    if status != SCHED_STATUS_PENDING:
        return False, status or "invalid_status", row
    if not _sched_is_draft(row):
        return True, "already_confirmed", row

    broadcast_at = _sched_parse_iso(row.get("broadcast_at"))
    if not broadcast_at or broadcast_at <= datetime.now(timezone.utc):
        db_sched_set_status(
            row_id,
            SCHED_STATUS_CANCELLED,
            error_msg="Cancelled: schedule time passed before admin confirmation.",
        )
        return False, "expired", row

    res = db_call_sync(
        f"sched_confirm:{row_id}",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .update({"status": SCHED_STATUS_PENDING, "error_msg": None})
            .eq("id", int(row_id))
            .eq("admin_id", int(admin_id))
            .eq("status", SCHED_STATUS_PENDING)
            .eq("error_msg", SCHED_DRAFT_MARKER)
            .execute()
        ),
        default=None,
    )
    saved = (getattr(res, "data", None) or [None])[0]
    if not saved:
        return False, "race_lost", db_sched_fetch_one(row_id)
    return True, "confirmed", saved

def db_sched_set_status(row_id: int, status: str, **extra) -> None:
    if not supabase:
        return
    update = {"status": str(status), **extra}
    try:
        db_call_sync(
            f"sched_set_status:{row_id}:{status}",
            lambda: (
                supabase.table("scheduled_broadcasts")
                .update(update)
                .eq("id", int(row_id))
                .execute()
            ),
            default=None,
        )
    except Exception as e:
        logger.error(f"db_sched_set_status #{row_id} -> {status}: {e}")


def db_sched_mark_stale_sending_failed() -> int:
    """Recover schedules stuck in `sending` and expire unconfirmed drafts.

    This version uses `sending_started_at` stored in `error_msg` when claiming.
    That fixes false-failure for old due broadcasts that are actively sending.
    """
    if not supabase:
        return 0

    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - max(60, int(_SCHED_SENDING_STALE_SECONDS))
    changed = 0

    sending_res = db_call_sync(
        "sched_fetch_sending_for_stale_check",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .select("id, broadcast_at, error_msg, status")
            .eq("status", SCHED_STATUS_SENDING)
            .limit(100)
            .execute()
        ),
        default=None,
    )
    for row in list(getattr(sending_res, "data", None) or []):
        started_at = _sched_claim_time(row)
        if started_at and started_at.timestamp() <= cutoff:
            db_sched_set_status(
                int(row["id"]),
                SCHED_STATUS_FAILED,
                error_msg=f"Marked failed: stuck in sending for more than {_SCHED_SENDING_STALE_SECONDS}s",
            )
            changed += 1

    # Draft rows are previews that were never confirmed. They must never fire.
    # Once their requested time has passed, mark them cancelled to keep the DB clean.
    draft_res = db_call_sync(
        "sched_expire_old_drafts",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .update({
                "status": SCHED_STATUS_CANCELLED,
                "error_msg": "Cancelled: schedule draft expired before confirmation.",
            })
            .eq("status", SCHED_STATUS_PENDING)
            .eq("error_msg", SCHED_DRAFT_MARKER)
            .lte("broadcast_at", _sched_iso(now))
            .execute()
        ),
        default=None,
    )
    changed += len(getattr(draft_res, "data", None) or [])
    return changed


def db_sched_fetch_admin_pending(admin_id: int) -> list[dict]:
    if not supabase:
        return []
    res = db_call_sync(
        f"sched_fetch_admin_pending:{admin_id}",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .select("id, broadcast_at, plain_text, caption, photo_file_id, status, error_msg")
            .eq("admin_id", int(admin_id))
            .eq("status", SCHED_STATUS_PENDING)
            .order("broadcast_at")
            .execute()
        ),
        default=None,
    )
    rows = list(getattr(res, "data", None) or [])
    return [r for r in rows if _sched_is_draft(r) or _sched_is_confirmed_pending(r)]

def db_sched_fetch_one(row_id: int) -> dict | None:
    if not supabase:
        return None
    res = db_call_sync(
        f"sched_fetch_one:{row_id}",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .select("*")
            .eq("id", int(row_id))
            .limit(1)
            .execute()
        ),
        default=None,
    )
    rows = list(getattr(res, "data", None) or [])
    return rows[0] if rows else None


def _sched_can_edit(row: dict | None, admin_id: int | None = None) -> tuple[bool, str]:
    if not row:
        return False, "not_found"
    if admin_id is not None and int(row.get("admin_id") or 0) != int(admin_id):
        return False, "not_owner"
    if str(row.get("status") or "").lower() != SCHED_STATUS_PENDING:
        return False, str(row.get("status") or "invalid_status")
    broadcast_at = _sched_parse_iso(row.get("broadcast_at"))
    if broadcast_at and broadcast_at <= datetime.now(timezone.utc):
        return False, "time_passed"
    return True, "editable"


def db_sched_update_time(row_id: int, admin_id: int, broadcast_at: datetime) -> tuple[bool, str, dict | None]:
    row = db_sched_fetch_one(row_id)
    ok, reason = _sched_can_edit(row, admin_id)
    if not ok:
        return False, reason, row
    if _sched_to_utc(broadcast_at) <= datetime.now(timezone.utc):
        return False, "new_time_in_past", row

    res = db_call_sync(
        f"sched_update_time:{row_id}",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .update({"broadcast_at": _sched_iso(broadcast_at)})
            .eq("id", int(row_id))
            .eq("admin_id", int(admin_id))
            .eq("status", SCHED_STATUS_PENDING)
            .execute()
        ),
        default=None,
    )
    saved = (getattr(res, "data", None) or [None])[0]
    return (True, "updated", saved) if saved else (False, "race_lost", db_sched_fetch_one(row_id))


def db_sched_update_text(row_id: int, admin_id: int, text: str) -> tuple[bool, str, dict | None]:
    text = (text or "").strip()
    row = db_sched_fetch_one(row_id)
    ok, reason = _sched_can_edit(row, admin_id)
    if not ok:
        return False, reason, row
    if not text:
        return False, "empty_text", row

    has_photo = bool(row.get("photo_file_id"))
    if has_photo and len(text) > 1024:
        return False, "caption_too_long", row
    if not has_photo and len(text) > TELE_MSG_LIMIT:
        return False, "text_too_long", row

    update = {"caption": text, "plain_text": None} if has_photo else {"plain_text": text, "caption": None}
    res = db_call_sync(
        f"sched_update_text:{row_id}",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .update(update)
            .eq("id", int(row_id))
            .eq("admin_id", int(admin_id))
            .eq("status", SCHED_STATUS_PENDING)
            .execute()
        ),
        default=None,
    )
    saved = (getattr(res, "data", None) or [None])[0]
    return (True, "updated", saved) if saved else (False, "race_lost", db_sched_fetch_one(row_id))


def db_sched_update_photo(row_id: int, admin_id: int, photo_file_id: str, caption: str = "") -> tuple[bool, str, dict | None]:
    photo_file_id = (photo_file_id or "").strip()
    caption = (caption or "").strip()
    row = db_sched_fetch_one(row_id)
    ok, reason = _sched_can_edit(row, admin_id)
    if not ok:
        return False, reason, row
    if not photo_file_id:
        return False, "empty_photo", row
    if len(caption) > 1024:
        return False, "caption_too_long", row

    res = db_call_sync(
        f"sched_update_photo:{row_id}",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .update({"photo_file_id": photo_file_id, "caption": caption, "plain_text": None})
            .eq("id", int(row_id))
            .eq("admin_id", int(admin_id))
            .eq("status", SCHED_STATUS_PENDING)
            .execute()
        ),
        default=None,
    )
    saved = (getattr(res, "data", None) or [None])[0]
    return (True, "updated", saved) if saved else (False, "race_lost", db_sched_fetch_one(row_id))


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------
def _parse_dt(text: str) -> datetime | None:
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Keyboard helpers
# ---------------------------------------------------------------------------
def get_sched_confirm_kb(row_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ បញ្ជាក់ Schedule", callback_data=f"sched_ok:{row_id}"),
         InlineKeyboardButton("❌ បោះបង់",           callback_data=f"sched_no:{row_id}")],
        [InlineKeyboardButton("✏️ Edit Time", callback_data=f"sched_edit_time:{row_id}"),
         InlineKeyboardButton("📝 Edit Text", callback_data=f"sched_edit_text:{row_id}")],
        [InlineKeyboardButton("🖼 Replace Photo", callback_data=f"sched_edit_photo:{row_id}")],
    ])


def get_sched_detail_kb(row: dict) -> InlineKeyboardMarkup:
    row_id = int(row.get("id") or 0)
    editable, _reason = _sched_can_edit(row, int(row.get("admin_id") or 0))
    rows: list[list[InlineKeyboardButton]] = []
    if _sched_is_draft(row):
        rows.append([
            InlineKeyboardButton("✅ Confirm", callback_data=f"sched_ok:{row_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"sched_no:{row_id}"),
        ])
    if editable:
        rows.extend([
            [InlineKeyboardButton("✏️ Edit Time", callback_data=f"sched_edit_time:{row_id}"),
             InlineKeyboardButton("📝 Edit Text", callback_data=f"sched_edit_text:{row_id}")],
            [InlineKeyboardButton("🖼 Replace Photo", callback_data=f"sched_edit_photo:{row_id}"),
             InlineKeyboardButton("🗑️ Cancel Schedule", callback_data=f"sched_cancel_confirm:{row_id}")],
        ])
    rows.append([InlineKeyboardButton("⬅️ Schedules", callback_data="admin_schedules"),
                 InlineKeyboardButton("❌ Close", callback_data="sched_close")])
    return InlineKeyboardMarkup(rows)


def get_admin_dashboard_kb() -> InlineKeyboardMarkup:
    """Admin Dashboard V2 keyboard with direct panels and settings."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Dashboard",   callback_data="admin_home"),
         InlineKeyboardButton("🩺 Health",      callback_data="admin_health")],
        [InlineKeyboardButton("👥 Users",       callback_data="admin_users"),
         InlineKeyboardButton("🕘 Recent History", callback_data="admin_history")],
        [InlineKeyboardButton("⚙️ Settings",    callback_data="admin_settings"),
         InlineKeyboardButton("📊 Stats",       callback_data="admin_stats")],
        [InlineKeyboardButton("📢 Broadcast",   callback_data="admin_broadcast"),
         InlineKeyboardButton("⏰ Schedules",   callback_data="admin_schedules")],
        [InlineKeyboardButton("🔑 API Keys",    callback_data="admin_api"),
         InlineKeyboardButton("🔄 Refresh",     callback_data="admin_home")],
        [InlineKeyboardButton("❌ Close",       callback_data="admin_close")],
    ])


def get_admin_action_kb(cancel_callback: str = "admin_cancel_state") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Admin Dashboard", callback_data="admin_home"),
         InlineKeyboardButton("❌ Cancel",          callback_data=cancel_callback)],
    ])


def get_api_admin_kb() -> InlineKeyboardMarkup:
    """Admin API management panel.

    Buttons are intentionally small and direct so admins can create/list/setup
    API keys without typing /api subcommands.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Create API Key", callback_data="api_create"),
         InlineKeyboardButton("📋 List Keys",      callback_data="api_list")],
        [InlineKeyboardButton("🧩 Setup SQL",      callback_data="api_sql"),
         InlineKeyboardButton("🩺 API Status",     callback_data="api_status")],
        [InlineKeyboardButton("❔ Help",           callback_data="api_help"),
         InlineKeyboardButton("⬅️ Admin",         callback_data="api_back")],
        [InlineKeyboardButton("❌ Close",          callback_data="api_close")],
    ])


def get_api_list_kb(rows: list[dict]) -> InlineKeyboardMarkup:
    kbd_rows: list[list[InlineKeyboardButton]] = []
    active_rows = [r for r in rows if r.get("active")]

    for row in active_rows[:10]:
        row_id = str(row.get("id") or "").strip()
        prefix = str(row.get("key_prefix") or "?").strip()
        label = f"🚫 Revoke #{row_id} {prefix}"[:60]
        if row_id:
            kbd_rows.append([InlineKeyboardButton(label, callback_data=f"api_revoke:{row_id}")])

    kbd_rows.extend([
        [InlineKeyboardButton("🔄 Refresh List", callback_data="api_list"),
         InlineKeyboardButton("➕ Create New",   callback_data="api_create")],
        [InlineKeyboardButton("⬅️ API Menu",    callback_data="api_menu"),
         InlineKeyboardButton("❌ Close",        callback_data="api_close")],
    ])
    return InlineKeyboardMarkup(kbd_rows)


def get_api_revoke_confirm_kb(identifier: str) -> InlineKeyboardMarkup:
    safe_ident = str(identifier or "")[:32]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Confirm Revoke", callback_data=f"api_revoke_confirm:{safe_ident}")],
        [InlineKeyboardButton("⬅️ Back to List", callback_data="api_list"),
         InlineKeyboardButton("❌ Close",        callback_data="api_close")],
    ])


def get_schedules_list_kb(rows: list[dict], page: int, page_size: int = 5) -> InlineKeyboardMarkup:
    total   = max(1, (len(rows) + page_size - 1) // page_size)
    chunk   = rows[page * page_size : (page + 1) * page_size]
    kbd_rows = []
    for r in chunk:
        try:
            dt_str = _fmt_dt(datetime.fromisoformat(str(r["broadcast_at"]).replace("Z", "+00:00")))
        except Exception:
            dt_str = str(r.get("broadcast_at", "?"))
        icon = "🟡" if _sched_is_draft(r) else "🟢"
        kbd_rows.append([InlineKeyboardButton(f"{icon} #{r['id']}  {dt_str}", callback_data=f"sched_view:{r['id']}")])
    nav = []
    if page > 0:            nav.append(InlineKeyboardButton("⬅️", callback_data=f"sched_page:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total}", callback_data="sched_noop"))
    if page < total - 1:    nav.append(InlineKeyboardButton("➡️", callback_data=f"sched_page:{page+1}"))
    if nav:
        kbd_rows.append(nav)
    kbd_rows.append([InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
                     InlineKeyboardButton("📅 New", callback_data="admin_schedule_new")])
    kbd_rows.append([InlineKeyboardButton("❌ បិទ", callback_data="sched_close")])
    return InlineKeyboardMarkup(kbd_rows)



def _sched_status_label(row: dict) -> str:
    if _sched_is_draft(row):
        return "preview"
    return str(row.get("status") or "?")


def _sched_content_preview(row: dict, limit: int = 500) -> str:
    if row.get("photo_file_id"):
        base = row.get("caption") or "(photo, no caption)"
    else:
        base = row.get("plain_text") or "(empty)"
    base = str(base).strip()
    if len(base) > limit:
        base = base[:limit] + "…"
    return base


def _sched_detail_text(row: dict) -> str:
    try:
        dt = _sched_parse_iso(row.get("broadcast_at"))
        dt_str = _fmt_dt(dt) if dt else str(row.get("broadcast_at", "?"))
    except Exception:
        dt_str = str(row.get("broadcast_at", "?"))
    media = "Photo + caption" if row.get("photo_file_id") else "Text"
    status = _sched_status_label(row)
    content = html.escape(_sched_content_preview(row))
    return (
        f"📋 <b>Schedule #{int(row.get('id') or 0)}</b>\n"
        f"⏰ {html.escape(dt_str)}\n"
        f"ស្ថានភាព: <b>{html.escape(status)}</b>\n"
        f"ប្រភេទ: <b>{html.escape(media)}</b>\n\n"
        f"{content}"
    )


# ---------------------------------------------------------------------------
# Status timer
# ---------------------------------------------------------------------------
_STATUS_FRAMES = [
    "⏳ កំពុងបង្កើតសំឡេង ·", "⏳ កំពុងបង្កើតសំឡេង ··",
    "⏳ កំពុងបង្កើតសំឡេង ···", "⏳ កំពុងបង្កើតសំឡេង ····",
]
_TRANSCRIBE_FRAMES = [
    "🎙️ កំពុង Transcribe ·", "🎙️ កំពុង Transcribe ··",
    "🎙️ កំពុង Transcribe ···", "🎙️ កំពុង Transcribe ····",
]
_OCR_FRAMES = [
    "🔍 កំពុង OCR រូបភាព ·", "🔍 កំពុង OCR រូបភាព ··",
    "🔍 កំពុង OCR រូបភាព ···", "🔍 កំពុង OCR រូបភាព ····",
]
_AUDIO_FILE_FRAMES = [
    "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ·", "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ··",
    "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ···", "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ····",
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
            with suppress(BadRequest, TelegramError):
                await msg.edit_text(frames[frame_idx % len(frames)])
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
def _rounded_speed(speed: float) -> float:
    return round(max(_SPEED_MIN, min(_SPEED_MAX, speed)), 4)


@functools.lru_cache(maxsize=64)
def _build_atempo_chain(speed: float) -> str:
    if abs(speed - 1.0) < 1e-6:
        return "atempo=1.0"
    stages: list[str] = []
    r = speed
    if r < 1.0:
        while r < 0.5 - 1e-9 and r > 1e-6:
            stages.append("atempo=0.5")
            r = round(r / 0.5, 6)
        stages.append(f"atempo={max(0.5, r):.6f}")
    else:
        while r > 2.0 + 1e-9:
            stages.append("atempo=2.0")
            r = round(r / 2.0, 6)
        stages.append(f"atempo={min(2.0, r):.6f}")
    return ",".join(stages)


def _detect_tts_lang_key(text: str) -> str:
    khmer_chars = len(_KHMER_RE.findall(text or ""))
    total_alpha = sum(1 for c in (text or "") if c.isalpha())
    is_khmer = khmer_chars > (total_alpha * 0.3) if total_alpha else khmer_chars > 0
    return "km" if is_khmer else "en"


def _detect_voice(text: str, gender: str) -> str:
    gender = gender if gender in ("female", "male") else "female"
    return VOICE_MAP[_detect_tts_lang_key(text)][gender]


def _clean_tts_text_for_edge(text: str) -> str:
    """Remove hidden/control chars that can make Edge TTS return no audio."""
    text = (text or "")
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d"):
        text = text.replace(ch, "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tts_voice_candidates(text: str, gender: str) -> list[str]:
    """Primary voice first, then same-language fallback; optional cross-language fallback."""
    gender = gender if gender in ("female", "male") else "female"
    lang = _detect_tts_lang_key(text)
    other_gender = "male" if gender == "female" else "female"
    candidates = [VOICE_MAP[lang][gender], VOICE_MAP[lang][other_gender]]
    if EDGE_TTS_CROSS_LANG_FALLBACK:
        other_lang = "en" if lang == "km" else "km"
        candidates.extend([VOICE_MAP[other_lang][gender], VOICE_MAP[other_lang][other_gender]])
    seen: set[str] = set()
    unique: list[str] = []
    for voice in candidates:
        if voice and voice not in seen:
            seen.add(voice)
            unique.append(voice)
    return unique


async def _edge_tts_stream_once(chunk_text: str, voice: str) -> bytes:
    async def _collect() -> bytes:
        audio_chunks: list[bytes] = []
        # Do not pass rate here; speed is applied by FFmpeg atempo. This avoids
        # invalid edge-tts parameter combinations that can produce NoAudioReceived.
        communicate = edge_tts.Communicate(chunk_text, voice)
        async for message in communicate.stream():
            if message.get("type") == "audio" and message.get("data"):
                audio_chunks.append(message["data"])
        return b"".join(audio_chunks)

    return await asyncio.wait_for(_collect(), timeout=EDGE_TTS_STREAM_TIMEOUT_S)


async def _edge_tts_stream_with_retry(chunk_text: str, voices: list[str]) -> tuple[bytes, str]:
    last_errors: list[str] = []
    for voice in voices:
        for attempt in range(1, EDGE_TTS_RETRIES + 1):
            try:
                mp3_data = await _edge_tts_stream_once(chunk_text, voice)
                if mp3_data:
                    if attempt > 1:
                        logger.info("edge-tts recovered on attempt %s with voice=%s", attempt, voice)
                    return mp3_data, voice
                raise RuntimeError("No audio was received from Edge TTS.")
            except asyncio.TimeoutError:
                last_errors.append(
                    f"voice={voice} attempt={attempt}: timeout after {EDGE_TTS_STREAM_TIMEOUT_S:.0f}s"
                )
            except Exception as e:
                last_errors.append(f"voice={voice} attempt={attempt}: {type(e).__name__}: {e}")

            if attempt < EDGE_TTS_RETRIES:
                await asyncio.sleep(EDGE_TTS_RETRY_DELAY_S * attempt)

        logger.warning("edge-tts voice failed, trying fallback if available: %s", voice)

    detail = " | ".join(last_errors[-8:])
    raise RuntimeError(f"edge-tts failed after retries/fallbacks. {detail}")


def _tts_user_error_message(exc: Exception | str) -> str:
    msg = str(exc).lower()
    if "no audio" in msg or "edge-tts failed" in msg:
        return (
            "❌ TTS service មិនបានបញ្ជូនសំឡេងមកវិញ។\n"
            "✅ Bot បាន retry និងសាក voice fallback រួចហើយ។ សូមសាកម្តងទៀត ឬប្តូរសំឡេង Male/Female។"
        )
    if "timeout" in msg:
        return "❌ TTS យឺតពេក/timeout។ សូមសាកអត្ថបទខ្លីជាងនេះ ឬសាកម្តងទៀត។"
    if "ffmpeg" in msg:
        return "❌ FFmpeg មានបញ្ហាក្នុងការបម្លែងសំឡេង។ សូមពិនិត្យ FFmpeg នៅលើ server។"
    return "❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"


async def generate_voice(text: str, gender: str, speed: float, output_path: str) -> bytes:
    text = _clean_tts_text_for_edge(text)
    if not text:
        raise ValueError("generate_voice: text must not be empty")

    voices = _tts_voice_candidates(text, gender)
    text_chunks = _split_text_chunks(text, max_chars=EDGE_TTS_CHUNK_CHARS)
    if not text_chunks:
        raise ValueError("generate_voice: no speakable text chunks")

    mp3_parts: list[bytes] = []
    used_voices: list[str] = []
    for idx, chunk_text in enumerate(text_chunks, 1):
        try:
            chunk_mp3, used_voice = await _edge_tts_stream_with_retry(chunk_text, voices)
            mp3_parts.append(chunk_mp3)
            used_voices.append(used_voice)
        except Exception as e:
            preview = chunk_text[:80].replace("\n", " ")
            raise RuntimeError(
                f"edge-tts failed at chunk {idx}/{len(text_chunks)} ({preview!r}): {e}"
            ) from e

    mp3_data = b"".join(mp3_parts)
    if not mp3_data:
        raise RuntimeError("edge-tts returned empty audio after retries")

    if len(set(used_voices)) > 1:
        logger.info("edge-tts used fallback voices: %s", sorted(set(used_voices)))

    speed_key = _rounded_speed(speed)
    af        = _build_atempo_chain(speed_key) if abs(speed_key - DEFAULT_SPEED) > 1e-4 else None
    cmd       = [_FFMPEG_EXE, "-y", "-f", "mp3", "-i", "pipe:0"]
    if af:
        cmd += ["-filter:a", af]
    cmd += ["-c:a", "libopus", "-b:a", "32k", output_path]

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5,
        )
        _, stderr_data = await asyncio.wait_for(proc.communicate(input=mp3_data), timeout=60)
    except asyncio.TimeoutError:
        raise RuntimeError("FFmpeg timed out after 60s")

    if proc.returncode != 0:
        snippet = (stderr_data or b"").decode(errors="replace")[-400:]
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode}): {snippet}")

    try:
        loop        = asyncio.get_running_loop()
        audio_bytes = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: open(output_path, "rb").read()),
            timeout=10,
        )
    except asyncio.TimeoutError:
        raise RuntimeError("Timed out reading output audio file")
    except OSError as e:
        raise RuntimeError(f"Failed to read output audio file: {e}") from e
    return audio_bytes


async def generate_voice_limited(text: str, gender: str, speed: float, output_path: str) -> bytes:
    sem = _TTS_CHUNK_SEMAPHORE
    if sem is None:
        return await generate_voice(text, gender, speed, output_path)
    async with sem:
        return await generate_voice(text, gender, speed, output_path)


# ---------------------------------------------------------------------------
# Gemini transcription helpers
# ---------------------------------------------------------------------------
async def transcribe_voice(ogg_path: str) -> str:
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(None, lambda: open(ogg_path, "rb").read())
    except OSError as e:
        raise RuntimeError(f"Cannot read voice file: {e}") from e

    prompt    = (
        "Transcribe this audio exactly as spoken. "
        "Output ONLY the transcribed text — no labels, no explanation. "
        "Support both Khmer and English."
    )
    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        raise RuntimeError("Gemini semaphore not initialised.")

    async def _guarded_call():
        async with semaphore:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                        prompt,
                    ],
                )
            return await loop.run_in_executor(_AI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(_guarded_call(), timeout=60)
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini transcription timed out after 60s")


async def transcribe_audio_file(file_path: str, mime_type: str) -> str:
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(None, lambda: open(file_path, "rb").read())
    except OSError as e:
        raise RuntimeError(f"Cannot read audio file: {e}") from e

    prompt    = (
        "Transcribe this audio exactly as spoken. "
        "Output ONLY the transcribed text — no labels, no explanation. "
        "Support both Khmer and English."
    )
    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        raise RuntimeError("Gemini semaphore not initialised.")

    async def _guarded_call():
        async with semaphore:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
            return await loop.run_in_executor(_AI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(_guarded_call(), timeout=90)
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini audio transcription timed out after 90s")


# ---------------------------------------------------------------------------
# Text chunking for TTS
# ---------------------------------------------------------------------------
def _split_text_chunks(text: str, max_chars: int = TTS_CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences     = [s for s in _SENTENCE_RE.split(text) if s.strip()]
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
                    space    = max_chars - len(current)
                    current += sent[pos: pos + space]
                    pos     += space
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
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Detect image MIME type from magic bytes
# ---------------------------------------------------------------------------
def _detect_image_mime(path: str) -> str:
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        if header[:8] == b"\x89PNG\r\n\x1a\n":   return "image/png"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP": return "image/webp"
        if header[:2] == b"\xff\xd8":              return "image/jpeg"
        if header[:6] in (b"GIF87a", b"GIF89a"):   return "image/gif"
    except Exception:
        pass
    return "image/jpeg"


# ---------------------------------------------------------------------------
# OCR (async wrapper)
# ---------------------------------------------------------------------------
async def ocr_image(image_path: str, mime_type: str = "image/jpeg") -> str:
    if not _ocr_configured():
        raise RuntimeError(_ocr_status_for_user())
    loop = asyncio.get_running_loop()
    try:
        image_bytes = await loop.run_in_executor(None, lambda: open(image_path, "rb").read())
    except OSError as e:
        raise RuntimeError(f"Cannot read image file: {e}") from e

    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        raise RuntimeError("AI semaphore not initialised.")

    async def _guarded_call():
        async with semaphore:
            return await loop.run_in_executor(
                _AI_EXECUTOR, lambda: ask_ocr_image(image_bytes, mime_type)[0]
            )

    try:
        return await asyncio.wait_for(_guarded_call(), timeout=OCR_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise RuntimeError(f"OCR timed out after {OCR_TIMEOUT_SECONDS}s")


# ---------------------------------------------------------------------------
# Cooldown check
# ---------------------------------------------------------------------------
async def _check_cooldown(reply_target, user_id: int) -> bool:
    lock = _get_user_lock(user_id)
    if lock.locked():
        await safe_send(lambda: reply_target.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return True
    now  = time.monotonic()
    last = _get_last_tts(user_id)
    if now - last < USER_COOLDOWN_S:
        rem = round(USER_COOLDOWN_S - (now - last), 1)
        await safe_send(lambda r=rem: reply_target.reply_text(f"⏳ សូមរង់ចាំ {r}s មុននឹងផ្ញើម្តងទៀត។"))
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
        await safe_send(lambda: bot.send_message(chat_id=chat_id, text="❌ រកអត្ថបទមិនឃើញ។"))
        return

    set_last_tts_text(user_id, text)
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        file_path = _make_temp_ogg()
        try:
            audio_bytes = await generate_voice_limited(chunk, gender, speed, file_path)
            sent = await safe_send(
                lambda ab=audio_bytes, ci=i, ct=total: bot.send_voice(
                    chat_id=chat_id,
                    voice=io.BytesIO(ab),
                    caption=f"🗣️ {BOT_TAG}  [{ci}/{ct}]",
                    reply_markup=get_main_kb(gender),
                )
            )
            if sent:
                save_text_cache(
                    sent.message_id, chunk,
                    chat_id=chat_id, user_id=user_id, username=username,
                )
                set_last_tts_text(user_id, chunk)
        except Exception as e:
            logger.error(f"paged TTS chunk {i}/{total} error: {e}", exc_info=True)
            await safe_send(
                lambda ci=i, ct=total, err=e: bot.send_message(
                    chat_id=chat_id, text=f"{_tts_user_error_message(err)}\nChunk {ci}/{ct}"
                )
            )
        finally:
            _cleanup(file_path)

        if i < total:
            await asyncio.sleep(0.25)

    _set_last_tts(user_id)


# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------
def get_main_kb(gender: str) -> InlineKeyboardMarkup:
    f_btn = "👩 សំឡេងស្រី" + (" ✅" if gender == "female" else "")
    m_btn = "👨 សំឡេងប្រុស" + (" ✅" if gender == "male" else "")
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f_btn, callback_data="tg_female"),
         InlineKeyboardButton(m_btn, callback_data="tg_male")],
        [InlineKeyboardButton("🎚️ ល្បឿនសំឡេង", callback_data="show_speed")],
    ])


def get_speed_kb(current_speed: float) -> InlineKeyboardMarkup:
    speed_row = [
        InlineKeyboardButton(
            lbl + (" ✅" if abs(val - current_speed) < 0.01 else ""),
            callback_data=cb,
        )
        for cb, (lbl, val) in SPEED_OPTIONS.items()
    ]
    return InlineKeyboardMarkup([
        speed_row,
        [InlineKeyboardButton("🔙 ត្រឡប់", callback_data="hide_speed")],
    ])


def get_transcription_kb(transcript_msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📢 AI អាន",  callback_data=f"tts_transcript:{transcript_msg_id}"),
        InlineKeyboardButton("🗑️ លុប",    callback_data=f"del_transcript:{transcript_msg_id}"),
    ]])


def get_audio_file_kb(msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📢 បំលែងទៅសំឡេង TTS", callback_data=f"audio_tts:{msg_id}"),
        InlineKeyboardButton("🗑️ លុប",               callback_data=f"audio_del:{msg_id}"),
    ]])


def get_ocr_confirm_kb(msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("▶️ អាន", callback_data=f"doc_read:{msg_id}"),
        InlineKeyboardButton("🗑️ លុប", callback_data=f"doc_del:{msg_id}"),
    ]])


def get_broadcast_confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ បញ្ជាក់ Broadcast", callback_data="bc_confirm"),
        InlineKeyboardButton("❌ បោះបង់",           callback_data="bc_cancel"),
    ]])


def _clamp_users_page(users: list[dict], page: int, page_size: int = 7) -> int:
    total_pages = max(1, (len(users) + page_size - 1) // page_size)
    return max(0, min(int(page or 0), total_pages - 1))


def _parse_user_back_ref(ref: str | None) -> tuple[str, int]:
    ref = (ref or "p0").strip().lower()
    if len(ref) >= 2 and ref[0] in ("p", "s", "h") and ref[1:].isdigit():
        return ref[0], int(ref[1:])
    if ref.isdigit():
        return "p", int(ref)
    return "p", 0


def _user_back_callback(ref: str | None) -> tuple[str, str]:
    kind, page = _parse_user_back_ref(ref)
    if kind == "s":
        return f"users_search_page:{page}", "⬅️ Search"
    if kind == "h":
        return f"history_page:{page}", "⬅️ History"
    return f"users_page:{page}", "⬅️ Users"


def get_users_page_kb(users: list[dict], page: int, page_size: int = 7) -> InlineKeyboardMarkup:
    page = _clamp_users_page(users, page, page_size)
    total_pages = max(1, (len(users) + page_size - 1) // page_size)
    chunk       = users[page * page_size : page * page_size + page_size]
    rows: list[list[InlineKeyboardButton]] = []
    for u in chunk:
        uid = int(u.get("user_id") or 0)
        label_name = (u.get("username") or str(uid))[:22]
        rows.append([InlineKeyboardButton(
            f"👤 {label_name}  ({uid})",
            callback_data=f"user_view:{uid}:p{page}",
        )])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"users_page:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"users_page:{page+1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("🔎 Search User", callback_data="users_search"),
                 InlineKeyboardButton("🔄 Refresh", callback_data=f"users_page:{page}")])
    rows.append([InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
                 InlineKeyboardButton("❌ Close", callback_data="users_close")])
    return InlineKeyboardMarkup(rows)


def get_user_search_page_kb(users: list[dict], page: int, page_size: int = 7) -> InlineKeyboardMarkup:
    page = _clamp_users_page(users, page, page_size)
    total_pages = max(1, (len(users) + page_size - 1) // page_size)
    chunk       = users[page * page_size : page * page_size + page_size]
    rows: list[list[InlineKeyboardButton]] = []
    for u in chunk:
        uid = int(u.get("user_id") or 0)
        label_name = (u.get("username") or str(uid))[:22]
        rows.append([InlineKeyboardButton(
            f"👤 {label_name}  ({uid})",
            callback_data=f"user_view:{uid}:s{page}",
        )])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"users_search_page:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"users_search_page:{page+1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("🔎 New Search", callback_data="users_search"),
                 InlineKeyboardButton("👥 All Users", callback_data="users_page:0")])
    rows.append([InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
                 InlineKeyboardButton("❌ Close", callback_data="users_close")])
    return InlineKeyboardMarkup(rows)


def get_user_search_prompt_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👥 All Users", callback_data="users_page:0")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ Close", callback_data="users_close")],
    ])


def _history_compact_text(text: str, max_len: int = 90) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def get_recent_history_kb(rows: list[dict], page: int, page_size: int = 7) -> InlineKeyboardMarkup:
    page = _clamp_users_page(rows, page, page_size)
    total_pages = max(1, (len(rows) + page_size - 1) // page_size)
    chunk = rows[page * page_size : page * page_size + page_size]

    kbd_rows: list[list[InlineKeyboardButton]] = []
    for item in chunk:
        uid = int(item.get("user_id") or 0)
        username = str(item.get("username") or "").strip().lstrip("@")
        label_name = (username or str(uid))[:20]
        turns = int(item.get("turns") or 1)
        role_icon = "📝"
        kbd_rows.append([InlineKeyboardButton(
            f"{role_icon} {label_name} ({uid}) · {turns}",
            callback_data=f"history_user:{uid}:{page}",
        )])

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"history_page:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"history_page:{page+1}"))
    if nav:
        kbd_rows.append(nav)

    kbd_rows.append([InlineKeyboardButton("🔄 Refresh", callback_data="history_refresh"),
                     InlineKeyboardButton("👥 Users", callback_data="users_page:0")])
    kbd_rows.append([InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
                     InlineKeyboardButton("❌ Close", callback_data="history_close")])
    return InlineKeyboardMarkup(kbd_rows)


def get_user_history_kb(user_id: int, back_ref: str = "p0", page: int = 0, total_rows: int = 0, page_size: int = ADMIN_HISTORY_PAGE_SIZE) -> InlineKeyboardMarkup:
    back_callback, back_label = _user_back_callback(back_ref)
    page_size = max(5, min(20, int(page_size or ADMIN_HISTORY_PAGE_SIZE)))
    total_pages = max(1, (int(total_rows or 0) + page_size - 1) // page_size)
    page = max(0, min(int(page or 0), total_pages - 1))

    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("🔄 Refresh History", callback_data=f"user_history:{user_id}:{back_ref}:{page}"),
         InlineKeyboardButton("🧹 Clear History", callback_data=f"user_clearhist:{user_id}:{back_ref}")],
    ]

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"user_history:{user_id}:{back_ref}:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("Next ➡️", callback_data=f"user_history:{user_id}:{back_ref}:{page+1}"))
    if total_pages > 1:
        rows.append(nav)

    rows.extend([
        [InlineKeyboardButton("👤 User Detail", callback_data=f"user_view:{user_id}:{back_ref}"),
         InlineKeyboardButton(back_label, callback_data=back_callback)],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ Close", callback_data="users_close")],
    ])
    return InlineKeyboardMarkup(rows)


def get_user_detail_kb(user_id: int, blocked: bool, back_ref: str = "p0") -> InlineKeyboardMarkup:
    back_callback, back_label = _user_back_callback(back_ref)
    rows = [
        [InlineKeyboardButton("💬 Chat", callback_data=f"user_chat:{user_id}"),
         InlineKeyboardButton("📜 Full History", callback_data=f"user_history:{user_id}:{back_ref}")],
        [InlineKeyboardButton("🧹 Clear History", callback_data=f"user_clearhist:{user_id}:{back_ref}"),
         InlineKeyboardButton("♻️ Reset Prefs", callback_data=f"user_resetprefs:{user_id}:{back_ref}")],
    ]
    if blocked:
        rows.append([InlineKeyboardButton("✅ Unblock User", callback_data=f"user_unblock:{user_id}:{back_ref}")])
    else:
        rows.append([InlineKeyboardButton("🚫 Block User", callback_data=f"user_block:{user_id}:{back_ref}")])
    rows.append([InlineKeyboardButton(back_label, callback_data=back_callback),
                 InlineKeyboardButton("⬅️ Admin", callback_data="admin_home")])
    return InlineKeyboardMarkup(rows)


def get_bot_settings_kb(settings: dict[str, str]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for key, label in BOT_SETTING_LABELS.items():
        enabled = _setting_bool_from(settings, key, default=True)
        if key == "maintenance_mode":
            state = "ON 🛠️" if enabled else "OFF ✅"
        else:
            state = "ON ✅" if enabled else "OFF ⚠️"
        rows.append([InlineKeyboardButton(f"{label}: {state}", callback_data=f"admin_set:{key}")])
    rows.extend([
        [InlineKeyboardButton("🔄 Refresh", callback_data="admin_settings_refresh"),
         InlineKeyboardButton("🧩 Setup SQL", callback_data="admin_settings_sql")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ Close", callback_data="admin_close")],
    ])
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Admin guard
# ---------------------------------------------------------------------------
def admin_only(handler):
    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        if uid not in ADMIN_IDS:
            if update.message:
                await safe_send(lambda: update.message.reply_text("⛔ អ្នកមិនមានសិទ្ធិប្រើពាក្យបញ្ជានេះទេ។"))
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
    _user_to_admin[target_id]    = admin_id


def _close_session(admin_id: int) -> int | None:
    target_id = _admin_chat_target.pop(admin_id, None)
    if target_id is not None:
        _user_to_admin.pop(target_id, None)
    return target_id


def _get_admin_for_user(user_id: int) -> int | None:
    return _user_to_admin.get(user_id)


# ===========================================================================
# Broadcast helper
# ===========================================================================
def _safe_broadcast_html(text: str | None, max_chars: int) -> str | None:
    """Escape admin/user broadcast content before using Telegram HTML mode."""
    raw = (text or "").strip()
    if not raw:
        return None
    if len(raw) > max_chars:
        raw = raw[: max_chars - 1] + "…"
    return html.escape(raw, quote=False)


async def _run_broadcast_to_all(
    bot,
    admin_id: int,
    pending: dict,
    label: str = "Broadcast",
) -> tuple[int, int, int]:
    broadcast_semaphore = _BROADCAST_SEMAPHORE
    if broadcast_semaphore is None:
        logger.error(f"{label}: _BROADCAST_SEMAPHORE not initialised.")
        return (0, 0, 0)

    loop     = asyncio.get_running_loop()
    user_ids = await loop.run_in_executor(None, get_all_user_ids)
    total    = len(user_ids)

    if total == 0:
        await safe_send(lambda: bot.send_message(
            chat_id=admin_id,
            text=f"⚠️ {label}: មិនមានអ្នកប្រើប្រាស់ registered ណាមួយទេ។",
        ))
        return (0, 0, 0)

    sent = failed = blocked = 0
    photo_file_id = pending.get("photo_file_id")
    safe_caption  = _safe_broadcast_html(pending.get("caption"), 1024)
    safe_text     = _safe_broadcast_html(pending.get("text"), 4096)

    if not photo_file_id and not safe_text:
        await safe_send(lambda: bot.send_message(
            chat_id=admin_id, text=f"❌ {label}: Broadcast text is empty."
        ))
        return (0, 0, 0)

    progress_msg = await safe_send(lambda: bot.send_message(
        chat_id=admin_id,
        text=f"📡 {label} — កំពុង Broadcast ទៅ {total} នាក់..."
    ))

    async def _send_one(uid: int) -> str:
        async with broadcast_semaphore:
            for attempt in range(2):
                try:
                    if photo_file_id:
                        await bot.send_photo(
                            chat_id=uid,
                            photo=photo_file_id,
                            caption=safe_caption,
                            parse_mode="HTML" if safe_caption else None,
                        )
                    else:
                        await bot.send_message(
                            chat_id=uid,
                            text=safe_text or " ",
                            parse_mode="HTML",
                        )
                    return "sent"
                except Forbidden:
                    return "blocked"
                except RetryAfter as e:
                    await asyncio.sleep(e.retry_after + 1)
                    if attempt == 1:
                        return "failed"
                except BadRequest as e:
                    logger.error(f"{label} Telegram BadRequest uid={uid}: {e}")
                    return "failed"
                except Exception as e:
                    logger.error(f"{label} error uid={uid} attempt={attempt}: {e}")
                    if attempt == 1:
                        return "failed"
                    await asyncio.sleep(0.5 * (attempt + 1))
        return "failed"

    batch_size = max(1, BROADCAST_BATCH_SIZE)
    for start in range(0, total, batch_size):
        batch = user_ids[start:start + batch_size]
        results = await asyncio.gather(
            *(_send_one(uid) for uid in batch),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"{label} batch task error: {result}")
                failed += 1
            elif result == "sent":
                sent += 1
            elif result == "blocked":
                blocked += 1
            else:
                failed += 1

        completed = min(start + len(batch), total)
        if progress_msg and (completed == total or completed % max(25, batch_size) == 0):
            with suppress(Exception):
                pct = int(completed / total * 100) if total else 0
                await progress_msg.edit_text(
                    f"📡 {label}: {pct}% ({completed}/{total})\n"
                    f"✅ {sent}  🚫 {blocked}  ❌ {failed}"
                )

        if completed < total:
            await asyncio.sleep(max(0.0, BROADCAST_INTER_BATCH_DELAY))

    report = (
        f"✅ <b>{html.escape(label)}</b> រួចរាល់!\n\n"
        f"👥 សរុប: {total}\n📨 បានផ្ញើ: {sent}\n"
        f"🚫 Blocked: {blocked}\n❌ Failed: {failed}"
    )
    try:
        if progress_msg:
            await safe_send(lambda: progress_msg.edit_text(report, parse_mode="HTML"))
        else:
            await safe_send(lambda: bot.send_message(chat_id=admin_id, text=report, parse_mode="HTML"))
    except Exception as e:
        logger.error(f"{label} report error: {e}")

    return (sent, failed, blocked)

# ===========================================================================
# BROADCAST (immediate)
# ===========================================================================
@admin_only
async def broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _pending_broadcast.pop(update.effective_user.id, None)
    context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
    await safe_send(lambda: update.message.reply_text(
        "📡 <b>Admin Broadcast</b>\n\n"
        "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Broadcast ។\n"
        "👉 អាចផ្ញើរូបភាព + Caption រួមគ្នា ឬ តែ text ។\n\n"
        "វាយ /cancel ដើម្បីបោះបង់។",
        parse_mode="HTML",
    ))


async def _admin_summary_counts(admin_id: int) -> dict:
    """Fetch lightweight admin dashboard counts without blocking the event loop."""
    def _fetch() -> dict:
        counts = {
            "total_users": 0,
            "pending_sched": 0,
            "blocked_users": 0,
            "active_api_keys": 0,
            "api_table_ok": False,
            "settings_db_ok": False,
            "maintenance_mode": False,
        }
        try:
            counts["total_users"] = len(get_all_user_ids()) if supabase else 0
        except Exception as e:
            logger.warning(f"admin user count failed: {e}")
        try:
            counts["pending_sched"] = len(db_sched_fetch_admin_pending(admin_id)) if supabase else 0
        except Exception as e:
            logger.warning(f"admin schedule count failed: {e}")
        try:
            counts["blocked_users"] = db_blocked_user_count()
        except Exception:
            counts["blocked_users"] = 0
        try:
            settings, status = db_bot_settings_fetch_all()
            counts["settings_db_ok"] = bool(status.get("db_ok"))
            counts["maintenance_mode"] = _setting_bool_from(settings, "maintenance_mode", False)
        except Exception:
            counts["settings_db_ok"] = False
        try:
            api_status = db_ai_api_key_status()
            counts["active_api_keys"] = int(api_status.get("active_count") or 0)
            counts["api_table_ok"] = bool(api_status.get("table_ok"))
        except Exception:
            counts["api_table_ok"] = False
        return counts

    return await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, _fetch)


def _ok_bad(ok: bool, ok_text: str = "OK", bad_text: str = "OFF") -> str:
    return f"✅ {ok_text}" if ok else f"⚠️ {bad_text}"


async def _admin_home_text(admin_id: int, title: str = "🛠️ Admin Dashboard V2") -> str:
    counts = await _admin_summary_counts(admin_id)
    settings, settings_status = await get_bot_settings_async()
    ffmpeg_ok = bool(_FFMPEG_EXE and os.path.exists(_FFMPEG_EXE))
    temp_ok = False
    temp_info = ""
    try:
        temp_dir = _get_temp_dir()
        temp_ok = os.path.isdir(temp_dir) and os.access(temp_dir, os.W_OK)
        temp_files = glob.glob(os.path.join(temp_dir, f"{_TMP_PREFIX}*"))
        temp_info = f"{len(temp_files)} files"
    except Exception:
        temp_info = "unknown"

    api_ready = bool(os.environ.get("AI_API_KEY", "").strip()) or bool(counts.get("active_api_keys"))
    maintenance = _setting_bool_from(settings, "maintenance_mode", False)
    tts_on = _setting_bool_from(settings, "tts_enabled", True)
    ocr_on = _setting_bool_from(settings, "ocr_enabled", True)
    voice_on = _setting_bool_from(settings, "voice_transcribe_enabled", True)
    return (
        f"{title}\n\n"
        "<b>System Status</b>\n"
        f"🤖 Telegram: <b>✅ OK</b>\n"
        f"🗄️ Supabase: <b>{_ok_bad(bool(supabase))}</b>\n"
        f"⚙️ Settings DB: <b>{_ok_bad(bool(settings_status.get('db_ok')), 'READY', 'MEMORY/SETUP')}</b>\n"
        f"🧠 Hugging Face: <b>{_ok_bad(bool(_hf_client))}</b>\n"
        f"🔍 OCR provider: <b>{_ok_bad(_ocr_configured(), 'READY', 'OFF')}</b>\n"
        f"🎧 FFmpeg: <b>{_ok_bad(ffmpeg_ok, 'OK', 'ERROR')}</b>\n"
        f"⏱️ Uptime: <b>{html.escape(_format_uptime())}</b>\n"
        f"📁 Temp: <b>{_ok_bad(temp_ok, 'OK', 'ERROR')}</b> <code>{html.escape(temp_info)}</code>\n\n"
        "<b>Live Controls</b>\n"
        f"🛠️ Maintenance: <b>{'ON ⚠️' if maintenance else 'OFF ✅'}</b>\n"
        f"🗣️ TTS: <b>{'ON ✅' if tts_on else 'OFF ⚠️'}</b>\n"
        f"🔍 OCR: <b>{'ON ✅' if ocr_on else 'OFF ⚠️'}</b>\n"
        f"🎙️ Voice: <b>{'ON ✅' if voice_on else 'OFF ⚠️'}</b>\n\n"
        "<b>Quick Stats</b>\n"
        f"👥 Users: <b>{int(counts.get('total_users') or 0)}</b>\n"
        f"🚫 Blocked: <b>{int(counts.get('blocked_users') or 0)}</b>\n"
        f"⏰ Pending schedules: <b>{int(counts.get('pending_sched') or 0)}</b>\n"
        f"🔑 API access: <b>{_ok_bad(api_ready, 'READY', 'SETUP')}</b>\n"
        f"💬 Admin chats: <b>{len(_admin_chat_target)}</b>\n\n"
        "<b>Runtime Since Restart</b>\n"
        f"🗣️ TTS: <b>{_RUNTIME_METRICS.get('tts', 0)}</b> | "
        f"🔍 OCR: <b>{_RUNTIME_METRICS.get('ocr', 0)}</b> | "
        f"🎙️ Voice: <b>{_RUNTIME_METRICS.get('voice', 0)}</b>\n"
        f"🎵 Audio: <b>{_RUNTIME_METRICS.get('audio', 0)}</b> | "
        f"⛔ Blocked hits: <b>{_RUNTIME_METRICS.get('blocked_hits', 0)}</b> | "
        f"⚠️ Disabled hits: <b>{_RUNTIME_METRICS.get('disabled_hits', 0)}</b>\n\n"
        "ចុចប៊ូតុងខាងក្រោម ដើម្បីគ្រប់គ្រង Bot។"
    )


async def _admin_health_text() -> str:
    ffmpeg_ok = bool(_FFMPEG_EXE and os.path.exists(_FFMPEG_EXE))
    temp_dir = ""
    temp_ok = False
    try:
        temp_dir = _get_temp_dir()
        temp_ok = os.path.isdir(temp_dir) and os.access(temp_dir, os.W_OK)
    except Exception:
        temp_dir = "ERROR"

    ocr_ready = _ocr_configured()
    return (
        "🩺 <b>Bot Health</b>\n\n"
        f"🤖 Telegram bot: <b>✅ OK</b>\n"
        f"🗄️ Supabase: <b>{_ok_bad(bool(supabase))}</b>\n"
        f"🧠 Hugging Face: <b>{_ok_bad(bool(_hf_client))}</b>\n"
        f"🔍 OCR: <b>{_ok_bad(bool(ocr_ready), 'READY', 'OFF')}</b>\n"
        f"🎧 FFmpeg: <b>{_ok_bad(ffmpeg_ok, 'OK', 'ERROR')}</b>\n"
        f"📁 Temp folder: <b>{_ok_bad(temp_ok, 'OK', 'ERROR')}</b>\n"
        f"<code>{html.escape(str(temp_dir))}</code>\n\n"
        f"⏰ Scheduler poll: <b>{int(_SCHED_POLL_INTERVAL)}s</b>\n"
        f"🔐 Scheduler lock: <b>{'ON' if _SCHED_LOCK_ENABLED else 'OFF'}</b> / required <b>{'YES' if _SCHED_LOCK_REQUIRED else 'NO'}</b>\n"
        f"📌 Lock status: <b>{html.escape(str(_scheduler_lock_last_status))}</b>\n"
        f"🔑 Lock owner: <code>{html.escape(_BOT_LOCK_OWNER[:80])}</code>\n"
        f"📦 Due batch: <b>{int(_SCHED_DUE_LIMIT)}</b> / scan <b>{int(_SCHED_SCAN_LIMIT)}</b>\n"
        f"🏃 Active schedule jobs: <b>{len(_scheduler_active_ids)}</b>"
    )


async def _admin_stats_text(admin_id: int) -> str:
    counts = await _admin_summary_counts(admin_id)
    settings, status = await get_bot_settings_async()
    return (
        "📊 <b>Admin Dashboard V2 Stats</b>\n\n"
        f"👥 Total users: <b>{int(counts.get('total_users') or 0)}</b>\n"
        f"🚫 Blocked users: <b>{int(counts.get('blocked_users') or 0)}</b>\n"
        f"⏰ Pending schedules: <b>{int(counts.get('pending_sched') or 0)}</b>\n"
        f"🔑 Active API keys: <b>{int(counts.get('active_api_keys') or 0)}</b>\n"
        f"💬 Active admin chats: <b>{len(_admin_chat_target)}</b>\n"
        f"🔒 Active user locks: <b>{len(_user_locks)}</b>\n"
        f"💭 History cache: <b>{len(_hist_cache)}</b> users\n"
        f"⚙️ Settings DB: <b>{_ok_bad(bool(status.get('db_ok')), 'READY', 'MEMORY/SETUP')}</b>\n"
        f"🛠️ Maintenance: <b>{'ON' if _setting_bool_from(settings, 'maintenance_mode', False) else 'OFF'}</b>\n\n"
        "<b>Runtime metrics since restart</b>\n"
        f"🗣️ TTS requests: <b>{_RUNTIME_METRICS.get('tts', 0)}</b>\n"
        f"🔍 OCR requests: <b>{_RUNTIME_METRICS.get('ocr', 0)}</b>\n"
        f"🎙️ Voice transcripts: <b>{_RUNTIME_METRICS.get('voice', 0)}</b>\n"
        f"🎵 Audio transcripts: <b>{_RUNTIME_METRICS.get('audio', 0)}</b>\n"
        f"⛔ Blocked hits: <b>{_RUNTIME_METRICS.get('blocked_hits', 0)}</b>\n"
        f"⚠️ Disabled hits: <b>{_RUNTIME_METRICS.get('disabled_hits', 0)}</b>\n"
        f"❌ Errors: <b>{_RUNTIME_METRICS.get('errors', 0)}</b>\n\n"
        f"🧠 HF model: <code>{html.escape(str(HF_MODEL or 'OFF'))}</code>\n"
        f"🔍 OCR provider: <code>{html.escape(str(OCR_PROVIDER or 'auto'))}</code>\n"
        f"⏱️ Uptime: <b>{html.escape(_format_uptime())}</b>"
    )


async def _admin_open_users_panel(query) -> None:
    users = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, get_all_users_with_names)
    if not users:
        await safe_send(lambda: query.message.edit_text(
            "👥 <b>Users</b>\n\n❌ គ្មានអ្នកប្រើប្រាស់ registered ទេ។",
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return
    await safe_send(lambda: query.message.edit_text(
        f"👥 <b>User Management ({len(users)} users)</b>\n"
        "Select a user, or press 🔎 Search User to find by ID/username.",
        parse_mode="HTML",
        reply_markup=get_users_page_kb(users, page=0),
    ))


async def _admin_open_schedules_panel(query, admin_id: int) -> None:
    rows = await asyncio.get_running_loop().run_in_executor(
        _DB_EXECUTOR, lambda: db_sched_fetch_admin_pending(admin_id)
    )
    if not rows:
        await safe_send(lambda: query.message.edit_text(
            "📋 <b>Scheduled Broadcasts</b>\n\n📭 មិនមាន Scheduled Broadcast ណាមួយទេ។",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📅 New Schedule", callback_data="admin_schedule_new")],
                [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
                 InlineKeyboardButton("❌ Close", callback_data="admin_close")],
            ]),
        ))
        return
    await safe_send(lambda: query.message.edit_text(
        f"📋 <b>Scheduled Broadcasts ({len(rows)} active/preview)</b>\n"
        "🟡 Preview = មិនទាន់ Confirm, 🟢 Pending = រង់ចាំផ្ញើ។\n"
        "ចុចលើ Schedule ដើម្បីមើលលម្អិត ឬ Cancel ។",
        parse_mode="HTML",
        reply_markup=get_schedules_list_kb(rows, page=0),
    ))


async def _admin_open_settings_panel(query, force: bool = False, notice: str = "") -> None:
    settings, status = await get_bot_settings_async(force=force)
    lines = []
    if notice:
        lines.append(f"{html.escape(notice)}\n")
    lines.extend([
        "⚙️ <b>Bot Settings Panel</b>",
        "",
        f"Storage: <b>{_ok_bad(bool(status.get('db_ok')), 'Supabase', 'Memory / setup needed')}</b>",
    ])
    if status.get("error"):
        lines.append(f"Setup note: <code>{html.escape(str(status.get('error'))[:500])}</code>")
    lines.append("")
    for key, label in BOT_SETTING_LABELS.items():
        enabled = _setting_bool_from(settings, key, True)
        if key == "maintenance_mode":
            state = "ON ⚠️" if enabled else "OFF ✅"
        else:
            state = "ON ✅" if enabled else "OFF ⚠️"
        desc = html.escape(BOT_SETTING_DESCRIPTIONS.get(key, ""))
        lines.append(f"{label}: <b>{state}</b> — {desc}")
    lines.extend(["", "ចុច setting ណាមួយដើម្បី ON/OFF។"])
    await safe_send(lambda: query.message.edit_text(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=get_bot_settings_kb(settings),
        disable_web_page_preview=True,
    ))


async def _admin_send_settings_sql(message) -> None:
    for page in _paginate_plain(ADMIN_V2_TABLES_SQL, limit=3800):
        await safe_send(lambda p=page: message.reply_text(
            "🧩 <b>Admin Dashboard V2 SQL</b>\n\n"
            f"<pre>{html.escape(p)}</pre>",
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))


async def _admin_start_broadcast_from_button(query, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    _pending_broadcast.pop(user_id, None)
    context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
    await safe_send(lambda: query.message.edit_text(
        "📢 <b>Admin Broadcast</b>\n\n"
        "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Broadcast ។\n"
        "Bot នឹងបង្ហាញ preview មុនពេលផ្ញើ។\n\n"
        "ចុច Cancel ឬវាយ /cancel ដើម្បីបោះបង់។",
        parse_mode="HTML",
        reply_markup=get_admin_action_kb(),
    ))


async def _admin_start_schedule_from_button(query, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    _sched_payload.pop(user_id, None)
    context.user_data["sched_state"] = SCHED_WAIT_MSG
    await safe_send(lambda: query.message.edit_text(
        "📅 <b>Scheduled Broadcast</b>\n\n"
        "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Schedule ។\n"
        "បន្ទាប់មក Bot នឹងសួរពេលវេលា UTC។\n\n"
        "ចុច Cancel ឬវាយ /cancel ដើម្បីបោះបង់។",
        parse_mode="HTML",
        reply_markup=get_admin_action_kb(),
    ))


@admin_only
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = await _admin_home_text(update.effective_user.id)
    await safe_send(lambda: update.message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
    ))




@admin_only
async def cmd_botsettings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings, status = await get_bot_settings_async(force=True)
    text = (
        "⚙️ <b>Bot Settings Panel</b>\n\n"
        f"Storage: <b>{_ok_bad(bool(status.get('db_ok')), 'Supabase', 'Memory / setup needed')}</b>\n"
        "Use /admin → ⚙️ Settings for button controls."
    )
    await safe_send(lambda: update.message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_bot_settings_kb(settings),
    ))


@admin_only
async def broadcast_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if context.user_data.get("bc_state") != BROADCAST_WAIT_MESSAGE:
        return

    msg           = update.message
    photo_file_id: str | None = None
    caption_text:  str | None = None
    plain_text:    str | None = None

    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text  = msg.caption or ""
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
        "caption":       caption_text,
        "text":          plain_text,
    }

    if photo_file_id:
        preview_cap = html.escape(caption_text) if caption_text else "<i>(គ្មាន Caption)</i>"
        await safe_send(lambda: msg.reply_photo(
            photo=photo_file_id,
            caption=(
                f"👁️ <b>Preview Broadcast</b>\n\n{preview_cap}\n\n"
                "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?"
            ),
            parse_mode="HTML",
            reply_markup=get_broadcast_confirm_kb(),
        ))
    else:
        await safe_send(lambda: msg.reply_text(
            f"👁️ <b>Preview Broadcast</b>\n\n{html.escape(plain_text)}\n\n"
            "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?",
            parse_mode="HTML",
            reply_markup=get_broadcast_confirm_kb(),
        ))


async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data

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
            await safe_send(lambda: query.message.reply_text("⚠️ រកទិន្នន័យ Broadcast មិនឃើញ។ សូមចាប់ផ្ដើមថ្មី។"))
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
    await safe_send(lambda: update.message.reply_text(
        "📅 <b>Scheduled Broadcast</b>\n\n"
        "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Schedule ។\n\n"
        "វាយ /cancel ដើម្បីបោះបង់។",
        parse_mode="HTML",
    ))


@admin_only
async def cmd_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    loop     = asyncio.get_running_loop()
    rows     = await loop.run_in_executor(None, db_sched_fetch_admin_pending, admin_id)
    if not rows:
        await safe_send(lambda: update.message.reply_text("📭 មិនមាន Scheduled Broadcast ណាមួយទេ។"))
        return
    await safe_send(lambda: update.message.reply_text(
        f"📋 <b>Scheduled Broadcasts ({len(rows)} pending)</b>\n"
        "ចុចលើ Schedule ដើម្បីមើលលម្អិត ឬ Cancel ។",
        parse_mode="HTML",
        reply_markup=get_schedules_list_kb(rows, page=0),
    ))


@admin_only
async def cmd_cancelschedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    args     = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(lambda: update.message.reply_text(
            "❌ Usage: /cancelschedule &lt;id&gt;\nឬប្រើ /schedules ដើម្បីជ្រើស។",
            parse_mode="HTML",
        ))
        return
    row_id = int(args[0])
    loop   = asyncio.get_running_loop()
    row    = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
    if not row:
        await safe_send(lambda: update.message.reply_text(f"❌ រកមិនឃើញ Schedule #{row_id}។"))
        return
    if row["admin_id"] != admin_id:
        await safe_send(lambda: update.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
        return
    if row["status"] != "pending":
        st = row["status"]
        await safe_send(lambda: update.message.reply_text(
            f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
            parse_mode="HTML",
        ))
        return
    await loop.run_in_executor(None, db_sched_set_status, row_id, "cancelled")
    await safe_send(lambda: update.message.reply_text(
        f"✅ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
    ))


async def _handle_sched_content(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        return False
    if context.user_data.get("sched_state") != SCHED_WAIT_MSG:
        return False

    msg           = update.message
    photo_file_id: str | None = None
    caption_text:  str | None = None
    plain_text:    str | None = None

    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text  = msg.caption or ""
    elif msg.text:
        plain_text = msg.text.strip()
        if not plain_text:
            await safe_send(lambda: msg.reply_text("⚠️ អត្ថបទមិនអាចទទេបាន។ សូមវាយសារ ឬ ផ្ញើរូបភាព។"))
            return True
    else:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return True

    _sched_payload[user_id] = {"photo_file_id": photo_file_id, "caption": caption_text, "text": plain_text}
    context.user_data["sched_state"] = SCHED_WAIT_TIME
    await safe_send(lambda: msg.reply_text(
        "🕐 <b>ពេលវេលា Broadcast (UTC)</b>\n\n"
        "វាយកាលបរិច្ឆេទ និងម៉ោង ។\n"
        "ទម្រង់: <code>YYYY-MM-DD HH:MM</code>\n"
        "ឧទាហរណ៍: <code>2025-12-25 09:00</code>",
        parse_mode="HTML",
    ))
    return True


async def _handle_sched_datetime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
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
        await safe_send(lambda: msg.reply_text(
            "❌ ទម្រង់ពេលវេលាខុស។\nឧទាហរណ៍ត្រឹមត្រូវ: <code>2025-12-25 09:00</code>",
            parse_mode="HTML",
        ))
        return True

    now = datetime.now(timezone.utc)
    if broadcast_at <= now:
        await safe_send(lambda: msg.reply_text(
            "❌ ពេលវេលាត្រូវតែជាអនាគត (UTC) ។\n"
            f"ឥឡូវ: <code>{_fmt_dt(now)}</code>",
            parse_mode="HTML",
        ))
        return True

    # Do not pop until Supabase save succeeds. If the DB call fails, the admin
    # can retry the time without re-sending the broadcast content.
    payload = _sched_payload.get(user_id)
    if not payload:
        context.user_data.pop("sched_state", None)
        await safe_send(lambda: msg.reply_text(
            "❌ រកទិន្នន័យ Schedule មិនឃើញ (session expired)។\n"
            "សូមចាប់ផ្ដើម /schedule ម្តងទៀត។"
        ))
        return True

    loop = asyncio.get_running_loop()
    try:
        row = await loop.run_in_executor(None, db_sched_insert, payload, user_id, broadcast_at)
    except Exception as e:
        logger.error(f"db_sched_insert failed: {e}", exc_info=True)
        await safe_send(lambda: msg.reply_text(
            "❌ មានបញ្ហាក្នុងការ Save Schedule ។ សូមព្យាយាមបញ្ចូលពេលវេលាម្តងទៀត។"
        ))
        return True

    # Save succeeded. Clear conversation state. The DB row is an unconfirmed
    # preview marker, so it cannot send until ✅ Confirm Schedule is pressed.
    _sched_payload.pop(user_id, None)
    context.user_data.pop("sched_state", None)

    row_id = row["id"]
    dt_str = _fmt_dt(broadcast_at)

    if payload.get("photo_file_id"):
        cap_preview = html.escape(payload["caption"]) if payload.get("caption") else "<i>(គ្មាន Caption)</i>"
        await safe_send(lambda: msg.reply_photo(
            photo=payload["photo_file_id"],
            caption=(
                f"📅 <b>Preview Schedule #{row_id}</b>\n"
                f"⏰ {dt_str}\n"
                "ស្ថានភាព: <b>Preview — មិនទាន់បញ្ជាក់</b>\n\n"
                f"{cap_preview}"
            ),
            parse_mode="HTML",
            reply_markup=get_sched_confirm_kb(row_id),
        ))
    else:
        await safe_send(lambda: msg.reply_text(
            f"📅 <b>Preview Schedule #{row_id}</b>\n"
            f"⏰ {dt_str}\n"
            "ស្ថានភាព: <b>Preview — មិនទាន់បញ្ជាក់</b>\n\n"
            f"{html.escape(payload.get('text') or '')}",
            parse_mode="HTML",
            reply_markup=get_sched_confirm_kb(row_id),
        ))
    return True


def _sched_clear_edit_state(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("sched_state", None)
    context.user_data.pop("sched_edit_row_id", None)


def _sched_edit_error_text(row_id: int, reason: str) -> str:
    mapping = {
        "not_found": "❌ រកមិនឃើញ Schedule ។",
        "not_owner": "⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។",
        "time_passed": f"⚠️ Schedule #{row_id} ជិត/ផុតពេលហើយ — មិនអាច Edit បាន។",
        "new_time_in_past": "❌ ពេលថ្មីត្រូវតែជាអនាគត (UTC)។",
        "empty_text": "⚠️ អត្ថបទមិនអាចទទេបាន។",
        "caption_too_long": "⚠️ Caption វែងពេក។ Telegram caption limit ប្រហែល 1024 តួអក្សរ។",
        "text_too_long": f"⚠️ Text វែងពេក។ សូមឲ្យក្រោម {TELE_MSG_LIMIT} តួអក្សរ។",
        "empty_photo": "⚠️ មិនមានរូបភាព។",
        "race_lost": "⚠️ Schedule ត្រូវបានផ្លាស់ប្តូរដោយ process ផ្សេង។ សូម refresh ម្តងទៀត។",
    }
    return mapping.get(str(reason), f"⚠️ មិនអាច Edit Schedule #{row_id}: <b>{html.escape(str(reason))}</b>")


async def _handle_sched_edit_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id) or context.user_data.get("sched_state") != SCHED_EDIT_WAIT_TIME:
        return False
    msg = update.message
    row_id = int(context.user_data.get("sched_edit_row_id") or 0)
    if not row_id:
        _sched_clear_edit_state(context)
        await safe_send(lambda: msg.reply_text("❌ Edit session expired. Open /schedules again."))
        return True
    if not msg.text:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើ Text ពេលវេលា UTC។"))
        return True
    new_dt = _parse_dt(msg.text)
    if new_dt is None:
        await safe_send(lambda: msg.reply_text(
            "❌ ទម្រង់ពេលវេលាខុស។\nឧទាហរណ៍: <code>2026-12-25 09:00</code>",
            parse_mode="HTML",
        ))
        return True
    ok, reason, row = await asyncio.get_running_loop().run_in_executor(
        None, db_sched_update_time, row_id, user_id, new_dt
    )
    if not ok:
        await safe_send(lambda: msg.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
        return True
    _sched_clear_edit_state(context)
    await safe_send(lambda: msg.reply_text(
        f"✅ Schedule <b>#{row_id}</b> time updated.\n⏰ New time: <b>{_fmt_dt(_sched_to_utc(new_dt))}</b>",
        parse_mode="HTML",
        reply_markup=get_sched_detail_kb(row),
    ))
    return True


async def _handle_sched_edit_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id) or context.user_data.get("sched_state") != SCHED_EDIT_WAIT_TEXT:
        return False
    msg = update.message
    row_id = int(context.user_data.get("sched_edit_row_id") or 0)
    if not row_id:
        _sched_clear_edit_state(context)
        await safe_send(lambda: msg.reply_text("❌ Edit session expired. Open /schedules again."))
        return True
    if not msg.text or not msg.text.strip():
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើអត្ថបទថ្មី។"))
        return True
    ok, reason, row = await asyncio.get_running_loop().run_in_executor(
        None, db_sched_update_text, row_id, user_id, msg.text
    )
    if not ok:
        await safe_send(lambda: msg.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
        return True
    _sched_clear_edit_state(context)
    await safe_send(lambda: msg.reply_text(
        f"✅ Schedule <b>#{row_id}</b> text/caption updated.",
        parse_mode="HTML",
        reply_markup=get_sched_detail_kb(row),
    ))
    return True


async def _handle_sched_edit_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id) or context.user_data.get("sched_state") != SCHED_EDIT_WAIT_PHOTO:
        return False
    msg = update.message
    row_id = int(context.user_data.get("sched_edit_row_id") or 0)
    if not row_id:
        _sched_clear_edit_state(context)
        await safe_send(lambda: msg.reply_text("❌ Edit session expired. Open /schedules again."))
        return True
    if not msg.photo:
        await safe_send(lambda: msg.reply_text("⚠️ សូមផ្ញើរូបភាពថ្មី + Caption (optional)។"))
        return True
    photo_file_id = msg.photo[-1].file_id
    caption = msg.caption or ""
    ok, reason, row = await asyncio.get_running_loop().run_in_executor(
        None, db_sched_update_photo, row_id, user_id, photo_file_id, caption
    )
    if not ok:
        await safe_send(lambda: msg.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
        return True
    _sched_clear_edit_state(context)
    await safe_send(lambda: msg.reply_text(
        f"✅ Schedule <b>#{row_id}</b> photo replaced.",
        parse_mode="HTML",
        reply_markup=get_sched_detail_kb(row),
    ))
    return True


async def _cb_admin_dashboard(query, user_id: int, context, data: str):
    if not _is_admin(user_id):
        await safe_send(lambda: query.message.reply_text("⛔ Admin only."))
        return

    if data == "admin_close":
        context.user_data.pop("bc_state", None)
        context.user_data.pop("sched_state", None)
        context.user_data.pop("sched_edit_row_id", None)
        _pending_broadcast.pop(user_id, None)
        _sched_payload.pop(user_id, None)
        with suppress(Exception):
            await query.message.delete()
        return

    if data in ("admin_home", "admin_refresh"):
        context.user_data.pop("bc_state", None)
        context.user_data.pop("sched_state", None)
        context.user_data.pop("sched_edit_row_id", None)
        text = await _admin_home_text(user_id)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return

    if data == "admin_cancel_state":
        context.user_data.pop("bc_state", None)
        context.user_data.pop("sched_state", None)
        context.user_data.pop("sched_edit_row_id", None)
        _pending_broadcast.pop(user_id, None)
        _sched_payload.pop(user_id, None)
        text = await _admin_home_text(user_id, title="✅ Admin action cancelled")
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return

    if data == "admin_stats":
        text = await _admin_stats_text(user_id)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
            disable_web_page_preview=True,
        ))
        return

    if data == "admin_health":
        text = await _admin_health_text()
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return

    if data in ("admin_settings", "admin_settings_refresh"):
        await _admin_open_settings_panel(query, force=True)
        return

    if data == "admin_settings_sql":
        await _admin_send_settings_sql(query.message)
        with suppress(Exception):
            await _admin_open_settings_panel(query, force=True, notice="🧩 SQL sent. Run it in Supabase SQL editor.")
        return

    if data.startswith("admin_set:"):
        key = data.split(":", 1)[1].strip()
        settings, _status = await get_bot_settings_async(force=True)
        current = _setting_bool_from(settings, key, True)
        ok, info = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: db_bot_setting_set(key, not current, user_id),
        )
        notice = f"✅ {BOT_SETTING_LABELS.get(key, key)} updated." if ok else f"⚠️ Could not persist setting: {info}"
        await _admin_open_settings_panel(query, force=True, notice=notice)
        return

    if data == "admin_api":
        await _cb_api_dashboard(query, user_id, context, "api_menu")
        return

    if data == "admin_users":
        await _admin_open_users_panel(query)
        return

    if data == "admin_history":
        await _admin_open_recent_history_panel(query, page=0)
        return

    if data == "admin_schedules":
        await _admin_open_schedules_panel(query, user_id)
        return

    if data == "admin_broadcast":
        await _admin_start_broadcast_from_button(query, context, user_id)
        return

    if data == "admin_schedule_new":
        await _admin_start_schedule_from_button(query, context, user_id)
        return

    # Fallback: reopen dashboard instead of sending a separate command message.
    text = await _admin_home_text(user_id)
    await safe_send(lambda: query.message.edit_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
    ))


def _callback_int_arg(data: str, prefix: str) -> int | None:
    try:
        if not str(data or "").startswith(prefix):
            return None
        value = str(data).split(":", 1)[1]
        return int(value)
    except Exception:
        return None


async def sched_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    user_id = query.from_user.id
    data = query.data or ""

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    with suppress(Exception):
        await query.answer()

    if query.message is None:
        return

    loop = asyncio.get_running_loop()

    if data.startswith("sched_ok:"):
        row_id = _callback_int_arg(data, "sched_ok:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule id."))
            return

        ok, reason, row = await loop.run_in_executor(None, db_sched_confirm, row_id, user_id)
        if not ok:
            if reason == "not_found":
                text = "❌ រកមិនឃើញ Schedule ។"
            elif reason == "not_owner":
                text = "⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"
            elif reason == "expired":
                text = f"⚠️ Schedule #{row_id} ផុតពេលមុនពេលបញ្ជាក់ ដូច្នេះបានបោះបង់។"
            else:
                text = f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{html.escape(str(reason))}</b> — មិនអាចបញ្ជាក់ទេ។"
            with suppress(Exception):
                await query.message.edit_reply_markup(reply_markup=None)
            await safe_send(lambda: query.message.reply_text(text, parse_mode="HTML"))
            return

        try:
            dt_str = _fmt_dt(datetime.fromisoformat(str(row["broadcast_at"]).replace("Z", "+00:00")))
        except Exception:
            dt_str = str(row.get("broadcast_at", "?")) if row else "?"
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        status_note = "បានបញ្ជាក់រួចហើយ" if reason == "already_confirmed" else "បានបញ្ជាក់"
        await safe_send(lambda: query.message.reply_text(
            f"✅ <b>Schedule #{row_id} {status_note}!</b>\n⏰ នឹង Broadcast នៅ {dt_str}",
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_no:"):
        row_id = _callback_int_arg(data, "sched_no:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule id."))
            return
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text("❌ រកមិនឃើញ Schedule ។"))
            return
        if int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
            return
        if str(row.get("status")) in (SCHED_STATUS_DRAFT, SCHED_STATUS_PENDING):
            await loop.run_in_executor(None, db_sched_set_status, row_id, SCHED_STATUS_CANCELLED)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f"❌ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
        ))
        return

    if data == "sched_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "sched_noop":
        return

    if data.startswith("sched_page:"):
        page = _callback_int_arg(data, "sched_page:")
        if page is None:
            return
        rows = await loop.run_in_executor(None, db_sched_fetch_admin_pending, user_id)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=get_schedules_list_kb(rows, page=page))
        return

    if data.startswith("sched_view:"):
        row_id = _callback_int_arg(data, "sched_view:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule id."))
            return
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text("❌ រកមិនឃើញ Schedule ។"))
            return
        if int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
            return
        await safe_send(lambda: query.message.reply_text(
            _sched_detail_text(row),
            parse_mode="HTML",
            reply_markup=get_sched_detail_kb(row),
        ))
        return

    if data.startswith("sched_edit_time:"):
        row_id = _callback_int_arg(data, "sched_edit_time:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule id."))
            return
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = SCHED_EDIT_WAIT_TIME
        context.user_data["sched_edit_row_id"] = row_id
        await safe_send(lambda: query.message.reply_text(
            f"✏️ <b>Edit Schedule #{row_id} Time</b>\n\n"
            "ផ្ញើពេលវេលាថ្មីជា UTC។\n"
            "Format: <code>YYYY-MM-DD HH:MM</code>\n"
            "Example: <code>2026-12-25 09:00</code>\n\n"
            "វាយ /cancel ដើម្បីបោះបង់ edit។",
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_edit_text:"):
        row_id = _callback_int_arg(data, "sched_edit_text:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule id."))
            return
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = SCHED_EDIT_WAIT_TEXT
        context.user_data["sched_edit_row_id"] = row_id
        target = "caption" if row.get("photo_file_id") else "text"
        await safe_send(lambda: query.message.reply_text(
            f"📝 <b>Edit Schedule #{row_id} {target}</b>\n\n"
            "ផ្ញើអត្ថបទថ្មី។ វាយ /cancel ដើម្បីបោះបង់ edit។",
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_edit_photo:"):
        row_id = _callback_int_arg(data, "sched_edit_photo:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule id."))
            return
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = SCHED_EDIT_WAIT_PHOTO
        context.user_data["sched_edit_row_id"] = row_id
        await safe_send(lambda: query.message.reply_text(
            f"🖼 <b>Replace Schedule #{row_id} Photo</b>\n\n"
            "ផ្ញើរូបភាពថ្មី + Caption (optional)។ វាយ /cancel ដើម្បីបោះបង់ edit។",
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_cancel_confirm:"):
        row_id = _callback_int_arg(data, "sched_cancel_confirm:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule id."))
            return
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row or int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text("⛔ អ្នកមិនមានសិទ្ធិ cancel Schedule នេះ។"))
            return
        if row.get("status") not in (SCHED_STATUS_DRAFT, SCHED_STATUS_PENDING):
            st = html.escape(str(row.get("status") or "?"))
            await safe_send(lambda: query.message.reply_text(
                f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
                parse_mode="HTML",
            ))
            return
        await loop.run_in_executor(None, db_sched_set_status, row_id, SCHED_STATUS_CANCELLED)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f"✅ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
        ))

# ---------------------------------------------------------------------------
# Subtitle/text document helpers
# ---------------------------------------------------------------------------
def _is_subtitle_file(filename: str | None) -> bool:
    return bool(filename) and os.path.splitext(filename.lower())[1] in SUBTITLE_EXTENSIONS


def _decode_text_bytes(data: bytes) -> str:
    if not data:
        return ""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _clean_subtitle_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\ufeff?WEBVTT.*?(?:\n\n|\Z)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(NOTE|STYLE|REGION)\b.*?(?:\n\n|\Z)", "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    timestamp_re = re.compile(
        r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[\.,]\d{1,3}\s*-->\s*"
        r"(?:\d{1,2}:)?\d{1,2}:\d{2}[\.,]\d{1,3}.*$"
    )
    lines: list[str] = []
    previous = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.isdigit() or timestamp_re.match(line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\{\\.*?\}", "", line)
        line = re.sub(r"^\s*[-–—]\s*", "", line)
        line = re.sub(r"^\[[^\]]{1,40}\]$", "", line)
        line = re.sub(r"^\([^\)]{1,40}\)$", "", line)
        line = html.unescape(line).strip()
        if not line or line == previous:
            continue
        lines.append(line)
        previous = line

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(cleaned) > MAX_SUBTITLE_CHARS:
        cleaned = cleaned[:MAX_SUBTITLE_CHARS].rsplit("\n", 1)[0].strip()
    return cleaned


async def _download_telegram_file_to_bytes(tg_file, max_bytes: int) -> bytes:
    bio = io.BytesIO()
    await tg_file.download_to_memory(out=bio)
    data = bio.getvalue()
    if len(data) > max_bytes:
        raise ValueError(f"File too large. Max {max_bytes // 1024 // 1024} MB.")
    return data


# ---------------------------------------------------------------------------
# on_document — subtitle / text reader
# ---------------------------------------------------------------------------
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg      = update.message
    user     = update.effective_user
    if not msg or not user or not msg.document:
        return

    sync_user_data(user)
    user_id  = user.id
    chat_id  = msg.chat_id
    document = msg.document
    filename = document.file_name or ""
    mime_type = document.mime_type or ""

    if _is_subtitle_file(filename):
        status_msg = await safe_send(lambda: msg.reply_text("🎬 កំពុងអាន subtitle/text file..."))
        try:
            if document.file_size and document.file_size > MAX_SUBTITLE_BYTES:
                await safe_send(lambda: msg.reply_text(
                    f"❌ File ធំពេក។ Max {MAX_SUBTITLE_BYTES // 1024 // 1024} MB."
                ))
                return

            tg_file    = await context.bot.get_file(document.file_id)
            file_bytes = await _download_telegram_file_to_bytes(tg_file, MAX_SUBTITLE_BYTES)
            raw_text   = _decode_text_bytes(file_bytes)
            cleaned    = _clean_subtitle_text(raw_text)

            if not cleaned:
                await safe_send(lambda: msg.reply_text("❌ មិនមានអត្ថបទអាចអានបានក្នុង file នេះទេ។"))
                return

            preview = cleaned[:1200] + ("\n\n..." if len(cleaned) > 1200 else "")
            sent    = await safe_send(lambda: msg.reply_text(
                "🎬 <b>Subtitle/Text detected</b>\n\n"
                f"📄 File: <code>{html.escape(filename)}</code>\n"
                f"🔤 Characters: <b>{len(cleaned)}</b>\n\n"
                f"<blockquote>{html.escape(preview)}</blockquote>\n\n"
                "ចុច ▶️ អាន ដើម្បីបំលែងទៅសំឡេង។",
                parse_mode="HTML",
                reply_markup=get_ocr_confirm_kb(msg.message_id),
            ))
            save_text_cache(
                msg.message_id, cleaned,
                chat_id=chat_id, user_id=user_id,
                username=user.username or user.first_name,
            )
            set_last_tts_text(user_id, cleaned)
            record_turn(user_id, "user",      f"[subtitle file] {filename}")
            record_turn(user_id, "assistant", cleaned[:CONV_CONTEXT_MAX_CHARS])

        except Exception as e:
            logger.error(f"subtitle reader error: {e}", exc_info=True)
            await safe_send(lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការអាន subtitle/text file។"))
        finally:
            if status_msg:
                with suppress(Exception):
                    await status_msg.delete()
        return

    await safe_send(lambda: msg.reply_text(
        "❌ File នេះមិនទាន់ support ទេ។\n\nSupport subtitle/text:\n• .srt\n• .vtt\n• .txt"
    ))


# ---------------------------------------------------------------------------
# Callback: doc read / audio TTS
# ---------------------------------------------------------------------------
async def _cb_doc_read(query, user_id: int, context, data: str):
    try:
        src_msg_id = int(data.split(":")[1])
    except Exception:
        await safe_send(lambda: query.message.reply_text("❌ Invalid document cache id."))
        return
    if query.message is None:
        return

    chat_id = query.message.chat.id
    loop    = asyncio.get_running_loop()

    full_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, src_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not full_text:
        full_text = get_last_tts_text(user_id) or ""

    if not full_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ។ សូមផ្ញើ file ម្តងទៀត។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    speed  = prefs["speed"]
    uname  = query.from_user.username or query.from_user.first_name or str(user_id)

    with suppress(Exception):
        await query.message.delete()

    set_last_tts_text(user_id, full_text)
    tts_stop  = asyncio.Event()
    tts_timer = asyncio.create_task(send_status_timer(chat_id, context.bot, tts_stop))
    lock      = _get_user_lock(user_id)

    try:
        async with lock:
            await _deliver_paged_tts(
                chat_id=chat_id, bot=context.bot,
                text=full_text, gender=gender, speed=speed,
                user_id=user_id, username=uname,
            )
            record_turn(user_id, "assistant", full_text[:CONV_CONTEXT_MAX_CHARS])
            _set_last_tts(user_id)
    except Exception as e:
        logger.error(f"_cb_doc_read delivery error: {e}", exc_info=True)
        with suppress(Exception):
            await context.bot.send_message(chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។")
    finally:
        await _stop_timer(tts_stop, tts_timer)


async def _fire_scheduled_broadcast(bot, row: dict, already_claimed: bool = False) -> None:
    row_id = int(row["id"])
    admin_id = int(row["admin_id"])
    logger.info(f"Firing scheduled broadcast #{row_id} for admin {admin_id}")
    loop = asyncio.get_running_loop()

    if not already_claimed:
        claimed = await loop.run_in_executor(None, db_sched_claim, row_id)
        if not claimed:
            logger.warning(f"Scheduled broadcast #{row_id} already claimed/cancelled — skipping.")
            return

    sent = failed = blocked = 0
    try:
        pending = {
            "photo_file_id": row.get("photo_file_id"),
            "caption": row.get("caption") or "",
            "text": row.get("plain_text") or "",
        }
        if not pending["photo_file_id"] and not pending["text"]:
            raise RuntimeError("Scheduled broadcast has no photo and no text.")

        sent, failed, blocked = await _run_broadcast_to_all(
            bot, admin_id, pending, label=f"Scheduled #{row_id}"
        )
        await loop.run_in_executor(
            None,
            functools.partial(
                db_sched_set_status,
                row_id,
                SCHED_STATUS_DONE,
                sent_count=sent,
                failed_count=failed,
                blocked_count=blocked,
                error_msg=None,
            ),
        )
    except Exception as e:
        logger.error(f"Scheduled broadcast #{row_id} failed: {e}", exc_info=True)
        await loop.run_in_executor(
            None,
            functools.partial(
                db_sched_set_status,
                row_id,
                SCHED_STATUS_FAILED,
                sent_count=sent,
                failed_count=failed,
                blocked_count=blocked,
                error_msg=str(e)[:1000],
            ),
        )

_scheduler_tasks: set[asyncio.Task] = set()
_scheduler_active_ids: set[int] = set()
_scheduler_lock_last_status = "not_started"


def _scheduler_task_done(task: asyncio.Task, row_id: int) -> None:
    _scheduler_tasks.discard(task)
    _scheduler_active_ids.discard(row_id)
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.warning("Could not inspect scheduled task #%s result: %s", row_id, e)
        return
    if exc:
        logger.error("Scheduled broadcast task #%s crashed: %s", row_id, exc, exc_info=exc)


async def _scheduler_loop(bot, stop_event: asyncio.Event) -> None:
    global _scheduler_lock_last_status
    logger.info("Scheduled broadcast loop started.")
    while not stop_event.is_set():
        lock_acquired = False
        try:
            loop = asyncio.get_running_loop()
            lock_acquired = await loop.run_in_executor(
                None, db_lock_acquire, _SCHED_LOCK_KEY, _BOT_LOCK_OWNER, _SCHED_LOCK_TTL_S
            )
            _scheduler_lock_last_status = "owned" if lock_acquired else "not_owner"
            if not lock_acquired:
                _log_once(
                    logging.INFO,
                    "sched_lock_not_owner",
                    "Scheduler tick skipped because another bot instance owns lock %s",
                    _SCHED_LOCK_KEY,
                )
            else:
                stale_count = await loop.run_in_executor(None, db_sched_mark_stale_sending_failed)
                if stale_count:
                    logger.warning(f"Recovered/expired {stale_count} scheduled broadcast row(s).")
                due = await loop.run_in_executor(None, db_sched_fetch_due, _SCHED_DUE_LIMIT)
                for row in due:
                    row_id = int(row.get("id") or 0)
                    if not row_id or row_id in _scheduler_active_ids:
                        continue

                    # Claim before creating the async send task. This closes the
                    # small race where another process could fetch the same due row
                    # before the task gets CPU time. db_sched_claim also re-checks
                    # broadcast_at <= now, so edit-time changes are respected.
                    claimed = await loop.run_in_executor(None, db_sched_claim, row_id)
                    if not claimed:
                        continue

                    _scheduler_active_ids.add(row_id)
                    task = asyncio.create_task(_fire_scheduled_broadcast(bot, row, already_claimed=True))
                    _scheduler_tasks.add(task)
                    task.add_done_callback(lambda t, rid=row_id: _scheduler_task_done(t, rid))
        except Exception as e:
            _scheduler_lock_last_status = f"error:{type(e).__name__}"
            logger.error(f"Scheduler loop error: {e}")
        finally:
            # Releasing lets another healthy instance take over on the next tick.
            # The DB row claim protects individual broadcasts after task creation.
            if lock_acquired:
                with suppress(Exception):
                    await asyncio.get_running_loop().run_in_executor(
                        None, db_lock_release, _SCHED_LOCK_KEY, _BOT_LOCK_OWNER
                    )
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
    args = context.args or []
    if args:
        query_text = " ".join(args).strip()
        results = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: search_users_by_query(query_text),
        )
        context.user_data["users_search_query"] = query_text
        context.user_data["users_search_results"] = results
        if not results:
            await safe_send(lambda: update.message.reply_text(
                "🔎 <b>User Search</b>\n\n"
                f"No users found for: <code>{html.escape(query_text)}</code>",
                parse_mode="HTML",
                reply_markup=get_user_search_prompt_kb(),
            ))
            return
        await safe_send(lambda: update.message.reply_text(
            "🔎 <b>User Search Results</b>\n\n"
            f"Query: <code>{html.escape(query_text)}</code>\n"
            f"Found: <b>{len(results)}</b> user(s)\n\n"
            "Select a user to view details.",
            parse_mode="HTML",
            reply_markup=get_user_search_page_kb(results, page=0),
        ))
        return

    users = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, get_all_users_with_names)
    if not users:
        await safe_send(lambda: update.message.reply_text("❌ គ្មានអ្នកប្រើប្រាស់ registered ទេ។"))
        return
    await safe_send(lambda: update.message.reply_text(
        f"👥 <b>អ្នកប្រើប្រាស់ ({len(users)} នាក់)</b>\nចុចលើឈ្មោះ ដើម្បីមើល Detail ឬប្រើ 🔎 Search User ។",
        parse_mode="HTML",
        reply_markup=get_users_page_kb(users, page=0),
    ))


@admin_only
async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    args     = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(lambda: update.message.reply_text(
            "❌ Usage: /chat <user_id>\nឬប្រើ /users ដើម្បីជ្រើស user ។"
        ))
        return
    target_id = int(args[0])
    exists    = await asyncio.get_running_loop().run_in_executor(None, user_exists_in_db, target_id)
    if not exists:
        await safe_send(lambda: update.message.reply_text(
            f"❌ User <code>{target_id}</code> មិនមាននៅក្នុង Database ។", parse_mode="HTML"
        ))
        return
    await _open_chat_session(context.bot, admin_id, target_id, context)
    await safe_send(lambda: update.message.reply_text(
        f"💬 <b>Chat Mode បើក</b>\n\nកំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
        "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\nវាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
        parse_mode="HTML",
    ))


@admin_only
async def cmd_endchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id  = update.effective_user.id
    target_id = _close_session(admin_id)
    context.user_data.pop("chat_state", None)
    if target_id is None:
        await safe_send(lambda: update.message.reply_text("ℹ️ អ្នកមិនទាន់ open Chat ណាមួយទេ។"))
        return
    await safe_send(lambda: update.message.reply_text(
        f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។", parse_mode="HTML"
    ))
    with suppress(Exception):
        await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")


async def _open_chat_session(bot, admin_id: int, target_id: int, context):
    _open_session(admin_id, target_id)
    context.user_data["chat_state"] = CHAT_WAIT_MESSAGE
    with suppress(Exception):
        await bot.send_message(chat_id=target_id, text="🔔 Admin ចង់ Chat ជាមួយអ្នក។ ផ្ញើសារតបមកបាន!")


async def _show_user_search_results(query, context: ContextTypes.DEFAULT_TYPE, page: int = 0) -> None:
    search_query = (context.user_data.get("users_search_query") or "").strip()
    results = context.user_data.get("users_search_results")
    if results is None or not isinstance(results, list):
        results = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: search_users_by_query(search_query),
        )
        context.user_data["users_search_results"] = results

    page = _clamp_users_page(results, page)
    if not results:
        await safe_send(lambda: query.message.edit_text(
            "🔎 <b>User Search</b>\n\n"
            f"No users found for: <code>{html.escape(search_query)}</code>\n\n"
            "Press 🔎 New Search or return to all users.",
            parse_mode="HTML",
            reply_markup=get_user_search_prompt_kb(),
        ))
        return

    await safe_send(lambda: query.message.edit_text(
        "🔎 <b>User Search Results</b>\n\n"
        f"Query: <code>{html.escape(search_query)}</code>\n"
        f"Found: <b>{len(results)}</b> user(s)\n\n"
        "Select a user to view details.",
        parse_mode="HTML",
        reply_markup=get_user_search_page_kb(results, page=page),
    ))


async def _admin_open_recent_history_panel(query, page: int = 0) -> None:
    rows = await asyncio.get_running_loop().run_in_executor(
        _DB_EXECUTOR,
        lambda: db_recent_history_users(limit_users=100, scan_limit=700),
    )
    page = _clamp_users_page(rows, page)
    await safe_send(lambda: query.message.edit_text(
        _format_recent_history_panel_text(rows, page=page),
        parse_mode="HTML",
        reply_markup=get_recent_history_kb(rows, page=page),
        disable_web_page_preview=True,
    ))


async def _show_user_full_history(query, user_id: int, back_ref: str = "p0", page: int = 0) -> None:
    rows = await asyncio.get_running_loop().run_in_executor(
        _DB_EXECUTOR,
        lambda: db_user_history_fetch(user_id, limit=ADMIN_FULL_HISTORY_TURNS),
    )
    total_pages = max(1, (len(rows) + ADMIN_HISTORY_PAGE_SIZE - 1) // ADMIN_HISTORY_PAGE_SIZE)
    page = max(0, min(int(page or 0), total_pages - 1))
    await safe_send(lambda: query.message.edit_text(
        _format_user_full_history_text(user_id, rows, page=page, page_size=ADMIN_HISTORY_PAGE_SIZE),
        parse_mode="HTML",
        reply_markup=get_user_history_kb(user_id, back_ref=back_ref, page=page, total_rows=len(rows), page_size=ADMIN_HISTORY_PAGE_SIZE),
        disable_web_page_preview=True,
    ))


async def _handle_user_search_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if context.user_data.get("user_search_state") != USER_SEARCH_WAIT_QUERY:
        return False

    msg = update.message
    query_text = (msg.text or "").strip()
    context.user_data.pop("user_search_state", None)

    if not query_text:
        await safe_send(lambda: msg.reply_text(
            "⚠️ Search text is empty. Press /users and try again.",
            reply_markup=get_user_search_prompt_kb(),
        ))
        return True

    results = await asyncio.get_running_loop().run_in_executor(
        _DB_EXECUTOR,
        lambda: search_users_by_query(query_text),
    )
    context.user_data["users_search_query"] = query_text
    context.user_data["users_search_results"] = results

    if not results:
        await safe_send(lambda: msg.reply_text(
            "🔎 <b>User Search</b>\n\n"
            f"No users found for: <code>{html.escape(query_text)}</code>\n\n"
            "Search supports Telegram user ID and username.",
            parse_mode="HTML",
            reply_markup=get_user_search_prompt_kb(),
        ))
        return True

    await safe_send(lambda: msg.reply_text(
        "🔎 <b>User Search Results</b>\n\n"
        f"Query: <code>{html.escape(query_text)}</code>\n"
        f"Found: <b>{len(results)}</b> user(s)\n\n"
        "Select a user to view details.",
        parse_mode="HTML",
        reply_markup=get_user_search_page_kb(results, page=0),
    ))
    return True


async def users_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    with suppress(Exception):
        await query.answer()

    if data == "users_close":
        context.user_data.pop("user_search_state", None)
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "noop":
        return

    if data in ("history_refresh", "history_page:0"):
        await _admin_open_recent_history_panel(query, page=0)
        return

    if data == "history_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data.startswith("history_page:"):
        page = int(data.split(":", 1)[1])
        await _admin_open_recent_history_panel(query, page=page)
        return

    if data.startswith("history_user:"):
        parts = data.split(":")
        target_id = int(parts[1])
        page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
        await safe_send(lambda: query.message.edit_text(
            _format_user_detail_text(row),
            parse_mode="HTML",
            reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=f"h{page}"),
        ))
        return

    if data == "users_search":
        context.user_data["user_search_state"] = USER_SEARCH_WAIT_QUERY
        await safe_send(lambda: query.message.edit_text(
            "🔎 <b>Search User</b>\n\n"
            "Send a Telegram user ID or username.\n\n"
            "Examples:\n"
            "<code>1272791365</code>\n"
            "<code>heng</code>\n"
            "<code>@username</code>\n\n"
            "Use /cancel to stop search.",
            parse_mode="HTML",
            reply_markup=get_user_search_prompt_kb(),
        ))
        return

    if data.startswith("users_search_page:"):
        page = int(data.split(":", 1)[1])
        await _show_user_search_results(query, context, page=page)
        return

    if data.startswith("users_page:"):
        page  = int(data.split(":")[1])
        users = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, get_all_users_with_names)
        page = _clamp_users_page(users, page)
        await safe_send(lambda: query.message.edit_text(
            f"👥 <b>User Management ({len(users)} users)</b>\n"
            "Select a user, or press 🔎 Search User to find by ID/username.",
            parse_mode="HTML",
            reply_markup=get_users_page_kb(users, page=page),
        ))
        return

    if data.startswith("user_view:"):
        parts = data.split(":")
        target_id = int(parts[1])
        back_ref = parts[2] if len(parts) > 2 else "p0"
        row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
        await safe_send(lambda: query.message.edit_text(
            _format_user_detail_text(row),
            parse_mode="HTML",
            reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
        ))
        return

    if data.startswith("user_history:"):
        parts = data.split(":")
        target_id = int(parts[1])
        back_ref = parts[2] if len(parts) > 2 else "p0"
        page = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
        await _show_user_full_history(query, target_id, back_ref=back_ref, page=page)
        return

    if data.startswith("user_chat:"):
        target_id = int(data.split(":", 1)[1])
        exists = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: user_exists_in_db(target_id))
        if not exists:
            await safe_send(lambda: query.message.edit_text(
                f"❌ User <code>{target_id}</code> មិនមាននៅក្នុង Database ។",
                parse_mode="HTML",
                reply_markup=get_admin_dashboard_kb(),
            ))
            return
        await _open_chat_session(context.bot, user_id, target_id, context)
        await safe_send(lambda: query.message.edit_text(
            f"💬 <b>Chat Mode បើក</b>\n\nកំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
            "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
            "វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
                InlineKeyboardButton("❌ End Chat", callback_data="admin_cancel_state"),
            ]]),
        ))
        return

    if data.startswith(("user_block:", "user_unblock:")):
        parts = data.split(":")
        action = parts[0]
        target_id = int(parts[1])
        back_ref = parts[2] if len(parts) > 2 else "p0"
        blocked = action == "user_block"
        ok, info = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: db_user_set_blocked(target_id, user_id, blocked),
        )
        row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
        notice = "✅ User blocked." if blocked else "✅ User unblocked."
        if not ok:
            notice = f"⚠️ Saved memory only / DB issue: {info[:500]}"
        await safe_send(lambda: query.message.edit_text(
            notice + "\n\n" + _format_user_detail_text(row),
            parse_mode="HTML",
            reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
        ))
        return

    if data.startswith("user_resetprefs:"):
        parts = data.split(":")
        target_id = int(parts[1])
        back_ref = parts[2] if len(parts) > 2 else "p0"
        ok, info = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_reset_prefs(target_id))
        row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
        notice = "✅ User preferences reset." if ok else f"❌ Reset failed: {info[:500]}"
        await safe_send(lambda: query.message.edit_text(
            notice + "\n\n" + _format_user_detail_text(row),
            parse_mode="HTML",
            reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
        ))
        return

    if data.startswith("user_clearhist:"):
        parts = data.split(":")
        target_id = int(parts[1])
        back_ref = parts[2] if len(parts) > 2 else "p0"
        await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_history_clear(target_id))
        row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
        await safe_send(lambda: query.message.edit_text(
            "✅ User conversation history cleared.\n\n" + _format_user_detail_text(row),
            parse_mode="HTML",
            reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
        ))
        return


async def _fwd_admin_to_user(bot, admin_id: int, target_id: int, msg) -> bool:
    async def _do():
        if msg.text:
            await bot.send_message(chat_id=target_id, text=f"📩 <b>Admin:</b>\n{html.escape(msg.text)}", parse_mode="HTML")
        elif msg.photo:
            cap = f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>"
            await bot.send_photo(chat_id=target_id, photo=msg.photo[-1].file_id, caption=cap, parse_mode="HTML")
        elif msg.voice:
            await bot.send_voice(chat_id=target_id, voice=msg.voice.file_id, caption="📩 Admin voice message")
        elif msg.video_note:
            await bot.send_video_note(chat_id=target_id, video_note=msg.video_note.file_id)
        elif msg.sticker:
            await bot.send_sticker(chat_id=target_id, sticker=msg.sticker.file_id)
        elif msg.document:
            cap = f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>"
            await bot.send_document(chat_id=target_id, document=msg.document.file_id, caption=cap, parse_mode="HTML")
        elif msg.video:
            cap = f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>"
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


def _api_help_text() -> str:
    return (
        "🔑 <b>AI API Key Manager</b>\n\n"
        "Commands:\n"
        "• <code>/api create</code> — create a new AI API key\n"
        "• <code>/api create my app</code> — create with a note/name\n"
        "• <code>/api list</code> — show recent keys\n"
        "• <code>/api revoke KEY_PREFIX_OR_ID</code> — disable a key\n"
        "• <code>/api sql</code> — show Supabase table SQL\n\n"
        "Use key with:\n"
        "<code>X-Api-Key: YOUR_KEY</code>\n"
        "or\n"
        "<code>Authorization: Bearer YOUR_KEY</code>\n\n"
        "Endpoint:\n"
        "<code>/ai-assistant</code>"
    )


def _format_api_key_row(row: dict) -> str:
    prefix = html.escape(str(row.get("key_prefix") or "?"))
    row_id = html.escape(str(row.get("id") or "?"))
    note = html.escape(str(row.get("note") or "-"))
    active = "✅ active" if row.get("active") else "🚫 revoked"
    created = html.escape(str(row.get("created_at") or "-")[:19].replace("T", " "))
    last_used = html.escape(str(row.get("last_used_at") or "-")[:19].replace("T", " "))
    return (
        f"#{row_id} <code>{prefix}</code> — <b>{active}</b>\n"
        f"   note: {note}\n"
        f"   created: {created} | last used: {last_used}"
    )


@admin_only
async def cmd_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to create/list/revoke API keys for /ai-assistant."""
    msg = update.message
    admin_id = update.effective_user.id
    args = list(context.args or [])

    if not args or args[0].lower() in ("help", "-h", "--help"):
        await safe_send(lambda: msg.reply_text(
            _api_help_text(),
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
            disable_web_page_preview=True,
        ))
        return

    action = args[0].lower().strip()
    loop = asyncio.get_running_loop()

    if action == "sql":
        pages = _paginate_plain(AI_API_KEYS_TABLE_SQL, limit=3800)
        for page in pages:
            await safe_send(lambda p=page: msg.reply_text(
                f"<pre>{html.escape(p)}</pre>",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
        return

    if action == "create":
        note = " ".join(args[1:]).strip()
        try:
            raw_key, row, storage = await loop.run_in_executor(
                _DB_EXECUTOR,
                lambda: db_ai_api_key_create(admin_id=admin_id, note=note),
            )
        except Exception as e:
            logger.error(f"/api create failed: {e}", exc_info=True)
            err = str(e)
            pages = _paginate_plain(err, limit=3500)
            await safe_send(lambda: msg.reply_text(
                "❌ Cannot create API key.\n"
                "If this is first setup, press <b>🧩 Setup SQL</b> or run <code>/api sql</code> and execute it in Supabase.",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            if pages:
                await safe_send(lambda p=pages[0]: msg.reply_text(
                f"<pre>{html.escape(p)}</pre>",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        warning = ""
        if storage == "memory":
            warning = (
                "\n\n⚠️ Supabase is not configured, so this key is stored in memory only "
                "and will stop working after restart/deploy."
            )

        await safe_send(lambda: msg.reply_text(
            "✅ <b>New AI API key created</b>\n\n"
            "Copy it now. It will not be shown again.\n\n"
            f"<code>{html.escape(raw_key)}</code>\n\n"
            f"Prefix: <code>{html.escape(str(row.get('key_prefix') or _api_key_prefix(raw_key)))}</code>\n"
            f"Storage: <b>{html.escape(storage)}</b>\n\n"
            "Example:\n"
            "<pre>curl -X POST https://YOUR-APP.onrender.com/ai-assistant \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            f"  -H 'X-Api-Key: {html.escape(raw_key)}' \\\n"
            "  -d '{\"message\":\"Hello\"}'</pre>"
            f"{warning}",
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
            disable_web_page_preview=True,
        ))
        return

    if action == "list":
        try:
            rows = await loop.run_in_executor(_DB_EXECUTOR, lambda: db_ai_api_key_list(limit=20))
        except Exception as e:
            logger.error(f"/api list failed: {e}", exc_info=True)
            await safe_send(lambda: msg.reply_text(
                f"❌ Cannot list API keys.\n<pre>{html.escape(str(e)[:3500])}</pre>",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        if not rows:
            await safe_send(lambda: msg.reply_text(
                "ℹ️ No API keys found. Press <b>➕ Create API Key</b> or use <code>/api create</code>.",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        body = "\n\n".join(_format_api_key_row(r) for r in rows)
        for page in _paginate_plain("🔑 <b>AI API Keys</b>\n\n" + body, limit=3900):
            await safe_send(lambda p=page: msg.reply_text(
                p,
                parse_mode="HTML",
                reply_markup=get_api_list_kb(rows),
                disable_web_page_preview=True,
            ))
        return

    if action == "revoke":
        if len(args) < 2:
            await safe_send(lambda: msg.reply_text(
                "⚠️ Usage: <code>/api revoke KEY_PREFIX_OR_ID</code>\n\n"
                "Or press <b>📋 List Keys</b> and revoke with buttons.",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        identifier = args[1].strip()
        try:
            ok, info = await loop.run_in_executor(
                _DB_EXECUTOR,
                lambda: db_ai_api_key_revoke(identifier),
            )
        except Exception as e:
            logger.error(f"/api revoke failed: {e}", exc_info=True)
            await safe_send(lambda: msg.reply_text(
                f"❌ Cannot revoke API key.\n<pre>{html.escape(str(e)[:3500])}</pre>",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        if ok:
            await safe_send(lambda: msg.reply_text(
                f"✅ API key revoked: <code>{html.escape(info)}</code>",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
        else:
            await safe_send(lambda: msg.reply_text(
                f"❌ {html.escape(info)}",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
        return

    await safe_send(lambda: msg.reply_text(
        _api_help_text(),
        parse_mode="HTML",
        reply_markup=get_api_admin_kb(),
        disable_web_page_preview=True,
    ))

def _api_status_text(status: dict, notice: str = "") -> str:
    static_key = "✅ ON" if status.get("static_key") else "⚠️ OFF"
    supabase_ok = "✅ OK" if status.get("supabase") else "⚠️ OFF"
    service_role = "✅ SET" if status.get("service_role_key") else "⚠️ MISSING"
    table_ok = "✅ READY" if status.get("table_ok") else "❌ NOT READY"
    dynamic_ok = "✅ ON" if (status.get("active_count", 0) > 0 or status.get("memory_count", 0) > 0) else "⚠️ NO ACTIVE KEY"

    lines = []
    if notice:
        lines.append(f"{html.escape(notice)}\n")
    lines.extend([
        "🔑 <b>AI API Key Admin</b>",
        "",
        f"Static <code>AI_API_KEY</code>: <b>{static_key}</b>",
        f"Dynamic generated keys: <b>{dynamic_ok}</b>",
        f"Supabase: <b>{supabase_ok}</b>",
        f"Service role key: <b>{service_role}</b>",
        f"Table <code>ai_api_keys</code>: <b>{table_ok}</b>",
        f"Active keys: <b>{int(status.get('active_count') or 0)}</b>",
    ])
    if status.get("memory_count"):
        lines.append(f"Memory keys: <b>{int(status.get('memory_count') or 0)}</b> ⚠️ not persistent")
    if status.get("error"):
        lines.extend([
            "",
            "⚠️ <b>Setup error</b>",
            f"<pre>{html.escape(str(status.get('error'))[:900])}</pre>",
            "Press <b>🧩 Setup SQL</b>, run it in Supabase, then restart/redeploy bot with <code>SUPABASE_SERVICE_ROLE_KEY</code>.",
        ])
    else:
        lines.extend([
            "",
            "Use buttons below to create/list/revoke keys.",
        ])
    return "\n".join(lines)


async def _api_status_async() -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_DB_EXECUTOR, db_ai_api_key_status)


async def _api_send_sql_message(message) -> None:
    pages = _paginate_plain(AI_API_KEYS_TABLE_SQL, limit=3800)
    for page in pages:
        await safe_send(lambda p=page: message.reply_text(
            "🧩 <b>Supabase SQL for API keys</b>\n\n"
            f"<pre>{html.escape(p)}</pre>",
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
        ))


async def _api_edit_menu(query, notice: str = "") -> None:
    status = await _api_status_async()
    await safe_send(lambda: query.message.edit_text(
        _api_status_text(status, notice=notice),
        parse_mode="HTML",
        reply_markup=get_api_admin_kb(),
        disable_web_page_preview=True,
    ))


async def _api_create_from_button(query, user_id: int) -> None:
    loop = asyncio.get_running_loop()
    note = f"telegram-button admin {user_id}"
    try:
        raw_key, row, storage = await loop.run_in_executor(
            _DB_EXECUTOR,
            lambda: db_ai_api_key_create(admin_id=user_id, note=note),
        )
    except Exception as e:
        logger.error("api_create button failed: %s", e, exc_info=True)
        err = str(e)
        short_err = err[:1800]
        await safe_send(lambda: query.message.reply_text(
            "❌ <b>Cannot create API key</b>\n\n"
            "Usually this means the <code>ai_api_keys</code> table is not created yet, "
            "or <code>SUPABASE_SERVICE_ROLE_KEY</code> is missing on the server.\n\n"
            "Press <b>🧩 Setup SQL</b>, run it in Supabase, then redeploy/restart.\n\n"
            f"<pre>{html.escape(short_err)}</pre>",
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
        ))
        return

    warning = ""
    if storage == "memory":
        warning = (
            "\n\n⚠️ Supabase is not configured. This key is stored in memory only "
            "and will stop working after restart/deploy."
        )

    await safe_send(lambda: query.message.reply_text(
        "✅ <b>New AI API key created</b>\n\n"
        "Copy it now. It will not be shown again.\n\n"
        f"<code>{html.escape(raw_key)}</code>\n\n"
        f"Prefix: <code>{html.escape(str(row.get('key_prefix') or _api_key_prefix(raw_key)))}</code>\n"
        f"Storage: <b>{html.escape(storage)}</b>\n\n"
        "Use it like this:\n"
        "<pre>curl -X POST https://YOUR-APP.onrender.com/ai-assistant \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -H 'X-Api-Key: {html.escape(raw_key)}' \\\n"
        "  -d '{\"message\":\"Hello\"}'</pre>"
        f"{warning}",
        parse_mode="HTML",
        reply_markup=get_api_admin_kb(),
        disable_web_page_preview=True,
    ))

    with suppress(Exception):
        await _api_edit_menu(query, notice="✅ API key created. Copy it from the new message above.")


async def _api_list_from_button(query) -> None:
    loop = asyncio.get_running_loop()
    try:
        rows = await loop.run_in_executor(_DB_EXECUTOR, lambda: db_ai_api_key_list(limit=20))
    except Exception as e:
        logger.error("api_list button failed: %s", e, exc_info=True)
        await safe_send(lambda: query.message.edit_text(
            "❌ <b>Cannot list API keys</b>\n\n"
            "Run setup SQL first, then ensure <code>SUPABASE_SERVICE_ROLE_KEY</code> is set.\n\n"
            f"<pre>{html.escape(str(e)[:1800])}</pre>",
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
        ))
        return

    if not rows:
        await safe_send(lambda: query.message.edit_text(
            "📋 <b>AI API Keys</b>\n\n"
            "No API keys found yet. Press <b>➕ Create API Key</b> after setup is ready.",
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
        ))
        return

    body = "\n\n".join(_format_api_key_row(r) for r in rows)
    await safe_send(lambda: query.message.edit_text(
        "📋 <b>AI API Keys</b>\n\n" + body,
        parse_mode="HTML",
        reply_markup=get_api_list_kb(rows),
        disable_web_page_preview=True,
    ))


async def _cb_api_dashboard(query, user_id: int, context: ContextTypes.DEFAULT_TYPE, data: str):
    if not _is_admin(user_id):
        await safe_send(lambda: query.message.reply_text("⛔ Admin only."))
        return

    if data in ("api_menu", "admin_api"):
        await _api_edit_menu(query)
        return

    if data == "api_back":
        text = await _admin_home_text(user_id)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return

    if data == "api_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "api_help":
        await safe_send(lambda: query.message.edit_text(
            _api_help_text(),
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
            disable_web_page_preview=True,
        ))
        return

    if data == "api_status":
        await _api_edit_menu(query)
        return

    if data == "api_sql":
        await _api_send_sql_message(query.message)
        with suppress(Exception):
            await _api_edit_menu(query, notice="🧩 SQL sent below. Run it in Supabase SQL editor.")
        return

    if data == "api_create":
        await _api_create_from_button(query, user_id)
        return

    if data == "api_list":
        await _api_list_from_button(query)
        return

    if data.startswith("api_revoke:"):
        identifier = data.split(":", 1)[1].strip()
        await safe_send(lambda: query.message.edit_text(
            "⚠️ <b>Confirm revoke API key</b>\n\n"
            f"Key ID: <code>{html.escape(identifier)}</code>\n\n"
            "After revoke, apps using this key cannot access <code>/ai-assistant</code>.",
            parse_mode="HTML",
            reply_markup=get_api_revoke_confirm_kb(identifier),
        ))
        return

    if data.startswith("api_revoke_confirm:"):
        identifier = data.split(":", 1)[1].strip()
        loop = asyncio.get_running_loop()
        try:
            ok, info = await loop.run_in_executor(
                _DB_EXECUTOR,
                lambda: db_ai_api_key_revoke(identifier),
            )
        except Exception as e:
            logger.error("api_revoke button failed: %s", e, exc_info=True)
            await safe_send(lambda: query.message.edit_text(
                "❌ <b>Cannot revoke API key</b>\n\n"
                f"<pre>{html.escape(str(e)[:1800])}</pre>",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        if ok:
            await _api_list_from_button(query)
            await safe_send(lambda: query.message.reply_text(
                f"✅ API key revoked: <code>{html.escape(info)}</code>",
                parse_mode="HTML",
            ))
        else:
            await safe_send(lambda: query.message.edit_text(
                f"❌ {html.escape(info)}",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
        return

    await _api_edit_menu(query)


@admin_only
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loop   = asyncio.get_running_loop()
    user_ids, pending_scheds = await asyncio.gather(
        loop.run_in_executor(None, get_all_user_ids),
        loop.run_in_executor(None, db_sched_fetch_admin_pending, update.effective_user.id),
    )
    await safe_send(lambda: update.message.reply_text(
        f"📊 <b>Bot Statistics</b>\n\n"
        f"👥 អ្នកប្រើប្រាស់សរុប: <b>{len(user_ids)}</b>\n"
        f"💬 Active Admin Chats: <b>{len(_admin_chat_target)}</b>\n"
        f"📅 Scheduled (pending): <b>{len(pending_scheds)}</b>\n"
        f"🔒 Active user locks: <b>{len(_user_locks)}</b>\n"
        f"💭 History cache entries: <b>{len(_hist_cache)}</b>\n"
        f"🔑 Dynamic API auth: <b>{'ON' if _dynamic_ai_auth_configured() else 'OFF'}</b>\n"
        f"🤗 HF Model: <b>{HF_MODEL}</b>\nOCR: <b>{HF_OCR_MODEL}</b>",
        parse_mode="HTML",
    ))


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid     = update.effective_user.id
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
            await safe_send(lambda: update.message.reply_text(
                f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។", parse_mode="HTML"
            ))
            with suppress(Exception):
                await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")
        cleared = True

    if context.user_data.get("sched_state") in (SCHED_WAIT_MSG, SCHED_WAIT_TIME, SCHED_EDIT_WAIT_TIME, SCHED_EDIT_WAIT_TEXT, SCHED_EDIT_WAIT_PHOTO):
        _sched_payload.pop(uid, None)
        context.user_data.pop("sched_state", None)
        context.user_data.pop("sched_edit_row_id", None)
        await safe_send(lambda: update.message.reply_text("❌ Schedule/Edit flow បានបោះបង់។"))
        cleared = True

    if context.user_data.get("user_search_state") == USER_SEARCH_WAIT_QUERY:
        context.user_data.pop("user_search_state", None)
        await safe_send(lambda: update.message.reply_text("❌ User search cancelled."))
        cleared = True

    if not cleared:
        await safe_send(lambda: update.message.reply_text("ℹ️ មិនមាន operation ត្រូវ cancel ទេ។"))


# ===========================================================================
# REGULAR HANDLERS
# ===========================================================================
_BOT_START_TIME: float = 0.0


async def _drop_stale_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Drop old message updates received before bot started.

    FIX: For callback_query updates, we intentionally do NOT drop them based on
    message date — the message date is when the original message was sent, not
    when the user tapped the button. Dropping callbacks based on message age
    would break buttons on messages sent before the bot restarted. We only drop
    stale *message* updates.
    """
    if _BOT_START_TIME == 0.0:
        return

    # Only filter plain messages (not callbacks)
    msg = update.message or update.edited_message
    if msg and getattr(msg, "date", None):
        if msg.date.timestamp() < (_BOT_START_TIME - _STALE_GRACE_S):
            logger.debug(f"Dropping stale message update (id={update.update_id})")
            raise ApplicationHandlerStop


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sync_user_data(update.effective_user)
        if not await _ensure_user_allowed(update, context):
            return
        await safe_send(lambda: update.message.reply_text(
            WELCOME_TEXT,
            reply_markup=ReplyKeyboardRemove(),
            disable_web_page_preview=True,
        ))
    except Exception as e:
        logger.error(f"on_start error: {e}")


async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await on_start(update, context)


async def cmd_myprefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prefs   = await get_user_prefs_async(user_id)
    gender_label = "👩 ស្រី (Female)" if prefs["gender"] == "female" else "👨 ប្រុស (Male)"
    speed_label  = next(
        (lbl for _, (lbl, val) in SPEED_OPTIONS.items() if abs(val - prefs["speed"]) < 0.01),
        f"{prefs['speed']}x",
    )
    await safe_send(lambda: update.message.reply_text(
        f"⚙️ <b>ការកំណត់របស់អ្នក</b>\n\n"
        f"🗣️ សំឡេង: <b>{gender_label}</b>\n"
        f"🎚️ ល្បឿន: <b>{speed_label}</b>\n\n"
        "ផ្ញើ text ណាមួយ ហើយប្រើប៊ូតុងក្រោមសំឡេង ដើម្បីប្តូរ។",
        parse_mode="HTML",
        reply_markup=get_main_kb(prefs["gender"]),
    ))


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _hist_cache_clear(user_id)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_DB_EXECUTOR, db_history_clear, user_id)
    await safe_send(lambda: update.message.reply_text(
        "🗑️ ប្រវត្តិការសន្ទនារបស់អ្នកបានលុបចេញហើយ។\nBot នឹងចាប់ផ្ដើមការសន្ទនាថ្មី។"
    ))


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg     = update.message
    user    = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    # Admin flow intercepts
    if _is_admin(user_id):
        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_EDIT_WAIT_PHOTO:
            await _handle_sched_edit_photo(update, context)
            return
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok    = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                reply = (
                    f"✅ Photo ផ្ញើដល់ User <code>{target_id}</code> ។"
                    if ok else
                    f"❌ User <code>{target_id}</code> blocked bot ។"
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

    if not await _ensure_user_allowed(update, context, "ocr_enabled", "Image OCR"):
        return

    if not _ocr_configured():
        await safe_send(lambda: msg.reply_text(_ocr_status_for_user()))
        return

    if await _check_cooldown(msg, user_id):
        return

    _metric_inc("ocr")
    sync_user_data(user)
    uname       = user.username or user.first_name or str(user_id)
    stop_event  = asyncio.Event()
    timer_task  = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_OCR_FRAMES)
    )
    img_path    = None
    try:
        img_path  = _make_temp_img(suffix=".jpg")
        tg_file   = await safe_send(lambda: context.bot.get_file(msg.photo[-1].file_id))
        if not tg_file:
            raise RuntimeError("Could not download photo.")
        await tg_file.download_to_drive(img_path)
        mime_type = _detect_image_mime(img_path)
        ocr_text  = await ocr_image(img_path, mime_type=mime_type)

        if not ocr_text or ocr_text.upper() == "NOTEXT":
            await safe_send(lambda: msg.reply_text("🖼️ រូបភាពនេះមិនមានអត្ថបទដែលអាចអានបាន។"))
            return

        record_turn(user_id, "user", f"[Image OCR]: {ocr_text[:500]}")

        is_khmer   = bool(_KHMER_RE.search(ocr_text))
        lang_flag  = "🇰🇭" if is_khmer else "🇺🇸"
        header     = f"🔍 <b>OCR {lang_flag}</b>\n\n"
        plain_pages = _paginate_plain(ocr_text, limit=TELE_MSG_LIMIT - len(header))
        sent_pages  = []
        for idx, plain_page in enumerate(plain_pages):
            page_body = (header if idx == 0 else "") + html.escape(plain_page)
            sent      = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, ocr_text,
                chat_id=msg.chat_id, user_id=user_id, username=uname,
            )
            await safe_send(lambda: last_sent.edit_reply_markup(
                reply_markup=get_ocr_confirm_kb(last_sent.message_id)
            ))

    except Exception as e:
        err_msg  = str(e) or repr(e)
        logger.error(f"on_photo OCR error: {type(e).__name__}: {e!r}", exc_info=True)
        if _is_dns_or_network_error(err_msg):
            user_msg = (
                "❌ OCR network/DNS មានបញ្ហា។ Server មិនអាចភ្ជាប់ទៅ Provider OCR បាន។\n"
                "✅ Code នេះនឹងសាក fallback provider ដោយស្វ័យប្រវត្តិ បើបាន Set OCR_PROVIDER=auto និង GEMINI_API_KEY/GEMINI_MODEL។"
            )
        elif "model/provider is not available" in err_msg:
            user_msg = (
                "❌ Hugging Face OCR model មិន support hosted inference ទេ។\n"
                "សូមសាក HF_OCR_MODEL=microsoft/trocr-base-printed ជាមុនសិន។"
            )
        else:
            user_msg = f"❌ មានបញ្ហាក្នុងការ OCR រូបភាព។\n{html.escape(err_msg[:1000])}"
        await safe_send(lambda: msg.reply_text(user_msg))
    finally:
        await _stop_timer(stop_event, timer_task)
        if img_path:
            _cleanup(img_path)


async def on_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg     = update.message
    user    = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    # Admin chat-session forward
    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok    = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok else
                f"❌ User <code>{target_id}</code> blocked bot ។"
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
        await safe_send(lambda: msg.reply_text("✅ ឯកសារបានផ្ញើដល់ Admin ។"))
        return

    doc   = msg.document
    audio = msg.audio

    if doc is not None:
        filename  = doc.file_name or ""
        mime_type = doc.mime_type or ""
        file_id   = doc.file_id
        file_size = doc.file_size or 0
        # FIX: Check subtitle BEFORE audio — subtitle files (.txt/.srt/.vtt) must
        # be handled by on_document, not the audio transcriber.
        if _is_subtitle_file(filename):
            await on_document(update, context)
            return
        if not _is_audio_file(filename, mime_type):
            await on_document(update, context)
            return
    elif audio is not None:
        filename  = audio.file_name or ""
        mime_type = audio.mime_type or ""
        file_id   = audio.file_id
        file_size = audio.file_size or 0
    else:
        return

    if not await _ensure_user_allowed(update, context, "audio_transcribe_enabled", "Audio file transcribe"):
        return

    if not _gemini:
        await safe_send(lambda: msg.reply_text(
            "❌ Gemini API មិន Activate ទេ។ Transcribe ត្រូវការ GEMINI_API_KEY ។"
        ))
        return

    if file_size > MAX_AUDIO_FILE_BYTES:
        await safe_send(lambda: msg.reply_text(
            f"❌ ឯកសារអូឌីយ៉ូធំពេក (អតិបរមា {MAX_AUDIO_FILE_BYTES // 1024 // 1024}MB)។"
        ))
        return

    if await _check_cooldown(msg, user_id):
        return

    _metric_inc("audio")
    sync_user_data(user)
    uname       = user.username or user.first_name or str(user_id)
    ext         = os.path.splitext(filename)[1].lower() if filename else ".mp3"
    if ext not in _AUDIO_EXTENSIONS:
        ext = ".mp3"
    gemini_mime = _audio_mime_for_gemini(filename, mime_type)
    audio_path  = _make_temp_audio(suffix=ext)

    stop_event  = asyncio.Event()
    timer_task  = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_AUDIO_FILE_FRAMES)
    )
    try:
        tg_file = await safe_send(lambda: context.bot.get_file(file_id))
        if not tg_file:
            raise RuntimeError("Could not download audio file.")
        await tg_file.download_to_drive(audio_path)
        transcript = await transcribe_audio_file(audio_path, gemini_mime)

        if not transcript:
            await safe_send(lambda: msg.reply_text(
                "❌ រក Transcript មិនឃើញ — ឯកសារប្រហែលជាស្ងាត់ ឬ មិន Support ។"
            ))
            return

        record_turn(user_id, "user", f"[Audio File Transcript]: {transcript[:500]}")
        is_khmer   = bool(_KHMER_RE.search(transcript))
        lang_flag  = "🇰🇭" if is_khmer else "🇺🇸"
        fname_display = html.escape(filename[:50]) if filename else "audio"
        header     = f"🎵 <b>Transcript</b> {lang_flag} — <code>{fname_display}</code>\n\n"
        plain_pages = _paginate_plain(transcript, limit=TELE_MSG_LIMIT - len(header))
        sent_pages  = []
        for idx, plain_page in enumerate(plain_pages):
            page_body = (header if idx == 0 else "") + html.escape(plain_page)
            sent      = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, transcript,
                chat_id=msg.chat_id, user_id=user_id, username=uname,
            )
            await safe_send(lambda: last_sent.edit_reply_markup(
                reply_markup=get_audio_file_kb(last_sent.message_id)
            ))

    except Exception as e:
        logger.error(f"on_audio_file error: {e}", exc_info=True)
        await safe_send(lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ Transcribe ឯកសារអូឌីយ៉ូ។"))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(audio_path)


async def on_any_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg     = update.message
    user    = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok    = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok else
                f"❌ User <code>{target_id}</code> blocked bot ។"
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
    msg     = update.message
    user    = update.effective_user
    user_id = user.id

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok    = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ Voice ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok else
                f"❌ User <code>{target_id}</code> blocked bot ។"
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

    if not await _ensure_user_allowed(update, context, "voice_transcribe_enabled", "Voice transcribe"):
        return

    if not _gemini:
        await safe_send(lambda: msg.reply_text("❌ Gemini API មិន Activate ទេ។ សូម Set GEMINI_API_KEY ។"))
        return

    if msg.voice.file_size and msg.voice.file_size > MAX_VOICE_BYTES:
        await safe_send(lambda: msg.reply_text("❌ ឯកសារសំឡេងធំពេក (អតិបរមា 20MB)។"))
        return

    if await _check_cooldown(msg, user_id):
        return

    _metric_inc("voice")
    sync_user_data(user)
    ogg_path   = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_TRANSCRIBE_FRAMES)
    )
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
        is_khmer    = bool(_KHMER_RE.search(transcript))
        lang_flag   = "🇰🇭" if is_khmer else "🇺🇸"
        header      = f"📝 <b>Transcript</b> {lang_flag}\n\n"
        plain_pages = _paginate_plain(transcript, limit=TELE_MSG_LIMIT - len(header))
        sent_pages  = []
        for idx, plain_page in enumerate(plain_pages):
            page_body = (header if idx == 0 else "") + html.escape(plain_page)
            sent      = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, transcript,
                chat_id=msg.chat_id, user_id=user_id,
                username=user.username or user.first_name,
            )
            await safe_send(lambda: last_sent.edit_reply_markup(
                reply_markup=get_transcription_kb(last_sent.message_id)
            ))
    except Exception as e:
        logger.error(f"on_voice error: {e}")
        with suppress(Exception):
            await safe_send(lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ Transcribe។"))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(ogg_path)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg  = update.message
    if not msg or not msg.text:
        return

    text = msg.text
    user = update.effective_user
    if not user:
        return

    user_id = user.id

    # Admin flow intercepts
    if _is_admin(user_id):
        if await _handle_user_search_text(update, context):
            return

        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        if sched_state == SCHED_WAIT_TIME:
            await _handle_sched_datetime(update, context)
            return
        if sched_state == SCHED_EDIT_WAIT_TIME:
            await _handle_sched_edit_time(update, context)
            return
        if sched_state == SCHED_EDIT_WAIT_TEXT:
            await _handle_sched_edit_text(update, context)
            return
        if sched_state == SCHED_EDIT_WAIT_PHOTO:
            await safe_send(lambda: msg.reply_text("⚠️ សូមផ្ញើរូបភាពថ្មី ឬ /cancel។"))
            return
        if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                if ok:
                    await safe_send(lambda: msg.reply_text(
                        f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។", parse_mode="HTML"
                    ))
                else:
                    await safe_send(lambda: msg.reply_text(
                        f"❌ User <code>{target_id}</code> blocked bot ។ Chat session បានបិទ។",
                        parse_mode="HTML",
                    ))
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ សាររបស់អ្នកបានផ្ញើដល់ Admin ។"))
        return

    if not await _ensure_user_allowed(update, context, "tts_enabled", "Text to Voice"):
        return

    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    stripped = text.strip()
    if not stripped:
        return

    if len(stripped) > MAX_INPUT_CHARS:
        await safe_send(lambda: msg.reply_text(
            f"❌ អត្ថបទវែងពេក។ អតិបរមា {MAX_INPUT_CHARS} តួអក្សរ។\n"
            f"(អ្នកបានផ្ញើ {len(stripped)} តួ)"
        ))
        return

    if await _check_cooldown(msg, user_id):
        return

    _metric_inc("tts")
    sync_user_data(user)
    loop = asyncio.get_running_loop()
    prefs, tts_text = await asyncio.gather(
        get_user_prefs_async(user_id),
        resolve_tts_text(user_id, stripped, loop),
    )

    gender   = prefs["gender"]
    speed    = prefs["speed"]
    tts_text = tts_text.strip() or stripped

    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(update.effective_chat.id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)

    async with lock:
        try:
            audio_bytes = await generate_voice_limited(tts_text, gender, speed, file_path)
            sent_msg    = await safe_send(
                lambda ab=audio_bytes: msg.reply_voice(
                    voice=io.BytesIO(ab),
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
                set_last_tts_text(user_id, tts_text)
                record_turn(user_id, "user",      stripped)
                record_turn(user_id, "assistant", tts_text)
            _set_last_tts(user_id)

        except Exception as e:
            logger.error(f"on_text TTS error: {e}", exc_info=True)
            with suppress(Exception):
                await safe_send(lambda err=e: msg.reply_text(_tts_user_error_message(err)))

        finally:
            await _stop_timer(stop_event, timer_task)
            _cleanup(file_path)


# ---------------------------------------------------------------------------
# Callback helpers
# ---------------------------------------------------------------------------
async def _cb_show_speed(query, user_id: int, context):
    prefs = await get_user_prefs_async(user_id)
    await safe_send(lambda: query.message.edit_reply_markup(
        reply_markup=get_speed_kb(prefs["speed"])
    ))


async def _cb_hide_speed(query, user_id: int, context):
    prefs = await get_user_prefs_async(user_id)
    await safe_send(lambda: query.message.edit_reply_markup(
        reply_markup=get_main_kb(prefs["gender"])
    ))


async def get_callback_original_text(query, user_id: int) -> str | None:
    """Resolve the TTS text for a callback button (3-tier fallback)."""
    if query.message is None:
        return get_last_tts_text(user_id)

    chat_id = query.message.chat.id
    msg_id  = query.message.message_id
    loop    = asyncio.get_running_loop()

    # 1. Exact text_cache
    try:
        original_text = await loop.run_in_executor(None, get_text_cache, msg_id, chat_id)
        if original_text:
            return original_text
    except Exception as e:
        logger.warning(f"get_callback_original_text text_cache failed user={user_id}: {e}")

    # 2. Latest generated TTS from memory
    original_text = get_last_tts_text(user_id)
    if original_text:
        return original_text

    # 3. Latest assistant turn from conversation_history
    try:
        history = _hist_cache_get(user_id)
        if history is None:
            history = await loop.run_in_executor(_DB_EXECUTOR, db_history_fetch, user_id)
            for row in history or []:
                _hist_cache_append(user_id, row.get("role", "user"), row.get("content", ""))

        for row in reversed(history or []):
            role    = _normalize_role(row.get("role", ""))
            content = str(row.get("content", "") or "").strip()
            if role == "assistant" and content:
                set_last_tts_text(user_id, content)
                return content
    except Exception as e:
        logger.error(f"get_callback_original_text history fallback failed user={user_id}: {e}")

    return None


async def _cb_speed(query, user_id: int, context, data: str):
    _, new_speed = SPEED_OPTIONS[data]
    if query.message is None:
        return

    chat_id = query.message.chat.id
    original_text, prefs = await asyncio.gather(
        get_callback_original_text(query, user_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text(
            "❌ រកអត្ថបទដើមមិនឃើញ។\nសូមផ្ញើអត្ថបទម្តងទៀត រួចប្តូរល្បឿន។"
        ))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    update_user_speed(user_id, new_speed)

    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, context.bot, stop_event))
    lock       = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, gender, new_speed, file_path)
                with suppress(Exception):
                    await query.message.delete()
                # FIX: was query.message.chat.send_voice() which is not a valid method.
                # Chat object has no send_voice. Must use bot.send_voice(chat_id=...).
                new_msg = await safe_send(
                    lambda ab=audio_bytes, g=gender: context.bot.send_voice(
                        chat_id=chat_id,
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(g),
                    )
                )
                if new_msg:
                    save_text_cache(
                        new_msg.message_id, original_text,
                        chat_id=chat_id, user_id=user_id,
                        username=query.from_user.username or query.from_user.first_name,
                    )
                    set_last_tts_text(user_id, original_text)
                    record_turn(user_id, "assistant", original_text)
                _set_last_tts(user_id)
            except Exception as e:
                logger.error(f"speed regen error: {e}", exc_info=True)
                with suppress(Exception):
                    await safe_send(lambda: context.bot.send_message(
                        chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"
                    ))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(file_path)


async def _cb_gender(query, user_id: int, context, data: str):
    new_gender = data.replace("tg_", "")
    if query.message is None:
        return

    chat_id = query.message.chat.id
    original_text, prefs = await asyncio.gather(
        get_callback_original_text(query, user_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text(
            "❌ រកអត្ថបទដើមមិនឃើញ។\nសូមផ្ញើអត្ថបទម្តងទៀត រួចប្តូរសំឡេង។"
        ))
        return

    if await _check_cooldown(query.message, user_id):
        return

    speed = prefs["speed"]
    update_user_gender(user_id, new_gender)

    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, context.bot, stop_event))
    lock       = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, new_gender, speed, file_path)
                with suppress(Exception):
                    await query.message.delete()
                # FIX: was query.message.chat.send_voice() — Chat object has no send_voice.
                # Must use bot.send_voice(chat_id=...).
                new_msg = await safe_send(
                    lambda ab=audio_bytes, ng=new_gender: context.bot.send_voice(
                        chat_id=chat_id,
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(ng),
                    )
                )
                if new_msg:
                    save_text_cache(
                        new_msg.message_id, original_text,
                        chat_id=chat_id, user_id=user_id,
                        username=query.from_user.username or query.from_user.first_name,
                    )
                    set_last_tts_text(user_id, original_text)
                    record_turn(user_id, "assistant", original_text)
                _set_last_tts(user_id)
            except Exception as e:
                logger.error(f"gender regen error: {e}", exc_info=True)
                with suppress(Exception):
                    await safe_send(lambda: context.bot.send_message(
                        chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"
                    ))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(file_path)


async def _cb_tts_transcript(query, user_id: int, context, data: str):
    transcript_msg_id = int(data.split(":")[1])
    if query.message is None:
        return

    chat_id = query.message.chat.id
    loop    = asyncio.get_running_loop()
    original_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, transcript_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender     = prefs["gender"]
    speed      = prefs["speed"]
    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, context.bot, stop_event))
    lock       = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, gender, speed, file_path)
                # FIX: was query.message.chat.send_voice() — Chat object has no send_voice.
                new_msg = await safe_send(
                    lambda ab=audio_bytes, g=gender: context.bot.send_voice(
                        chat_id=chat_id,
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(g),
                    )
                )
                if new_msg:
                    save_text_cache(
                        new_msg.message_id, original_text,
                        chat_id=chat_id, user_id=user_id,
                        username=query.from_user.username or query.from_user.first_name,
                    )
                    set_last_tts_text(user_id, original_text)
                    record_turn(user_id, "assistant", original_text)
                _set_last_tts(user_id)
            except Exception as e:
                logger.error(f"transcript TTS error: {e}", exc_info=True)
                with suppress(Exception):
                    await safe_send(lambda: context.bot.send_message(
                        chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"
                    ))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(file_path)


async def _cb_audio_tts(query, user_id: int, context, data: str):
    src_msg_id = int(data.split(":")[1])
    chat_id    = query.message.chat.id
    loop       = asyncio.get_running_loop()

    full_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, src_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )
    if not full_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ (cache expired)។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender    = prefs["gender"]
    speed     = prefs["speed"]
    uname     = query.from_user.username or query.from_user.first_name or str(user_id)

    # FIX: Only remove the button markup AFTER TTS succeeds. If we remove it
    # before and TTS fails, the user has no way to retry. We suppress errors
    # so a stale markup doesn't block delivery.
    tts_stop  = asyncio.Event()
    tts_timer = asyncio.create_task(send_status_timer(chat_id, context.bot, tts_stop))
    lock      = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                await _deliver_paged_tts(
                    chat_id=chat_id, bot=context.bot,
                    text=full_text, gender=gender, speed=speed,
                    user_id=user_id, username=uname,
                )
                record_turn(user_id, "assistant", full_text[:CONV_CONTEXT_MAX_CHARS])
                # Remove markup only after successful delivery
                with suppress(Exception):
                    await query.message.edit_reply_markup(reply_markup=None)
            except Exception as e:
                logger.error(f"_cb_audio_tts delivery error: {e}")
                with suppress(Exception):
                    await context.bot.send_message(chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។")
    finally:
        await _stop_timer(tts_stop, tts_timer)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data

    if query.message is None:
        logger.debug(f"on_callback: no message for data={data!r}")
        # FIX: Still answer the query to prevent Telegram showing a loading spinner
        with suppress(Exception):
            await query.answer()
        return

    # FIX: These patterns are handled by dedicated CallbackQueryHandlers that
    # already call query.answer(). Do NOT answer here — it would cause a
    # "query is too old" double-answer error on Telegram's side.
    _HANDLED_EXACT = {"bc_confirm", "bc_cancel", "users_close", "noop"}
    _HANDLED_PREFIX = ("sched_", "users_page:", "users_search", "user_", "history_")
    if data in _HANDLED_EXACT or any(data.startswith(p) for p in _HANDLED_PREFIX):
        return

    # Answer all other callbacks here (only once)
    with suppress(Exception):
        await query.answer()

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
        elif data.startswith(("del_transcript:", "doc_del:", "audio_del:")):
            with suppress(Exception):
                await query.message.delete()
        elif data.startswith("doc_read:"):
            await _cb_doc_read(query, user_id, context, data)
        elif data.startswith("audio_tts:"):
            await _cb_audio_tts(query, user_id, context, data)
        elif data.startswith("api_"):
            await _cb_api_dashboard(query, user_id, context, data)
        elif data.startswith("admin_"):
            await _cb_admin_dashboard(query, user_id, context, data)
        else:
            logger.debug(f"on_callback: unhandled data={data!r}")
    except Exception as e:
        logger.error(f"on_callback unhandled [data={data}]: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    _metric_inc("errors")
    logger.error(f"Unhandled exception: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        with suppress(Exception):
            await safe_send(lambda: update.effective_message.reply_text(
                "⚠️ មានបញ្ហាបច្ចេកទេស។ Bot នៅដំណើរការ — សូមព្យាយាមម្តងទៀត។"
            ))


# ---------------------------------------------------------------------------
# Bot runner
# ---------------------------------------------------------------------------
async def _run_bot():
    global _BOT_START_TIME, _AI_SEMAPHORE, _BROADCAST_SEMAPHORE
    global _prefs_cache_lock, _TTS_CHUNK_SEMAPHORE

    _AI_SEMAPHORE        = asyncio.Semaphore(MAX_CONCURRENT_AI)
    _BROADCAST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_BROADCAST)
    _prefs_cache_lock    = asyncio.Lock()
    _TTS_CHUNK_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_TTS_USERS)

    # FIX: Reset ALL mutable in-memory state on restart to prevent stale data
    # accumulation across crash-restart cycles.
    for store in (
        _admin_chat_target, _user_to_admin, _pending_broadcast,
        _prefs_cache, _user_last_tts, _sched_payload,
        _hist_cache, _user_locks,
        # FIX: These two were missing from the original reset list:
        _last_tts_text, _text_cache_memory,
    ):
        store.clear()
    _api_key_cache_clear()
    _api_key_touch_last.clear()
    if not supabase:
        _api_keys_memory_by_hash.clear()
    _scheduler_tasks.clear()
    _scheduler_active_ids.clear()

    _BOT_START_TIME = time.time()

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    app.add_handler(TypeHandler(Update, _drop_stale_updates), group=-1)

    # Commands
    app.add_handler(CommandHandler("start",           on_start))
    app.add_handler(CommandHandler("help",            on_help))
    app.add_handler(CommandHandler("myprefs",         cmd_myprefs))
    app.add_handler(CommandHandler("clear",           cmd_clear))
    app.add_handler(CommandHandler("broadcast",       broadcast_start))
    app.add_handler(CommandHandler("schedule",        cmd_schedule))
    app.add_handler(CommandHandler("schedules",       cmd_schedules))
    app.add_handler(CommandHandler("cancelschedule",  cmd_cancelschedule))
    app.add_handler(CommandHandler("cancel",          cmd_cancel))
    app.add_handler(CommandHandler("stats",           admin_stats))
    app.add_handler(CommandHandler("admin",           cmd_admin))
    app.add_handler(CommandHandler("api",             cmd_api))
    app.add_handler(CommandHandler("botsettings",     cmd_botsettings))
    app.add_handler(CommandHandler("users",           cmd_users))
    app.add_handler(CommandHandler("chat",            cmd_chat))
    app.add_handler(CommandHandler("endchat",         cmd_endchat))

    # Callback handlers (priority order matters)
    app.add_handler(CallbackQueryHandler(broadcast_callback,  pattern=r"^bc_(confirm|cancel)$"))
    app.add_handler(CallbackQueryHandler(
        users_page_callback,
        pattern=r"^(users_page:\d+|users_search|users_search_page:\d+|users_close|noop|history_(page:\d+|refresh|close|user:\d+:\d+)|user_(view|chat|block|unblock|resetprefs|clearhist|history):\d+(?::[psh]\d+)?)$",
    ))
    app.add_handler(CallbackQueryHandler(sched_callback,      pattern=r"^sched_"))
    app.add_handler(CallbackQueryHandler(on_callback))

    # Message handlers
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(MessageHandler(
        (filters.Document.ALL | filters.AUDIO) & ~filters.VOICE,
        on_audio_file,
    ))
    app.add_handler(MessageHandler(
        filters.Sticker.ALL | filters.VIDEO | filters.VIDEO_NOTE,
        on_any_media,
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    logger.info(
        f"Bot polling started. Admins: {ADMIN_IDS or 'none configured'} | "
        f"HF: {HF_MODEL} | OCR provider: {OCR_PROVIDER} | "
        f"HF OCR: {HF_OCR_MODEL} | Gemini: {_gemini is not None}"
    )

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

    print(
        f"Bot is starting... (AI: {AI_PROVIDER} | HF: {HF_MODEL} | "
        f"OCR: {OCR_PROVIDER} | HF OCR: {HF_OCR_MODEL} | "
        f"Redis: {'on' if redis_client is not None else 'off'})"
    )

    while True:
        try:
            asyncio.run(_run_bot())
        except Exception as e:
            logger.error(f"Bot crashed: {e} — restarting in 5s...", exc_info=True)
        else:
            logger.warning("Polling stopped — restarting in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
