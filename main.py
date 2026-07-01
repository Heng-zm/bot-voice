import logging
import os
import re
import asyncio
import inspect
import threading
import time
import contextvars
import html
import json as _json
import functools
import tempfile
import gc
import hashlib
import hmac
import secrets
import socket
import atexit
import base64
import httpx
import imageio_ffmpeg as _iio_ffmpeg
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception as exc:
    logging.getLogger(__name__).warning("zoneinfo import failed; timezone fallback will be used: %s", exc)
    ZoneInfo = None
from urllib.parse import urlencode, quote
try:
    from google import genai
    from google.genai import types as genai_types
except Exception as exc:
    logging.getLogger(__name__).warning("Optional dependency google-genai is not available; Gemini features disabled: %s", exc)
    genai = None
    genai_types = None
from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response as StarletteResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.middleware.sessions import SessionMiddleware
from starlette.concurrency import run_in_threadpool
from jinja2 import Environment, Template, select_autoescape
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception as exc:
    logging.getLogger(__name__).warning("Optional dependency pydantic-settings is not available; env fallback settings will be used: %s", exc)
    BaseSettings = object
    SettingsConfigDict = dict
try:
    from supabase import create_client, Client
except Exception as exc:
    logging.getLogger(__name__).warning("Dependency supabase is not available; Supabase-backed features disabled: %s", exc)
    create_client = None
    Client = object
try:
    from supabase import acreate_client, AsyncClient
except Exception as exc:
    logging.getLogger(__name__).warning("Async Supabase client import failed; async Supabase calls disabled: %s", exc)
    acreate_client = None
    AsyncClient = object
from typing import Any, Callable


_ASYNC_RESOLVER_LOOP: asyncio.AbstractEventLoop | None = None
_ASYNC_RESOLVER_THREAD: threading.Thread | None = None
_ASYNC_RESOLVER_LOCK = threading.Lock()


def _async_resolver_loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    """Own exactly one background event loop for sync worker awaitables."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _get_async_resolver_loop() -> asyncio.AbstractEventLoop:
    """Return a process-wide resolver loop instead of creating per-call loops.

    This is used only from sync worker threads when a library unexpectedly
    returns an awaitable.  It avoids asyncio.run(...) inside ThreadPoolExecutor
    workers, which repeatedly creates/destroys event loops under load.
    """
    global _ASYNC_RESOLVER_LOOP, _ASYNC_RESOLVER_THREAD
    loop = _ASYNC_RESOLVER_LOOP
    if loop is not None and loop.is_running():
        return loop

    with _ASYNC_RESOLVER_LOCK:
        loop = _ASYNC_RESOLVER_LOOP
        if loop is not None and loop.is_running():
            return loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_async_resolver_loop_worker,
            args=(loop,),
            name="async-awaitable-resolver",
            daemon=True,
        )
        thread.start()
        _ASYNC_RESOLVER_LOOP = loop
        _ASYNC_RESOLVER_THREAD = thread
        return loop


def _shutdown_async_resolver_loop() -> None:
    """Stop the background resolver loop cleanly during process shutdown."""
    loop = _ASYNC_RESOLVER_LOOP
    thread = _ASYNC_RESOLVER_THREAD
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive():
        with suppress(Exception):
            thread.join(timeout=2.0)


atexit.register(_shutdown_async_resolver_loop)


def _close_unawaited(value: Any) -> None:
    """Best-effort coroutine cleanup when sync resolution cannot continue."""
    close = getattr(value, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


async def _await_sync_value(value: Any) -> Any:
    return await value


def _resolve_maybe_awaitable_sync(value: Any) -> Any:
    """Resolve sync or async SDK return values from worker/sync code paths.

    ThreadPoolExecutor workers must not call asyncio.run() for every Supabase
    awaitable.  Instead, submit the awaitable to one dedicated resolver event
    loop with asyncio.run_coroutine_threadsafe().  If this function is called
    from an already-running event loop, fail loudly so the async caller awaits
    the value directly instead of blocking its own loop.
    """
    if not inspect.isawaitable(value):
        return value

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        _close_unawaited(value)
        raise RuntimeError(
            "Async Supabase result returned inside a running event loop. "
            "Await it explicitly or move the call to db_call/db_call_sync."
        )

    resolver_loop = _get_async_resolver_loop()
    future = asyncio.run_coroutine_threadsafe(_await_sync_value(value), resolver_loop)
    return future.result()


class _SyncExecuteProxy:
    """Proxy query builders so .execute() works with sync or async Supabase clients."""

    __slots__ = ("_obj",)

    def __init__(self, obj: Any):
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, "_obj"), name)
        if not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if name == "execute":
                return _resolve_maybe_awaitable_sync(result)
            return _wrap_supabase_object(result)

        return _wrapped


def _wrap_supabase_object(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bytes, bytearray, int, float, bool, list, tuple, dict, set)):
        return obj
    if isinstance(obj, _SyncExecuteProxy):
        return obj
    if hasattr(obj, "execute"):
        return _SyncExecuteProxy(obj)
    return obj


class _SupabaseClientV2Compat:
    """Small compatibility wrapper for existing sync-style Supabase call sites."""

    __slots__ = ("_client",)

    def __init__(self, client: Any):
        object.__setattr__(self, "_client", client)

    def table(self, *args: Any, **kwargs: Any) -> Any:
        return _wrap_supabase_object(object.__getattribute__(self, "_client").table(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, "_client"), name)
        if not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return _wrap_supabase_object(attr(*args, **kwargs))

        return _wrapped


def _supabase_v2_compat_client(client: Any) -> Any:
    return None if client is None else _SupabaseClientV2Compat(client)


# ── Fast JSON codec helpers ────────────────────────────────────────────────
# Prefer binary JSON parsers to avoid raw_body.decode(...) copies on hot paths.
# The fallback is the stdlib json module, so the app still boots without extras.
try:
    import orjson as _orjson
except Exception:
    _orjson = None
try:
    import ujson as _ujson
except Exception:
    _ujson = None


def _json_loads_fast(data: bytes | bytearray | memoryview | str | None) -> Any:
    if data is None:
        return None
    if _orjson is not None:
        return _orjson.loads(data)
    if _ujson is not None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data).decode("utf-8")
        return _ujson.loads(data)
    if isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data).decode("utf-8")
    return _json.loads(data)


def _json_dumps_fast(content: Any) -> bytes:
    if _orjson is not None:
        return _orjson.dumps(content)
    if _ujson is not None:
        return _ujson.dumps(content, ensure_ascii=False).encode("utf-8")
    return _json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _FastJSONResponse(StarletteResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return _json_dumps_fast(content)

# Load local .env before any environment-driven config is read.
# This keeps local/dev runs consistent with Render environment variables.
try:
    from dotenv import load_dotenv as _early_load_dotenv
    _early_load_dotenv()
except Exception as exc:
    logging.getLogger(__name__).warning("Optional dependency python-dotenv is not available; .env file was not loaded: %s", exc)

try:
    import redis as redis_lib
except Exception as exc:
    logging.getLogger(__name__).warning("Optional dependency redis is not available; Redis cache/locks disabled: %s", exc)
    redis_lib = None


# ── Built-in smooth performance defaults ───────────────────────────────────
# These values are now code defaults, so the bot runs smoothly without adding
# performance knobs to Render/Railway env. Environment variables still override
# them for emergency deployments, and /admin settings can change the hot-safe
# values at runtime.
PERFORMANCE_CODE_DEFAULTS: dict[str, Any] = {
    "BOT_MODE": "WEBHOOK",
    "TELEGRAM_CONCURRENT_UPDATES": 4,
    "TELEGRAM_CONNECTION_POOL_SIZE": 24,
    "HTTP_MAX_CONNECTIONS": 100,
    "HTTP_MAX_KEEPALIVE_CONNECTIONS": 20,
    "DB_EXECUTOR_MAX_WORKERS": 3,
    "MAX_CONCURRENT_TTS_USERS": 2,
    "USER_SYNC_TTL_S": 1800.0,
    "PREFS_CACHE_TTL_S": 600.0,
    "TTS_AUDIO_CACHE_ENABLED": True,
    "TTS_AUDIO_CACHE_MAX_MB": 64,
    "TTS_AUDIO_CACHE_ITEM_MAX_MB": 8,
    "TTS_AUDIO_CACHE_TTL_S": 1200.0,
    "EDGE_TTS_PARALLEL_CHUNKS": 1,
    "PAGED_TTS_SEND_DELAY_S": 0.10,
    "WEB_STATUS_POLL_SECONDS": 30,
    "WEB_LIVE_POLL_SECONDS": 30,
}


def _perf_default(name: str, fallback: Any = None) -> Any:
    return PERFORMANCE_CODE_DEFAULTS.get(name, fallback)


# ── Admin cookie defaults ──────────────────────────────────────────────────
# Default cookie mode for separated frontend/backend deployments.
# SameSite=None requires Secure=True in modern browsers.
DEFAULT_WEB_COOKIE_SAMESITE = "none"
DEFAULT_WEB_COOKIE_SECURE = True
ADMIN_BACKEND_RELEASE = "v21-pdf-activity-graph-style"


# ── FastAPI Web Server + typed settings ─────────────────────────────────────
class AppSettings(BaseSettings):
    """Centralised environment configuration.

    The old helper functions (_env_bool/_env_int/_env_float) remain as safe
    compatibility wrappers, but critical web/server values now come from this
    typed settings object so bad env values fail predictably instead of being
    spread through the file.
    """
    if hasattr(BaseSettings, "model_config") or BaseSettings is not object:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    PORT: int = 8080
    RENDER: bool = False
    RENDER_EXTERNAL_URL: str = ""
    WEB_TRUST_PROXY: bool = False
    FLASK_SECRET_KEY: str = ""
    WEB_SECRET_KEY: str = ""
    # Redis-backed web session secret. The app reads this before normal Redis
    # startup so WEB_SECRET_KEY no longer needs to live in Render/Railway env.
    WEB_SECRET_REDIS_KEY: str = ""
    WEB_SESSION_COOKIE_NAME: str = "bot_admin_session"
    WEB_COOKIE_SAMESITE: str = DEFAULT_WEB_COOKIE_SAMESITE
    WEB_COOKIE_SECURE: bool = DEFAULT_WEB_COOKIE_SECURE
    WEB_ADMIN_SESSION_DAYS: int = 14
    WEB_MAX_CONTENT_LENGTH: int = 64 * 1024 * 1024
    # Backend/API-only mode: HTML admin dashboard pages are disabled by default.
    # Build/deploy the frontend separately and point it to this backend API.
    BACKEND_ONLY: bool = True
    WEB_ADMIN_FRONTEND_ENABLED: bool = False
    # Keep the backend API-only by default. Do not use "*" as a frontend URL.
    # Set ADMIN_FRONTEND_URL to the deployed frontend URL, and set
    # FRONTEND_ALLOWED_ORIGINS to a comma-separated allowlist.
    ADMIN_FRONTEND_URL: str = ""
    FRONTEND_ALLOWED_ORIGINS: str = ""
    # Emergency/dev-only escape hatch. This is now forcibly ignored in
    # production/runtime Render environments. Use exact frontend origins there.
    FRONTEND_ALLOW_ALL_ORIGINS: bool = False
    # Local Vite convenience only. When not in production and no frontend origins
    # are configured, allow http://localhost:5173 and http://127.0.0.1:5173.
    FRONTEND_LOCAL_DEV_ORIGINS_ENABLED: bool = True
    ADMIN_LOGIN_RATE_LIMIT_ATTEMPTS: int = 5
    ADMIN_LOGIN_RATE_LIMIT_WINDOW_S: float = 300.0
    ADMIN_LOGIN_RATE_LIMIT_REDIS_ENABLED: bool = True
    ADMIN_ORIGIN_GUARD_ENABLED: bool = True
    # Optional bearer fallback for separated frontends when browser cookies are
    # blocked by local dev, preview domains, or missing credentials: include.
    # The token is short-lived, HMAC-signed with WEB_SECRET_KEY, and accepted
    # only for explicitly configured ADMIN_IDS.
    ADMIN_API_TOKEN_FALLBACK_ENABLED: bool = True
    # Keep the separated-frontend bearer fallback short-lived by default.
    # Set ADMIN_API_TOKEN_FALLBACK_ENABLED=false after cookie auth is verified.
    ADMIN_API_TOKEN_TTL_SECONDS: int = 3600
    WEB_REQUIRE_STABLE_SECRET_IN_PRODUCTION: bool = True
    MAX_CONCURRENT_TTS_USERS: int = int(_perf_default("MAX_CONCURRENT_TTS_USERS", 2))
    MAX_CONCURRENT_AI: int = 3
    MAX_CONCURRENT_GEMINI: int = 3
    MAX_CONCURRENT_BROADCAST: int = 3
    BROADCAST_BATCH_SIZE: int = 3
    BROADCAST_INTER_BATCH_DELAY: float = 0.20
    TTS_RESOLVER_AI_ENABLED: bool = False
    APP_TIMEZONE: str = "Asia/Phnom_Penh"
    WEB_ADMIN_TIMEZONE: str = ""
    APP_TIMEZONE_ALIAS: str = "ICT"
    APP_TIMEZONE_UTC_LABEL: str = "UTC+7"
    BOT_SETTINGS_CACHE_TTL_S: float = 30.0
    WEB_BROADCAST_JOBS_MAX: int = 50
    WEB_BROADCAST_WORKERS: int = 3
    WEB_BROADCAST_DELAY_S: float = 0.05
    WEB_TABLE_PAGE_SIZE: int = 50
    WEB_COUNTS_CACHE_TTL_S: float = 30.0
    WEB_STATUS_POLL_SECONDS: int = int(_perf_default("WEB_STATUS_POLL_SECONDS", 30))
    WEB_LIVE_POLL_SECONDS: int = int(_perf_default("WEB_LIVE_POLL_SECONDS", 30))
    WEB_LIVE_SCHEDULES_CACHE_TTL_S: float = 30.0
    AI_API_KEY_CACHE_TTL_S: float = 60.0
    AI_API_KEY_TOUCH_INTERVAL_S: float = 60.0
    WEB_SLOW_REQUEST_MS: float = 1500.0
    WEB_ADMIN_AUDIT_MAX: int = 300
    TELEGRAM_CONCURRENT_UPDATES: int = int(_perf_default("TELEGRAM_CONCURRENT_UPDATES", 4))
    TELEGRAM_CONNECTION_POOL_SIZE: int = int(_perf_default("TELEGRAM_CONNECTION_POOL_SIZE", 24))
    DB_EXECUTOR_MAX_WORKERS: int = int(_perf_default("DB_EXECUTOR_MAX_WORKERS", 3))
    USER_SYNC_TTL_S: float = float(_perf_default("USER_SYNC_TTL_S", 1800.0))
    PREFS_CACHE_TTL_S: float = float(_perf_default("PREFS_CACHE_TTL_S", 600.0))
    TTS_AUDIO_CACHE_ENABLED: bool = bool(_perf_default("TTS_AUDIO_CACHE_ENABLED", True))
    TTS_AUDIO_CACHE_MAX_MB: int = int(_perf_default("TTS_AUDIO_CACHE_MAX_MB", 64))
    TTS_AUDIO_CACHE_ITEM_MAX_MB: int = int(_perf_default("TTS_AUDIO_CACHE_ITEM_MAX_MB", 8))
    TTS_AUDIO_CACHE_TTL_S: float = float(_perf_default("TTS_AUDIO_CACHE_TTL_S", 1200.0))
    EDGE_TTS_PARALLEL_CHUNKS: int = int(_perf_default("EDGE_TTS_PARALLEL_CHUNKS", 1))
    PAGED_TTS_SEND_DELAY_S: float = float(_perf_default("PAGED_TTS_SEND_DELAY_S", 0.10))
    BOT_MODE: str = str(_perf_default("BOT_MODE", "WEBHOOK"))
    TELEGRAM_WEBHOOK_URL: str | None = None
    TELEGRAM_WEBHOOK_SECRET_TOKEN: str | None = None
    TELEGRAM_WEBHOOK_DROP_PENDING_UPDATES: bool = False
    TELEGRAM_ALLOWED_UPDATES: str = "message,callback_query"
    HTTP_MAX_CONNECTIONS: int = int(_perf_default("HTTP_MAX_CONNECTIONS", 100))
    HTTP_MAX_KEEPALIVE_CONNECTIONS: int = int(_perf_default("HTTP_MAX_KEEPALIVE_CONNECTIONS", 20))
    REDIS_MAX_CONNECTIONS: int = 100
    USER_RATE_LIMIT_PER_SECOND: int = 3
    USER_RATE_LIMIT_WINDOW_S: float = 1.0
    API_RATE_LIMIT_PER_SECOND: int = 20
    API_RATE_LIMIT_WINDOW_S: float = 1.0
    # Security defaults: keep public surfaces tight unless explicitly opened.
    AI_API_ALLOWED_ORIGINS: str = ""
    AI_API_QUERY_KEY_ENABLED: bool = False
    WEBHOOK_REPLAY_TTL_S: int = 600
    ADMIN_CALLBACK_GUARD_ENABLED: bool = True
    RATE_LIMIT_REDIS_TTL_S: int = 30
    CACHE_ASIDE_DEFAULT_TTL_S: int = 3600
    WEB_BROADCAST_QUEUE_MAXSIZE: int = 200

try:
    SETTINGS = AppSettings()
except Exception as exc:
    # Keep the bot bootable even if pydantic-settings is not installed locally.
    logging.getLogger(__name__).warning("Pydantic settings fallback activated: %s", exc)
    class _FallbackSettings:
        def __getattr__(self, name):
            return os.environ.get(name, "")
    SETTINGS = _FallbackSettings()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        raw = getattr(SETTINGS, name, default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.environ.get(name, getattr(SETTINGS, name, default))
    try:
        value = int(str(raw).strip())
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = os.environ.get(name, getattr(SETTINGS, name, default))
    try:
        value = float(str(raw).strip())
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def _default_cookie_secure() -> bool:
    """Return a safe default for admin session cookies.

    - Explicit WEB_COOKIE_SECURE always wins.
    - SameSite=None requires Secure cookies in modern browsers.
    - Render/proxy HTTPS deployments should default to Secure.
    - Local development can stay non-secure unless configured otherwise.
    """
    explicit = os.environ.get("WEB_COOKIE_SECURE")
    if explicit is not None:
        return _env_bool("WEB_COOKIE_SECURE", DEFAULT_WEB_COOKIE_SECURE)
    same_site = str(os.environ.get("WEB_COOKIE_SAMESITE") or getattr(SETTINGS, "WEB_COOKIE_SAMESITE", DEFAULT_WEB_COOKIE_SAMESITE) or DEFAULT_WEB_COOKIE_SAMESITE).strip().lower()
    if same_site == "none":
        return True
    if _env_bool("WEB_TRUST_PROXY", _env_bool("RENDER", False)):
        return True
    return bool(getattr(SETTINGS, "WEB_COOKIE_SECURE", DEFAULT_WEB_COOKIE_SECURE))


def _cookie_samesite() -> str:
    """Return a Starlette-safe SameSite value.

    A mistyped WEB_COOKIE_SAMESITE used to make the dashboard fail during
    startup. Keep bad env values safe by falling back to lax, while the normal code default is SameSite=None.
    """
    value = str(getattr(SETTINGS, "WEB_COOKIE_SAMESITE", DEFAULT_WEB_COOKIE_SAMESITE) or DEFAULT_WEB_COOKIE_SAMESITE).strip().lower()
    return value if value in {"lax", "strict", "none"} else "lax"


def _web_max_content_length() -> int:
    return _env_int(
        "WEB_MAX_CONTENT_LENGTH",
        64 * 1024 * 1024,
        minimum=1 * 1024 * 1024,
        maximum=256 * 1024 * 1024,
    )


def _is_production_runtime() -> bool:
    env_name = str(os.environ.get("ENV") or os.environ.get("APP_ENV") or os.environ.get("PYTHON_ENV") or "").strip().lower()
    return _env_bool("RENDER", False) or env_name in {"prod", "production"}


_WEB_SESSION_SECRET_CACHE: str | None = None
_WEB_SESSION_SECRET_SOURCE_CACHE: str = ""
_WEB_SESSION_SECRET_REDIS_CLIENT: Any | None = None
_WEB_SESSION_SECRET_REDIS_LOCK = threading.RLock()
_WEB_SESSION_SECRET_REDIS_LAST_ERROR: str = ""


def _generate_web_secret_value() -> str:
    """Generate a strong Starlette/FastAPI session signing secret."""
    return secrets.token_urlsafe(48)


def _web_secret_fingerprint(secret_value: str) -> str:
    value = str(secret_value or "")
    if not value:
        return "not-set"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _web_secret_redis_key() -> str:
    """Redis key that stores the stable WEB_SECRET_KEY replacement."""
    configured = str(
        os.environ.get("WEB_SECRET_REDIS_KEY")
        or getattr(SETTINGS, "WEB_SECRET_REDIS_KEY", "")
        or ""
    ).strip()
    if configured:
        return configured
    prefix = str(os.environ.get("REDIS_CACHE_PREFIX") or "tgbot").strip() or "tgbot"
    return f"{prefix}:web_secret_key:v1"


def _web_secret_redis_url() -> str:
    return str(os.environ.get("REDIS_URL") or "").strip()


def _web_secret_redis_client() -> Any | None:
    """Return a Redis client usable before the normal app Redis init runs.

    FastAPI SessionMiddleware is created during module import, before load_env()
    initializes the shared redis_client.  This small early client exists only so
    the web session secret can be read/generated from Redis at startup.  After
    the normal redis_client exists, this helper reuses it.
    """
    global _WEB_SESSION_SECRET_REDIS_CLIENT, _WEB_SESSION_SECRET_REDIS_LAST_ERROR

    shared_client = globals().get("redis_client")
    if shared_client is not None:
        return shared_client

    if redis_lib is None:
        _WEB_SESSION_SECRET_REDIS_LAST_ERROR = "redis package is not installed"
        return None

    redis_url = _web_secret_redis_url()
    if not redis_url:
        _WEB_SESSION_SECRET_REDIS_LAST_ERROR = "REDIS_URL is not configured"
        return None

    with _WEB_SESSION_SECRET_REDIS_LOCK:
        if _WEB_SESSION_SECRET_REDIS_CLIENT is not None:
            return _WEB_SESSION_SECRET_REDIS_CLIENT
        try:
            timeout = _env_float("WEB_SECRET_REDIS_SOCKET_TIMEOUT_S", 3.0, minimum=0.2, maximum=30.0)
            max_connections = _env_int("WEB_SECRET_REDIS_MAX_CONNECTIONS", 3, minimum=1, maximum=20)
            client = redis_lib.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
                health_check_interval=30,
                retry_on_timeout=True,
                max_connections=max_connections,
            )
            client.ping()
            _WEB_SESSION_SECRET_REDIS_CLIENT = client
            _WEB_SESSION_SECRET_REDIS_LAST_ERROR = ""
            logging.getLogger(__name__).info(
                "Redis web session secret client initialised key=%s max_connections=%s.",
                _web_secret_redis_key(),
                max_connections,
            )
            return client
        except Exception as exc:
            _WEB_SESSION_SECRET_REDIS_LAST_ERROR = str(exc)[:300]
            logging.getLogger(__name__).warning(
                "Redis web session secret client failed; falling back if possible: %s",
                _WEB_SESSION_SECRET_REDIS_LAST_ERROR,
            )
            return None


def _web_secret_get_from_redis() -> tuple[str, str]:
    client = _web_secret_redis_client()
    if client is None:
        return "", "redis-unavailable"
    key = _web_secret_redis_key()
    try:
        raw = client.get(key)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        value = str(raw or "").strip()
        if len(value) >= 32:
            return value, f"redis:{key}"
        if value:
            logging.getLogger(__name__).warning(
                "Ignoring too-short Redis WEB_SECRET_KEY value key=%s fingerprint=%s.",
                key,
                _web_secret_fingerprint(value),
            )
        return "", f"redis:{key}:missing"
    except Exception as exc:
        logging.getLogger(__name__).warning("Redis WEB_SECRET_KEY read failed key=%s: %s", key, exc)
        return "", f"redis:{key}:error"


def _web_secret_set_in_redis_sync(secret_value: str, *, overwrite: bool = True) -> tuple[bool, str]:
    secret_value = str(secret_value or "").strip()
    if len(secret_value) < 32:
        return False, "Generated secret is too short; refusing to store it."
    client = _web_secret_redis_client()
    if client is None:
        detail = _WEB_SESSION_SECRET_REDIS_LAST_ERROR or "Redis is not available"
        return False, f"Cannot store WEB_SECRET_KEY in Redis: {detail}"
    key = _web_secret_redis_key()
    try:
        if overwrite:
            ok = client.set(key, secret_value)
        else:
            ok = client.set(key, secret_value, nx=True)
        if not ok and not overwrite:
            return False, f"Redis key already exists: {key}"
        return True, f"Stored in Redis key {key}"
    except Exception as exc:
        logging.getLogger(__name__).error("Redis WEB_SECRET_KEY write failed key=%s: %s", key, exc, exc_info=True)
        return False, f"Redis WEB_SECRET_KEY write failed: {str(exc)[:300]}"


def _web_secret_get_or_create_from_redis() -> tuple[str, str]:
    existing, source = _web_secret_get_from_redis()
    if existing:
        return existing, source

    client = _web_secret_redis_client()
    if client is None:
        return "", source

    key = _web_secret_redis_key()
    generated = _generate_web_secret_value()
    try:
        # NX makes first boot safe in two-server mode: one instance creates the
        # shared secret, the other reads the winning value immediately after.
        created = bool(client.set(key, generated, nx=True))
        if created:
            logging.getLogger(__name__).warning(
                "Generated and stored new Redis WEB_SECRET_KEY replacement key=%s fingerprint=%s. Restart all backend instances after manual rotations only.",
                key,
                _web_secret_fingerprint(generated),
            )
            return generated, f"redis:{key}:generated"
        winner, winner_source = _web_secret_get_from_redis()
        if winner:
            return winner, winner_source
        return "", f"redis:{key}:race-empty"
    except Exception as exc:
        logging.getLogger(__name__).warning("Redis WEB_SECRET_KEY auto-create failed key=%s: %s", key, exc)
        return "", f"redis:{key}:create-error"


def _web_session_secret_source() -> str:
    return _WEB_SESSION_SECRET_SOURCE_CACHE or "unknown"


def _web_session_secret_fingerprint() -> str:
    return _web_secret_fingerprint(_WEB_SESSION_SECRET_CACHE or "")


def _web_stable_secret_configured() -> bool:
    if _WEB_SESSION_SECRET_CACHE and not (_WEB_SESSION_SECRET_SOURCE_CACHE or "").startswith("random-startup"):
        return True
    redis_value, _redis_source = _web_secret_get_from_redis()
    if redis_value:
        return True
    return bool(
        os.environ.get("WEB_SECRET_KEY")
        or os.environ.get("FLASK_SECRET_KEY")
        or getattr(SETTINGS, "WEB_SECRET_KEY", "")
        or getattr(SETTINGS, "FLASK_SECRET_KEY", "")
    )


def _web_session_secret_key() -> str:
    global _WEB_SESSION_SECRET_CACHE, _WEB_SESSION_SECRET_SOURCE_CACHE

    if _WEB_SESSION_SECRET_CACHE:
        return _WEB_SESSION_SECRET_CACHE

    # Primary path: Redis.  This removes the need to put WEB_SECRET_KEY in the
    # deployment environment and keeps multiple backend instances in sync.
    redis_secret, redis_source = _web_secret_get_or_create_from_redis()
    if redis_secret:
        _WEB_SESSION_SECRET_CACHE = redis_secret
        _WEB_SESSION_SECRET_SOURCE_CACHE = redis_source
        return redis_secret

    # Compatibility fallback only.  Redis remains the preferred source.
    secret = (
        os.environ.get("WEB_SECRET_KEY")
        or os.environ.get("FLASK_SECRET_KEY")
        or getattr(SETTINGS, "WEB_SECRET_KEY", "")
        or getattr(SETTINGS, "FLASK_SECRET_KEY", "")
        or ""
    )
    secret = str(secret or "").strip()
    if secret:
        _WEB_SESSION_SECRET_CACHE = secret
        _WEB_SESSION_SECRET_SOURCE_CACHE = "env:web_secret_key"
        logging.getLogger(__name__).warning(
            "Using env WEB_SECRET_KEY fallback. Recommended: remove WEB_SECRET_KEY from env and store the key in Redis key=%s.",
            _web_secret_redis_key(),
        )
        return secret

    msg = (
        "Redis-backed WEB_SECRET_KEY is not available. "
        "Production requires a stable web session secret so admin sessions do "
        "not reset after restart and do not break across multiple servers. "
        "Set REDIS_URL, or explicitly set WEB_SECRET_KEY only as a temporary "
        "compatibility fallback."
    )
    if _is_production_runtime() and _env_bool("WEB_REQUIRE_STABLE_SECRET_IN_PRODUCTION", True):
        logging.getLogger(__name__).critical(msg)
        raise RuntimeError(msg)

    logging.getLogger(__name__).warning(
        "%s Using a temporary in-memory secret because this is not production "
        "or WEB_REQUIRE_STABLE_SECRET_IN_PRODUCTION=false.",
        msg,
    )
    _WEB_SESSION_SECRET_CACHE = secrets.token_hex(32)
    _WEB_SESSION_SECRET_SOURCE_CACHE = "random-startup-fallback"
    return _WEB_SESSION_SECRET_CACHE


_request_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("fastapi_request_ctx")
_session_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("fastapi_session_ctx")


class _LocalProxy:
    def __init__(self, ctx: contextvars.ContextVar):
        object.__setattr__(self, "_ctx", ctx)

    def _get_current_object(self):
        return object.__getattribute__(self, "_ctx").get()

    def __getattr__(self, name):
        return getattr(self._get_current_object(), name)

    def __setattr__(self, name, value):
        setattr(self._get_current_object(), name, value)

    def __getitem__(self, key):
        return self._get_current_object()[key]

    def __setitem__(self, key, value):
        self._get_current_object()[key] = value

    def __delitem__(self, key):
        del self._get_current_object()[key]

    def get(self, *args, **kwargs):
        return self._get_current_object().get(*args, **kwargs)

    def pop(self, *args, **kwargs):
        return self._get_current_object().pop(*args, **kwargs)

    def clear(self):
        return self._get_current_object().clear()

    def setdefault(self, *args, **kwargs):
        return self._get_current_object().setdefault(*args, **kwargs)

    def __contains__(self, item):
        return item in self._get_current_object()


class _MultiDictCompat(dict):
    def get(self, key, default=None, type=None):
        value = super().get(key, default)
        if isinstance(value, list):
            value = value[-1] if value else default
        if type is not None and value is not None:
            try:
                return type(value)
            except Exception:
                return default
        return value

    def getlist(self, key):
        value = super().get(key, [])
        if isinstance(value, list):
            return list(value)
        if value is None:
            return []
        return [value]


class _UploadFileCompat:
    def __init__(self, filename: str, content_type: str, data: bytes):
        self.filename = filename or "upload.bin"
        self.content_type = content_type or "application/octet-stream"
        self.mimetype = self.content_type
        self._data = data or b""

    def read(self) -> bytes:
        return self._data


class _RequestCompat:
    def __init__(self, req: FastAPIRequest, form: dict, files: dict, raw_body: bytes, json_body: Any):
        self._req = req
        self.method = req.method
        self.headers = req.headers
        self.args = _MultiDictCompat(dict(req.query_params))
        self.form = _MultiDictCompat(form)
        self.files = _MultiDictCompat(files)
        self.content_type = req.headers.get("content-type", "")
        self.referrer = req.headers.get("referer", "")
        self.path = req.url.path
        self.full_path = req.url.path + (("?" + req.url.query) if req.url.query else "")
        self._raw_body = raw_body or b""
        self._json_body = json_body

    def get_json(self, force: bool = False, silent: bool = False):
        if self._json_body is not None:
            return self._json_body
        if not self._raw_body:
            return None if silent else {}
        try:
            return _json_loads_fast(self._raw_body)
        except Exception:
            if silent:
                return None
            raise


request = _LocalProxy(_request_ctx)
session = _LocalProxy(_session_ctx)


def jsonify(obj: Any = None, **kwargs: Any):
    # FastAPI/Starlette JSONResponse cannot serialize datetime, Decimal, UUID,
    # or Pydantic objects directly. jsonable_encoder keeps dashboard/API JSON
    # responses safe even when Supabase/client libraries return richer types.
    # _FastJSONResponse uses orjson/ujson when installed and falls back safely.
    return _FastJSONResponse(jsonable_encoder(obj if obj is not None else kwargs))


def Response(content: Any = "", status: int = 200, status_code: int | None = None, mimetype: str | None = None, media_type: str | None = None, headers: dict | None = None):
    return StarletteResponse(content=content, status_code=status_code or status, media_type=media_type or mimetype, headers=headers)


def redirect(location: str, code: int = 302):
    checker = globals().get("_should_return_backend_json_redirect")
    if callable(checker):
        try:
            if checker(location):
                return _admin_api_success({"redirect": location}, status_code=200)
        except Exception:
            pass
    return RedirectResponse(url=location, status_code=code)


def stream_with_context(generator):
    return generator


def abort(status_code: int = 400):
    raise HTTPException(status_code=status_code)


_JINJA_TEMPLATE_CACHE_MAX = 256
_JINJA_TEMPLATE_CACHE: OrderedDict[tuple[int, str], tuple[str, Any]] = OrderedDict()
_JINJA_TEMPLATE_CACHE_LOCK = threading.RLock()
_JINJA_ENV = Environment(autoescape=select_autoescape(["html", "xml"]), enable_async=False)


def render_template_string(template: str, **context: Any) -> str:
    # Compile each unique template string once. The compatibility layer uses
    # large inline dashboard templates, so this avoids repeated Jinja parsing
    # on every admin page render while keeping Flask's function name intact.
    template = str(template or "")
    key = (len(template), hashlib.sha256(template.encode("utf-8")).hexdigest())
    compiled = None
    with _JINJA_TEMPLATE_CACHE_LOCK:
        cached = _JINJA_TEMPLATE_CACHE.get(key)
        if cached and cached[0] == template:
            compiled = cached[1]
            _JINJA_TEMPLATE_CACHE.move_to_end(key)
    if compiled is None:
        try:
            compiled = _JINJA_ENV.from_string(template)
        except Exception:
            # Preserve old fallback behavior if a third-party Jinja extension or
            # unusual template edge case is not accepted by the shared env.
            return Template(template, autoescape=select_autoescape(["html", "xml"])).render(**context)
        with _JINJA_TEMPLATE_CACHE_LOCK:
            _JINJA_TEMPLATE_CACHE[key] = (template, compiled)
            while len(_JINJA_TEMPLATE_CACHE) > _JINJA_TEMPLATE_CACHE_MAX:
                _JINJA_TEMPLATE_CACHE.popitem(last=False)
    return compiled.render(**context)


def flask_flash(message: str, category: str = "message") -> None:
    flashes = list(session.get("_flashes", []))
    flashes.append((category, message))
    session["_flashes"] = flashes


def get_flashed_messages(with_categories: bool = False):
    flashes = list(session.pop("_flashes", []))
    if with_categories:
        return flashes
    return [msg for _cat, msg in flashes]


def url_for(endpoint: str, **params: Any) -> str:
    try:
        clean = {k: str(v) for k, v in params.items() if v is not None}
        return str(app_flask.fastapi.url_path_for(endpoint, **clean))
    except Exception:
        return "/" + endpoint.replace("web_admin_", "admin/").replace("_", "-")


class FastAPICompatApp:
    """Small Flask-style compatibility layer over FastAPI.

    This lets the existing admin/AI route bodies keep working while the actual
    ASGI server, sessions, CORS/security headers, and lifecycle run on FastAPI.
    It removes the old separate Flask thread and makes future route migrations
    incremental instead of risky all-at-once rewrites.
    """
    def __init__(self):
        self.config: dict[str, Any] = {}
        self.after_request_funcs: list[Callable[[Any], Any]] = []
        secret_key = _web_session_secret_key()
        api_docs_enabled = _env_bool("API_DOCS_ENABLED", False)
        self.fastapi = FastAPI(
            title="Telegram Bot Admin",
            docs_url="/docs" if api_docs_enabled else None,
            redoc_url="/redoc" if api_docs_enabled else None,
            openapi_url="/openapi.json" if api_docs_enabled else None,
        )
        self.fastapi.add_middleware(
            SessionMiddleware,
            secret_key=secret_key,
            session_cookie=str(getattr(SETTINGS, "WEB_SESSION_COOKIE_NAME", "bot_admin_session") or "bot_admin_session"),
            https_only=_default_cookie_secure(),
            same_site=_cookie_samesite(),
            max_age=max(3600, _env_int("WEB_ADMIN_SESSION_DAYS", 14, minimum=1, maximum=365) * 86400),
        )

    def after_request(self, fn: Callable[[Any], Any]):
        self.after_request_funcs.append(fn)
        return fn

    @staticmethod
    def _convert_path(path: str) -> str:
        path = re.sub(r"<int:(\w+)>", r"{\1}", path)
        path = re.sub(r"<str:(\w+)>", r"{\1}", path)
        path = re.sub(r"<(\w+)>", r"{\1}", path)
        return path

    @staticmethod
    def _path_converters(path: str) -> dict[str, str]:
        converters: dict[str, str] = {}
        for kind, name in re.findall(r"<(?:(int|str):)?(\w+)>", path):
            converters[name] = kind or "str"
        return converters

    @staticmethod
    def _cast_path_params(params: dict[str, Any], converters: dict[str, str]) -> dict[str, Any]:
        casted = dict(params)
        for name, kind in converters.items():
            if name not in casted:
                continue
            if kind == "int":
                try:
                    casted[name] = int(casted[name])
                except (TypeError, ValueError):
                    raise HTTPException(status_code=404)
            else:
                casted[name] = str(casted[name])
        return casted

    async def _prepare_context(self, req: FastAPIRequest):
        raw_body = b""
        json_body = None
        form_data: dict[str, Any] = {}
        files_data: dict[str, Any] = {}
        content_type = req.headers.get("content-type", "")
        max_body = _web_max_content_length()

        # Do the hard limit check before Starlette reads/parses the request
        # body. This prevents obvious memory-flood uploads from ever reaching
        # req.body(), req.form(), or UploadFile.read().
        content_length = req.headers.get("content-length")
        if content_length:
            try:
                content_length_i = int(str(content_length).strip())
                if content_length_i < 0:
                    raise ValueError
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Content-Length header.")
            if content_length_i > max_body:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Max {max_body} bytes.",
                )

        def _store_multi(target: dict[str, Any], key: str, value: Any) -> None:
            if key in target:
                current = target[key]
                if isinstance(current, list):
                    current.append(value)
                else:
                    target[key] = [current, value]
            else:
                target[key] = value

        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            try:
                form = await req.form(max_part_size=max_body)
            except TypeError:
                # Older Starlette versions do not expose max_part_size. The
                # upfront Content-Length guard above still protects normal
                # browser/API clients; the manual read guard below remains a
                # second line of defense.
                form = await req.form()
            total_read = 0
            for key, value in form.multi_items():
                if hasattr(value, "read") and hasattr(value, "filename"):
                    data = await value.read()
                    total_read += len(data)
                    if total_read > max_body:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Uploaded data too large. Max {max_body} bytes.",
                        )
                    _store_multi(
                        files_data,
                        key,
                        _UploadFileCompat(
                            value.filename,
                            getattr(value, "content_type", "") or "application/octet-stream",
                            data,
                        ),
                    )
                else:
                    _store_multi(form_data, key, value)
        else:
            # Stream instead of await req.body() so requests without a trusted
            # Content-Length cannot allocate unbounded memory before rejection.
            chunks: list[bytes] = []
            total_read = 0
            async for chunk in req.stream():
                if not chunk:
                    continue
                total_read += len(chunk)
                if total_read > max_body:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request body too large. Max {max_body} bytes.",
                    )
                chunks.append(chunk)
            raw_body = b"".join(chunks)
            if raw_body and "application/json" in content_type:
                try:
                    json_body = _json_loads_fast(raw_body)
                except Exception:
                    json_body = None
        compat = _RequestCompat(req, form_data, files_data, raw_body, json_body)
        req_token = _request_ctx.set(compat)
        sess_token = _session_ctx.set(req.session)
        return req_token, sess_token

    def _finalize_response(self, result: Any):
        status_code = None
        if isinstance(result, tuple):
            if len(result) >= 2:
                result, status_code = result[0], int(result[1])
        if isinstance(result, StarletteResponse):
            resp = result
            if status_code is not None:
                resp.status_code = status_code
        elif isinstance(result, (dict, list)):
            resp = _FastJSONResponse(jsonable_encoder(result), status_code=status_code or 200)
        elif isinstance(result, bytes):
            resp = Response(result, status=status_code or 200)
        elif result is None:
            resp = Response("", status=status_code or 204)
        else:
            resp = HTMLResponse(str(result), status_code=status_code or 200)
        for fn in self.after_request_funcs:
            maybe_resp = fn(resp)
            if maybe_resp is not None:
                resp = maybe_resp
        return resp

    def route(self, path: str, methods: list[str] | tuple[str, ...] | None = None):
        route_methods = list(methods or ["GET"])
        fastapi_path = self._convert_path(path)
        converters = self._path_converters(path)
        def decorator(func: Callable):
            async def endpoint(starlette_request: FastAPIRequest):
                req_token = sess_token = None
                try:
                    req_token, sess_token = await self._prepare_context(starlette_request)
                    kwargs = self._cast_path_params(dict(starlette_request.path_params), converters)
                    if inspect.iscoroutinefunction(func):
                        result = await func(**kwargs)
                    else:
                        result = await run_in_threadpool(lambda: func(**kwargs))
                    return self._finalize_response(result)
                finally:
                    if sess_token is not None:
                        _session_ctx.reset(sess_token)
                    if req_token is not None:
                        _request_ctx.reset(req_token)
            endpoint.__name__ = func.__name__ + "_asgi"
            self.fastapi.add_api_route(
                fastapi_path,
                endpoint,
                methods=route_methods,
                name=func.__name__,
                include_in_schema=_env_bool("API_DOCS_INCLUDE_COMPAT_ROUTES", True),
            )
            return func
        return decorator

    def run(self, host: str = "0.0.0.0", port: int = 8080, **kwargs: Any):
        import uvicorn
        uvicorn.run(self.fastapi, host=host, port=port, **kwargs)


app_flask = FastAPICompatApp()
app = app_flask.fastapi


# ── Backend/API-only mode + separated frontend CORS ────────────────────────
def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, getattr(SETTINGS, name, default)) or default).strip()


def _admin_frontend_url() -> str:
    value = _env_str("ADMIN_FRONTEND_URL", "").rstrip("/")
    # "*" is valid for an origin allowlist, but never for a concrete frontend URL.
    return "" if value == "*" else value


def _backend_only_mode() -> bool:
    # Default is API-only so the Python service no longer serves dashboard UI.
    # To temporarily restore old inline dashboard pages, set:
    #   BACKEND_ONLY=false
    #   WEB_ADMIN_FRONTEND_ENABLED=true
    return _env_bool("BACKEND_ONLY", True) or not _env_bool("WEB_ADMIN_FRONTEND_ENABLED", False)


DEFAULT_LOCAL_FRONTEND_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def _frontend_allow_all_origins_enabled() -> bool:
    # Dev/emergency only. Production must use FRONTEND_ALLOWED_ORIGINS with
    # exact URLs because wildcard credentialed CORS is unsafe with admin cookies.
    requested = _env_bool("FRONTEND_ALLOW_ALL_ORIGINS", False)
    if requested and _is_production_runtime():
        logging.getLogger(__name__).warning(
            "Ignoring FRONTEND_ALLOW_ALL_ORIGINS=true in production. "
            "Set ADMIN_FRONTEND_URL and FRONTEND_ALLOWED_ORIGINS to exact frontend origins."
        )
        return False
    return requested


def _frontend_local_dev_origins_enabled() -> bool:
    explicit = os.environ.get("FRONTEND_LOCAL_DEV_ORIGINS_ENABLED")
    if explicit is not None:
        return _env_bool("FRONTEND_LOCAL_DEV_ORIGINS_ENABLED", True)
    return not _is_production_runtime() and _env_bool("FRONTEND_LOCAL_DEV_ORIGINS_ENABLED", True)


def _frontend_allowed_origins() -> list[str]:
    raw = _env_str("FRONTEND_ALLOWED_ORIGINS", "")
    admin_url = _admin_frontend_url()
    origins: list[str] = []
    allow_all_requested = False
    for item in (raw.split(",") if raw else []):
        item = item.strip().rstrip("/")
        if not item:
            continue
        if item == "*":
            allow_all_requested = True
            continue
        if item not in origins:
            origins.append(item)
    if admin_url and admin_url not in origins:
        origins.append(admin_url)
    if _frontend_local_dev_origins_enabled():
        for item in DEFAULT_LOCAL_FRONTEND_ORIGINS:
            if item not in origins:
                origins.append(item)
    if allow_all_requested:
        if _frontend_allow_all_origins_enabled():
            origins.append("*")
        else:
            logging.getLogger(__name__).warning(
                "Ignoring FRONTEND_ALLOWED_ORIGINS='*'. Wildcard admin CORS is allowed only "
                "outside production and only when FRONTEND_ALLOW_ALL_ORIGINS=true. "
                "Set exact frontend origins in production."
            )
    return origins


def _normalise_origin(value: str) -> str:
    return str(value or "").strip().rstrip("/")


def _admin_origin_guard_enabled() -> bool:
    return _env_bool("ADMIN_ORIGIN_GUARD_ENABLED", True)


def _admin_origin_allowed(origin: str) -> bool:
    origin = _normalise_origin(origin)
    if not origin:
        return True
    allowed = [_normalise_origin(item) for item in _frontend_allowed_origins()]
    if "*" in allowed and _frontend_allow_all_origins_enabled():
        return True
    return origin in allowed


def _admin_origin_guard_payload(origin: str) -> dict[str, Any]:
    return {
        "ok": False,
        "code": "admin_origin_forbidden",
        "error": "This admin API origin is not allowed. Set ADMIN_FRONTEND_URL and FRONTEND_ALLOWED_ORIGINS to your frontend URL.",
        "origin": _normalise_origin(origin),
        "frontend_url": _admin_frontend_url(),
        "frontend_allowed_origins": _frontend_allowed_origins(),
        "backend_only": _backend_only_mode(),
    }


def _request_wants_json() -> bool:
    try:
        accept = str(request.headers.get("accept") or "").lower()
        content_type = str(request.headers.get("content-type") or "").lower()
        requested_with = str(request.headers.get("x-requested-with") or "").lower()
        return (
            "application/json" in accept
            or "application/json" in content_type
            or requested_with in {"fetch", "xmlhttprequest"}
            or str(getattr(request, "path", "")).startswith("/api/")
            or str(getattr(request, "path", "")).endswith(".json")
        )
    except Exception:
        return False


def _admin_api_error(message: str, status_code: int = 400, *, code: str = "admin_error", **extra: Any):
    payload = {
        "ok": False,
        "error": str(message),
        "code": code,
        "backend_only": _backend_only_mode(),
        "frontend_url": _admin_frontend_url(),
    }
    payload.update(extra)
    resp = jsonify(payload)
    resp.status_code = int(status_code)
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


def _admin_api_success(payload: dict | None = None, *, status_code: int = 200):
    data = {"ok": True, "backend_only": _backend_only_mode(), "frontend_url": _admin_frontend_url()}
    if payload:
        data.update(payload)
    resp = jsonify(data)
    resp.status_code = int(status_code)
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


def _legacy_dashboard_disabled_payload(title: str = "Admin dashboard disabled", *, status_code: int = 410):
    return _admin_api_error(
        "HTML admin dashboard pages are disabled. Deploy the frontend as a separate app and call this backend API.",
        status_code,
        code="admin_frontend_disabled",
        title=title,
        auth={
            "login": "/api/admin/auth/login",
            "me": "/api/admin/auth/me",
            "logout": "/api/admin/auth/logout",
        },
        endpoints={
            "health": "/readyz",
            "status": "/admin/status.json",
            "live": "/admin/live.json",
            "performance": "/admin/performance.json",
            "errors": "/admin/errors.json",
            "broadcast_jobs": "/admin/broadcast/jobs.json",
            "report_pdf": "/admin/report.pdf",
        },
    )


def _should_return_backend_json_redirect(location: str) -> bool:
    try:
        path = str(getattr(request, "path", "") or "")
    except Exception:
        path = ""
    return _backend_only_mode() and (path == "/dashboard" or path.startswith("/admin"))


_allowed_frontend_origins = _frontend_allowed_origins()
if _allowed_frontend_origins:
    _allow_all_frontends = "*" in _allowed_frontend_origins
    # Exact origin allowlist is the production-safe path for cookie sessions.
    # Wildcard origin matching is enabled only when FRONTEND_ALLOW_ALL_ORIGINS=true.
    cors_kwargs = {
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type", "X-CSRF-Token", "X-Requested-With", "X-Api-Key"],
        "expose_headers": ["X-Request-ID", "X-Response-Time-ms"],
    }
    if _allow_all_frontends:
        logging.getLogger(__name__).warning(
            "FRONTEND_ALLOW_ALL_ORIGINS=true enables wildcard credentialed CORS. "
            "Use only for development or trusted isolated deployments."
        )
        cors_kwargs.update({"allow_origins": [], "allow_origin_regex": r"https?://.*"})
    else:
        cors_kwargs.update({"allow_origins": _allowed_frontend_origins})
    app.add_middleware(CORSMiddleware, **cors_kwargs)


@app.middleware("http")
async def _admin_origin_guard_middleware(req: FastAPIRequest, call_next):
    """Reject browser state-changing admin API calls from unknown origins.

    CORS controls what browsers can read, but this guard also prevents
    credentialed writes from untrusted sites when cookies are enabled.  Requests
    with no Origin header (curl/server-to-server/same-origin form posts) are
    allowed. Configure ADMIN_FRONTEND_URL / FRONTEND_ALLOWED_ORIGINS for a
    separated frontend.
    """
    method = str(req.method or "GET").upper()
    path = str(req.url.path or "")
    if (
        _admin_origin_guard_enabled()
        and method in {"POST", "PUT", "PATCH", "DELETE"}
        and (path.startswith("/api/admin") or path.startswith("/admin/"))
    ):
        origin = _normalise_origin(req.headers.get("origin") or "")
        if origin and not _admin_origin_allowed(origin):
            logging.getLogger(__name__).warning(
                "Blocked admin API request from disallowed origin origin=%s method=%s path=%s",
                origin,
                method,
                path,
            )
            return _FastJSONResponse(_admin_origin_guard_payload(origin), status_code=403)
    return await call_next(req)


def _is_legacy_admin_frontend_request(path: str, method: str) -> bool:
    path = str(path or "")
    method = str(method or "GET").upper()
    if path == "/dashboard":
        return True
    if path in {"/admin", "/admin/", "/admin/login"}:
        return True
    if not path.startswith("/admin/"):
        return False
    # Backend resources remain available to the separated frontend.
    if path.endswith((".json", ".pdf", ".csv")):
        return False
    # Keep write/action endpoints callable; they return JSON redirects in API-only mode.
    if method != "GET" or path.endswith("/action"):
        return False
    return True


@app.middleware("http")
async def _backend_only_legacy_dashboard_guard(req: FastAPIRequest, call_next):
    if _backend_only_mode() and _is_legacy_admin_frontend_request(req.url.path, req.method):
        return _FastJSONResponse({
            "ok": False,
            "code": "admin_frontend_disabled",
            "error": "HTML admin dashboard pages are disabled. Deploy the frontend separately and call this backend API.",
            "backend_only": True,
            "frontend_url": _admin_frontend_url(),
            "auth": {
                "login": "/api/admin/auth/login",
                "me": "/api/admin/auth/me",
                "logout": "/api/admin/auth/logout",
            },
            "endpoints": {
                "health": "/readyz",
                "bootstrap": "/api/admin/bootstrap",
                "status": "/admin/status.json",
                "live": "/admin/live.json",
                "performance": "/admin/performance.json",
                "errors": "/admin/errors.json",
                "broadcast_jobs": "/admin/broadcast/jobs.json",
                "report_pdf": "/admin/report.pdf",
            },
        }, status_code=410)
    return await call_next(req)
app_flask.config["SECRET_KEY"] = _web_session_secret_key()
app_flask.config["SESSION_COOKIE_NAME"] = getattr(SETTINGS, "WEB_SESSION_COOKIE_NAME", "bot_admin_session")
app_flask.config["SESSION_COOKIE_HTTPONLY"] = True
app_flask.config["SESSION_COOKIE_SAMESITE"] = _cookie_samesite().capitalize()
app_flask.config["SESSION_COOKIE_SECURE"] = _default_cookie_secure()
app_flask.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=_env_int("WEB_ADMIN_SESSION_DAYS", 14, minimum=1))
app_flask.config["SESSION_REFRESH_EACH_REQUEST"] = True
app_flask.config["MAX_CONTENT_LENGTH"] = _web_max_content_length()

# ── Admin Performance V5 runtime knobs ──────────────────────────────────────
# These are intentionally env-tunable so Render small instances can stay stable.
WEB_SLOW_REQUEST_MS = _env_float("WEB_SLOW_REQUEST_MS", 1500.0, minimum=100.0, maximum=60_000.0)
WEB_ADMIN_AUDIT_MAX = _env_int("WEB_ADMIN_AUDIT_MAX", 300, minimum=50, maximum=5_000)
TELEGRAM_CONCURRENT_UPDATES = _env_int("TELEGRAM_CONCURRENT_UPDATES", int(_perf_default("TELEGRAM_CONCURRENT_UPDATES", 4)), minimum=1, maximum=64)
TELEGRAM_CONNECTION_POOL_SIZE = _env_int("TELEGRAM_CONNECTION_POOL_SIZE", int(_perf_default("TELEGRAM_CONNECTION_POOL_SIZE", 24)), minimum=4, maximum=256)
_WEB_ACTIVE_REQUESTS = 0
_WEB_ACTIVE_REQUESTS_LOCK = threading.Lock()
_WEB_SLOW_REQUESTS = deque(maxlen=200)
_WEB_ADMIN_AUDIT = deque(maxlen=WEB_ADMIN_AUDIT_MAX)


@app.middleware("http")
async def _web_request_id_and_timing(req: FastAPIRequest, call_next):
    """Attach request id, latency, active-request count, and slow-request logs.

    Admin V5 adds lightweight performance telemetry without exposing secrets.
    It helps identify slow dashboard endpoints, Supabase pressure, and browser
    polling storms while keeping the response body unchanged.
    """
    global _WEB_ACTIVE_REQUESTS
    request_id = (req.headers.get("X-Request-ID") or secrets.token_hex(8)).strip()[:64]
    started = time.monotonic()
    with _WEB_ACTIVE_REQUESTS_LOCK:
        _WEB_ACTIVE_REQUESTS += 1
        active_now = _WEB_ACTIVE_REQUESTS
    try:
        resp = await call_next(req)
    except Exception:
        logging.getLogger(__name__).exception(
            "Unhandled web request error request_id=%s method=%s path=%s active=%s",
            request_id,
            req.method,
            req.url.path,
            active_now,
        )
        raise
    finally:
        with _WEB_ACTIVE_REQUESTS_LOCK:
            _WEB_ACTIVE_REQUESTS = max(0, _WEB_ACTIVE_REQUESTS - 1)

    elapsed_ms = (time.monotonic() - started) * 1000.0
    resp.headers.setdefault("X-Request-ID", request_id)
    resp.headers.setdefault("X-Response-Time-ms", f"{elapsed_ms:.1f}")
    resp.headers.setdefault("X-Active-Requests", str(active_now))

    # Telegram webhook updates can legitimately wait for handler processing.
    # Keep dashboard/API slow logs sensitive, but avoid noisy webhook warnings
    # unless the request is truly stuck.
    slow_threshold_ms = WEB_SLOW_REQUEST_MS
    if req.url.path.startswith("/tg-webhook-") or req.url.path == "/telegram-webhook":
        slow_threshold_ms = max(WEB_SLOW_REQUEST_MS, _env_float("WEBHOOK_SLOW_REQUEST_MS", 10_000.0, minimum=1000.0, maximum=120_000.0))

    if elapsed_ms >= slow_threshold_ms:
        item = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "method": req.method,
            "path": req.url.path,
            "status": int(getattr(resp, "status_code", 0) or 0),
            "elapsed_ms": round(elapsed_ms, 1),
            "active": int(active_now),
        }
        _WEB_SLOW_REQUESTS.appendleft(item)
        logging.getLogger(__name__).warning(
            "Slow web request request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f active=%s",
            request_id,
            req.method,
            req.url.path,
            item["status"],
            elapsed_ms,
            active_now,
        )
    return resp


@app.middleware("http")
async def _hot_api_rate_limit_middleware(req: FastAPIRequest, call_next):
    # Protect expensive public AI endpoints without touching admin dashboard or
    # Telegram webhook traffic. Telegram updates are throttled per chat/user in
    # the Telegram handler pipeline, not by shared Telegram source IP.
    if req.url.path.startswith("/ai-assistant"):
        if not await _api_rate_limit_allowed(req):
            return _FastJSONResponse({"ok": False, "error": "Too many requests. Please slow down.", "code": 429}, status_code=429)
    return await call_next(req)


@app_flask.after_request
def _web_security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data: https:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    if _default_cookie_secure():
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    try:
        if str(getattr(request, "path", "")).startswith("/admin"):
            resp.headers.setdefault("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            resp.headers.setdefault("Pragma", "no-cache")
    except Exception:
        pass
    return resp

@app_flask.route("/")
def health_check():
    return "Bot is running!", 200

@app_flask.route("/ping")
def ping():
    return "pong", 200


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app_flask.route("/readyz")
def readyz():
    """Public deployment health endpoint.

    Keep this intentionally minimal for Render/UptimeRobot. Detailed runtime,
    secret-source, CORS, cookie, Redis, webhook, and leader data is available
    only to authenticated admins at /api/admin/diagnostics.
    """
    return jsonify({
        "ok": True,
        "web": True,
        "status": "ready",
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def _admin_diagnostics_payload() -> dict[str, Any]:
    """Authenticated admin diagnostics; never expose this data on /readyz."""
    ffmpeg_path = globals().get("_FFMPEG_EXE", "")
    payload = {
        "web": True,
        "telegram_token_configured": bool(globals().get("TELEGRAM_BOT_TOKEN")),
        "supabase_configured": bool(globals().get("supabase")),
        "redis_configured": bool(globals().get("redis_client")),
        "ffmpeg_ok": bool(ffmpeg_path and os.path.exists(str(ffmpeg_path))),
        "web_secret_configured": _web_stable_secret_configured(),
        "web_secret_source": _web_session_secret_source(),
        "web_secret_redis_key": _web_secret_redis_key(),
        "frontend_allowed_origins": _frontend_allowed_origins(),
        "origin_guard_enabled": _admin_origin_guard_enabled(),
        "release": ADMIN_BACKEND_RELEASE,
        "bot_mode": _run_state_bot_mode() if "_run_state_bot_mode" in globals() else globals().get("BOT_MODE", "POLLING"),
        "webhook_ready": bool(globals().get("_TELEGRAM_APP")),
        "telegram_leader": _telegram_leader_snapshot() if "_telegram_leader_snapshot" in globals() else {},
    }
    cookie_policy = globals().get("_admin_cookie_policy_payload")
    if callable(cookie_policy):
        with suppress(Exception):
            payload["cookie_policy"] = cookie_policy()
    fmt = globals().get("_format_uptime")
    if callable(fmt):
        with suppress(Exception):
            payload["uptime"] = fmt()
    return payload


@app.get("/api-docs", include_in_schema=False)
async def api_docs_redirect():
    """Convenience alias for Swagger UI."""
    if not _env_bool("API_DOCS_ENABLED", False):
        return _FastJSONResponse({
            "ok": False,
            "error": "API docs are disabled. Set API_DOCS_ENABLED=true and restart.",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        }, status_code=404)
    return RedirectResponse(url="/docs", status_code=302)


@app.get("/swagger", include_in_schema=False)
async def swagger_redirect():
    """Convenience alias for Swagger UI."""
    return await api_docs_redirect()


def _runtime_webhook_secret_token() -> str:
    """Return the active Telegram webhook secret without exposing it in logs.

    RUN_STATE is authoritative so in-bot secret rotation takes effect
    immediately for webhook path validation and setWebhook registration.
    """
    run_state = globals().get("RUN_STATE")
    if isinstance(run_state, dict):
        value = run_state.get("TELEGRAM_WEBHOOK_SECRET_TOKEN")
        if value:
            return str(value).strip()
    return str(
        globals().get("TELEGRAM_WEBHOOK_SECRET_TOKEN")
        or getattr(SETTINGS, "TELEGRAM_WEBHOOK_SECRET_TOKEN", "")
        or ""
    ).strip()


def generate_new_webhook_token() -> str:
    """Generate a new 32-character URL-safe Telegram webhook secret token."""
    token = secrets.token_urlsafe(24)
    if len(token) < 32:
        token = (token + secrets.token_urlsafe(24))[:32]
    return token[:32]


def _runtime_webhook_base_url() -> str:
    """Return this Render service's public webhook base URL.

    In two-server mode, TELEGRAM_WEBHOOK_URL / RENDER_EXTERNAL_URL is
    intentionally treated as service-local.  Redis RUN_STATE may contain the
    other Render service's URL, so local env must win before persisted state.
    """
    env_value = str(
        os.environ.get("TELEGRAM_WEBHOOK_URL")
        or getattr(SETTINGS, "TELEGRAM_WEBHOOK_URL", "")
        or os.environ.get("RENDER_EXTERNAL_URL")
        or getattr(SETTINGS, "RENDER_EXTERNAL_URL", "")
        or ""
    ).strip().rstrip("/")
    if bool(globals().get("TELEGRAM_MULTI_SERVER_ENABLED", False)) and env_value:
        return env_value

    run_state = globals().get("RUN_STATE")
    if isinstance(run_state, dict):
        value = run_state.get("TELEGRAM_WEBHOOK_URL")
        if value:
            return str(value).strip().rstrip("/")
    return env_value or str(globals().get("TELEGRAM_WEBHOOK_URL") or "").strip().rstrip("/")


def _telegram_webhook_target_url_for_secret(secret_token: str) -> str:
    """Return the canonical webhook URL for an explicit secret token.

    This helper is intentionally separate from _runtime_webhook_secret_token()
    so secret rotation can safely call Telegram setWebhook with a new token
    before persisting that token into Redis/runtime state.  That prevents a
    failed setWebhook call from making the app reject Telegram's still-active
    old webhook path with 403 Invalid path secret.
    """
    base_url = _runtime_webhook_base_url()
    secret_token = str(secret_token or "").strip()
    if not base_url:
        raise RuntimeError("BOT_MODE=WEBHOOK requires TELEGRAM_WEBHOOK_URL.")
    if not secret_token:
        raise RuntimeError("BOT_MODE=WEBHOOK requires TELEGRAM_WEBHOOK_SECRET_TOKEN.")
    return f"{base_url}/tg-webhook-{quote(secret_token, safe='')}"


def _telegram_webhook_target_url() -> str:
    """Canonical public Telegram webhook URL registered with setWebhook.

    Telegram calls this exact route.  The token is URL-escaped for path safety
    while the Bot API secret header still carries the raw token value.
    """
    return _telegram_webhook_target_url_for_secret(_runtime_webhook_secret_token())


def _telegram_allowed_updates() -> list[str]:
    """Return setWebhook allowed_updates from env/runtime safely.

    Defaults to the update types this bot actually handles. Operators can add
    edited_message, my_chat_member, etc. with TELEGRAM_ALLOWED_UPDATES without
    touching code.
    """
    raw = str(
        os.environ.get("TELEGRAM_ALLOWED_UPDATES")
        or getattr(SETTINGS, "TELEGRAM_ALLOWED_UPDATES", "message,callback_query")
        or "message,callback_query"
    )
    allowed = []
    for item in raw.split(","):
        item = item.strip()
        if item and re.fullmatch(r"[a-z_]+", item) and item not in allowed:
            allowed.append(item)
    return allowed or ["message", "callback_query"]


_WEBHOOK_UPDATE_MEMORY: OrderedDict[int, float] = OrderedDict()
_WEBHOOK_UPDATE_MEMORY_LOCK = asyncio.Lock()


def _webhook_replay_ttl_seconds() -> int:
    return _env_int("WEBHOOK_REPLAY_TTL_S", 600, minimum=60, maximum=86400)


async def _telegram_webhook_update_seen_or_mark(update_id: Any) -> bool:
    """Return True if Telegram update_id was already accepted recently.

    Telegram may retry webhooks after network errors.  This guard prevents
    duplicate processing/sending while keeping a bounded TTL. Redis gives
    cross-instance protection; memory fallback protects single-instance runs.
    """
    try:
        uid = int(update_id)
    except Exception:
        return False

    ttl = _webhook_replay_ttl_seconds()
    redis_obj = globals().get("redis_client")
    if redis_obj is not None:
        key = _redis_key("tg_update_seen", uid) if "_redis_key" in globals() else f"tg_update_seen:{uid}"
        try:
            ok = await asyncio.to_thread(lambda: redis_obj.set(key, "1", ex=ttl, nx=True))
            return not bool(ok)
        except Exception as exc:
            _log_once(logging.WARNING, "webhook_replay_redis_fallback", "Webhook replay Redis fallback: %s", exc)

    now = time.monotonic()
    async with _WEBHOOK_UPDATE_MEMORY_LOCK:
        # Trim old entries opportunistically.
        stale_before = now - ttl
        for old_uid, old_ts in list(_WEBHOOK_UPDATE_MEMORY.items())[:5000]:
            if old_ts < stale_before:
                _WEBHOOK_UPDATE_MEMORY.pop(old_uid, None)
            else:
                break
        if uid in _WEBHOOK_UPDATE_MEMORY:
            _WEBHOOK_UPDATE_MEMORY.move_to_end(uid)
            return True
        _WEBHOOK_UPDATE_MEMORY[uid] = now
        while len(_WEBHOOK_UPDATE_MEMORY) > 50_000:
            _WEBHOOK_UPDATE_MEMORY.popitem(last=False)
        return False


async def _read_limited_webhook_body(req: FastAPIRequest, max_body: int) -> bytes:
    """Read Telegram webhook payloads with a hard body cap.

    Content-Length is rejected before body streaming.  The chunked stream path
    also enforces the cap so a missing/lying header cannot flood memory.
    """
    content_length = req.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_body:
                raise HTTPException(status_code=413, detail=f"Request body too large. Max {max_body} bytes.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

    chunks: list[bytes] = []
    total = 0
    async for chunk in req.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_body:
            raise HTTPException(status_code=413, detail=f"Request body too large. Max {max_body} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


async def _process_telegram_webhook_request(req: FastAPIRequest, path_secret_token: str | None = None):
    """Validate, parse, and dispatch a Telegram webhook update.

    Security errors intentionally return 403.  Internal parse/handler failures
    are logged and acknowledged with HTTP 200 so Telegram does not repeatedly
    redeliver a bad or already-accepted update forever.
    """
    if "_run_state_bot_mode" in globals() and _run_state_bot_mode() != "WEBHOOK":
        webhook_logger.info("Webhook update ignored because BOT_MODE=%s.", _run_state_bot_mode())
        return _FastJSONResponse({"status": "ignored", "reason": "not_webhook_mode"}, status_code=200)

    expected_secret = _runtime_webhook_secret_token()
    if not expected_secret:
        webhook_logger.error("Rejected Telegram webhook request because TELEGRAM_WEBHOOK_SECRET_TOKEN is not configured.")
        raise HTTPException(status_code=503, detail="Telegram webhook secret is not configured.")

    if path_secret_token is not None and not hmac.compare_digest(str(path_secret_token), expected_secret):
        webhook_logger.warning("Rejected Telegram webhook request with invalid path secret from %s", req.client.host if req.client else "unknown")
        raise HTTPException(status_code=403, detail="Invalid webhook path secret.")

    got_secret = (req.headers.get("X-Telegram-Bot-Api-Secret-Token") or "").strip()
    if not hmac.compare_digest(got_secret, expected_secret):
        webhook_logger.warning("Rejected Telegram webhook request with invalid secret header from %s", req.client.host if req.client else "unknown")
        raise HTTPException(status_code=403, detail="Invalid webhook secret.")

    app_obj = globals().get("telegram_application") or globals().get("_TELEGRAM_APP")
    if app_obj is None or not bool(globals().get("_TELEGRAM_APP_READY", False)):
        # Do NOT acknowledge startup-time updates. Returning 503 tells Telegram
        # to retry instead of losing user messages while the Render service,
        # Supabase, Redis, or python-telegram-bot application is still starting.
        webhook_logger.warning("Telegram webhook rejected with 503 because application is not ready yet.")
        raise HTTPException(status_code=503, detail="Telegram application is starting. Please retry.")

    if not _telegram_should_process_webhook_update():
        webhook_logger.info("Webhook update acknowledged by standby instance; active owner=%s", _telegram_leader_snapshot().get("owner"))
        return _FastJSONResponse({"status": "ignored", "reason": "standby_instance"}, status_code=200)

    max_body = min(_web_max_content_length(), 2 * 1024 * 1024)
    try:
        raw = await _read_limited_webhook_body(req, max_body)
        data = _json_loads_fast(raw)
        update = Update.de_json(data, app_obj.bot)
        if await _telegram_webhook_update_seen_or_mark(getattr(update, "update_id", None)):
            webhook_logger.info("Duplicate Telegram webhook update ignored update_id=%s", getattr(update, "update_id", None))
            _metric_inc("replay_dropped")
            return _FastJSONResponse({"status": "ok", "duplicate": True}, status_code=200)
        await app_obj.process_update(update)
    except HTTPException:
        raise
    except Exception as exc:
        webhook_logger.error("Telegram webhook ingestion failed but was acknowledged: %s", exc, exc_info=True)
        return _FastJSONResponse({"status": "ok", "warning": "processing_failed"}, status_code=200)

    return _FastJSONResponse({"status": "ok"}, status_code=200)


@app.post("/tg-webhook-{secret_token}", include_in_schema=False)
async def telegram_webhook_ingest(secret_token: str, req: FastAPIRequest):
    return await _process_telegram_webhook_request(req, secret_token)


@app.post("/telegram-webhook", include_in_schema=False)
async def telegram_webhook(req: FastAPIRequest):
    """Backward-compatible webhook route kept for older deployments.

    New deployments should use /tg-webhook-{secret_token}; setWebhook now
    registers that canonical route automatically.
    """
    return await _process_telegram_webhook_request(req, None)


async def _configure_telegram_webhook_via_http_for_secret(secret_token: str) -> None:
    """Configure Telegram webhook for an explicit secret token with 429 retry.

    Used by rotation so the remote Telegram webhook is updated first.  Only
    after this function succeeds should the new token be saved to RUN_STATE.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    secret_token = str(secret_token or "").strip()
    target_url = _telegram_webhook_target_url_for_secret(secret_token)

    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
    payload = {
        "url": target_url,
        "allowed_updates": _telegram_allowed_updates(),
        # Keep pending updates by default so a Render restart/deploy does not
        # silently lose user messages. Set TELEGRAM_WEBHOOK_DROP_PENDING_UPDATES=true
        # only when intentionally clearing a broken backlog.
        "drop_pending_updates": _env_bool("TELEGRAM_WEBHOOK_DROP_PENDING_UPDATES", False),
        "secret_token": secret_token,
    }
    timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)
    max_attempts = _env_int("TELEGRAM_SETWEBHOOK_MAX_RETRIES", 3, minimum=1, maximum=10)
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        async with httpx.AsyncClient(timeout=timeout, limits=_make_httpx_limits_from_run_state()) as client:
            resp = await client.post(api_url, json=payload)
        try:
            data = _json_loads_fast(resp.content)
        except Exception:
            data = {"ok": False, "description": resp.text[:500]}

        if resp.status_code == 429:
            retry_after = 1
            try:
                retry_after = int((data.get("parameters") or {}).get("retry_after") or 1)
            except Exception:
                retry_after = 1
            last_error = f"Telegram setWebhook rate-limited status=429 response={str(data)[:500]}"
            if attempt < max_attempts:
                webhook_logger.warning(
                    "Telegram setWebhook rate-limited; retrying after %ss attempt=%s/%s",
                    retry_after,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(max(1, retry_after))
                continue

        if resp.status_code >= 400 or not bool(data.get("ok")):
            raise RuntimeError(f"Telegram setWebhook failed status={resp.status_code} response={str(data)[:500]}")

        webhook_logger.info(
            "Telegram webhook configured via HTTPX url=%s max_connections=%s keepalive=%s",
            target_url,
            _run_state_http_max_connections(),
            min(max(2, int(HTTP_MAX_KEEPALIVE_CONNECTIONS or 20)), _run_state_http_max_connections()),
        )
        return

    raise RuntimeError(last_error or "Telegram setWebhook rate-limited after retries.")


async def _configure_telegram_webhook_via_http() -> None:
    """Configure Telegram webhook using the currently persisted runtime token."""
    await _configure_telegram_webhook_via_http_for_secret(_runtime_webhook_secret_token())


async def set_telegram_webhook() -> None:
    """Public compatibility wrapper used by runtime admin orchestration."""
    await _configure_telegram_webhook_via_http()


# Webhook secret rotation guard.  Telegram rate-limits setWebhook calls, and
# rapid repeated rotation can desynchronise Telegram's registered path from the
# app's Redis RUN_STATE secret.  Keep rotation serialized and cooled down.
_WEBHOOK_ROTATE_ASYNC_LOCK: asyncio.Lock | None = None
_WEBHOOK_ROTATE_SYNC_LOCK = threading.Lock()
_WEBHOOK_ROTATE_IN_PROGRESS = False
_WEBHOOK_LAST_ROTATE_AT = 0.0


def _webhook_rotate_cooldown_seconds() -> float:
    return _env_float("WEBHOOK_ROTATE_COOLDOWN_S", 60.0, minimum=5.0, maximum=3600.0)


def _webhook_rotate_lock() -> asyncio.Lock:
    global _WEBHOOK_ROTATE_ASYNC_LOCK
    if _WEBHOOK_ROTATE_ASYNC_LOCK is None:
        _WEBHOOK_ROTATE_ASYNC_LOCK = asyncio.Lock()
    return _WEBHOOK_ROTATE_ASYNC_LOCK


def _webhook_rotate_cooldown_remaining() -> float:
    elapsed = time.monotonic() - float(globals().get("_WEBHOOK_LAST_ROTATE_AT", 0.0) or 0.0)
    return max(0.0, _webhook_rotate_cooldown_seconds() - elapsed)


def _webhook_rotate_begin_or_remaining() -> float:
    """Reserve one rotation attempt.

    Returns:
      0.0  -> reservation acquired
      >0.0 -> cooldown seconds remaining
      -1.0 -> another rotation is already running
    """
    global _WEBHOOK_ROTATE_IN_PROGRESS
    with _WEBHOOK_ROTATE_SYNC_LOCK:
        if _WEBHOOK_ROTATE_IN_PROGRESS:
            return -1.0
        remaining = _webhook_rotate_cooldown_remaining()
        if remaining > 0:
            return remaining
        _WEBHOOK_ROTATE_IN_PROGRESS = True
        return 0.0


def _webhook_rotate_finish(success: bool) -> None:
    global _WEBHOOK_ROTATE_IN_PROGRESS, _WEBHOOK_LAST_ROTATE_AT
    with _WEBHOOK_ROTATE_SYNC_LOCK:
        if success:
            _WEBHOOK_LAST_ROTATE_AT = time.monotonic()
        _WEBHOOK_ROTATE_IN_PROGRESS = False


async def run_fastapi():
    """Run the FastAPI dashboard inside the main asyncio runtime."""
    import uvicorn
    port = _env_int("PORT", 8080, minimum=1, maximum=65535)
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info").lower(),
        proxy_headers=_env_bool("WEB_TRUST_PROXY", _env_bool("RENDER", False)),
        forwarded_allow_ips="*" if _env_bool("WEB_TRUST_PROXY", _env_bool("RENDER", False)) else None,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def keep_alive_async(stop_event: asyncio.Event | None = None):
    """Self-ping every 4 min with non-blocking HTTPX instead of requests."""
    logger_ka = logging.getLogger("keep_alive")
    render_url = (os.environ.get("RENDER_EXTERNAL_URL") or getattr(SETTINGS, "RENDER_EXTERNAL_URL", "") or "").strip().rstrip("/")
    if not render_url:
        logger_ka.warning("RENDER_EXTERNAL_URL not set — self-ping disabled.")
        return

    await asyncio.sleep(10)
    headers = {"User-Agent": "Mozilla/5.0 (AsyncKeepAlive/2.0)", "Accept": "text/plain,*/*"}

    async with httpx.AsyncClient(timeout=10, follow_redirects=True, limits=_make_httpx_limits_from_run_state()) as client:
        while stop_event is None or not stop_event.is_set():
            urls = [render_url]
            if not render_url.endswith("/ping"):
                urls.append(f"{render_url}/ping")

            ok = False
            for url in urls:
                try:
                    r = await client.get(url, headers=headers)
                    if 200 <= r.status_code < 400:
                        logger_ka.info(f"Keep-alive OK -> {r.status_code} {url}")
                        ok = True
                        break
                    logger_ka.warning(f"Keep-alive non-OK -> {r.status_code} {url}")
                except Exception as e:
                    logger_ka.warning(f"Keep-alive failed for {url}: {e}")

            if not ok:
                logger_ka.warning("Keep-alive failed for all URLs. Check RENDER_EXTERNAL_URL.")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=240) if stop_event else await asyncio.sleep(240)
            except asyncio.TimeoutError:
                pass


def run_flask():
    """Backward-compatible alias. Prefer run_fastapi()."""
    asyncio.run(run_fastapi())

def keep_alive():
    """Backward-compatible alias. Prefer keep_alive_async()."""
    asyncio.run(keep_alive_async())

# ── AI Assistant REST API ──────────────────────────────────────────────────
import base64

try:
    from huggingface_hub import InferenceClient
except Exception as exc:
    logging.getLogger(__name__).warning("Optional dependency huggingface_hub is not available; Hugging Face inference disabled: %s", exc)
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
MAX_CONCURRENT_TTS_USERS   = _env_int("MAX_CONCURRENT_TTS_USERS", int(_perf_default("MAX_CONCURRENT_TTS_USERS", 2)), minimum=1, maximum=50)
MAX_CONCURRENT_AI          = _env_int("MAX_CONCURRENT_AI", _env_int("MAX_CONCURRENT_GEMINI", 3, minimum=1), minimum=1, maximum=50)
MAX_CONCURRENT_BROADCAST   = _env_int("MAX_CONCURRENT_BROADCAST", 3, minimum=1, maximum=50)
BROADCAST_BATCH_SIZE       = _env_int("BROADCAST_BATCH_SIZE", MAX_CONCURRENT_BROADCAST, minimum=1, maximum=500)
BROADCAST_INTER_BATCH_DELAY = _env_float("BROADCAST_INTER_BATCH_DELAY", 0.20, minimum=0.0, maximum=30.0)
TTS_RESOLVER_AI_ENABLED    = _env_bool("TTS_RESOLVER_AI_ENABLED", False)

# ---------------------------------------------------------------------------
# Subtitle/Text document support
# ---------------------------------------------------------------------------
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".txt"}
MAX_SUBTITLE_BYTES  = 2 * 1024 * 1024
MAX_SUBTITLE_CHARS  = 20_000


def _ai_error(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg, "code": code}), code


def _ai_allowed_origin() -> str:
    """Return a safe CORS origin for public AI API responses.

    Default is same-origin only (no wildcard).  Set AI_API_ALLOWED_ORIGINS to a
    comma-separated allowlist, or to * only if you intentionally expose the API
    to browser clients from any origin.
    """
    origin = (request.headers.get("Origin") or "").strip()
    raw = str(os.environ.get("AI_API_ALLOWED_ORIGINS") or getattr(SETTINGS, "AI_API_ALLOWED_ORIGINS", "") or "").strip()
    if not raw or not origin:
        return ""
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    if "*" in allowed:
        return "*"
    return origin if origin in allowed else ""


def _apply_ai_cors_headers(resp):
    allowed_origin = _ai_allowed_origin()
    if allowed_origin:
        resp.headers["Access-Control-Allow-Origin"] = allowed_origin
        resp.headers["Vary"] = "Origin"
    resp.headers.setdefault("Access-Control-Allow-Methods", "POST, OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Api-Key")
    resp.headers.setdefault("Access-Control-Max-Age", "600")
    return resp


def _ai_cors(response_or_tuple):
    if isinstance(response_or_tuple, tuple):
        resp, code = response_or_tuple
        return _apply_ai_cors_headers(resp), code
    return _apply_ai_cors_headers(response_or_tuple)


def _ai_cors_preflight():
    return _apply_ai_cors_headers(Response("", status=204))


def _detect_lang(text: str) -> str:
    khmer = sum(1 for c in text if "\u1780" <= c <= "\u17FF")
    alpha = sum(1 for c in text if c.isalpha())
    return "km" if alpha and khmer / alpha > 0.25 else "en"


# ---------------------------------------------------------------------------
# Local time display / schedule input
# ---------------------------------------------------------------------------
# Keep database/scheduler timestamps in UTC, but show and accept admin-facing
# times in Phnom Penh local time by default: ICT, UTC+7, AM/PM format.
APP_TIMEZONE_NAME = (
    os.environ.get("APP_TIMEZONE")
    or os.environ.get("WEB_ADMIN_TIMEZONE")
    or "Asia/Phnom_Penh"
).strip() or "Asia/Phnom_Penh"
APP_TIMEZONE_ALIAS = (os.environ.get("APP_TIMEZONE_ALIAS") or "ICT").strip() or "ICT"
APP_TIMEZONE_UTC_LABEL = (os.environ.get("APP_TIMEZONE_UTC_LABEL") or "UTC+7").strip() or "UTC+7"


def _load_app_timezone():
    if ZoneInfo is not None:
        try:
            return ZoneInfo(APP_TIMEZONE_NAME)
        except Exception:
            pass
    # Safe fallback for Phnom Penh / Cambodia ICT if tzdata is unavailable.
    return timezone(timedelta(hours=7), APP_TIMEZONE_ALIAS)


APP_TIMEZONE = _load_app_timezone()


def _local_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def _to_local_time(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # Database values should normally be timezone-aware UTC. If a raw/naive
        # datetime reaches the UI, treat it as UTC to avoid double-shifting.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(APP_TIMEZONE)


def _local_to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=APP_TIMEZONE)
    return dt.astimezone(timezone.utc)


def _fmt_local_dt(dt: datetime | None = None) -> str:
    local_dt = _to_local_time(dt or datetime.now(timezone.utc))
    return f"{local_dt.strftime('%Y-%m-%d %I:%M %p')} {APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})"


def _fmt_local_time_hint() -> str:
    return f"Phnom Penh local time — AM/PM, {APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})"


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
_AI_API_KEY_VALIDATION_CACHE_TTL_S = _env_float("AI_API_KEY_CACHE_TTL_S", 60.0, minimum=1.0, maximum=3600.0)
_AI_API_KEY_TOUCH_INTERVAL_S = _env_float("AI_API_KEY_TOUCH_INTERVAL_S", 60.0, minimum=1.0, maximum=3600.0)
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
  ('ai_resolver_enabled', '1'),
  ('TELEGRAM_CONCURRENT_UPDATES', '4'),
  ('TELEGRAM_CONNECTION_POOL_SIZE', '24'),
  ('HTTP_MAX_CONNECTIONS', '100'),
  ('HTTP_MAX_KEEPALIVE_CONNECTIONS', '20'),
  ('DB_EXECUTOR_MAX_WORKERS', '3'),
  ('MAX_CONCURRENT_TTS_USERS', '2'),
  ('USER_SYNC_TTL_S', '1800.0'),
  ('PREFS_CACHE_TTL_S', '600.0'),
  ('TTS_AUDIO_CACHE_ENABLED', '1'),
  ('TTS_AUDIO_CACHE_MAX_MB', '64'),
  ('TTS_AUDIO_CACHE_ITEM_MAX_MB', '8'),
  ('TTS_AUDIO_CACHE_TTL_S', '1200.0'),
  ('EDGE_TTS_PARALLEL_CHUNKS', '1'),
  ('PAGED_TTS_SEND_DELAY_S', '0.10'),
  ('WEB_STATUS_POLL_SECONDS', '30'),
  ('WEB_LIVE_POLL_SECONDS', '30')
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
    """Extract AI API key without leaking credentials through URLs by default."""
    auth_header = (request.headers.get("Authorization") or "").strip()
    bearer_key = ""
    if auth_header.lower().startswith("bearer "):
        bearer_key = auth_header[7:].strip()
    key = (request.headers.get("X-Api-Key") or "").strip() or bearer_key
    if key:
        return key
    # Query-string API keys leak into logs/history/referrers. Keep disabled
    # unless explicitly enabled for old integrations during a temporary migration.
    if _env_bool("AI_API_QUERY_KEY_ENABLED", False):
        return (request.args.get("api_key") or "").strip()
    return ""


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
            history = _json_loads_fast(raw_history)
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
        return _ai_cors_preflight()

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
        return _ai_cors_preflight()

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
        return _ai_cors_preflight()

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
        "features": ["chat", "multi-turn-history", "image-ocr", "khmer-language", "english-language", "chinese-language", "admin-generated-api-keys"],
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

# ── FastAPI Web Admin Dashboard ───────────────────────────────────────────────
# Full-control web dashboard for the Telegram bot. It reuses the existing
# Supabase helpers, Redis state, Telegram token, bot settings, API-key manager,
# scheduled-broadcast functions, and distributed scheduler lock.
_WEB_BROADCAST_JOBS: OrderedDict[str, dict] = OrderedDict()
_WEB_BROADCAST_JOBS_LOCK = threading.Lock()
_WEB_BROADCAST_JOBS_MAX = _env_int("WEB_BROADCAST_JOBS_MAX", 50, minimum=10, maximum=2000)
WEB_BROADCAST_WORKERS = max(1, _env_int("WEB_BROADCAST_WORKERS", MAX_CONCURRENT_BROADCAST))
WEB_BROADCAST_DELAY_S = max(0.0, _env_float("WEB_BROADCAST_DELAY_S", 0.05))
WEB_BROADCAST_MAX_ACTIVE_JOBS = max(1, _env_int("WEB_BROADCAST_MAX_ACTIVE_JOBS", 2, minimum=1, maximum=10))
_WEB_BROADCAST_EXECUTOR = ThreadPoolExecutor(
    max_workers=WEB_BROADCAST_MAX_ACTIVE_JOBS,
    thread_name_prefix="web_broadcast_job",
)
_WEB_BROADCAST_QUEUE: asyncio.Queue | None = None
_WEB_BROADCAST_QUEUE_LOOP: asyncio.AbstractEventLoop | None = None
_WEB_BROADCAST_QUEUE_TASKS: list[asyncio.Task] = []
_WEB_BROADCAST_QUEUE_STOP = object()
WEB_TABLE_PAGE_SIZE = max(10, min(200, _env_int("WEB_TABLE_PAGE_SIZE", 50)))
# V4.1: avoid hammering Supabase from mobile/live dashboard polling.
WEB_COUNTS_CACHE_TTL_S = max(5.0, _env_float("WEB_COUNTS_CACHE_TTL_S", 30.0))
WEB_STATUS_POLL_SECONDS = max(10, _env_int("WEB_STATUS_POLL_SECONDS", int(_perf_default("WEB_STATUS_POLL_SECONDS", 30))))
WEB_LIVE_POLL_SECONDS = max(15, _env_int("WEB_LIVE_POLL_SECONDS", int(_perf_default("WEB_LIVE_POLL_SECONDS", 30))))
WEB_LIVE_SCHEDULES_CACHE_TTL_S = max(5.0, _env_float("WEB_LIVE_SCHEDULES_CACHE_TTL_S", 30.0))
_WEB_COUNTS_CACHE = {"ts": 0.0, "data": {}}
_WEB_COUNTS_CACHE_LOCK = threading.Lock()
_WEB_COUNTS_BUILD_LOCK = threading.Lock()
_WEB_LIVE_SCHEDULES_CACHE = {"ts": 0.0, "rows": []}
_WEB_LIVE_SCHEDULES_LOCK = threading.Lock()


def _web_admin_enabled() -> bool:
    # Keep the dashboard available by default, but never public.  Without
    # ADMIN_WEB_PASSWORD / WEB_ADMIN_PASSWORD, protected admin routes show a
    # locked page instead of auto-opening an admin session.
    return _env_bool("WEB_ADMIN_ENABLED", True)


def _web_admin_password() -> str:
    return (
        os.environ.get("ADMIN_WEB_PASSWORD", "").strip()
        or os.environ.get("WEB_ADMIN_PASSWORD", "").strip()
    )


def _web_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _web_current_admin_id() -> int:
    # Prefer the signed cookie session, but support the API-only bearer
    # fallback below so separated frontends can still hydrate /auth/me when a
    # browser refuses third-party/cross-site cookies.
    try:
        saved = _web_int(session.get("web_admin_id"), 0)
    except Exception:
        saved = 0
    if saved:
        return saved
    bearer_admin_id = _web_bearer_admin_id() if "_web_bearer_admin_id" in globals() else 0
    if bearer_admin_id:
        return int(bearer_admin_id)
    if ADMIN_IDS:
        return int(sorted(ADMIN_IDS)[0])
    return 0


def _web_valid_admin_id(admin_id: int) -> bool:
    """Return True only for explicitly configured Telegram admin IDs.

    Security fix: an empty ADMIN_IDS used to mean "allow any web login that
    knows ADMIN_WEB_PASSWORD".  In production that is too permissive.  The
    web dashboard is now locked until ADMIN_IDS is configured.
    """
    return bool(ADMIN_IDS) and int(admin_id or 0) in ADMIN_IDS


def _web_csrf_token() -> str:
    token = session.get("web_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["web_csrf_token"] = token
    return str(token)


def _admin_api_token_fallback_enabled() -> bool:
    return _env_bool("ADMIN_API_TOKEN_FALLBACK_ENABLED", True)


def _admin_api_token_ttl_seconds() -> int:
    # Production hardening: never silently stretch bearer fallback tokens to the
    # full cookie/session lifetime.  Cookie auth remains the primary path; the
    # bearer is only a short-lived bridge for separated frontends and local dev.
    return _env_int("ADMIN_API_TOKEN_TTL_SECONDS", 3600, minimum=60, maximum=24 * 86400)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    data = str(data or "")
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _admin_api_token_secret() -> bytes:
    # Derive a dedicated signing key from the web session secret instead of
    # reusing it directly.  Token verification automatically fails after
    # WEB_SECRET_KEY rotation/restart, matching admin session behavior.
    secret = _web_session_secret_key().encode("utf-8")
    return hmac.new(secret, b"bot-admin-api-token-v1", hashlib.sha256).digest()


def _admin_issue_api_token(admin_id: int) -> tuple[str | None, int]:
    if not _admin_api_token_fallback_enabled():
        return None, 0
    admin_id = int(admin_id or 0)
    if not _web_valid_admin_id(admin_id):
        return None, 0
    now = int(time.time())
    ttl = _admin_api_token_ttl_seconds()
    payload = {
        "v": 1,
        "admin_id": admin_id,
        "iat": now,
        "exp": now + ttl,
        "nonce": secrets.token_urlsafe(16),
    }
    body = _b64url_encode(_json_dumps_fast(payload))
    sig = _b64url_encode(hmac.new(_admin_api_token_secret(), body.encode("ascii"), hashlib.sha256).digest())
    return f"v1.{body}.{sig}", ttl


def _admin_verify_api_token(token: str) -> int | None:
    if not _admin_api_token_fallback_enabled():
        return None
    token = str(token or "").strip()
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        return None
    _version, body, got_sig = parts
    expected_sig = _b64url_encode(hmac.new(_admin_api_token_secret(), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(got_sig, expected_sig):
        return None
    try:
        payload = _json_loads_fast(_b64url_decode(body))
        admin_id = int(payload.get("admin_id") or 0)
        exp = int(payload.get("exp") or 0)
    except Exception:
        return None
    if exp < int(time.time()):
        return None
    if not _web_valid_admin_id(admin_id):
        return None
    return admin_id


def _web_bearer_token_from_request() -> str:
    try:
        auth = str(request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            return auth.split(None, 1)[1].strip()
        # Convenience for fetch clients that cannot set Authorization during an
        # early migration.  Prefer Authorization: Bearer in new frontends.
        header_token = str(request.headers.get("x-admin-session-token") or "").strip()
        return header_token
    except Exception:
        return ""


def _web_bearer_admin_id() -> int:
    token = _web_bearer_token_from_request()
    if not token:
        return 0
    admin_id = _admin_verify_api_token(token)
    return int(admin_id or 0)


def _web_using_bearer_auth() -> bool:
    return bool(_web_bearer_admin_id())


def _web_check_csrf() -> None:
    # Cookie-authenticated browser writes require CSRF.  Bearer fallback writes
    # are allowed without CSRF because browsers do not attach Authorization
    # headers automatically across origins.
    if _web_using_bearer_auth():
        return
    expected = str(session.get("web_csrf_token") or "")
    got = str(request.form.get("csrf_token") or request.headers.get("X-CSRF-Token") or "")
    if not expected or not got or not hmac.compare_digest(expected, got):
        abort(403)


def _web_h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _web_short(value: Any, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _web_badge(label: str, kind: str = "muted") -> str:
    return f'<span class="badge badge-{_web_h(kind)}">{_web_h(label)}</span>'


def _web_status_badge(status: Any, row: dict | None = None) -> str:
    if row and _sched_is_draft(row):
        return _web_badge("preview", "warn")
    st = str(status or "unknown").lower().strip()
    kind = {
        "pending": "ok",
        "sending": "info",
        "done": "ok",
        "failed": "danger",
        "cancelled": "muted",
        "running": "info",
        "paused": "warn",
        "cancelling": "danger",
        "skipped": "muted",
        "queued": "warn",
    }.get(st, "muted")
    return _web_badge(st, kind)


def _web_dt(value: Any) -> str:
    """Format timestamps for the web dashboard in Phnom Penh local time."""
    if not value:
        return ""
    dt = value if isinstance(value, datetime) else _sched_parse_iso(value)
    if dt:
        return _fmt_local_dt(dt)
    return _web_short(value, 32)


def _web_dt_input_value(value: Any) -> str:
    """Return a datetime-local value in Phnom Penh local time for admin forms."""
    dt = _sched_parse_iso(value) if value else None
    if not dt:
        return ""
    return _to_local_time(dt).strftime("%Y-%m-%dT%H:%M")


def _web_url(endpoint: str, **params: Any) -> str:
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or value == "":
            continue
        clean[str(key)] = value
    base = endpoint if endpoint.startswith("/") else url_for(endpoint)
    return base + (("?" + urlencode(clean, doseq=True)) if clean else "")


def _web_safe_return(default_endpoint: str = "web_admin_home") -> str:
    target = str(request.form.get("return_to") or request.referrer or "").strip()
    if target.startswith("/") and not target.startswith("//"):
        return target
    return url_for(default_endpoint)


def _web_safe_next_url(target: str | None, default_endpoint: str = "web_admin_home") -> str:
    target = str(target or "").strip()
    if target.startswith("/") and not target.startswith("//"):
        return target
    return url_for(default_endpoint)


def _web_return_input() -> str:
    return f"<input type='hidden' name='return_to' value='{_web_h(request.full_path)}'>"


def _web_progress_bar(done: Any, total: Any) -> str:
    try:
        done_i = max(0, int(done or 0))
        total_i = max(0, int(total or 0))
    except Exception:
        done_i = total_i = 0
    pct = int((done_i / total_i) * 100) if total_i else 0
    pct = max(0, min(100, pct))
    return f"<div class='progress' title='{done_i}/{total_i}'><span style='width:{pct}%'></span></div><span class='muted'>{pct}%</span>"


def _web_blocked_ids_for_users(users: list[Any]) -> set[int]:
    """Batch-load blocked status to avoid one Supabase query per table row/user."""
    ids: list[int] = []
    seen_ids: set[int] = set()
    for item in users or []:
        raw = item.get("user_id") if isinstance(item, dict) else item
        try:
            uid = int(raw)
        except Exception:
            continue
        if uid not in seen_ids:
            seen_ids.add(uid)
            ids.append(uid)

    if not ids:
        return set()

    blocked = {uid for uid in ids if uid in _blocked_users_memory}
    unknown: list[int] = []
    for uid in ids:
        cached = _blocked_cache_get(uid)
        if cached is True:
            blocked.add(uid)
        elif cached is None:
            unknown.append(uid)

    if not unknown or not supabase:
        for uid in unknown:
            _blocked_cache_set(uid, uid in blocked)
        return blocked

    try:
        found: set[int] = set()
        chunk_size = 200
        for start in range(0, len(unknown), chunk_size):
            chunk = unknown[start:start + chunk_size]
            res = db_call_sync(
                f"web_blocked_batch:{start}",
                lambda c=chunk: supabase.table("blocked_users").select("user_id").in_("user_id", c).execute(),
                default=None,
                attempts=2,
                critical=False,
            )
            for row in list(getattr(res, "data", None) or []):
                try:
                    found.add(int(row.get("user_id")))
                except Exception:
                    pass
        blocked.update(found)
        for uid in unknown:
            _blocked_cache_set(uid, uid in blocked)
        return blocked
    except Exception as exc:
        logger.warning("web blocked batch lookup failed; falling back to per-user cache: %s", exc)
        return {uid for uid in ids if db_user_is_blocked(uid)}


def _web_status_card(label: str, value: Any, hint: str = "", kind: str = "") -> str:
    kind_cls = f" stat-{kind}" if kind else ""
    return (
        f"<div class='stat{kind_cls}'><span class='muted'>{_web_h(label)}</span>"
        f"<b>{_web_h(value)}</b>"
        f"<small>{_web_h(hint)}</small></div>"
    )


def _web_admin_feature_groups() -> list[tuple[str, list[tuple[str, str, str, str, str]]]]:
    """Central feature map used by the Admin Center launcher.

    Keeping this list in one place makes the dashboard feel consistent and
    prevents feature links from drifting across pages as the admin system grows.
    """
    return [
        ("Operate", [
            ("Live Dashboard", "/admin", "⌘", "Realtime counts, schedules, jobs, and system status.", "dashboard"),
            ("Broadcast", "/admin/broadcast", "📣", "Send, queue, pause, resume, and inspect broadcast jobs.", "broadcast"),
            ("Schedules", "/admin/schedules", "⏱", "Create, preview, edit, cancel, and monitor scheduled sends.", "schedules"),
            ("Calendar", "/admin/schedules/calendar", "🗓", "View upcoming scheduled messages in a calendar-style flow.", "calendar"),
        ]),
        ("Understand", [
            ("Analytics", "/admin/analytics", "📈", "Usage trends, recent activity, and delivery performance.", "analytics"),
            ("Insights", "/admin/insights", "💡", "Smart warnings, roadmap ideas, and next-feature recommendations.", "insights"),
            ("Error Inbox", "/admin/errors", "🚨", "Group repeated errors, mute noise, copy logs, and see recommended fixes.", "errors"),
            ("Report", "/admin/report", "📄", "Generate and download an admin PDF system report.", "report"),
            ("Users", "/admin/users", "👥", "Search users, inspect profile details, and manage access.", "users"),
            ("CRM", "/admin/crm", "⭐", "Labels, notes, trust status, and CSV export for user care.", "crm"),
            ("Health", "/admin/health", "🩺", "Deployment checks, runtime metrics, cache and database status.", "health"),
        ]),
        ("Control", [
            ("Optimize", "/admin/optimize", "⚡", "Cleanup, cache refresh, and safe performance tuning actions.", "optimize"),
            ("Control Center", "/admin/control", "🛡️", "Maintenance mode, polling/webhook mode, and protected controls.", "control"),
            ("Runtime", "/admin/runtime", "🛠️", "Live runtime knobs for rate limits, pools, and bot mode.", "runtime"),
            ("Settings", "/admin/settings", "⚙", "Feature toggles for TTS, OCR, AI resolver, and admin behavior.", "settings"),
            ("API Keys", "/admin/api-keys", "🔑", "Create, view, and revoke AI assistant API access keys.", "api"),
            ("Locks", "/admin/locks", "🔒", "Inspect or release distributed scheduler locks safely.", "locks"),
            ("SQL Setup", "/admin/sql", "☷", "Copy required Supabase schema and setup instructions.", "sql"),
        ]),
    ]


def _web_feature_launcher_html(active: str = "dashboard", compact: bool = False) -> str:
    """Render a touch-friendly launcher for every admin feature."""
    groups: list[str] = []
    for title, items in _web_admin_feature_groups():
        cards: list[str] = []
        for label, url, icon, desc, key in items:
            active_cls = " feature-card-active" if key == active else ""
            cards.append(
                "<a class='feature-card%s' href='%s'>"
                "<span class='feature-icon'>%s</span>"
                "<span><b>%s</b><small>%s</small></span>"
                "<em>Open</em></a>"
                % (
                    active_cls,
                    _web_h(url),
                    _web_h(icon),
                    _web_h(label),
                    _web_h(desc),
                )
            )
        groups.append(
            "<section class='feature-section'><div class='section-title'><h2>%s</h2>"
            "<span class='muted'>%s tools</span></div><div class='feature-grid%s'>%s</div></section>"
            % (
                _web_h(title),
                len(items),
                " feature-grid-compact" if compact else "",
                "".join(cards),
            )
        )
    return "<div class='feature-launcher'>" + "".join(groups) + "</div>"


def _web_command_hero_html(counts: dict[str, Any]) -> str:
    """Hero summary shown at the top of Admin Center."""
    total_jobs = int(counts.get("pending") or 0) + int(counts.get("sending") or 0) + int(counts.get("failed") or 0)
    return f"""
    <div class='command-hero'>
      <div>
        <span class='hero-kicker'><span class='v3-live-dot'></span>Admin System Center</span>
        <h2>Control every bot feature from one smooth dashboard.</h2>
        <p>Fast access to broadcasts, schedules, users, CRM, settings, runtime controls, health checks, locks, and SQL setup.</p>
        <div class='actions hero-actions'>
          <a class='btn' href='/admin/broadcast'>New Broadcast</a>
          <a class='btn secondary' href='/admin/users'>Search Users</a>
          <a class='btn ghost' href='/admin/insights'>Insights</a>
          <a class='btn ghost' href='/admin/errors'>Error Inbox</a>
          <a class='btn ghost' href='/admin/optimize'>Optimize Now</a>
        </div>
      </div>
      <div class='hero-panel'>
        <div><span class='muted'>Users</span><b data-count='users'>{_web_h(counts.get('users', 0))}</b></div>
        <div><span class='muted'>Jobs</span><b>{_web_h(total_jobs)}</b></div>
        <div><span class='muted'>Blocked</span><b data-count='blocked'>{_web_h(counts.get('blocked', 0))}</b></div>
        <div><span class='muted'>API Keys</span><b data-count='api_keys'>{_web_h(counts.get('api_keys', 0))}</b></div>
      </div>
    </div>
    """


def _web_render(title: str, body: str, *, active: str = "dashboard", status_code: int = 200):
    if _backend_only_mode():
        return _legacy_dashboard_disabled_payload(title, status_code=status_code if status_code >= 400 else 410)
    csrf = _web_csrf_token() if session.get("web_admin_ok") else ""
    nav = [
        ("dashboard", "Admin Center", "/admin", "⌘"),
        ("optimize", "Optimize", "/admin/optimize", "⚡"),
        ("analytics", "Analytics", "/admin/analytics", "📈"),
        ("insights", "Insights", "/admin/insights", "💡"),
        ("errors", "Errors", "/admin/errors", "🚨"),
        ("report", "Report", "/admin/report", "📄"),
        ("users", "Users", "/admin/users", "👥"),
        ("crm", "CRM", "/admin/crm", "⭐"),
        ("schedules", "Schedules", "/admin/schedules", "⏱"),
        ("calendar", "Calendar", "/admin/schedules/calendar", "🗓"),
        ("broadcast", "Broadcast", "/admin/broadcast", "📣"),
        ("health", "Health", "/admin/health", "🩺"),
        ("control", "Control", "/admin/control", "🛡️"),
        ("runtime", "Runtime", "/admin/runtime", "🛠️"),
        ("settings", "Settings", "/admin/settings", "⚙"),
        ("api", "API Keys", "/admin/api-keys", "🔑"),
        ("locks", "Locks", "/admin/locks", "🔒"),
        ("sql", "SQL Setup", "/admin/sql", "☷"),
    ]
    bottom_nav = [item for item in nav if item[0] in {"dashboard", "insights", "errors", "broadcast", "optimize"}]
    template = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0f172a">
<title>{{ title }} - Bot Admin</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: { sans: ['Inter','ui-sans-serif','system-ui','-apple-system','Segoe UI','Roboto','Arial','Noto Sans Khmer','sans-serif'] },
      boxShadow: { soft: '0 20px 60px rgba(15,23,42,.28)' }
    }
  }
}
</script>
<style type="text/tailwindcss">
@layer base{
  *{@apply box-border}
  html{@apply scroll-smooth bg-slate-950}
  body{@apply min-h-screen m-0 bg-slate-950 text-slate-100 font-sans antialiased; background:radial-gradient(circle at top left,rgba(59,130,246,.22),transparent 30rem),radial-gradient(circle at bottom right,rgba(16,185,129,.12),transparent 28rem),linear-gradient(135deg,#020617,#0f172a 58%,#111827); -webkit-tap-highlight-color:transparent}
  a{@apply text-blue-200 no-underline transition hover:text-white}
  code{@apply rounded-lg border border-white/10 bg-slate-950/80 px-1.5 py-0.5 text-xs text-slate-100}
  pre{@apply max-h-[65vh] overflow-auto whitespace-pre-wrap rounded-2xl border border-white/10 bg-slate-950/90 p-4 text-sm text-slate-100}
  input,select,textarea{@apply w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-blue-400 focus:ring-4 focus:ring-blue-500/15}
  textarea{@apply min-h-[130px] resize-y}
  button,input[type=submit],.btn{@apply inline-flex min-h-10 items-center justify-center gap-2 rounded-2xl bg-blue-600 px-4 py-2.5 text-sm font-extrabold text-white shadow-sm transition hover:bg-blue-500 hover:text-white disabled:cursor-not-allowed disabled:opacity-60}
  h2{@apply text-lg font-black tracking-tight text-white}
  h3{@apply text-base font-extrabold text-white}
  summary{@apply cursor-pointer font-extrabold text-blue-200}
  details{@apply mt-3}
}
@layer components{
  .layout{@apply grid min-h-screen grid-cols-[280px_minmax(0,1fr)]}
  .side{@apply sticky top-0 h-screen overflow-y-auto border-r border-white/10 bg-slate-950/80 p-5 backdrop-blur-2xl}
  .brand{@apply flex items-center gap-3 text-2xl font-black tracking-tight text-white}
  .brand-mark{@apply grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-blue-500 to-violet-600 shadow-lg shadow-blue-500/20}
  .sub,.muted,.help{@apply text-slate-400}
  .sub{@apply my-4 text-xs}
  .help{@apply mt-1.5 text-xs}
  .nav{@apply grid gap-1.5}
  .nav-search{@apply mb-3 w-full rounded-2xl border border-white/10 bg-slate-950/60 px-3 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10}
  .nav a{@apply flex items-center gap-3 rounded-2xl border border-transparent px-3 py-3 text-sm font-bold text-slate-200 hover:border-blue-300/20 hover:bg-blue-400/10}
  .nav a.active{@apply border-blue-300/25 bg-blue-400/15 text-white shadow-sm}
  .nav-ico{@apply w-6 text-center}
  .main{@apply mx-auto w-full max-w-[1560px] px-6 py-6 pb-28 lg:pb-6}
  .top{@apply mb-5 flex items-start justify-between gap-4}
  .h1{@apply text-3xl font-black tracking-tight text-white}
  .top-right{@apply text-right text-xs text-slate-400}
  .mobilebar{@apply sticky top-0 z-50 hidden items-center justify-between border-b border-white/10 bg-slate-950/90 px-4 py-3 backdrop-blur-2xl}
  .menu-toggle{@apply hidden}
  .card{@apply mb-4 rounded-3xl border border-white/10 bg-slate-900/75 p-5 shadow-soft backdrop-blur-xl transition duration-200 hover:border-white/15}
  .grid{@apply mb-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4}
  .grid2{@apply grid grid-cols-1 gap-4 xl:grid-cols-2}
  .row{@apply grid grid-cols-1 gap-3 md:grid-cols-2}
  .row3{@apply grid grid-cols-1 gap-3 lg:grid-cols-3}
  .stat{@apply min-h-[110px] rounded-3xl border border-white/10 bg-gradient-to-b from-blue-500/10 to-slate-900/90 p-4 shadow-sm}
  .stat b{@apply block text-3xl font-black tracking-tight text-white}
  .stat small{@apply mt-1 block text-xs text-slate-400}
  .stat-ok{@apply from-emerald-500/15}
  .stat-warn{@apply from-amber-500/15}
  .stat-danger{@apply from-red-500/15}
  .badge{@apply inline-flex items-center gap-1 rounded-full border border-white/10 bg-slate-800/90 px-2.5 py-1 text-xs font-bold text-slate-300 whitespace-nowrap}
  .badge-ok{@apply border-emerald-400/30 bg-emerald-500/15 text-emerald-200}
  .badge-warn{@apply border-amber-400/30 bg-amber-500/15 text-amber-100}
  .badge-danger{@apply border-red-400/30 bg-red-500/15 text-red-200}
  .badge-info{@apply border-sky-400/30 bg-sky-500/15 text-sky-200}
  .badge-muted{@apply bg-slate-700/50 text-slate-300}
  .table-wrap{@apply overflow-auto rounded-2xl border border-white/10}
  .table{@apply w-full min-w-[760px] border-collapse text-sm}
  .table th{@apply sticky top-0 border-b border-white/10 bg-slate-950/80 px-3 py-3 text-left text-[11px] font-black uppercase tracking-wider text-slate-300 backdrop-blur}
  .table td{@apply border-b border-white/10 px-3 py-3 align-top text-slate-100}
  .table tr:last-child td{@apply border-b-0}
  .table tbody tr{@apply transition hover:bg-white/[.035]}
  .actions{@apply flex flex-wrap items-center gap-2}
  .secondary{@apply bg-slate-700 hover:bg-slate-600}
  .danger{@apply bg-red-600 hover:bg-red-500}
  .ok{@apply bg-emerald-600 hover:bg-emerald-500}
  .warn{@apply bg-amber-600 hover:bg-amber-500}
  .ghost{@apply border border-white/10 bg-transparent text-slate-100 hover:bg-white/10}
  .field{@apply my-2.5}
  .field label{@apply mb-1.5 flex items-center justify-between gap-3 text-sm font-extrabold text-slate-300}
  .flash{@apply mb-3 rounded-2xl border border-white/10 bg-slate-800/90 px-4 py-3 text-sm text-slate-100}
  .flash.success{@apply border-emerald-400/30 bg-emerald-500/15 text-emerald-100}
  .flash.error{@apply border-red-400/30 bg-red-500/15 text-red-100}
  .flash.warning{@apply border-amber-400/30 bg-amber-500/15 text-amber-100}
  .inline-form{@apply inline-flex}
  .footer{@apply mt-6 grid gap-1.5 text-xs text-slate-400}
  .progress{@apply mr-2 inline-flex h-2.5 w-32 overflow-hidden rounded-full border border-white/10 bg-slate-950 align-middle}
  .progress span{@apply block h-full rounded-full bg-gradient-to-r from-blue-400 to-emerald-400 transition-all}
  .kbd{@apply inline-flex rounded-lg border border-white/10 border-b-white/20 bg-slate-950 px-1.5 py-0.5 text-[11px] text-slate-300}
  .pillbar{@apply flex flex-wrap gap-2}
  .empty{@apply p-6 text-center text-slate-400}
  .copybox{@apply break-all select-all}
  .error-message{@apply max-w-[520px] whitespace-pre-wrap break-words text-slate-200}
  .danger-zone{@apply border-red-400/25 bg-red-500/10}
  .v3-live-dot{@apply mr-2 inline-block h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_0_6px_rgba(52,211,153,.12)]}
  .mini-grid{@apply grid grid-cols-1 gap-3 md:grid-cols-3}
  .mini-stat{@apply rounded-2xl border border-white/10 bg-slate-950/35 p-4}
  .mini-stat b{@apply block text-2xl font-black text-white}
  .mobile-card{@apply hidden}
  .sticky-actions{@apply sticky bottom-24 z-20 rounded-2xl border border-white/10 bg-slate-950/85 p-3 shadow-soft backdrop-blur-xl lg:bottom-3}
  .job-controls form{@apply inline-flex}
  .live-note{@apply inline-flex items-center text-xs text-slate-400}
  .nowrap{@apply whitespace-nowrap}
  .detail-hero{@apply grid grid-cols-1 gap-4 xl:grid-cols-[1.2fr_.8fr]}
  .history-item{@apply my-2 rounded-2xl border border-white/10 bg-slate-950/40 p-4}
  .history-meta{@apply mb-2 flex flex-wrap gap-2 text-xs text-slate-400}
  .touch-list{@apply grid gap-3}
  .touch-item{@apply rounded-3xl border border-white/10 bg-slate-950/35 p-4}
  .bottom-nav{@apply fixed inset-x-0 bottom-0 z-50 grid grid-cols-5 border-t border-white/10 bg-slate-950/95 px-2 pb-[calc(.5rem+env(safe-area-inset-bottom))] pt-2 backdrop-blur-2xl lg:hidden}
  .bottom-nav a{@apply flex flex-col items-center justify-center gap-1 rounded-2xl px-1 py-2 text-[11px] font-bold text-slate-400}
  .bottom-nav a.active{@apply bg-blue-500/15 text-white}
  .mobile-search-sticky{@apply sticky top-[60px] z-30 bg-slate-950/70 backdrop-blur-xl}
  .command-hero{@apply mb-4 grid grid-cols-1 gap-4 overflow-hidden rounded-[2rem] border border-white/10 bg-gradient-to-br from-blue-500/20 via-slate-900/85 to-emerald-500/10 p-5 shadow-soft backdrop-blur-xl xl:grid-cols-[1.4fr_.6fr]}
  .command-hero h2{@apply mt-3 text-3xl font-black leading-tight tracking-tight text-white md:text-4xl}
  .command-hero p{@apply mt-2 max-w-3xl text-sm leading-6 text-slate-300}
  .hero-kicker{@apply inline-flex items-center rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs font-black uppercase tracking-wider text-emerald-100}
  .hero-actions{@apply mt-4}
  .hero-panel{@apply grid grid-cols-2 gap-3}
  .hero-panel div{@apply rounded-3xl border border-white/10 bg-slate-950/45 p-4}
  .hero-panel b{@apply mt-1 block text-3xl font-black text-white}
  .feature-launcher{@apply grid gap-4}
  .feature-section{@apply rounded-3xl border border-white/10 bg-slate-900/55 p-4 shadow-sm backdrop-blur-xl}
  .section-title{@apply mb-3 flex items-center justify-between gap-3}
  .feature-grid{@apply grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-4}
  .feature-grid-compact{@apply 2xl:grid-cols-3}
  .feature-card{@apply group relative flex min-h-[112px] items-start gap-3 rounded-3xl border border-white/10 bg-slate-950/35 p-4 text-slate-100 transition duration-200 hover:-translate-y-0.5 hover:border-blue-300/35 hover:bg-blue-500/10 hover:text-white hover:shadow-lg hover:shadow-blue-950/20}
  .feature-card-active{@apply border-blue-300/30 bg-blue-500/15}
  .feature-icon{@apply grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-white/10 text-xl}
  .feature-card b{@apply block text-sm font-black text-white}
  .feature-card small{@apply mt-1 block text-xs leading-5 text-slate-400 group-hover:text-slate-300}
  .feature-card em{@apply absolute bottom-3 right-4 text-[11px] font-black not-italic text-blue-200 opacity-0 transition group-hover:opacity-100}
  .toast{@apply fixed bottom-24 left-1/2 z-[70] hidden -translate-x-1/2 rounded-2xl border border-white/10 bg-slate-950/95 px-4 py-3 text-sm font-bold text-white shadow-soft backdrop-blur-xl}
  .toast.show{@apply block}
}

</style>
<style>
@media(max-width:860px){.mobilebar{display:flex}.layout{display:block}.side{display:none;position:fixed;z-index:60;inset:58px 0 auto 0;height:calc(100vh - 58px);border-right:0;border-bottom:1px solid rgba(255,255,255,.1)}.menu-toggle:checked~.layout .side{display:block}.main{padding:16px 16px calc(92px + env(safe-area-inset-bottom))}.top{display:block}.top-right{text-align:left;margin-top:.4rem}.h1{font-size:1.55rem}.desktop-table{display:none}.mobile-card{display:grid;gap:.75rem}.table{font-size:12px;min-width:720px}.card{padding:16px}.sticky-actions .actions{display:grid;grid-template-columns:1fr 1fr}.sticky-actions .actions>*{width:100%}input,select,textarea{font-size:16px}button,input[type=submit],.btn{min-height:44px}.actions .btn,.actions button{width:auto}}
@media(max-width:520px){.grid{grid-template-columns:1fr}.actions .btn,.actions button{width:100%}.sticky-actions .actions{grid-template-columns:1fr}.pillbar{display:grid}.pillbar .btn{width:100%}.mini-stat b{font-size:18px}.command-hero h2{font-size:1.75rem}.hero-panel b{font-size:1.35rem}.feature-card{min-height:auto}}
</style>
</head>
<body>
<input id="menu-toggle" class="menu-toggle" type="checkbox">
<div class="mobilebar"><label class="btn ghost" for="menu-toggle">☰ Menu</label><b>Bot Admin V8</b></div>
<div class="layout">
<aside class="side">
  <div class="brand"><span class="brand-mark">🤖</span><span>Bot Admin</span></div>
  <div class="sub">System V8 · Admin ID: <code>{{ admin_id }}</code></div>
  <input class="nav-search" data-nav-filter placeholder="Search admin features…" aria-label="Search admin features">
  <nav class="nav" data-nav-list>{% for key,label,url,ico in nav %}<a class="{{ 'active' if key==active else '' }}" href="{{ url }}" data-nav-item data-label="{{ label|lower }}"><span class="nav-ico">{{ ico }}</span><span>{{ label }}</span></a>{% endfor %}</nav>
  <div class="footer"><div>Supabase: <b>{{ 'ON' if supabase_on else 'OFF' }}</b></div><div>Redis: <b>{{ 'ON' if redis_on else 'OFF' }}</b></div><div><a href="/ping">Ping</a> · <a href="/readyz">Ready</a></div><div class="muted">Tip: press <span class="kbd">Ctrl</span> + <span class="kbd">K</span> to search.</div></div>
</aside>
<main class="main">
  <div class="top"><div><div class="h1">{{ title }}</div><div class="muted">Unified Admin Center V8 · error inbox · safer Telegram operations</div></div><div class="top-right"><div data-local-time>{{ now }}</div><div>{{ time_hint }}</div></div></div>
  {% for cat,msg in messages %}<div class="flash {{ cat }}">{{ msg }}</div>{% endfor %}
  {{ body|safe }}
</main>
</div>
<nav class="bottom-nav">{% for key,label,url,ico in bottom_nav %}<a class="{{ 'active' if key==active else '' }}" href="{{ url }}"><span>{{ ico }}</span><span>{{ label }}</span></a>{% endfor %}</nav>
<div class="toast" data-toast></div>
<script>
window.WEB_CSRF={{ csrf|tojson }};
(function(){
  function qs(sel,root){return (root||document).querySelector(sel)}
  function qsa(sel,root){return Array.from((root||document).querySelectorAll(sel))}
  qsa('form[data-confirm]').forEach(function(form){form.addEventListener('submit',function(e){var msg=form.getAttribute('data-confirm')||'Are you sure?';if(!confirm(msg)){e.preventDefault();return false;}})});
  qsa('form').forEach(function(form){form.addEventListener('submit',function(){var btn=form.querySelector('button[type="submit"],button:not([type]),input[type="submit"]');if(btn&&!form.dataset.noDisable){setTimeout(function(){btn.disabled=true;btn.dataset.oldText=btn.innerText;btn.innerText='Working…'},0)}})});
  qsa('[data-count-target]').forEach(function(el){var target=qs(el.getAttribute('data-count-target'));function update(){if(target){el.textContent=target.value.length}};if(target){target.addEventListener('input',update);update();}});
  function toast(msg){var el=qs('[data-toast]');if(!el)return;el.textContent=msg;el.classList.add('show');clearTimeout(el._t);el._t=setTimeout(function(){el.classList.remove('show')},1800)}
  qsa('[data-copy]').forEach(function(btn){btn.addEventListener('click',function(){var text=btn.getAttribute('data-copy')||'';navigator.clipboard&&navigator.clipboard.writeText(text).then(function(){btn.innerText='Copied';toast('Copied to clipboard')}).catch(function(){toast('Copy failed')})})});
  var navFilter=qs('[data-nav-filter]');
  if(navFilter){navFilter.addEventListener('input',function(){var q=navFilter.value.trim().toLowerCase();qsa('[data-nav-item]').forEach(function(a){var hay=(a.getAttribute('data-label')||a.textContent||'').toLowerCase();a.style.display=!q||hay.indexOf(q)>=0?'flex':'none';});});document.addEventListener('keydown',function(e){if((e.ctrlKey||e.metaKey)&&String(e.key).toLowerCase()==='k'){e.preventDefault();navFilter.focus();navFilter.select();}})}
  qsa('[data-nav-item],.bottom-nav a').forEach(function(a){a.addEventListener('click',function(){var t=qs('#menu-toggle');if(t)t.checked=false;})});
  var live=qs('[data-live-status]');
  function applyStatus(data){if(!data||!data.ok)return;qsa('[data-metric]').forEach(function(el){var k=el.getAttribute('data-metric');if(data.metrics&&Object.prototype.hasOwnProperty.call(data.metrics,k)){el.textContent=data.metrics[k]}});var up=qs('[data-uptime]');if(up)up.textContent=data.uptime||'';var lt=qs('[data-local-time]');if(lt&&data.local_time)lt.textContent=data.local_time;}
  function refreshStatus(){fetch('/admin/status.json?light=1',{credentials:'same-origin',cache:'no-store'}).then(function(r){return r.ok?r.json():null}).then(applyStatus).catch(function(){});}
  function refreshLiveDashboard(){var root=qs('[data-realtime-dashboard]');if(!root)return;fetch('/admin/live.json',{credentials:'same-origin',cache:'no-store'}).then(function(r){return r.ok?r.json():null}).then(function(data){if(!data||!data.ok)return;applyStatus(data);if(data.counts){qsa('[data-count]').forEach(function(el){var k=el.getAttribute('data-count');if(Object.prototype.hasOwnProperty.call(data.counts,k)){el.textContent=data.counts[k]}})}var jobs=qs('[data-live-jobs]');if(jobs&&typeof data.jobs_html==='string')jobs.innerHTML=data.jobs_html;var sch=qs('[data-live-schedules]');if(sch&&typeof data.schedules_html==='string')sch.innerHTML=data.schedules_html;var stamp=qs('[data-live-updated]');if(stamp&&data.local_time)stamp.textContent='Updated '+data.local_time;}).catch(function(){});}
  function refreshBroadcastJobs(){var target=qs('[data-broadcast-jobs]');if(!target)return;fetch('/admin/broadcast/jobs.json',{credentials:'same-origin',cache:'no-store'}).then(function(r){return r.ok?r.json():null}).then(function(data){if(!data||!data.ok)return;if(typeof data.rows_html==='string')target.innerHTML=data.rows_html;}).catch(function(){});}
  var hasRealtime=!!qs('[data-realtime-dashboard]');
  if(live&&!hasRealtime){refreshStatus();setInterval(refreshStatus,{{ status_poll_ms }})}
  if(hasRealtime){refreshLiveDashboard();setInterval(refreshLiveDashboard,{{ live_poll_ms }})}
  if(qs('[data-broadcast-jobs]')){refreshBroadcastJobs();setInterval(refreshBroadcastJobs,5000)}
})();
</script>
</body>
</html>
"""
    return render_template_string(
        template,
        title=title,
        body=body,
        active=active,
        nav=nav,
        bottom_nav=bottom_nav,
        admin_id=_web_current_admin_id(),
        csrf=csrf,
        messages=get_flashed_messages(with_categories=True),
        now=_fmt_local_dt(),
        time_hint=_fmt_local_time_hint(),
        supabase_on=bool(supabase),
        redis_on=bool(redis_client),
        status_poll_ms=WEB_STATUS_POLL_SECONDS * 1000,
        live_poll_ms=WEB_LIVE_POLL_SECONDS * 1000,
    ), status_code


def _web_admin_session_valid() -> bool:
    # Primary auth path: signed Starlette session cookie.
    try:
        if bool(session.get("web_admin_ok")):
            admin_id = _web_int(session.get("web_admin_id"), 0)
            if _web_valid_admin_id(admin_id):
                return True
            session.clear()
            return False
    except Exception:
        pass

    # API-only fallback: short-lived HMAC bearer token returned by login.
    # This fixes separated frontends that can log in but cannot hydrate
    # /api/admin/auth/me because the browser did not persist the cookie.
    return bool(_web_bearer_admin_id())


def _web_locked_admin_html(message: str) -> str:
    return (
        "<div class='card'>"
        "<h2>🔒 Admin dashboard locked</h2>"
        f"<p>{_web_h(message)}</p>"
        "<p class='muted'>Required production env: <code>ADMIN_IDS</code> plus "
        "<code>ADMIN_WEB_PASSWORD</code> or <code>WEB_ADMIN_PASSWORD</code>. "
        "Restart after changing deployment environment variables.</p>"
        "</div>"
    )


def _web_login_page(error: str = ""):
    if _backend_only_mode():
        return _admin_api_error(
            error or "HTML login page is disabled. Use POST /api/admin/auth/login from the separate frontend.",
            401,
            code="admin_login_page_disabled",
            login_endpoint="/api/admin/auth/login",
        )
    if not session.get("web_login_csrf"):
        session["web_login_csrf"] = secrets.token_urlsafe(32)
    next_url = _web_safe_next_url(request.args.get("next") or request.form.get("next"), "web_admin_home")
    admin_hint = str(sorted(ADMIN_IDS)[0]) if ADMIN_IDS else ""
    error_html = f"<div class='flash danger'>{_web_h(error)}</div>" if error else ""
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Admin Login</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0f172a;color:#e5e7eb;font-family:system-ui,-apple-system,Segoe UI,sans-serif}.card{width:min(440px,calc(100vw - 32px));background:#111827;border:1px solid #334155;border-radius:22px;padding:28px;box-shadow:0 24px 80px rgba(0,0,0,.35)}h1{margin:.2rem 0 1rem;font-size:1.6rem}.muted{color:#94a3b8;font-size:.94rem}label{display:block;margin-top:14px;color:#cbd5e1;font-weight:700}input{box-sizing:border-box;width:100%;margin-top:7px;border-radius:14px;border:1px solid #475569;background:#020617;color:#fff;padding:13px 14px;font-size:16px}button{width:100%;margin-top:20px;border:0;border-radius:14px;padding:13px 14px;background:#38bdf8;color:#082f49;font-weight:800;font-size:16px;cursor:pointer}.flash{margin:12px 0;padding:10px 12px;border-radius:12px}.danger{background:#7f1d1d;color:#fecaca}</style>
</head><body><form class="card" method="post"><h1>🔐 Admin Login</h1><p class="muted">Protected dashboard. Enter your web admin password.</p>{{ error|safe }}<input type="hidden" name="csrf_token" value="{{ csrf }}"><input type="hidden" name="next" value="{{ next_url }}"><label>Telegram Admin ID</label><input name="admin_id" value="{{ admin_hint }}" inputmode="numeric" autocomplete="username"><label>Password</label><input name="password" type="password" autocomplete="current-password" required autofocus><button>Login</button><p class="muted">Tip: set ADMIN_WEB_PASSWORD in Render/Railway environment.</p></form></body></html>"""
    return HTMLResponse(render_template_string(
        template,
        csrf=session.get("web_login_csrf"),
        next_url=next_url,
        admin_hint=admin_hint,
        error=error_html,
    ), status_code=401 if error else 200)


_ADMIN_LOGIN_ATTEMPTS: dict[str, deque[float]] = {}
_ADMIN_LOGIN_ATTEMPTS_LOCK = threading.RLock()


def _web_client_ip() -> str:
    # Respect common proxy headers only when WEB_TRUST_PROXY/RENDER is enabled.
    # Otherwise use the ASGI client host so spoofed X-Forwarded-For headers do
    # not bypass login throttling on direct/local deployments.
    try:
        if _env_bool("WEB_TRUST_PROXY", _env_bool("RENDER", False)):
            for header in ("cf-connecting-ip", "x-real-ip", "x-forwarded-for"):
                value = str(request.headers.get(header) or "").strip()
                if value:
                    return value.split(",", 1)[0].strip()[:80]
        req_obj = getattr(request, "_req", None)
        client = getattr(req_obj, "client", None)
        host = str(getattr(client, "host", "") or "").strip()
        return host[:80] or "unknown"
    except Exception:
        return "unknown"


def _admin_login_rate_limit_key(admin_id: int | None) -> str:
    raw = f"{_web_client_ip()}:{admin_id or 0}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _admin_login_rate_limit_config() -> tuple[int, float]:
    attempts = _env_int("ADMIN_LOGIN_RATE_LIMIT_ATTEMPTS", 5, minimum=1, maximum=100)
    window_s = _env_float("ADMIN_LOGIN_RATE_LIMIT_WINDOW_S", 300.0, minimum=10.0, maximum=86400.0)
    return attempts, window_s


def _admin_login_redis_enabled() -> bool:
    return _env_bool("ADMIN_LOGIN_RATE_LIMIT_REDIS_ENABLED", True) and globals().get("redis_client") is not None


def _admin_login_redis_key(admin_id: int | None) -> str:
    key_hash = _admin_login_rate_limit_key(admin_id)
    key_builder = globals().get("_redis_key")
    if callable(key_builder):
        with suppress(Exception):
            return str(key_builder("admin_login_fail", key_hash))
    return f"admin_login_fail:{key_hash}"


def _admin_login_redis_count(admin_id: int | None) -> tuple[int | None, int | None]:
    if not _admin_login_redis_enabled():
        return None, None
    redis_obj = globals().get("redis_client")
    key = _admin_login_redis_key(admin_id)
    try:
        raw = redis_obj.get(key)
        count = int(raw or 0)
        ttl = redis_obj.ttl(key)
        ttl_i = int(ttl) if ttl is not None and int(ttl) > 0 else None
        return count, ttl_i
    except Exception as exc:
        _log_once(logging.WARNING, "admin_login_redis_limit_fallback", "Admin login Redis rate-limit fallback: %s", exc)
        return None, None


def _admin_login_is_limited(admin_id: int | None) -> tuple[bool, int]:
    limit, window_s = _admin_login_rate_limit_config()
    redis_count, redis_ttl = _admin_login_redis_count(admin_id)
    if redis_count is not None:
        if redis_count >= limit:
            return True, max(1, int(redis_ttl or window_s))
        return False, 0

    key = _admin_login_rate_limit_key(admin_id)
    now = time.monotonic()
    with _ADMIN_LOGIN_ATTEMPTS_LOCK:
        dq = _ADMIN_LOGIN_ATTEMPTS.setdefault(key, deque())
        stale_before = now - window_s
        while dq and dq[0] < stale_before:
            dq.popleft()
        if len(dq) >= limit:
            retry_after = max(1, int(window_s - (now - dq[0])))
            return True, retry_after
        return False, 0


def _admin_login_record_failure(admin_id: int | None) -> None:
    _limit, window_s = _admin_login_rate_limit_config()
    if _admin_login_redis_enabled():
        redis_obj = globals().get("redis_client")
        key = _admin_login_redis_key(admin_id)
        try:
            count = int(redis_obj.incr(key) or 1)
            if count == 1 or int(redis_obj.ttl(key) or -1) < 0:
                redis_obj.expire(key, int(window_s))
            return
        except Exception as exc:
            _log_once(logging.WARNING, "admin_login_redis_record_fallback", "Admin login Redis failure-record fallback: %s", exc)

    key = _admin_login_rate_limit_key(admin_id)
    now = time.monotonic()
    with _ADMIN_LOGIN_ATTEMPTS_LOCK:
        dq = _ADMIN_LOGIN_ATTEMPTS.setdefault(key, deque())
        stale_before = now - window_s
        while dq and dq[0] < stale_before:
            dq.popleft()
        dq.append(now)
        # Opportunistic cleanup to avoid unbounded memory growth under scans.
        if len(_ADMIN_LOGIN_ATTEMPTS) > 20_000:
            for old_key in list(_ADMIN_LOGIN_ATTEMPTS.keys())[:1000]:
                old_dq = _ADMIN_LOGIN_ATTEMPTS.get(old_key)
                if not old_dq or old_dq[-1] < stale_before:
                    _ADMIN_LOGIN_ATTEMPTS.pop(old_key, None)


def _admin_login_clear_failures(admin_id: int | None) -> None:
    if _admin_login_redis_enabled():
        redis_obj = globals().get("redis_client")
        try:
            redis_obj.delete(_admin_login_redis_key(admin_id))
        except Exception:
            pass
    key = _admin_login_rate_limit_key(admin_id)
    with _ADMIN_LOGIN_ATTEMPTS_LOCK:
        _ADMIN_LOGIN_ATTEMPTS.pop(key, None)


def _admin_cookie_policy_payload() -> dict[str, Any]:
    same_site = _cookie_samesite()
    secure = bool(_default_cookie_secure())
    return {
        "same_site": same_site,
        "secure": secure,
        "session_cookie_name": str(getattr(SETTINGS, "WEB_SESSION_COOKIE_NAME", "bot_admin_session") or "bot_admin_session"),
        "requires_fetch_credentials_include": True,
        "same_site_none_requires_secure": same_site == "none",
        "bearer_fallback_enabled": _admin_api_token_fallback_enabled(),
        "note": "For separated frontend calls, use fetch/axios credentials include for cookies, or Authorization: Bearer <access_token> fallback.",
    }


def _admin_auth_profile_payload(admin_id: int | None = None, *, authenticated: bool = True) -> dict[str, Any]:
    """Return a stable admin profile shape for separated frontends.

    Some frontend builds hydrate auth state from login response directly while
    others call /api/admin/auth/me after login. Keep both endpoints returning
    the same profile/admin/user objects so the UI never treats a successful
    login as incomplete just because profile data was missing.
    """
    resolved_admin_id = int(admin_id or _web_current_admin_id() or 0) if authenticated else None
    profile = {
        "id": resolved_admin_id,
        "admin_id": resolved_admin_id,
        "telegram_id": resolved_admin_id,
        "role": "admin",
        "name": f"Admin {resolved_admin_id}" if resolved_admin_id else "Admin",
        "display_name": f"Admin {resolved_admin_id}" if resolved_admin_id else "Admin",
        "username": "",
    } if authenticated else None
    return {
        "authenticated": bool(authenticated),
        "admin_id": resolved_admin_id,
        "csrf_token": _web_csrf_token() if authenticated else None,
        "profile": profile,
        "admin": profile,
        "user": profile,
        "admin_ids_configured": bool(ADMIN_IDS),
        "admin_password_configured": bool(_web_admin_password()),
        "frontend_allowed_origins": _frontend_allowed_origins(),
        "origin_guard_enabled": _admin_origin_guard_enabled(),
        "cookie_policy": _admin_cookie_policy_payload(),
        "auth_methods": ["cookie", "bearer"] if _admin_api_token_fallback_enabled() else ["cookie"],
        "bearer_fallback_enabled": _admin_api_token_fallback_enabled(),
        "release": ADMIN_BACKEND_RELEASE,
    }


def web_admin_required(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _web_admin_enabled():
            return _admin_api_error("Admin backend is disabled by WEB_ADMIN_ENABLED=0.", 403, code="admin_disabled")
        if not ADMIN_IDS:
            return _admin_api_error(
                "No ADMIN_IDS are configured. Configure at least one Telegram admin ID before using admin APIs.",
                403,
                code="admin_ids_missing",
            )
        if not _web_admin_password():
            return _admin_api_error(
                "No web admin password is configured. Set ADMIN_WEB_PASSWORD or WEB_ADMIN_PASSWORD.",
                403,
                code="admin_password_missing",
            )
        if not _web_admin_session_valid():
            if _backend_only_mode() or _request_wants_json():
                return _admin_api_error(
                    "Authentication required. Login with POST /api/admin/auth/login.",
                    401,
                    code="admin_auth_required",
                    login_endpoint="/api/admin/auth/login",
                )
            return redirect(_web_url("web_admin_login", next=request.full_path))
        _web_csrf_token()
        return fn(*args, **kwargs)
    return wrapper


@app_flask.route("/api/admin/diagnostics", methods=["GET"])
@web_admin_required
def api_admin_diagnostics():
    return _admin_api_success(_admin_diagnostics_payload())


@app_flask.route("/api/admin/auth/login", methods=["POST", "OPTIONS"])
def api_admin_auth_login():
    if request.method == "OPTIONS":
        return _admin_api_success({"preflight": True}, status_code=204)
    if not _web_admin_enabled():
        return _admin_api_error("Admin backend is disabled by WEB_ADMIN_ENABLED=0.", 403, code="admin_disabled")
    if not ADMIN_IDS:
        return _admin_api_error("No ADMIN_IDS are configured.", 403, code="admin_ids_missing")
    expected_password = _web_admin_password()
    if not expected_password:
        return _admin_api_error("No web admin password is configured. Set ADMIN_WEB_PASSWORD or WEB_ADMIN_PASSWORD.", 403, code="admin_password_missing")

    data = request.get_json(silent=True) or {}
    got_password = str(data.get("password") or request.form.get("password") or "")
    admin_id = _web_int(data.get("admin_id") or request.form.get("admin_id"), _web_current_admin_id())

    limited, retry_after = _admin_login_is_limited(admin_id)
    if limited:
        resp = _admin_api_error(
            "Too many failed login attempts. Please wait and try again.",
            429,
            code="admin_login_rate_limited",
            retry_after_seconds=retry_after,
        )
        resp.headers["Retry-After"] = str(retry_after)
        return resp

    valid_admin = _web_valid_admin_id(admin_id)
    valid_password = hmac.compare_digest(got_password, expected_password)
    if not (valid_admin and valid_password):
        _admin_login_record_failure(admin_id)
        return _admin_api_error("Invalid admin credentials.", 401, code="invalid_admin_credentials")

    _admin_login_clear_failures(admin_id)
    session.clear()
    session["web_admin_ok"] = True
    session["web_admin_id"] = int(admin_id or _web_current_admin_id())
    session["web_login_at"] = datetime.now(timezone.utc).isoformat()
    csrf = _web_csrf_token()
    auth_payload = _admin_auth_profile_payload(int(session["web_admin_id"]), authenticated=True)
    api_token, token_ttl = _admin_issue_api_token(int(session["web_admin_id"]))
    auth_payload.update({
        "csrf_token": csrf,
        "session": "cookie",
        "message": "Login successful.",
    })
    if api_token:
        auth_payload.update({
            "access_token": api_token,
            "token_type": "Bearer",
            "expires_in": token_ttl,
            "authorization_header": "Bearer <access_token>",
        })
    return _admin_api_success(auth_payload)


@app_flask.route("/api/admin/auth/me", methods=["GET"])
def api_admin_auth_me():
    authenticated = _web_admin_session_valid()
    payload = _admin_auth_profile_payload(_web_current_admin_id() if authenticated else None, authenticated=authenticated)
    return _admin_api_success(payload) if authenticated else _admin_api_error("Not authenticated.", 401, code="admin_auth_required", **payload)


@app_flask.route("/api/admin/auth/logout", methods=["POST", "OPTIONS"])
def api_admin_auth_logout():
    if request.method == "OPTIONS":
        return _admin_api_success({"preflight": True}, status_code=204)
    session.clear()
    return _admin_api_success({
        "message": "Logged out.",
        "bearer_token_note": "Stateless bearer fallback tokens expire automatically; discard the token in the frontend on logout.",
    })


@app_flask.route("/api/admin/bootstrap", methods=["GET"])
@web_admin_required
def api_admin_bootstrap():
    return _admin_api_success({
        "admin_id": _web_current_admin_id(),
        "csrf_token": _web_csrf_token(),
        "timezone": APP_TIMEZONE_NAME,
        "timezone_label": f"{APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})",
        "health_endpoint": "/readyz",
        "diagnostics_endpoint": "/api/admin/diagnostics",
        "status_endpoint": "/admin/status.json",
        "release": ADMIN_BACKEND_RELEASE,
        "cookie_policy": _admin_cookie_policy_payload(),
        "origin_guard_enabled": _admin_origin_guard_enabled(),
        "frontend_allowed_origins": _frontend_allowed_origins(),
        "live_endpoint": "/admin/live.json",
        "performance_endpoint": "/admin/performance.json",
        "errors_endpoint": "/admin/errors.json",
        "broadcast_jobs_endpoint": "/admin/broadcast/jobs.json",
        "report_pdf_endpoint": "/admin/report.pdf",
        "auth_methods": ["cookie", "bearer"] if _admin_api_token_fallback_enabled() else ["cookie"],
        "bearer_fallback_enabled": _admin_api_token_fallback_enabled(),
    })


@app_flask.route("/admin/login", methods=["GET", "POST"])
def web_admin_login():
    if _backend_only_mode():
        return _legacy_dashboard_disabled_payload("Admin Login", status_code=410)
    if not _web_admin_enabled():
        return _web_render("Disabled", "<div class='card'>WEB_ADMIN_ENABLED=0</div>", status_code=403)
    if not ADMIN_IDS:
        return _web_render(
            "Admin Locked",
            _web_locked_admin_html("No ADMIN_IDS are configured. Web admin login is disabled until at least one Telegram admin ID is explicitly allowed."),
            status_code=403,
        )

    expected_password = _web_admin_password()
    if not expected_password:
        return _web_render(
            "Admin Locked",
            _web_locked_admin_html("No web admin password is configured."),
            status_code=403,
        )

    if request.method == "POST":
        expected_csrf = str(session.get("web_login_csrf") or "")
        got_csrf = str(request.form.get("csrf_token") or "")
        if not expected_csrf or not hmac.compare_digest(expected_csrf, got_csrf):
            return _web_login_page("Session expired. Please try again.")

        got_password = str(request.form.get("password") or "")
        admin_id = _web_int(request.form.get("admin_id"), _web_current_admin_id())
        if not hmac.compare_digest(got_password, expected_password):
            return _web_login_page("Invalid password.")
        if not _web_valid_admin_id(admin_id):
            return _web_login_page("This Telegram Admin ID is not in ADMIN_IDS.")

        session.clear()
        session["web_admin_ok"] = True
        session["web_admin_id"] = int(admin_id or _web_current_admin_id())
        session["web_login_at"] = datetime.now(timezone.utc).isoformat()
        _web_csrf_token()
        return redirect(_web_safe_next_url(request.form.get("next"), "web_admin_home"))

    return _web_login_page()


@app_flask.route("/admin/logout")
def web_admin_logout():
    session.clear()
    if _backend_only_mode():
        return _admin_api_success({"message": "Logged out."})
    return redirect(url_for("web_admin_login"))

def _web_table_count(table: str, select_field: str = "id") -> int | None:
    if not supabase:
        return None
    try:
        res = (
            supabase.table(table)
            .select(select_field, count="exact")
            .limit(1)
            .execute()
        )
        count = getattr(res, "count", None)
        return int(count) if count is not None else None
    except Exception:
        return None


def _web_counts(force: bool = False) -> dict:
    """Return dashboard counts without opening many simultaneous Supabase calls.

    V4.1 fix: /admin/status.json and /admin/live.json can be requested close
    together by the browser. Before this guard, both could run count queries at
    the same time and produce noisy [Errno 11] Resource temporarily unavailable
    warnings. This function now coalesces refreshes and serves safe stale data
    while another request is rebuilding the cache.
    """
    now = time.monotonic()
    with _WEB_COUNTS_CACHE_LOCK:
        cached = dict(_WEB_COUNTS_CACHE.get("data") or {})
        age = now - float(_WEB_COUNTS_CACHE.get("ts") or 0.0)
        if cached and not force and age < _run_state_web_counts_cache_ttl():
            return cached

    acquired = _WEB_COUNTS_BUILD_LOCK.acquire(blocking=False)
    if not acquired:
        # Another request is already refreshing counts. Return stale values instead
        # of creating extra Supabase connections.
        if cached:
            return cached
        _WEB_COUNTS_BUILD_LOCK.acquire()

    try:
        now = time.monotonic()
        with _WEB_COUNTS_CACHE_LOCK:
            cached = dict(_WEB_COUNTS_CACHE.get("data") or {})
            age = now - float(_WEB_COUNTS_CACHE.get("ts") or 0.0)
            if cached and not force and age < _run_state_web_counts_cache_ttl():
                return cached

        counts = {"users": 0, "blocked": 0, "schedules": 0, "pending": 0, "sending": 0, "failed": 0, "api_keys": 0}

        try:
            counted = _web_table_count("user_prefs", "user_id")
            counts["users"] = counted if counted is not None else (len(get_all_user_ids()) if supabase else 0)
        except Exception:
            pass

        try:
            counted = _web_table_count("blocked_users", "user_id")
            counts["blocked"] = counted if counted is not None else db_blocked_user_count()
        except Exception:
            pass

        if supabase:
            try:
                res = db_call_sync(
                    "web_sched_counts",
                    lambda: supabase.table("scheduled_broadcasts").select("id,status,error_msg").limit(5000).execute(),
                    default=None,
                    attempts=1,
                    critical=False,
                )
                rows = list(getattr(res, "data", None) or [])
                counts["schedules"] = len(rows)
                for r in rows:
                    st = str(r.get("status") or "").lower()
                    if st == "pending" and not _sched_is_draft(r):
                        counts["pending"] += 1
                    elif st == "sending":
                        counts["sending"] += 1
                    elif st == "failed":
                        counts["failed"] += 1
            except Exception:
                pass

        try:
            counts["api_keys"] = int(db_ai_api_key_status().get("active_count") or 0)
        except Exception:
            pass

        with _WEB_COUNTS_CACHE_LOCK:
            _WEB_COUNTS_CACHE["data"] = dict(counts)
            _WEB_COUNTS_CACHE["ts"] = time.monotonic()
        return counts
    finally:
        with suppress(Exception):
            _WEB_COUNTS_BUILD_LOCK.release()




def _web_count_card(key: str, label: str, value: Any, hint: str = "", kind: str = "") -> str:
    kind_cls = f" stat-{kind}" if kind else ""
    return (
        f"<div class='stat{kind_cls}'><span class='muted'>{_web_h(label)}</span>"
        f"<b data-count='{_web_h(key)}'>{_web_h(value)}</b>"
        f"<small>{_web_h(hint)}</small></div>"
    )


def _web_broadcast_job_get(job_id: str) -> dict:
    with _WEB_BROADCAST_JOBS_LOCK:
        return dict(_WEB_BROADCAST_JOBS.get(str(job_id), {}))


def _web_broadcast_job_control(job_id: str, action: str) -> tuple[bool, str]:
    job_id = str(job_id or "").strip()
    action = str(action or "").strip().lower()
    with _WEB_BROADCAST_JOBS_LOCK:
        row = _WEB_BROADCAST_JOBS.get(job_id)
        if not row:
            return False, "Broadcast job not found."
        status = str(row.get("status") or "").lower()
        if status in {"done", "failed", "cancelled"}:
            return False, f"Job is already {status}."
        if action == "pause":
            row["control"] = "pause"
            row["status"] = "paused"
            row["paused_at"] = _sched_iso()
            msg = "paused"
        elif action == "resume":
            row["control"] = "run"
            row["status"] = "running"
            row["resumed_at"] = _sched_iso()
            msg = "resumed"
        elif action == "cancel":
            row["control"] = "cancel"
            row["status"] = "cancelling"
            row["cancel_requested_at"] = _sched_iso()
            msg = "cancel requested"
        else:
            return False, "Unknown broadcast control action."
        _WEB_BROADCAST_JOBS[job_id] = row
        _WEB_BROADCAST_JOBS.move_to_end(job_id)
        return True, msg


def _web_broadcast_active_count() -> int:
    active_states = {"queued", "running", "paused", "cancelling"}
    with _WEB_BROADCAST_JOBS_LOCK:
        return sum(
            1
            for row in _WEB_BROADCAST_JOBS.values()
            if str(row.get("status") or "").lower() in active_states
        )


async def _web_broadcast_queue_consumer(worker_id: int) -> None:
    logger.info("Web broadcast queue worker %s started.", worker_id)
    while True:
        item = await _WEB_BROADCAST_QUEUE.get()  # type: ignore[union-attr]
        try:
            if item is _WEB_BROADCAST_QUEUE_STOP:
                return
            if len(item) == 4:
                job_id, admin_id, text, parse_mode = item
            else:
                job_id, admin_id, text = item
                parse_mode = _BROADCAST_PARSE_MODE_AUTO
            await asyncio.to_thread(_web_broadcast_worker, job_id, admin_id, text, parse_mode)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("web broadcast queue worker failed: %s", exc, exc_info=True)
            with suppress(Exception):
                _web_broadcast_job_set(str(item[0]), status="failed", error=str(exc)[:500], finished_at=_sched_iso())
        finally:
            with suppress(Exception):
                _WEB_BROADCAST_QUEUE.task_done()  # type: ignore[union-attr]


def _start_web_broadcast_queue_workers() -> None:
    global _WEB_BROADCAST_QUEUE, _WEB_BROADCAST_QUEUE_LOOP, _WEB_BROADCAST_QUEUE_TASKS
    loop = asyncio.get_running_loop()
    if _WEB_BROADCAST_QUEUE is not None and _WEB_BROADCAST_QUEUE_LOOP is loop:
        return
    _WEB_BROADCAST_QUEUE = asyncio.Queue(maxsize=_run_state_web_broadcast_queue_maxsize())
    _WEB_BROADCAST_QUEUE_LOOP = loop
    _WEB_BROADCAST_QUEUE_TASKS = [
        asyncio.create_task(_web_broadcast_queue_consumer(i + 1), name=f"web-broadcast-queue-{i + 1}")
        for i in range(WEB_BROADCAST_MAX_ACTIVE_JOBS)
    ]


async def _stop_web_broadcast_queue_workers() -> None:
    global _WEB_BROADCAST_QUEUE_TASKS
    if _WEB_BROADCAST_QUEUE is not None:
        for _ in _WEB_BROADCAST_QUEUE_TASKS:
            with suppress(Exception):
                _WEB_BROADCAST_QUEUE.put_nowait(_WEB_BROADCAST_QUEUE_STOP)
    for task in list(_WEB_BROADCAST_QUEUE_TASKS):
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
    _WEB_BROADCAST_QUEUE_TASKS = []


def _submit_web_broadcast_job(
    job_id: str,
    admin_id: int,
    text: str,
    parse_mode: str | None = "auto",
) -> None:
    """Submit a web broadcast without blocking the FastAPI/admin request path."""
    item = (
        str(job_id),
        int(admin_id or 0),
        str(text or ""),
        _broadcast_normalize_parse_mode(parse_mode),
    )
    queue = _WEB_BROADCAST_QUEUE
    loop = _WEB_BROADCAST_QUEUE_LOOP
    if queue is not None and loop is not None and loop.is_running():
        def _enqueue() -> None:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                _web_broadcast_job_set(job_id, status="failed", error="Broadcast queue is full; try again later.", finished_at=_sched_iso())
        loop.call_soon_threadsafe(_enqueue)
        return

    # Safe fallback for early startup/tests: keep old executor behavior.
    future = _WEB_BROADCAST_EXECUTOR.submit(
        _web_broadcast_worker,
        job_id,
        admin_id,
        text,
        _broadcast_normalize_parse_mode(parse_mode),
    )
    future.add_done_callback(_log_future_exception)


def _web_broadcast_job_rows_html(csrf: str | None = None, *, include_actions: bool = True) -> str:
    csrf = csrf or _web_csrf_token()
    with _WEB_BROADCAST_JOBS_LOCK:
        jobs = list(reversed(list(_WEB_BROADCAST_JOBS.items())))
    rows = []
    for jid, row in jobs:
        sent = int(row.get("sent") or 0)
        failed = int(row.get("failed") or 0)
        blocked = int(row.get("blocked") or 0)
        total = int(row.get("total") or 0)
        processed = sent + failed + blocked
        status = str(row.get("status") or "queued").lower()
        controls = ""
        if include_actions and status not in {"done", "failed", "cancelled"}:
            if status == "paused":
                controls += f"<form class='inline-form' method='post' action='/admin/broadcast/action'><input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='job_id' value='{_web_h(jid)}'><input type='hidden' name='action' value='resume'><button class='ok'>Resume</button></form>"
            else:
                controls += f"<form class='inline-form' method='post' action='/admin/broadcast/action'><input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='job_id' value='{_web_h(jid)}'><input type='hidden' name='action' value='pause'><button class='warn'>Pause</button></form>"
            controls += f"<form class='inline-form' method='post' action='/admin/broadcast/action' data-confirm='Cancel this running broadcast job?'><input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='job_id' value='{_web_h(jid)}'><input type='hidden' name='action' value='cancel'><button class='danger'>Cancel</button></form>"
        mode_label = _broadcast_parse_mode_label(row.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO)
        rows.append(
            f"<tr><td><code>{_web_h(jid)}</code><br><span class='muted'>{_web_h(row.get('error') or row.get('note') or '')}</span><br>{_web_badge(mode_label, 'info')}</td>"
            f"<td>{_web_status_badge(status)}</td>"
            f"<td><span class='nowrap'>{sent}/{total}</span><br>{_web_progress_bar(processed, total)}</td>"
            f"<td>{blocked}</td><td>{failed}</td>"
            f"<td>{_web_h(_web_dt(row.get('started_at') or row.get('created_at') or ''))}</td>"
            f"<td><div class='actions job-controls'>{controls or '<span class=\"muted\">-</span>'}</div></td></tr>"
        )
    return "".join(rows) or '<tr><td colspan=7><div class="empty">No jobs yet.</div></td></tr>'


def _web_schedule_rows_html(rows: list[dict], csrf: str | None = None, return_input: str = "") -> str:
    csrf = csrf or _web_csrf_token()
    table_rows = []
    for row in rows:
        rid = _web_int(row.get("id"), 0)
        content = row.get("caption") or row.get("plain_text") or ""
        preview_content, preview_mode = _broadcast_strip_format_directive(content, _BROADCAST_PARSE_MODE_AUTO)
        preview_mode_label = _broadcast_parse_mode_label(preview_mode)
        edit_parse_options = _broadcast_parse_mode_select_html("parse_mode", preview_mode)
        can_edit, edit_reason = _sched_can_edit(row, _web_current_admin_id())
        is_pending = str(row.get("status") or "").lower() == SCHED_STATUS_PENDING
        confirm_btn = ""
        if _sched_is_draft(row):
            confirm_btn = f"<form class='inline-form' method='post' action='/admin/schedules/action' data-confirm='Confirm and allow this schedule to send at its time?'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='confirm'><input type='hidden' name='row_id' value='{rid}'><button class='ok'>Confirm</button></form>"
        cancel_btn = ""
        if is_pending:
            cancel_btn = f"<form class='inline-form' method='post' action='/admin/schedules/action' data-confirm='Cancel this scheduled broadcast?'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='cancel'><input type='hidden' name='row_id' value='{rid}'><button class='danger'>Cancel</button></form>"
        duplicate_at = (_local_now() + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M")
        duplicate_btn = f"<form class='inline-form' method='post' action='/admin/schedules/action' data-confirm='Duplicate this schedule as a preview?'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='duplicate'><input type='hidden' name='row_id' value='{rid}'><input type='hidden' name='broadcast_at' value='{duplicate_at}'><button class='secondary'>Duplicate</button></form>"
        edit_block = ""
        if can_edit:
            edit_block = f"""
            <details><summary>Edit schedule</summary>
              <form method='post' action='/admin/schedules/action'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='edit_time'><input type='hidden' name='row_id' value='{rid}'><div class='field'><label>Phnom Penh time</label><input type='datetime-local' name='broadcast_at' value='{_web_h(_web_dt_input_value(row.get('broadcast_at')))}' required><div class='help'>Local {APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL}); stored as UTC automatically.</div></div><button>Save Time</button></form>
              <form method='post' action='/admin/schedules/action'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='edit_text'><input type='hidden' name='row_id' value='{rid}'><div class='field'><label>Format mode</label><select name='parse_mode'>{edit_parse_options}</select></div><div class='field'><label>{'Caption' if row.get('photo_file_id') else 'Text message'}</label><textarea name='text' required>{_web_h(preview_content)}</textarea></div><button>Save Text</button></form>
              <form method='post' action='/admin/schedules/action'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='edit_photo'><input type='hidden' name='row_id' value='{rid}'><div class='field'><label>Format mode</label><select name='parse_mode'>{edit_parse_options}</select></div><div class='row'><div class='field'><label>Photo file ID</label><input name='photo_file_id' value='{_web_h(row.get('photo_file_id') or '')}' required></div><div class='field'><label>Caption</label><input name='caption' value='{_web_h(_broadcast_strip_format_directive(row.get('caption') or '', preview_mode)[0])}' maxlength='1024'></div></div><button class='secondary'>Save / Replace Photo</button></form>
              <form method='post' action='/admin/schedules/action'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='duplicate'><input type='hidden' name='row_id' value='{rid}'><div class='field'><label>Duplicate to Phnom Penh time</label><input type='datetime-local' name='broadcast_at' value='{duplicate_at}' required><div class='help'>Creates a preview copy so you can confirm it after checking.</div></div><button class='secondary'>Duplicate Preview</button></form>
            </details>
            """
        elif is_pending:
            edit_block = f"<div class='help'>Not editable: {_web_h(edit_reason)}</div>"
        table_rows.append(
            f"<tr><td><code>#{rid}</code><br><span class='muted'>admin {_web_h(row.get('admin_id'))}</span></td>"
            f"<td>{_web_status_badge(row.get('status'), row)}<br><span class='muted'>{_web_h(row.get('error_msg') or '')}</span></td>"
            f"<td>{_web_h(_web_dt(row.get('broadcast_at')))}</td>"
            f"<td>{_web_h(_web_short(preview_content, 180))}<br>{_web_badge('photo' if row.get('photo_file_id') else 'text', 'info' if row.get('photo_file_id') else 'muted')} {_web_badge(preview_mode_label, 'info')}</td>"
            f"<td><div class='actions'>{confirm_btn}{cancel_btn}{duplicate_btn}</div>{edit_block}</td></tr>"
        )
    return "".join(table_rows) or '<tr><td colspan=5><div class="empty">No schedules found.</div></td></tr>'


def _web_sched_duplicate(row_id: int, admin_id: int, broadcast_at: datetime | None = None) -> tuple[bool, str]:
    row = db_sched_fetch_one(int(row_id))
    if not row:
        return False, "Schedule not found."
    if int(row.get("admin_id") or 0) != int(admin_id):
        return False, "Schedule belongs to another admin."
    new_dt = broadcast_at or (_local_now() + timedelta(minutes=10))
    if _sched_to_utc(new_dt) <= datetime.now(timezone.utc):
        return False, "Duplicate time must be in the future."
    text = row.get("plain_text") or ""
    photo_file_id = row.get("photo_file_id") or ""
    caption = row.get("caption") or ""
    ok, msg = _web_sched_create(admin_id, new_dt, text or caption, photo_file_id, caption, confirmed=False)
    return ok, "duplicate preview " + msg if ok else msg


def _web_live_schedules(limit: int = 8) -> list[dict]:
    """Light cached schedule preview for realtime dashboard.

    Do not call the full schedules page query on every live poll. This keeps
    mobile dashboard refreshes from competing with bot callbacks for Supabase
    connections.
    """
    limit = max(1, int(limit or 8))
    now = time.monotonic()
    with _WEB_COUNTS_CACHE_LOCK:
        cached_rows = list(_WEB_LIVE_SCHEDULES_CACHE.get("rows") or [])
        age = now - float(_WEB_LIVE_SCHEDULES_CACHE.get("ts") or 0.0)
        if cached_rows and age < WEB_LIVE_SCHEDULES_CACHE_TTL_S:
            return cached_rows[:limit]

    if not _WEB_LIVE_SCHEDULES_LOCK.acquire(blocking=False):
        return cached_rows[:limit]

    try:
        if not supabase:
            rows: list[dict] = []
        else:
            res = db_call_sync(
                "web_live_schedules",
                lambda: (
                    supabase.table("scheduled_broadcasts")
                    .select("id,admin_id,broadcast_at,plain_text,caption,photo_file_id,status,error_msg")
                    .eq("status", SCHED_STATUS_PENDING)
                    .order("broadcast_at")
                    .limit(max(limit * 4, 20))
                    .execute()
                ),
                default=None,
                attempts=1,
                critical=False,
            )
            rows = [r for r in list(getattr(res, "data", None) or []) if _sched_is_confirmed_pending(r)]
        rows = sorted(rows, key=lambda r: str(r.get("broadcast_at") or ""))[:limit]
        with _WEB_COUNTS_CACHE_LOCK:
            _WEB_LIVE_SCHEDULES_CACHE["rows"] = list(rows)
            _WEB_LIVE_SCHEDULES_CACHE["ts"] = time.monotonic()
        return rows
    finally:
        with suppress(Exception):
            _WEB_LIVE_SCHEDULES_LOCK.release()



def _web_health_item(label: str, ok: bool, detail: str = "", warn: bool = False) -> str:
    kind = "ok" if ok and not warn else ("warn" if warn else "danger")
    badge = _web_badge("OK" if ok and not warn else ("WARN" if warn else "ERROR"), kind)
    return (
        f"<tr><td><b>{_web_h(label)}</b><br><span class='muted'>{_web_h(detail)}</span></td>"
        f"<td>{badge}</td></tr>"
    )


def _web_system_v4_rows() -> str:
    """System V4 health checks for the web dashboard.

    These checks are intentionally light and local. They do not send Telegram
    messages or mutate Supabase/Redis state, so the health page is safe to
    refresh from mobile.
    """
    rows: list[str] = []
    rows.append(_web_health_item("FastAPI web admin", True, "ASGI dashboard route is responding."))
    rows.append(_web_health_item("Admin password", bool(_web_admin_password()), "ADMIN_WEB_PASSWORD or WEB_ADMIN_PASSWORD"))
    rows.append(_web_health_item("Telegram token", bool(TELEGRAM_BOT_TOKEN), "Required for direct messages and broadcasts."))
    rows.append(_web_health_item("Admin IDs", bool(ADMIN_IDS), f"{len(ADMIN_IDS)} admin id(s) configured.", warn=not bool(ADMIN_IDS)))
    rows.append(_web_health_item("Supabase", bool(supabase), "Database client is configured."))
    rows.append(_web_health_item("Redis", bool(redis_client), "Cache is optional but recommended.", warn=not bool(redis_client)))
    try:
        api_status = db_ai_api_key_status()
        api_ok = bool(api_status.get("static_key") or api_status.get("active_count") or api_status.get("memory_count"))
        api_detail = f"static={api_status.get('static_key')} active_dynamic={api_status.get('active_count')} memory={api_status.get('memory_count')}"
        if api_status.get("error"):
            api_detail += f" · {api_status.get('error')}"
        rows.append(_web_health_item("AI API auth", api_ok, api_detail, warn=not api_ok))
    except Exception as exc:
        rows.append(_web_health_item("AI API auth", False, str(exc)[:220]))
    rows.append(_web_health_item("AI provider", bool(_hf_client or _gemini), f"provider={AI_PROVIDER} model={HF_MODEL if AI_PROVIDER == 'hf' else GEMINI_MODEL}", warn=not bool(_hf_client or _gemini)))
    ocr_disabled = _hf_ocr_is_temporarily_disabled() or _ocr_provider_is_temporarily_disabled("hf") or _ocr_provider_is_temporarily_disabled("gemini")
    rows.append(_web_health_item("OCR provider", _ocr_configured(), f"provider={OCR_PROVIDER} model={HF_OCR_MODEL}; temporary_disabled={ocr_disabled}", warn=ocr_disabled or not _ocr_configured()))
    hf_tts_configured = bool(GradioClient is not None and HF_TTS_SPACE and HF_TTS_API_NAME)
    hf_tts_disabled = _hf_tts_is_temporarily_disabled()
    rows.append(_web_health_item("Khmer HF Space TTS", hf_tts_configured, _tts_provider_summary() + f"; cooldown_remaining={_hf_tts_disabled_remaining_s()}s", warn=hf_tts_disabled or not hf_tts_configured))
    if _SCHED_LOCK_ENABLED and supabase:
        try:
            lock = db_lock_read(_SCHED_LOCK_KEY)
            if lock:
                dt = _sched_parse_iso(lock.get("locked_until"))
                expired = bool(dt and dt < datetime.now(timezone.utc))
                rows.append(_web_health_item("Scheduler lock", True, f"owner={lock.get('owner')} until={_web_dt(lock.get('locked_until'))}", warn=expired))
            else:
                rows.append(_web_health_item("Scheduler lock", True, "No active lock row.", warn=True))
        except Exception as exc:
            rows.append(_web_health_item("Scheduler lock", False, str(exc)[:220]))
    else:
        rows.append(_web_health_item("Scheduler lock", bool(not _SCHED_LOCK_ENABLED), "Disabled or Supabase unavailable.", warn=True))
    rows.append(_web_health_item("Timezone", True, f"{APP_TIMEZONE_NAME} · {_fmt_local_time_hint()}"))
    rows.append(_web_health_item("Broadcast workers", WEB_BROADCAST_WORKERS > 0, f"workers={WEB_BROADCAST_WORKERS}, delay={WEB_BROADCAST_DELAY_S}s, max_jobs={_WEB_BROADCAST_JOBS_MAX}"))
    return "".join(rows)


def _web_env_check_rows() -> str:
    checks = [
        ("TELEGRAM_BOT_TOKEN", bool(TELEGRAM_BOT_TOKEN), "Required"),
        ("ADMIN_IDS", bool(ADMIN_IDS), "Recommended"),
        ("ADMIN_WEB_PASSWORD / WEB_ADMIN_PASSWORD", bool(_web_admin_password()), "Required for dashboard"),
        ("Redis WEB_SECRET_KEY", _web_stable_secret_configured(), f"Stored at {_web_secret_redis_key()}"),
        ("SUPABASE_URL", bool(os.environ.get("SUPABASE_URL") or globals().get("SB_URL")), "Recommended"),
        ("SUPABASE_SERVICE_ROLE_KEY / SUPABASE_KEY", bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or globals().get("SB_KEY")), "Recommended"),
        ("REDIS_URL", bool(os.environ.get("REDIS_URL") or globals().get("REDIS_URL")), "Optional cache"),
        ("HF_TOKEN / GEMINI_API_KEY", bool(os.environ.get("HF_TOKEN") or os.environ.get("GEMINI_API_KEY")), "Required for AI/OCR"),
        ("gradio_client", GradioClient is not None, "Required for Khmer HF Space TTS"),
        ("HF_TTS_SPACE", bool(HF_TTS_SPACE), "Default: mrrtmob/khmer-tts"),
        ("RENDER_EXTERNAL_URL", bool(os.environ.get("RENDER_EXTERNAL_URL")), "Recommended for keep-alive"),
    ]
    out = []
    for key, ok, note in checks:
        out.append(f"<tr><td><code>{_web_h(key)}</code><br><span class='muted'>{_web_h(note)}</span></td><td>{_web_badge('SET' if ok else 'MISSING', 'ok' if ok else 'warn')}</td></tr>")
    return "".join(out)


def _web_user_detail_metrics(row: dict) -> dict:
    history_rows = list(row.get("history") or [])
    messages = len(history_rows)
    chars = sum(len(str(h.get("content") or h.get("original_text") or "")) for h in history_rows)
    first_dt = ""
    last_dt = ""
    if history_rows:
        first_dt = _web_dt(history_rows[0].get("created_at") or "")
        last_dt = _web_dt(history_rows[-1].get("created_at") or "")
    last_text = ""
    if history_rows:
        last_text = _history_compact_text(history_rows[-1].get("content") or history_rows[-1].get("original_text") or "", 180)
    return {
        "messages": messages,
        "chars": chars,
        "first_seen": first_dt,
        "last_seen": last_dt,
        "last_text": last_text,
    }


# ── Admin Insights / Feature Recommendation Center ─────────────────────────
def _web_admin_recommendation_items(counts: dict | None = None, perf: dict | None = None, settings_status: dict | None = None) -> list[dict[str, str]]:
    """Return prioritized feature ideas tailored to this running dashboard.

    The list is intentionally static + lightweight so /admin/insights never
    creates heavy DB pressure.  Runtime warnings are prepended when the current
    process shows a clear operational risk.
    """
    counts = counts or {}
    perf = perf or {}
    settings_status = settings_status or {}
    items: list[dict[str, str]] = []

    try:
        failed_count = int(counts.get("failed") or 0)
    except Exception:
        failed_count = 0
    try:
        active_requests = int(((perf.get("web") or {}).get("active_requests")) or 0)
    except Exception:
        active_requests = 0

    if not bool(settings_status.get("db_ok", True)):
        items.append({
            "priority": "High",
            "feature": "Settings DB Repair Wizard",
            "status": "Recommended now",
            "why": "Bot settings are using memory fallback, so changes may not survive restart.",
            "action": "Add one-click SQL check + copy missing table SQL + re-test button.",
        })
    if failed_count > 0:
        items.append({
            "priority": "High",
            "feature": "Failed Job Auto-Recovery",
            "status": "Recommended now",
            "why": f"There are {failed_count} failed schedule/broadcast item(s) visible in admin counts.",
            "action": "Add Retry Failed, Retry Selected, and failure reason grouping.",
        })
    if active_requests >= 10:
        items.append({
            "priority": "Medium",
            "feature": "Traffic Pressure Guard",
            "status": "Recommended now",
            "why": "The dashboard is seeing high active web requests.",
            "action": "Auto-switch live polling to slow mode and show a pressure banner.",
        })
    if not bool(globals().get("redis_client")):
        items.append({
            "priority": "Medium",
            "feature": "Redis Setup Assistant",
            "status": "Recommended",
            "why": "Redis is offline, so locks/cache/replay protection use local fallback only.",
            "action": "Add setup checklist, connection tester, and safe fallback explanation.",
        })

    items.extend([
        {
            "priority": "High",
            "feature": "Smart Error Inbox Pro",
            "status": "Added in V8",
            "why": "Group repeated errors by fingerprint, affected feature, first seen, last seen, and count.",
            "action": "Open /admin/errors to review, mute, clear, copy logs, and see recommended fixes.",
        },
        {
            "priority": "High",
            "feature": "Broadcast Quality Checker",
            "status": "Next update",
            "why": "Admins need to preview Telegram formatting, link safety, length, and blocked-user impact before sending.",
            "action": "Add pre-send validation score and a test-send-to-admin button.",
        },
        {
            "priority": "Medium",
            "feature": "User Segmentation Auto Labels",
            "status": "Good for CRM",
            "why": "CRM can automatically tag active, warm, inactive, blocked, and power users.",
            "action": "Add scheduled segment refresh plus filter chips in /admin/crm.",
        },
        {
            "priority": "Medium",
            "feature": "Admin Activity Timeline",
            "status": "Good for teams",
            "why": "Important actions like pause, resume, key creation, lock release, and broadcast changes should be searchable.",
            "action": "Persist audit logs to Supabase instead of memory-only deque.",
        },
        {
            "priority": "Medium",
            "feature": "Backup & Restore Center",
            "status": "Safety feature",
            "why": "Settings, schedules, API key metadata, and CRM notes should be exportable before big updates.",
            "action": "Add JSON export/import with admin confirmation and dry-run validation.",
        },
        {
            "priority": "Low",
            "feature": "Theme & Compact Mode",
            "status": "UI polish",
            "why": "Small laptop screens and mobile admins benefit from dense tables and collapsible cards.",
            "action": "Add localStorage theme, compact density toggle, and sticky filter toolbar.",
        },
    ])

    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in items:
        key = item.get("feature", "")
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique[:12]


def _web_admin_alert_items(counts: dict | None = None, perf: dict | None = None, settings_status: dict | None = None) -> list[tuple[str, str, str]]:
    counts = counts or {}
    perf = perf or {}
    settings_status = settings_status or {}
    alerts: list[tuple[str, str, str]] = []

    if not bool(globals().get("supabase")):
        alerts.append(("Supabase offline", "Database features are disabled or using fallback memory.", "danger"))
    if not bool(settings_status.get("db_ok", True)):
        alerts.append(("Settings DB fallback", settings_status.get("error") or "bot_settings table may be missing.", "warn"))
    if not bool(globals().get("redis_client")):
        alerts.append(("Redis offline", "Replay guard, locks, and cache use process-local fallback only.", "warn"))
    try:
        failed_count = int(counts.get("failed") or 0)
        if failed_count:
            alerts.append(("Failed jobs", f"{failed_count} failed job(s) need review.", "danger"))
    except Exception:
        pass
    try:
        error_count_fn = globals().get("_admin_error_center_total_count")
        error_count = int(error_count_fn() if callable(error_count_fn) else len(list(globals().get("_ADMIN_ERROR_CENTER") or [])))
        if error_count:
            alerts.append(("Admin error inbox", f"{error_count} recent error event(s) captured. Open Error Inbox.", "danger"))
    except Exception:
        pass
    try:
        slow_count = len(list(globals().get("_WEB_SLOW_REQUESTS") or []))
        if slow_count:
            alerts.append(("Slow web requests", f"{slow_count} slow request(s) recorded in memory.", "warn"))
    except Exception:
        pass
    try:
        web_active = int(((perf.get("web") or {}).get("active_requests")) or 0)
        if web_active >= 10:
            alerts.append(("High web pressure", f"{web_active} active web request(s) right now.", "warn"))
    except Exception:
        pass

    if not alerts:
        alerts.append(("System looks stable", "No urgent admin warning detected from lightweight checks.", "ok"))
    return alerts[:8]


def _web_admin_recommendations_rows_html(items: list[dict[str, str]]) -> str:
    rows = []
    for item in items:
        priority = str(item.get("priority") or "Medium")
        kind = "danger" if priority.lower() == "high" else "warn" if priority.lower() == "medium" else "muted"
        rows.append(
            "<tr>"
            f"<td>{_web_badge(priority, kind)}</td>"
            f"<td><b>{_web_h(item.get('feature'))}</b><br><span class='muted'>{_web_h(item.get('why'))}</span></td>"
            f"<td>{_web_badge(item.get('status') or 'Idea', 'info')}</td>"
            f"<td>{_web_h(item.get('action'))}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="4"><div class="empty">No recommendations.</div></td></tr>'


def _web_admin_alerts_html(alerts: list[tuple[str, str, str]]) -> str:
    return "".join(
        f"<div class='mini-stat'><b>{_web_badge(title, kind)}</b><span class='muted'>{_web_h(detail)}</span></div>"
        for title, detail, kind in alerts
    )


@app_flask.route("/admin/insights")
@web_admin_required
def web_admin_insights():
    counts = _web_counts(force=False)
    _settings, settings_status = db_bot_settings_fetch_all()
    perf = _runtime_performance_snapshot(light=True)
    recommendations = _web_admin_recommendation_items(counts, perf, settings_status)
    alerts = _web_admin_alert_items(counts, perf, settings_status)
    body = f"""
    <div data-live-status></div>
    <div class='card'>
      <div class='actions' style='justify-content:space-between'>
        <div><h2>💡 Admin Insights & Roadmap</h2><p class='muted'>Smart warnings plus recommended next features for this bot admin panel.</p></div>
        <div class='actions'><a class='btn secondary' href='/admin/errors'>Error Inbox</a><a class='btn secondary' href='/admin/health'>Health</a><a class='btn secondary' href='/admin/optimize'>Optimize</a><a class='btn secondary' href='/admin/report.pdf'>PDF Report</a></div>
      </div>
    </div>
    <div class='grid'>
      {_web_status_card('Users', counts.get('users', 0), 'total user_prefs', 'ok' if counts.get('users') else '')}
      {_web_status_card('Pending', counts.get('pending', 0), 'scheduled broadcasts', 'warn' if counts.get('pending') else '')}
      {_web_status_card('Failed', counts.get('failed', 0), 'needs review', 'danger' if counts.get('failed') else 'ok')}
      {_web_status_card('Active requests', (perf.get('web') or {}).get('active_requests', 0), 'dashboard/API pressure', 'warn' if int((perf.get('web') or {}).get('active_requests') or 0) >= 10 else 'ok')}
    </div>
    <div class='card'><h2>Current warnings</h2><div class='mini-grid'>{_web_admin_alerts_html(alerts)}</div></div>
    <div class='card'>
      <h2>Recommended new features</h2>
      <p class='muted'>Prioritized for your current bot system: admin safety, broadcast reliability, CRM, and smoother mobile control.</p>
      <div class='table-wrap'><table class='table'><thead><tr><th>Priority</th><th>Feature</th><th>Status</th><th>Recommended implementation</th></tr></thead><tbody>{_web_admin_recommendations_rows_html(recommendations)}</tbody></table></div>
    </div>
    <div class='grid2'>
      <div class='card'><h2>Best next update package</h2><div class='mini-grid'>
        <div class='mini-stat'><b>Admin V8 Pro</b><span class='muted'>Insights, Smart Error Inbox Pro, broadcast checker, backup center.</span></div>
        <div class='mini-stat'><b>Safe Operations</b><span class='muted'>Group errors, retry failed, pause pressure, audit timeline, lock safety.</span></div>
        <div class='mini-stat'><b>Mobile Friendly</b><span class='muted'>Error Inbox in bottom nav, compact pages, sticky filters.</span></div>
      </div></div>
      <div class='card'><h2>Implementation order</h2><ol class='muted' style='padding-left:20px;line-height:1.9'>
        <li>Smart Error Inbox Pro is added. Open <a href='/admin/errors'>/admin/errors</a>.</li>
        <li>Add Broadcast Quality Checker before every send.</li>
        <li>Persist admin audit timeline to Supabase.</li>
        <li>Add Backup & Restore Center for settings/schedules.</li>
        <li>Add compact mode for small screens.</li>
      </ol></div>
    </div>
    """
    return _web_render("Insights", body, active="insights")


@app_flask.route("/admin/report")
@web_admin_required
def web_admin_report():
    body = f"""
    <div class='card'>
      <h2>📄 Admin PDF Report</h2>
      <p class='muted'>Generate a PDF with graphs, selected-day/range analytics, system status, runtime settings, feature toggles, Smart Error Inbox summary, and recent errors.</p>
      <div class='actions'>
        <a class='btn ok' href='/admin/report.pdf?range=today'>Today</a>
        <a class='btn secondary' href='/admin/report.pdf?range=yesterday'>Yesterday</a>
        <a class='btn secondary' href='/admin/report.pdf?range=last7'>Last 7 Days</a>
        <a class='btn secondary' href='/admin/report.pdf?range=last30'>Last 30 Days</a>
        <a class='btn secondary' href='/admin'>Back to Dashboard</a>
      </div>
      <p class='muted'>Custom day URL example: <code>/admin/report.pdf?date=2026-06-29</code></p>
    </div>
    """
    return _web_render("Admin Report", body, active="report")


@app_flask.route("/admin/report.pdf")
@web_admin_required
def web_admin_report_pdf():
    path = ""
    try:
        requested_date = request.args.get("date", "")
        requested_range = request.args.get("range", "today")
        report_range = _admin_report_parse_day_text(requested_date) if requested_date else _admin_report_range_from_key(requested_range)
        path = build_admin_report_pdf_sync(_web_current_admin_id(), report_range)
        with open(path, "rb") as f:
            data = f.read()
        filename = os.path.basename(path)
        return Response(
            data,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    finally:
        if path:
            with suppress(Exception):
                os.remove(path)


@app_flask.route("/admin")
@web_admin_required
def web_admin_home():
    counts = _web_counts()
    settings, settings_status = db_bot_settings_fetch_all()
    lock = db_lock_read(_SCHED_LOCK_KEY) if supabase else None
    if _SCHED_LOCK_ENABLED:
        if lock:
            lock_dt = _sched_parse_iso(lock.get("locked_until"))
            expired = bool(lock_dt and lock_dt < datetime.now(timezone.utc))
            lock_html = f"{_web_badge('expired' if expired else 'active', 'warn' if expired else 'ok')}<br><span class='muted'>{_web_h(lock.get('owner'))}</span><br><code>{_web_h(_web_dt(lock.get('locked_until')))}</code>"
        else:
            lock_html = _web_badge("no row", "warn")
    else:
        lock_html = _web_badge("disabled", "muted")

    metric_rows = []
    for k, v in _RUNTIME_METRICS.items():
        metric_rows.append(f"<tr><td>{_web_h(k.replace('_', ' ').title())}</td><td><b data-metric='{_web_h(k)}'>{int(v)}</b></td></tr>")
    metrics_rows = "".join(metric_rows)

    setting_rows = "".join(
        f"<tr><td>{_web_h(BOT_SETTING_LABELS.get(k,k))}<br><span class='muted'>{_web_h(BOT_SETTING_DESCRIPTIONS.get(k,''))}</span></td>"
        f"<td>{_web_badge('ON' if _setting_bool_from(settings,k) else 'OFF', 'ok' if _setting_bool_from(settings,k) else 'muted')}</td></tr>"
        for k in BOT_SETTING_DEFAULTS
    )
    live_schedule_rows = _web_schedule_rows_html(_web_live_schedules(), _web_csrf_token(), "")
    live_job_rows = _web_broadcast_job_rows_html(_web_csrf_token(), include_actions=False)

    feature_launcher = _web_feature_launcher_html(active="dashboard")
    body = f"""
    <div data-live-status data-realtime-dashboard></div>
    {_web_command_hero_html(counts)}
    <div class='actions' style='justify-content:space-between;margin-bottom:12px'>
      <span class='live-note'><span class='v3-live-dot'></span>Realtime dashboard refresh every {WEB_LIVE_POLL_SECONDS} seconds</span>
      <span class='muted' data-live-updated>Updated {_web_h(_fmt_local_dt())}</span>
    </div>
    <div class='grid'>
      {_web_count_card('users', 'Users', counts['users'], 'saved in user_prefs', 'ok' if counts['users'] else '')}
      {_web_count_card('blocked', 'Blocked', counts['blocked'], 'auto skipped during sends', 'danger' if counts['blocked'] else '')}
      {_web_count_card('pending', 'Pending schedules', counts['pending'], f"{counts['sending']} sending / {counts['failed']} failed", 'warn' if counts['pending'] else '')}
      {_web_count_card('api_keys', 'Active API keys', counts['api_keys'], 'generated admin access keys', 'ok' if counts['api_keys'] else '')}
    </div>
    {_optimization_score_card_html(_runtime_performance_snapshot(light=True))}
    {feature_launcher}
    <div class='grid2'>
      <div class='card'><h2>System health</h2><div class='table-wrap'><table class='table'>
        <tr><td>Uptime</td><td><b data-uptime>{_web_h(_format_uptime())}</b></td></tr>
        <tr><td>Supabase</td><td>{_web_badge('ON' if supabase else 'OFF', 'ok' if supabase else 'danger')}</td></tr>
        <tr><td>Redis</td><td>{_web_badge('ON' if redis_client else 'OFF', 'ok' if redis_client else 'warn')}</td></tr>
        <tr><td>AI</td><td>{_web_h(AI_PROVIDER)} / <code>{_web_h(HF_MODEL)}</code></td></tr>
        <tr><td>OCR</td><td>{_web_h(OCR_PROVIDER)} / <code>{_web_h(HF_OCR_MODEL)}</code></td></tr>
        <tr><td>Scheduler lock</td><td>{lock_html}</td></tr>
        <tr><td>Settings DB</td><td>{_web_badge('OK' if settings_status.get('db_ok') else 'MEMORY', 'ok' if settings_status.get('db_ok') else 'warn')} <span class='muted'>{_web_h(settings_status.get('error') or '')}</span></td></tr>
      </table></div></div>
      <div class='card'><h2>Runtime metrics</h2><div class='table-wrap'><table class='table'>{metrics_rows}</table></div></div>
    </div>
    <div class='grid2'>
      <div class='card'><div class='actions' style='justify-content:space-between'><h2>Live schedules</h2><a class='btn secondary' href='/admin/schedules'>Open</a></div><div class='table-wrap'><table class='table'><thead><tr><th>ID</th><th>Status</th><th>Time</th><th>Content</th><th>Actions</th></tr></thead><tbody data-live-schedules>{live_schedule_rows}</tbody></table></div></div>
      <div class='card'><div class='actions' style='justify-content:space-between'><h2>Live broadcast jobs</h2><a class='btn secondary' href='/admin/broadcast'>Open</a></div><div class='table-wrap'><table class='table'><thead><tr><th>Job</th><th>Status</th><th>Progress</th><th>Blocked</th><th>Failed</th><th>Started</th><th>Action</th></tr></thead><tbody data-live-jobs>{live_job_rows}</tbody></table></div></div>
    </div>
    <div class='grid2'>
      <div class='card'><h2>Feature settings</h2><div class='table-wrap'><table class='table'>{setting_rows}</table></div><p><a class='btn secondary' href='/admin/settings'>Open settings</a></p></div>
      <div class='card'><h2>Quick actions</h2><p class='muted'>Common admin tasks. Dangerous actions still require confirmation on their pages.</p><div class='actions'><a class='btn' href='/admin/optimize'>Optimize</a><a class='btn' href='/admin/users'>Manage Users</a><a class='btn' href='/admin/analytics'>Analytics V2</a><a class='btn' href='/admin/insights'>Insights</a><a class='btn' href='/admin/errors'>Error Inbox</a><a class='btn' href='/admin/schedules/calendar'>Calendar</a><a class='btn' href='/admin/schedules'>Schedules</a><a class='btn' href='/admin/broadcast'>Broadcast</a><a class='btn secondary' href='/admin/health'>Health Center</a><a class='btn secondary' href='/admin/control'>Control Center</a><a class='btn secondary' href='/admin/sql'>SQL Setup</a></div></div>
    </div>
    """
    return _web_render("Admin Center", body, active="dashboard")


@app_flask.route("/admin/center")
@web_admin_required
def web_admin_center_alias():
    return redirect(url_for("web_admin_home"))


@app_flask.route("/admin/health")
@web_admin_required
def web_admin_health():
    counts = _web_counts(force=False)
    settings, settings_status = db_bot_settings_fetch_all()
    metric_rows = "".join(
        f"<tr><td>{_web_h(k.replace('_', ' ').title())}</td><td><b data-metric='{_web_h(k)}'>{int(v)}</b></td></tr>"
        for k, v in _RUNTIME_METRICS.items()
    ) or '<tr><td colspan="2"><div class="empty">No runtime metrics yet.</div></td></tr>'
    settings_rows = "".join(
        f"<tr><td>{_web_h(BOT_SETTING_LABELS.get(k,k))}<br><span class='muted'>{_web_h(BOT_SETTING_DESCRIPTIONS.get(k,''))}</span></td>"
        f"<td>{_web_badge('ON' if _setting_bool_from(settings,k) else 'OFF', 'ok' if _setting_bool_from(settings,k) else 'muted')}</td></tr>"
        for k in BOT_SETTING_DEFAULTS
    )
    body = f"""
    <div data-live-status></div>
    <div class='grid'>
      {_web_count_card('users', 'Users', counts.get('users', 0), 'user_prefs total', 'ok' if counts.get('users') else '')}
      {_web_count_card('pending', 'Pending', counts.get('pending', 0), 'confirmed schedules', 'warn' if counts.get('pending') else '')}
      {_web_count_card('blocked', 'Blocked', counts.get('blocked', 0), 'skipped during sends', 'danger' if counts.get('blocked') else '')}
      {_web_status_card('Uptime', _format_uptime(), 'process runtime', 'ok')}
    </div>
    <div class='grid2'>
      <div class='card'><div class='actions' style='justify-content:space-between'><h2>System V4 Health Center</h2><span class='live-note'><span class='v3-live-dot'></span>safe local checks</span></div><div class='table-wrap'><table class='table'><tbody>{_web_system_v4_rows()}</tbody></table></div></div>
      <div class='card'><h2>Environment Checklist</h2><div class='table-wrap'><table class='table'><tbody>{_web_env_check_rows()}</tbody></table></div><p class='help'>This page does not show secrets. It only shows SET/MISSING.</p></div>
    </div>
    <div class='grid2'>
      <div class='card'><h2>Runtime Metrics</h2><div class='table-wrap'><table class='table'><tbody>{metric_rows}</tbody></table></div></div>
      <div class='card'><h2>Feature Switches</h2><div class='table-wrap'><table class='table'><tbody>{settings_rows}</tbody></table></div><p class='help'>Settings DB: {_web_h(settings_status.get('error') or ('OK' if settings_status.get('db_ok') else 'memory fallback'))}</p><p><a class='btn secondary' href='/admin/settings'>Open settings</a></p></div>
    </div>
    <div class='card'><h2>System V4 Notes</h2><div class='mini-grid'>
      <div class='mini-stat'><b>Mobile V4</b><span class='muted'>bottom nav, touch buttons, sticky actions</span></div>
      <div class='mini-stat'><b>User Detail+</b><span class='muted'>profile, usage cards, recent text_cache</span></div>
      <div class='mini-stat'><b>Cleaner UI</b><span class='muted'>Tailwind powered layout and responsive cards</span></div>
    </div></div>
    """
    return _web_render("Health Center", body, active="health")


def _web_fetch_users(q: str = "", page: int = 0, page_size: int = WEB_TABLE_PAGE_SIZE) -> list[dict]:
    q = (q or "").strip()
    if q:
        return search_users_by_query(q, limit=page_size)
    if not supabase:
        return []
    start = max(0, int(page)) * page_size
    end = start + page_size - 1
    fields_try = (
        "user_id, username, first_name, gender, speed, tts_model, last_active",
        "user_id, username, gender, speed, last_active",
        "user_id, username",
    )
    res = _supabase_select_schema_safe_sync(
        "user_prefs",
        fields_try,
        lambda builder, _fields: builder.range(start, end).execute(),
        name=f"web_users:{page}:{page_size}",
        default=None,
    )
    return list(getattr(res, "data", None) or []) if res is not None else []




# ── Admin CRM / User Quality Center ─────────────────────────────────────────
CRM_CACHE_TTL_S = _env_float("CRM_CACHE_TTL_S", 45.0, minimum=5.0, maximum=600.0)
CRM_DEFAULT_LIMIT = _env_int("CRM_DEFAULT_LIMIT", 80, minimum=20, maximum=500)
_CRM_SNAPSHOT_CACHE: dict[str, Any] = {"ts": 0.0, "key": "", "rows": [], "summary": {}}
_CRM_SNAPSHOT_LOCK = threading.Lock()


def _crm_parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = _sched_parse_iso(str(value))
    if dt and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _crm_age_days(value: Any) -> int | None:
    dt = _crm_parse_dt(value)
    if not dt:
        return None
    return max(0, int((datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() // 86400))


def _crm_segment_label(row: dict) -> tuple[str, str]:
    """Return (segment, badge-kind) for CRM list rows."""
    if row.get("blocked"):
        return "blocked", "danger"
    age = row.get("age_days")
    messages = int(row.get("recent_messages") or 0)
    if age is None:
        return "unknown", "muted"
    if age <= 1 and messages >= 20:
        return "power", "ok"
    if age <= 1:
        return "active", "ok"
    if age <= 7 and messages >= 8:
        return "warm", "info"
    if age <= 7:
        return "new/warm", "info"
    if age <= 30:
        return "inactive 7d", "warn"
    return "inactive 30d", "danger"


def _crm_quality_score(row: dict) -> int:
    """Score users 0-100 using cached profile + recent text_cache metrics."""
    if row.get("blocked"):
        return 0
    score = 35
    age = row.get("age_days")
    messages = int(row.get("recent_messages") or 0)
    chars = int(row.get("recent_chars") or 0)
    if age is None:
        score += 5
    elif age <= 1:
        score += 35
    elif age <= 3:
        score += 28
    elif age <= 7:
        score += 20
    elif age <= 30:
        score += 8
    else:
        score -= 10
    score += min(20, messages * 2)
    score += min(10, chars // 800)
    if row.get("username"):
        score += 3
    if row.get("tts_model"):
        score += 2
    return max(0, min(100, int(score)))


def _crm_score_badge(score: int) -> str:
    if score >= 80:
        return _web_badge(f"{score} excellent", "ok")
    if score >= 55:
        return _web_badge(f"{score} good", "info")
    if score >= 30:
        return _web_badge(f"{score} watch", "warn")
    return _web_badge(f"{score} risk", "danger")


def _crm_segment_matches(row: dict, segment: str) -> bool:
    segment = (segment or "all").strip().lower()
    if segment in ("", "all"):
        return True
    label = str(row.get("crm_segment") or "").lower()
    age = row.get("age_days")
    score = int(row.get("quality_score") or 0)
    if segment == "blocked":
        return bool(row.get("blocked"))
    if segment == "active":
        return not row.get("blocked") and age is not None and age <= 1
    if segment == "warm":
        return not row.get("blocked") and age is not None and 1 < age <= 7
    if segment == "inactive7":
        return not row.get("blocked") and age is not None and 7 < age <= 30
    if segment == "inactive30":
        return not row.get("blocked") and age is not None and age > 30
    if segment == "power":
        return label == "power" or score >= 80
    if segment == "risk":
        return score < 35 or label.startswith("inactive")
    if segment == "unknown":
        return age is None
    return segment in label


def _crm_fetch_base_users(q: str, limit: int) -> list[dict]:
    q = (q or "").strip()
    limit = max(1, min(500, int(limit or CRM_DEFAULT_LIMIT)))
    if q:
        return search_users_by_query(q, limit=limit)
    if not supabase:
        return []
    fields_try = (
        "user_id, username, first_name, gender, speed, tts_model, last_active",
        "user_id, username, gender, speed, tts_model, last_active",
        "user_id, username, gender, speed, last_active",
        "user_id, username",
    )

    def _query(builder: Any, fields: str) -> Any:
        builder = builder.limit(limit)
        if _supabase_has_selected_field(fields, "last_active"):
            builder = builder.order("last_active", desc=True)
        return builder.execute()

    res = _supabase_select_schema_safe_sync(
        "user_prefs",
        fields_try,
        _query,
        name=f"crm_base_users:{limit}",
        default=None,
    )
    rows = list(getattr(res, "data", None) or []) if res is not None else []
    return rows[:limit]


def _crm_text_cache_metrics(user_ids: list[int], scan_limit: int) -> dict[int, dict]:
    """Batch recent text_cache metrics with schema-safe optional text columns."""
    metrics: dict[int, dict] = {uid: {"recent_messages": 0, "recent_chars": 0, "last_text_at": None, "last_text": ""} for uid in user_ids}
    if not supabase or not user_ids:
        return metrics
    scan_limit = max(len(user_ids), min(3000, int(scan_limit or 1000)))
    chunk_size = 120
    for start in range(0, len(user_ids), chunk_size):
        chunk = user_ids[start:start + chunk_size]

        def _query(builder: Any, fields: str, c: list[int] = chunk) -> Any:
            builder = builder.in_("user_id", c)
            if _supabase_has_selected_field(fields, "created_at"):
                builder = builder.order("created_at", desc=True)
            return builder.limit(scan_limit).execute()

        res = _supabase_select_schema_safe_sync(
            "text_cache",
            (
                "user_id, original_text, content, created_at",
                "user_id, original_text, created_at",
                "user_id, content, created_at",
                "user_id, original_text",
                "user_id, content",
            ),
            _query,
            name=f"crm_text_cache:{start}:{len(chunk)}",
            default=None,
        )
        for item in list(getattr(res, "data", None) or []):
            try:
                uid = int(item.get("user_id") or 0)
            except Exception:
                uid = 0
            if uid not in metrics:
                continue
            text = str(item.get("original_text") or item.get("content") or item.get("text") or "")
            m = metrics[uid]
            m["recent_messages"] = int(m.get("recent_messages") or 0) + 1
            m["recent_chars"] = int(m.get("recent_chars") or 0) + len(text)
            if not m.get("last_text_at"):
                m["last_text_at"] = item.get("created_at")
                m["last_text"] = _history_compact_text(text, 150)
    return metrics


def db_crm_users_snapshot(q: str = "", segment: str = "all", limit: int = CRM_DEFAULT_LIMIT, force: bool = False) -> tuple[list[dict], dict]:
    """Return optimized CRM rows and summary.

    No new table is required. It reads user_prefs for profile/last_active,
    blocked_users in one batch, and text_cache in chunked batches for recent
    activity. A short cache prevents admin auto-refresh from hammering Supabase.
    """
    q = (q or "").strip()
    segment = (segment or "all").strip().lower() or "all"
    limit = max(10, min(500, int(limit or CRM_DEFAULT_LIMIT)))
    cache_key = f"{q}|{segment}|{limit}"
    now = time.monotonic()
    with _CRM_SNAPSHOT_LOCK:
        if not force and _CRM_SNAPSHOT_CACHE.get("key") == cache_key and now - float(_CRM_SNAPSHOT_CACHE.get("ts") or 0.0) < CRM_CACHE_TTL_S:
            return list(_CRM_SNAPSHOT_CACHE.get("rows") or []), dict(_CRM_SNAPSHOT_CACHE.get("summary") or {})

    base_rows = _crm_fetch_base_users(q, limit=max(limit, CRM_DEFAULT_LIMIT))
    user_ids: list[int] = []
    clean_rows: list[dict] = []
    seen: set[int] = set()
    for row in base_rows:
        try:
            uid = int(row.get("user_id") or 0)
        except Exception:
            uid = 0
        if not uid or uid in seen:
            continue
        seen.add(uid)
        user_ids.append(uid)
        clean_rows.append(dict(row))

    blocked_ids = _web_blocked_ids_for_users(clean_rows)
    tc_metrics = _crm_text_cache_metrics(user_ids, scan_limit=max(500, limit * 20))
    out: list[dict] = []
    scored_rows: list[dict] = []
    for row in clean_rows:
        uid = int(row.get("user_id") or 0)
        row.update(tc_metrics.get(uid) or {})
        row["blocked"] = uid in blocked_ids
        row["age_days"] = _crm_age_days(row.get("last_active") or row.get("last_text_at"))
        seg_label, seg_kind = _crm_segment_label(row)
        row["crm_segment"] = seg_label
        row["crm_segment_kind"] = seg_kind
        row["quality_score"] = _crm_quality_score(row)
        scored_rows.append(row)
        if _crm_segment_matches(row, segment):
            out.append(row)

    out.sort(key=lambda r: (int(r.get("quality_score") or 0), str(r.get("last_active") or r.get("last_text_at") or "")), reverse=True)
    out = out[:limit]

    summary = {
        "total_loaded": len(scored_rows),
        "shown": len(out),
        "active": sum(1 for r in scored_rows if _crm_segment_matches(r, "active")),
        "blocked": len(blocked_ids),
        "risk": sum(1 for r in scored_rows if int(r.get("quality_score") or 0) < 35 or str(r.get("crm_segment") or "").startswith("inactive")),
        "power": sum(1 for r in scored_rows if int(r.get("quality_score") or 0) >= 80),
        "cache_ttl_s": CRM_CACHE_TTL_S,
    }
    with _CRM_SNAPSHOT_LOCK:
        _CRM_SNAPSHOT_CACHE.update({"ts": time.monotonic(), "key": cache_key, "rows": list(out), "summary": dict(summary)})
    return out, summary


def _crm_clear_cache() -> None:
    with _CRM_SNAPSHOT_LOCK:
        _CRM_SNAPSHOT_CACHE.update({"ts": 0.0, "key": "", "rows": [], "summary": {}})


def _crm_segment_tabs(active_segment: str, q: str, limit: int) -> str:
    segments = [
        ("all", "All"),
        ("power", "Power"),
        ("active", "Active"),
        ("warm", "Warm"),
        ("inactive7", "Inactive 7d"),
        ("inactive30", "Inactive 30d"),
        ("risk", "Risk"),
        ("blocked", "Blocked"),
    ]
    return "".join(
        f"<a class='btn {'secondary' if key != active_segment else ''}' href='{_web_url('/admin/crm', segment=key, q=q, limit=limit)}'>{_web_h(label)}</a>"
        for key, label in segments
    )


def _crm_rows_html(rows: list[dict], csrf: str, return_input: str) -> tuple[str, str]:
    table_rows: list[str] = []
    mobile_cards: list[str] = []
    for row in rows:
        uid = int(row.get("user_id") or 0)
        username = _admin_username_display(row)
        detail_url = url_for("web_admin_user_detail", user_id=uid)
        last_seen = row.get("last_active") or row.get("last_text_at") or ""
        age = row.get("age_days")
        age_txt = "unknown" if age is None else ("today" if age == 0 else f"{age}d ago")
        score = int(row.get("quality_score") or 0)
        seg = _web_badge(row.get("crm_segment") or "unknown", row.get("crm_segment_kind") or "muted")
        block_action = "unblock" if row.get("blocked") else "block"
        block_label = "Unblock" if row.get("blocked") else "Block"
        block_cls = "ok" if row.get("blocked") else "danger"
        action_forms = f"""
          <a class='btn secondary' href='{detail_url}'>Detail</a>
          <form class='inline-form' method='post' action='/admin/users/action' data-confirm='{_web_h(block_label)} this user?'>
            <input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='user_id' value='{uid}'><input type='hidden' name='action' value='{block_action}'><button class='{block_cls}'>{block_label}</button>
          </form>
        """
        table_rows.append(f"""
        <tr>
          <td><a href='{detail_url}'><code>{uid}</code></a><br><span class='muted'>{_web_h(username)}</span></td>
          <td>{_crm_score_badge(score)}<br>{seg}</td>
          <td><b>{_web_h(row.get('recent_messages') or 0)}</b><br><span class='muted'>{_web_h(row.get('recent_chars') or 0)} chars</span></td>
          <td>{_web_h(_web_dt(last_seen))}<br><span class='muted'>{_web_h(age_txt)}</span></td>
          <td>{_web_h(_web_short(row.get('last_text') or 'No cached text', 160))}</td>
          <td><div class='actions'>{action_forms}</div></td>
        </tr>
        """)
        mobile_cards.append(f"""
        <div class='touch-item'>
          <div class='actions' style='justify-content:space-between'><b><a href='{detail_url}'><code>{uid}</code></a></b>{_crm_score_badge(score)}</div>
          <div class='muted'>{_web_h(username)} · {seg} · {age_txt}</div>
          <div class='help'>{_web_h(_web_short(row.get('last_text') or 'No cached text', 150))}</div>
          <div class='actions' style='margin-top:10px'>{action_forms}<button type='button' class='btn ghost' data-copy='{uid}'>Copy ID</button></div>
        </div>
        """)
    empty = '<tr><td colspan=6><div class="empty">No CRM users found for this filter.</div></td></tr>'
    return "".join(table_rows) or empty, "".join(mobile_cards) or '<div class="empty">No CRM users found for this filter.</div>'


def _crm_csv(rows: list[dict]) -> str:
    header = ["user_id", "username", "score", "segment", "blocked", "recent_messages", "recent_chars", "last_seen", "last_text"]
    lines = [",".join(header)]
    def esc(v: Any) -> str:
        s = str(v if v is not None else "")
        s = s.replace('"', '""')
        return f'"{s}"' if any(ch in s for ch in [",", "\n", "\r", '"']) else s
    for r in rows:
        last_seen = r.get("last_active") or r.get("last_text_at") or ""
        values = [
            r.get("user_id"),
            r.get("username") or r.get("first_name") or "",
            r.get("quality_score"),
            r.get("crm_segment"),
            bool(r.get("blocked")),
            r.get("recent_messages"),
            r.get("recent_chars"),
            _web_dt(last_seen),
            r.get("last_text") or "",
        ]
        lines.append(",".join(esc(v) for v in values))
    return "\n".join(lines) + "\n"
def _web_send_telegram_message(
    chat_id: int,
    text: str,
    *,
    admin_id: int | None = None,
    client: httpx.Client | None = None,
    parse_mode: str | None = "plain",
) -> tuple[bool, str]:
    if not TELEGRAM_BOT_TOKEN:
        return False, "TELEGRAM_BOT_TOKEN missing."

    try:
        prepared_text, prepared_mode = _broadcast_prepare_text(
            text,
            parse_mode,
            max_chars=TELE_MSG_LIMIT,
        )
    except ValueError as exc:
        return False, str(exc)

    if not prepared_text:
        return False, "Message is empty."

    created_client = False
    tg_client = client
    try:
        if tg_client is None:
            tg_client = httpx.Client(timeout=20)
            created_client = True

        last_error = ""
        for telegram_parse_mode in _broadcast_candidate_parse_modes(prepared_text, prepared_mode):
            payload = {
                "chat_id": int(chat_id),
                "text": prepared_text,
                "disable_web_page_preview": True,
            }
            if telegram_parse_mode:
                payload["parse_mode"] = telegram_parse_mode

            resp = tg_client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json=payload,
                timeout=20,
            )
            if resp.status_code == 403:
                blocker_admin_id = int(admin_id or 0)
                if not blocker_admin_id:
                    with suppress(Exception):
                        blocker_admin_id = _web_current_admin_id()
                db_user_set_blocked(int(chat_id), blocker_admin_id, True, "Telegram Forbidden from web send")
            if not (200 <= resp.status_code < 300):
                last_error = f"Telegram HTTP {resp.status_code}: {resp.text[:300]}"
                if telegram_parse_mode and _is_telegram_parse_error(last_error):
                    continue
                return False, last_error

            try:
                response_payload = resp.json()
            except Exception:
                response_payload = {}
            if response_payload.get("ok"):
                return True, "sent"

            last_error = str(response_payload)[:300]
            if telegram_parse_mode and _is_telegram_parse_error(last_error):
                continue
            return False, last_error or "Telegram send failed."

        return False, last_error or "Telegram parse mode rejected message."
    except Exception as exc:
        return False, str(exc)
    finally:
        if created_client and tg_client is not None:
            with suppress(Exception):
                tg_client.close()


@app_flask.route("/admin/users")
@web_admin_required
def web_admin_users():
    q = (request.args.get("q") or "").strip()
    page = max(0, _web_int(request.args.get("page"), 0))
    page_size = max(10, min(200, _web_int(request.args.get("page_size"), WEB_TABLE_PAGE_SIZE)))
    rows = _web_fetch_users(q, page=page, page_size=page_size)
    blocked_ids = _web_blocked_ids_for_users(rows)
    table_rows = []
    mobile_cards = []
    csrf = _web_csrf_token()
    return_input = _web_return_input()
    for row in rows:
        uid = _web_int(row.get("user_id"), 0)
        blocked = uid in blocked_ids
        username = row.get("username") or "-"
        detail_url = url_for("web_admin_user_detail", user_id=uid)
        profile_link = f"<a href='{detail_url}'><code>{uid}</code></a>"
        action_forms = f"""
          <form class='inline-form' method='post' action='/admin/users/action' {'data-confirm="Unblock this user?"' if blocked else 'data-confirm="Block this user?"'}>
            <input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='user_id' value='{uid}'><input type='hidden' name='action' value='{'unblock' if blocked else 'block'}'><button class='{'ok' if blocked else 'danger'}'>{'Unblock' if blocked else 'Block'}</button>
          </form>
          <a class='btn secondary' href='{detail_url}'>Detail</a>
        """
        table_rows.append(f"""
        <tr>
          <td>{profile_link}</td>
          <td>{_web_h(username)}</td>
          <td>{_web_h(row.get('gender') or '-')}</td>
          <td>{_web_h(row.get('speed') or '-')}</td>
          <td>{_web_h(_web_dt(row.get('last_active') or '-'))}</td>
          <td>{_web_badge('blocked' if blocked else 'active', 'danger' if blocked else 'ok')}</td>
          <td><div class='actions'>{action_forms}</div></td>
        </tr>
        """)
        mobile_cards.append(f"""
        <div class='touch-item'>
          <div class='actions' style='justify-content:space-between'><b><a href='{detail_url}'><code>{uid}</code></a></b>{_web_badge('blocked' if blocked else 'active', 'danger' if blocked else 'ok')}</div>
          <div class='muted'>{_web_h(username)} · Voice {_web_h(row.get('gender') or '-')} · Speed {_web_h(row.get('speed') or '-')}</div>
          <div class='help'>Last active: {_web_h(_web_dt(row.get('last_active') or '-'))}</div>
          <div class='actions' style='margin-top:10px'>{action_forms}</div>
        </div>
        """)
    next_link = f"<a class='btn secondary' href='{_web_url('/admin/users', page=page+1, page_size=page_size, q=q)}'>Next</a>" if not q and len(rows) >= page_size else ""
    prev_link = f"<a class='btn secondary' href='{_web_url('/admin/users', page=max(0,page-1), page_size=page_size, q=q)}'>Previous</a>" if page > 0 else ""
    body = f"""
    <div class='actions' style='justify-content:space-between;margin-bottom:12px'>
      <div class='actions'><a class='btn secondary' href='/admin/schedules/calendar'>🗓 Calendar View</a><a class='btn secondary' href='/admin/analytics'>📈 Analytics V2</a></div>
      <span class='muted'>List view · {_web_h(_fmt_local_time_hint())}</span>
    </div>
    <div class='card'>
      <form method='get' class='row3'>
        <div class='field'><label>Search user ID or username <span class='kbd'>Enter</span></label><input name='q' value='{_web_h(q)}' placeholder='123456789 or @username' autofocus></div>
        <div class='field'><label>Rows per page</label><select name='page_size'>
          {''.join(f"<option value='{n}' {'selected' if page_size==n else ''}>{n}</option>" for n in (25,50,100,200))}
        </select></div>
        <div class='field'><label>&nbsp;</label><div class='actions'><button>Search</button><a class='btn secondary' href='/admin/users'>Reset</a></div></div>
      </form>
    </div>
    <div class='card'><div class='actions' style='justify-content:space-between'><h2>Users</h2><span class='muted'>Page {page + 1}{' · filtered' if q else ''}</span></div>
      <div class='table-wrap desktop-table'><table class='table'><thead><tr><th>ID</th><th>Username</th><th>Voice</th><th>Speed</th><th>Last active</th><th>Status</th><th>Actions</th></tr></thead><tbody>{''.join(table_rows) or '<tr><td colspan=7><div class="empty">No users found.</div></td></tr>'}</tbody></table></div>
      <div class='mobile-card'>{''.join(mobile_cards) or '<div class="empty">No users found.</div>'}</div>
      <p class='actions'>{prev_link}{next_link}</p>
    </div>
    <div class='card'><h2>Send direct message</h2><form method='post' action='/admin/users/action' data-confirm='Send this message now?'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='send_message'><div class='row'><div class='field'><label>User ID</label><input name='user_id' inputmode='numeric' required></div><div class='field'><label>&nbsp;</label><button>Send</button></div></div><div class='field'><label>Message <span><span data-count-target='#direct-message-text'>0</span>/{TELE_MSG_LIMIT}</span></label><textarea id='direct-message-text' name='message' maxlength='{TELE_MSG_LIMIT}' required></textarea></div></form></div>
    """
    return _web_render("Users", body, active="users")





@app_flask.route("/admin/crm")
@web_admin_required
def web_admin_crm():
    q = (request.args.get("q") or "").strip()
    segment = (request.args.get("segment") or "all").strip().lower() or "all"
    limit = max(20, min(500, _web_int(request.args.get("limit"), CRM_DEFAULT_LIMIT)))
    force = str(request.args.get("refresh") or "").lower() in {"1", "true", "yes"}
    rows, summary = db_crm_users_snapshot(q=q, segment=segment, limit=limit, force=force)
    csrf = _web_csrf_token()
    return_input = _web_return_input()
    table_html, mobile_html = _crm_rows_html(rows, csrf, return_input)
    tabs = _crm_segment_tabs(segment, q, limit)
    export_url = _web_url("/admin/crm/export.csv", q=q, segment=segment, limit=limit)
    refresh_url = _web_url("/admin/crm", q=q, segment=segment, limit=limit, refresh=1)
    body = f"""
    <div class='actions' style='justify-content:space-between;margin-bottom:12px'>
      <div class='actions'><a class='btn secondary' href='/admin/users'>👥 Users</a><a class='btn secondary' href='/admin/broadcast'>📣 Broadcast</a><a class='btn secondary' href='{_web_h(export_url)}'>Export CSV</a></div>
      <span class='muted'>CRM cache TTL {int(CRM_CACHE_TTL_S)}s · {_web_h(_fmt_local_time_hint())}</span>
    </div>
    <div class='grid'>
      {_web_count_card('crm_shown', 'Shown', summary.get('shown', 0), f"loaded {summary.get('total_loaded', 0)} users", 'ok' if summary.get('shown') else '')}
      {_web_count_card('crm_power', 'Power users', summary.get('power', 0), 'score 80+', 'ok' if summary.get('power') else '')}
      {_web_count_card('crm_risk', 'At risk', summary.get('risk', 0), 'low score or inactive', 'warn' if summary.get('risk') else '')}
      {_web_count_card('crm_blocked', 'Blocked', summary.get('blocked', 0), 'skipped during sends', 'danger' if summary.get('blocked') else '')}
    </div>
    <div class='card'>
      <div class='actions' style='justify-content:space-between'><div><h2>User Quality / CRM Panel</h2><p class='muted'>Scores users from <code>user_prefs</code>, <code>text_cache</code>, and <code>blocked_users</code> with batched queries and short cache.</p></div><div class='actions'><a class='btn secondary' href='{_web_h(refresh_url)}'>Refresh</a><form class='inline-form' method='post' action='/admin/crm/action'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='return_to' value='/admin/crm'><input type='hidden' name='action' value='clear_cache'><button class='secondary'>Clear CRM Cache</button></form></div></div>
      <form method='get' class='row3'>
        <div class='field'><label>Search user ID or username</label><input name='q' value='{_web_h(q)}' placeholder='123456789 or @username'></div>
        <div class='field'><label>Segment</label><select name='segment'>
          {''.join(f"<option value='{k}' {'selected' if segment==k else ''}>{label}</option>" for k,label in [('all','All'),('power','Power users'),('active','Active today'),('warm','Warm 2-7d'),('inactive7','Inactive 7-30d'),('inactive30','Inactive 30d+'),('risk','At risk'),('blocked','Blocked')])}
        </select></div>
        <div class='field'><label>Limit</label><select name='limit'>{''.join(f"<option value='{n}' {'selected' if limit==n else ''}>{n}</option>" for n in (50,80,120,200,500))}</select></div>
        <div class='field'><label>&nbsp;</label><div class='actions'><button>Apply Filter</button><a class='btn secondary' href='/admin/crm'>Reset</a></div></div>
      </form>
      <div class='actions' style='margin-top:12px'>{tabs}</div>
    </div>
    <div class='card'>
      <div class='actions' style='justify-content:space-between'><h2>CRM Users</h2><span class='muted'>Sorted by quality score</span></div>
      <div class='table-wrap desktop-table'><table class='table'><thead><tr><th>User</th><th>Quality</th><th>Recent Usage</th><th>Last Seen</th><th>Latest Text</th><th>Actions</th></tr></thead><tbody>{table_html}</tbody></table></div>
      <div class='mobile-card'>{mobile_html}</div>
    </div>
    <div class='grid2'>
      <div class='card'><h2>How score works</h2><div class='touch-list'>
        <div class='touch-item'><b>Recency</b><div class='muted'>Recent last_active / last text_cache activity increases score.</div></div>
        <div class='touch-item'><b>Engagement</b><div class='muted'>More recent text_cache messages and characters increase score.</div></div>
        <div class='touch-item'><b>Risk</b><div class='muted'>Blocked users score 0; inactive users appear in risk segments.</div></div>
      </div></div>
      <div class='card'><h2>CRM Actions</h2><p class='muted'>Use this panel to find high-value users, inactive users, blocked users, and users who need follow-up. For mass sends, open Broadcast and use a test message first.</p><div class='actions'><a class='btn' href='/admin/broadcast'>Open Broadcast</a><a class='btn secondary' href='/admin/users'>Open Users</a></div></div>
    </div>
    """
    return _web_render("User Quality CRM", body, active="crm")


@app_flask.route("/admin/crm/export.csv")
@web_admin_required
def web_admin_crm_export_csv():
    q = (request.args.get("q") or "").strip()
    segment = (request.args.get("segment") or "all").strip().lower() or "all"
    limit = max(20, min(500, _web_int(request.args.get("limit"), CRM_DEFAULT_LIMIT)))
    rows, _summary = db_crm_users_snapshot(q=q, segment=segment, limit=limit)
    filename = f"crm-users-{segment}-{datetime.now(APP_TIMEZONE).strftime('%Y%m%d-%H%M')}.csv"
    return Response(
        _crm_csv(rows),
        status=200,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app_flask.route("/admin/crm/action", methods=["POST"])
@web_admin_required
def web_admin_crm_action():
    _web_check_csrf()
    action = str(request.form.get("action") or "").strip()
    if action == "clear_cache":
        _crm_clear_cache()
        flask_flash("CRM cache cleared.", "success")
    else:
        flask_flash("Unknown CRM action.", "error")
    return redirect(_web_safe_return("web_admin_crm"))




def _admin_username_display(row: dict | None) -> str:
    """Return a clean Telegram username for admin screens.

    Telegram gives `username` without @.  Store/display it as @username when
    it looks like a real Telegram handle, but do not invent @ for a normal
    first name fallback.
    """
    row = row or {}
    raw_username = str(row.get("username") or "").strip()
    first_name = str(row.get("first_name") or "").strip()

    if raw_username and raw_username != "-":
        clean = raw_username.lstrip("@").strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{4,31}", clean):
            return f"@{clean}"
        return raw_username

    return first_name or "-"

@app_flask.route("/admin/users/<int:user_id>")
@web_admin_required
def web_admin_user_detail(user_id: int):
    row = db_user_detail(int(user_id))
    blocked = bool(row.get("blocked"))
    csrf = _web_csrf_token()
    return_input = _web_return_input()
    username = _admin_username_display(row)
    history_rows = list(row.get("history") or [])
    metrics = _web_user_detail_metrics(row)
    tg_link = f"tg://user?id={int(user_id)}"
    status_badge = _web_badge("blocked" if blocked else "active", "danger" if blocked else "ok")

    history_html = []
    for h in reversed(history_rows[-60:]):
        created = _web_dt(h.get("created_at") or "")
        msg_id = h.get("message_id") or "-"
        chat_id = h.get("chat_id") or user_id
        content_raw = _history_compact_text(h.get("content") or h.get("original_text") or "", 1100)
        content = _web_h(content_raw)
        history_html.append(
            "<div class='history-item'>"
            f"<div class='history-meta'><span>{_web_h(created)}</span><span>chat <code>{_web_h(chat_id)}</code></span><span>msg <code>{_web_h(msg_id)}</code></span><span>{_web_badge('text_cache','info')}</span></div>"
            f"<div class='whitespace-pre-wrap'>{content or '<span class=\"muted\">empty</span>'}</div>"
            "</div>"
        )
    if not history_html:
        history_html.append("<div class='empty'>No recent text_cache history found for this user.</div>")

    block_action = "unblock" if blocked else "block"
    block_button_cls = "ok" if blocked else "danger"
    block_button_text = "Unblock" if blocked else "Block"
    block_confirm = "Unblock this user?" if blocked else "Block this user?"
    error_html = f"<div class='flash warning'>{_web_h(row.get('error'))}</div>" if row.get("error") else ""
    last_text_html = _web_h(metrics.get("last_text") or "No recent message")

    body = f"""
    <div class='actions' style='margin-bottom:12px'>
      <a class='btn secondary' href='/admin/users'>← Back</a>
      <a class='btn secondary' href='/admin/users?q={int(user_id)}'>Search ID</a>
      <a class='btn ghost' href='{_web_h(tg_link)}'>Open Telegram</a>
      <button type='button' class='btn ghost' data-copy='{int(user_id)}'>Copy ID</button>
    </div>

    <div class='detail-hero'>
      <div class='card'>
        <div class='actions' style='justify-content:space-between'>
          <div><h2>User Detail Upgrade</h2><p class='muted'>Profile, usage summary, and text_cache history</p></div>
          {status_badge}
        </div>
        <div class='mini-grid' style='margin:14px 0'>
          <div class='mini-stat'><span class='muted'>Messages</span><b>{_web_h(metrics.get('messages'))}</b><small class='muted'>loaded from text_cache</small></div>
          <div class='mini-stat'><span class='muted'>Text size</span><b>{_web_h(metrics.get('chars'))}</b><small class='muted'>characters loaded</small></div>
          <div class='mini-stat'><span class='muted'>Status</span><b>{'Blocked' if blocked else 'Active'}</b><small class='muted'>{_web_h(_web_dt(row.get('last_active') or '')) or 'no last_active'}</small></div>
        </div>
        <div class='table-wrap'><table class='table'>
          <tr><td>User ID</td><td><code>{int(user_id)}</code></td></tr>
          <tr><td>Username</td><td>{_web_h(username)}</td></tr>
          <tr><td>Status</td><td>{status_badge}</td></tr>
          <tr><td>Voice</td><td>{_web_h(row.get('gender') or 'female')}</td></tr>
          <tr><td>Speed</td><td>{_web_h(row.get('speed') or DEFAULT_SPEED)}</td></tr>
          <tr><td>Last active</td><td>{_web_h(_web_dt(row.get('last_active') or ''))}</td></tr>
          <tr><td>First cached message</td><td>{_web_h(metrics.get('first_seen') or '-')}</td></tr>
          <tr><td>Last cached message</td><td>{_web_h(metrics.get('last_seen') or '-')}</td></tr>
        </table></div>
        {error_html}
      </div>

      <div class='card'>
        <h2>Mobile Quick Actions</h2>
        <p class='muted'>Large touch controls for phone admin work.</p>
        <div class='sticky-actions'><div class='actions'>
          <form method='post' action='/admin/users/action' data-confirm='{_web_h(block_confirm)}'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='user_id' value='{int(user_id)}'><input type='hidden' name='action' value='{block_action}'><button class='{block_button_cls}'>{block_button_text}</button></form>
          <form method='post' action='/admin/users/action' data-confirm='Reset this user voice preferences?'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='user_id' value='{int(user_id)}'><input type='hidden' name='action' value='reset'><button class='secondary'>Reset prefs</button></form>
          <form method='post' action='/admin/users/action' data-confirm='Clear text_cache and conversation history for this user?'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='user_id' value='{int(user_id)}'><input type='hidden' name='action' value='clear_history'><button class='secondary'>Clear history</button></form>
        </div></div>
        <form method='post' action='/admin/users/action' data-confirm='Send this message now?'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='send_message'><input type='hidden' name='user_id' value='{int(user_id)}'><div class='field'><label>Direct message <span><span data-count-target='#detail-direct-message'>0</span>/{TELE_MSG_LIMIT}</span></label><textarea id='detail-direct-message' name='message' maxlength='{TELE_MSG_LIMIT}' placeholder='Write message to this user...' required></textarea></div><button>Send Message</button></form>
      </div>
    </div>

    <div class='grid2'>
      <div class='card'><h2>Latest Message Preview</h2><p class='muted whitespace-pre-wrap'>{last_text_html}</p></div>
      <div class='card'><h2>Admin Notes</h2><div class='touch-list'>
        <div class='touch-item'><b>History source</b><div class='muted'>Recent history uses <code>text_cache</code>, not conversation_history.</div></div>
        <div class='touch-item'><b>Broadcast safety</b><div class='muted'>Blocked users are skipped automatically during sends.</div></div>
      </div></div>
    </div>

    <div class='card'>
      <div class='actions' style='justify-content:space-between'><div><h2>Recent text_cache history</h2><p class='muted'>Newest first · {len(history_rows)} loaded</p></div><a class='btn secondary' href='/admin/users?q={int(user_id)}'>Open in users list</a></div>
      {''.join(history_html)}
    </div>
    """
    return _web_render("User Detail", body, active="users")

@app_flask.route("/admin/users/action", methods=["POST"])
@web_admin_required
def web_admin_users_action():
    _web_check_csrf()
    admin_id = _web_current_admin_id()
    user_id = _web_int(request.form.get("user_id"), 0)
    action = str(request.form.get("action") or "").strip()
    if not user_id:
        flask_flash("Missing user ID.", "error")
        return redirect(_web_safe_return("web_admin_users"))
    if action == "block":
        ok, msg = db_user_set_blocked(user_id, admin_id, True, "Blocked from web dashboard")
    elif action == "unblock":
        ok, msg = db_user_set_blocked(user_id, admin_id, False)
    elif action == "reset":
        ok, msg = db_user_reset_prefs(user_id)
    elif action == "clear_history":
        db_history_clear(user_id)
        ok, msg = True, "history clear queued"
    elif action == "send_message":
        ok, msg = _web_send_telegram_message(user_id, request.form.get("message") or "", admin_id=_web_current_admin_id())
    else:
        ok, msg = False, "Unknown action."
    flask_flash(("OK: " if ok else "ERROR: ") + msg, "success" if ok else "error")
    return redirect(_web_safe_return("web_admin_users"))


def _web_sched_list(status: str = "all", q: str = "", limit: int = 150) -> list[dict]:
    if not supabase:
        return []
    status = (status or "all").strip().lower()
    q = (q or "").strip().lower()
    def _query():
        builder = supabase.table("scheduled_broadcasts").select("*").order("broadcast_at", desc=True).limit(max(1, int(limit)))
        if status not in ("", "all", "preview"):
            builder = builder.eq("status", status)
        elif status == "preview":
            builder = builder.eq("status", SCHED_STATUS_PENDING)
        return builder.execute()
    res = db_call_sync("web_sched_list", _query, default=None, attempts=2, critical=False)
    rows = list(getattr(res, "data", None) or [])
    if status == "preview":
        rows = [r for r in rows if _sched_is_draft(r)]
    elif status == "pending":
        rows = [r for r in rows if _sched_is_confirmed_pending(r)]
    if q:
        rows = [r for r in rows if q in str(r.get("id") or "").lower() or q in str(r.get("caption") or "").lower() or q in str(r.get("plain_text") or "").lower()]
    return rows


def _web_sched_confirm_any(row_id: int, admin_id: int) -> tuple[bool, str]:
    """Confirm a schedule from the web dashboard using the same safe path as Telegram.

    This avoids owner-bypass bugs and keeps the DB-compatible draft design:
    status="pending" plus SCHED_DRAFT_MARKER until confirmation.
    """
    ok, reason, _row = db_sched_confirm(int(row_id), int(admin_id))
    messages = {
        "confirmed": "confirmed",
        "already_confirmed": "already confirmed",
        "not_found": "Schedule not found.",
        "not_owner": "Schedule belongs to another admin.",
        "expired": "Schedule expired and was cancelled.",
        "race_lost": "Schedule changed before confirmation. Refresh and try again.",
    }
    return bool(ok), messages.get(str(reason), f"Cannot confirm schedule: {reason}")


def _web_sched_cancel_any(row_id: int, admin_id: int) -> tuple[bool, str]:
    row = db_sched_fetch_one(int(row_id))
    if not row:
        return False, "Schedule not found."
    if int(row.get("admin_id") or 0) != int(admin_id):
        return False, "Schedule belongs to another admin."
    if str(row.get("status") or "").lower() != SCHED_STATUS_PENDING:
        return False, f"Cannot cancel schedule with status {row.get('status')}."
    db_sched_set_status(int(row_id), SCHED_STATUS_CANCELLED, error_msg="Cancelled from web dashboard")
    return True, "cancelled"


def _web_sched_create(
    admin_id: int,
    broadcast_at: datetime,
    text: str,
    photo_file_id: str = "",
    caption: str = "",
    confirmed: bool = False,
    parse_mode: str | None = "auto",
) -> tuple[bool, str]:
    if not supabase:
        return False, "Supabase is not configured."
    text = (text or "").strip()
    photo_file_id = (photo_file_id or "").strip()
    caption = (caption or "").strip()
    parse_mode = _broadcast_normalize_parse_mode(parse_mode)
    if photo_file_id and not caption and text:
        caption = text
    if photo_file_id and caption:
        caption = _broadcast_apply_format_directive(caption, parse_mode)
    elif not photo_file_id and text:
        text = _broadcast_apply_format_directive(text, parse_mode)
    if not photo_file_id and not text:
        return False, "Broadcast text is required when no photo_file_id is provided."
    if photo_file_id and len(caption) > 1024:
        return False, "Caption too long for Telegram photo."
    if not photo_file_id and len(text) > TELE_MSG_LIMIT:
        return False, "Text too long for Telegram."
    if _sched_to_utc(broadcast_at) <= datetime.now(timezone.utc):
        return False, "Broadcast time must be in the future."
    row = {"admin_id": int(admin_id), "photo_file_id": photo_file_id or None, "caption": caption if photo_file_id else None, "plain_text": None if photo_file_id else text, "broadcast_at": _sched_iso(broadcast_at), "status": SCHED_STATUS_PENDING, "error_msg": None if confirmed else SCHED_DRAFT_MARKER}
    res = db_call_sync("web_sched_create", lambda: supabase.table("scheduled_broadcasts").insert(row).execute(), default=None, attempts=2, critical=False)
    if getattr(res, "data", None):
        return True, f"created schedule #{res.data[0].get('id')}"
    return False, "insert failed"



# ---------------------------------------------------------------------------
# Admin Dashboard V5 — Analytics V2 + Schedule Calendar
# ---------------------------------------------------------------------------

def _web_safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return int(default)


def _web_pct(part: int | float, total: int | float) -> int:
    try:
        total_f = float(total or 0)
        if total_f <= 0:
            return 0
        return max(0, min(100, int((float(part or 0) / total_f) * 100)))
    except Exception:
        return 0


def _web_metric_bar(label: str, value: int, total: int, *, kind: str = "info") -> str:
    pct = _web_pct(value, total)
    return (
        "<div class='mini-stat'>"
        f"<div class='actions' style='justify-content:space-between'><b>{_web_h(label)}</b><span class='muted'>{_web_h(value)}/{_web_h(total)}</span></div>"
        f"<div class='progress' style='width:100%;margin-top:10px'><span style='width:{pct}%'></span></div>"
        f"<small class='muted'>{pct}%</small>"
        "</div>"
    )


def _web_schedule_status_key(row: dict) -> str:
    if _sched_is_draft(row):
        return "preview"
    return str(row.get("status") or "unknown").strip().lower() or "unknown"


def _web_schedule_select_fields() -> str:
    return "id,admin_id,broadcast_at,plain_text,caption,photo_file_id,status,error_msg,sent_count,failed_count,blocked_count,created_at"


def _web_sched_fetch_recent(limit: int = 5000) -> list[dict]:
    if not supabase:
        return []
    res = db_call_sync(
        "web_sched_recent_v5",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .select(_web_schedule_select_fields())
            .order("broadcast_at", desc=True)
            .limit(max(1, min(10000, int(limit or 5000))))
            .execute()
        ),
        default=None,
        attempts=2,
        critical=False,
    )
    return list(getattr(res, "data", None) or [])


def _web_sched_fetch_range(start_utc: datetime, end_utc: datetime, limit: int = 2000) -> list[dict]:
    if not supabase:
        return []
    start_iso = _sched_iso(start_utc)
    end_iso = _sched_iso(end_utc)
    res = db_call_sync(
        "web_sched_range_v5",
        lambda: (
            supabase.table("scheduled_broadcasts")
            .select(_web_schedule_select_fields())
            .gte("broadcast_at", start_iso)
            .lt("broadcast_at", end_iso)
            .order("broadcast_at")
            .limit(max(1, min(5000, int(limit or 2000))))
            .execute()
        ),
        default=None,
        attempts=2,
        critical=False,
    )
    return list(getattr(res, "data", None) or [])


def _web_sched_fetch_analytics_window(
    since_utc: datetime,
    limit: int = 20000,
    *,
    until_utc: datetime | None = None,
) -> list[dict]:
    """Fetch analytics rows using broadcast_at window filtering and pagination.

    Changes:
    - Analytics must be based on the scheduled send time (`broadcast_at`), not
      row creation time (`created_at`). This keeps charts, global stats, and
      recent stats aligned with the real broadcast window.
    - Supports an optional upper bound so the dashboard can count exactly the
      selected local-day window and avoid leaking future rows into stats.
    - Falls back to recent broadcast rows if the DB-side window query fails.
    """
    if not supabase:
        return []

    since_utc = _sched_to_utc(since_utc)
    until_utc = _sched_to_utc(until_utc) if until_utc else None
    since_iso = _sched_iso(since_utc)
    until_iso = _sched_iso(until_utc) if until_utc else None

    max_rows = max(
        100,
        min(
            100000,
            _env_int(
                "WEB_ANALYTICS_MAX_ROWS",
                int(limit or 20000),
                minimum=100,
                maximum=100000,
            ),
        ),
    )
    page_size = max(
        100,
        min(
            1000,
            _env_int(
                "WEB_ANALYTICS_PAGE_SIZE",
                1000,
                minimum=100,
                maximum=1000,
            ),
        ),
    )

    def _row_in_window(row: dict) -> bool:
        # Same rule as the DB query, used for fallback rows and defensive
        # filtering. Missing/invalid broadcast_at rows are excluded from
        # analytics because they cannot be placed on a broadcast timeline.
        dt = _sched_parse_iso(row.get("broadcast_at"))
        if not dt:
            return False
        if dt < since_utc:
            return False
        if until_utc and dt >= until_utc:
            return False
        return True

    rows: list[dict] = []
    query_failed = False
    offset = 0

    while len(rows) < max_rows:
        upper = min(offset + page_size - 1, max_rows - 1)

        def _query(lo: int = offset, hi: int = upper):
            # IMPORTANT: filter/order by broadcast_at, not created_at.
            builder = (
                supabase.table("scheduled_broadcasts")
                .select(_web_schedule_select_fields())
                .gte("broadcast_at", since_iso)
            )
            if until_iso:
                builder = builder.lt("broadcast_at", until_iso)
            return builder.order("broadcast_at", desc=True).range(lo, hi).execute()

        try:
            res = db_call_sync(
                f"web_sched_analytics_window:{offset}",
                _query,
                default=None,
                attempts=2,
                critical=False,
            )
        except Exception as exc:
            query_failed = True
            logger.warning("analytics broadcast_at query failed offset=%s: %s", offset, exc)
            break

        if res is None:
            query_failed = True
            logger.warning("analytics broadcast_at query returned no response offset=%s", offset)
            break

        batch = list(getattr(res, "data", None) or [])
        if not batch:
            break

        rows.extend(batch)
        if len(batch) < (upper - offset + 1):
            break
        offset += page_size

    if query_failed:
        # Keep the dashboard usable when Supabase/network has a temporary issue.
        # _web_sched_fetch_recent already orders by broadcast_at, then we apply
        # the exact same window in Python.
        try:
            fallback_rows = [row for row in _web_sched_fetch_recent(max_rows) if _row_in_window(row)]
            if fallback_rows:
                return fallback_rows[:max_rows]
        except Exception as exc:
            logger.warning("analytics fallback recent fetch failed: %s", exc)

        if rows:
            return [row for row in rows if _row_in_window(row)][:max_rows]

        raise RuntimeError("analytics DB query failed and no fallback rows were available")

    if len(rows) >= max_rows:
        logger.warning(
            "analytics row cap reached max_rows=%s since=%s until=%s; raise WEB_ANALYTICS_MAX_ROWS or add DB aggregates",
            max_rows,
            since_iso,
            until_iso or "-",
        )

    return [row for row in rows if _row_in_window(row)][:max_rows]


def _web_analytics_v2(days: int = 30) -> dict:
    """Build scheduled-broadcast analytics for the selected local-day window.

    Changes:
    - Uses the same broadcast_at window for global stats, recent stats, and the
      chart so the cards no longer disagree.
    - Initializes day_counts from local dates, avoiding UTC/local date drift.
    - Adds a small in-memory TTL cache to reduce dashboard DB pressure.
    - On DB failure, returns stale cached analytics when available; otherwise
      returns an empty safe payload instead of crashing the admin page.
    """
    try:
        days = max(1, min(365, int(days or 30)))
    except Exception:
        days = 30

    now_utc = datetime.now(timezone.utc)

    # Count complete local calendar days including today. Example for 30 days:
    # local midnight 29 days ago <= broadcast_at < tomorrow local midnight.
    since_local = (_local_now() - timedelta(days=days - 1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    until_local = (since_local + timedelta(days=days)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    since_utc = _local_to_utc(since_local)
    until_utc = _local_to_utc(until_local)

    cache_ttl = _env_float("WEB_ANALYTICS_CACHE_TTL_S", 45.0, minimum=30.0, maximum=60.0)
    cache_key = (days, _sched_iso(since_utc), _sched_iso(until_utc), APP_TIMEZONE_NAME)
    cache_lock = globals().setdefault("_WEB_ANALYTICS_CACHE_LOCK", threading.RLock())
    cache = globals().setdefault("_WEB_ANALYTICS_CACHE", OrderedDict())
    monotonic_now = time.monotonic()

    with cache_lock:
        cached = cache.get(cache_key)
        if cached:
            cached_at, cached_data = cached
            if monotonic_now - float(cached_at or 0) <= cache_ttl:
                cache.move_to_end(cache_key)
                data = dict(cached_data)
                data["cache_hit"] = True
                data["cache_stale"] = False
                return data

    def _new_day_counts() -> OrderedDict[str, dict]:
        # Initialize by local date directly. The old UTC + timedelta approach
        # could skip/duplicate labels around timezone conversions.
        buckets: OrderedDict[str, dict] = OrderedDict()
        for i in range(days):
            day_key = (since_local.date() + timedelta(days=i)).isoformat()
            buckets[day_key] = {"schedules": 0, "sent": 0, "failed": 0, "blocked": 0}
        return buckets

    def _empty_result(error: str = "") -> dict:
        return {
            "days": days,
            "rows": [],
            "status_counts": {},
            "total_schedules": 0,
            "sent": 0,
            "failed": 0,
            "blocked": 0,
            "delivery_total": 0,
            "delivery_rate": 0,
            "recent_sent": 0,
            "recent_failed": 0,
            "recent_blocked": 0,
            "recent_delivery_total": 0,
            "recent_delivery_rate": 0,
            "upcoming": 0,
            "overdue": 0,
            "previews": 0,
            "photo": 0,
            "text": 0,
            "day_counts": _new_day_counts(),
            "recent_errors": [],
            "cache_hit": False,
            "cache_stale": False,
            "analytics_error": error,
        }

    try:
        rows = _web_sched_fetch_analytics_window(since_utc, until_utc=until_utc)
    except Exception as exc:
        logger.warning("analytics build failed; using cache/empty fallback: %s", exc)
        with cache_lock:
            cached = cache.get(cache_key)
            if cached:
                cached_at, cached_data = cached
                cache.move_to_end(cache_key)
                data = dict(cached_data)
                data["cache_hit"] = True
                data["cache_stale"] = True
                data["analytics_error"] = str(exc)
                return data

        data = _empty_result(str(exc))
        with cache_lock:
            cache[cache_key] = (time.monotonic(), data)
        return data

    status_counts: dict[str, int] = {}
    day_counts = _new_day_counts()
    counted_rows: list[dict] = []

    sent = failed = blocked = 0
    upcoming = overdue = previews = photo = text = 0
    recent_errors: list[dict] = []

    for row in rows:
        dt = _sched_parse_iso(row.get("broadcast_at"))
        if not dt:
            continue

        # Defensive window check keeps fallback rows and any DB edge cases honest.
        if dt < since_utc or dt >= until_utc:
            continue

        local_key = _to_local_time(dt).date().isoformat()
        bucket = day_counts.get(local_key)
        if bucket is None:
            continue

        status = _web_schedule_status_key(row)
        sent_i = _web_safe_int(row.get("sent_count"))
        failed_i = _web_safe_int(row.get("failed_count"))
        blocked_i = _web_safe_int(row.get("blocked_count"))

        counted_rows.append(row)
        status_counts[status] = status_counts.get(status, 0) + 1

        sent += sent_i
        failed += failed_i
        blocked += blocked_i

        # Count schedules and delivery totals in the same local bucket.
        bucket["schedules"] += 1
        bucket["sent"] += sent_i
        bucket["failed"] += failed_i
        bucket["blocked"] += blocked_i

        if row.get("photo_file_id"):
            photo += 1
        else:
            text += 1

        if status == "preview":
            previews += 1

        if status == SCHED_STATUS_PENDING:
            if dt >= now_utc:
                upcoming += 1
            else:
                overdue += 1

        if status == "failed" or row.get("error_msg"):
            recent_errors.append(row)

    delivery_total = sent + failed + blocked

    data = {
        "days": days,
        "rows": counted_rows,
        "status_counts": status_counts,
        "total_schedules": len(counted_rows),
        "sent": sent,
        "failed": failed,
        "blocked": blocked,
        "delivery_total": delivery_total,
        "delivery_rate": _web_pct(sent, delivery_total),

        # Recent stats are intentionally equal to global stats for this page,
        # because both are now based on the selected last-N-days broadcast_at
        # window. This removes the old mismatch caused by mixed counting logic.
        "recent_sent": sent,
        "recent_failed": failed,
        "recent_blocked": blocked,
        "recent_delivery_total": delivery_total,
        "recent_delivery_rate": _web_pct(sent, delivery_total),

        "upcoming": upcoming,
        "overdue": overdue,
        "previews": previews,
        "photo": photo,
        "text": text,
        "day_counts": day_counts,
        "recent_errors": recent_errors[:12],
        "cache_hit": False,
        "cache_stale": False,
        "analytics_error": "",
    }

    with cache_lock:
        cache[cache_key] = (time.monotonic(), data)
        cache.move_to_end(cache_key)

        # Keep cache tiny and remove old entries for long-running Render workers.
        expired_before = time.monotonic() - max(cache_ttl, 60.0) * 4
        for key, (cached_at, _cached_data) in list(cache.items()):
            if float(cached_at or 0) < expired_before:
                cache.pop(key, None)
        while len(cache) > 24:
            cache.popitem(last=False)

    return data


def _web_analytics_chart_html(day_counts: OrderedDict[str, dict]) -> str:
    """Render the local-day analytics chart.

    Changes:
    - Uses _web_safe_int for robust counting.
    - Tooltip includes schedule, sent, failed, and blocked counts, matching the
      improved day_counts structure.
    """
    if not day_counts:
        return "<div class='empty'>No analytics data.</div>"

    max_value = max(
        (_web_safe_int(v.get("schedules")) for v in day_counts.values()),
        default=0,
    ) or 1

    bars = []
    for day, vals in day_counts.items():
        count = _web_safe_int(vals.get("schedules"))
        sent = _web_safe_int(vals.get("sent"))
        failed = _web_safe_int(vals.get("failed"))
        blocked = _web_safe_int(vals.get("blocked"))
        height = max(8, int((count / max_value) * 120)) if count else 8
        label = str(day)[5:] if len(str(day)) >= 10 else str(day)
        title = f"{day}: {count} schedule(s) · {sent} sent · {failed} failed · {blocked} blocked"

        bars.append(
            "<div style='display:flex;flex-direction:column;align-items:center;gap:7px;min-width:34px'>"
            f"<div title='{_web_h(title)}' style='height:130px;display:flex;align-items:end;width:100%'>"
            f"<div style='height:{height}px;width:100%;border-radius:12px 12px 4px 4px;background:linear-gradient(to top,rgba(59,130,246,.85),rgba(16,185,129,.85));border:1px solid rgba(255,255,255,.12)'></div>"
            "</div>"
            f"<small class='muted'>{_web_h(label)}</small>"
            "</div>"
        )

    return "<div style='display:flex;gap:10px;overflow-x:auto;padding:8px 2px 2px'>" + "".join(bars) + "</div>"

def _web_analytics_error_rows(rows: list[dict]) -> str:
    out = []
    for row in rows:
        content = row.get("caption") or row.get("plain_text") or ""
        out.append(
            f"<tr><td><code>#{_web_h(row.get('id'))}</code><br>{_web_status_badge(row.get('status'), row)}</td>"
            f"<td>{_web_h(_web_dt(row.get('broadcast_at')))}</td>"
            f"<td>{_web_h(_web_short(row.get('error_msg') or '-', 160))}</td>"
            f"<td>{_web_h(_web_short(content, 120))}</td>"
            f"<td><a class='btn secondary' href='/admin/schedules?q={_web_h(row.get('id'))}'>Open</a></td></tr>"
        )
    return "".join(out) or '<tr><td colspan=5><div class="empty">No recent schedule errors.</div></td></tr>'


def _web_calendar_month_params(raw_month: str | None = None) -> tuple[int, int]:
    raw = (raw_month or "").strip()
    if re.match(r"^\d{4}-\d{2}$", raw):
        year, month = raw.split("-", 1)
        try:
            y, m = int(year), int(month)
            if 1 <= m <= 12:
                return y, m
        except Exception:
            pass
    now = _local_now()
    return now.year, now.month


def _web_calendar_month_shift(year: int, month: int, delta: int) -> tuple[int, int]:
    month0 = (int(year) * 12 + (int(month) - 1)) + int(delta)
    return month0 // 12, (month0 % 12) + 1


def _web_calendar_bounds(year: int, month: int) -> tuple[datetime, datetime, datetime, datetime]:
    start_local = datetime(int(year), int(month), 1, tzinfo=APP_TIMEZONE)
    ny, nm = _web_calendar_month_shift(year, month, 1)
    end_local = datetime(ny, nm, 1, tzinfo=APP_TIMEZONE)
    return start_local, end_local, _local_to_utc(start_local), _local_to_utc(end_local)


def _web_schedule_calendar_html(year: int, month: int, rows: list[dict]) -> str:
    start_local, end_local, _start_utc, _end_utc = _web_calendar_bounds(year, month)
    days_in_month = (end_local.date() - start_local.date()).days
    first_weekday = start_local.weekday()  # Monday=0
    today = _local_now().date()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        dt = _sched_parse_iso(row.get("broadcast_at"))
        if not dt:
            continue
        key = _to_local_time(dt).date().isoformat()
        grouped.setdefault(key, []).append(row)

    weekday_heads = "".join(f"<div class='muted' style='font-weight:900;text-align:center;padding:8px'>{d}</div>" for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    cells = ["<div></div>" for _ in range(first_weekday)]
    for day in range(1, days_in_month + 1):
        d = datetime(year, month, day, tzinfo=APP_TIMEZONE).date()
        key = d.isoformat()
        items = grouped.get(key, [])
        border = "border:1px solid rgba(59,130,246,.45);box-shadow:0 0 0 1px rgba(59,130,246,.15) inset;" if d == today else "border:1px solid rgba(255,255,255,.10);"
        item_html = []
        for row in items[:4]:
            dt = _sched_parse_iso(row.get("broadcast_at"))
            local_time = _to_local_time(dt).strftime("%I:%M %p") if dt else ""
            content = row.get("caption") or row.get("plain_text") or ""
            status = _web_schedule_status_key(row)
            kind = {"done": "ok", "failed": "danger", "cancelled": "muted", "sending": "info", "preview": "warn", "pending": "ok"}.get(status, "muted")
            item_html.append(
                f"<a href='/admin/schedules?q={_web_h(row.get('id'))}' style='display:block;margin-top:7px;padding:8px;border-radius:14px;background:rgba(15,23,42,.72);border:1px solid rgba(255,255,255,.08)'>"
                f"<span class='badge badge-{kind}'>{_web_h(status)}</span><br>"
                f"<small class='muted'>{_web_h(local_time)} · #{_web_h(row.get('id'))}</small><br>"
                f"<span style='font-size:12px;color:#e2e8f0'>{_web_h(_web_short(content, 54))}</span>"
                "</a>"
            )
        more = len(items) - 4
        if more > 0:
            item_html.append(f"<div class='help'>+{more} more schedule(s)</div>")
        cells.append(
            f"<div style='{border}min-height:160px;border-radius:22px;padding:12px;background:rgba(15,23,42,.45)'>"
            f"<div class='actions' style='justify-content:space-between'><b>{day}</b><span class='badge badge-muted'>{len(items)}</span></div>"
            + "".join(item_html) +
            "</div>"
        )
    while len(cells) % 7:
        cells.append("<div></div>")
    return weekday_heads + "".join(cells)


@app_flask.route("/admin/analytics")
@web_admin_required
def web_admin_analytics():
    days = max(7, min(90, _web_int(request.args.get("days"), 30)))
    data = _web_analytics_v2(days)
    status_counts = data["status_counts"]
    total_schedules = max(1, int(data["total_schedules"] or 0))
    status_rows = "".join(
        _web_metric_bar(label.title(), int(status_counts.get(label, 0)), total_schedules, kind="info")
        for label in ["preview", "pending", "sending", "done", "failed", "cancelled"]
    )
    day_chart = _web_analytics_chart_html(data["day_counts"])
    error_rows = _web_analytics_error_rows(data["recent_errors"])
    body = f"""
    <div class='actions' style='justify-content:space-between;margin-bottom:12px'>
      <div class='pillbar'>
        <a class='btn {'secondary' if days != 7 else ''}' href='/admin/analytics?days=7'>7 days</a>
        <a class='btn {'secondary' if days != 30 else ''}' href='/admin/analytics?days=30'>30 days</a>
        <a class='btn {'secondary' if days != 90 else ''}' href='/admin/analytics?days=90'>90 days</a>
      </div>
      <div class='actions'><a class='btn secondary' href='/admin/schedules/calendar'>Calendar</a><a class='btn secondary' href='/admin/broadcast'>Broadcast</a></div>
    </div>
    <div class='grid'>
      {_web_status_card('Delivery rate', str(data['delivery_rate']) + '%', f"{data['sent']} sent / {data['delivery_total']} total", 'ok' if data['delivery_rate'] >= 90 else 'warn')}
      {_web_status_card('Recent delivery', str(data['recent_delivery_rate']) + '%', f"last {days} days", 'ok' if data['recent_delivery_rate'] >= 90 else 'warn')}
      {_web_status_card('Upcoming', data['upcoming'], f"{data['previews']} preview / {data['overdue']} overdue", 'warn' if data['upcoming'] else '')}
      {_web_status_card('Blocked users hit', data['blocked'], 'broadcast blocked/unreachable count', 'danger' if data['blocked'] else '')}
    </div>
    <div class='grid2'>
      <div class='card'><h2>Schedule Activity · Last {days} days</h2><p class='muted'>Counts are grouped using Phnom Penh local dates.</p>{day_chart}</div>
      <div class='card'><h2>Delivery Breakdown</h2><div class='mini-grid'>
        {_web_metric_bar('Sent', data['sent'], max(1, data['delivery_total']))}
        {_web_metric_bar('Failed', data['failed'], max(1, data['delivery_total']))}
        {_web_metric_bar('Blocked', data['blocked'], max(1, data['delivery_total']))}
      </div><div class='mini-grid' style='margin-top:12px'>
        {_web_metric_bar('Text schedules', data['text'], total_schedules)}
        {_web_metric_bar('Photo schedules', data['photo'], total_schedules)}
        {_web_metric_bar('Overdue pending', data['overdue'], total_schedules)}
      </div></div>
    </div>
    <div class='card'><h2>Status Mix</h2><div class='mini-grid'>{status_rows}</div></div>
    <div class='card'><div class='actions' style='justify-content:space-between'><h2>Recent Failed / Warning Schedules</h2><a class='btn secondary' href='/admin/schedules?status=failed'>Open failed</a></div><div class='table-wrap'><table class='table'><thead><tr><th>ID</th><th>Time</th><th>Error</th><th>Content</th><th>Action</th></tr></thead><tbody>{error_rows}</tbody></table></div></div>
    """
    return _web_render("Analytics V2", body, active="analytics")


@app_flask.route("/admin/schedules/calendar")
@web_admin_required
def web_admin_schedule_calendar():
    year, month = _web_calendar_month_params(request.args.get("month"))
    prev_y, prev_m = _web_calendar_month_shift(year, month, -1)
    next_y, next_m = _web_calendar_month_shift(year, month, 1)
    start_local, end_local, start_utc, end_utc = _web_calendar_bounds(year, month)
    rows = _web_sched_fetch_range(start_utc, end_utc, limit=2500)
    calendar_html = _web_schedule_calendar_html(year, month, rows)
    month_label = start_local.strftime("%B %Y")
    body = f"""
    <div class='actions' style='justify-content:space-between;margin-bottom:12px'>
      <div class='actions'>
        <a class='btn secondary' href='/admin/schedules/calendar?month={prev_y:04d}-{prev_m:02d}'>← Previous</a>
        <a class='btn' href='/admin/schedules/calendar'>This Month</a>
        <a class='btn secondary' href='/admin/schedules/calendar?month={next_y:04d}-{next_m:02d}'>Next →</a>
      </div>
      <div class='actions'><a class='btn secondary' href='/admin/schedules'>List View</a><a class='btn secondary' href='/admin/analytics'>Analytics</a></div>
    </div>
    <div class='grid'>
      {_web_status_card('Month', month_label, f"{_web_h(_fmt_local_time_hint())}", 'ok')}
      {_web_status_card('Schedules', len(rows), 'inside this calendar month', 'ok' if rows else '')}
      {_web_status_card('Preview', sum(1 for r in rows if _sched_is_draft(r)), 'must confirm before sending', 'warn')}
      {_web_status_card('Confirmed pending', sum(1 for r in rows if _sched_is_confirmed_pending(r)), 'scheduler can send these', 'ok')}
    </div>
    <div class='card'>
      <div class='actions' style='justify-content:space-between'><div><h2>Schedule Calendar View · {month_label}</h2><p class='muted'>Month range: {_web_h(_web_dt(start_utc))} → {_web_h(_web_dt(end_utc))}</p></div><a class='btn' href='/admin/schedules'>Create Schedule</a></div>
      <div style='display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:10px;min-width:880px'>{calendar_html}</div>
    </div>
    <div class='card'><h2>Mobile note</h2><p class='muted'>On phone, swipe horizontally inside the calendar card. Tap any schedule card to open it in the list view.</p></div>
    """
    return _web_render("Schedule Calendar", body, active="calendar")

@app_flask.route("/admin/schedules")
@web_admin_required
def web_admin_schedules():
    status = (request.args.get("status") or "all").strip().lower()
    q = (request.args.get("q") or "").strip()
    rows = _web_sched_list(status, q)
    csrf = _web_csrf_token()
    return_input = _web_return_input()
    options = "".join(f"<option value='{s}' {'selected' if status==s else ''}>{s.title()}</option>" for s in ["all", "preview", "pending", "sending", "done", "failed", "cancelled"])
    now_plus = (_local_now() + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M")
    table_rows = _web_schedule_rows_html(rows, csrf, return_input)
    parse_options = _broadcast_parse_mode_select_html("parse_mode", _BROADCAST_PARSE_MODE_AUTO)
    body = f"""
    <div class='card'>
      <form method='get' class='row3'>
        <div class='field'><label>Status</label><select name='status'>{options}</select></div>
        <div class='field'><label>Search ID/text</label><input name='q' value='{_web_h(q)}' placeholder='ID, caption, or text'></div>
        <div class='field'><label>&nbsp;</label><div class='actions'><button>Filter</button><a class='btn secondary' href='/admin/schedules'>Reset</a></div></div>
      </form>
    </div>
    <div class='card'><h2>Create Schedule</h2><p class='muted'>Use Phnom Penh local time in AM/PM style ({APP_TIMEZONE_ALIAS}, {APP_TIMEZONE_UTC_LABEL}). The bot stores the final timestamp in UTC automatically. Use Preview first for safer broadcasts.</p><form method='post' action='/admin/schedules/action'><input type='hidden' name='csrf_token' value='{csrf}'>{return_input}<input type='hidden' name='action' value='create'><div class='row'><div class='field'><label>Broadcast Phnom Penh time</label><input type='datetime-local' name='broadcast_at' value='{now_plus}' required><div class='help'>Example: 2026-12-25 09:00 AM {APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})</div></div><div class='field'><label>Mode</label><select name='confirmed'><option value='0'>Preview first</option><option value='1'>Confirm immediately</option></select></div></div><div class='field'><label>Format mode</label><select name='parse_mode'>{parse_options}</select><div class='help'>Supports Telegram HTML, MarkdownV2, Markdown, and plain text. First-line directives also work: <code>::html</code>, <code>::mdv2</code>, <code>::md</code>, <code>::plain</code>.</div></div><div class='field'><label>Text message <span><span data-count-target='#schedule-text'>0</span>/{TELE_MSG_LIMIT}</span></label><textarea id='schedule-text' name='text' maxlength='{TELE_MSG_LIMIT}' placeholder='Required for text-only schedule; used as photo caption when caption is empty'></textarea></div><div class='row'><div class='field'><label>Telegram photo_file_id optional</label><input name='photo_file_id'></div><div class='field'><label>Photo caption optional</label><input name='caption' maxlength='1024'></div></div><button>Create Schedule</button></form></div>
    <div class='card'><div class='actions' style='justify-content:space-between'><h2>Schedules</h2><span class='muted'>{len(rows)} shown</span></div><div class='table-wrap'><table class='table'><thead><tr><th>ID</th><th>Status</th><th>Time</th><th>Content</th><th>Actions</th></tr></thead><tbody>{table_rows}</tbody></table></div></div>
    """
    return _web_render("Schedules", body, active="schedules")


@app_flask.route("/admin/schedules/action", methods=["POST"])
@web_admin_required
def web_admin_schedules_action():
    _web_check_csrf()
    action = str(request.form.get("action") or "").strip()
    row_id = _web_int(request.form.get("row_id"), 0)
    ok, msg = False, "Unknown action."
    try:
        if action == "create":
            dt = _parse_dt(request.form.get("broadcast_at") or "")
            if not dt:
                ok, msg = False, "Invalid time. Use Phnom Penh local time: YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM AM/PM."
            else:
                ok, msg = _web_sched_create(
                    _web_current_admin_id(),
                    dt,
                    request.form.get("text") or "",
                    request.form.get("photo_file_id") or "",
                    request.form.get("caption") or "",
                    str(request.form.get("confirmed") or "0") == "1",
                    request.form.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO,
                )
        elif not row_id:
            ok, msg = False, "Missing schedule ID."
        elif action == "confirm":
            ok, msg = _web_sched_confirm_any(row_id, _web_current_admin_id())
        elif action == "cancel":
            ok, msg = _web_sched_cancel_any(row_id, _web_current_admin_id())
        elif action == "duplicate":
            dt = _parse_dt(request.form.get("broadcast_at") or "")
            if not dt:
                ok, msg = False, "Invalid duplicate time. Use Phnom Penh local time."
            else:
                ok, msg = _web_sched_duplicate(row_id, _web_current_admin_id(), dt)
        elif action == "edit_time":
            dt = _parse_dt(request.form.get("broadcast_at") or "")
            if not dt:
                ok, msg = False, "Invalid time. Use Phnom Penh local time: YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM AM/PM."
            else:
                ok, reason, _row = db_sched_update_time(row_id, _web_current_admin_id(), dt)
                msg = "time updated" if ok else f"time not updated: {reason}"
        elif action == "edit_text":
            edit_text = _broadcast_apply_format_directive(
                request.form.get("text") or "",
                request.form.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO,
            )
            ok, reason, _row = db_sched_update_text(row_id, _web_current_admin_id(), edit_text)
            msg = "text updated" if ok else f"text not updated: {reason}"
        elif action == "edit_photo":
            edit_caption = _broadcast_apply_format_directive(
                request.form.get("caption") or "",
                request.form.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO,
            )
            ok, reason, _row = db_sched_update_photo(row_id, _web_current_admin_id(), request.form.get("photo_file_id") or "", edit_caption)
            msg = "photo updated" if ok else f"photo not updated: {reason}"
    except Exception as exc:
        ok, msg = False, str(exc)
    flask_flash(("OK: " if ok else "ERROR: ") + msg, "success" if ok else "error")
    return redirect(_web_safe_return("web_admin_schedules"))


def _web_broadcast_job_set(job_id: str, **updates) -> None:
    with _WEB_BROADCAST_JOBS_LOCK:
        row = _WEB_BROADCAST_JOBS.get(job_id, {})
        row.update(updates)
        _WEB_BROADCAST_JOBS[job_id] = row
        _WEB_BROADCAST_JOBS.move_to_end(job_id)
        while len(_WEB_BROADCAST_JOBS) > _WEB_BROADCAST_JOBS_MAX:
            _WEB_BROADCAST_JOBS.popitem(last=False)


def _web_broadcast_worker(
    job_id: str,
    admin_id: int,
    text: str,
    parse_mode: str | None = "auto",
) -> None:
    """Run web dashboard broadcast in bounded concurrent batches with pause/cancel."""
    parse_mode = _broadcast_normalize_parse_mode(parse_mode)
    try:
        raw_users = get_all_user_ids()
    except Exception as exc:
        logger.error("web broadcast could not load users: %s", exc, exc_info=True)
        _web_broadcast_job_set(job_id, status="failed", error=str(exc)[:500], finished_at=_sched_iso())
        return

    users: list[int] = []
    seen: set[int] = set()
    for raw_uid in raw_users:
        try:
            uid = int(raw_uid)
        except Exception:
            logger.warning("web broadcast skipped invalid user id: %r", raw_uid)
            continue
        if uid not in seen:
            users.append(uid)
            seen.add(uid)

    total = len(users)
    sent = failed = blocked = skipped = 0
    _web_broadcast_job_set(
        job_id,
        status="running",
        control="run",
        total=total,
        sent=0,
        failed=0,
        blocked=0,
        skipped=0,
        parse_mode=parse_mode,
        started_at=_sched_iso(),
    )

    if not users:
        _web_broadcast_job_set(job_id, status="done", finished_at=_sched_iso())
        return

    blocked_ids = _web_blocked_ids_for_users(users)

    def _control_state() -> str:
        return str(_web_broadcast_job_get(job_id).get("control") or "run").lower()

    def _wait_if_paused() -> bool:
        while True:
            control = _control_state()
            if control == "cancel":
                return False
            if control != "pause":
                return True
            _web_broadcast_job_set(job_id, status="paused")
            time.sleep(0.5)

    def _send_one(uid: int) -> str:
        try:
            if _control_state() == "cancel":
                return "skipped"
            if uid in blocked_ids:
                return "blocked"
            ok, send_msg = _web_send_telegram_message(
                uid,
                text,
                admin_id=admin_id,
                client=tg_client,
                parse_mode=parse_mode,
            )
            if ok:
                return "sent"
            low = str(send_msg or "").lower()
            if "403" in low or "forbidden" in low or "bot was blocked" in low:
                db_user_set_blocked(uid, admin_id, True, "Telegram blocked during web broadcast")
                return "blocked"
            logger.warning("web broadcast failed uid=%s: %s", uid, str(send_msg)[:240])
            return "failed"
        except Exception as exc:
            logger.warning("web broadcast exception uid=%s: %s", uid, exc)
            return "failed"

    workers = max(1, min(WEB_BROADCAST_WORKERS, _run_state_max_concurrent_broadcast(), max(1, len(users))))
    batch_size = max(1, max(_run_state_broadcast_batch_size(), workers))
    delay_s = _run_state_broadcast_delay()
    limits = httpx.Limits(
        max_keepalive_connections=max(_run_state_http_keepalive_connections(), workers),
        max_connections=max(_run_state_http_max_connections(), workers * 2),
    )
    try:
        with httpx.Client(timeout=20, limits=limits) as tg_client:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="web_broadcast") as pool:
                for start in range(0, total, batch_size):
                    if not _wait_if_paused():
                        skipped += max(0, total - start)
                        _web_broadcast_job_set(job_id, status="cancelled", skipped=skipped, sent=sent, failed=failed, blocked=blocked, finished_at=_sched_iso())
                        return
                    _web_broadcast_job_set(job_id, status="running")
                    batch = users[start:start + batch_size]
                    futures = [pool.submit(_send_one, uid) for uid in batch]
                    last_progress_update = 0.0
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                        except Exception as exc:
                            logger.warning("web broadcast future failed job=%s: %s", job_id, exc)
                            result = "failed"

                        if result == "sent":
                            sent += 1
                        elif result == "blocked":
                            blocked += 1
                        elif result == "skipped":
                            skipped += 1
                        else:
                            failed += 1

                        now = time.monotonic()
                        if now - last_progress_update >= 1.0:
                            _web_broadcast_job_set(job_id, sent=sent, failed=failed, blocked=blocked, skipped=skipped)
                            last_progress_update = now

                    _web_broadcast_job_set(job_id, sent=sent, failed=failed, blocked=blocked, skipped=skipped)
                    if _control_state() == "cancel":
                        skipped += max(0, total - (start + len(batch)))
                        _web_broadcast_job_set(job_id, status="cancelled", skipped=skipped, sent=sent, failed=failed, blocked=blocked, finished_at=_sched_iso())
                        return
                    if start + len(batch) < total and delay_s:
                        time.sleep(delay_s)

        _web_broadcast_job_set(job_id, status="done", control="done", sent=sent, failed=failed, blocked=blocked, skipped=skipped, finished_at=_sched_iso())
    except Exception as exc:
        logger.error("web broadcast worker failed job=%s: %s", job_id, exc, exc_info=True)
        _web_broadcast_job_set(job_id, status="failed", sent=sent, failed=failed, blocked=blocked, skipped=skipped, error=str(exc)[:500], finished_at=_sched_iso())


@app_flask.route("/admin/broadcast", methods=["GET", "POST"])
@web_admin_required
def web_admin_broadcast():
    if request.method == "POST":
        _web_check_csrf()
        text = (request.form.get("text") or "").strip()
        parse_mode = _broadcast_normalize_parse_mode(request.form.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO)
        if not text:
            flask_flash("Broadcast text is empty.", "error")
        elif len(text) > TELE_MSG_LIMIT:
            flask_flash(f"Broadcast too long. Max {TELE_MSG_LIMIT} characters.", "error")
        else:
            active_jobs = _web_broadcast_active_count()
            if active_jobs >= WEB_BROADCAST_MAX_ACTIVE_JOBS:
                flask_flash(
                    f"Too many active broadcast jobs ({active_jobs}/{WEB_BROADCAST_MAX_ACTIVE_JOBS}). Pause/cancel or wait for one to finish.",
                    "error",
                )
            else:
                job_id = secrets.token_hex(6)
                _web_broadcast_job_set(
                    job_id,
                    status="queued",
                    control="run",
                    total=0,
                    sent=0,
                    failed=0,
                    blocked=0,
                    skipped=0,
                    parse_mode=parse_mode,
                    created_at=_sched_iso(),
                )
                _submit_web_broadcast_job(job_id, _web_current_admin_id(), text, parse_mode)
                flask_flash(f"Broadcast job {job_id} started.", "success")
                return redirect(url_for("web_admin_broadcast"))
    csrf = _web_csrf_token()
    job_rows = _web_broadcast_job_rows_html(csrf, include_actions=True)
    parse_options = _broadcast_parse_mode_select_html("parse_mode", _BROADCAST_PARSE_MODE_AUTO)
    body = f"""
    <div class='card danger-zone'><h2>Immediate text broadcast</h2><p class='muted'>Safer broadcast system: bounded workers, blocked-user skip, live progress, pause, resume, cancel, Telegram 403 auto-block, and Telegram formatting fallback.</p><form method='post' data-confirm='Start this broadcast now? This sends to all users.'><input type='hidden' name='csrf_token' value='{csrf}'><div class='field'><label>Format mode</label><select name='parse_mode'>{parse_options}</select><div class='help'>Supports Telegram HTML, MarkdownV2, Markdown, and plain text. You can also prefix the first line with <code>::html</code>, <code>::mdv2</code>, <code>::md</code>, or <code>::plain</code>.</div></div><div class='field'><label>Message <span><span data-count-target='#broadcast-text'>0</span>/{TELE_MSG_LIMIT}</span></label><textarea id='broadcast-text' name='text' maxlength='{TELE_MSG_LIMIT}' required></textarea><div class='help'>For scheduled sending, use the Schedules page instead. If formatting is malformed, the bot retries as plain text instead of failing the whole job.</div></div><button class='danger'>Start Broadcast</button></form></div>
    <div class='card'><div class='actions' style='justify-content:space-between'><h2>Recent web broadcast jobs</h2><span class='live-note'><span class='v3-live-dot'></span>Auto-refresh every 5 seconds</span></div><div class='table-wrap'><table class='table'><thead><tr><th>Job</th><th>Status</th><th>Progress</th><th>Blocked</th><th>Failed</th><th>Started</th><th>Action</th></tr></thead><tbody data-broadcast-jobs>{job_rows}</tbody></table></div></div>
    """
    return _web_render("Broadcast", body, active="broadcast")




@app_flask.route("/admin/broadcast/action", methods=["POST"])
@web_admin_required
def web_admin_broadcast_action():
    _web_check_csrf()
    job_id = str(request.form.get("job_id") or "").strip()
    action = str(request.form.get("action") or "").strip().lower()
    ok, msg = _web_broadcast_job_control(job_id, action)
    flask_flash(("OK: " if ok else "ERROR: ") + msg, "success" if ok else "error")
    return redirect(request.referrer or url_for("web_admin_broadcast"))

@app_flask.route("/admin/broadcast/jobs.json")
@web_admin_required
def web_admin_broadcast_jobs_json():
    with _WEB_BROADCAST_JOBS_LOCK:
        jobs = [{"id": jid, **dict(row)} for jid, row in reversed(list(_WEB_BROADCAST_JOBS.items()))]
    resp = jsonify({"ok": True, "jobs": jobs, "rows_html": _web_broadcast_job_rows_html(_web_csrf_token(), include_actions=True)})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app_flask.route("/admin/settings", methods=["GET", "POST"])
@web_admin_required
def web_admin_settings():
    if request.method == "POST":
        _web_check_csrf()
        key = str(request.form.get("key") or "").strip()
        enabled = str(request.form.get("enabled") or "0") == "1"
        ok, msg = db_bot_setting_set(key, enabled, _web_current_admin_id())
        flask_flash(("OK: " if ok else "ERROR: ") + msg, "success" if ok else "error")
        return redirect(url_for("web_admin_settings"))
    settings, status = db_bot_settings_fetch_all()
    csrf = _web_csrf_token()
    rows = []
    for key in BOT_FEATURE_SETTING_KEYS:
        enabled = _setting_bool_from(settings, key)
        rows.append(
            f"<tr><td><b>{_web_h(BOT_SETTING_LABELS.get(key,key))}</b><br><span class='muted'>{_web_h(BOT_SETTING_DESCRIPTIONS.get(key,''))}</span></td>"
            f"<td>{_web_badge('ON' if enabled else 'OFF', 'ok' if enabled else 'muted')}</td>"
            f"<td><form method='post' data-confirm='Change this bot setting?'><input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='key' value='{_web_h(key)}'><input type='hidden' name='enabled' value='{'0' if enabled else '1'}'><button class='{'danger' if enabled else 'ok'}'>{'Turn OFF' if enabled else 'Turn ON'}</button></form></td></tr>"
        )
    perf_rows = []
    for key, spec in BOT_PERFORMANCE_SETTING_SPECS.items():
        value = _setting_raw_from(settings, key, BOT_SETTING_DEFAULTS.get(key, ''))
        perf_rows.append(
            f"<tr><td><b>{_web_h(BOT_SETTING_LABELS.get(key,key))}</b><br><span class='muted'>{_web_h(BOT_SETTING_DESCRIPTIONS.get(key,''))}</span></td>"
            f"<td><code>{_web_h(value)}</code></td><td><a class='btn secondary' href='/admin/runtime'>Advanced Runtime</a></td></tr>"
        )
    body = f"""<div class='card'><h2>Bot Settings</h2><p>DB status: {_web_badge('OK' if status.get('db_ok') else 'MEMORY', 'ok' if status.get('db_ok') else 'warn')} <span class='muted'>{_web_h(status.get('error') or '')}</span></p><div class='table-wrap'><table class='table'><thead><tr><th>Feature</th><th>Status</th><th>Action</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div><div class='card'><h2>Performance Settings</h2><p class='muted'>Use Telegram /admin → Settings → Performance Settings for quick mobile edits, or open Runtime for advanced web tuning.</p><div class='table-wrap'><table class='table'><thead><tr><th>Setting</th><th>Value</th><th>Action</th></tr></thead><tbody>{''.join(perf_rows)}</tbody></table></div></div>"""
    return _web_render("Settings", body, active="settings")


@app_flask.route("/admin/api-keys", methods=["GET", "POST"])
@web_admin_required
def web_admin_api_keys():
    if request.method == "POST":
        _web_check_csrf()
        action = str(request.form.get("action") or "").strip()
        try:
            if action == "create":
                raw, row, source = db_ai_api_key_create(_web_current_admin_id(), request.form.get("note") or "web-dashboard")
                flask_flash(f"Created key ({source}). Copy now: {raw}", "success")
            elif action == "revoke":
                ok, msg = db_ai_api_key_revoke(request.form.get("identifier") or "")
                flask_flash(("Revoked " if ok else "Failed: ") + msg, "success" if ok else "error")
        except Exception as exc:
            flask_flash(str(exc), "error")
        return redirect(url_for("web_admin_api_keys"))
    try:
        rows = db_ai_api_key_list(50)
    except Exception as exc:
        rows = []
        flask_flash(str(exc), "error")
    csrf = _web_csrf_token()
    trs = "".join(
        f"<tr><td><code>{_web_h(r.get('id'))}</code></td><td><code>{_web_h(r.get('key_prefix'))}</code></td><td>{_web_h(r.get('note') or '')}</td><td>{_web_badge('active' if r.get('active') else 'revoked', 'ok' if r.get('active') else 'muted')}</td><td>{_web_h(_web_dt(r.get('created_at')))}</td><td><form class='inline-form' method='post' data-confirm='Revoke this API key? Existing clients using it will fail.'><input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='action' value='revoke'><input type='hidden' name='identifier' value='{_web_h(r.get('id'))}'><button class='danger'>Revoke</button></form></td></tr>"
        for r in rows
    )
    body = f"""
    <div class='card'><h2>Create API Key</h2><p class='muted'>The raw key is shown once in the success message. Store it in your frontend/backend env immediately.</p><form method='post'><input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='action' value='create'><div class='field'><label>Note</label><input name='note' placeholder='frontend, mobile app, test'></div><button>Create key</button></form></div>
    <div class='card'><h2>API Keys</h2><div class='table-wrap'><table class='table'><thead><tr><th>ID</th><th>Prefix</th><th>Note</th><th>Status</th><th>Created</th><th>Action</th></tr></thead><tbody>{trs or '<tr><td colspan=6><div class="empty">No keys.</div></td></tr>'}</tbody></table></div></div>
    """
    return _web_render("API Keys", body, active="api")


@app_flask.route("/admin/locks", methods=["GET", "POST"])
@web_admin_required
def web_admin_locks():
    if request.method == "POST":
        _web_check_csrf()
        if not supabase:
            flask_flash("Supabase is not configured.", "error")
        else:
            res = db_call_sync("web_lock_release", lambda: supabase.table("bot_locks").delete().eq("lock_key", _SCHED_LOCK_KEY).execute(), default=None, attempts=2, critical=False)
            flask_flash("Scheduler lock release attempted.", "success" if res is not None else "warning")
        return redirect(url_for("web_admin_locks"))
    lock = db_lock_read(_SCHED_LOCK_KEY) if supabase else None
    body = f"""<div class='card'><h2>Distributed Scheduler Lock</h2><div class='table-wrap'><table class='table'><tr><td>Enabled</td><td>{_web_badge('ON' if _SCHED_LOCK_ENABLED else 'OFF', 'ok' if _SCHED_LOCK_ENABLED else 'muted')}</td></tr><tr><td>Required</td><td>{_web_badge('YES' if _SCHED_LOCK_REQUIRED else 'NO', 'warn' if _SCHED_LOCK_REQUIRED else 'muted')}</td></tr><tr><td>Key</td><td><code>{_web_h(_SCHED_LOCK_KEY)}</code></td></tr><tr><td>Current row</td><td><pre>{_web_h(_json.dumps(lock, ensure_ascii=False, indent=2) if lock else 'No lock row found.')}</pre></td></tr></table></div><form method='post' data-confirm='Force release scheduler lock? Only do this if the old owner is dead.'><input type='hidden' name='csrf_token' value='{_web_csrf_token()}'><button class='danger'>Force release scheduler lock</button></form><p class='muted'>Only force release when the old owner is dead or Render restarted.</p></div>"""
    return _web_render("Locks", body, active="locks")


@app_flask.route("/admin/sql")
@web_admin_required
def web_admin_sql():
    locks_sql = """create table if not exists public.bot_locks (
  lock_key text primary key,
  owner text not null,
  locked_until timestamptz not null,
  updated_at timestamptz not null default now()
);

create index if not exists bot_locks_locked_until_idx on public.bot_locks (locked_until);

alter table public.bot_locks enable row level security;

drop policy if exists "service_role_bot_locks_all" on public.bot_locks;
create policy "service_role_bot_locks_all"
on public.bot_locks
for all
to service_role
using (true)
with check (true);"""
    schedule_indexes = """create index if not exists scheduled_broadcasts_due_idx
on public.scheduled_broadcasts (broadcast_at)
where status = 'pending' and error_msg is null;

create index if not exists scheduled_broadcasts_admin_pending_idx
on public.scheduled_broadcasts (admin_id, broadcast_at)
where status = 'pending';

-- Recommended for Admin Dashboard V5 Analytics + Calendar month range queries
create index if not exists scheduled_broadcasts_calendar_idx
on public.scheduled_broadcasts (broadcast_at, status);"""
    body = f"<div class='card'><h2>Required / Recommended SQL</h2><p>Run in Supabase SQL Editor.</p></div><div class='card'><h3>bot_locks</h3><pre>{_web_h(locks_sql)}</pre></div><div class='card'><h3>Admin V2 tables</h3><pre>{_web_h(ADMIN_V2_TABLES_SQL)}</pre></div><div class='card'><h3>Schedule indexes</h3><pre>{_web_h(schedule_indexes)}</pre></div>"
    return _web_render("SQL", body, active="sql")


# ── Admin Control Center V5 ─────────────────────────────────────────────────
def _web_counts_invalidate() -> None:
    with _WEB_COUNTS_CACHE_LOCK:
        _WEB_COUNTS_CACHE["ts"] = 0.0
        _WEB_COUNTS_CACHE["data"] = {}
    with _WEB_LIVE_SCHEDULES_LOCK:
        _WEB_LIVE_SCHEDULES_CACHE["ts"] = 0.0
        _WEB_LIVE_SCHEDULES_CACHE["rows"] = []


def _web_admin_audit(action: str, detail: str = "") -> None:
    item = {
        "at": _fmt_local_dt(),
        "admin_id": _web_current_admin_id(),
        "action": str(action or "")[:80],
        "detail": _web_short(detail, 300),
        "path": getattr(request, "path", ""),
    }
    _WEB_ADMIN_AUDIT.appendleft(item)
    logger.info("admin_audit admin_id=%s action=%s detail=%s", item["admin_id"], item["action"], item["detail"])


def _semaphore_snapshot(sem: asyncio.Semaphore | None, configured: int) -> dict:
    if sem is None:
        return {"configured": int(configured), "available": None, "in_use": None}
    available = int(getattr(sem, "_value", 0))
    return {
        "configured": int(configured),
        "available": max(0, available),
        "in_use": max(0, int(configured) - max(0, available)),
    }


def _breaker_snapshot(breaker: Any) -> dict:
    if breaker is None:
        return {"configured": False}
    return {
        "configured": True,
        "name": getattr(breaker, "name", ""),
        "failures": int(getattr(breaker, "failures", 0) or 0),
        "open": bool(breaker.is_open()) if hasattr(breaker, "is_open") else False,
        "open_remaining_s": max(0, int(float(getattr(breaker, "open_until", 0.0) or 0.0) - time.monotonic())),
    }


def _runtime_performance_snapshot(light: bool = False) -> dict:
    with _WEB_ACTIVE_REQUESTS_LOCK:
        active_requests = int(_WEB_ACTIVE_REQUESTS)

    active_jobs = _web_broadcast_active_count() if "_web_broadcast_active_count" in globals() else 0
    pending_sched = len(_scheduler_active_ids) if "_scheduler_active_ids" in globals() else 0

    snap = {
        "uptime": _format_uptime(),
        "local_time": _fmt_local_dt(),
        "web": {
            "active_requests": active_requests,
            "slow_request_threshold_ms": WEB_SLOW_REQUEST_MS,
            "recent_slow_requests": [] if light else list(_WEB_SLOW_REQUESTS)[:20],
            "admin_error_events": (globals().get("_admin_error_center_total_count")() if callable(globals().get("_admin_error_center_total_count")) else len(list(globals().get("_ADMIN_ERROR_CENTER") or []))),
            "admin_error_groups": len(globals().get("_admin_error_center_grouped", lambda *a, **k: [])(limit=200, include_muted=True)) if callable(globals().get("_admin_error_center_grouped")) else 0,
            "counts_cache_ttl_s": _run_state_web_counts_cache_ttl(),
            "status_poll_s": _run_state_int("WEB_STATUS_POLL_SECONDS", WEB_STATUS_POLL_SECONDS, minimum=10, maximum=300),
            "live_poll_s": _run_state_int("WEB_LIVE_POLL_SECONDS", WEB_LIVE_POLL_SECONDS, minimum=15, maximum=300),
        },
        "telegram": {
            "concurrent_updates": TELEGRAM_CONCURRENT_UPDATES,
            "connection_pool_size": TELEGRAM_CONNECTION_POOL_SIZE,
            "http_max_connections": _run_state_http_max_connections(),
            "http_keepalive_connections": _run_state_http_keepalive_connections(),
            "user_rate_limit": _run_state_user_rate_limit(),
            "user_rate_window_s": _run_state_user_rate_window(),
            "api_rate_limit": _run_state_api_rate_limit(),
            "api_rate_window_s": _run_state_api_rate_window(),
            "allowed_updates": ["message", "callback_query"],
        },
        "semaphores": {
            "tts": _semaphore_snapshot(_TTS_CHUNK_SEMAPHORE, MAX_CONCURRENT_TTS_USERS),
            "ai": _semaphore_snapshot(_AI_SEMAPHORE, MAX_CONCURRENT_AI),
            "broadcast": _semaphore_snapshot(_BROADCAST_SEMAPHORE, MAX_CONCURRENT_BROADCAST),
        },
        "broadcast": {
            "active_jobs": active_jobs,
            "max_active_jobs": WEB_BROADCAST_MAX_ACTIVE_JOBS,
            "workers": min(WEB_BROADCAST_WORKERS, _run_state_max_concurrent_broadcast()),
            "batch_size": _run_state_broadcast_batch_size(),
            "delay_s": _run_state_broadcast_delay(),
            "queue_size": _WEB_BROADCAST_QUEUE.qsize() if _WEB_BROADCAST_QUEUE is not None else None,
            "queue_maxsize": _WEB_BROADCAST_QUEUE.maxsize if _WEB_BROADCAST_QUEUE is not None else _run_state_web_broadcast_queue_maxsize(),
            "recent_jobs": [] if light else list(_WEB_BROADCAST_JOBS.values())[:20],
        },
        "scheduler": {
            "active_ids": pending_sched,
            "lock_enabled": bool(_SCHED_LOCK_ENABLED),
            "lock_required": bool(_SCHED_LOCK_REQUIRED),
        },
        "stores": {
            "supabase": bool(supabase),
            "redis": bool(redis_client),
            "supabase_breaker": _breaker_snapshot(globals().get("supabase_breaker")),
            "redis_breaker": _breaker_snapshot(globals().get("redis_breaker")),
        },
        "memory": {
            "rate_limit_keys": len(_RATE_LIMIT_MEMORY),
            "rate_limit_notice_keys": len(_RATE_LIMIT_NOTICE_MEMORY),
            "user_locks": len(_user_locks),
            "prefs_cache": len(_prefs_cache),
            "history_cache": len(_hist_cache),
            "text_cache_memory": len(_text_cache_memory),
            "tts_audio_cache_items": len(_TTS_AUDIO_CACHE) if "_TTS_AUDIO_CACHE" in globals() else 0,
            "tts_audio_cache_mb": round((_TTS_AUDIO_CACHE_BYTES if "_TTS_AUDIO_CACHE_BYTES" in globals() else 0) / 1024 / 1024, 2),
            "prefs_load_locks": len(_prefs_load_locks) if "_prefs_load_locks" in globals() else 0,
            "api_key_cache": len(_api_key_validation_cache),
        },
        "executors": {
            "db": _executor_status("db", globals().get("_DB_EXECUTOR")),
            "ai": _executor_status("ai", globals().get("_AI_EXECUTOR")),
            "web_broadcast": _executor_status("web_broadcast", globals().get("_WEB_BROADCAST_EXECUTOR")),
        },
        "cooldowns": {
            "hf_tts_remaining_s": _hf_tts_disabled_remaining_s() if "_hf_tts_disabled_remaining_s" in globals() else 0,
            "hf_ocr_disabled": bool(_hf_ocr_is_temporarily_disabled()) if "_hf_ocr_is_temporarily_disabled" in globals() else False,
            "hf_ocr_provider_disabled": bool(_ocr_provider_is_temporarily_disabled("hf")) if "_ocr_provider_is_temporarily_disabled" in globals() else False,
            "gemini_ocr_provider_disabled": bool(_ocr_provider_is_temporarily_disabled("gemini")) if "_ocr_provider_is_temporarily_disabled" in globals() else False,
        },
        "metrics": dict(_RUNTIME_METRICS),
    }

    if not light:
        with suppress(Exception):
            import resource
            rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            snap["process"] = {"max_rss_kb": rss_kb}
        with suppress(Exception):
            snap.setdefault("process", {})["loadavg"] = list(os.getloadavg())
    return snap


def _reset_hf_tts_cooldown() -> None:
    if "_HF_TTS_STATE_LOCK" in globals():
        with globals()["_HF_TTS_STATE_LOCK"]:
            globals()["_HF_TTS_FAILURES"] = 0
            globals()["_HF_TTS_DISABLED_UNTIL"] = 0.0


def _reset_ocr_cooldowns() -> None:
    if "_hf_ocr_state_lock" in globals():
        with globals()["_hf_ocr_state_lock"]:
            globals()["_hf_ocr_failures"] = 0
            globals()["_hf_ocr_disabled_until"] = 0.0
    if "_ocr_provider_state_lock" in globals():
        with globals()["_ocr_provider_state_lock"]:
            for key in list(globals().get("_ocr_provider_failures", {}).keys()):
                globals()["_ocr_provider_failures"][key] = 0
            for key in list(globals().get("_ocr_provider_disabled_until", {}).keys()):
                globals()["_ocr_provider_disabled_until"][key] = 0.0


@app_flask.route("/admin/performance.json")
@web_admin_required
def web_admin_performance_json():
    resp = jsonify({"ok": True, "performance": _runtime_performance_snapshot(light=False)})
    resp.headers["Cache-Control"] = "no-store"
    return resp




def _run_admin_coro_sync(coro: Any, timeout: float = 20.0) -> Any:
    """Run an async runtime admin mutation from a sync web-admin route.

    FastAPI compatibility routes run sync handlers in a worker thread. This
    helper safely schedules mutations onto the bot/background loop when present,
    cancels timed-out work, and refuses to block a currently-running event loop.
    """
    loop = _WEB_BROADCAST_QUEUE_LOOP
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=timeout)
        except Exception:
            if not future.done():
                future.cancel()
            raise

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is not None and running_loop.is_running():
        _close_unawaited(coro)
        raise RuntimeError("Cannot run admin coroutine synchronously from an active event loop.")
    return asyncio.run(coro)



def _admin_runtime_update_many_sync(values: dict[str, Any], *, timeout: float = 20.0) -> list[str]:
    """Persist multiple runtime knobs from sync admin routes."""
    changed: list[str] = []
    for key, value in values.items():
        if key not in globals().get("_RUN_STATE_REDIS_KEYS", ()):  # safe guard for future edits
            continue
        current = RUN_STATE.get(key, globals().get(key))
        coerced = _coerce_run_state_value(key, value)
        if str(current) == str(coerced):
            continue
        _run_admin_coro_sync(_update_run_state(key, coerced), timeout=timeout)
        changed.append(key)
    return changed


def _optimization_preset_values(name: str) -> dict[str, Any]:
    """Runtime tuning presets for common Render/Railway sizes."""
    name = (name or "balanced").strip().lower()
    if name in {"small", "render_small", "safe"}:
        return {
            "HTTP_MAX_CONNECTIONS": 50,
            "HTTP_MAX_KEEPALIVE_CONNECTIONS": 12,
            "USER_RATE_LIMIT_PER_SECOND": 2,
            "USER_RATE_LIMIT_WINDOW_S": 1.0,
            "API_RATE_LIMIT_PER_SECOND": 12,
            "API_RATE_LIMIT_WINDOW_S": 1.0,
            "RATE_LIMIT_REDIS_TTL_S": 30,
            "CACHE_ASIDE_DEFAULT_TTL_S": 3600,
            "BROADCAST_BATCH_SIZE": 2,
            "BROADCAST_INTER_BATCH_DELAY": 0.35,
            "MAX_CONCURRENT_BROADCAST": 2,
            "WEB_COUNTS_CACHE_TTL_S": 60.0,
            "WEB_STATUS_POLL_SECONDS": 45,
            "WEB_LIVE_POLL_SECONDS": 45,
        }
    if name in {"high", "high_throughput", "fast"}:
        return {
            "HTTP_MAX_CONNECTIONS": 180,
            "HTTP_MAX_KEEPALIVE_CONNECTIONS": 50,
            "USER_RATE_LIMIT_PER_SECOND": 5,
            "USER_RATE_LIMIT_WINDOW_S": 1.0,
            "API_RATE_LIMIT_PER_SECOND": 60,
            "API_RATE_LIMIT_WINDOW_S": 1.0,
            "RATE_LIMIT_REDIS_TTL_S": 45,
            "CACHE_ASIDE_DEFAULT_TTL_S": 1800,
            "BROADCAST_BATCH_SIZE": 8,
            "BROADCAST_INTER_BATCH_DELAY": 0.08,
            "MAX_CONCURRENT_BROADCAST": 6,
            "WEB_COUNTS_CACHE_TTL_S": 20.0,
            "WEB_STATUS_POLL_SECONDS": 20,
            "WEB_LIVE_POLL_SECONDS": 20,
        }
    # balanced default
    return {
        "HTTP_MAX_CONNECTIONS": 100,
        "HTTP_MAX_KEEPALIVE_CONNECTIONS": 24,
        "USER_RATE_LIMIT_PER_SECOND": 3,
        "USER_RATE_LIMIT_WINDOW_S": 1.0,
        "API_RATE_LIMIT_PER_SECOND": 25,
        "API_RATE_LIMIT_WINDOW_S": 1.0,
        "RATE_LIMIT_REDIS_TTL_S": 30,
        "CACHE_ASIDE_DEFAULT_TTL_S": 3600,
        "BROADCAST_BATCH_SIZE": 4,
        "BROADCAST_INTER_BATCH_DELAY": 0.18,
        "MAX_CONCURRENT_BROADCAST": 3,
        "WEB_COUNTS_CACHE_TTL_S": 30.0,
        "WEB_STATUS_POLL_SECONDS": 30,
        "WEB_LIVE_POLL_SECONDS": 30,
    }


def _optimization_score(snap: dict[str, Any] | None = None) -> tuple[int, list[str], list[str]]:
    """Return a simple health score plus warnings/recommendations for /admin."""
    snap = snap or _runtime_performance_snapshot(light=True)
    score = 100
    warnings: list[str] = []
    tips: list[str] = []

    web = snap.get("web", {}) if isinstance(snap, dict) else {}
    memory = snap.get("memory", {}) if isinstance(snap, dict) else {}
    broadcast = snap.get("broadcast", {}) if isinstance(snap, dict) else {}
    stores = snap.get("stores", {}) if isinstance(snap, dict) else {}
    semaphores = snap.get("semaphores", {}) if isinstance(snap, dict) else {}

    if int(web.get("active_requests") or 0) >= 10:
        score -= 15
        warnings.append("High active web requests")
        tips.append("Increase poll interval or use the Small Render preset during traffic spikes.")
    if int(memory.get("rate_limit_keys") or 0) > 50_000:
        score -= 10
        warnings.append("Large in-memory rate-limit cache")
        tips.append("Enable Redis or use Trim Runtime Memory from Optimize.")
    if int(memory.get("rate_limit_notice_keys") or 0) > 20_000:
        score -= 8
        warnings.append("Large rate-limit notice cache")
        tips.append("Trim notices after a flood event to reduce memory pressure.")
    if not bool(stores.get("redis")):
        score -= 8
        warnings.append("Redis is OFF")
        tips.append("Redis improves cache-aside, runtime config, and rate limiting stability.")
    if not bool(stores.get("supabase")):
        score -= 12
        warnings.append("Supabase is OFF")
        tips.append("Supabase is required for persistent admin data and user management.")
    qsize = broadcast.get("queue_size")
    qmax = broadcast.get("queue_maxsize")
    try:
        if qsize is not None and qmax and int(qsize) > int(qmax) * 0.7:
            score -= 10
            warnings.append("Broadcast queue is near capacity")
            tips.append("Pause new broadcasts or increase queue size then restart.")
    except Exception:
        pass
    for name, sem in (semaphores or {}).items():
        try:
            configured = int(sem.get("configured") or 0)
            in_use = int(sem.get("in_use") or 0)
            if configured and in_use >= configured:
                score -= 8
                warnings.append(f"{name.upper()} concurrency is saturated")
                tips.append(f"Lower incoming load or increase MAX_CONCURRENT_{name.upper()} if CPU/RAM allows.")
        except Exception:
            pass
    if not warnings:
        tips.append("System looks stable. Keep Balanced preset unless you see queue growth or slow requests.")
    return max(0, min(100, score)), warnings, tips[:6]


def _optimization_score_card_html(snap: dict[str, Any] | None = None) -> str:
    score, warnings, tips = _optimization_score(snap)
    kind = "ok" if score >= 85 else ("warn" if score >= 65 else "danger")
    warning_html = "".join(f"<li>{_web_h(w)}</li>" for w in warnings) or "<li>No critical warnings.</li>"
    tip_html = "".join(f"<li>{_web_h(t)}</li>" for t in tips)
    return f"""
    <div class='card'>
      <div class='actions' style='justify-content:space-between;align-items:center'>
        <div><h2>Optimization Score</h2><p class='muted'>Live score from web load, queues, caches, stores, and concurrency.</p></div>
        {_web_badge(str(score) + '/100', kind)}
      </div>
      <div style='height:10px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;margin:10px 0 12px'>
        <div style='height:10px;width:{score}%;border-radius:999px;background:linear-gradient(90deg,#38bdf8,#22c55e)'></div>
      </div>
      <div class='grid2'>
        <div><b>Warnings</b><ul class='muted' style='margin:.5rem 0 0 1.1rem'>{warning_html}</ul></div>
        <div><b>Recommendations</b><ul class='muted' style='margin:.5rem 0 0 1.1rem'>{tip_html}</ul></div>
      </div>
    </div>
    """


def _optimization_preset_cards_html(csrf: str) -> str:
    cards = [
        ("preset_small", "Small Render", "Lowest DB/HTTP pressure. Best for free/small instances.", "secondary"),
        ("preset_balanced", "Balanced", "Safe default for normal traffic and admin usage.", "ok"),
        ("preset_high", "High Throughput", "Faster broadcast/API throughput. Use only when CPU/RAM are healthy.", "warn"),
    ]
    out = []
    for action, label, desc, kind in cards:
        out.append(
            f"<div class='card'><h3>{_web_h(label)}</h3><p class='muted'>{_web_h(desc)}</p>"
            f"<form method='post' data-confirm='Apply {_web_h(label)} runtime preset?'>"
            f"<input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='action' value='{action}'>"
            f"<button class='{_web_h(kind)}'>Apply Preset</button></form></div>"
        )
    return "<div class='grid'>" + "".join(out) + "</div>"


def _runtime_core_rows_html() -> str:
    keys = [
        "BOT_MODE", "HTTP_MAX_CONNECTIONS", "HTTP_MAX_KEEPALIVE_CONNECTIONS",
        "USER_RATE_LIMIT_PER_SECOND", "API_RATE_LIMIT_PER_SECOND",
        "BROADCAST_BATCH_SIZE", "BROADCAST_INTER_BATCH_DELAY", "MAX_CONCURRENT_BROADCAST",
        "WEB_COUNTS_CACHE_TTL_S", "WEB_STATUS_POLL_SECONDS", "WEB_LIVE_POLL_SECONDS",
    ]
    rows = []
    for key in keys:
        rows.append(
            f"<tr><td><code>{_web_h(key)}</code></td><td><b>{_web_h(_runtime_display_value(key))}</b></td>"
            f"<td class='muted'>{_web_h(_RUNTIME_CONFIG_SPECS.get(key, {}).get('help', ''))}</td></tr>"
        )
    return "".join(rows)


def _slow_request_rows_html(limit: int = 12) -> str:
    rows = []
    for item in list(_WEB_SLOW_REQUESTS)[:max(1, int(limit or 12))]:
        rows.append(
            f"<tr><td>{_web_h(item.get('ts'))}</td><td><code>{_web_h(item.get('method'))}</code></td>"
            f"<td>{_web_h(item.get('path'))}</td><td>{_web_h(item.get('status'))}</td>"
            f"<td><b>{_web_h(item.get('elapsed_ms'))} ms</b></td><td>{_web_h(item.get('active'))}</td></tr>"
        )
    return "".join(rows) or '<tr><td colspan="6"><div class="empty">No slow requests recorded.</div></td></tr>'


def _admin_trim_runtime_memory_sync() -> str:
    """Safe in-process cleanup for admin Optimize page."""
    now = time.monotonic()
    removed = 0
    stale_notice_before = now - max(USER_RATE_LIMIT_NOTICE_COOLDOWN_S * 4, 300.0)
    for key, old_ts in list(_RATE_LIMIT_NOTICE_MEMORY.items()):
        if old_ts < stale_notice_before:
            _RATE_LIMIT_NOTICE_MEMORY.pop(key, None)
            removed += 1
    # Do not clear active rate buckets blindly; keep recent buckets and remove empty/stale ones.
    wall_now = time.time()
    stale_bucket_before = wall_now - max(_run_state_user_rate_window() * 4, 120.0)
    for key, dq in list(_RATE_LIMIT_MEMORY.items()):
        if not dq or (dq and dq[-1] < stale_bucket_before):
            _RATE_LIMIT_MEMORY.pop(key, None)
            removed += 1
    return f"Removed {removed} stale in-memory limiter entries."

def _runtime_display_value(key: str) -> str:
    value = RUN_STATE.get(key, globals().get(key, ""))
    if key == "TELEGRAM_WEBHOOK_SECRET_TOKEN":
        token = str(value or "")
        if not token:
            return ""
        return token[:6] + "…" + token[-4:]
    if isinstance(value, float):
        return f"{value:g}"
    return str(value if value is not None else "")


def _runtime_config_input_html(key: str) -> str:
    spec = _RUNTIME_CONFIG_SPECS.get(key, {})
    kind = spec.get("kind", "str")
    value = _runtime_display_value(key)
    if kind == "mode":
        mode = _run_state_bot_mode()
        return (
            "<select name='cfg_" + _web_h(key) + "'>"
            f"<option value='POLLING' {'selected' if mode == 'POLLING' else ''}>POLLING</option>"
            f"<option value='WEBHOOK' {'selected' if mode == 'WEBHOOK' else ''}>WEBHOOK</option>"
            "</select>"
        )
    if kind == "secret":
        return (
            f"<input type='text' value='{_web_h(value)}' disabled>"
            "<div class='muted'>Use the Rotate Secret button. The full token is never displayed.</div>"
        )
    if kind == "int":
        return f"<input type='number' name='cfg_{_web_h(key)}' value='{_web_h(value)}' min='{_web_h(spec.get('min', ''))}' max='{_web_h(spec.get('max', ''))}' step='1'>"
    if kind == "float":
        return f"<input type='number' name='cfg_{_web_h(key)}' value='{_web_h(value)}' min='{_web_h(spec.get('min', ''))}' max='{_web_h(spec.get('max', ''))}' step='0.05'>"
    return f"<input type='text' name='cfg_{_web_h(key)}' value='{_web_h(value)}'>"


def _runtime_config_rows_html() -> str:
    rows = []
    for key, spec in _RUNTIME_CONFIG_SPECS.items():
        rows.append(
            f"<tr><td><b>{_web_h(spec.get('label', key))}</b><br>"
            f"<code>{_web_h(key)}</code><br><span class='muted'>{_web_h(spec.get('help', ''))}</span></td>"
            f"<td>{_runtime_config_input_html(key)}</td></tr>"
        )
    return "".join(rows)


@app_flask.route("/admin/runtime", methods=["GET", "POST"])
@web_admin_required
def web_admin_runtime_config():
    """Dashboard-controlled runtime configuration persisted in Redis."""
    csrf = _web_csrf_token()
    admin_id = _web_current_admin_id()
    if request.method == "POST":
        _web_check_csrf()
        action = str(request.form.get("action") or "save").strip().lower()
        ok = True
        messages: list[str] = []
        try:
            if action == "rotate_secret":
                remaining = _webhook_rotate_begin_or_remaining()
                if remaining < 0:
                    raise RuntimeError("Webhook secret rotation is already running. Please wait.")
                if remaining > 0:
                    raise RuntimeError(f"Please wait {int(remaining) + 1}s before rotating the webhook secret again.")

                success = False
                new_token = generate_new_webhook_token()
                try:
                    # Important order: ask Telegram to accept the new webhook
                    # first.  Only persist RUN_STATE after setWebhook succeeds.
                    if _run_state_bot_mode() == "WEBHOOK":
                        _run_admin_coro_sync(
                            _configure_telegram_webhook_via_http_for_secret(new_token),
                            timeout=45,
                        )
                    _run_admin_coro_sync(
                        _update_run_state("TELEGRAM_WEBHOOK_SECRET_TOKEN", new_token),
                        timeout=10,
                    )
                    success = True
                finally:
                    _webhook_rotate_finish(success)

                messages.append(f"Webhook secret rotated. New path: /tg-webhook-{new_token}")
                logger.info("Admin %s rotated Webhook Secret Token from web dashboard.", admin_id)
            else:
                target_mode = str(request.form.get("cfg_BOT_MODE") or _run_state_bot_mode()).strip().upper()
                mode_changed = target_mode != _run_state_bot_mode()
                webhook_url_changed = False
                for key, spec in _RUNTIME_CONFIG_SPECS.items():
                    if key in {"BOT_MODE", "TELEGRAM_WEBHOOK_SECRET_TOKEN"}:
                        continue
                    field = f"cfg_{key}"
                    if field not in request.form:
                        continue
                    raw = request.form.get(field)
                    current = RUN_STATE.get(key, globals().get(key))
                    new_value = _coerce_run_state_value(key, raw)
                    if str(new_value) != str(current):
                        _run_admin_coro_sync(_update_run_state(key, new_value), timeout=10)
                        messages.append(f"{key} updated")
                        if key == "TELEGRAM_WEBHOOK_URL":
                            webhook_url_changed = True
                if mode_changed:
                    _run_admin_coro_sync(_switch_telegram_runtime_mode(target_mode, admin_id), timeout=30)
                    messages.append(f"BOT_MODE switched to {target_mode}")
                elif webhook_url_changed and _run_state_bot_mode() == "WEBHOOK":
                    _run_admin_coro_sync(set_telegram_webhook(), timeout=20)
                    messages.append("Webhook re-registered with new URL")
                if not messages:
                    messages.append("No runtime changes detected.")
        except Exception as exc:
            ok = False
            messages.append(str(exc)[:500])
        _web_admin_audit("runtime_config", "; ".join(messages))
        flask_flash(("OK: " if ok else "ERROR: ") + "; ".join(messages), "success" if ok else "error")
        return redirect(url_for("web_admin_runtime_config"))

    webhook_ready = bool(_runtime_webhook_base_url() and _runtime_webhook_secret_token())
    minimal_env_text = 'PORT=8080\nRENDER=true\nRENDER_EXTERNAL_URL=https://your-service.onrender.com\nTELEGRAM_BOT_TOKEN=CHANGE_ME\nADMIN_IDS=1272791365\nADMIN_WEB_PASSWORD=CHANGE_ME\nSUPABASE_URL=https://your-project.supabase.co\nSUPABASE_SERVICE_ROLE_KEY=CHANGE_ME\nREDIS_URL=redis://default:password@host:6379\nHF_TOKEN=CHANGE_ME_OPTIONAL\nGEMINI_API_KEY=CHANGE_ME_OPTIONAL'
    body = f"""
    <div data-live-status></div>
    <div class='card'>
      <h2>Runtime Config Center</h2>
      <p class='muted'>Use this page to move performance and runtime tuning out of your .env. Values save to Redis and restore after Render/Railway restarts. Keep only secrets and connection URLs in environment variables.</p>
      <div class='grid'>
        {_web_status_card('Current mode', _run_state_bot_mode(), 'runtime state', 'ok' if _run_state_bot_mode() == 'POLLING' else 'info')}
        {_web_status_card('Webhook ready', 'YES' if webhook_ready else 'NO', 'base URL + secret', 'ok' if webhook_ready else 'warn')}
        {_web_status_card('User rate limit', f"{_run_state_user_rate_limit()} req/{_run_state_user_rate_window():g}s", 'live anti-flood', 'ok')}
        {_web_status_card('HTTP pool', f"{_run_state_http_max_connections()}/{_run_state_http_keepalive_connections()}", 'max/keepalive', 'ok')}
      </div>
    </div>
    <form method='post' data-confirm='Save runtime config changes?'>
      <input type='hidden' name='csrf_token' value='{csrf}'>
      <input type='hidden' name='action' value='save'>
      <div class='card'><h2>Dashboard-controlled Settings</h2><div class='table-wrap'><table class='table'><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>{_runtime_config_rows_html()}</tbody></table></div><p><button class='ok'>Save Runtime Config</button> <a class='btn secondary' href='/admin/control'>Control Center</a></p></div>
    </form>
    <div class='card'>
      <h2>Webhook Secret</h2>
      <p class='muted'>Rotate the webhook secret without editing env. If the bot is in WEBHOOK mode, Telegram is re-registered immediately.</p>
      <form method='post' data-confirm='Rotate webhook secret now? Old webhook path will stop working.'>
        <input type='hidden' name='csrf_token' value='{csrf}'>
        <input type='hidden' name='action' value='rotate_secret'>
        <button class='danger'>🔄 Rotate Webhook Secret</button>
      </form>
    </div>
    <div class='card'><h2>Minimal env after using this page</h2><pre>{_web_h(minimal_env_text)}</pre></div>
    """
    return _web_render("Runtime Config", body, active="runtime")


@app_flask.route("/admin/optimize", methods=["GET", "POST"])
@web_admin_required
def web_admin_optimize():
    """One-screen admin optimization center for performance presets and cleanup."""
    csrf = _web_csrf_token()
    admin_id = _web_current_admin_id()
    if request.method == "POST":
        _web_check_csrf()
        action = str(request.form.get("action") or "").strip().lower()
        ok = True
        msg = "Done."
        try:
            if action in {"preset_small", "preset_balanced", "preset_high"}:
                preset = action.replace("preset_", "")
                changed = _admin_runtime_update_many_sync(_optimization_preset_values(preset), timeout=12.0)
                _web_counts_invalidate()
                msg = f"Applied {preset} optimization preset. Updated: {', '.join(changed) if changed else 'no changes needed'}."
            elif action == "clear_dashboard_caches":
                _web_counts_invalidate()
                _api_key_cache_clear()
                with _blocked_cache_lock:
                    _blocked_user_cache.clear()
                _bot_settings_cache["ts"] = 0.0
                msg = "Dashboard, settings, API key, and blocked-user caches cleared."
            elif action == "trim_runtime_memory":
                msg = _admin_trim_runtime_memory_sync()
                gc.collect()
            elif action == "slow_admin_polling":
                changed = _admin_runtime_update_many_sync({
                    "WEB_STATUS_POLL_SECONDS": 60,
                    "WEB_LIVE_POLL_SECONDS": 60,
                    "WEB_COUNTS_CACHE_TTL_S": 90.0,
                }, timeout=10.0)
                msg = f"Admin polling slowed to reduce DB/API pressure. Updated: {', '.join(changed) if changed else 'no changes needed'}."
            elif action == "normal_admin_polling":
                changed = _admin_runtime_update_many_sync({
                    "WEB_STATUS_POLL_SECONDS": 30,
                    "WEB_LIVE_POLL_SECONDS": 30,
                    "WEB_COUNTS_CACHE_TTL_S": 30.0,
                }, timeout=10.0)
                msg = f"Admin polling restored to balanced settings. Updated: {', '.join(changed) if changed else 'no changes needed'}."
            elif action == "purge_temp":
                with suppress(Exception):
                    _sweep_stale_temps()
                gc.collect()
                msg = "Temp files swept and garbage collection completed."
            else:
                ok, msg = False, "Unknown optimization action."
        except Exception as exc:
            ok, msg = False, str(exc)[:500]
        _web_admin_audit(f"optimize:{action}", msg)
        flask_flash(("OK: " if ok else "ERROR: ") + msg, "success" if ok else "error")
        return redirect(url_for("web_admin_optimize"))

    snap = _runtime_performance_snapshot(light=False)
    quick_actions = [
        ("clear_dashboard_caches", "Clear dashboard caches", "Refresh counts, settings, API-key, and blocked-user caches.", "secondary"),
        ("trim_runtime_memory", "Trim runtime memory", "Remove stale in-memory limiter/notice entries and run GC.", "secondary"),
        ("slow_admin_polling", "Reduce admin polling", "Use 60s live polling and longer count cache TTL to reduce Supabase/API pressure.", "warn"),
        ("normal_admin_polling", "Restore normal polling", "Return dashboard live refresh to balanced 30s settings.", "ok"),
        ("purge_temp", "Purge temp files", "Run temp sweeper and garbage collector now.", "secondary"),
    ]
    quick_rows = "".join(
        f"<tr><td><b>{_web_h(label)}</b><br><span class='muted'>{_web_h(desc)}</span></td>"
        f"<td><form method='post' data-confirm='Run optimization action: {_web_h(label)}?'>"
        f"<input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='action' value='{_web_h(action)}'>"
        f"<button class='{_web_h(kind)}'>{_web_h(label)}</button></form></td></tr>"
        for action, label, desc, kind in quick_actions
    )
    body = f"""
    <div data-live-status></div>
    {_optimization_score_card_html(snap)}
    {_optimization_preset_cards_html(csrf)}
    <div class='grid2'>
      <div class='card'><h2>Optimization Actions</h2><p class='muted'>Safe cleanup and admin-poll tuning without restarting the bot.</p><div class='table-wrap'><table class='table'><thead><tr><th>Action</th><th>Run</th></tr></thead><tbody>{quick_rows}</tbody></table></div></div>
      <div class='card'><h2>Current Runtime Core</h2><div class='table-wrap'><table class='table'><thead><tr><th>Key</th><th>Value</th><th>Note</th></tr></thead><tbody>{_runtime_core_rows_html()}</tbody></table></div><p><a class='btn secondary' href='/admin/runtime'>Advanced runtime config</a> <a class='btn secondary' href='/admin/performance.json'>JSON snapshot</a></p></div>
    </div>
    <div class='card'><h2>Recent Slow Requests</h2><p class='muted'>Useful for finding heavy admin pages or browser polling storms.</p><div class='table-wrap'><table class='table'><thead><tr><th>Time</th><th>Method</th><th>Path</th><th>Status</th><th>Latency</th><th>Active</th></tr></thead><tbody>{_slow_request_rows_html()}</tbody></table></div></div>
    <div class='card'><h2>Full Performance Snapshot</h2><pre>{_web_h(_json.dumps(snap, ensure_ascii=False, indent=2, default=str))}</pre></div>
    """
    return _web_render("Optimize", body, active="optimize")

@app_flask.route("/admin/control")
@web_admin_required
def web_admin_control():
    perf = _runtime_performance_snapshot(light=False)
    csrf = _web_csrf_token()
    actions = [
        ("refresh_caches", "Refresh dashboard caches", "Clear dashboard counts/live schedule cache and reload bot settings on next read.", "secondary"),
        ("clear_runtime_metrics", "Reset runtime metrics", "Set in-memory runtime counters back to zero.", "warn"),
        ("reset_hf_tts", "Reset HF TTS cooldown", "Use after fixing Hugging Face / Gradio Space errors.", "warn"),
        ("reset_ocr", "Reset OCR cooldowns", "Use after fixing HF/Gemini OCR networking or quota problems.", "warn"),
        ("purge_temp", "Purge temp files", "Run the temp sweeper and garbage collector now.", "secondary"),
        ("cancel_web_broadcasts", "Cancel active web broadcasts", "Ask all running web broadcast jobs to stop safely.", "danger"),
        ("pause_all", "Emergency pause", "Maintenance ON and public TTS/OCR/audio/voice features OFF.", "danger"),
        ("resume_all", "Resume normal service", "Maintenance OFF and public features ON.", "ok"),
        ("release_lock", "Force release scheduler lock", "Only use if Render restarted and the old scheduler owner is dead.", "danger"),
    ]
    action_rows = "".join(
        f"<tr><td><b>{_web_h(label)}</b><br><span class='muted'>{_web_h(desc)}</span></td>"
        f"<td><form method='post' action='/admin/control/action' data-confirm='Run admin action: {_web_h(label)}?'><input type='hidden' name='csrf_token' value='{csrf}'><input type='hidden' name='action' value='{_web_h(action)}'><button class='{_web_h(kind)}'>{_web_h(label)}</button></form></td></tr>"
        for action, label, desc, kind in actions
    )
    audit_rows = "".join(
        f"<tr><td>{_web_h(item.get('at'))}</td><td><code>{_web_h(item.get('admin_id'))}</code></td><td>{_web_h(item.get('action'))}</td><td>{_web_h(item.get('detail'))}</td></tr>"
        for item in list(_WEB_ADMIN_AUDIT)[:50]
    ) or '<tr><td colspan="4"><div class="empty">No admin actions recorded in this process yet.</div></td></tr>'
    body = f"""
    <div data-live-status></div>
    <div class='grid'>
      {_web_status_card('Web active requests', perf['web']['active_requests'], f"slow threshold {WEB_SLOW_REQUEST_MS:.0f} ms", 'ok' if perf['web']['active_requests'] < 5 else 'warn')}
      {_web_status_card('TTS in use', perf['semaphores']['tts']['in_use'], f"max {MAX_CONCURRENT_TTS_USERS}", 'warn' if perf['semaphores']['tts']['in_use'] else '')}
      {_web_status_card('AI in use', perf['semaphores']['ai']['in_use'], f"max {MAX_CONCURRENT_AI}", 'warn' if perf['semaphores']['ai']['in_use'] else '')}
      {_web_status_card('Broadcast jobs', perf['broadcast']['active_jobs'], f"max active {WEB_BROADCAST_MAX_ACTIVE_JOBS}", 'warn' if perf['broadcast']['active_jobs'] else '')}
    </div>
    <div class='grid2'>
      <div class='card'><h2>Admin Control Actions</h2><p class='muted'>High-impact actions are protected by CSRF and confirmation. They affect this running process immediately.</p><div class='table-wrap'><table class='table'><thead><tr><th>Action</th><th>Run</th></tr></thead><tbody>{action_rows}</tbody></table></div></div>
      <div class='card'><h2>Performance Snapshot</h2><pre>{_web_h(_json.dumps(perf, ensure_ascii=False, indent=2, default=str))}</pre><p><a class='btn secondary' href='/admin/performance.json'>Open JSON</a></p></div>
    </div>
    <div class='card'><h2>Recent Admin Audit</h2><div class='table-wrap'><table class='table'><thead><tr><th>Time</th><th>Admin</th><th>Action</th><th>Detail</th></tr></thead><tbody>{audit_rows}</tbody></table></div></div>
    """
    return _web_render("Control Center", body, active="control")


@app_flask.route("/admin/control/action", methods=["POST"])
@web_admin_required
def web_admin_control_action():
    _web_check_csrf()
    action = str(request.form.get("action") or "").strip()
    ok = True
    msg = "Done."
    try:
        if action == "refresh_caches":
            _web_counts_invalidate()
            _api_key_cache_clear()
            with _blocked_cache_lock:
                _blocked_user_cache.clear()
            _bot_settings_cache["ts"] = 0.0
            msg = "Dashboard, settings, API-key, and blocked-user caches refreshed."
        elif action == "clear_runtime_metrics":
            for key in list(_RUNTIME_METRICS.keys()):
                _RUNTIME_METRICS[key] = 0
            msg = "Runtime metrics reset."
        elif action == "reset_hf_tts":
            _reset_hf_tts_cooldown()
            msg = "HF TTS cooldown reset."
        elif action == "reset_ocr":
            _reset_ocr_cooldowns()
            msg = "OCR cooldowns reset."
        elif action == "purge_temp":
            with suppress(Exception):
                _sweep_stale_temps()
            gc.collect()
            msg = "Temp sweep and garbage collection completed."
        elif action == "cancel_web_broadcasts":
            count = 0
            with _WEB_BROADCAST_JOBS_LOCK:
                for job in _WEB_BROADCAST_JOBS.values():
                    if str(job.get("status")) in {"queued", "running", "paused"}:
                        job["control"] = "cancel"
                        count += 1
            msg = f"Cancel requested for {count} active web broadcast job(s)."
        elif action == "pause_all":
            for key in ("maintenance_mode", "tts_enabled", "ocr_enabled", "voice_transcribe_enabled", "audio_transcribe_enabled", "ai_resolver_enabled"):
                db_bot_setting_set(key, key == "maintenance_mode", _web_current_admin_id())
            _bot_settings_cache["ts"] = 0.0
            msg = "Emergency pause enabled. Maintenance is ON and public features are OFF."
        elif action == "resume_all":
            db_bot_setting_set("maintenance_mode", False, _web_current_admin_id())
            for key in ("tts_enabled", "ocr_enabled", "voice_transcribe_enabled", "audio_transcribe_enabled", "ai_resolver_enabled"):
                db_bot_setting_set(key, True, _web_current_admin_id())
            _bot_settings_cache["ts"] = 0.0
            msg = "Normal service resumed."
        elif action == "release_lock":
            if not supabase:
                ok, msg = False, "Supabase is not configured."
            else:
                res = db_call_sync("web_control_lock_release", lambda: supabase.table("bot_locks").delete().eq("lock_key", _SCHED_LOCK_KEY).execute(), default=None, attempts=2, critical=False)
                ok = res is not None
                msg = "Scheduler lock release attempted." if ok else "Scheduler lock release failed."
        else:
            ok, msg = False, "Unknown control action."
    except Exception as exc:
        ok, msg = False, str(exc)[:500]
    _web_admin_audit(action, msg)
    flask_flash(("OK: " if ok else "ERROR: ") + msg, "success" if ok else "error")
    return redirect(url_for("web_admin_control"))


@app_flask.route("/admin/status.json")
@web_admin_required
def web_admin_status_json():
    light = str(request.args.get("light") or "0") == "1"
    payload = {
        "ok": True,
        "metrics": dict(_RUNTIME_METRICS),
        "uptime": _format_uptime(),
        "local_time": _fmt_local_dt(),
        "timezone": APP_TIMEZONE_NAME,
        "timezone_label": f"{APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})",
        "performance": _runtime_performance_snapshot(light=True),
    }
    if not light:
        payload.update({
            "counts": _web_counts(),
            "scheduler_lock": db_lock_read(_SCHED_LOCK_KEY) if supabase else None,
        })
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store"
    return resp




@app_flask.route("/admin/live.json")
@web_admin_required
def web_admin_live_json():
    counts = _web_counts(force=False)
    payload = {
        "ok": True,
        "counts": counts,
        "metrics": dict(_RUNTIME_METRICS),
        "uptime": _format_uptime(),
        "local_time": _fmt_local_dt(),
        "timezone_label": f"{APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})",
        "performance": _runtime_performance_snapshot(light=True),
        "jobs_html": _web_broadcast_job_rows_html(_web_csrf_token(), include_actions=False),
        "schedules_html": _web_schedule_rows_html(_web_live_schedules(), _web_csrf_token(), ""),
    }
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app_flask.route("/dashboard")
def web_admin_dashboard_alias():
    return redirect(url_for("web_admin_home"))


# ── FFmpeg ─────────────────────────────────────────────────────────────────
_FFMPEG_EXE = _iio_ffmpeg.get_ffmpeg_exe()

try:
    import edge_tts
except Exception as exc:
    edge_tts = None
    logging.getLogger(__name__).warning("Optional dependency edge_tts is not available; Edge TTS is disabled: %s", exc)
try:
    from gradio_client import Client as GradioClient
except Exception:
    GradioClient = None
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
try:
    from telegram.request import HTTPXRequest
except Exception:
    HTTPXRequest = None

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
webhook_logger = logging.getLogger("telegram_webhook")
rate_limit_logger = logging.getLogger("rate_limit")
broadcast_queue_logger = logging.getLogger("broadcast_queue")


# ---------------------------------------------------------------------------
# Admin Error Inbox Pro V8 — grouped in-memory recent error center for /admin
# ---------------------------------------------------------------------------
ADMIN_ERROR_CENTER_MAX = _env_int("ADMIN_ERROR_CENTER_MAX", 300, minimum=20, maximum=5000)
# Backward-compatible alias used by the Telegram /admin Error Center text.
_ADMIN_ERROR_CENTER_MAX = ADMIN_ERROR_CENTER_MAX
ADMIN_ERROR_MESSAGE_MAX = _env_int("ADMIN_ERROR_MESSAGE_MAX", 900, minimum=120, maximum=3000)
_ADMIN_ERROR_CENTER: deque[dict[str, Any]] = deque(maxlen=ADMIN_ERROR_CENTER_MAX)
_ADMIN_ERROR_CENTER_LOCK = threading.RLock()
_ADMIN_ERROR_CENTER_MUTED: set[str] = set()
_ADMIN_ERROR_HANDLER_INSTALLED = False


_SECRET_PATTERNS = [
    (re.compile(r"(bot\d{5,}:[A-Za-z0-9_-]{20,})"), "<telegram-token>"),
    (re.compile(r"(sk-[A-Za-z0-9_-]{16,})"), "<api-key>"),
    (re.compile(r"(eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"), "<jwt>"),
    (re.compile(r"(?i)(password|secret|token|key)=([^\s&]+)"), r"\1=<hidden>"),
]


def _admin_error_sanitize(value: Any, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text[: int(limit or ADMIN_ERROR_MESSAGE_MAX)]


def _admin_error_fingerprint(source: str, message: str) -> str:
    normalized = _admin_error_sanitize(message, 700).lower()
    normalized = re.sub(r"\b\d{4,}\b", "#", normalized)
    normalized = re.sub(r"[0-9a-f]{12,}", "#hex", normalized)
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    raw = f"{_admin_error_sanitize(source, 80).lower()}:{normalized}"[:2000]
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _admin_error_feature(source: str, message: str) -> str:
    hay = f"{source} {message}".lower()
    if "webhook" in hay or "setwebhook" in hay or "tg-webhook" in hay:
        return "Telegram Webhook"
    if "broadcast" in hay or "scheduled_broadcast" in hay:
        return "Broadcast / Schedule"
    if "supabase" in hay or "postgrest" in hay or "pgrst" in hay or "database" in hay:
        return "Supabase / DB"
    if "redis" in hay or "lock" in hay:
        return "Redis / Locks"
    if "edge-tts" in hay or "speech.platform.bing" in hay or "tts" in hay:
        return "TTS Engine"
    if "gemini" in hay or "huggingface" in hay or "hf_" in hay or "ai-assistant" in hay:
        return "AI / OCR"
    if "telegram" in hay or "bot api" in hay or "bad request" in hay:
        return "Telegram API"
    if "rate" in hay or "429" in hay:
        return "Rate Limit"
    return "Runtime"


def _admin_error_recommended_fix(feature: str, message: str) -> str:
    hay = message.lower()
    if "429" in hay or "rate" in hay:
        return "Slow down retries, increase delay settings, and check Telegram/API rate limits."
    if "message to edit not found" in hay:
        return "Use safe edit fallback: send a new message when edit/delete target no longer exists."
    if "forbidden" in hay or "bot was blocked" in hay:
        return "Mark user as blocked/inactive and skip future broadcast sends to that user."
    if "supabase" in feature.lower() or "column" in hay or "pgrst" in hay:
        return "Open SQL Setup, run missing table/column migration, then retest Health Center."
    if "redis" in feature.lower() or "lock" in feature.lower():
        return "Check REDIS_URL, Redis connection limits, and release stale locks only when no worker is active."
    if "webhook" in feature.lower():
        return "Open Control Center, verify WEBHOOK mode, TELEGRAM_WEBHOOK_URL, secret token, and active leader."
    if "tts" in feature.lower() or "403" in hay:
        return "Reset TTS cooldown, reduce parallel chunks, check proxy/VPN/network, or switch fallback provider."
    if "broadcast" in feature.lower():
        return "Open failed jobs, inspect Telegram parse/blocked errors, fix content, then retry selected users."
    return "Open the related admin page, copy this grouped log, and fix the repeated root cause first."


def _record_admin_error(source: str, message: str, *, level: str = "ERROR", context: str = "") -> None:
    """Store a small, sanitized error entry for Admin Error Inbox Pro."""
    source_s = _admin_error_sanitize(source or "unknown", 80)
    message_s = _admin_error_sanitize(message or "(empty error)", ADMIN_ERROR_MESSAGE_MAX)
    context_s = _admin_error_sanitize(context or "", 300)
    fingerprint = _admin_error_fingerprint(source_s, message_s)
    item = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source_s,
        "level": _admin_error_sanitize(level or "ERROR", 16),
        "message": message_s,
        "context": context_s,
        "feature": _admin_error_feature(source_s, message_s),
        "fingerprint": fingerprint,
    }
    with _ADMIN_ERROR_CENTER_LOCK:
        _ADMIN_ERROR_CENTER.appendleft(item)


class _AdminErrorCenterHandler(logging.Handler):
    """Capture ERROR+ log records without changing normal logging output."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.ERROR:
                return
            _record_admin_error(
                record.name,
                self.format(record),
                level=record.levelname,
                context=f"{record.pathname}:{record.lineno}",
            )
        except Exception:
            # Logging handlers must never crash the bot.
            pass


def _install_admin_error_handler() -> None:
    global _ADMIN_ERROR_HANDLER_INSTALLED
    if _ADMIN_ERROR_HANDLER_INSTALLED:
        return
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_admin_error_inbox_pro", False):
            _ADMIN_ERROR_HANDLER_INSTALLED = True
            return
    handler = _AdminErrorCenterHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._admin_error_inbox_pro = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    _ADMIN_ERROR_HANDLER_INSTALLED = True


_install_admin_error_handler()


def _admin_error_center_total_count() -> int:
    with _ADMIN_ERROR_CENTER_LOCK:
        return len(_ADMIN_ERROR_CENTER)


def _admin_error_center_snapshot(limit: int = 10) -> list[dict[str, Any]]:
    with _ADMIN_ERROR_CENTER_LOCK:
        return list(_ADMIN_ERROR_CENTER)[: max(1, min(100, int(limit or 10)))]


def _admin_error_center_clear() -> int:
    with _ADMIN_ERROR_CENTER_LOCK:
        count = len(_ADMIN_ERROR_CENTER)
        _ADMIN_ERROR_CENTER.clear()
    return count


def _admin_error_center_remove_fingerprint(fingerprint: str) -> int:
    fingerprint = str(fingerprint or "").strip()
    with _ADMIN_ERROR_CENTER_LOCK:
        old = list(_ADMIN_ERROR_CENTER)
        kept = [item for item in old if str(item.get("fingerprint")) != fingerprint]
        removed = len(old) - len(kept)
        _ADMIN_ERROR_CENTER.clear()
        _ADMIN_ERROR_CENTER.extend(kept)
    return removed


def _admin_error_center_grouped(limit: int = 50, include_muted: bool = False) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with _ADMIN_ERROR_CENTER_LOCK:
        items = list(_ADMIN_ERROR_CENTER)
        muted = set(_ADMIN_ERROR_CENTER_MUTED)
    for item in items:
        fp = str(item.get("fingerprint") or "")
        if not fp:
            continue
        is_muted = fp in muted
        if is_muted and not include_muted:
            continue
        current = grouped.get(fp)
        ts = str(item.get("ts") or "")
        if current is None:
            grouped[fp] = {
                "fingerprint": fp,
                "count": 1,
                "first_seen": ts,
                "last_seen": ts,
                "level": item.get("level") or "ERROR",
                "source": item.get("source") or "unknown",
                "feature": item.get("feature") or _admin_error_feature(item.get("source"), item.get("message")),
                "message": item.get("message") or "",
                "context": item.get("context") or "",
                "muted": is_muted,
            }
            continue
        current["count"] += 1
        current["first_seen"] = min(str(current.get("first_seen") or ts), ts)
        current["last_seen"] = max(str(current.get("last_seen") or ts), ts)
    rows = list(grouped.values())
    rows.sort(key=lambda x: (bool(x.get("muted")), str(x.get("last_seen") or "")), reverse=False)
    rows.sort(key=lambda x: str(x.get("last_seen") or ""), reverse=True)
    return rows[: max(1, min(200, int(limit or 50)))]


def _admin_error_details(fingerprint: str, limit: int = 30) -> list[dict[str, Any]]:
    fingerprint = str(fingerprint or "").strip()
    with _ADMIN_ERROR_CENTER_LOCK:
        rows = [item for item in list(_ADMIN_ERROR_CENTER) if str(item.get("fingerprint")) == fingerprint]
    return rows[: max(1, min(100, int(limit or 30)))]


def _admin_error_rows_html(rows: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for row in rows:
        fp = str(row.get("fingerprint") or "")
        priority_kind = "danger" if str(row.get("level", "")).upper() in {"ERROR", "CRITICAL"} else "warn"
        muted_badge = _web_badge("Muted", "muted") if row.get("muted") else ""
        fix = _admin_error_recommended_fix(str(row.get("feature") or ""), str(row.get("message") or ""))
        copy_text = f"[{row.get('level')}] {row.get('feature')} {row.get('source')} {row.get('message')} fp={fp} fix={fix}"
        out.append(
            "<tr>"
            f"<td>{_web_badge(row.get('level') or 'ERROR', priority_kind)}<br>{muted_badge}<br><code>{_web_h(fp)}</code></td>"
            f"<td><b>{_web_h(row.get('feature'))}</b><br><span class='muted'>{_web_h(row.get('source'))}</span></td>"
            f"<td><b>{_web_h(row.get('count'))}x</b><br><span class='muted'>First: {_web_h(_web_dt(row.get('first_seen')))}</span><br><span class='muted'>Last: {_web_h(_web_dt(row.get('last_seen')))}</span></td>"
            f"<td><div class='error-message'>{_web_h(row.get('message'))}</div><details><summary>Recommended fix</summary><p class='muted'>{_web_h(fix)}</p></details></td>"
            f"<td class='nowrap'><div class='actions'>"
            f"<a class='btn secondary' href='/admin/errors/group/{_web_h(fp)}'>Open</a>"
            f"<button type='button' class='secondary' data-copy='{_web_h(copy_text)}'>Copy</button>"
            f"<form method='post' action='/admin/errors/action' class='inline-form' data-confirm='Mute this error group?'><input type='hidden' name='csrf_token' value='{_web_h(_web_csrf_token())}'><input type='hidden' name='fingerprint' value='{_web_h(fp)}'><input type='hidden' name='action' value='mute'><button class='ghost'>Mute</button></form>"
            f"<form method='post' action='/admin/errors/action' class='inline-form' data-confirm='Clear this error group from memory?'><input type='hidden' name='csrf_token' value='{_web_h(_web_csrf_token())}'><input type='hidden' name='fingerprint' value='{_web_h(fp)}'><input type='hidden' name='action' value='clear_fingerprint'><button class='danger'>Clear</button></form>"
            f"</div></td></tr>"
        )
    return "".join(out) or '<tr><td colspan="5"><div class="empty">No unmuted errors captured yet.</div></td></tr>'


def _admin_error_detail_rows_html(rows: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{_web_h(_web_dt(row.get('ts')))}</td>"
            f"<td>{_web_badge(row.get('level') or 'ERROR', 'danger')}</td>"
            f"<td>{_web_h(row.get('source'))}<br><span class='muted'>{_web_h(row.get('context'))}</span></td>"
            f"<td><div class='error-message'>{_web_h(row.get('message'))}</div></td>"
            "</tr>"
        )
    return "".join(out) or '<tr><td colspan="4"><div class="empty">This error group is empty.</div></td></tr>'


@app_flask.route("/admin/errors")
@web_admin_required
def web_admin_errors():
    include_muted = str(request.args.get("muted") or "").lower() in {"1", "true", "yes", "on"}
    groups = _admin_error_center_grouped(limit=80, include_muted=include_muted)
    total_events = _admin_error_center_total_count()
    muted_count = len(_ADMIN_ERROR_CENTER_MUTED)
    visible_groups = len(groups)
    slow_count = len(list(globals().get("_WEB_SLOW_REQUESTS") or []))
    rows_html = _admin_error_rows_html(groups)
    csrf = _web_csrf_token()
    muted_link = "/admin/errors" if include_muted else "/admin/errors?muted=1"
    muted_label = "Hide muted" if include_muted else "Show muted"
    body = f"""
    <div data-live-status></div>
    <div class='card'>
      <div class='actions' style='justify-content:space-between'>
        <div><h2>🚨 Smart Error Inbox Pro</h2><p class='muted'>Grouped recent ERROR/CRITICAL logs by fingerprint. Memory-based, safe, no database migration required.</p></div>
        <div class='actions'>
          <a class='btn secondary' href='{_web_h(muted_link)}'>{_web_h(muted_label)}</a>
          <a class='btn secondary' href='/admin/health'>Health</a>
          <a class='btn secondary' href='/admin/optimize'>Optimize</a>
        </div>
      </div>
    </div>
    <div class='grid'>
      {_web_status_card('Error events', total_events, 'recent captured logs', 'danger' if total_events else 'ok')}
      {_web_status_card('Error groups', visible_groups, 'fingerprint groups visible', 'warn' if visible_groups else 'ok')}
      {_web_status_card('Muted groups', muted_count, 'hidden noisy fingerprints', 'muted' if muted_count else 'ok')}
      {_web_status_card('Slow requests', slow_count, 'from web telemetry', 'warn' if slow_count else 'ok')}
    </div>
    <div class='card'>
      <div class='actions' style='justify-content:space-between'>
        <div><h2>Error groups</h2><p class='muted'>Use Open for details, Copy for Claude/AI debugging, Mute for noisy known issues, Clear after fixing.</p></div>
        <div class='actions'>
          <form method='post' action='/admin/errors/action' data-confirm='Create a sample test error?'><input type='hidden' name='csrf_token' value='{_web_h(csrf)}'><input type='hidden' name='action' value='test_error'><button class='secondary'>Test Capture</button></form>
          <form method='post' action='/admin/errors/action' data-confirm='Unmute all groups?'><input type='hidden' name='csrf_token' value='{_web_h(csrf)}'><input type='hidden' name='action' value='unmute_all'><button class='ghost'>Unmute All</button></form>
          <form method='post' action='/admin/errors/action' data-confirm='Clear all captured errors from memory?'><input type='hidden' name='csrf_token' value='{_web_h(csrf)}'><input type='hidden' name='action' value='clear_all'><button class='danger'>Clear All</button></form>
        </div>
      </div>
      <div class='table-wrap'><table class='table'><thead><tr><th>Level / ID</th><th>Feature</th><th>Count / Time</th><th>Message</th><th>Actions</th></tr></thead><tbody>{rows_html}</tbody></table></div>
    </div>
    <div class='grid2'>
      <div class='card'><h2>What this fixes</h2><div class='mini-grid'>
        <div class='mini-stat'><b>Less noise</b><span class='muted'>Repeated errors are grouped instead of flooding the admin.</span></div>
        <div class='mini-stat'><b>Faster debug</b><span class='muted'>Copy one grouped log with feature, source, message, and suggested fix.</span></div>
        <div class='mini-stat'><b>Safe ops</b><span class='muted'>Mute known issues without deleting real logs.</span></div>
      </div></div>
      <div class='card'><h2>Recommended next step</h2><p class='muted'>After this, add Broadcast Quality Checker to validate parse mode, length, links, and target count before sending.</p><p><a class='btn secondary' href='/admin/insights'>Open Insights</a></p></div>
    </div>
    """
    return _web_render("Smart Error Inbox", body, active="errors")


@app_flask.route("/admin/errors/group/<fingerprint>")
@web_admin_required
def web_admin_error_detail(fingerprint: str):
    fingerprint = str(fingerprint or "").strip()
    rows = _admin_error_details(fingerprint, limit=80)
    first = rows[0] if rows else {}
    feature = first.get("feature") or "Error Group"
    message = first.get("message") or ""
    fix = _admin_error_recommended_fix(str(feature), str(message))
    copy_all = "\n".join(
        f"[{row.get('ts')}] [{row.get('level')}] {row.get('source')} {row.get('context')} {row.get('message')}"
        for row in rows[:50]
    )
    csrf = _web_csrf_token()
    body = f"""
    <div class='card'>
      <div class='actions' style='justify-content:space-between'>
        <div><h2>🚨 Error Group: <code>{_web_h(fingerprint)}</code></h2><p class='muted'>{_web_h(feature)} · {len(rows)} recent event(s)</p></div>
        <div class='actions'><a class='btn secondary' href='/admin/errors'>Back</a><button type='button' class='secondary' data-copy='{_web_h(copy_all)}'>Copy All</button></div>
      </div>
    </div>
    <div class='grid2'>
      <div class='card'><h2>Recommended fix</h2><p class='muted'>{_web_h(fix)}</p><div class='actions'>
        <form method='post' action='/admin/errors/action' data-confirm='Mute this error group?'><input type='hidden' name='csrf_token' value='{_web_h(csrf)}'><input type='hidden' name='fingerprint' value='{_web_h(fingerprint)}'><input type='hidden' name='action' value='mute'><button class='ghost'>Mute</button></form>
        <form method='post' action='/admin/errors/action' data-confirm='Clear this error group from memory?'><input type='hidden' name='csrf_token' value='{_web_h(csrf)}'><input type='hidden' name='fingerprint' value='{_web_h(fingerprint)}'><input type='hidden' name='action' value='clear_fingerprint'><button class='danger'>Clear Group</button></form>
      </div></div>
      <div class='card'><h2>Message sample</h2><pre>{_web_h(message or 'No message')}</pre></div>
    </div>
    <div class='card'><h2>Recent events</h2><div class='table-wrap'><table class='table'><thead><tr><th>Time</th><th>Level</th><th>Source</th><th>Message</th></tr></thead><tbody>{_admin_error_detail_rows_html(rows)}</tbody></table></div></div>
    """
    return _web_render("Error Detail", body, active="errors")


@app_flask.route("/admin/errors/action", methods=["POST"])
@web_admin_required
def web_admin_errors_action():
    _web_check_csrf()
    action = str(request.form.get("action") or "").strip()
    fp = str(request.form.get("fingerprint") or "").strip()
    ok = True
    msg = "No action."
    if action == "clear_all":
        count = _admin_error_center_clear()
        msg = f"Cleared {count} captured error event(s)."
    elif action == "clear_fingerprint" and fp:
        count = _admin_error_center_remove_fingerprint(fp)
        with _ADMIN_ERROR_CENTER_LOCK:
            _ADMIN_ERROR_CENTER_MUTED.discard(fp)
        msg = f"Cleared {count} event(s) for {fp}."
    elif action == "mute" and fp:
        with _ADMIN_ERROR_CENTER_LOCK:
            _ADMIN_ERROR_CENTER_MUTED.add(fp)
        msg = f"Muted error group {fp}."
    elif action == "unmute" and fp:
        with _ADMIN_ERROR_CENTER_LOCK:
            _ADMIN_ERROR_CENTER_MUTED.discard(fp)
        msg = f"Unmuted error group {fp}."
    elif action == "unmute_all":
        with _ADMIN_ERROR_CENTER_LOCK:
            count = len(_ADMIN_ERROR_CENTER_MUTED)
            _ADMIN_ERROR_CENTER_MUTED.clear()
        msg = f"Unmuted {count} group(s)."
    elif action == "test_error":
        _record_admin_error("admin.error_inbox.test", "Sample captured error from Smart Error Inbox Pro test button.", level="ERROR", context="manual-test")
        msg = "Sample error captured."
    else:
        ok = False
        msg = "Unknown or incomplete error inbox action."
    _web_admin_audit(f"errors:{action}", msg)
    flask_flash(("OK: " if ok else "ERROR: ") + msg, "success" if ok else "error")
    return redirect(_web_safe_next_url(request.form.get("return_to"), "web_admin_errors"))


@app_flask.route("/admin/errors.json")
@web_admin_required
def web_admin_errors_json():
    return jsonify({
        "ok": True,
        "events": _admin_error_center_total_count(),
        "groups": _admin_error_center_grouped(limit=80, include_muted=True),
        "muted": list(_ADMIN_ERROR_CENTER_MUTED)[:200],
        "local_time": _fmt_local_dt(),
    })


# ---------------------------------------------------------------------------
# Config  (populated by _init_clients)
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = ""
SB_URL:             str = ""
SB_KEY:             str = ""
REDIS_URL:          str = ""
REDIS_CACHE_PREFIX       = os.environ.get("REDIS_CACHE_PREFIX", "tgbot")
REDIS_PREFS_TTL_S        = _env_int("REDIS_PREFS_TTL_S", 1800, minimum=30, maximum=86400)
REDIS_TEXT_CACHE_TTL_S   = _env_int("REDIS_TEXT_CACHE_TTL_S", 86400, minimum=60, maximum=604800)
REDIS_HISTORY_TTL_S      = _env_int("REDIS_HISTORY_TTL_S", 86400, minimum=60, maximum=604800)
REDIS_SOCKET_TIMEOUT_S   = _env_float("REDIS_SOCKET_TIMEOUT_S", 3.0, minimum=0.2, maximum=30.0)
CACHE_ASIDE_DEFAULT_TTL_S = _env_int("CACHE_ASIDE_DEFAULT_TTL_S", 3600, minimum=30, maximum=86400)
HTTP_MAX_CONNECTIONS = _env_int("HTTP_MAX_CONNECTIONS", int(_perf_default("HTTP_MAX_CONNECTIONS", 100)), minimum=10, maximum=1000)
HTTP_MAX_KEEPALIVE_CONNECTIONS = _env_int("HTTP_MAX_KEEPALIVE_CONNECTIONS", int(_perf_default("HTTP_MAX_KEEPALIVE_CONNECTIONS", 20)), minimum=2, maximum=500)
REDIS_MAX_CONNECTIONS = _env_int("REDIS_MAX_CONNECTIONS", 100, minimum=2, maximum=1000)
HTTPX_HIGH_CONCURRENCY_LIMITS = httpx.Limits(
    max_connections=HTTP_MAX_CONNECTIONS,
    max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
)
BOT_MODE = (os.environ.get("BOT_MODE") or getattr(SETTINGS, "BOT_MODE", _perf_default("BOT_MODE", "WEBHOOK")) or _perf_default("BOT_MODE", "WEBHOOK")).strip().upper()
if BOT_MODE not in {"POLLING", "WEBHOOK"}:
    BOT_MODE = str(_perf_default("BOT_MODE", "WEBHOOK")).upper()
TELEGRAM_WEBHOOK_URL = (
    os.environ.get("TELEGRAM_WEBHOOK_URL")
    or getattr(SETTINGS, "TELEGRAM_WEBHOOK_URL", None)
    or os.environ.get("RENDER_EXTERNAL_URL")
    or getattr(SETTINGS, "RENDER_EXTERNAL_URL", "")
    or ""
).strip().rstrip("/")
TELEGRAM_WEBHOOK_SECRET_TOKEN = (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or getattr(SETTINGS, "TELEGRAM_WEBHOOK_SECRET_TOKEN", None) or "").strip()

# ── Render two-web-service safety ─────────────────────────────────────────
# Telegram allows only one active webhook URL per bot token.  When the same
# code is deployed as two Render Web Services, both processes must NOT call
# setWebhook/process webhook updates at the same time.  This leader lease lets
# both web dashboards stay online while only one service owns Telegram traffic.
_IS_RENDER_ENV = (
    _env_bool("RENDER", False)
    or bool(os.environ.get("RENDER_SERVICE_ID"))
    or bool(os.environ.get("RENDER_EXTERNAL_URL"))
)
TELEGRAM_MULTI_SERVER_ENABLED = _env_bool("TELEGRAM_MULTI_SERVER_ENABLED", _IS_RENDER_ENV)
TELEGRAM_ACTIVE_LOCK_ENABLED = _env_bool("TELEGRAM_ACTIVE_LOCK_ENABLED", TELEGRAM_MULTI_SERVER_ENABLED)
TELEGRAM_ACTIVE_LOCK_REQUIRED = _env_bool("TELEGRAM_ACTIVE_LOCK_REQUIRED", TELEGRAM_MULTI_SERVER_ENABLED)
TELEGRAM_ACTIVE_LOCK_KEY = (os.environ.get("TELEGRAM_ACTIVE_LOCK_KEY") or "telegram_webhook_owner").strip() or "telegram_webhook_owner"
TELEGRAM_ACTIVE_LOCK_TTL_S = _env_int("TELEGRAM_ACTIVE_LOCK_TTL_S", 90, minimum=30, maximum=3600)
TELEGRAM_ACTIVE_LOCK_RENEW_S = _env_float(
    "TELEGRAM_ACTIVE_LOCK_RENEW_S",
    max(10.0, TELEGRAM_ACTIVE_LOCK_TTL_S / 3.0),
    minimum=5.0,
    maximum=max(5.0, TELEGRAM_ACTIVE_LOCK_TTL_S - 5.0),
)
TELEGRAM_WEBHOOK_RECONFIGURE_INTERVAL_S = _env_int("TELEGRAM_WEBHOOK_RECONFIGURE_INTERVAL_S", 600, minimum=60, maximum=86400)
_PROCESS_BOOT_ID = secrets.token_hex(6)
_TELEGRAM_LEADER_OWNED = False
_TELEGRAM_LEADER_LAST_CHANGE_AT = 0.0
_TELEGRAM_LEADER_LAST_RENEW_AT = 0.0
_TELEGRAM_LEADER_LAST_WEBHOOK_SIGNATURE: tuple[Any, ...] | None = None
_TELEGRAM_LEADER_LAST_WEBHOOK_SET_AT = 0.0
_TELEGRAM_LEADER_LAST_ERROR = ""
_TELEGRAM_LEADER_OWNER_ID_CACHE: str | None = None

SUPABASE_DB_POOLER_URL = (
    os.environ.get("SUPABASE_DB_POOLER_URL")
    or os.environ.get("DATABASE_URL_POOLER")
    or os.environ.get("SUPABASE_POOLER_URL")
    or ""
).strip()
USER_RATE_LIMIT_PER_SECOND = _env_int("USER_RATE_LIMIT_PER_SECOND", 3, minimum=1, maximum=100)
USER_RATE_LIMIT_WINDOW_S = _env_float("USER_RATE_LIMIT_WINDOW_S", 1.0, minimum=0.25, maximum=60.0)
USER_RATE_LIMIT_NOTICE_COOLDOWN_S = _env_float("USER_RATE_LIMIT_NOTICE_COOLDOWN_S", 8.0, minimum=1.0, maximum=300.0)
API_RATE_LIMIT_PER_SECOND = _env_int("API_RATE_LIMIT_PER_SECOND", 20, minimum=1, maximum=1000)
API_RATE_LIMIT_WINDOW_S = _env_float("API_RATE_LIMIT_WINDOW_S", 1.0, minimum=0.25, maximum=60.0)
RATE_LIMIT_REDIS_TTL_S = _env_int("RATE_LIMIT_REDIS_TTL_S", 30, minimum=2, maximum=3600)
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
WEB_BROADCAST_QUEUE_MAXSIZE = _env_int("WEB_BROADCAST_QUEUE_MAXSIZE", 200, minimum=10, maximum=10000)

# ---------------------------------------------------------------------------
# Live runtime configuration state for in-bot admin controls
# ---------------------------------------------------------------------------
# Keep the legacy globals above for compatibility with the rest of the script,
# but hot-path logic reads through RUN_STATE so verified admins can safely tune
# selected values without restarting the process.
RUN_STATE: dict[str, Any] = {
    "BOT_MODE": BOT_MODE,
    "TELEGRAM_WEBHOOK_URL": TELEGRAM_WEBHOOK_URL,
    "TELEGRAM_WEBHOOK_SECRET_TOKEN": TELEGRAM_WEBHOOK_SECRET_TOKEN,
    "HTTP_MAX_CONNECTIONS": HTTP_MAX_CONNECTIONS,
    "HTTP_MAX_KEEPALIVE_CONNECTIONS": HTTP_MAX_KEEPALIVE_CONNECTIONS,
    "USER_RATE_LIMIT_PER_SECOND": USER_RATE_LIMIT_PER_SECOND,
    "USER_RATE_LIMIT_WINDOW_S": USER_RATE_LIMIT_WINDOW_S,
    "API_RATE_LIMIT_PER_SECOND": API_RATE_LIMIT_PER_SECOND,
    "API_RATE_LIMIT_WINDOW_S": API_RATE_LIMIT_WINDOW_S,
    "RATE_LIMIT_REDIS_TTL_S": RATE_LIMIT_REDIS_TTL_S,
    "CACHE_ASIDE_DEFAULT_TTL_S": CACHE_ASIDE_DEFAULT_TTL_S,
    "WEB_BROADCAST_QUEUE_MAXSIZE": WEB_BROADCAST_QUEUE_MAXSIZE,
    "BROADCAST_BATCH_SIZE": BROADCAST_BATCH_SIZE,
    "BROADCAST_INTER_BATCH_DELAY": BROADCAST_INTER_BATCH_DELAY,
    "MAX_CONCURRENT_TTS_USERS": MAX_CONCURRENT_TTS_USERS,
    "MAX_CONCURRENT_AI": MAX_CONCURRENT_AI,
    "MAX_CONCURRENT_BROADCAST": MAX_CONCURRENT_BROADCAST,
    "WEB_COUNTS_CACHE_TTL_S": WEB_COUNTS_CACHE_TTL_S,
    "WEB_STATUS_POLL_SECONDS": WEB_STATUS_POLL_SECONDS,
    "WEB_LIVE_POLL_SECONDS": WEB_LIVE_POLL_SECONDS,
    "TELEGRAM_CONCURRENT_UPDATES": TELEGRAM_CONCURRENT_UPDATES,
    "TELEGRAM_CONNECTION_POOL_SIZE": TELEGRAM_CONNECTION_POOL_SIZE,
    "DB_EXECUTOR_MAX_WORKERS": int(_perf_default("DB_EXECUTOR_MAX_WORKERS", 3)),
    "USER_SYNC_TTL_S": _USER_SYNC_TTL if "_USER_SYNC_TTL" in globals() else float(_perf_default("USER_SYNC_TTL_S", 1800.0)),
    "PREFS_CACHE_TTL_S": _PREFS_TTL if "_PREFS_TTL" in globals() else float(_perf_default("PREFS_CACHE_TTL_S", 600.0)),
    "TTS_AUDIO_CACHE_ENABLED": TTS_AUDIO_CACHE_ENABLED if "TTS_AUDIO_CACHE_ENABLED" in globals() else bool(_perf_default("TTS_AUDIO_CACHE_ENABLED", True)),
    "TTS_AUDIO_CACHE_MAX_MB": TTS_AUDIO_CACHE_MAX_MB if "TTS_AUDIO_CACHE_MAX_MB" in globals() else int(_perf_default("TTS_AUDIO_CACHE_MAX_MB", 64)),
    "TTS_AUDIO_CACHE_ITEM_MAX_MB": TTS_AUDIO_CACHE_ITEM_MAX_MB if "TTS_AUDIO_CACHE_ITEM_MAX_MB" in globals() else int(_perf_default("TTS_AUDIO_CACHE_ITEM_MAX_MB", 8)),
    "TTS_AUDIO_CACHE_TTL_S": TTS_AUDIO_CACHE_TTL_S if "TTS_AUDIO_CACHE_TTL_S" in globals() else float(_perf_default("TTS_AUDIO_CACHE_TTL_S", 1200.0)),
    "EDGE_TTS_PARALLEL_CHUNKS": EDGE_TTS_PARALLEL_CHUNKS if "EDGE_TTS_PARALLEL_CHUNKS" in globals() else int(_perf_default("EDGE_TTS_PARALLEL_CHUNKS", 1)),
    "PAGED_TTS_SEND_DELAY_S": PAGED_TTS_SEND_DELAY_S if "PAGED_TTS_SEND_DELAY_S" in globals() else float(_perf_default("PAGED_TTS_SEND_DELAY_S", 0.10)),
}
RUN_STATE_LOCK = asyncio.Lock()
ACTIVE_ADMIN_CONVERSATIONS: dict[int, dict[str, Any]] = {}
ACTIVE_ADMIN_CONVERSATIONS_LOCK = asyncio.Lock()
_TELEGRAM_POLLING_ACTIVE = False
_TELEGRAM_POLLING_LOCK: asyncio.Lock | None = None
# Tracks the single active long-polling lifecycle guard.  It prevents duplicate
# getUpdates workers and gives runtime mode switches one concrete task to
# cancel before Telegram webhook operations, avoiding 409 polling/webhook
# conflicts.
ACTIVE_POLLING_TASK: asyncio.Task | None = None
_RUNTIME_STATE_RESTORED_FROM_REDIS = False
_RUN_STATE_REDIS_KEYS = (
    "BOT_MODE",
    "TELEGRAM_WEBHOOK_URL",
    "TELEGRAM_WEBHOOK_SECRET_TOKEN",
    "HTTP_MAX_CONNECTIONS",
    "HTTP_MAX_KEEPALIVE_CONNECTIONS",
    "USER_RATE_LIMIT_PER_SECOND",
    "USER_RATE_LIMIT_WINDOW_S",
    "API_RATE_LIMIT_PER_SECOND",
    "API_RATE_LIMIT_WINDOW_S",
    "RATE_LIMIT_REDIS_TTL_S",
    "CACHE_ASIDE_DEFAULT_TTL_S",
    "WEB_BROADCAST_QUEUE_MAXSIZE",
    "BROADCAST_BATCH_SIZE",
    "BROADCAST_INTER_BATCH_DELAY",
    "MAX_CONCURRENT_TTS_USERS",
    "MAX_CONCURRENT_AI",
    "MAX_CONCURRENT_BROADCAST",
    "WEB_COUNTS_CACHE_TTL_S",
    "WEB_STATUS_POLL_SECONDS",
    "WEB_LIVE_POLL_SECONDS",
    "TELEGRAM_CONCURRENT_UPDATES",
    "TELEGRAM_CONNECTION_POOL_SIZE",
    "DB_EXECUTOR_MAX_WORKERS",
    "USER_SYNC_TTL_S",
    "PREFS_CACHE_TTL_S",
    "TTS_AUDIO_CACHE_ENABLED",
    "TTS_AUDIO_CACHE_MAX_MB",
    "TTS_AUDIO_CACHE_ITEM_MAX_MB",
    "TTS_AUDIO_CACHE_TTL_S",
    "EDGE_TTS_PARALLEL_CHUNKS",
    "PAGED_TTS_SEND_DELAY_S",
)

_RUNTIME_CONFIG_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("BOT_MODE", {"kind": "mode", "label": "Bot mode", "help": "POLLING for one node; WEBHOOK for scalable deployment."}),
    ("TELEGRAM_WEBHOOK_URL", {"kind": "url", "label": "Webhook base URL", "help": "Public HTTPS base URL only. Do not include /tg-webhook-..."}),
    ("TELEGRAM_WEBHOOK_SECRET_TOKEN", {"kind": "secret", "label": "Webhook secret", "help": "Rotatable secret used in the webhook path and Telegram secret header."}),
    ("USER_RATE_LIMIT_PER_SECOND", {"kind": "int", "min": 1, "max": 100, "label": "User rate limit / second", "help": "Anti-flood limit per chat/user."}),
    ("USER_RATE_LIMIT_WINDOW_S", {"kind": "float", "min": 0.25, "max": 60.0, "label": "User rate window seconds", "help": "Sliding window duration for user anti-flood."}),
    ("API_RATE_LIMIT_PER_SECOND", {"kind": "int", "min": 1, "max": 1000, "label": "API rate limit / second", "help": "Anti-flood limit for hot web/API endpoints."}),
    ("API_RATE_LIMIT_WINDOW_S", {"kind": "float", "min": 0.25, "max": 60.0, "label": "API rate window seconds", "help": "Sliding window duration for API limiter."}),
    ("RATE_LIMIT_REDIS_TTL_S", {"kind": "int", "min": 2, "max": 3600, "label": "Rate limit Redis TTL", "help": "TTL for Redis-backed limiter keys."}),
    ("HTTP_MAX_CONNECTIONS", {"kind": "int", "min": 10, "max": 1000, "label": "HTTP max connections", "help": "httpx pooled max connections for Telegram/API calls."}),
    ("HTTP_MAX_KEEPALIVE_CONNECTIONS", {"kind": "int", "min": 2, "max": 500, "label": "HTTP keepalive connections", "help": "httpx keep-alive pool size."}),
    ("CACHE_ASIDE_DEFAULT_TTL_S", {"kind": "int", "min": 30, "max": 86400, "label": "Cache-aside TTL seconds", "help": "Default Redis cache TTL for cache-aside data."}),
    ("WEB_BROADCAST_QUEUE_MAXSIZE", {"kind": "int", "min": 10, "max": 10000, "label": "Broadcast queue size", "help": "Max queued admin broadcast jobs. Resize takes effect after restart for the active queue."}),
    ("BROADCAST_BATCH_SIZE", {"kind": "int", "min": 1, "max": 500, "label": "Broadcast batch size", "help": "Users per broadcast batch."}),
    ("BROADCAST_INTER_BATCH_DELAY", {"kind": "float", "min": 0.0, "max": 30.0, "label": "Broadcast batch delay", "help": "Delay between broadcast batches."}),
    ("MAX_CONCURRENT_TTS_USERS", {"kind": "int", "min": 1, "max": 50, "label": "Max concurrent TTS", "help": "Concurrent TTS workload cap. New semaphore applies to future requests."}),
    ("MAX_CONCURRENT_AI", {"kind": "int", "min": 1, "max": 50, "label": "Max concurrent AI", "help": "Concurrent AI workload cap. New semaphore applies to future requests."}),
    ("MAX_CONCURRENT_BROADCAST", {"kind": "int", "min": 1, "max": 50, "label": "Max concurrent broadcast", "help": "Bot broadcast semaphore cap."}),
    ("WEB_COUNTS_CACHE_TTL_S", {"kind": "float", "min": 5.0, "max": 600.0, "label": "Dashboard counts cache TTL", "help": "Admin dashboard count cache duration."}),
    ("WEB_STATUS_POLL_SECONDS", {"kind": "int", "min": 10, "max": 300, "label": "Dashboard status poll seconds", "help": "Browser live status refresh interval."}),
    ("WEB_LIVE_POLL_SECONDS", {"kind": "int", "min": 15, "max": 300, "label": "Dashboard live poll seconds", "help": "Live dashboard refresh interval."}),
    ("TELEGRAM_CONCURRENT_UPDATES", {"kind": "int", "min": 1, "max": 64, "label": "Telegram concurrent updates", "help": "Update processor concurrency. Restart required."}),
    ("TELEGRAM_CONNECTION_POOL_SIZE", {"kind": "int", "min": 4, "max": 256, "label": "Telegram connection pool", "help": "Telegram HTTP pool size. Restart required."}),
    ("DB_EXECUTOR_MAX_WORKERS", {"kind": "int", "min": 1, "max": 16, "label": "DB executor workers", "help": "Supabase background worker threads."}),
    ("USER_SYNC_TTL_S", {"kind": "float", "min": 60.0, "max": 86400.0, "label": "User sync TTL", "help": "Minimum seconds between user sync writes."}),
    ("PREFS_CACHE_TTL_S", {"kind": "float", "min": 30.0, "max": 86400.0, "label": "Prefs cache TTL", "help": "User prefs cache duration."}),
    ("TTS_AUDIO_CACHE_ENABLED", {"kind": "bool", "label": "TTS audio cache", "help": "Reuse matching generated audio."}),
    ("TTS_AUDIO_CACHE_MAX_MB", {"kind": "int", "min": 4, "max": 512, "label": "TTS cache max MB", "help": "Total in-memory TTS audio cache."}),
    ("TTS_AUDIO_CACHE_ITEM_MAX_MB", {"kind": "int", "min": 1, "max": 64, "label": "TTS cache item max MB", "help": "Max single cached audio item."}),
    ("TTS_AUDIO_CACHE_TTL_S", {"kind": "float", "min": 30.0, "max": 86400.0, "label": "TTS cache TTL", "help": "Seconds before cached audio expires."}),
    ("EDGE_TTS_PARALLEL_CHUNKS", {"kind": "int", "min": 1, "max": 4, "label": "Edge TTS parallel chunks", "help": "Use 1 for stability; 2+ may trigger 403."}),
    ("PAGED_TTS_SEND_DELAY_S", {"kind": "float", "min": 0.0, "max": 2.0, "label": "Paged TTS send delay", "help": "Delay between chunked voice messages."}),
])


def _run_state_get(key: str, default: Any = None) -> Any:
    return RUN_STATE.get(key, default)


def _run_state_bot_mode() -> str:
    mode = str(RUN_STATE.get("BOT_MODE") or BOT_MODE or _perf_default("BOT_MODE", "WEBHOOK")).strip().upper()
    return mode if mode in {"POLLING", "WEBHOOK"} else str(_perf_default("BOT_MODE", "WEBHOOK")).upper()


def _run_state_user_rate_limit() -> int:
    try:
        return max(1, int(RUN_STATE.get("USER_RATE_LIMIT_PER_SECOND", USER_RATE_LIMIT_PER_SECOND)))
    except Exception:
        return max(1, int(USER_RATE_LIMIT_PER_SECOND or 3))


def _run_state_float(key: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Read a float from RUN_STATE with bounds for hot-path runtime tuning."""
    try:
        value = float(RUN_STATE.get(key, default))
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def _run_state_int(key: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Read an int from RUN_STATE with bounds for hot-path runtime tuning."""
    try:
        value = int(RUN_STATE.get(key, default))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


def _run_state_user_rate_window() -> float:
    return _run_state_float("USER_RATE_LIMIT_WINDOW_S", USER_RATE_LIMIT_WINDOW_S, minimum=0.25, maximum=60.0)


def _run_state_api_rate_limit() -> int:
    return _run_state_int("API_RATE_LIMIT_PER_SECOND", API_RATE_LIMIT_PER_SECOND, minimum=1, maximum=1000)


def _run_state_api_rate_window() -> float:
    return _run_state_float("API_RATE_LIMIT_WINDOW_S", API_RATE_LIMIT_WINDOW_S, minimum=0.25, maximum=60.0)


def _run_state_rate_limit_redis_ttl() -> int:
    return _run_state_int("RATE_LIMIT_REDIS_TTL_S", RATE_LIMIT_REDIS_TTL_S, minimum=2, maximum=3600)


def _run_state_cache_ttl() -> int:
    return _run_state_int("CACHE_ASIDE_DEFAULT_TTL_S", CACHE_ASIDE_DEFAULT_TTL_S, minimum=30, maximum=86400)


def _run_state_web_broadcast_queue_maxsize() -> int:
    return _run_state_int("WEB_BROADCAST_QUEUE_MAXSIZE", WEB_BROADCAST_QUEUE_MAXSIZE, minimum=10, maximum=10000)


def _run_state_broadcast_batch_size() -> int:
    return _run_state_int("BROADCAST_BATCH_SIZE", BROADCAST_BATCH_SIZE, minimum=1, maximum=500)


def _run_state_broadcast_delay() -> float:
    return _run_state_float("BROADCAST_INTER_BATCH_DELAY", BROADCAST_INTER_BATCH_DELAY, minimum=0.0, maximum=30.0)


def _run_state_max_concurrent_broadcast() -> int:
    return _run_state_int("MAX_CONCURRENT_BROADCAST", MAX_CONCURRENT_BROADCAST, minimum=1, maximum=50)


def _run_state_web_counts_cache_ttl() -> float:
    return _run_state_float("WEB_COUNTS_CACHE_TTL_S", WEB_COUNTS_CACHE_TTL_S, minimum=5.0, maximum=600.0)


def _run_state_http_max_connections() -> int:
    try:
        return max(10, int(RUN_STATE.get("HTTP_MAX_CONNECTIONS", HTTP_MAX_CONNECTIONS)))
    except Exception:
        return max(10, int(HTTP_MAX_CONNECTIONS or 100))


def _run_state_http_keepalive_connections() -> int:
    try:
        max_conn = _run_state_http_max_connections()
        keepalive = int(RUN_STATE.get("HTTP_MAX_KEEPALIVE_CONNECTIONS", HTTP_MAX_KEEPALIVE_CONNECTIONS))
        return min(max(2, keepalive), max_conn)
    except Exception:
        return min(max(2, int(HTTP_MAX_KEEPALIVE_CONNECTIONS or 20)), _run_state_http_max_connections())


def _make_httpx_limits_from_run_state() -> httpx.Limits:
    max_connections = _run_state_http_max_connections()
    keepalive = _run_state_http_keepalive_connections()
    return httpx.Limits(max_connections=max_connections, max_keepalive_connections=keepalive)


def _coerce_run_state_value(key: str, value: Any) -> Any:
    spec = _RUNTIME_CONFIG_SPECS.get(key, {})
    kind = spec.get("kind", "str")
    if kind == "mode":
        mode = str(value or "POLLING").strip().upper()
        if mode not in {"POLLING", "WEBHOOK"}:
            raise ValueError("BOT_MODE must be POLLING or WEBHOOK")
        return mode
    if kind == "int":
        v = int(str(value).strip())
        return max(int(spec.get("min", v)), min(int(spec.get("max", v)), v))
    if kind == "float":
        v = float(str(value).strip())
        return max(float(spec.get("min", v)), min(float(spec.get("max", v)), v))
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        raise ValueError(f"{key} must be true/false or 1/0")
    if kind == "url":
        return str(value or "").strip().rstrip("/")
    if kind == "secret":
        token = str(value or "").strip()
        if not token:
            raise ValueError(f"{key} cannot be empty")
        return token
    return str(value if value is not None else "").strip()


def _sync_run_state_from_globals() -> None:
    for key in _RUN_STATE_REDIS_KEYS:
        if key in globals():
            try:
                RUN_STATE[key] = _coerce_run_state_value(key, globals().get(key))
            except Exception:
                RUN_STATE[key] = globals().get(key)


async def _persist_run_state_key(key: str, value: Any) -> bool:
    """Persist runtime admin overrides so container restarts keep the last value.

    The colon key is the primary format.  The underscore key is written as a
    compatibility mirror because earlier deployment notes referenced that name.
    """
    if redis_client is None:
        return False
    raw_value = str(value)
    primary_key = f"RUN_STATE:{key}"
    compat_key = f"RUN_STATE_{key}"
    extra_keys: list[str] = []
    if key == "TELEGRAM_WEBHOOK_SECRET_TOKEN":
        # Required stable alias for webhook secret rotation persistence.
        extra_keys.append("RUN_STATE_WEBHOOK_SECRET")

    def _run():
        pipe = redis_client.pipeline(transaction=False)
        pipe.set(primary_key, raw_value)
        pipe.set(compat_key, raw_value)
        for extra_key in extra_keys:
            pipe.set(extra_key, raw_value)
        pipe.execute()
        return True

    try:
        ok = await redis_call(
            f"run_state_set:{key}",
            _run,
            default=False,
            attempts=2,
            timeout=2.0,
            critical=False,
        )
        if ok:
            safe_value = "<redacted>" if key == "TELEGRAM_WEBHOOK_SECRET_TOKEN" else raw_value
            webhook_logger.info("Runtime state persisted to Redis keys=%s,%s value=%s", primary_key, compat_key, safe_value)
        return bool(ok)
    except Exception as exc:
        webhook_logger.warning("Runtime state Redis persist failed key=%s: %s", key, exc)
        return False


async def _restore_run_state_from_redis() -> bool:
    """Load persisted runtime/admin dashboard overrides after Redis is initialised.

    Redis is optional. Startup pings Redis before key reads. If Redis is down,
    the app keeps .env/SETTINGS defaults and continues booting.
    """
    global _RUNTIME_STATE_RESTORED_FROM_REDIS
    if redis_client is None:
        webhook_logger.info("Runtime state restore skipped: Redis is not configured.")
        return False

    keys = list(_RUN_STATE_REDIS_KEYS)
    primary_keys = [f"RUN_STATE:{key}" for key in keys]
    compat_keys = [f"RUN_STATE_{key}" for key in keys]
    extra_restore_keys = ["RUN_STATE_WEBHOOK_SECRET"]

    def _ping():
        return redis_client.ping()

    def _fetch_values():
        pipe = redis_client.pipeline(transaction=False)
        for redis_key in primary_keys + compat_keys + extra_restore_keys:
            pipe.get(redis_key)
        return pipe.execute()

    try:
        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=2.5)
        values = await redis_call(
            "run_state_restore",
            _fetch_values,
            default=None,
            attempts=2,
            timeout=3.0,
            critical=False,
        )
    except Exception as rexc:
        webhook_logger.warning("Redis fallback triggered. Connection not ready during boot recovery: %s", rexc)
        return False

    if not values:
        webhook_logger.info("Runtime state restore found no Redis overrides; using env defaults.")
        return False

    n = len(keys)
    primary_values = list(values[:n])
    compat_values = list(values[n:n * 2])
    restored: dict[str, Any] = {}

    for idx, key in enumerate(keys):
        raw = primary_values[idx] if idx < len(primary_values) else None
        if raw in (None, "") and idx < len(compat_values):
            raw = compat_values[idx]
        if raw in (None, "") and key == "TELEGRAM_WEBHOOK_SECRET_TOKEN":
            extra_idx = n * 2
            if extra_idx < len(values):
                raw = values[extra_idx]
        if raw is None or raw == "":
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            coerced = _coerce_run_state_value(key, raw)
            await _update_run_state(key, coerced, persist=False)
            restored[key] = "<redacted>" if key == "TELEGRAM_WEBHOOK_SECRET_TOKEN" else coerced
        except Exception as exc:
            webhook_logger.warning("Ignoring invalid persisted runtime state %s=%r: %s", key, raw, exc)

    _RUNTIME_STATE_RESTORED_FROM_REDIS = bool(restored)
    if restored:
        webhook_logger.info("Runtime state recovered from Redis before startup activation: %s", restored)
    else:
        webhook_logger.info("Runtime state restore completed with no valid overrides.")
    return bool(restored)


async def _update_run_state(key: str, value: Any, *, persist: bool = True) -> None:
    """Update runtime config from Telegram admin or Web Admin Dashboard.

    Values are persisted to Redis when available, mirrored into legacy globals,
    and applied to hot runtime objects where safe. Secrets are never logged in
    clear text.
    """
    global HTTPX_HIGH_CONCURRENCY_LIMITS
    global _AI_SEMAPHORE, _BROADCAST_SEMAPHORE, _TTS_CHUNK_SEMAPHORE, _DB_EXECUTOR
    global TTS_AUDIO_CACHE_MAX_BYTES, TTS_AUDIO_CACHE_ITEM_MAX_BYTES

    if key not in _RUN_STATE_REDIS_KEYS:
        raise ValueError(f"Unsupported runtime setting: {key}")

    persisted_value = _coerce_run_state_value(key, value)
    async with RUN_STATE_LOCK:
        RUN_STATE[key] = persisted_value
        globals()[key] = persisted_value
        if key in {"HTTP_MAX_CONNECTIONS", "HTTP_MAX_KEEPALIVE_CONNECTIONS"}:
            HTTPX_HIGH_CONCURRENCY_LIMITS = _make_httpx_limits_from_run_state()
        elif key == "MAX_CONCURRENT_AI" and _AI_SEMAPHORE is not None:
            _AI_SEMAPHORE = asyncio.Semaphore(int(persisted_value))
        elif key == "MAX_CONCURRENT_BROADCAST" and _BROADCAST_SEMAPHORE is not None:
            _BROADCAST_SEMAPHORE = asyncio.Semaphore(int(persisted_value))
        elif key == "MAX_CONCURRENT_TTS_USERS" and _TTS_CHUNK_SEMAPHORE is not None:
            _TTS_CHUNK_SEMAPHORE = asyncio.Semaphore(int(persisted_value))
        elif key == "DB_EXECUTOR_MAX_WORKERS":
            old_executor = _DB_EXECUTOR
            _DB_EXECUTOR = ThreadPoolExecutor(max_workers=int(persisted_value), thread_name_prefix="db_write")
            with suppress(Exception):
                old_executor.shutdown(wait=False, cancel_futures=False)
        elif key == "PREFS_CACHE_TTL_S":
            globals()["_PREFS_TTL"] = float(persisted_value)
        elif key == "USER_SYNC_TTL_S":
            globals()["_USER_SYNC_TTL"] = float(persisted_value)
        elif key in {"TTS_AUDIO_CACHE_MAX_MB", "TTS_AUDIO_CACHE_ITEM_MAX_MB"}:
            globals()[key] = int(persisted_value)
            TTS_AUDIO_CACHE_MAX_BYTES = int(globals().get("TTS_AUDIO_CACHE_MAX_MB", 64)) * 1024 * 1024
            TTS_AUDIO_CACHE_ITEM_MAX_BYTES = int(globals().get("TTS_AUDIO_CACHE_ITEM_MAX_MB", 8)) * 1024 * 1024
        elif key in {"TTS_AUDIO_CACHE_TTL_S", "PAGED_TTS_SEND_DELAY_S"}:
            globals()[key] = float(persisted_value)
        elif key in {"TTS_AUDIO_CACHE_ENABLED"}:
            globals()[key] = bool(persisted_value)
        elif key in {"EDGE_TTS_PARALLEL_CHUNKS", "TELEGRAM_CONCURRENT_UPDATES", "TELEGRAM_CONNECTION_POOL_SIZE"}:
            globals()[key] = int(persisted_value)
        elif key in {"WEB_COUNTS_CACHE_TTL_S", "WEB_STATUS_POLL_SECONDS", "WEB_LIVE_POLL_SECONDS"}:
            _web_counts_invalidate()

    if persist:
        await _persist_run_state_key(key, persisted_value)

def _runtime_admin_text() -> str:
    mode = _run_state_bot_mode()
    rate_limit = _run_state_user_rate_limit()
    http_conn = _run_state_http_max_connections()
    polling_status = "ON" if globals().get("_TELEGRAM_POLLING_ACTIVE") else "OFF"
    webhook_ready = "YES" if TELEGRAM_WEBHOOK_URL and _runtime_webhook_secret_token() else "NO"
    return (
        "🛠️ <b>ផ្ទាំងគ្រប់គ្រង Admin Runtime</b>\n\n"
        "តម្លៃសំខាន់ៗបច្ចុប្បន្ន៖\n"
        f"• Bot Mode: <b>{html.escape(mode)}</b>\n"
        f"• Polling Active: <b>{html.escape(polling_status)}</b>\n"
        f"• Webhook Config Ready: <b>{html.escape(webhook_ready)}</b>\n"
        f"• User Rate Limit: <b>{rate_limit} req/{_run_state_user_rate_window():g}s</b>\n"
        f"• HTTP Max Connections: <b>{http_conn}</b>\n\n"
        "ជ្រើសរើសសកម្មភាពខាងក្រោម ដើម្បីកែប្រែ Runtime ដោយមិនបាច់ Restart។"
    )


def get_runtime_admin_kb() -> Any:
    mode = _run_state_bot_mode()
    target = "WEBHOOK" if mode != "WEBHOOK" else "POLLING"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🔄 Switch to {target}", callback_data=f"rtadmin_switch:{target}")],
        [InlineKeyboardButton("⚡ Modify Rate Limit", callback_data="rtadmin_rate")],
        [InlineKeyboardButton("🔄 Rotate Webhook Secret", callback_data="rtadmin_rotate_secret")],
        [InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home")],
        [InlineKeyboardButton("❌ Close Menu", callback_data="rtadmin_close")],
    ])


def _refresh_runtime_admin_markup() -> Any:
    return get_runtime_admin_kb()


def _runtime_admin_log_state(action: str, admin_id: int, extra: str = "") -> None:
    logger.info(
        "Runtime admin action=%s admin_id=%s mode=%s user_rate_limit=%s http_max_connections=%s %s",
        action,
        admin_id,
        _run_state_bot_mode(),
        _run_state_user_rate_limit(),
        _run_state_http_max_connections(),
        extra,
    )


async def _cancel_active_polling_task(reason: str = "runtime_switch") -> None:
    """Cancel the tracked polling lifecycle task exactly once.

    This guard prevents duplicate long-polling workers and ensures mode changes
    cannot leave getUpdates running while a webhook is active.
    """
    global ACTIVE_POLLING_TASK
    task = ACTIVE_POLLING_TASK
    if task is None:
        return
    if task.done():
        ACTIVE_POLLING_TASK = None
        return
    current = asyncio.current_task()
    webhook_logger.info("Cancelling active polling task reason=%s task=%s", reason, task.get_name() if hasattr(task, "get_name") else task)
    task.cancel()
    if task is not current:
        with suppress(asyncio.CancelledError, Exception):
            await task
    ACTIVE_POLLING_TASK = None
    webhook_logger.info("Active polling task cancellation completed reason=%s", reason)


async def _telegram_polling_task_guard(app_obj: Any) -> None:
    """Small lifecycle guard for the active updater polling session.

    python-telegram-bot owns the actual getUpdates worker internally after
    updater.start_polling(); this task gives the runtime controller a single
    cancellable handle and exits if BOT_MODE changes away from POLLING.
    """
    global ACTIVE_POLLING_TASK
    try:
        while True:
            if _run_state_bot_mode() != "POLLING":
                webhook_logger.info("Polling guard detected BOT_MODE=%s; stopping updater to prevent 409 conflicts.", _run_state_bot_mode())
                await _telegram_stop_polling_runtime(app_obj, cancel_task=False)
                break
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        webhook_logger.info("Polling guard task cancelled.")
        raise
    except Exception as exc:
        webhook_logger.error("Polling guard failed: %s", exc, exc_info=True)
    finally:
        if ACTIVE_POLLING_TASK is asyncio.current_task():
            ACTIVE_POLLING_TASK = None


def _telegram_polling_lock() -> asyncio.Lock:
    global _TELEGRAM_POLLING_LOCK
    if _TELEGRAM_POLLING_LOCK is None:
        _TELEGRAM_POLLING_LOCK = asyncio.Lock()
    return _TELEGRAM_POLLING_LOCK


def _telegram_app_updater(app_obj: Any) -> Any:
    return getattr(app_obj, "updater", None) if app_obj is not None else None


async def _telegram_start_polling_runtime(app_obj: Any) -> bool:
    global _TELEGRAM_POLLING_ACTIVE, ACTIVE_POLLING_TASK
    updater = _telegram_app_updater(app_obj)
    if updater is None:
        webhook_logger.warning("Cannot start polling dynamically: Telegram updater is unavailable.")
        return False
    if _run_state_bot_mode() != "POLLING":
        webhook_logger.info("Polling start skipped because BOT_MODE=%s.", _run_state_bot_mode())
        return False

    # Always cancel any stale guard before starting a fresh polling session.
    await _cancel_active_polling_task("before_start_polling")

    async with _telegram_polling_lock():
        if _TELEGRAM_POLLING_ACTIVE:
            webhook_logger.info("Polling updater already active; duplicate start suppressed.")
            if ACTIVE_POLLING_TASK is None or ACTIVE_POLLING_TASK.done():
                ACTIVE_POLLING_TASK = asyncio.create_task(_telegram_polling_task_guard(app_obj), name="telegram-polling-guard")
            return True
        await updater.start_polling(
            allowed_updates=_telegram_allowed_updates(),
            drop_pending_updates=_env_bool("TELEGRAM_POLLING_DROP_PENDING_UPDATES", True),
        )
        _TELEGRAM_POLLING_ACTIVE = True
        ACTIVE_POLLING_TASK = asyncio.create_task(_telegram_polling_task_guard(app_obj), name="telegram-polling-guard")
        webhook_logger.info("Telegram long polling started dynamically; guard task created.")
        return True


async def _telegram_stop_polling_runtime(app_obj: Any, *, cancel_task: bool = True) -> bool:
    global _TELEGRAM_POLLING_ACTIVE
    updater = _telegram_app_updater(app_obj)
    if cancel_task:
        await _cancel_active_polling_task("stop_polling")
    if updater is None:
        _TELEGRAM_POLLING_ACTIVE = False
        return False
    async with _telegram_polling_lock():
        if not _TELEGRAM_POLLING_ACTIVE:
            return True
        with suppress(Exception):
            await updater.stop()
        _TELEGRAM_POLLING_ACTIVE = False
        webhook_logger.info("Telegram long polling stopped dynamically.")
        return True


async def _delete_telegram_webhook_via_http(drop_pending: bool = True) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")
    api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout, limits=_make_httpx_limits_from_run_state()) as client:
        resp = await client.post(api_url, json={"drop_pending_updates": bool(drop_pending)})
    try:
        data = _json_loads_fast(resp.content)
    except Exception:
        data = {"ok": False, "description": resp.text[:500]}
    if resp.status_code >= 400 or not bool(data.get("ok")):
        raise RuntimeError(f"Telegram deleteWebhook failed status={resp.status_code} response={str(data)[:500]}")
    webhook_logger.info("Telegram webhook deleted via HTTPX.")


async def _switch_telegram_runtime_mode(target_mode: str, admin_id: int = 0) -> str:
    target = str(target_mode or "POLLING").strip().upper()
    if target not in {"POLLING", "WEBHOOK"}:
        raise ValueError("Invalid target Telegram mode.")
    app_obj = globals().get("_TELEGRAM_APP")
    if app_obj is None:
        raise RuntimeError("Telegram application is not ready yet.")

    if target == "WEBHOOK":
        if not _runtime_webhook_base_url() or not _runtime_webhook_secret_token():
            raise RuntimeError("TELEGRAM_WEBHOOK_URL/RENDER_EXTERNAL_URL and TELEGRAM_WEBHOOK_SECRET_TOKEN are required before switching to WEBHOOK.")
        webhook_logger.info("Runtime switch to WEBHOOK requested by admin_id=%s", admin_id)
        # Hard-stop any getUpdates worker before setWebhook to avoid Telegram 409.
        await _cancel_active_polling_task("switch_to_webhook")
        await _telegram_stop_polling_runtime(app_obj, cancel_task=False)
        await _configure_telegram_webhook_via_http()
        await _update_run_state("BOT_MODE", "WEBHOOK")
        _runtime_admin_log_state("switch_webhook", admin_id)
        return "WEBHOOK"

    webhook_logger.info("Runtime switch to POLLING requested by admin_id=%s", admin_id)
    # Remove webhook first, verify Telegram accepted it, then start one polling worker.
    await _cancel_active_polling_task("switch_to_polling_pre_delete")
    await _delete_telegram_webhook_via_http(drop_pending=True)
    # Telegram may still release the previous webhook/getUpdates state shortly
    # after deleteWebhook returns. A short grace delay prevents intermittent
    # 409 Conflict errors when polling starts immediately.
    await asyncio.sleep(1.5)
    await _update_run_state("BOT_MODE", "POLLING")
    await _telegram_start_polling_runtime(app_obj)
    _runtime_admin_log_state("switch_polling", admin_id)
    return "POLLING"


def _runtime_admin_notice_text() -> str:
    return "⏳ Too many requests. Please slow down."


def _is_runtime_admin_callback(data: str) -> bool:
    return str(data or "").startswith("rtadmin_")


async def _runtime_admin_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    if query is None:
        return
    admin_id = query.from_user.id if query.from_user else 0
    data = query.data or ""
    if not _is_admin(admin_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    if query.message is None:
        with suppress(Exception):
            await query.answer()
        return

    if data == "rtadmin_close":
        async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS.pop(admin_id, None)
        with suppress(Exception):
            await query.answer("Closed")
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "rtadmin_rate":
        async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS[admin_id] = {"state": "awaiting_rate_limit", "ts": time.monotonic()}
        with suppress(Exception):
            await query.answer("Send a number", show_alert=False)
        await safe_send(lambda: query.message.edit_text(
            "⚡ <b>កែប្រែ Rate Limit</b>\n\n"
            f"តម្លៃបច្ចុប្បន្ន: <b>{_run_state_user_rate_limit()} req/{_run_state_user_rate_window():g}s</b>\n\n"
            "សូមផ្ញើលេខគត់ថ្មី ឧទាហរណ៍ <code>3</code> ឬ <code>5</code>។\n"
            "បើចង់កែ HTTP pool សូមផ្ញើ <code>http 120</code>។",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="rtadmin_cancel")]]),
        ))
        return

    if data == "rtadmin_cancel":
        async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS.pop(admin_id, None)
        with suppress(Exception):
            await query.answer("Cancelled")
        await safe_send(lambda: query.message.edit_text(
            _runtime_admin_text(),
            parse_mode="HTML",
            reply_markup=_refresh_runtime_admin_markup(),
        ))
        return

    if data == "rtadmin_rotate_secret":
        with suppress(Exception):
            await query.answer("Rotating secret…", show_alert=False)

        async with _webhook_rotate_lock():
            remaining = _webhook_rotate_begin_or_remaining()
            if remaining < 0:
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text()
                    + "\n\n⚠️ Webhook secret rotation is already running. Please wait.",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
                return
            if remaining > 0:
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text()
                    + f"\n\n⚠️ Please wait {int(remaining) + 1}s before rotating the webhook secret again.",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
                return

            success = False
            new_token = generate_new_webhook_token()
            try:
                # Important order: ask Telegram to accept the new webhook first.
                # Only persist RUN_STATE after setWebhook succeeds.
                if _run_state_bot_mode() == "WEBHOOK":
                    await _configure_telegram_webhook_via_http_for_secret(new_token)

                await _update_run_state("TELEGRAM_WEBHOOK_SECRET_TOKEN", new_token, persist=True)
                success = True

                logger.info("Admin %s rotated Webhook Secret Token.", admin_id)
                webhook_logger.info("Webhook secret rotated by admin_id=%s mode=%s", admin_id, _run_state_bot_mode())

                new_path = f"/tg-webhook-{new_token}"
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text()
                    + "\n\n✅ Webhook secret updated!"
                    + f"\nNew URL path: <code>{html.escape(new_path)}</code>",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
            except Exception as exc:
                webhook_logger.error("Webhook secret rotation failed admin_id=%s: %s", admin_id, exc, exc_info=True)
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text() + f"\n\n❌ Rotate secret failed: <code>{html.escape(str(exc)[:800])}</code>",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
            finally:
                _webhook_rotate_finish(success)
        return

    if data.startswith("rtadmin_switch:"):
        target = data.split(":", 1)[1].strip().upper()
        with suppress(Exception):
            await query.answer("Switching…", show_alert=False)
        try:
            mode = await _switch_telegram_runtime_mode(target, admin_id=admin_id)
            await safe_send(lambda: query.message.edit_text(
                _runtime_admin_text() + f"\n\n✅ បានប្ដូរទៅ <b>{html.escape(mode)}</b> រួចរាល់។",
                parse_mode="HTML",
                reply_markup=_refresh_runtime_admin_markup(),
                disable_web_page_preview=True,
            ))
        except Exception as exc:
            webhook_logger.error("Runtime mode switch failed admin_id=%s target=%s: %s", admin_id, target, exc, exc_info=True)
            await safe_send(lambda: query.message.edit_text(
                _runtime_admin_text() + f"\n\n❌ ប្ដូរ Mode មិនបាន: <code>{html.escape(str(exc)[:800])}</code>",
                parse_mode="HTML",
                reply_markup=_refresh_runtime_admin_markup(),
                disable_web_page_preview=True,
            ))
        return

    with suppress(Exception):
        await query.answer()


async def _handle_runtime_admin_text(update: Any, context: Any) -> bool:
    msg = update.message
    user = update.effective_user
    if not msg or not user or not _is_admin(user.id):
        return False
    async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
        state = dict(ACTIVE_ADMIN_CONVERSATIONS.get(user.id) or {})
    if state.get("state") == "awaiting_bot_perf_value":
        raw = (msg.text or "").strip()
        key = str(state.get("key") or "").strip()
        if raw.lower() in {"/cancel", "cancel", "បោះបង់"}:
            async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
                ACTIVE_ADMIN_CONVERSATIONS.pop(user.id, None)
            await safe_send(lambda: msg.reply_text("Cancelled.", reply_markup=get_admin_dashboard_kb()))
            return True
        ok, info = await _apply_bot_performance_setting(key, raw, admin_id=user.id)
        async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS.pop(user.id, None)
        if ok:
            await safe_send(lambda: msg.reply_text(
                f"✅ <b>{html.escape(BOT_SETTING_LABELS.get(key, key))}</b> set to <code>{html.escape(str(raw))}</code>.\n"
                "Open /admin → Settings → Performance Settings to review.",
                parse_mode="HTML",
                reply_markup=get_admin_dashboard_kb(),
            ))
        else:
            await safe_send(lambda: msg.reply_text(
                f"⚠️ Could not update <b>{html.escape(key)}</b>: <code>{html.escape(str(info)[:500])}</code>",
                parse_mode="HTML",
                reply_markup=get_admin_dashboard_kb(),
            ))
        return True

    if state.get("state") != "awaiting_rate_limit":
        return False

    raw = (msg.text or "").strip()
    if raw.lower() in {"/cancel", "cancel", "បោះបង់"}:
        async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS.pop(user.id, None)
        await safe_send(lambda: msg.reply_text(
            _runtime_admin_text(),
            parse_mode="HTML",
            reply_markup=_refresh_runtime_admin_markup(),
        ))
        return True

    http_match = re.fullmatch(r"(?i)(?:http|connections?|pool)\s*[:= ]\s*(\d+)", raw)
    if http_match:
        value = int(http_match.group(1))
        if value < 10 or value > 1000:
            await safe_send(lambda: msg.reply_text("⚠️ HTTP Max Connections ត្រូវនៅចន្លោះ 10 ដល់ 1000។"))
            return True
        await _update_run_state("HTTP_MAX_CONNECTIONS", value)
        async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS.pop(user.id, None)
        _runtime_admin_log_state("update_http_max_connections", user.id, extra=f"new_http_max={value}")
        await safe_send(lambda: msg.reply_text(
            _runtime_admin_text() + f"\n\n✅ HTTP Max Connections ត្រូវបានកែទៅ <b>{value}</b>។",
            parse_mode="HTML",
            reply_markup=_refresh_runtime_admin_markup(),
            disable_web_page_preview=True,
        ))
        return True

    if not raw.isdigit():
        await safe_send(lambda: msg.reply_text(
            "⚠️ សូមផ្ញើលេខគត់សម្រាប់ Rate Limit ឧទាហរណ៍ <code>3</code> ឬ <code>http 120</code> សម្រាប់ HTTP pool។",
            parse_mode="HTML",
        ))
        return True

    value = int(raw)
    if value < 1 or value > 100:
        await safe_send(lambda: msg.reply_text("⚠️ តម្លៃ Rate Limit ត្រូវនៅចន្លោះ 1 ដល់ 100។"))
        return True

    await _update_run_state("USER_RATE_LIMIT_PER_SECOND", value)
    async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
        ACTIVE_ADMIN_CONVERSATIONS.pop(user.id, None)
    _runtime_admin_log_state("update_rate_limit", user.id, extra=f"new_limit={value}")
    await safe_send(lambda: msg.reply_text(
        _runtime_admin_text() + f"\n\n✅ Rate Limit ត្រូវបានកែទៅ <b>{value}</b> req/{_run_state_user_rate_window():g}s។",
        parse_mode="HTML",
        reply_markup=_refresh_runtime_admin_markup(),
        disable_web_page_preview=True,
    ))
    return True


def _runtime_admin_bootstrap_state() -> None:
    with suppress(Exception):
        _restore_tts_provider_runtime_overrides_sync()
    logger.info(
        "Runtime admin state ready mode=%s user_rate_limit=%s http_max_connections=%s tts_provider=%s khmer_tts=%s restored_from_redis=%s",
        _run_state_bot_mode(),
        _run_state_user_rate_limit(),
        _run_state_http_max_connections(),
        globals().get("TTS_PROVIDER", "auto"),
        globals().get("KHMER_TTS_PROVIDER", "hf_space"),
        bool(globals().get("_RUNTIME_STATE_RESTORED_FROM_REDIS")),
    )


def _refresh_arch_runtime_settings() -> None:
    """Re-read high-concurrency env settings after load_dotenv().

    The globals are preserved for backward compatibility, but startup now
    refreshes them from SETTINGS/env so Render secret updates and local .env
    changes are reflected before clients, queues, webhook setup, and rate
    limiters are initialised.
    """
    global CACHE_ASIDE_DEFAULT_TTL_S, HTTP_MAX_CONNECTIONS, HTTP_MAX_KEEPALIVE_CONNECTIONS
    global REDIS_MAX_CONNECTIONS, HTTPX_HIGH_CONCURRENCY_LIMITS
    global BOT_MODE, TELEGRAM_WEBHOOK_URL, TELEGRAM_WEBHOOK_SECRET_TOKEN
    global USER_RATE_LIMIT_PER_SECOND, USER_RATE_LIMIT_WINDOW_S
    global API_RATE_LIMIT_PER_SECOND, API_RATE_LIMIT_WINDOW_S, RATE_LIMIT_REDIS_TTL_S
    global WEB_BROADCAST_QUEUE_MAXSIZE

    CACHE_ASIDE_DEFAULT_TTL_S = _env_int("CACHE_ASIDE_DEFAULT_TTL_S", 3600, minimum=30, maximum=86400)
    HTTP_MAX_CONNECTIONS = _env_int("HTTP_MAX_CONNECTIONS", 100, minimum=10, maximum=1000)
    HTTP_MAX_KEEPALIVE_CONNECTIONS = _env_int("HTTP_MAX_KEEPALIVE_CONNECTIONS", 20, minimum=2, maximum=500)
    REDIS_MAX_CONNECTIONS = _env_int("REDIS_MAX_CONNECTIONS", 100, minimum=2, maximum=1000)
    HTTPX_HIGH_CONCURRENCY_LIMITS = httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
    )
    BOT_MODE = (os.environ.get("BOT_MODE") or getattr(SETTINGS, "BOT_MODE", _perf_default("BOT_MODE", "WEBHOOK")) or _perf_default("BOT_MODE", "WEBHOOK")).strip().upper()
    if BOT_MODE not in {"POLLING", "WEBHOOK"}:
        webhook_logger.warning("Unsupported BOT_MODE=%r; falling back to WEBHOOK.", BOT_MODE)
        BOT_MODE = "WEBHOOK"
    TELEGRAM_WEBHOOK_URL = (
        os.environ.get("TELEGRAM_WEBHOOK_URL")
        or getattr(SETTINGS, "TELEGRAM_WEBHOOK_URL", None)
        or os.environ.get("RENDER_EXTERNAL_URL")
        or getattr(SETTINGS, "RENDER_EXTERNAL_URL", "")
        or ""
    ).strip().rstrip("/")
    TELEGRAM_WEBHOOK_SECRET_TOKEN = (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or getattr(SETTINGS, "TELEGRAM_WEBHOOK_SECRET_TOKEN", None) or "").strip()
    USER_RATE_LIMIT_PER_SECOND = _env_int("USER_RATE_LIMIT_PER_SECOND", 3, minimum=1, maximum=100)
    USER_RATE_LIMIT_WINDOW_S = _env_float("USER_RATE_LIMIT_WINDOW_S", 1.0, minimum=0.25, maximum=60.0)
    API_RATE_LIMIT_PER_SECOND = _env_int("API_RATE_LIMIT_PER_SECOND", 20, minimum=1, maximum=1000)
    API_RATE_LIMIT_WINDOW_S = _env_float("API_RATE_LIMIT_WINDOW_S", 1.0, minimum=0.25, maximum=60.0)
    RATE_LIMIT_REDIS_TTL_S = _env_int("RATE_LIMIT_REDIS_TTL_S", 30, minimum=2, maximum=3600)
    WEB_BROADCAST_QUEUE_MAXSIZE = _env_int("WEB_BROADCAST_QUEUE_MAXSIZE", 200, minimum=10, maximum=10000)
    _sync_run_state_from_globals()
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
OCR_PROVIDER_FAILURE_LIMIT = _env_int("OCR_PROVIDER_FAILURE_LIMIT", 3, minimum=1, maximum=50)
OCR_PROVIDER_COOLDOWN_S    = _env_float("OCR_PROVIDER_COOLDOWN_S", 300.0, minimum=1.0, maximum=86400.0)
MAX_VOICE_BYTES         = 20 * 1024 * 1024
MAX_AUDIO_FILE_BYTES    = 50 * 1024 * 1024
MAX_INPUT_CHARS         = 5_000
# Long text converted as one giant voice can make FFmpeg/libopus exceed Render's
# small-instance CPU budget.  Above this length, send multiple smaller voice
# messages instead of one huge OGG conversion.
TTS_SINGLE_VOICE_MAX_CHARS = _env_int("TTS_SINGLE_VOICE_MAX_CHARS", 1200, minimum=300, maximum=MAX_INPUT_CHARS)
TTS_CHUNK_CHARS         = 900
# Edge TTS can intermittently return NoAudioReceived on Render when a voice
# is busy, text is too long, or websocket audio chunks stop arriving.
# These values keep each Edge request small and allow retry/fallback voices.
EDGE_TTS_CHUNK_CHARS      = _env_int("EDGE_TTS_CHUNK_CHARS", 600, minimum=120, maximum=700)
EDGE_TTS_RETRIES          = _env_int("EDGE_TTS_RETRIES", 3, minimum=1, maximum=8)
EDGE_TTS_RETRY_DELAY_S    = _env_float("EDGE_TTS_RETRY_DELAY_S", 0.8, minimum=0.2, maximum=10.0)
EDGE_TTS_STREAM_TIMEOUT_S = _env_float("EDGE_TTS_STREAM_TIMEOUT_S", 45.0, minimum=15.0, maximum=180.0)
EDGE_TTS_CROSS_LANG_FALLBACK = _env_bool("EDGE_TTS_CROSS_LANG_FALLBACK", False)
# Smooth performance knobs. Keep Edge chunk parallelism conservative because
# too many websocket calls can trigger throttling/403 on small Render services.
EDGE_TTS_PARALLEL_CHUNKS   = _env_int("EDGE_TTS_PARALLEL_CHUNKS", int(_perf_default("EDGE_TTS_PARALLEL_CHUNKS", 1)), minimum=1, maximum=4)
TTS_AUDIO_CACHE_ENABLED    = _env_bool("TTS_AUDIO_CACHE_ENABLED", bool(_perf_default("TTS_AUDIO_CACHE_ENABLED", True)))
TTS_AUDIO_CACHE_TTL_S      = _env_float("TTS_AUDIO_CACHE_TTL_S", float(_perf_default("TTS_AUDIO_CACHE_TTL_S", 1200.0)), minimum=30.0, maximum=86400.0)
TTS_AUDIO_CACHE_MAX_MB     = _env_int("TTS_AUDIO_CACHE_MAX_MB", int(_perf_default("TTS_AUDIO_CACHE_MAX_MB", 64)), minimum=4, maximum=512)
TTS_AUDIO_CACHE_ITEM_MAX_MB = _env_int("TTS_AUDIO_CACHE_ITEM_MAX_MB", int(_perf_default("TTS_AUDIO_CACHE_ITEM_MAX_MB", 8)), minimum=1, maximum=64)
TTS_AUDIO_CACHE_MAX_BYTES  = TTS_AUDIO_CACHE_MAX_MB * 1024 * 1024
TTS_AUDIO_CACHE_ITEM_MAX_BYTES = TTS_AUDIO_CACHE_ITEM_MAX_MB * 1024 * 1024
PAGED_TTS_SEND_DELAY_S     = _env_float("PAGED_TTS_SEND_DELAY_S", float(_perf_default("PAGED_TTS_SEND_DELAY_S", 0.10)), minimum=0.0, maximum=2.0)

# Khmer TTS V2: optional Hugging Face Gradio Space provider.
# Confirmed normal endpoint from `Client.view_api()` for mrrtmob/khmer-tts:
#   client.predict(text, "kiri", 0.6, 0.95, 1.1, 2048, api_name="/lambda")
# Keep Edge TTS as fallback because public Spaces can cold-start, sleep, or rate-limit.
TTS_PROVIDER             = (os.environ.get("TTS_PROVIDER") or "auto").strip().lower()       # auto | edge | hf_space | khmer_hf_space
KHMER_TTS_PROVIDER       = (os.environ.get("KHMER_TTS_PROVIDER") or "hf_space").strip().lower()  # hf_space | edge
HF_TTS_SPACE             = (os.environ.get("HF_TTS_SPACE") or "mrrtmob/khmer-tts").strip()
HF_TTS_API_NAME          = (os.environ.get("HF_TTS_API_NAME") or "/lambda").strip()
HF_TTS_TOKEN             = (os.environ.get("HF_TTS_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
HF_TTS_VOICE             = (os.environ.get("HF_TTS_VOICE") or "kiri").strip()
HF_TTS_TEMP              = _env_float("HF_TTS_TEMP", 0.6, minimum=0.05, maximum=2.0)
HF_TTS_TOP_P             = _env_float("HF_TTS_TOP_P", 0.95, minimum=0.05, maximum=1.0)
HF_TTS_REP_PEN           = _env_float("HF_TTS_REP_PEN", 1.1, minimum=0.5, maximum=3.0)
HF_TTS_MAX_TOK           = _env_int("HF_TTS_MAX_TOK", 2048, minimum=128, maximum=4096)
HF_TTS_MAX_CHARS         = _env_int("HF_TTS_MAX_CHARS", 300, minimum=50, maximum=1200)
HF_TTS_TIMEOUT_S         = _env_float("HF_TTS_TIMEOUT_S", 90.0, minimum=15.0, maximum=240.0)
HF_TTS_RETRIES           = _env_int("HF_TTS_RETRIES", 3, minimum=1, maximum=5)
HF_TTS_RETRY_DELAY_S     = _env_float("HF_TTS_RETRY_DELAY_S", 2.0, minimum=0.2, maximum=20.0)
HF_TTS_EDGE_FALLBACK     = _env_bool("HF_TTS_EDGE_FALLBACK", True)
HF_TTS_FAILURE_LIMIT     = _env_int("HF_TTS_FAILURE_LIMIT", 3, minimum=1, maximum=20)
HF_TTS_COOLDOWN_S        = _env_float("HF_TTS_COOLDOWN_S", 300.0, minimum=30.0, maximum=3600.0)
# Public HF ZeroGPU Spaces can run out of daily quota. When this happens, retrying
# every user request only creates log spam and slow responses, so quota errors use
# a longer cooldown and immediately fall back to Edge TTS when fallback is enabled.
HF_TTS_QUOTA_COOLDOWN_S  = _env_float("HF_TTS_QUOTA_COOLDOWN_S", 1800.0, minimum=300.0, maximum=86400.0)
HF_TTS_NO_AUDIO_COOLDOWN_S = _env_float("HF_TTS_NO_AUDIO_COOLDOWN_S", 600.0, minimum=60.0, maximum=3600.0)
HF_TTS_CLIENT_CACHE      = _env_bool("HF_TTS_CLIENT_CACHE", True)
HF_TTS_SERIALIZE_CALLS   = _env_bool("HF_TTS_SERIALIZE_CALLS", True)
FFMPEG_START_TIMEOUT_S  = _env_float("FFMPEG_START_TIMEOUT_S", 5.0, minimum=1.0, maximum=30.0)
FFMPEG_CONVERT_TIMEOUT_S = _env_float("FFMPEG_CONVERT_TIMEOUT_S", 120.0, minimum=15.0, maximum=600.0)
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
    # Keep Supabase writes bounded. Too many SDK threads can trigger
    # [Errno 11] Resource temporarily unavailable on small Render instances.
    max_workers=_env_int(
        "DB_EXECUTOR_MAX_WORKERS",
        int(_perf_default("DB_EXECUTOR_MAX_WORKERS", 3)),
        minimum=1,
        maximum=16,
    ),
    thread_name_prefix="db_write",
)
_AI_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, MAX_CONCURRENT_AI), thread_name_prefix="ai"
)


def _shutdown_thread_executors() -> None:
    """Stop long-lived thread pools cleanly on deploy restarts/shutdown."""
    for name in ("_WEB_BROADCAST_EXECUTOR", "_DB_EXECUTOR", "_AI_EXECUTOR"):
        executor = globals().get(name)
        if executor is None:
            continue
        with suppress(Exception):
            executor.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_thread_executors)


def _executor_status(name: str, executor: Any) -> dict[str, Any]:
    """Return lightweight ThreadPoolExecutor status without touching private state unsafely."""
    if executor is None:
        return {"configured": False}
    queue_size = None
    try:
        queue_size = executor._work_queue.qsize()  # type: ignore[attr-defined]
    except Exception:
        pass
    threads = None
    try:
        threads = len(executor._threads)  # type: ignore[attr-defined]
    except Exception:
        pass
    max_workers = getattr(executor, "_max_workers", None)
    return {
        "configured": True,
        "name": name,
        "max_workers": max_workers,
        "threads": threads,
        "queued": queue_size,
    }


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
    max_failures=_env_int("SUPABASE_BREAKER_FAILURES", 5, minimum=1, maximum=100),
    reset_after=_env_float("SUPABASE_BREAKER_RESET_S", 30.0, minimum=1.0, maximum=3600.0),
)
redis_breaker = CircuitBreaker(
    "Redis",
    max_failures=_env_int("REDIS_BREAKER_FAILURES", 5, minimum=1, maximum=100),
    reset_after=_env_float("REDIS_BREAKER_RESET_S", 20.0, minimum=1.0, maximum=3600.0),
)

_LOG_ONCE_TTL_S = _env_float("LOG_ONCE_TTL_S", 60.0, minimum=1.0, maximum=3600.0)
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
        "invalid input syntax", "check constraint", "violates check constraint",
        "23514", "duplicate key", "unique constraint",
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
            result = _resolve_maybe_awaitable_sync(factory())
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
            result = await asyncio.wait_for(
                asyncio.to_thread(lambda: _resolve_maybe_awaitable_sync(factory())),
                timeout=timeout,
            )
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


# ── Supabase schema-safe optional column helpers ───────────────────────────
# Older deployments may not have newer optional columns such as
# user_prefs.first_name, user_prefs.tts_model, or text_cache.content.  The
# normal retry wrapper logs those as non-retryable errors.  Dashboard/CRM reads
# should instead omit missing optional columns, cache that decision, and keep the
# UI fast/noiseless.  Required columns still surface real errors.
_SUPABASE_SCHEMA_LOCK = threading.RLock()
_SUPABASE_MISSING_COLUMNS: dict[str, set[str]] = {}


def _supabase_field_column(field: str) -> str:
    raw = str(field or "").strip()
    if not raw:
        return ""
    # Handles simple PostgREST select fields used in this file:
    #   "username", "username:display_name", "count".
    raw = raw.split(":", 1)[0].strip()
    raw = raw.split(".")[-1].strip()
    raw = raw.split("(", 1)[0].strip()
    return raw.strip('"')


def _supabase_split_fields(select_fields: str) -> list[str]:
    return [part.strip() for part in str(select_fields or "").split(",") if part.strip()]


def _supabase_missing_column_name(exc: BaseException, table: str | None = None) -> str | None:
    msg = str(exc or "")
    table_l = str(table or "").strip().lower()
    patterns = (
        r"column\s+(?:(?P<table>[A-Za-z_][\w]*)\.)?(?P<col>[A-Za-z_][\w]*)\s+does\s+not\s+exist",
        r"Could\s+not\s+find\s+the\s+'(?P<col>[^']+)'\s+column\s+of\s+'(?P<table>[^']+)'",
    )
    for pat in patterns:
        m = re.search(pat, msg, re.IGNORECASE)
        if not m:
            continue
        found_table = str(m.groupdict().get("table") or "").strip().lower()
        found_col = str(m.groupdict().get("col") or "").strip().strip('"')
        if not found_col:
            continue
        if table_l and found_table and found_table != table_l:
            continue
        return found_col
    return None


def _supabase_mark_missing_column(table: str, column: str, source: str = "") -> None:
    table = str(table or "").strip()
    column = str(column or "").strip()
    if not table or not column:
        return
    with _SUPABASE_SCHEMA_LOCK:
        bucket = _SUPABASE_MISSING_COLUMNS.setdefault(table, set())
        first_seen = column not in bucket
        bucket.add(column)
    if first_seen:
        _log_once(
            logging.INFO,
            f"supabase_schema_missing:{table}:{column}",
            "Supabase optional column %s.%s is missing; %s will omit it from future schema-safe reads.",
            table,
            column,
            source or "schema-safe query",
        )


def _supabase_filter_select_fields(table: str, select_fields: str) -> str:
    fields = _supabase_split_fields(select_fields)
    if not fields:
        return ""
    with _SUPABASE_SCHEMA_LOCK:
        missing = set(_SUPABASE_MISSING_COLUMNS.get(str(table), set()))
    kept = [field for field in fields if _supabase_field_column(field) not in missing]
    return ", ".join(kept)


def _supabase_has_selected_field(select_fields: str, column: str) -> bool:
    wanted = str(column or "").strip()
    return any(_supabase_field_column(field) == wanted for field in _supabase_split_fields(select_fields))


def _supabase_select_schema_safe_sync(
    table: str,
    field_candidates: tuple[str, ...] | list[str],
    build_query: Callable[[Any, str], Any],
    *,
    name: str,
    default: Any = None,
    max_schema_retries: int = 8,
) -> Any:
    """Execute a Supabase select while silently dropping missing optional columns.

    The helper retries the same candidate after each discovered missing column.
    This prevents noisy 42703 logs and keeps old Supabase schemas compatible
    with newer dashboard/CRM code.
    """
    if not supabase:
        return default

    tried_effective_fields: set[str] = set()
    for raw_fields in field_candidates:
        for _ in range(max(1, int(max_schema_retries))):
            fields = _supabase_filter_select_fields(table, raw_fields)
            if not fields or fields in tried_effective_fields:
                break
            tried_effective_fields.add(fields)
            try:
                result = build_query(supabase.table(table).select(fields), fields)
                if hasattr(result, "execute"):
                    result = result.execute()
                result = _resolve_maybe_awaitable_sync(result)
                with suppress(Exception):
                    supabase_breaker.record_success()
                return result
            except Exception as exc:
                missing_col = _supabase_missing_column_name(exc, table)
                if missing_col:
                    _supabase_mark_missing_column(table, missing_col, name)
                    continue

                with suppress(Exception):
                    supabase_breaker.record_failure(exc)
                level = logging.WARNING if _is_retryable_store_error(exc) else logging.ERROR
                _log_once(
                    level,
                    f"schema_safe_select:{name}:{type(exc).__name__}:{str(exc)[:120]}",
                    "Supabase:%s failed: %s",
                    name,
                    exc,
                )
                return default
    return default


def _supabase_filter_payload_columns(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Drop optional columns already known to be missing for this table.

    Required columns such as user_id are never marked missing by normal app
    paths, so real schema mistakes still surface as failures.
    """
    with _SUPABASE_SCHEMA_LOCK:
        missing = set(_SUPABASE_MISSING_COLUMNS.get(str(table), set()))
    return {key: value for key, value in dict(payload or {}).items() if key not in missing}


def _supabase_upsert_schema_safe_sync(
    table: str,
    payload: dict[str, Any],
    *,
    on_conflict: str | None = None,
    name: str,
    default: Any = None,
    max_schema_retries: int = 8,
) -> Any:
    """Execute an upsert while silently retrying without missing optional columns.

    This prevents expected migration drift such as missing user_prefs.first_name
    from producing noisy PGRST204 non-retryable errors on every user sync.
    """
    if not supabase:
        return default

    tried_keys: set[tuple[str, ...]] = set()
    for _ in range(max(1, int(max_schema_retries))):
        filtered_payload = _supabase_filter_payload_columns(table, payload)
        key_tuple = tuple(sorted(filtered_payload.keys()))
        if not filtered_payload or key_tuple in tried_keys:
            break
        tried_keys.add(key_tuple)
        try:
            builder = (
                supabase.table(table).upsert(filtered_payload, on_conflict=on_conflict)
                if on_conflict
                else supabase.table(table).upsert(filtered_payload)
            )
            result = builder.execute()
            result = _resolve_maybe_awaitable_sync(result)
            with suppress(Exception):
                supabase_breaker.record_success()
            return result
        except Exception as exc:
            missing_col = _supabase_missing_column_name(exc, table)
            if missing_col:
                _supabase_mark_missing_column(table, missing_col, name)
                continue

            with suppress(Exception):
                supabase_breaker.record_failure(exc)
            level = logging.WARNING if _is_retryable_store_error(exc) else logging.ERROR
            _log_once(
                level,
                f"schema_safe_upsert:{name}:{type(exc).__name__}:{str(exc)[:120]}",
                "Supabase:%s failed: %s",
                name,
                exc,
            )
            return default
    return default


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
        return _json_loads_fast(raw)
    except Exception as exc:
        _log_once(logging.WARNING, f"redis_bad_json:{key}", "Invalid Redis JSON for %s: %s", key, exc)
        return default


def _redis_set_json_sync(key: str, value: Any, ttl: int) -> bool:
    raw = _json_dumps_fast(value)
    ok = redis_call_sync(f"set:{key}", lambda: redis_client.set(key, raw, ex=max(1, int(ttl))), default=False)
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
        return _json_loads_fast(raw)
    except Exception as exc:
        _log_once(logging.WARNING, f"redis_bad_json:{key}", "Invalid Redis JSON for %s: %s", key, exc)
        return default


async def _redis_set_json(key: str, value: Any, ttl: int) -> bool:
    raw = _json_dumps_fast(value)
    ok = await redis_call(f"set:{key}", lambda: redis_client.set(key, raw, ex=max(1, int(ttl))), default=False)
    return bool(ok)


async def _redis_delete(key: str) -> None:
    await redis_call(f"delete:{key}", lambda: redis_client.delete(key), default=None)


# ---------------------------------------------------------------------------
# High-concurrency Redis cache-aside + rate limiting helpers
# ---------------------------------------------------------------------------
_RATE_LIMIT_MEMORY: dict[str, deque[float]] = {}
_RATE_LIMIT_MEMORY_LOCK = asyncio.Lock()
_RATE_LIMIT_NOTICE_MEMORY: dict[str, float] = {}


def _cache_payload(value: Any) -> dict[str, Any]:
    return {"v": value, "cached_at": time.time()}


def _cache_payload_value(payload: Any, default: Any = None) -> Any:
    if isinstance(payload, dict) and "v" in payload:
        return payload.get("v", default)
    return payload if payload is not None else default


async def _redis_cache_get_many_json(keys: list[str]) -> dict[str, Any]:
    """Fetch multiple JSON cache keys through one Redis round trip.

    redis-py is synchronous in this project, so the pipeline runs via the
    existing async retry wrapper, which offloads it from the event loop.
    """
    if redis_client is None or not keys:
        return {}

    def _run():
        pipe = redis_client.pipeline(transaction=False)
        for key in keys:
            pipe.get(key)
        return pipe.execute()

    raw_values = await redis_call("pipeline_get_json", _run, default=None, timeout=2.0, attempts=1)
    if not raw_values:
        return {}

    out: dict[str, Any] = {}
    for key, raw in zip(keys, raw_values):
        if not raw:
            continue
        try:
            out[key] = _json_loads_fast(raw)
        except Exception as exc:
            _log_once(logging.WARNING, f"redis_bad_json:{key}", "Invalid Redis JSON for %s: %s", key, exc)
    return out


async def _redis_cache_set_many_json(items: dict[str, Any], ttl: int) -> bool:
    if redis_client is None or not items:
        return False
    ttl_i = max(1, int(ttl or _run_state_cache_ttl()))
    encoded = {str(k): _json_dumps_fast(v) for k, v in items.items()}

    def _run():
        pipe = redis_client.pipeline(transaction=False)
        for key, raw in encoded.items():
            pipe.set(key, raw, ex=ttl_i)
        pipe.execute()
        return True

    return bool(await redis_call("pipeline_set_json", _run, default=False, timeout=3.0, attempts=1))


async def _rate_limit_check(key: str, limit: int, window_s: float) -> tuple[bool, int]:
    """Sliding-window limiter with Redis ZSET backing and in-memory fallback."""
    if not RATE_LIMIT_ENABLED:
        return True, 0
    limit_i = max(1, int(limit or 1))
    window = max(0.25, float(window_s or 1.0))
    now = time.time()
    redis_key = _redis_key("rl", key)

    if redis_client is not None:
        cutoff = now - window
        member = f"{now:.6f}:{secrets.token_hex(4)}"

        def _run():
            # One pipeline round trip; non-transactional for speed. Minor race is
            # acceptable here because this is a protective throttle, not billing.
            pipe = redis_client.pipeline(transaction=False)
            pipe.zremrangebyscore(redis_key, 0, cutoff)
            pipe.zcard(redis_key)
            pipe.zadd(redis_key, {member: now})
            pipe.expire(redis_key, max(_run_state_rate_limit_redis_ttl(), int(window) + 2))
            results = pipe.execute()
            return int(results[1] or 0)

        try:
            count_before = await redis_call("rate_limit", _run, default=None, timeout=1.25, attempts=1)
            if count_before is not None:
                allowed = int(count_before) < limit_i
                return allowed, max(0, limit_i - int(count_before) - (1 if allowed else 0))
        except Exception as exc:
            _log_once(logging.WARNING, "rate_limit_redis_fallback", "Redis rate limiter fallback: %s", exc)

    async with _RATE_LIMIT_MEMORY_LOCK:
        dq = _RATE_LIMIT_MEMORY.setdefault(key, deque())
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        allowed = len(dq) < limit_i
        if allowed:
            dq.append(now)
        # Opportunistic cleanup so offline Redis mode stays bounded.
        if len(_RATE_LIMIT_MEMORY) > 100_000:
            stale_before = now - max(window * 4, 60.0)
            for old_key in list(_RATE_LIMIT_MEMORY.keys())[:5000]:
                old_dq = _RATE_LIMIT_MEMORY.get(old_key)
                if not old_dq or (old_dq and old_dq[-1] < stale_before):
                    _RATE_LIMIT_MEMORY.pop(old_key, None)
        return allowed, max(0, limit_i - len(dq))


def _update_rate_limit_key(update: Any) -> str:
    uid = getattr(getattr(update, "effective_user", None), "id", None)
    chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
    if uid is not None:
        return f"tg:user:{int(uid)}"
    if chat_id is not None:
        return f"tg:chat:{int(chat_id)}"
    return "tg:anonymous"


async def _telegram_rate_limit_guard(update: Any, context: Any) -> None:
    if not isinstance(update, Update):
        return
    key = _update_rate_limit_key(update)
    allowed, _remaining = await _rate_limit_check(key, _run_state_user_rate_limit(), _run_state_user_rate_window())
    if allowed:
        return
    _metric_inc("rate_limited")
    now = time.monotonic()
    last = _RATE_LIMIT_NOTICE_MEMORY.get(key, 0.0)
    if now - last >= USER_RATE_LIMIT_NOTICE_COOLDOWN_S:
        _RATE_LIMIT_NOTICE_MEMORY[key] = now
        if len(_RATE_LIMIT_NOTICE_MEMORY) > 50_000:
            stale_before = now - max(USER_RATE_LIMIT_NOTICE_COOLDOWN_S * 4, 300.0)
            for old_key, old_ts in list(_RATE_LIMIT_NOTICE_MEMORY.items())[:5000]:
                if old_ts < stale_before:
                    _RATE_LIMIT_NOTICE_MEMORY.pop(old_key, None)
        msg = getattr(update, "effective_message", None)
        if msg is not None:
            with suppress(Exception):
                await msg.reply_text(_runtime_admin_notice_text())
    raise ApplicationHandlerStop


_ADMIN_ONLY_COMMANDS = {
    "admin", "stats", "broadcast", "schedule", "schedules", "cancelschedule",
    "runtime", "api", "botsettings", "users", "chat", "endchat",
}
_ADMIN_SECURITY_NOTICE_MEMORY: dict[str, float] = {}


def _telegram_command_name(update: Update) -> str:
    msg = update.effective_message
    text = str(getattr(msg, "text", None) or getattr(msg, "caption", None) or "").strip()
    if not text.startswith("/"):
        return ""
    return text[1:].split(maxsplit=1)[0].split("@", 1)[0].lower()


async def _security_notice_once(update: Update, key: str, text: str, *, alert: bool = False) -> None:
    now = time.monotonic()
    last = _ADMIN_SECURITY_NOTICE_MEMORY.get(key, 0.0)
    if now - last < USER_RATE_LIMIT_NOTICE_COOLDOWN_S:
        return
    _ADMIN_SECURITY_NOTICE_MEMORY[key] = now
    if len(_ADMIN_SECURITY_NOTICE_MEMORY) > 50_000:
        stale_before = now - max(USER_RATE_LIMIT_NOTICE_COOLDOWN_S * 4, 300.0)
        for old_key, old_ts in list(_ADMIN_SECURITY_NOTICE_MEMORY.items())[:5000]:
            if old_ts < stale_before:
                _ADMIN_SECURITY_NOTICE_MEMORY.pop(old_key, None)
    query = update.callback_query
    if query is not None:
        with suppress(Exception):
            await query.answer(text, show_alert=alert)
        return
    msg = update.effective_message
    if msg is not None:
        with suppress(Exception):
            await msg.reply_text(text)


async def _telegram_user_security_guard(update: Any, context: Any) -> None:
    """Cheap global user-safety gate before expensive handlers.

    - Blocks non-admin access to admin commands/callbacks.
    - Stops blocked users before OCR/TTS/AI work begins.
    - Uses cached blocked lookups to avoid DB pressure under normal traffic.
    """
    if not isinstance(update, Update):
        return
    user = update.effective_user
    if user is None:
        return
    user_id = int(user.id)
    if _is_admin(user_id):
        return

    query = update.callback_query
    data = str(getattr(query, "data", "") or "") if query is not None else ""
    admin_callback_prefixes = ("admin_", "api_", "rtadmin_", "user_", "users_", "history_", "sched_", "bc_", "admin_report_")
    if _env_bool("ADMIN_CALLBACK_GUARD_ENABLED", True) and data.startswith(admin_callback_prefixes):
        _metric_inc("admin_denied")
        await _security_notice_once(update, f"admin_cb:{user_id}", "⛔ Admin only.", alert=True)
        raise ApplicationHandlerStop

    cmd = _telegram_command_name(update)
    if cmd in _ADMIN_ONLY_COMMANDS:
        _metric_inc("admin_denied")
        await _security_notice_once(update, f"admin_cmd:{user_id}", "⛔ ពាក្យបញ្ជានេះសម្រាប់ Admin ប៉ុណ្ណោះ។")
        raise ApplicationHandlerStop
    if cmd in {"security", "privacy", "deleteme"}:
        # Privacy/self-service commands stay available even if the user is blocked.
        return

    # Blocked-user guard.  db_user_is_blocked() has memory cache; executor keeps
    # the event loop safe if Supabase fallback is needed.
    try:
        blocked = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_is_blocked(user_id))
    except Exception as exc:
        blocked = False
        _log_once(logging.WARNING, f"blocked_guard_failed:{user_id}", "Blocked-user guard skipped user=%s: %s", user_id, exc)
    if blocked:
        _metric_inc("blocked_hits")
        await _security_notice_once(update, f"blocked:{user_id}", "⛔ អ្នកត្រូវបាន Block មិនអាចប្រើ Bot នេះបានទេ។")
        raise ApplicationHandlerStop


async def _api_rate_limit_allowed(req: FastAPIRequest) -> bool:
    if not RATE_LIMIT_ENABLED:
        return True
    host = req.client.host if req.client else "unknown"
    auth = (req.headers.get("authorization") or req.headers.get("x-api-key") or "").strip()
    if auth.lower().startswith("bearer "):
        auth = auth[7:].strip()
    identity = hashlib.sha256(auth.encode("utf-8")).hexdigest()[:24] if auth else host
    allowed, _remaining = await _rate_limit_check(f"api:{req.url.path}:{identity}", _run_state_api_rate_limit(), _run_state_api_rate_window())
    return allowed


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
ADMIN_REPORT_WAIT_DAY = 9

# Broadcast Template Library (stored in bot_settings as JSON; no new DB table).
BROADCAST_TEMPLATES_SETTING_KEY = "broadcast_templates_json"
BROADCAST_TEMPLATE_LIBRARY_MAX = 20
BROADCAST_TEMPLATE_TITLE_MAX = 48
BROADCAST_TEMPLATE_PREVIEW_MAX = 700
BROADCAST_TEMPLATE_BUTTON_TITLE_MAX = 34
_BROADCAST_TEMPLATES_LOCK = threading.RLock()

# Broadcast Sent Delete Center. Telegram normally allows deleting bot-sent
# messages only for a limited time, so keep delete jobs short-lived and
# in-memory to avoid a DB migration for this safe rollback action.
BROADCAST_SENT_DELETE_TTL_S = 47 * 3600
BROADCAST_SENT_DELETE_MAX_JOBS = 25
BROADCAST_SENT_DELETE_BATCH_SIZE = 20
BROADCAST_SENT_DELETE_DELAY_S = 0.08
_BROADCAST_SENT_DELETE_LOCK = threading.RLock()
_BROADCAST_SENT_DELETE_JOBS: OrderedDict[str, dict] = OrderedDict()

_SCHED_POLL_INTERVAL   = _env_int("SCHED_POLL_INTERVAL", 60, minimum=5, maximum=3600)
_SCHED_SENDING_STALE_SECONDS = _env_int("SCHED_SENDING_STALE_SECONDS", 1800, minimum=60, maximum=86400)
_SCHED_DUE_LIMIT      = _env_int("SCHED_DUE_LIMIT", 5, minimum=1, maximum=100)
_SCHED_SCAN_LIMIT     = _env_int("SCHED_SCAN_LIMIT", 250, minimum=25, maximum=5000)
_SCHED_LOCK_ENABLED   = os.environ.get("SCHED_LOCK_ENABLED", "1") == "1"
_SCHED_LOCK_REQUIRED  = os.environ.get("SCHED_LOCK_REQUIRED", "0") == "1"
_SCHED_ADMIN_PENDING_CACHE_TTL_S = _env_float("SCHED_ADMIN_PENDING_CACHE_TTL_S", 8.0, minimum=1.0, maximum=300.0)
_sched_admin_pending_cache: dict[int, tuple[float, list[dict]]] = {}
_sched_admin_pending_locks: dict[int, threading.Lock] = {}
_sched_admin_pending_cache_lock = threading.Lock()
_SCHED_LOCK_KEY       = os.environ.get("SCHED_LOCK_KEY", "scheduled_broadcast_runner").strip() or "scheduled_broadcast_runner"
_SCHED_LOCK_TTL_S     = _env_int("SCHED_LOCK_TTL_S", max(90, _SCHED_POLL_INTERVAL * 3), minimum=30, maximum=86400)
_BOT_LOCK_OWNER       = os.environ.get("BOT_LOCK_OWNER", "").strip() or (
    f"{os.environ.get('RENDER_SERVICE_NAME', 'bot')}:{os.environ.get('RENDER_INSTANCE_ID', 'local')}:{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(4)}"
)
_DT_FORMATS = [
    "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
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
supabase_async: Any = None
redis_client = None
_gemini    = None
_hf_client = None
_TELEGRAM_APP: Any = None
_TELEGRAM_APP_READY = False
# Public alias used by the FastAPI webhook dispatcher. Keep both names for
# compatibility with older helper code and current Telegram ingestion logic.
telegram_application: Any = None

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
                max_tokens=_env_int("HF_MAX_TOKENS", 1024, minimum=64, maximum=8192),
                temperature=_env_float("HF_TEMPERATURE", 0.7, minimum=0.0, maximum=2.0),
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
    url_model = quote(model_id, safe="")
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
            resp = httpx.post(url, headers=headers, content=image_data, timeout=OCR_TIMEOUT_SECONDS)
            last_body = resp.text[:700]

            if resp.status_code in (503, 504, 429) and attempt < 2:
                wait_s = 1.5 * (attempt + 1)
                try:
                    wait_s = max(wait_s, float(resp.json().get("estimated_time", wait_s)))
                except Exception:
                    pass
                time.sleep(min(wait_s, 10.0))  # sync retry sleep; function runs in AI executor
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
        logger.error("TELEGRAM_BOT_TOKEN is not set. Telegram polling will not start until the env var is configured.")

    if SB_URL and SB_KEY:
        try:
            if SB_KEY.startswith("sb_publishable_") or "publishable" in SB_KEY.lower():
                logger.warning("SUPABASE_KEY looks like a publishable key. Admin tables such as ai_api_keys should use SUPABASE_SERVICE_ROLE_KEY on the server.")
            supabase = _supabase_v2_compat_client(create_client(SB_URL, SB_KEY))
            if SUPABASE_DB_POOLER_URL:
                logger.info("Supabase DB pooler URL configured for direct PostgreSQL workloads (transaction pooler / Supavisor recommended on port 6543).")
            else:
                logger.info("Supabase client initialised. For any direct PostgreSQL queries, configure SUPABASE_DB_POOLER_URL with the Supabase pooler/Supavisor transaction endpoint on port 6543 to avoid socket exhaustion.")
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
                    max_connections=REDIS_MAX_CONNECTIONS,
                )
                redis_client.ping()
                logger.info("Redis cache initialised with max_connections=%s.", REDIS_MAX_CONNECTIONS)
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
    # Khmer
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    # English
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
    # Chinese / Mandarin (Simplified + Traditional text detection routes here)
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
}
TTS_LANGUAGE_LABELS = {
    "km": "Khmer",
    "en": "English",
    "zh": "Chinese",
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
    "🇰🇭 ភាសាខ្មែរ | 🇺🇸 English | 🇨🇳 中文 Chinese\n\n"
    "⚙️ ប្រើ /myprefs ដើម្បីមើលការកំណត់របស់អ្នក\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)
BOT_TAG = "@voicekhaibot"

# ---------------------------------------------------------------------------
# Prefs cache — memory -> Redis -> Supabase -> safe defaults
# ---------------------------------------------------------------------------
_PREFS_TTL      = _env_float("PREFS_CACHE_TTL_S", float(_perf_default("PREFS_CACHE_TTL_S", 600.0)), minimum=30.0, maximum=86400.0)
_PREFS_MAX_SIZE = _env_int("PREFS_CACHE_MAX_SIZE", 10_000, minimum=500, maximum=200_000)
_prefs_cache: OrderedDict[int, tuple[dict, float]] = OrderedDict()
_prefs_cache_lock: asyncio.Lock | None = None
_prefs_cache_thread_lock = threading.RLock()
_PREFS_LOAD_LOCKS_MAX = _env_int("PREFS_LOAD_LOCKS_MAX", 5000, minimum=100, maximum=50000)
_prefs_load_locks: OrderedDict[int, asyncio.Lock] = OrderedDict()
_prefs_load_locks_guard = threading.RLock()


def _get_prefs_load_lock(user_id: int) -> asyncio.Lock:
    """Per-user single-flight lock for cache misses.

    Without this, several fast callbacks from the same user can all miss memory
    cache and hit Redis/Supabase at the same time. This lock makes only one
    coroutine load prefs while the others wait and reuse the refreshed cache.
    """
    user_id = int(user_id)
    with _prefs_load_locks_guard:
        lock = _prefs_load_locks.get(user_id)
        if lock is not None:
            _prefs_load_locks.move_to_end(user_id)
            return lock
        lock = asyncio.Lock()
        _prefs_load_locks[user_id] = lock
        while len(_prefs_load_locks) > _PREFS_LOAD_LOCKS_MAX:
            victim = None
            for uid, item in _prefs_load_locks.items():
                if not item.locked():
                    victim = uid
                    break
            if victim is None:
                break
            _prefs_load_locks.pop(victim, None)
        return lock

TTS_MODEL_OPTIONS = {
    "auto": ("Auto", "Kiri → Edge TTS"),
    "hf_space": ("Kiri", ""),
    "edge": ("Edge TTS", ""),
}
TTS_MODEL_ALIASES = {
    "auto": "auto",
    "default": "auto",
    "server": "auto",
    "hf": "hf_space",
    "hf_space": "hf_space",
    "khmer_hf": "hf_space",
    "khmer_hf_space": "hf_space",
    "khmer-tts": "hf_space",
    "mrrtmob": "hf_space",
    "edge": "edge",
    "edge_tts": "edge",
    "msedge": "edge",
}
DEFAULT_TTS_MODEL = (os.environ.get("DEFAULT_TTS_MODEL") or os.environ.get("USER_DEFAULT_TTS_MODEL") or "auto").strip().lower()


def _normalize_tts_model(value: Any) -> str:
    raw = str(value or DEFAULT_TTS_MODEL or "auto").strip().lower().replace("-", "_")
    return TTS_MODEL_ALIASES.get(raw, "auto")


def _tts_model_label(value: Any) -> str:
    key = _normalize_tts_model(value)
    label, hint = TTS_MODEL_OPTIONS.get(key, TTS_MODEL_OPTIONS["auto"])
    return f"{label} — {hint}"


DEFAULT_USER_PREFS: dict = {"gender": "female", "speed": DEFAULT_SPEED, "tts_model": _normalize_tts_model(DEFAULT_TTS_MODEL)}

# Optional Supabase column state. Older deployments may not have
# user_prefs.tts_model yet, and PostgREST may keep a stale schema cache right
# after ALTER TABLE. Treat that as a soft feature-migration issue, never as a
# Telegram callback failure.
_USER_PREFS_TTS_MODEL_COLUMN_OK: bool | None = None
_USER_PREFS_TTS_MODEL_LAST_CHECK = 0.0
_USER_PREFS_TTS_MODEL_RECHECK_S = _env_float("TTS_MODEL_COLUMN_RECHECK_S", 300.0, minimum=60.0, maximum=86400.0)

USER_PREFS_TTS_MODEL_SQL = """-- Run in Supabase SQL Editor if users cannot save TTS model choice
alter table public.user_prefs
add column if not exists tts_model text default 'auto';

update public.user_prefs
set tts_model = 'auto'
where tts_model is null;

-- Refresh PostgREST/Supabase API schema cache so writes stop returning PGRST204.
notify pgrst, 'reload schema';
"""


def _is_missing_tts_model_column_error(exc: Exception | str) -> bool:
    msg = str(exc).lower()
    return (
        "tts_model" in msg
        and (
            "42703" in msg
            or "pgrst204" in msg
            or "schema cache" in msg
            or "does not exist" in msg
            or "could not find" in msg
        )
    )


def _set_tts_model_column_status(value: bool) -> None:
    global _USER_PREFS_TTS_MODEL_COLUMN_OK, _USER_PREFS_TTS_MODEL_LAST_CHECK
    _USER_PREFS_TTS_MODEL_COLUMN_OK = bool(value)
    _USER_PREFS_TTS_MODEL_LAST_CHECK = time.monotonic()


def _should_try_tts_model_column() -> bool:
    if _USER_PREFS_TTS_MODEL_COLUMN_OK is True:
        return True
    if _USER_PREFS_TTS_MODEL_COLUMN_OK is False:
        # Recheck occasionally so a running bot can recover after SQL + schema
        # cache reload, without a forced redeploy.
        return (time.monotonic() - _USER_PREFS_TTS_MODEL_LAST_CHECK) >= _USER_PREFS_TTS_MODEL_RECHECK_S
    return True


def _user_prefs_select_sync(user_id: int, include_tts_model: bool):
    fields = "gender, speed, tts_model" if include_tts_model else "gender, speed"
    return (
        supabase.table("user_prefs")
        .select(fields)
        .eq("user_id", int(user_id))
        .limit(1)
        .execute()
    )


def _cache_user_tts_model_preference_sync(user_id: int, model: str) -> None:
    """Save the chosen TTS model immediately in local/Redis cache.

    This keeps the UI responsive and prevents missing optional Supabase columns
    from resetting the user's choice during the current session.
    """
    user_id = int(user_id)
    model = _normalize_tts_model(model)
    prefs = _get_cached_prefs_sync(user_id)
    if prefs is None and redis_client is not None:
        cached = _redis_get_json_sync(_prefs_redis_key(user_id), default=None)
        prefs = cached if isinstance(cached, dict) else None
    prefs = _normalize_user_prefs(prefs)
    prefs["tts_model"] = model
    _cache_prefs_sync(user_id, prefs)
    if redis_client is not None:
        _redis_set_json_sync(_prefs_redis_key(user_id), prefs, REDIS_PREFS_TTL_S)


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

    prefs["tts_model"] = _normalize_tts_model(row.get("tts_model", prefs.get("tts_model")))

    return prefs


def _cache_prefs_sync(user_id: int, prefs: dict) -> None:
    with _prefs_cache_thread_lock:
        _prefs_cache.pop(user_id, None)
        _prefs_cache[user_id] = (_normalize_user_prefs(prefs), time.monotonic())
        while len(_prefs_cache) > _PREFS_MAX_SIZE:
            _prefs_cache.popitem(last=False)


def _get_cached_prefs_sync(user_id: int) -> dict | None:
    with _prefs_cache_thread_lock:
        entry = _prefs_cache.get(user_id)
        if entry and time.monotonic() - entry[1] < _PREFS_TTL:
            _prefs_cache.move_to_end(user_id)
            return dict(entry[0])
        if entry:
            _prefs_cache.pop(user_id, None)
        return None


def _invalidate_prefs(user_id: int) -> None:
    with _prefs_cache_thread_lock:
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


async def _load_user_prefs_uncached_async(user_id: int, defaults: dict) -> dict:
    redis_key = _prefs_redis_key(user_id)
    redis_map = await _redis_cache_get_many_json([redis_key])
    redis_prefs = redis_map.get(redis_key)
    redis_prefs = _cache_payload_value(redis_prefs, redis_prefs)
    if isinstance(redis_prefs, dict):
        prefs = _normalize_user_prefs(redis_prefs)
        await _async_cache_prefs(user_id, prefs, write_redis=False)
        return prefs

    if not supabase:
        return dict(defaults)

    try:
        rows = None
        loop = asyncio.get_running_loop()

        if _should_try_tts_model_column():
            try:
                res = await asyncio.wait_for(
                    loop.run_in_executor(
                        _DB_EXECUTOR,
                        functools.partial(_user_prefs_select_sync, user_id, True),
                    ),
                    timeout=8,
                )
                _set_tts_model_column_status(True)
                rows = getattr(res, "data", None) if res else None
            except Exception as exc:
                if _is_missing_tts_model_column_error(exc):
                    _set_tts_model_column_status(False)
                    _log_once(
                        logging.WARNING,
                        "user_prefs_tts_model_missing_read",
                        "user_prefs.tts_model is missing or Supabase schema cache is stale. Using legacy prefs until SQL is applied. SQL:\n%s",
                        USER_PREFS_TTS_MODEL_SQL,
                    )
                else:
                    raise

        if rows is None:
            res = await asyncio.wait_for(
                loop.run_in_executor(
                    _DB_EXECUTOR,
                    functools.partial(_user_prefs_select_sync, user_id, False),
                ),
                timeout=8,
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
        return dict(defaults)


async def get_user_prefs_async(user_id: int) -> dict:
    """
    Smooth single-flight prefs flow:
      1) memory cache
      2) one coroutine per user loads Redis/Supabase on cache miss
      3) Redis cache
      4) Supabase with schema-safe tts_model fallback
      5) safe defaults on any outage
    """
    user_id = int(user_id)
    defaults = dict(DEFAULT_USER_PREFS)

    cached = await _async_get_cached_prefs(user_id)
    if cached is not None:
        return cached

    lock = _get_prefs_load_lock(user_id)
    async with lock:
        # Another callback may have refreshed the cache while we waited.
        cached = await _async_get_cached_prefs(user_id)
        if cached is not None:
            return cached
        return await _load_user_prefs_uncached_async(user_id, defaults)

# ---------------------------------------------------------------------------
# Per-user async locks (LRU-capped)
# ---------------------------------------------------------------------------
_USER_LOCK_MAX = 5_000
_user_locks: OrderedDict[int, asyncio.Lock] = OrderedDict()
_user_locks_guard = threading.RLock()


def _evict_idle_user_locks() -> int:
    """Evict only unlocked user locks from the LRU cache.

    Never remove a locked/in-use asyncio.Lock. Evicting an active lock creates a
    second lock for the same user and silently bypasses the per-user concurrency
    limit. If every cached lock is active, keep the cache temporarily above the
    soft limit and trim it on a later request when idle locks exist.
    """
    evicted = 0
    while len(_user_locks) > _USER_LOCK_MAX:
        victim_id = None
        for uid, lock in _user_locks.items():
            if not lock.locked():
                victim_id = uid
                break
        if victim_id is None:
            logger.warning(
                "User lock cache exceeded soft limit (%s/%s), but all locks are active; "
                "skipping eviction to preserve concurrency limits.",
                len(_user_locks),
                _USER_LOCK_MAX,
            )
            break
        _user_locks.pop(victim_id, None)
        evicted += 1
    return evicted


def _get_user_lock(user_id: int) -> asyncio.Lock:
    with _user_locks_guard:
        lock = _user_locks.get(user_id)
        if lock is not None:
            _user_locks.move_to_end(user_id)
            return lock

        lock = asyncio.Lock()
        _user_locks[user_id] = lock
        _evict_idle_user_locks()
        return lock


# ---------------------------------------------------------------------------
# DB executor helpers
# ---------------------------------------------------------------------------
def _log_future_exception(future):
    with suppress(Exception):
        exc = future.exception()
        if exc:
            logger.error(
                "DB executor task raised: %s",
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )


def _db_executor_queue_size() -> int:
    try:
        return int(_DB_EXECUTOR._work_queue.qsize())  # type: ignore[attr-defined]
    except Exception:
        return 0


def _submit_db(fn, *args, drop_if_backlog: bool = False, backlog_limit: int | None = None, **kwargs):
    """Submit background DB/cache work with optional backpressure.

    Low-priority telemetry writes such as last_active must not build an
    unbounded queue during traffic spikes. Critical preference/history writes
    still use the normal path by leaving drop_if_backlog=False.
    """
    if drop_if_backlog:
        limit = backlog_limit if backlog_limit is not None else _env_int("DB_DROP_BACKLOG_LIMIT", 500, minimum=20, maximum=100_000)
        queued = _db_executor_queue_size()
        if queued >= limit:
            _log_once(
                logging.WARNING,
                f"db_submit_backpressure:{getattr(fn, '__name__', 'task')}",
                "Dropped low-priority DB task because db queue is backed up: queued=%s limit=%s",
                queued,
                limit,
            )
            return None
    future = _DB_EXECUTOR.submit(fn, *args, **kwargs)
    future.add_done_callback(_log_future_exception)
    return future


# ---------------------------------------------------------------------------
# Safe Telegram send with retry
# ---------------------------------------------------------------------------
def _is_message_not_modified_error(exc: Exception | str) -> bool:
    return "message is not modified" in str(exc).lower()


def _is_stale_telegram_message_error(exc: Exception | str) -> bool:
    """Return True for harmless Telegram edit/delete races on stale messages.

    Inline keyboards can outlive the message Telegram is asked to edit/delete:
    users may delete the message, the chat may auto-clean it, or Telegram may no
    longer expose it for editing.  These should not become noisy unhandled
    callback errors, especially for simple menu callbacks like `show_speed`.
    """
    msg = str(exc).lower()
    return any(token in msg for token in (
        "message to edit not found",
        "message to delete not found",
        "message can't be deleted",
        "message can't be edited",
        "message is not found",
    ))


def _is_nonfatal_telegram_edit_error(exc: Exception | str) -> bool:
    return _is_message_not_modified_error(exc) or _is_stale_telegram_message_error(exc)


def _retry_after_seconds(exc: Any, default: float = 3.0) -> float:
    """Return a safe sleep duration for Telegram RetryAfter variants."""
    value = getattr(exc, "retry_after", default)
    if isinstance(value, timedelta):
        value = value.total_seconds()
    try:
        return max(0.5, min(float(value) + 1.0, 60.0))
    except Exception:
        return float(default)


async def safe_send(coro_factory, retries: int = 3, delay: float = 2.0):
    """Run Telegram send/edit/delete calls with safe retry behavior.

    The helper keeps callback/menu flows stable by treating unchanged or stale
    Telegram messages as non-fatal, honoring RetryAfter sleeps, and returning
    None when a user blocked the bot instead of crashing the whole handler.
    """
    if inspect.isawaitable(coro_factory):
        raise TypeError(
            "safe_send requires a zero-arg callable (lambda), not a raw coroutine."
        )
    if not callable(coro_factory):
        raise TypeError("safe_send requires a zero-arg callable.")

    retries = max(1, int(retries or 1))
    delay = max(0.1, float(delay or 0.1))
    last_exc = None
    for attempt in range(retries):
        try:
            result = coro_factory()
            if inspect.isawaitable(result):
                return await result
            return result
        except RetryAfter as e:
            wait = _retry_after_seconds(e)
            logger.warning("Telegram RetryAfter — sleeping %.1fs (attempt %s/%s)", wait, attempt + 1, retries)
            last_exc = e
            if attempt >= retries - 1:
                raise
            await asyncio.sleep(wait)
        except Forbidden as e:
            # User/group blocked the bot or bot lost access.  Broadcast-specific
            # code still marks recipients blocked; generic UI sends should not
            # take down the handler.
            logger.info("Telegram Forbidden ignored by safe_send: %s", str(e)[:300])
            return None
        except BadRequest as e:
            if _is_nonfatal_telegram_edit_error(e):
                logger.debug("Telegram edit/delete skipped because message is unchanged or stale: %s", e)
                return None
            logger.warning("Telegram BadRequest (not retried): %s", e)
            raise
        except (TimedOut, NetworkError) as e:
            last_exc = e
            if attempt < retries - 1:
                wait = min(delay * (attempt + 1), 15.0)
                logger.warning("Telegram network error (attempt %s/%s), retrying in %.1fs: %s", attempt + 1, retries, wait, e)
                await asyncio.sleep(wait)
            else:
                raise
        except TelegramError as e:
            if _is_nonfatal_telegram_edit_error(e):
                logger.debug("Telegram edit/delete skipped because message is unchanged or stale: %s", e)
                return None
            logger.error("Telegram error: %s", e)
            raise
    if last_exc:
        logger.warning("safe_send giving up after %s attempts: %s", retries, last_exc)
    return None


# ---------------------------------------------------------------------------
# Telegram-safe text pagination
# ---------------------------------------------------------------------------
def _paginate_plain(text: str, limit: int = TELE_MSG_LIMIT, header: str = "") -> list[str]:
    """Split plain text into Telegram-sized pages.

    `header` is prepended to every page and included in the limit calculation.
    This keeps command output consistent without forcing callers to subtract
    header sizes manually.
    """
    text = str(text or "").strip()
    header = str(header or "")
    if not text and not header:
        return []

    limit = max(1, int(limit or TELE_MSG_LIMIT))
    body_limit = max(1, limit - len(header))
    pages: list[str] = []

    while text:
        if len(text) <= body_limit:
            pages.append(header + text)
            break

        cut = text.rfind("\n", 0, body_limit + 1)
        if cut <= 0:
            cut = text.rfind(" ", 0, body_limit + 1)
        if cut <= 0:
            cut = body_limit

        page = text[:cut].rstrip()
        if page:
            pages.append(header + page)
        text = text[cut:].lstrip()

    if not pages and header:
        pages.append(header.rstrip())
    return [p for p in pages if p]


def _take_escaped_prefix(text: str, escaped_limit: int) -> tuple[str, str]:
    """Return the largest raw prefix whose html.escape() fits escaped_limit."""
    if escaped_limit <= 0:
        return "", text
    if len(html.escape(text)) <= escaped_limit:
        return text, ""

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(html.escape(text[:mid])) <= escaped_limit:
            lo = mid
        else:
            hi = mid - 1
    if lo <= 0:
        lo = 1
    return text[:lo], text[lo:]


def _paginate_pre_html(text: str, limit: int = TELE_MSG_LIMIT, header: str = "") -> list[str]:
    """Paginate plain text as HTML <pre> blocks without splitting entities/tags."""
    text = str(text or "").strip()
    header = str(header or "")
    wrapper_len = len("<pre></pre>")
    body_limit = max(1, int(limit or TELE_MSG_LIMIT) - len(header) - wrapper_len)
    pages: list[str] = []

    while text:
        raw, rest = _take_escaped_prefix(text, body_limit)
        if rest:
            # Prefer a clean boundary if it still leaves useful content.
            for sep in ("\n", " "):
                cut = raw.rfind(sep)
                if cut > 0 and len(html.escape(raw[:cut].rstrip())) <= body_limit:
                    rest = raw[cut:] + rest
                    raw = raw[:cut].rstrip()
                    break
        raw = raw.rstrip()
        if raw:
            pages.append(f"{header}<pre>{html.escape(raw)}</pre>")
        text = rest.lstrip()

    if not pages and header:
        pages.append(header.rstrip())
    return pages


def _html_safe_cut(text: str, limit: int) -> int:
    """Pick a cut position that does not land inside an HTML tag or entity."""
    if len(text) <= limit:
        return len(text)
    cut = max(1, min(limit, len(text)))

    # Avoid splitting a tag token such as <code> or </b>.
    last_lt = text.rfind("<", 0, cut)
    last_gt = text.rfind(">", 0, cut)
    if last_lt > last_gt:
        cut = max(1, last_lt)

    # Avoid splitting an HTML entity such as &amp; or &#123;.
    last_amp = text.rfind("&", 0, cut)
    last_semi = text.rfind(";", 0, cut)
    if last_amp > last_semi and cut - last_amp <= 12:
        cut = max(1, last_amp)

    # Prefer human-readable boundaries before falling back to a hard safe cut.
    for sep in ("\n\n", "\n", " "):
        boundary = text.rfind(sep, 0, cut)
        if boundary > 0:
            return boundary
    return cut


def _paginate_html(text: str, limit: int = TELE_MSG_LIMIT, header: str = "") -> list[str]:
    """Split already-escaped Telegram HTML without cutting through tags.

    This function is intentionally conservative: it first paginates on paragraph
    and line boundaries, then uses _html_safe_cut only for unusually long blocks.
    `header` is prepended to every page and included in the limit calculation.
    """
    text = str(text or "").strip()
    header = str(header or "")
    if not text and not header:
        return []

    limit = max(1, int(limit or TELE_MSG_LIMIT))
    body_limit = max(1, limit - len(header))
    pages: list[str] = []
    current = ""

    blocks = re.split(r"(\n{2,})", text)
    for block in blocks:
        if not block:
            continue
        candidate = current + block
        if len(candidate) <= body_limit:
            current = candidate
            continue
        if current.strip():
            pages.append(header + current.strip())
            current = ""

        block = block.lstrip()
        while len(block) > body_limit:
            cut = _html_safe_cut(block, body_limit)
            piece = block[:cut].strip()
            if piece:
                pages.append(header + piece)
            block = block[cut:].lstrip()
        current = block

    if current.strip():
        pages.append(header + current.strip())
    if not pages and header:
        pages.append(header.rstrip())
    return [p for p in pages if p]


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------
_TMP_PREFIX = "tgbot_"
_TEMP_DIR_CACHE: str | None = None
_TEMP_DIR_CACHE_LOCK = threading.Lock()


def _get_temp_dir() -> str:
    """Return the bot temp directory, creating it once per process.

    The old implementation called os.makedirs on every temp-file creation.
    That is safe, but it becomes noisy on high-volume TTS/media workloads.
    """
    global _TEMP_DIR_CACHE
    configured = os.environ.get("BOT_TMP_DIR") or tempfile.gettempdir()
    temp_dir = os.path.abspath(configured)
    cached = _TEMP_DIR_CACHE
    if cached == temp_dir:
        return cached
    with _TEMP_DIR_CACHE_LOCK:
        if _TEMP_DIR_CACHE != temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            _TEMP_DIR_CACHE = temp_dir
        return _TEMP_DIR_CACHE


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
_STALE_TEMP_EXTENSIONS = frozenset({
    ".ogg", ".jpg", ".jpeg", ".png", ".webp",
    ".mp3", ".wav", ".mp4", ".m4a", ".flac", ".aac", ".opus", ".webm",
})


def _sweep_stale_temps() -> None:
    """Delete old bot temp files using one directory scan.

    This replaces one glob pass per extension with os.scandir(), reducing
    filesystem work when the temp directory contains many files.
    """
    temp_dir = _get_temp_dir()
    now = time.time()
    try:
        entries = list(os.scandir(temp_dir))
    except OSError as exc:
        logger.warning("Temp sweep could not scan %s: %s", temp_dir, exc)
        return

    for entry in entries:
        try:
            if not entry.is_file():
                continue
            name = entry.name
            if not name.startswith(_TMP_PREFIX):
                continue
            if os.path.splitext(name.lower())[1] not in _STALE_TEMP_EXTENSIONS:
                continue
            if now - entry.stat().st_mtime > _STALE_TEMP_AGE_S:
                os.remove(entry.path)
                logger.info("Swept stale temp file: %s", entry.path)
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
            await asyncio.to_thread(_sweep_stale_temps)


# ---------------------------------------------------------------------------
# Database helpers — user prefs
# ---------------------------------------------------------------------------
_USER_SYNC_TTL = _env_float("USER_SYNC_TTL_S", float(_perf_default("USER_SYNC_TTL_S", 1800.0)), minimum=60.0, maximum=86400.0)
_USER_SYNC_MAX = _env_int("USER_SYNC_MAX", 20_000, minimum=1000, maximum=500_000)
_user_sync_seen: OrderedDict[int, float] = OrderedDict()
_user_sync_seen_lock = threading.RLock()


def sync_user_data(user) -> None:
    if not supabase or not user:
        return
    now = time.monotonic()
    user_id = int(getattr(user, "id", 0) or 0)
    if not user_id:
        return

    with _user_sync_seen_lock:
        last = _user_sync_seen.get(user_id, 0.0)
        if now - last < _USER_SYNC_TTL:
            return
        _user_sync_seen.pop(user_id, None)
        _user_sync_seen[user_id] = now
        while len(_user_sync_seen) > _USER_SYNC_MAX:
            _user_sync_seen.popitem(last=False)

    def _run():
        # Telegram usernames are stored without @ by Telegram. Store first_name
        # only when the database has that optional column; older schemas are
        # retried automatically without first_name/last_active and no noisy
        # PGRST204 non-retryable error is logged.
        username = (getattr(user, "username", None) or "").strip()
        first_name = (getattr(user, "first_name", None) or "").strip()
        payload = {
            "user_id": user_id,
            "username": username or first_name,
            "first_name": first_name,
            "last_active": datetime.now(timezone.utc).isoformat(),
        }
        _supabase_upsert_schema_safe_sync(
            "user_prefs",
            payload,
            on_conflict="user_id",
            name=f"sync_user_data:{user_id}",
            default=None,
            max_schema_retries=4,
        )

    _submit_db(_run, drop_if_backlog=True)


def _paginated_fetch(select_fields: str) -> list[dict]:
    if not supabase:
        return []

    all_rows, page, page_size = [], 0, 1000
    while True:
        res = _supabase_select_schema_safe_sync(
            "user_prefs",
            (select_fields,),
            lambda builder, _fields, p=page: builder.range(p * page_size, (p + 1) * page_size - 1).execute(),
            name=f"paginated_fetch:{select_fields}:page{page}",
            default=None,
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



USER_SEARCH_CACHE_TTL_S = _env_float("USER_SEARCH_CACHE_TTL_S", 20.0, minimum=0.0, maximum=600.0)
_user_search_cache: dict[str, Any] = {"ts": 0.0, "rows": []}
_user_search_cache_lock = threading.Lock()


def get_all_user_ids()         -> list[int]:  return [row["user_id"] for row in _paginated_fetch("user_id")]
def get_all_users_with_names() -> list[dict]: return _paginated_fetch("user_id, username")


def _get_user_search_rows_cached(force: bool = False) -> list[dict]:
    now = time.monotonic()
    with _user_search_cache_lock:
        rows = list(_user_search_cache.get("rows") or [])
        ts = float(_user_search_cache.get("ts") or 0.0)
        if rows and not force and USER_SEARCH_CACHE_TTL_S > 0 and now - ts < USER_SEARCH_CACHE_TTL_S:
            return rows

    if not supabase:
        return []

    fields = "user_id, username, first_name, gender, speed, tts_model, last_active"
    rows = _paginated_fetch(fields)
    if rows:
        with _user_search_cache_lock:
            _user_search_cache["ts"] = now
            _user_search_cache["rows"] = list(rows)
        return rows
    return []



def _dedupe_user_rows(rows: list[dict], limit: int) -> list[dict]:
    seen: set[int] = set()
    out: list[dict] = []
    for row in rows:
        try:
            uid_int = int(row.get("user_id") or 0)
        except Exception:
            uid_int = 0
        if not uid_int or uid_int in seen:
            continue
        seen.add(uid_int)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _normalize_user_search_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lstrip("@")).lower()


def search_users_by_query(query: str, limit: int = 80) -> list[dict]:
    """Search users by Telegram user ID or username with schema-safe selects."""
    q = _normalize_user_search_query(query)
    limit = max(1, min(200, int(limit or 80)))
    if not q:
        return []

    direct_rows: list[dict] = []
    fields_try = (
        "user_id, username, first_name, gender, speed, tts_model, last_active",
        "user_id, username, gender, speed, last_active",
        "user_id, username",
    )

    if supabase:
        if q.isdigit():
            res = _supabase_select_schema_safe_sync(
                "user_prefs",
                fields_try,
                lambda builder, _fields: builder.eq("user_id", int(q)).limit(limit).execute(),
                name=f"search_user_exact:{q}",
                default=None,
            )
            direct_rows.extend(list(getattr(res, "data", None) or []) if res is not None else [])

        username_q = q.lstrip("@")
        if username_q and not username_q.isdigit():
            res = _supabase_select_schema_safe_sync(
                "user_prefs",
                fields_try,
                lambda builder, _fields: builder.ilike("username", f"%{username_q}%").limit(limit).execute(),
                name=f"search_username_ilike:{username_q}",
                default=None,
            )
            direct_rows.extend(list(getattr(res, "data", None) or []) if res is not None else [])

    users = _get_user_search_rows_cached()
    if direct_rows:
        user_map: dict[int, dict] = {}
        for row in direct_rows + users:
            try:
                uid = int(row.get("user_id") or 0)
            except Exception:
                uid = 0
            if uid and uid not in user_map:
                user_map[uid] = row
        users = list(user_map.values())
    elif not users:
        return []

    exact: list[dict] = []
    prefix: list[dict] = []
    contains: list[dict] = []

    for row in users:
        uid = str(row.get("user_id") or "").strip()
        username = str(row.get("username") or row.get("first_name") or "").strip()
        username_l = username.lower().lstrip("@")
        if uid == q or username_l == q:
            exact.append(row)
        elif username_l.startswith(q):
            prefix.append(row)
        elif q in username_l or q in uid:
            contains.append(row)

    return _dedupe_user_rows(exact + prefix + contains, limit)


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


def update_user_tts_model(user_id: int, tts_model: str) -> str:
    """Persist the user's preferred TTS model/provider.

    Returns the normalized model key immediately so callback handlers can
    regenerate voice without waiting on Supabase. The preference is saved to
    memory/Redis first, then persisted in Supabase in the background.

    Important production behavior:
      - Missing optional user_prefs.tts_model never breaks Telegram callbacks.
      - Upsert is used instead of update so first-time users persist correctly.
      - Existing cached gender/speed are included to avoid NOT NULL/default
        problems on older user_prefs schemas.
    """
    user_id = int(user_id)
    model = _normalize_tts_model(tts_model)

    # Fast path: update local + Redis cache before any database work. This makes
    # the button state correct immediately and survives Render process restarts
    # when Redis is configured, even before the optional Supabase column exists.
    _cache_user_tts_model_preference_sync(user_id, model)

    if not supabase:
        return model

    if not _should_try_tts_model_column():
        _log_once(
            logging.WARNING,
            "tts_model_column_missing_write_skipped",
            "Skipped Supabase tts_model write because user_prefs.tts_model is not available yet. SQL:\n%s",
            USER_PREFS_TTS_MODEL_SQL,
        )
        return model

    cached_prefs = _normalize_user_prefs(_get_cached_prefs_sync(user_id))
    payload = {
        "user_id": user_id,
        "gender": cached_prefs.get("gender", "female"),
        "speed": cached_prefs.get("speed", DEFAULT_SPEED),
        "tts_model": model,
    }

    def _run():
        try:
            supabase.table("user_prefs").upsert(payload, on_conflict="user_id").execute()
            _set_tts_model_column_status(True)
        except Exception as exc:
            if _is_missing_tts_model_column_error(exc):
                _set_tts_model_column_status(False)
                _log_once(
                    logging.WARNING,
                    "tts_model_column_missing_write",
                    "Could not save tts_model because the optional column is missing or Supabase schema cache is stale. SQL:\n%s",
                    USER_PREFS_TTS_MODEL_SQL,
                )
                return
            _log_once(
                logging.WARNING,
                f"update_user_tts_model_failed:{type(exc).__name__}:{str(exc)[:120]}",
                "update_user_tts_model failed for user=%s: %s",
                user_id,
                exc,
            )

    _submit_db(_run)
    return model


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
    keep = _env_int("ADMIN_TEXT_CACHE_REDIS_TURNS", 120, minimum=50, maximum=5000)
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

    This remains sync for backwards compatibility. Async handlers should call
    get_text_cache_async() so Redis/Supabase blocking clients never stall the
    Telegram/FastAPI event loop.
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


async def get_text_cache_async(msg_id: int, chat_id: int = 0) -> str | None:
    """Async-safe Redis cache-aside text reader for callbacks/web handlers.

    Hot path stays memory/Redis-first. On Redis miss, Supabase is queried in the
    bounded DB executor, then both memory and Redis are refreshed. If Redis is
    offline, this falls back to the existing Supabase/default flow.
    """
    cached = _get_text_cache_memory_sync(msg_id, chat_id)
    if cached:
        return cached

    redis_key = _text_cache_redis_key(msg_id, chat_id)
    redis_values = await _redis_cache_get_many_json([redis_key])
    redis_payload = redis_values.get(redis_key)
    if isinstance(redis_payload, dict):
        redis_text = str(redis_payload.get("text") or _cache_payload_value(redis_payload, "") or "").strip()
        if redis_text:
            _remember_text_cache_sync(msg_id, chat_id, redis_text)
            return redis_text

    if not supabase:
        return None

    try:
        res = await db_call(
            f"get_text_cache_async:{chat_id}:{msg_id}",
            lambda: supabase.table("text_cache")
                .select("original_text")
                .eq("message_id", int(msg_id))
                .eq("chat_id", int(chat_id))
                .limit(1)
                .execute(),
            default=None,
            attempts=2,
            timeout=6.0,
            critical=False,
        )
        rows = getattr(res, "data", None) if res else None
        if rows:
            text_value = (rows[0].get("original_text") or "").strip()
            if text_value:
                _remember_text_cache_sync(msg_id, chat_id, text_value)
                await _redis_cache_set_many_json({redis_key: {"text": text_value}}, REDIS_TEXT_CACHE_TTL_S)
                return text_value
    except Exception as exc:
        _log_once(
            logging.WARNING,
            f"text_cache_async_fallback:{type(exc).__name__}:{str(exc)[:120]}",
            "get_text_cache_async fallback chat=%s msg=%s: %s",
            chat_id,
            msg_id,
            exc,
        )

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



def ensure_tts_model_column() -> None:
    if not supabase:
        logger.info("ensure_tts_model_column: Supabase not configured, skipping.")
        return

    try:
        supabase.table("user_prefs").select("tts_model").limit(1).execute()
        _set_tts_model_column_status(True)
        logger.info("tts_model column present.")
    except Exception as exc:
        if _is_missing_tts_model_column_error(exc):
            _set_tts_model_column_status(False)
            logger.warning(
                "Optional user_prefs.tts_model is not available yet. User TTS model choice will work in cache, but permanent save needs SQL:\n%s",
                USER_PREFS_TTS_MODEL_SQL,
            )
            return
        logger.warning("Could not verify user_prefs.tts_model: %s", exc)


def startup_self_check() -> None:
    """Log actionable setup problems once at startup without crashing the bot."""
    checks: list[str] = []
    if not TELEGRAM_BOT_TOKEN:
        checks.append("TELEGRAM_BOT_TOKEN is missing")
    if not ADMIN_IDS:
        checks.append("ADMIN_IDS is empty; admin-only commands will reject everyone")
    if not _web_admin_password() and _web_admin_enabled():
        checks.append("ADMIN_WEB_PASSWORD / WEB_ADMIN_PASSWORD is missing; /admin web dashboard is locked")
    if not _web_stable_secret_configured():
        checks.append("Redis WEB_SECRET_KEY is not available; set REDIS_URL so web admin sessions persist across restarts/two servers")
    if supabase and not os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
        checks.append("SUPABASE_SERVICE_ROLE_KEY is not set; admin tables may fail under RLS/publishable key")
    if supabase and not SUPABASE_DB_POOLER_URL:
        checks.append("SUPABASE_DB_POOLER_URL is not set; direct PostgreSQL workloads should use the Supabase pooler/Supavisor transaction URL on port 6543")
    if BOT_MODE == "WEBHOOK" and not TELEGRAM_WEBHOOK_URL:
        checks.append("BOT_MODE=WEBHOOK but TELEGRAM_WEBHOOK_URL is missing; /telegram-webhook will receive updates only after setWebhook is configured")
    if AI_PROVIDER == "hf" and not HF_TOKEN:
        checks.append("HF_TOKEN is missing; /ai-assistant chat will be unavailable")
    if OCR_PROVIDER in ("auto", "hf") and not HF_TOKEN:
        checks.append("HF_TOKEN is missing; Hugging Face OCR will be unavailable")
    if OCR_PROVIDER in ("auto", "gemini") and not GEMINI_API_KEY:
        checks.append("GEMINI_API_KEY is missing; Gemini OCR/audio fallback will be unavailable")
    if (TTS_PROVIDER in {"auto", "edge", "edge_tts"} or KHMER_TTS_PROVIDER in {"auto", "edge", "edge_tts"}) and edge_tts is None:
        checks.append("edge-tts is missing; Edge TTS will be unavailable. Add `edge-tts` to requirements.txt or choose another TTS provider")
    if (TTS_PROVIDER in {"auto", "hf", "hf_space", "khmer_hf_space", "khmer-tts"} or KHMER_TTS_PROVIDER in {"hf", "hf_space", "khmer_hf_space", "khmer-tts"}) and GradioClient is None:
        checks.append("gradio_client is missing; Khmer HF Space TTS will fall back to Edge. Add `gradio_client` to requirements.txt")
    if _should_try_hf_khmer_tts("សាកល្បង", "hf_space") and not HF_TTS_SPACE:
        checks.append("HF_TTS_SPACE is empty; Khmer HF Space TTS is disabled")
    if not REDIS_URL:
        checks.append("REDIS_URL is missing; cache/history fallback will use memory + Supabase only")

    if checks:
        logger.warning("Startup self-check warnings:\n- %s", "\n- ".join(checks))
    else:
        logger.info("Startup self-check passed.")



# ---------------------------------------------------------------------------
# Admin Dashboard V2 — settings, blocks, metrics, user tools
# ---------------------------------------------------------------------------
BOT_FEATURE_SETTING_KEYS: tuple[str, ...] = (
    "maintenance_mode",
    "tts_enabled",
    "ocr_enabled",
    "voice_transcribe_enabled",
    "audio_transcribe_enabled",
    "ai_resolver_enabled",
)

BOT_PERFORMANCE_SETTING_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("TELEGRAM_CONCURRENT_UPDATES", {"kind": "int", "min": 1, "max": 64, "label": "⚙️ Telegram Updates", "help": "Concurrent Telegram update processing. Restart recommended."}),
    ("TELEGRAM_CONNECTION_POOL_SIZE", {"kind": "int", "min": 4, "max": 256, "label": "🌐 Telegram Pool", "help": "Telegram HTTP connection pool. Restart recommended."}),
    ("HTTP_MAX_CONNECTIONS", {"kind": "int", "min": 10, "max": 1000, "label": "🌍 HTTP Max Connections", "help": "Shared HTTPX max connections. Hot reload."}),
    ("HTTP_MAX_KEEPALIVE_CONNECTIONS", {"kind": "int", "min": 2, "max": 500, "label": "♻️ HTTP Keepalive", "help": "Shared HTTPX keepalive pool. Hot reload."}),
    ("DB_EXECUTOR_MAX_WORKERS", {"kind": "int", "min": 1, "max": 16, "label": "🗄️ DB Workers", "help": "Background Supabase worker threads. Hot reload."}),
    ("MAX_CONCURRENT_TTS_USERS", {"kind": "int", "min": 1, "max": 50, "label": "🗣️ Max TTS Users", "help": "Concurrent TTS cap. Hot reload."}),
    ("USER_SYNC_TTL_S", {"kind": "float", "min": 60.0, "max": 86400.0, "label": "👤 User Sync TTL", "help": "Seconds between user activity DB writes. Hot reload."}),
    ("PREFS_CACHE_TTL_S", {"kind": "float", "min": 30.0, "max": 86400.0, "label": "🧠 Prefs Cache TTL", "help": "User preferences cache lifetime. Hot reload."}),
    ("TTS_AUDIO_CACHE_ENABLED", {"kind": "bool", "label": "💾 TTS Audio Cache", "help": "Reuse same generated audio. Hot reload."}),
    ("TTS_AUDIO_CACHE_MAX_MB", {"kind": "int", "min": 4, "max": 512, "label": "💽 TTS Cache MB", "help": "Total audio cache memory. Hot reload."}),
    ("TTS_AUDIO_CACHE_ITEM_MAX_MB", {"kind": "int", "min": 1, "max": 64, "label": "📦 TTS Item MB", "help": "Max size per cached audio. Hot reload."}),
    ("TTS_AUDIO_CACHE_TTL_S", {"kind": "float", "min": 30.0, "max": 86400.0, "label": "⏱️ TTS Cache TTL", "help": "Audio cache expiry seconds. Hot reload."}),
    ("EDGE_TTS_PARALLEL_CHUNKS", {"kind": "int", "min": 1, "max": 4, "label": "⚡ Edge Parallel", "help": "Keep 1 for stability; 2+ can trigger 403."}),
    ("PAGED_TTS_SEND_DELAY_S", {"kind": "float", "min": 0.0, "max": 2.0, "label": "📨 Send Delay", "help": "Delay between paged TTS chunks. Hot reload."}),
    ("WEB_STATUS_POLL_SECONDS", {"kind": "int", "min": 10, "max": 300, "label": "📊 Status Poll", "help": "Admin status refresh seconds. Hot reload."}),
    ("WEB_LIVE_POLL_SECONDS", {"kind": "int", "min": 15, "max": 300, "label": "📡 Live Poll", "help": "Dashboard live refresh seconds. Hot reload."}),
])

BOT_SETTING_DEFAULTS: dict[str, str] = {
    "maintenance_mode": "0",
    "tts_enabled": "1",
    "ocr_enabled": "1",
    "voice_transcribe_enabled": "1",
    "audio_transcribe_enabled": "1",
    "ai_resolver_enabled": "1",
    **{key: str(_perf_default(key, spec.get("default", ""))) for key, spec in BOT_PERFORMANCE_SETTING_SPECS.items()},
}
BOT_SETTING_LABELS: dict[str, str] = {
    "maintenance_mode": "🛠️ Maintenance Mode",
    "tts_enabled": "🗣️ Text → Voice",
    "ocr_enabled": "🔍 Image OCR",
    "voice_transcribe_enabled": "🎙️ Voice Transcribe",
    "audio_transcribe_enabled": "🎵 Audio File Transcribe",
    "ai_resolver_enabled": "🧠 AI Text Resolver",
    **{key: str(spec.get("label", key)) for key, spec in BOT_PERFORMANCE_SETTING_SPECS.items()},
}
BOT_SETTING_DESCRIPTIONS: dict[str, str] = {
    "maintenance_mode": "When ON, normal users cannot use the bot.",
    "tts_enabled": "Allow normal text messages to generate voice.",
    "ocr_enabled": "Allow photo OCR reading.",
    "voice_transcribe_enabled": "Allow Telegram voice transcription.",
    "audio_transcribe_enabled": "Allow uploaded audio-file transcription.",
    "ai_resolver_enabled": "Allow AI to rewrite/resolve text before TTS when enabled by env.",
    **{key: str(spec.get("help", "")) for key, spec in BOT_PERFORMANCE_SETTING_SPECS.items()},
}
_SETTINGS_CACHE_TTL_S = _env_float("BOT_SETTINGS_CACHE_TTL_S", 30.0, minimum=0.0, maximum=3600.0)
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
    ("admin_denied", 0),
    ("replay_dropped", 0),
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


def _setting_raw_from(settings: dict | None, key: str, default: Any = None) -> str:
    source = settings or _bot_settings_cache.get("data") or BOT_SETTING_DEFAULTS
    if default is None:
        default = BOT_SETTING_DEFAULTS.get(key, "")
    return str(source.get(key, default)).strip()


def _coerce_bot_perf_setting(key: str, raw: Any) -> Any:
    spec = BOT_PERFORMANCE_SETTING_SPECS.get(key)
    if not spec:
        raise ValueError(f"Unknown performance setting: {key}")
    kind = str(spec.get("kind", "str"))
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        raise ValueError("Use on/off, true/false, or 1/0.")
    if kind == "int":
        try:
            value = int(float(str(raw).strip()))
        except Exception as exc:
            raise ValueError("Use a whole number.") from exc
        if "min" in spec and value < int(spec["min"]):
            raise ValueError(f"Minimum is {spec['min']}.")
        if "max" in spec and value > int(spec["max"]):
            raise ValueError(f"Maximum is {spec['max']}.")
        return value
    if kind == "float":
        try:
            value = float(str(raw).strip())
        except Exception as exc:
            raise ValueError("Use a number.") from exc
        if "min" in spec and value < float(spec["min"]):
            raise ValueError(f"Minimum is {spec['min']}.")
        if "max" in spec and value > float(spec["max"]):
            raise ValueError(f"Maximum is {spec['max']}.")
        return value
    return str(raw).strip()


def _format_bot_setting_value(key: str, value: Any) -> str:
    spec = BOT_PERFORMANCE_SETTING_SPECS.get(key)
    if spec and spec.get("kind") == "bool":
        return "ON ✅" if _setting_bool_from({key: str(value)}, key, True) else "OFF ⚠️"
    return str(value)


def _bot_settings_redis_key() -> str:
    return _redis_key("settings", "bot", "v1")


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

    redis_key = _bot_settings_redis_key()
    if not force:
        cached_map = await _redis_cache_get_many_json([redis_key])
        payload = cached_map.get(redis_key)
        cached_data = _cache_payload_value(payload, None)
        if isinstance(cached_data, dict):
            data = dict(BOT_SETTING_DEFAULTS)
            data.update({k: str(v) for k, v in cached_data.items() if k in BOT_SETTING_DEFAULTS})
            status = {"db_ok": True, "error": "", "memory": False, "cache": "redis"}
            _bot_settings_cache["data"] = dict(data)
            _bot_settings_cache["status"] = dict(status)
            _bot_settings_cache["ts"] = now
            return data, status

    data, status = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, db_bot_settings_fetch_all)
    _bot_settings_cache["data"] = dict(data)
    _bot_settings_cache["status"] = dict(status)
    _bot_settings_cache["ts"] = now
    if status.get("db_ok") or not supabase:
        await _redis_cache_set_many_json({redis_key: _cache_payload(dict(data))}, CACHE_ASIDE_DEFAULT_TTL_S)
    return data, status


def db_bot_setting_value_set(key: str, value: Any, admin_id: int) -> tuple[bool, str]:
    if key not in BOT_SETTING_DEFAULTS:
        return False, f"Unknown setting: {key}"
    value = str(value).strip()
    _bot_settings_memory[key] = value
    if not supabase:
        _bot_settings_cache["ts"] = 0.0
        _submit_db(lambda: _redis_delete_sync(_bot_settings_redis_key()))
        return True, "saved in memory only"
    try:
        supabase.table("bot_settings").upsert({
            "key": key,
            "value": value,
            "updated_by": int(admin_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="key").execute()
        _bot_settings_cache["ts"] = 0.0
        _redis_delete_sync(_bot_settings_redis_key())
        return True, "saved"
    except Exception as e:
        _bot_settings_cache["ts"] = 0.0
        _redis_delete_sync(_bot_settings_redis_key())
        return False, str(e)


def db_bot_setting_set(key: str, enabled: bool, admin_id: int) -> tuple[bool, str]:
    return db_bot_setting_value_set(key, _bool_to_setting_value(enabled), admin_id)


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
        res = (
            supabase.table("blocked_users")
            .select("user_id", count="exact")
            .limit(1)
            .execute()
        )
        db_count = int(getattr(res, "count", None) or 0)
        return max(db_count, len(_blocked_users_memory))
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
    _cache_user_tts_model_preference_sync(user_id, _normalize_tts_model(DEFAULT_TTS_MODEL))
    if not supabase:
        return True, "memory-only"

    full_payload = {
        "user_id": user_id,
        "gender": "female",
        "speed": DEFAULT_SPEED,
        "tts_model": _normalize_tts_model(DEFAULT_TTS_MODEL),
    }
    legacy_payload = {
        "user_id": user_id,
        "gender": "female",
        "speed": DEFAULT_SPEED,
    }

    try:
        if _should_try_tts_model_column():
            supabase.table("user_prefs").upsert(full_payload, on_conflict="user_id").execute()
            _set_tts_model_column_status(True)
            return True, "reset"
    except Exception as e:
        if _is_missing_tts_model_column_error(e):
            _set_tts_model_column_status(False)
        else:
            return False, str(e)

    try:
        supabase.table("user_prefs").upsert(legacy_payload, on_conflict="user_id").execute()
        return True, "reset (tts_model column pending)"
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
    username = html.escape(_admin_username_display(row))
    gender = html.escape(str(row.get("gender") or "female"))
    speed = html.escape(str(row.get("speed") or DEFAULT_SPEED))
    tts_model = html.escape(_tts_model_label(row.get("tts_model") or DEFAULT_TTS_MODEL))
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
        f"TTS model: <b>{tts_model}</b>\n"
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
ADMIN_DETAIL_HISTORY_TURNS = _env_int("ADMIN_DETAIL_HISTORY_TURNS", 10, minimum=3, maximum=20)
ADMIN_FULL_HISTORY_TURNS   = _env_int("ADMIN_FULL_HISTORY_TURNS", 50, minimum=10, maximum=100)
ADMIN_HISTORY_PAGE_SIZE    = _env_int("ADMIN_HISTORY_PAGE_SIZE", 10, minimum=5, maximum=20)

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
    """Admin full history view from text_cache: Redis first, then schema-safe Supabase."""
    user_id = int(user_id)
    limit = max(1, min(120, int(limit or 30)))

    redis_rows = _redis_get_json_sync(_text_cache_user_history_redis_key(user_id), default=None)
    if isinstance(redis_rows, list) and redis_rows:
        clean = _admin_history_rows_normalized(redis_rows, limit=limit)
        if clean:
            return clean[-limit:]

    if not supabase:
        return []

    def _query(builder: Any, fields: str) -> Any:
        builder = builder.eq("user_id", user_id)
        if _supabase_has_selected_field(fields, "created_at"):
            builder = builder.order("created_at", desc=True)
        return builder.limit(limit).execute()

    res = _supabase_select_schema_safe_sync(
        "text_cache",
        (
            "user_id, username, message_id, chat_id, original_text, content, created_at",
            "user_id, username, message_id, chat_id, original_text, created_at",
            "user_id, message_id, chat_id, original_text, created_at",
            "user_id, message_id, chat_id, content, created_at",
        ),
        _query,
        name=f"admin_text_cache_history_fetch:{user_id}",
        default=None,
    )
    rows = list(reversed(getattr(res, "data", None) or [])) if res else []
    clean = _admin_history_rows_normalized(rows, limit=limit)
    if clean and redis_client is not None:
        _redis_set_json_sync(_text_cache_user_history_redis_key(user_id), clean[-max(limit, 50):], REDIS_HISTORY_TTL_S)
    return clean


def db_recent_history_users(limit_users: int = 80, scan_limit: int = 500) -> list[dict]:
    """Return users ordered by latest text_cache activity using schema-safe selects."""
    limit_users = max(1, min(200, int(limit_users or 80)))
    scan_limit = max(limit_users, min(1000, int(scan_limit or 500)))

    name_map: dict[int, str] = {}
    try:
        for u in get_all_users_with_names():
            uid = int(u.get("user_id") or 0)
            if uid:
                name_map[uid] = str(u.get("username") or u.get("first_name") or "").strip()
    except Exception as exc:
        _log_once(logging.WARNING, "recent_text_cache_names_failed", f"recent text_cache username lookup failed: {exc}")

    rows: list[dict] = []
    if supabase:
        def _query(builder: Any, fields: str) -> Any:
            if _supabase_has_selected_field(fields, "created_at"):
                builder = builder.order("created_at", desc=True)
            return builder.limit(scan_limit).execute()

        res = _supabase_select_schema_safe_sync(
            "text_cache",
            (
                "user_id, username, message_id, chat_id, original_text, content, created_at",
                "user_id, username, message_id, chat_id, original_text, created_at",
                "user_id, message_id, chat_id, original_text, created_at",
                "user_id, message_id, chat_id, content, created_at",
            ),
            _query,
            name="admin_recent_text_cache_scan",
            default=None,
        )
        rows = getattr(res, "data", None) or [] if res else []

    # Fallback only to Redis per-user text_cache history if Supabase is down.
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




async def _init_async_clients() -> None:
    """Initialise optional async clients for new async code paths.

    Existing database helpers still use the proven sync Supabase client through
    the guarded DB executor. New FastAPI-native routes can use supabase_async
    when the installed supabase-py version exposes acreate_client().
    """
    global supabase_async
    if not SB_URL or not SB_KEY or acreate_client is None:
        supabase_async = None
        return
    try:
        maybe_client = acreate_client(SB_URL, SB_KEY)
        supabase_async = await maybe_client if inspect.isawaitable(maybe_client) else maybe_client
        logger.info("Async Supabase client initialised.")
    except Exception as e:
        supabase_async = None
        logger.warning(f"Async Supabase init unavailable; using guarded sync DB executor: {e}")

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


def db_named_lock_acquire(lock_key: str, owner: str, ttl_s: int | float) -> bool:
    """Acquire/renew a named lock in Supabase without depending on scheduler flags."""
    if not supabase:
        return False
    lock_key = str(lock_key or "lock")[:240]
    owner = str(owner or "unknown")[:240]
    now_iso = _sched_iso()
    until_iso = _lock_until_iso(ttl_s)
    update = {"owner": owner, "locked_until": until_iso, "updated_at": now_iso}
    try:
        res = (
            supabase.table("bot_locks")
            .update(update)
            .eq("lock_key", lock_key)
            .eq("owner", owner)
            .execute()
        )
        if getattr(res, "data", None):
            return True

        res = (
            supabase.table("bot_locks")
            .update(update)
            .eq("lock_key", lock_key)
            .lt("locked_until", now_iso)
            .execute()
        )
        if getattr(res, "data", None):
            return True

        try:
            res = supabase.table("bot_locks").insert({"lock_key": lock_key, **update}).execute()
            return bool(getattr(res, "data", None))
        except Exception as insert_exc:
            low = str(insert_exc).lower()
            if "duplicate" in low or "23505" in low or "unique" in low:
                return False
            raise
    except Exception as exc:
        _log_once(
            logging.WARNING,
            f"named_lock_unavailable:{lock_key}:{type(exc).__name__}:{str(exc)[:120]}",
            "Named distributed lock unavailable key=%s error=%s",
            lock_key,
            exc,
        )
        return False


def db_named_lock_release(lock_key: str, owner: str) -> bool:
    if not supabase:
        return False
    try:
        res = (
            supabase.table("bot_locks")
            .delete()
            .eq("lock_key", str(lock_key)[:240])
            .eq("owner", str(owner)[:240])
            .execute()
        )
        return bool(getattr(res, "data", None))
    except Exception as exc:
        _log_once(logging.WARNING, f"named_lock_release_failed:{lock_key}", "Named lock release failed key=%s: %s", lock_key, exc)
        return False


# ---------------------------------------------------------------------------
# Telegram active-owner lease for two Render Web Services
# ---------------------------------------------------------------------------
def _telegram_leader_lock_enabled() -> bool:
    return bool(globals().get("TELEGRAM_ACTIVE_LOCK_ENABLED", False))


def _telegram_leader_require_store() -> bool:
    return bool(globals().get("TELEGRAM_ACTIVE_LOCK_REQUIRED", False))


def _telegram_leader_lock_key() -> str:
    return str(globals().get("TELEGRAM_ACTIVE_LOCK_KEY", "telegram_webhook_owner") or "telegram_webhook_owner")


def _telegram_leader_ttl_s() -> int:
    return max(30, int(globals().get("TELEGRAM_ACTIVE_LOCK_TTL_S", 90) or 90))


def _telegram_leader_renew_interval_s() -> float:
    ttl = _telegram_leader_ttl_s()
    return max(5.0, min(float(globals().get("TELEGRAM_ACTIVE_LOCK_RENEW_S", ttl / 3.0) or ttl / 3.0), ttl - 5.0))


def _telegram_leader_owner_id() -> str:
    global _TELEGRAM_LEADER_OWNER_ID_CACHE
    cached = globals().get("_TELEGRAM_LEADER_OWNER_ID_CACHE")
    if cached:
        return str(cached)
    service = (
        os.environ.get("TELEGRAM_INSTANCE_ID")
        or os.environ.get("RENDER_SERVICE_ID")
        or os.environ.get("RENDER_SERVICE_NAME")
        or os.environ.get("RENDER_EXTERNAL_HOSTNAME")
        or socket.gethostname()
        or "render-service"
    )
    owner = f"{service}:{socket.gethostname()}:{os.getpid()}:{globals().get('_PROCESS_BOOT_ID', secrets.token_hex(6))}"
    owner = re.sub(r"[^A-Za-z0-9_.:@/-]+", "_", owner)[:240]
    _TELEGRAM_LEADER_OWNER_ID_CACHE = owner
    return owner


def _telegram_leader_meta() -> dict[str, Any]:
    return {
        "owner": _telegram_leader_owner_id(),
        "mode": _run_state_bot_mode() if "_run_state_bot_mode" in globals() else str(globals().get("BOT_MODE", "WEBHOOK")),
        "url": _runtime_webhook_base_url() if "_runtime_webhook_base_url" in globals() else str(globals().get("TELEGRAM_WEBHOOK_URL", "")),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "boot_id": str(globals().get("_PROCESS_BOOT_ID", "")),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ttl_s": _telegram_leader_ttl_s(),
    }


def _telegram_leader_redis_lock_key() -> str:
    return _redis_key("locks", _telegram_leader_lock_key())


def _telegram_leader_redis_meta_key() -> str:
    return _redis_key("locks", _telegram_leader_lock_key(), "meta")


def _telegram_leader_store_available() -> bool:
    return bool(globals().get("redis_client") is not None or globals().get("supabase") is not None)


def _telegram_leader_acquire_sync() -> bool:
    """Acquire/renew active Telegram ownership. Redis is preferred; Supabase is fallback."""
    global _TELEGRAM_LEADER_LAST_ERROR
    if not _telegram_leader_lock_enabled():
        return True

    owner = _telegram_leader_owner_id()
    ttl = _telegram_leader_ttl_s()

    if redis_client is not None:
        key = _telegram_leader_redis_lock_key()
        meta_key = _telegram_leader_redis_meta_key()
        try:
            ok = redis_call_sync(
                "telegram_leader_set_nx",
                lambda: redis_client.set(key, owner, ex=ttl, nx=True),
                default=False,
                attempts=2,
                critical=False,
            )
            current = None
            if ok:
                current = owner
            else:
                current = redis_call_sync(
                    "telegram_leader_get",
                    lambda: redis_client.get(key),
                    default=None,
                    attempts=2,
                    critical=False,
                )
                if isinstance(current, bytes):
                    current = current.decode("utf-8", errors="ignore")

            if hmac.compare_digest(str(current or ""), owner):
                # Renew only if the lock still belongs to this process.
                redis_call_sync(
                    "telegram_leader_expire",
                    lambda: redis_client.expire(key, ttl),
                    default=False,
                    attempts=2,
                    critical=False,
                )
                redis_call_sync(
                    "telegram_leader_meta",
                    lambda: redis_client.set(meta_key, _json_dumps_fast(_telegram_leader_meta()).decode("utf-8"), ex=ttl),
                    default=False,
                    attempts=1,
                    critical=False,
                )
                _TELEGRAM_LEADER_LAST_ERROR = ""
                return True
            _TELEGRAM_LEADER_LAST_ERROR = f"owned_by:{str(current or '')[:120]}"
            return False
        except Exception as exc:
            _TELEGRAM_LEADER_LAST_ERROR = f"redis:{type(exc).__name__}:{str(exc)[:160]}"
            _log_once(logging.WARNING, "telegram_leader_redis_failed", "Telegram leader Redis lock failed: %s", exc)

    if supabase is not None:
        ok = db_named_lock_acquire(_telegram_leader_lock_key(), owner, ttl)
        _TELEGRAM_LEADER_LAST_ERROR = "" if ok else "supabase:not_owner"
        return bool(ok)

    _TELEGRAM_LEADER_LAST_ERROR = "no_redis_or_supabase"
    if _telegram_leader_require_store():
        _log_once(
            logging.ERROR,
            "telegram_leader_no_store_required",
            "Two-server Telegram mode requires REDIS_URL or Supabase bot_locks table. This instance stays standby.",
        )
        return False
    _log_once(
        logging.WARNING,
        "telegram_leader_no_store_single",
        "Telegram leader lock store unavailable; continuing as single active instance because TELEGRAM_ACTIVE_LOCK_REQUIRED=0.",
    )
    return True


def _telegram_leader_release_sync() -> bool:
    if not _telegram_leader_lock_enabled():
        return True
    owner = _telegram_leader_owner_id()
    released = False
    if redis_client is not None:
        key = _telegram_leader_redis_lock_key()
        meta_key = _telegram_leader_redis_meta_key()
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            redis.call('del', KEYS[1])
            redis.call('del', KEYS[2])
            return 1
        end
        return 0
        """
        try:
            released = bool(redis_call_sync(
                "telegram_leader_release",
                lambda: redis_client.eval(script, 2, key, meta_key, owner),
                default=False,
                attempts=1,
                critical=False,
            ))
        except Exception:
            current = redis_call_sync("telegram_leader_release_get", lambda: redis_client.get(key), default=None, attempts=1, critical=False)
            if isinstance(current, bytes):
                current = current.decode("utf-8", errors="ignore")
            if hmac.compare_digest(str(current or ""), owner):
                redis_call_sync("telegram_leader_release_del", lambda: redis_client.delete(key, meta_key), default=None, attempts=1, critical=False)
                released = True
    elif supabase is not None:
        released = db_named_lock_release(_telegram_leader_lock_key(), owner)
    return bool(released)


def _telegram_leader_snapshot() -> dict[str, Any]:
    owned = bool(globals().get("_TELEGRAM_LEADER_OWNED", False))
    snapshot = {
        "enabled": _telegram_leader_lock_enabled(),
        "required": _telegram_leader_require_store(),
        "owned_by_this_instance": owned,
        "owner": None,
        "this_owner": _telegram_leader_owner_id(),
        "backend": "none",
        "error": str(globals().get("_TELEGRAM_LEADER_LAST_ERROR", "") or ""),
        "url": _runtime_webhook_base_url() if "_runtime_webhook_base_url" in globals() else "",
        "last_change_at": float(globals().get("_TELEGRAM_LEADER_LAST_CHANGE_AT", 0.0) or 0.0),
        "last_renew_at": float(globals().get("_TELEGRAM_LEADER_LAST_RENEW_AT", 0.0) or 0.0),
    }
    if redis_client is not None:
        snapshot["backend"] = "redis"
        with suppress(Exception):
            owner = redis_call_sync("telegram_leader_snapshot_get", lambda: redis_client.get(_telegram_leader_redis_lock_key()), default=None, attempts=1, critical=False)
            if isinstance(owner, bytes):
                owner = owner.decode("utf-8", errors="ignore")
            snapshot["owner"] = owner
            meta = _redis_get_json_sync(_telegram_leader_redis_meta_key(), default=None)
            if isinstance(meta, dict):
                snapshot["meta"] = meta
        return snapshot
    if supabase is not None:
        snapshot["backend"] = "supabase"
        row = db_lock_read(_telegram_leader_lock_key())
        if row:
            snapshot["owner"] = row.get("owner")
            snapshot["meta"] = row
        return snapshot
    return snapshot


def _telegram_should_process_webhook_update() -> bool:
    if not _telegram_leader_lock_enabled():
        return True
    return bool(globals().get("_TELEGRAM_LEADER_OWNED", False))


def _telegram_should_run_telegram_workers() -> bool:
    if not _telegram_leader_lock_enabled():
        return True
    return bool(globals().get("_TELEGRAM_LEADER_OWNED", False))


def _telegram_webhook_signature() -> tuple[Any, ...]:
    return (
        _run_state_bot_mode() if "_run_state_bot_mode" in globals() else str(globals().get("BOT_MODE", "WEBHOOK")),
        _runtime_webhook_base_url() if "_runtime_webhook_base_url" in globals() else str(globals().get("TELEGRAM_WEBHOOK_URL", "")),
        _runtime_webhook_secret_token() if "_runtime_webhook_secret_token" in globals() else str(globals().get("TELEGRAM_WEBHOOK_SECRET_TOKEN", "")),
        tuple(_telegram_allowed_updates() if "_telegram_allowed_updates" in globals() else ["message", "callback_query"]),
    )


async def _telegram_leader_refresh_once(app_obj: Any | None = None) -> bool:
    """Renew/acquire Telegram active ownership and configure webhook once per owner."""
    global _TELEGRAM_LEADER_OWNED, _TELEGRAM_LEADER_LAST_CHANGE_AT, _TELEGRAM_LEADER_LAST_RENEW_AT
    global _TELEGRAM_LEADER_LAST_WEBHOOK_SIGNATURE, _TELEGRAM_LEADER_LAST_WEBHOOK_SET_AT

    if not _telegram_leader_lock_enabled():
        now = time.monotonic()
        if not _TELEGRAM_LEADER_OWNED:
            _TELEGRAM_LEADER_LAST_CHANGE_AT = now
        _TELEGRAM_LEADER_OWNED = True
        _TELEGRAM_LEADER_LAST_RENEW_AT = now

        # Single-server webhook mode has no Redis leader-election branch, but
        # it still must register/refresh Telegram's webhook.  Without this, a
        # normal one-instance WEBHOOK deployment can boot successfully while
        # Telegram still points at an old URL/secret or has no webhook set.
        if _run_state_bot_mode() == "WEBHOOK":
            sig = _telegram_webhook_signature()
            reconfigure_after = max(60, int(globals().get("TELEGRAM_WEBHOOK_RECONFIGURE_INTERVAL_S", 600) or 600))
            needs_reconfigure = (
                _TELEGRAM_LEADER_LAST_WEBHOOK_SIGNATURE != sig
                or (now - float(_TELEGRAM_LEADER_LAST_WEBHOOK_SET_AT or 0.0)) >= reconfigure_after
            )
            if needs_reconfigure:
                if app_obj is not None:
                    with suppress(Exception):
                        await _telegram_stop_polling_runtime(app_obj)
                await _configure_telegram_webhook_via_http()
                _TELEGRAM_LEADER_LAST_WEBHOOK_SIGNATURE = sig
                _TELEGRAM_LEADER_LAST_WEBHOOK_SET_AT = now
                webhook_logger.info("Telegram webhook confirmed with leader lock disabled.")

        return True

    loop = asyncio.get_running_loop()
    acquired = await loop.run_in_executor(None, _telegram_leader_acquire_sync)
    now = time.monotonic()
    if not acquired:
        if _TELEGRAM_LEADER_OWNED:
            webhook_logger.warning("This Render instance lost Telegram active ownership; switching to standby.")
            if app_obj is not None:
                with suppress(Exception):
                    await _telegram_stop_polling_runtime(app_obj)
        _TELEGRAM_LEADER_OWNED = False
        _TELEGRAM_LEADER_LAST_CHANGE_AT = now
        _TELEGRAM_LEADER_LAST_WEBHOOK_SIGNATURE = None
        return False

    if not _TELEGRAM_LEADER_OWNED:
        webhook_logger.warning(
            "This Render instance is now Telegram ACTIVE owner key=%s owner=%s url=%s",
            _telegram_leader_lock_key(),
            _telegram_leader_owner_id(),
            _runtime_webhook_base_url(),
        )
        _TELEGRAM_LEADER_LAST_CHANGE_AT = now
    _TELEGRAM_LEADER_OWNED = True
    _TELEGRAM_LEADER_LAST_RENEW_AT = now

    if _run_state_bot_mode() == "WEBHOOK":
        sig = _telegram_webhook_signature()
        reconfigure_after = max(60, int(globals().get("TELEGRAM_WEBHOOK_RECONFIGURE_INTERVAL_S", 600) or 600))
        needs_reconfigure = (
            _TELEGRAM_LEADER_LAST_WEBHOOK_SIGNATURE != sig
            or (now - float(_TELEGRAM_LEADER_LAST_WEBHOOK_SET_AT or 0.0)) >= reconfigure_after
        )
        if needs_reconfigure:
            if app_obj is not None:
                with suppress(Exception):
                    await _telegram_stop_polling_runtime(app_obj)
            await _configure_telegram_webhook_via_http()
            _TELEGRAM_LEADER_LAST_WEBHOOK_SIGNATURE = sig
            _TELEGRAM_LEADER_LAST_WEBHOOK_SET_AT = now
            webhook_logger.info("Telegram webhook confirmed by active owner.")
    return True


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


def _sched_admin_pending_cache_clear(admin_id: int | None = None) -> None:
    with _sched_admin_pending_cache_lock:
        if admin_id is None:
            _sched_admin_pending_cache.clear()
        else:
            _sched_admin_pending_cache.pop(int(admin_id), None)


def _sched_admin_pending_lock(admin_id: int) -> threading.Lock:
    with _sched_admin_pending_cache_lock:
        lock = _sched_admin_pending_locks.get(int(admin_id))
        if lock is None:
            lock = threading.Lock()
            _sched_admin_pending_locks[int(admin_id)] = lock
        return lock


def db_sched_insert(payload: dict, admin_id: int, broadcast_at: datetime) -> dict:
    """Create a schedule preview row.

    Important fix: new rows remain DB-compatible with status="pending", but
    are tagged with SCHED_DRAFT_MARKER in error_msg. The scheduler only sends
    pending rows where error_msg IS NULL, so the broadcast cannot fire until
    the admin taps ✅ Confirm Schedule.
    """
    if not supabase:
        raise RuntimeError("Supabase not configured.")

    parse_mode = _broadcast_normalize_parse_mode(payload.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO)
    caption = payload.get("caption")
    plain_text = payload.get("text")
    if payload.get("photo_file_id"):
        caption = _broadcast_apply_format_directive(caption, parse_mode) if caption else caption
    elif plain_text:
        plain_text = _broadcast_apply_format_directive(plain_text, parse_mode)

    row = {
        "admin_id":      int(admin_id),
        "photo_file_id": payload.get("photo_file_id"),
        "caption":       caption,
        "plain_text":    plain_text,
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
    _sched_admin_pending_cache_clear(admin_id)
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
    _sched_admin_pending_cache_clear(admin_id)
    return True, "confirmed", saved

def db_sched_set_status(row_id: int, status: str, *, critical: bool = False, **extra) -> bool:
    """Update one scheduled broadcast status and report whether DB saved it.

    `critical=True` is used by the scheduler finalizer. Without it, a failed
    final update could be swallowed by db_call_sync and the row would remain
    stuck in `sending` even though Telegram sending already finished.
    """
    if not supabase:
        return False

    update = {"status": str(status), **extra}
    try:
        res = db_call_sync(
            f"sched_set_status:{row_id}:{status}",
            lambda: (
                supabase.table("scheduled_broadcasts")
                .update(update)
                .eq("id", int(row_id))
                .execute()
            ),
            default=None,
            attempts=3 if critical else 2,
            critical=critical,
        )
        ok = bool(getattr(res, "data", None))
        if ok:
            _sched_admin_pending_cache_clear(None)
            return True
        if critical:
            raise RuntimeError(f"scheduled_broadcasts status update returned no row for id={row_id} status={status}")
        logger.warning("db_sched_set_status #%s -> %s returned no saved row", row_id, status)
        return False
    except Exception as e:
        logger.error(f"db_sched_set_status #{row_id} -> {status}: {e}")
        if critical:
            raise
        return False


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
    admin_id = int(admin_id)
    now = time.monotonic()
    with _sched_admin_pending_cache_lock:
        cached = _sched_admin_pending_cache.get(admin_id)
        if cached and now - cached[0] < _SCHED_ADMIN_PENDING_CACHE_TTL_S:
            return list(cached[1])

    lock = _sched_admin_pending_lock(admin_id)
    if not lock.acquire(blocking=False):
        # A concurrent callback/page is already fetching this list. Reuse stale
        # data if available instead of opening a second Supabase request.
        if cached:
            return list(cached[1])
        return []

    try:
        res = db_call_sync(
            f"sched_fetch_admin_pending:{admin_id}",
            lambda: (
                supabase.table("scheduled_broadcasts")
                .select("id, broadcast_at, plain_text, caption, photo_file_id, status, error_msg")
                .eq("admin_id", admin_id)
                .eq("status", SCHED_STATUS_PENDING)
                .order("broadcast_at")
                .execute()
            ),
            default=None,
            attempts=1,
        )
        rows = list(getattr(res, "data", None) or [])
        rows = [r for r in rows if _sched_is_draft(r) or _sched_is_confirmed_pending(r)]
        with _sched_admin_pending_cache_lock:
            _sched_admin_pending_cache[admin_id] = (time.monotonic(), list(rows))
        return rows
    finally:
        with suppress(Exception):
            lock.release()

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
    """Parse admin-entered schedule time as Phnom Penh local time, return UTC."""
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"\s+", " ", raw)
    # Accept common local inputs, including AM/PM. HTML datetime-local sends
    # 24-hour values, while Telegram admins may type AM/PM manually.
    local_formats = list(_DT_FORMATS) + [
        "%Y-%m-%d %I:%M %p", "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%dT%I:%M %p", "%Y-%m-%dT%I:%M:%S %p",
        "%d/%m/%Y %I:%M %p", "%d-%m-%Y %I:%M %p",
    ]
    for fmt in local_formats:
        try:
            local_dt = datetime.strptime(raw, fmt).replace(tzinfo=APP_TIMEZONE)
            return _local_to_utc(local_dt)
        except ValueError:
            continue

    # Also accept ISO strings with explicit timezone offsets.
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=APP_TIMEZONE)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _fmt_dt(dt: datetime) -> str:
    return _fmt_local_dt(dt)


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
                 InlineKeyboardButton("❌ បិទ", callback_data="sched_close")])
    return InlineKeyboardMarkup(rows)


def get_admin_dashboard_kb() -> InlineKeyboardMarkup:
    """Telegram Admin System Center V7 keyboard.

    This menu now exposes the CRM and Optimize panels directly in Telegram,
    not only in the web dashboard.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Dashboard",   callback_data="admin_home"),
         InlineKeyboardButton("🩺 Health",      callback_data="admin_health")],
        [InlineKeyboardButton("👥 Users",       callback_data="admin_users"),
         InlineKeyboardButton("⭐ CRM",         callback_data="admin_crm")],
        [InlineKeyboardButton("⚡ Optimize",    callback_data="admin_optimize"),
         InlineKeyboardButton("🕘 Recent History", callback_data="admin_history")],
        [InlineKeyboardButton("⚙️ Settings",    callback_data="admin_settings"),
         InlineKeyboardButton("📊 Stats",       callback_data="admin_stats")],
        [InlineKeyboardButton("📢 Broadcast សុវត្ថិភាព", callback_data="admin_broadcast"),
         InlineKeyboardButton("🗓 Schedule Calendar", callback_data="admin_calendar")],
        [InlineKeyboardButton("⏰ Schedules",   callback_data="admin_schedules"),
         InlineKeyboardButton("🎛 TTS Provider", callback_data="admin_tts")],
        [InlineKeyboardButton("🚨 Error Center", callback_data="admin_errors"),
         InlineKeyboardButton("🛠️ Runtime",    callback_data="admin_runtime")],
        [InlineKeyboardButton("🔑 API Keys",    callback_data="admin_api"),
         InlineKeyboardButton("📄 Report PDF",  callback_data="admin_report")],
        [InlineKeyboardButton("🔐 WEB_KEY",     callback_data="admin_web_key")],
        [InlineKeyboardButton("🔄 Refresh",     callback_data="admin_home")],
        [InlineKeyboardButton("❌ បិទ",       callback_data="admin_close")],
    ])


def get_admin_report_day_kb() -> InlineKeyboardMarkup:
    """Date/range picker shown when admin taps PDF Report in Telegram."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📅 Today", callback_data="admin_report_range:today"),
         InlineKeyboardButton("◀️ Yesterday", callback_data="admin_report_range:yesterday")],
        [InlineKeyboardButton("7 Days", callback_data="admin_report_range:last7"),
         InlineKeyboardButton("30 Days", callback_data="admin_report_range:last30")],
        [InlineKeyboardButton("✍️ Type Date", callback_data="admin_report_range:custom")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ Cancel", callback_data="admin_cancel_state")],
    ])


def generate_web_secret_key() -> str:
    """Return a strong WEB_SECRET_KEY value for Redis-backed web sessions."""
    return _generate_web_secret_value()


def _web_secret_key_fingerprint(secret_value: str) -> str:
    return _web_secret_fingerprint(secret_value)


def _configured_web_secret_source() -> tuple[str, str]:
    """Return (source, value) for the configured web session secret.

    Redis is primary.  Env values are kept only as compatibility fallback.
    This never exposes the secret to logs or public API responses.
    """
    redis_value, redis_source = _web_secret_get_from_redis()
    if redis_value:
        return redis_source, redis_value
    web_key = os.environ.get("WEB_SECRET_KEY", "").strip()
    if web_key:
        return "env:WEB_SECRET_KEY:fallback", web_key
    flask_key = os.environ.get("FLASK_SECRET_KEY", "").strip()
    if flask_key:
        return "env:FLASK_SECRET_KEY:fallback", flask_key
    settings_key = str(getattr(SETTINGS, "WEB_SECRET_KEY", "") or "").strip()
    if settings_key:
        return "settings:WEB_SECRET_KEY:fallback", settings_key
    settings_flask_key = str(getattr(SETTINGS, "FLASK_SECRET_KEY", "") or "").strip()
    if settings_flask_key:
        return "settings:FLASK_SECRET_KEY:fallback", settings_flask_key
    if _WEB_SESSION_SECRET_CACHE:
        return _web_session_secret_source(), _WEB_SESSION_SECRET_CACHE
    return "missing", ""


def _web_key_private_message(new_key: str) -> str:
    stored_source, stored_value = _configured_web_secret_source()
    stored_status = "stored in Redis" if stored_value else "missing"
    active_source = _web_session_secret_source()
    active_fp = _web_session_secret_fingerprint()
    stored_fp = _web_secret_key_fingerprint(stored_value or new_key)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (
        "🔐 <b>New WEB_SECRET_KEY stored in Redis</b>\n\n"
        "You do <b>not</b> need to put <code>WEB_SECRET_KEY</code> in Render/Railway env anymore. "
        "Keep <code>REDIS_URL</code> configured so every backend server reads the same session secret.\n\n"
        f"Redis key: <code>{html.escape(_web_secret_redis_key())}</code>\n"
        f"Stored source: <code>{html.escape(stored_source)}</code>\n"
        f"Stored status: <code>{html.escape(stored_status)}</code>\n"
        f"Stored fingerprint: <code>{html.escape(stored_fp)}</code>\n"
        f"Active running source: <code>{html.escape(active_source)}</code>\n"
        f"Active running fingerprint: <code>{html.escape(active_fp)}</code>\n\n"
        "Important:\n"
        "• Restart/deploy <b>all</b> backend instances after rotation so SessionMiddleware uses the Redis key.\n"
        "• Existing admin browser sessions will be logged out after rotation.\n"
        "• The raw secret is not shown because it is already persisted in Redis.\n"
        "• Recommended cookie defaults are already in code: <code>WEB_COOKIE_SAMESITE=none</code>, <code>WEB_COOKIE_SECURE=true</code>.\n\n"
        f"Generated UTC: <code>{html.escape(generated_at)}</code>"
    )


async def _admin_generate_web_key(query, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate and store the web session secret in Redis from Telegram /admin.

    The raw secret is not displayed.  The admin receives only fingerprints and
    restart guidance, because Redis is now the source of truth.
    """
    new_key = generate_web_secret_key()
    fingerprint = _web_secret_key_fingerprint(new_key)
    ok, store_message = await asyncio.to_thread(_web_secret_set_in_redis_sync, new_key, overwrite=True)
    if not ok:
        await safe_send(lambda: query.message.edit_text(
            "⚠️ <b>WEB_KEY Redis save failed.</b>\n\n"
            f"Reason: <code>{html.escape(str(store_message)[:500])}</code>\n\n"
            "Set <code>REDIS_URL</code> first, restart the backend, then tap <b>WEB_KEY</b> again.",
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
            disable_web_page_preview=True,
        ))
        return
    try:
        await context.bot.send_message(
            chat_id=int(user_id),
            text=_web_key_private_message(new_key),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Forbidden:
        await safe_send(lambda: query.message.edit_text(
            "⚠️ <b>Cannot send WEB_KEY privately.</b>\n\n"
            "Please open a private chat with this bot, press /start, then return to /admin and tap <b>WEB_KEY</b> again.\n\n"
            "The key was already stored in Redis; restart all backend instances to apply it.",
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
            disable_web_page_preview=True,
        ))
        return
    except Exception as exc:
        logger.error("WEB_SECRET_KEY private delivery failed admin_id=%s fingerprint=%s: %s", user_id, fingerprint, exc, exc_info=True)
        await safe_send(lambda: query.message.edit_text(
            "⚠️ <b>WEB_KEY generation failed to deliver.</b>\n\n"
            f"Reason: <code>{html.escape(str(exc)[:300])}</code>\n\n"
            "The key was already stored in Redis; restart all backend instances to apply it.",
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
            disable_web_page_preview=True,
        ))
        return

    logger.info("WEB_SECRET_KEY stored in Redis for admin_id=%s fingerprint=%s", user_id, fingerprint)
    await safe_send(lambda: query.message.edit_text(
        "✅ <b>WEB_KEY stored in Redis.</b>\n\n"
        "I sent the Redis status and restart guidance to your private Telegram chat.\n"
        "No <code>WEB_SECRET_KEY</code> env value is needed now. Restart/deploy all backend instances to use the rotated Redis key.\n\n"
        f"Redis key: <code>{html.escape(_web_secret_redis_key())}</code>\n"
        f"Fingerprint: <code>{html.escape(fingerprint)}</code>",
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
        disable_web_page_preview=True,
    ))


_CRM_TELEGRAM_SEGMENTS = {
    "all": "All",
    "power": "Power",
    "active": "Active",
    "warm": "Warm",
    "risk": "At-risk",
    "blocked": "Blocked",
}


def get_admin_crm_kb(active_segment: str = "all") -> InlineKeyboardMarkup:
    """Compact Telegram CRM controls."""
    active_segment = (active_segment or "all").strip().lower()
    def _label(key: str, text: str) -> str:
        return f"✅ {text}" if key == active_segment else text

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(_label("all", "All"), callback_data="admin_crm:all"),
         InlineKeyboardButton(_label("power", "Power"), callback_data="admin_crm:power"),
         InlineKeyboardButton(_label("active", "Active"), callback_data="admin_crm:active")],
        [InlineKeyboardButton(_label("warm", "Warm"), callback_data="admin_crm:warm"),
         InlineKeyboardButton(_label("risk", "At-risk"), callback_data="admin_crm:risk"),
         InlineKeyboardButton(_label("blocked", "Blocked"), callback_data="admin_crm:blocked")],
        [InlineKeyboardButton("👥 Users", callback_data="admin_users"),
         InlineKeyboardButton("📢 Broadcast", callback_data="admin_broadcast")],
        [InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
    ])


def _format_crm_row_for_telegram(row: dict) -> str:
    uid = int(row.get("user_id") or 0)
    name = html.escape(str(row.get("username") or row.get("first_name") or uid))
    score = int(row.get("quality_score") or 0)
    segment = html.escape(str(row.get("crm_segment") or "unknown"))
    age = row.get("age_days")
    age_text = "unknown" if age is None else f"{int(age)}d"
    messages = int(row.get("recent_messages") or 0)
    blocked = " 🚫" if row.get("blocked") else ""
    return f"• <b>{score}</b> · {name}{blocked} · <code>{uid}</code> · {segment} · {age_text} · {messages} msg"


async def _admin_crm_text(segment: str = "all") -> str:
    segment = (segment or "all").strip().lower() or "all"
    if segment not in _CRM_TELEGRAM_SEGMENTS:
        segment = "all"

    rows, summary = await asyncio.get_running_loop().run_in_executor(
        _DB_EXECUTOR,
        lambda: db_crm_users_snapshot(segment=segment, limit=12, force=False),
    )
    lines = [_format_crm_row_for_telegram(row) for row in rows[:10]]
    users_text = "\n".join(lines) if lines else "No users found for this CRM segment."
    return (
        f"⭐ <b>User Quality / CRM Panel</b>\n"
        f"Segment: <b>{html.escape(_CRM_TELEGRAM_SEGMENTS.get(segment, segment))}</b>\n\n"
        "<b>Summary</b>\n"
        f"👥 Shown: <b>{int(summary.get('shown') or 0)}</b> / loaded <b>{int(summary.get('total_loaded') or 0)}</b>\n"
        f"🔥 Power users: <b>{int(summary.get('power') or 0)}</b>\n"
        f"✅ Active today: <b>{int(summary.get('active') or 0)}</b>\n"
        f"⚠️ At-risk: <b>{int(summary.get('risk') or 0)}</b>\n"
        f"🚫 Blocked: <b>{int(summary.get('blocked') or 0)}</b>\n"
        f"🧠 Cache TTL: <b>{int(summary.get('cache_ttl_s') or CRM_CACHE_TTL_S)}s</b>\n\n"
        "<b>Top users</b>\n"
        f"{users_text}\n\n"
        "Tip: open 👥 Users to view detail, block/unblock, reset prefs, or start admin chat."
    )


async def _admin_open_crm_panel(query, segment: str = "all") -> None:
    segment = (segment or "all").strip().lower() or "all"
    text = await _admin_crm_text(segment)
    await safe_send(lambda: query.message.edit_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_crm_kb(segment),
        disable_web_page_preview=True,
    ))


def get_admin_optimize_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🛠 Runtime", callback_data="admin_runtime"),
         InlineKeyboardButton("🩺 Health", callback_data="admin_health")],
        [InlineKeyboardButton("🚨 Error Center", callback_data="admin_errors"),
         InlineKeyboardButton("🔄 Refresh", callback_data="admin_optimize")],
        [InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
    ])


def _admin_optimize_text() -> str:
    snap = _runtime_performance_snapshot(light=True)
    score, warnings, tips = _optimization_score(snap)
    runtime = snap.get("runtime", {}) if isinstance(snap, dict) else {}
    web = snap.get("web", {}) if isinstance(snap, dict) else {}
    memory = snap.get("memory", {}) if isinstance(snap, dict) else {}
    broadcast = snap.get("broadcast", {}) if isinstance(snap, dict) else {}
    warning_text = "\n".join(f"• {html.escape(w)}" for w in warnings[:5]) if warnings else "• No critical warnings."
    tips_text = "\n".join(f"• {html.escape(t)}" for t in tips[:5]) if tips else "• Keep Balanced preset."
    return (
        "⚡ <b>Optimization Center</b>\n\n"
        f"Score: <b>{score}/100</b>\n"
        f"Mode: <b>{html.escape(str(runtime.get('BOT_MODE') or _run_state_bot_mode()))}</b>\n"
        f"Active web requests: <b>{int(web.get('active_requests') or 0)}</b>\n"
        f"Broadcast queue: <b>{html.escape(str(broadcast.get('queue_size', 0)))}</b>\n"
        f"Rate-limit memory: <b>{int(memory.get('rate_limit_keys') or 0)}</b>\n"
        f"Notice memory: <b>{int(memory.get('rate_limit_notice_keys') or 0)}</b>\n\n"
        "<b>Warnings</b>\n"
        f"{warning_text}\n\n"
        "<b>Recommendations</b>\n"
        f"{tips_text}\n\n"
        "For one-click cleanup and presets, use the web page <code>/admin/optimize</code>."
    )


async def _admin_open_optimize_panel(query) -> None:
    await safe_send(lambda: query.message.edit_text(
        _admin_optimize_text(),
        parse_mode="HTML",
        reply_markup=get_admin_optimize_kb(),
        disable_web_page_preview=True,
    ))


def get_admin_action_kb(cancel_callback: str = "admin_cancel_state") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home"),
         InlineKeyboardButton("❌ បោះបង់",          callback_data=cancel_callback)],
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
        [InlineKeyboardButton("❌ បិទ",          callback_data="api_close")],
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
         InlineKeyboardButton("❌ បិទ",        callback_data="api_close")],
    ])
    return InlineKeyboardMarkup(kbd_rows)


def get_api_revoke_confirm_kb(identifier: str) -> InlineKeyboardMarkup:
    safe_ident = str(identifier or "")[:32]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Confirm Revoke", callback_data=f"api_revoke_confirm:{safe_ident}")],
        [InlineKeyboardButton("⬅️ Back to List", callback_data="api_list"),
         InlineKeyboardButton("❌ បិទ",        callback_data="api_close")],
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
    base, mode = _broadcast_strip_format_directive(str(base), _BROADCAST_PARSE_MODE_AUTO)
    base = str(base).strip()
    if len(base) > limit:
        base = base[:limit] + "…"
    if mode != _BROADCAST_PARSE_MODE_AUTO:
        return f"[{_broadcast_parse_mode_label(mode)}] {base}"
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


_CHINESE_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _detect_tts_lang_key(text: str) -> str:
    """Detect the best Edge/HF TTS language bucket for a text chunk.

    Supports Khmer, English, and Chinese. Chinese detection is done before the
    English fallback because CJK ideographs are also treated as alphabetic by
    Python's str.isalpha().
    """
    text = text or ""
    khmer_chars = len(_KHMER_RE.findall(text))
    chinese_chars = len(_CHINESE_RE.findall(text))
    latin_chars = len(_LATIN_RE.findall(text))
    signal_total = khmer_chars + chinese_chars + latin_chars

    if signal_total <= 0:
        return "en"

    # Prefer the strongest non-Latin script when it has meaningful presence.
    # This keeps short Chinese phrases like "你好" or mixed captions readable.
    if chinese_chars and chinese_chars >= khmer_chars and chinese_chars / signal_total >= 0.15:
        return "zh"
    if khmer_chars and khmer_chars / signal_total >= 0.15:
        return "km"
    return "en"


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
    """Primary voice first, same-language gender fallback, then optional cross-language fallback."""
    gender = gender if gender in ("female", "male") else "female"
    lang = _detect_tts_lang_key(text)
    if lang not in VOICE_MAP:
        lang = "en"
    other_gender = "male" if gender == "female" else "female"
    candidates = [VOICE_MAP[lang][gender], VOICE_MAP[lang][other_gender]]

    if EDGE_TTS_CROSS_LANG_FALLBACK:
        fallback_order = ("km", "zh", "en")
        for other_lang in fallback_order:
            if other_lang == lang or other_lang not in VOICE_MAP:
                continue
            candidates.extend([VOICE_MAP[other_lang][gender], VOICE_MAP[other_lang][other_gender]])

    seen: set[str] = set()
    unique: list[str] = []
    for voice in candidates:
        if voice and voice not in seen:
            seen.add(voice)
            unique.append(voice)
    return unique


async def _edge_tts_stream_once(chunk_text: str, voice: str) -> bytes:
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed. Add `edge-tts` to requirements.txt or choose another TTS provider.")

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


# Khmer HF Space provider state. Protected by a lock because Gradio calls run in worker threads.
_HF_TTS_STATE_LOCK = threading.Lock()
_HF_TTS_CLIENT_LOCK = threading.Lock()
_HF_TTS_CLIENT_CALL_LOCK = threading.Lock()
_HF_TTS_FAILURES = 0
_HF_TTS_DISABLED_UNTIL = 0.0
_HF_TTS_CLIENT = None
_HF_TTS_CLIENT_KEY: tuple[str, str] | None = None


def _tts_provider_summary() -> str:
    return (
        f"provider={TTS_PROVIDER}, khmer={KHMER_TTS_PROVIDER}, "
        f"hf_space={HF_TTS_SPACE}{HF_TTS_API_NAME}, "
        f"gradio_client={'on' if GradioClient is not None else 'missing'}, "
        f"edge_fallback={'on' if HF_TTS_EDGE_FALLBACK else 'off'}"
    )


def _hf_tts_is_temporarily_disabled() -> bool:
    with _HF_TTS_STATE_LOCK:
        return time.monotonic() < _HF_TTS_DISABLED_UNTIL


def _hf_tts_disabled_remaining_s() -> int:
    with _HF_TTS_STATE_LOCK:
        return max(0, int(_HF_TTS_DISABLED_UNTIL - time.monotonic()))


def _hf_tts_record_success() -> None:
    global _HF_TTS_FAILURES, _HF_TTS_DISABLED_UNTIL
    with _HF_TTS_STATE_LOCK:
        _HF_TTS_FAILURES = 0
        _HF_TTS_DISABLED_UNTIL = 0.0


def _hf_tts_error_text(exc: BaseException | str) -> str:
    return str(exc or "").strip()


def _hf_tts_is_quota_error(exc: BaseException | str) -> bool:
    msg = _hf_tts_error_text(exc).lower()
    return (
        "zerogpu quota" in msg
        or "exceeded your free" in msg
        or "quota" in msg and "exceeded" in msg
        or "0s left" in msg
    )


def _hf_tts_is_no_audio_error(exc: BaseException | str) -> bool:
    msg = _hf_tts_error_text(exc).lower()
    return "returned no audio path" in msg or "returned empty audio" in msg or "no audio" in msg


def _hf_tts_record_failure(exc: BaseException | str) -> None:
    """Record real HF TTS failures and open a cooldown when retries are harmful.

    Cooldown-only errors are intentionally not counted by generate_voice(). Quota
    failures are disabled immediately because the next request cannot fix the
    account-level ZeroGPU quota. "No audio path" gets a medium cooldown after
    the normal failure threshold because Spaces sometimes return malformed tuples
    during cold starts or overloaded periods.
    """
    global _HF_TTS_FAILURES, _HF_TTS_DISABLED_UNTIL
    exc_text = _hf_tts_error_text(exc)[:500]
    now = time.monotonic()
    cooldown = 0.0
    reason = "failure-threshold"

    with _HF_TTS_STATE_LOCK:
        _HF_TTS_FAILURES += 1
        previous_until = _HF_TTS_DISABLED_UNTIL
        already_disabled = now < previous_until

        if _hf_tts_is_quota_error(exc):
            cooldown = HF_TTS_QUOTA_COOLDOWN_S
            reason = "quota"
        elif _HF_TTS_FAILURES >= HF_TTS_FAILURE_LIMIT:
            cooldown = HF_TTS_NO_AUDIO_COOLDOWN_S if _hf_tts_is_no_audio_error(exc) else HF_TTS_COOLDOWN_S
            reason = "no-audio" if _hf_tts_is_no_audio_error(exc) else "failure-threshold"

        if cooldown > 0:
            _HF_TTS_DISABLED_UNTIL = max(previous_until, now + cooldown)
            should_log = (not already_disabled) or (_HF_TTS_DISABLED_UNTIL > previous_until + 1.0)
        else:
            should_log = False

        failures = _HF_TTS_FAILURES
        remaining = max(0, int(_HF_TTS_DISABLED_UNTIL - now))

    if should_log:
        logger.warning(
            "HF Khmer TTS temporarily disabled for %ss after %s failure(s), reason=%s: %s",
            remaining,
            failures,
            reason,
            exc_text,
        )


def _should_try_hf_khmer_tts(text: str, tts_model: str = "auto") -> bool:
    if _detect_tts_lang_key(text) != "km":
        return False

    user_model = _normalize_tts_model(tts_model)
    if user_model == "edge":
        return False

    # When HF is cooling down and Edge fallback is enabled, do not call the HF
    # Space at all. This avoids repeated "Loaded as API" messages, avoids
    # extending cooldown on cooldown-only errors, and gives users a fast reply.
    if _hf_tts_is_temporarily_disabled() and HF_TTS_EDGE_FALLBACK:
        return False

    if user_model == "hf_space":
        return True

    # Auto mode follows server-level env config.
    mode = (TTS_PROVIDER or "auto").strip().lower()
    khmer_mode = (KHMER_TTS_PROVIDER or "hf_space").strip().lower()
    if mode in {"edge", "edge_tts"} or khmer_mode in {"edge", "edge_tts"}:
        return False
    if mode in {"hf", "hf_space", "khmer_hf_space", "khmer-tts", "auto"}:
        return True
    return khmer_mode in {"hf", "hf_space", "khmer_hf_space", "khmer-tts"}


_HF_TTS_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".mp4", ".webm", ".opus"}


def _hf_tts_is_probable_audio_ref(value: Any) -> bool:
    """Return True only for real Gradio audio references.

    mrrtmob/khmer-tts may include status strings in its tuple result, for
    example "Characters: 238/300". Treating those strings as file paths causes
    FileNotFoundError. We therefore accept only HTTP(S) URLs or existing local
    files, and reject known UI/status labels.
    """
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False

    low = text.lower()
    if low.startswith(("characters:", "character:", "duration:", "status:", "error:", "warning:")):
        return False
    if "characters:" in low and re.search(r"\b\d+\s*/\s*\d+\b", low):
        return False

    if re.match(r"^https?://", text, re.I):
        return True

    # Gradio normally returns a real temporary path for local files. Do not
    # accept arbitrary strings just because they look like filenames.
    try:
        return os.path.isfile(text)
    except (OSError, ValueError):
        return False


def _extract_hf_audio_path_or_url(audio_obj: Any) -> str:
    """Extract a valid audio file path/URL from Gradio result shapes.

    Some Gradio Spaces return nested tuples/lists such as (audio, metadata) or
    dictionaries containing path/url fields. Status strings like
    "Characters: 238/300" must be ignored because they are not audio files.
    """
    if audio_obj is None:
        return ""
    if isinstance(audio_obj, str):
        return audio_obj.strip() if _hf_tts_is_probable_audio_ref(audio_obj) else ""
    if isinstance(audio_obj, dict):
        # Prefer explicit audio references first. Validate them before returning.
        for key in ("path", "url", "name"):
            val = audio_obj.get(key)
            if _hf_tts_is_probable_audio_ref(val):
                return str(val).strip()
        for val in audio_obj.values():
            nested = _extract_hf_audio_path_or_url(val)
            if nested:
                return nested
    if isinstance(audio_obj, (tuple, list)):
        for item in audio_obj:
            nested = _extract_hf_audio_path_or_url(item)
            if nested:
                return nested
    for attr in ("path", "url", "name"):
        val = getattr(audio_obj, attr, None)
        if _hf_tts_is_probable_audio_ref(val):
            return str(val).strip()
    return ""


def _hf_tts_dict_get_any(data: dict, keys: tuple[str, ...]) -> Any:
    """Return the first present dict value without using `or` on numpy arrays."""
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return None


def _hf_tts_array_like_to_wav_bytes(sample_rate: Any, samples: Any) -> bytes:
    """Encode Gradio Audio(type='numpy') output into WAV bytes.

    mrrtmob/khmer-tts declares `gr.Audio(type="numpy")`, so the Python-side
    result can be `(24000, audio_samples)` instead of a temporary file path.
    Telegram/FFmpeg can consume WAV bytes reliably, so convert int/float arrays
    or nested Python lists to a small WAV file in memory.
    """
    if samples is None:
        return b""
    try:
        sr = int(sample_rate)
    except Exception:
        return b""
    if sr < 8000 or sr > 192000:
        return b""

    import io
    import wave
    channels = 1
    pcm = b""

    try:
        import numpy as _np  # Optional; gradio/numpy is usually present with HF TTS.
        arr = _np.asarray(samples)
        if arr.size <= 0:
            return b""
        arr = _np.squeeze(arr)
        if arr.ndim == 0:
            return b""
        if arr.ndim == 2:
            # Accept either [frames, channels] or [channels, frames].
            if arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
                arr = arr.T
            if arr.shape[1] in (1, 2):
                channels = int(arr.shape[1])
            else:
                arr = arr[:, 0]
                channels = 1
        elif arr.ndim > 2:
            arr = arr.reshape(-1)
            channels = 1

        if _np.issubdtype(arr.dtype, _np.floating):
            arr = _np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            arr = _np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767).astype("<i2")
        elif arr.dtype != _np.int16:
            arr = _np.clip(arr, -32768, 32767).astype("<i2")
        else:
            arr = arr.astype("<i2", copy=False)
        pcm = arr.tobytes()
    except Exception:
        # Pure-Python fallback for JSON/list-like arrays.
        from array import array
        try:
            seq = samples.tolist() if hasattr(samples, "tolist") else samples
        except Exception:
            seq = samples
        if not isinstance(seq, (list, tuple)) or not seq:
            return b""
        flat: list[Any] = []
        if isinstance(seq[0], (list, tuple)):
            first_len = len(seq[0])
            if first_len in (1, 2):
                channels = first_len
                for frame in seq:
                    for value in list(frame)[:channels]:
                        flat.append(value)
            else:
                for row in seq:
                    if isinstance(row, (list, tuple)):
                        flat.extend(row)
                    else:
                        flat.append(row)
                channels = 1
        else:
            flat = list(seq)
            channels = 1
        if not flat:
            return b""
        pcm_arr = array("h")
        for value in flat:
            try:
                f = float(value)
                if -1.0 <= f <= 1.0:
                    f = f * 32767
                pcm_arr.append(max(-32768, min(32767, int(f))))
            except Exception:
                pcm_arr.append(0)
        pcm = pcm_arr.tobytes()

    if not pcm:
        return b""

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(max(1, min(2, int(channels or 1))))
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm)
    return out.getvalue()


def _extract_hf_audio_bytes_from_result(audio_obj: Any) -> bytes:
    """Extract actual audio bytes from Gradio result shapes.

    Supports both common result formats:
      - file/path/url dictionaries or strings handled elsewhere
      - numpy audio tuples such as `(24000, audio_samples)` returned by
        `gr.Audio(type="numpy")`
    Non-audio UI/status strings such as `Characters: 238/300` are ignored.
    """
    if audio_obj is None:
        return b""

    if isinstance(audio_obj, dict):
        sr = _hf_tts_dict_get_any(audio_obj, ("sample_rate", "sampling_rate", "rate", "sr"))
        samples = _hf_tts_dict_get_any(audio_obj, ("data", "array", "samples", "audio"))
        data = _hf_tts_array_like_to_wav_bytes(sr, samples)
        if data:
            return data
        for value in audio_obj.values():
            data = _extract_hf_audio_bytes_from_result(value)
            if data:
                return data
        return b""

    if isinstance(audio_obj, (tuple, list)):
        # Gradio Audio(type='numpy') convention: (sample_rate, audio_samples).
        if len(audio_obj) >= 2:
            data = _hf_tts_array_like_to_wav_bytes(audio_obj[0], audio_obj[1])
            if data:
                return data
        for item in audio_obj:
            data = _extract_hf_audio_bytes_from_result(item)
            if data:
                return data
    return b""


def _audio_suffix_from_bytes(data: bytes) -> str:
    head = data[:16]
    if head.startswith(b"RIFF"):
        return ".wav"
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"ID3") or head[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return ".mp3"
    if b"ftyp" in head:
        return ".mp4"
    return ".audio"


async def _terminate_subprocess(proc: asyncio.subprocess.Process | None, label: str) -> None:
    """Best-effort cleanup for timed-out FFmpeg/subprocess jobs.

    Without this, a timed-out conversion can leave an orphan process alive on
    small Render instances, causing the next TTS request to become slower.
    """
    if proc is None or proc.returncode is not None:
        return
    with suppress(ProcessLookupError):
        proc.kill()
    try:
        await asyncio.wait_for(proc.communicate(), timeout=5)
    except Exception as exc:
        logger.warning("Could not fully clean up %s subprocess: %s", label, exc)


def _ffmpeg_tts_timeout_s(*, chunk_count: int = 1, input_bytes: int = 0) -> float:
    """Compute a bounded FFmpeg timeout for TTS conversion.

    The old fixed 75s timeout was too tight for long Khmer/Edge TTS jobs on
    small Render instances.  Keep short jobs fast, but give longer/multi-chunk
    conversions enough CPU time before killing FFmpeg.
    """
    chunks = max(1, int(chunk_count or 1))
    size_mb = max(0.0, float(input_bytes or 0) / (1024 * 1024))
    dynamic = 45.0 + (25.0 * chunks) + (12.0 * size_mb)
    return min(600.0, max(float(FFMPEG_CONVERT_TIMEOUT_S), dynamic))


def _hf_tts_make_client_sync():
    if GradioClient is None:
        raise RuntimeError("gradio_client is not installed. Add `gradio_client` to requirements.txt.")
    if not HF_TTS_SPACE:
        raise RuntimeError("HF_TTS_SPACE is empty.")
    if HF_TTS_TOKEN:
        try:
            return GradioClient(HF_TTS_SPACE, hf_token=HF_TTS_TOKEN)
        except TypeError:
            return GradioClient(HF_TTS_SPACE)
    return GradioClient(HF_TTS_SPACE)


def _hf_tts_get_client_sync():
    """Return a cached Gradio client so each chunk/request does not reload the API."""
    global _HF_TTS_CLIENT, _HF_TTS_CLIENT_KEY
    if not HF_TTS_CLIENT_CACHE:
        return _hf_tts_make_client_sync()

    # Do not store the full token in the cache key; the process env is static in
    # normal deployments, and this is enough to rebuild if token presence changes.
    key = (HF_TTS_SPACE, "token" if HF_TTS_TOKEN else "public")
    with _HF_TTS_CLIENT_LOCK:
        if _HF_TTS_CLIENT is not None and _HF_TTS_CLIENT_KEY == key:
            return _HF_TTS_CLIENT
        _HF_TTS_CLIENT = _hf_tts_make_client_sync()
        _HF_TTS_CLIENT_KEY = key
        return _HF_TTS_CLIENT


def _hf_tts_predict_should_retry(exc: Exception) -> bool:
    """Return True for transient Gradio/ZeroGPU cold-start style failures."""
    msg = str(exc).lower()
    non_retryable = (
        "invalid api",
        "api_name",
        "not found",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "invalid token",
    )
    if any(token in msg for token in non_retryable):
        return False
    retryable = (
        "cold",
        "starting",
        "loading",
        "zerogpu",
        "zero gpu",
        "queue",
        "queued",
        "timeout",
        "timed out",
        "temporarily",
        "connection",
        "read error",
        "503",
        "504",
        "502",
        "too busy",
        "no valid audio",
        "empty audio",
    )
    return any(token in msg for token in retryable)


def _hf_tts_sleep_before_retry(attempt: int) -> None:
    delay = min(float(HF_TTS_RETRY_DELAY_S) * (2 ** max(0, attempt - 1)), 30.0)
    time.sleep(delay)


def _hf_tts_space_predict_sync(chunk_text: str) -> bytes:
    """Blocking Gradio call for mrrtmob/khmer-tts. Run only in an executor.

    Public ZeroGPU Spaces often cold-start or sit in a queue.  Retry the
    blocking predict call here with exponential backoff so one cold start does
    not immediately fail the whole Telegram TTS request.
    """
    client = _hf_tts_get_client_sync()

    def _predict():
        return client.predict(
            chunk_text,
            HF_TTS_VOICE,
            HF_TTS_TEMP,
            HF_TTS_TOP_P,
            HF_TTS_REP_PEN,
            HF_TTS_MAX_TOK,
            api_name=HF_TTS_API_NAME,
        )

    def _predict_once() -> bytes:
        if HF_TTS_SERIALIZE_CALLS:
            with _HF_TTS_CLIENT_CALL_LOCK:
                result = _predict()
        else:
            result = _predict()

        # The Space uses `gr.Audio(type="numpy")`, so successful responses may be
        # `(24000, np.ndarray)` instead of a file path. Extract audio bytes first,
        # then fall back to file/URL references. Ignore secondary UI strings such as
        # `Characters: 238/300`.
        data = _extract_hf_audio_bytes_from_result(result)
        if data:
            return data

        path_or_url = _extract_hf_audio_path_or_url(result)
        if not path_or_url:
            result_preview = _web_short(result, 240) if "_web_short" in globals() else str(result)[:240]
            raise RuntimeError(
                f"HF TTS returned no valid audio data/file/url. result_type={type(result).__name__} result={result_preview!r}"
            )

        if re.match(r"^https?://", path_or_url, re.I):
            resp = httpx.get(path_or_url, timeout=HF_TTS_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.content or b""
        else:
            if not os.path.isfile(path_or_url):
                raise RuntimeError(f"HF TTS returned missing audio file: {path_or_url!r}")
            with open(path_or_url, "rb") as fh:
                data = fh.read()
        if not data:
            raise RuntimeError("HF TTS returned empty audio.")
        return data

    last_exc: Exception | None = None
    retries = max(1, int(HF_TTS_RETRIES or 3))
    for attempt in range(1, retries + 1):
        try:
            return _predict_once()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries or not _hf_tts_predict_should_retry(exc):
                raise
            logger.warning(
                "HF Khmer TTS Gradio call failed attempt=%s/%s; retrying with exponential backoff: %s",
                attempt,
                retries,
                exc,
            )
            _hf_tts_sleep_before_retry(attempt)

    raise RuntimeError(f"HF TTS failed after {retries} attempts: {last_exc}")


async def _convert_audio_files_to_telegram_voice(input_paths: list[str], speed: float, output_path: str) -> bytes:
    """Convert one or more audio files into Telegram voice-compatible OGG/Opus."""
    if not input_paths:
        raise RuntimeError("No audio files to convert.")

    speed_key = _rounded_speed(speed)
    af = _build_atempo_chain(speed_key) if abs(speed_key - DEFAULT_SPEED) > 1e-4 else None

    cmd = [_FFMPEG_EXE, "-y"]
    for path in input_paths:
        cmd += ["-i", path]

    if len(input_paths) > 1:
        concat_inputs = "".join(f"[{idx}:a]" for idx in range(len(input_paths)))
        filter_complex = f"{concat_inputs}concat=n={len(input_paths)}:v=0:a=1[a0]"
        map_label = "[a0]"
        if af:
            filter_complex += f";[a0]{af}[aout]"
            map_label = "[aout]"
        cmd += ["-filter_complex", filter_complex, "-map", map_label]
    elif af:
        cmd += ["-filter:a", af]

    cmd += ["-vn", "-c:a", "libopus", "-b:a", "32k", output_path]

    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=FFMPEG_START_TIMEOUT_S,
        )
        timeout_s = _ffmpeg_tts_timeout_s(chunk_count=len(input_paths))
        _, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except FileNotFoundError as exc:
        _cleanup(output_path)
        raise RuntimeError(f"FFmpeg executable not found: {_FFMPEG_EXE}") from exc
    except asyncio.TimeoutError:
        _cleanup(output_path)
        await _terminate_subprocess(proc, "HF TTS FFmpeg")
        raise RuntimeError("FFmpeg timed out while converting HF TTS audio")

    if proc.returncode != 0:
        _cleanup(output_path)
        snippet = (stderr_data or b"").decode(errors="replace")[-600:]
        raise RuntimeError(f"FFmpeg failed converting HF TTS audio (code {proc.returncode}): {snippet}")

    try:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: open(output_path, "rb").read()),
            timeout=10,
        )
    except asyncio.TimeoutError:
        _cleanup(output_path)
        raise RuntimeError("Timed out reading converted HF TTS output")
    except OSError as exc:
        _cleanup(output_path)
        raise RuntimeError(f"Failed to read converted HF TTS output: {exc}") from exc


async def _generate_voice_hf_space(text: str, speed: float, output_path: str) -> bytes:
    """Generate Khmer TTS with mrrtmob/khmer-tts and convert to Telegram voice."""
    if _hf_tts_is_temporarily_disabled():
        raise RuntimeError(f"HF Khmer TTS cooldown active ({_hf_tts_disabled_remaining_s()}s remaining)")

    chunks = _split_text_chunks(text, max_chars=HF_TTS_MAX_CHARS)
    if not chunks:
        raise ValueError("HF Khmer TTS: no speakable text chunks")

    loop = asyncio.get_running_loop()
    with tempfile.TemporaryDirectory(prefix="hf_khmer_tts_") as tmpdir:
        input_paths: list[str] = []
        for idx, chunk_text in enumerate(chunks, 1):
            # _hf_tts_space_predict_sync owns the retry/backoff policy.  Keep a
            # single executor submission per chunk so retries do not multiply
            # and leave timed-out worker calls running in the background.
            try:
                retry_budget = max(1, int(HF_TTS_RETRIES or 3))
                backoff_budget = sum(
                    min(float(HF_TTS_RETRY_DELAY_S) * (2 ** max(0, attempt - 1)), 30.0)
                    for attempt in range(1, retry_budget)
                )
                predict_timeout_s = (HF_TTS_TIMEOUT_S * retry_budget) + backoff_budget + 10.0
                audio_bytes = await asyncio.wait_for(
                    loop.run_in_executor(_AI_EXECUTOR, _hf_tts_space_predict_sync, chunk_text),
                    timeout=predict_timeout_s,
                )
            except Exception as exc:
                preview = chunk_text[:80].replace("\n", " ")
                raise RuntimeError(
                    f"HF Khmer TTS failed at chunk {idx}/{len(chunks)} ({preview!r}): {exc}"
                ) from exc
            if not audio_bytes:
                preview = chunk_text[:80].replace("\n", " ")
                raise RuntimeError(
                    f"HF Khmer TTS failed at chunk {idx}/{len(chunks)} ({preview!r}): empty audio"
                )
            suffix = _audio_suffix_from_bytes(audio_bytes)
            in_path = os.path.join(tmpdir, f"chunk_{idx:03d}{suffix}")
            with open(in_path, "wb") as fh:
                fh.write(audio_bytes)
            input_paths.append(in_path)

        converted = await _convert_audio_files_to_telegram_voice(input_paths, speed, output_path)
        if not converted:
            raise RuntimeError("HF Khmer TTS produced empty converted output")
        _hf_tts_record_success()
        logger.info("HF Khmer TTS generated %s chunk(s) via %s%s", len(chunks), HF_TTS_SPACE, HF_TTS_API_NAME)
        return converted


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


# ---------------------------------------------------------------------------
# In-memory TTS audio cache
# ---------------------------------------------------------------------------
_TTS_AUDIO_CACHE: OrderedDict[str, tuple[bytes, float, int]] = OrderedDict()
_TTS_AUDIO_CACHE_LOCK = threading.RLock()
_TTS_AUDIO_CACHE_BYTES = 0


def _tts_audio_cache_key(text: str, gender: str, speed: float, tts_model: str) -> str:
    cleaned = _clean_tts_text_for_edge(text)
    lang = _detect_tts_lang_key(cleaned)
    payload = {
        "v": 2,
        "lang": lang,
        "gender": gender if gender in ("female", "male") else "female",
        "speed": _rounded_speed(speed),
        "model": _normalize_tts_model(tts_model),
        "tts_provider": TTS_PROVIDER,
        "khmer_provider": KHMER_TTS_PROVIDER,
        "text_hash": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
    }
    raw = _json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tts_audio_cache_get(key: str) -> bytes | None:
    if not TTS_AUDIO_CACHE_ENABLED:
        return None
    now = time.monotonic()
    with _TTS_AUDIO_CACHE_LOCK:
        item = _TTS_AUDIO_CACHE.get(key)
        if not item:
            return None
        data, created, size = item
        if now - created > TTS_AUDIO_CACHE_TTL_S:
            global _TTS_AUDIO_CACHE_BYTES
            _TTS_AUDIO_CACHE.pop(key, None)
            _TTS_AUDIO_CACHE_BYTES = max(0, _TTS_AUDIO_CACHE_BYTES - size)
            return None
        _TTS_AUDIO_CACHE.move_to_end(key)
        return bytes(data)


def _tts_audio_cache_set(key: str, data: bytes) -> None:
    if not TTS_AUDIO_CACHE_ENABLED or not data:
        return
    size = len(data)
    if size > TTS_AUDIO_CACHE_ITEM_MAX_BYTES:
        return
    global _TTS_AUDIO_CACHE_BYTES
    now = time.monotonic()
    with _TTS_AUDIO_CACHE_LOCK:
        old = _TTS_AUDIO_CACHE.pop(key, None)
        if old:
            _TTS_AUDIO_CACHE_BYTES = max(0, _TTS_AUDIO_CACHE_BYTES - old[2])
        _TTS_AUDIO_CACHE[key] = (bytes(data), now, size)
        _TTS_AUDIO_CACHE_BYTES += size
        while _TTS_AUDIO_CACHE_BYTES > TTS_AUDIO_CACHE_MAX_BYTES and _TTS_AUDIO_CACHE:
            _old_key, (_old_data, _old_created, old_size) = _TTS_AUDIO_CACHE.popitem(last=False)
            _TTS_AUDIO_CACHE_BYTES = max(0, _TTS_AUDIO_CACHE_BYTES - old_size)


def _write_cached_audio_to_path(path: str, data: bytes) -> None:
    if not path or not data:
        return
    try:
        with open(path, "wb") as fh:
            fh.write(data)
    except OSError:
        # The caller primarily uses the returned bytes. Disk cache write-through
        # is only for compatibility with cleanup/output_path expectations.
        pass

async def _generate_voice_edge(text: str, gender: str, speed: float, output_path: str) -> bytes:
    text = _clean_tts_text_for_edge(text)
    if not text:
        raise ValueError("generate_voice: text must not be empty")

    voices = _tts_voice_candidates(text, gender)
    text_chunks = _split_text_chunks(text, max_chars=EDGE_TTS_CHUNK_CHARS)
    if not text_chunks:
        raise ValueError("generate_voice: no speakable text chunks")

    async def _render_edge_chunk(idx: int, chunk_text: str) -> tuple[int, bytes, str]:
        try:
            chunk_mp3, used_voice = await _edge_tts_stream_with_retry(chunk_text, voices)
            return idx, chunk_mp3, used_voice
        except Exception as e:
            preview = chunk_text[:80].replace("\n", " ")
            raise RuntimeError(
                f"edge-tts failed at chunk {idx}/{len(text_chunks)} ({preview!r}): {e}"
            ) from e

    used_voices: list[str] = []
    if len(text_chunks) > 1 and EDGE_TTS_PARALLEL_CHUNKS > 1:
        chunk_sem = asyncio.Semaphore(min(EDGE_TTS_PARALLEL_CHUNKS, len(text_chunks)))

        async def _guarded(idx: int, chunk_text: str) -> tuple[int, bytes, str]:
            async with chunk_sem:
                return await _render_edge_chunk(idx, chunk_text)

        results = await asyncio.gather(*(
            _guarded(idx, chunk_text) for idx, chunk_text in enumerate(text_chunks, 1)
        ))
        results.sort(key=lambda item: item[0])
        mp3_parts = [item[1] for item in results]
        used_voices = [item[2] for item in results]
    else:
        mp3_parts: list[bytes] = []
        for idx, chunk_text in enumerate(text_chunks, 1):
            _idx, chunk_mp3, used_voice = await _render_edge_chunk(idx, chunk_text)
            mp3_parts.append(chunk_mp3)
            used_voices.append(used_voice)

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

    proc: asyncio.subprocess.Process | None = None
    timeout_s = float(FFMPEG_CONVERT_TIMEOUT_S)
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=FFMPEG_START_TIMEOUT_S,
        )
        timeout_s = _ffmpeg_tts_timeout_s(chunk_count=len(text_chunks), input_bytes=len(mp3_data))
        _, stderr_data = await asyncio.wait_for(proc.communicate(input=mp3_data), timeout=timeout_s)
    except FileNotFoundError as exc:
        _cleanup(output_path)
        raise RuntimeError(f"FFmpeg executable not found: {_FFMPEG_EXE}") from exc
    except asyncio.TimeoutError:
        _cleanup(output_path)
        await _terminate_subprocess(proc, "Edge TTS FFmpeg")
        raise RuntimeError(f"FFmpeg timed out after {timeout_s:.0f}s")

    if proc.returncode != 0:
        _cleanup(output_path)
        snippet = (stderr_data or b"").decode(errors="replace")[-400:]
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode}): {snippet}")

    try:
        loop        = asyncio.get_running_loop()
        audio_bytes = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: open(output_path, "rb").read()),
            timeout=10,
        )
    except asyncio.TimeoutError:
        _cleanup(output_path)
        raise RuntimeError("Timed out reading output audio file")
    except OSError as e:
        _cleanup(output_path)
        raise RuntimeError(f"Failed to read output audio file: {e}") from e
    return audio_bytes


async def generate_voice(text: str, gender: str, speed: float, output_path: str, tts_model: str = "auto") -> bytes:
    """Generate Telegram voice audio using the user's selected TTS model.

    User model routing:
      - auto: server default; Khmer usually uses HF Space, English uses Edge.
      - hf_space: force mrrtmob/khmer-tts for Khmer text; English still uses Edge.
      - edge: force Edge TTS for all supported languages.
    """
    text = _clean_tts_text_for_edge(text)
    if not text:
        raise ValueError("generate_voice: text must not be empty")

    user_model = _normalize_tts_model(tts_model)
    if _should_try_hf_khmer_tts(text, user_model):
        try:
            return await _generate_voice_hf_space(text, speed, output_path)
        except Exception as exc:
            low = str(exc).lower()
            cooldown_only = "cooldown active" in low
            if not cooldown_only:
                _hf_tts_record_failure(exc)
                logger.warning(
                    "HF Khmer TTS failed; user_model=%s edge fallback=%s cooldown_remaining=%ss: %s",
                    user_model,
                    HF_TTS_EDGE_FALLBACK,
                    _hf_tts_disabled_remaining_s(),
                    exc,
                )
            else:
                logger.info(
                    "HF Khmer TTS skipped due to cooldown; user_model=%s edge fallback=%s cooldown_remaining=%ss",
                    user_model,
                    HF_TTS_EDGE_FALLBACK,
                    _hf_tts_disabled_remaining_s(),
                )
            if not HF_TTS_EDGE_FALLBACK:
                raise RuntimeError(f"HF Khmer TTS failed and Edge fallback is disabled: {exc}") from exc

    return await _generate_voice_edge(text, gender, speed, output_path)


async def generate_voice_limited(text: str, gender: str, speed: float, output_path: str, tts_model: str = "auto") -> bytes:
    cleaned = _clean_tts_text_for_edge(text)
    cache_key = _tts_audio_cache_key(cleaned, gender, speed, tts_model)
    cached = _tts_audio_cache_get(cache_key)
    if cached is not None:
        await asyncio.to_thread(_write_cached_audio_to_path, output_path, cached)
        logger.debug("TTS audio cache hit lang=%s bytes=%s", _detect_tts_lang_key(cleaned), len(cached))
        return cached

    sem = _TTS_CHUNK_SEMAPHORE
    if sem is None:
        audio = await generate_voice(cleaned, gender, speed, output_path, tts_model)
    else:
        async with sem:
            # Re-check after waiting; another request may have generated the same audio.
            cached = _tts_audio_cache_get(cache_key)
            if cached is not None:
                await asyncio.to_thread(_write_cached_audio_to_path, output_path, cached)
                logger.debug("TTS audio cache hit after wait lang=%s bytes=%s", _detect_tts_lang_key(cleaned), len(cached))
                return cached
            audio = await generate_voice(cleaned, gender, speed, output_path, tts_model)
    _tts_audio_cache_set(cache_key, audio)
    return audio


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
        "Support Khmer, English, and Chinese."
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
        "Support Khmer, English, and Chinese."
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
    tts_model: str = "auto",
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
            audio_bytes = await generate_voice_limited(chunk, gender, speed, file_path, tts_model)
            sent = await safe_send(
                lambda ab=audio_bytes, ci=i, ct=total: bot.send_voice(
                    chat_id=chat_id,
                    voice=io.BytesIO(ab),
                    caption=f"🗣️ {BOT_TAG}  [{ci}/{ct}]",
                    reply_markup=get_main_kb(gender, tts_model),
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
            await asyncio.sleep(float(PAGED_TTS_SEND_DELAY_S))

    _set_last_tts(user_id)


# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------
def get_main_kb(gender: str, tts_model: str = "auto") -> InlineKeyboardMarkup:
    f_btn = "👩 សំឡេងស្រី" + (" ✅" if gender == "female" else "")
    m_btn = "👨 សំឡេងប្រុស" + (" ✅" if gender == "male" else "")
    model_key = _normalize_tts_model(tts_model)
    model_btn = f"🤖 ម៉ូដែល TTS: {TTS_MODEL_OPTIONS.get(model_key, TTS_MODEL_OPTIONS['auto'])[0]}"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f_btn, callback_data="tg_female"),
         InlineKeyboardButton(m_btn, callback_data="tg_male")],
        [InlineKeyboardButton("🎚️ ល្បឿនសំឡេង", callback_data="show_speed"),
         InlineKeyboardButton(model_btn, callback_data="show_tts_model")],
    ])


def get_tts_model_kb(current_model: str = "auto") -> InlineKeyboardMarkup:
    current = _normalize_tts_model(current_model)
    rows: list[list[InlineKeyboardButton]] = []
    for key, (label, hint) in TTS_MODEL_OPTIONS.items():
        suffix = " ✅" if key == current else ""
        rows.append([InlineKeyboardButton(f"{label}{suffix} — {hint}", callback_data=f"ttsmodel_{key}")])
    rows.append([InlineKeyboardButton("🔙 ត្រឡប់", callback_data="hide_tts_model")])
    return InlineKeyboardMarkup(rows)


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
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ បញ្ជាក់ផ្ញើទៅអ្នកប្រើសកម្ម", callback_data="bc_confirm")],
        [InlineKeyboardButton("📝 រក្សាទុក Template", callback_data="bc_save_template"),
         InlineKeyboardButton("📚 បណ្ណាល័យ Template", callback_data="bc_templates")],
        [InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home"),
         InlineKeyboardButton("❌ បោះបង់", callback_data="bc_cancel")],
    ])


def get_broadcast_entry_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📚 បណ្ណាល័យ Template", callback_data="bc_templates")],
        [InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home"),
         InlineKeyboardButton("❌ បោះបង់", callback_data="bc_cancel")],
    ])


def get_broadcast_templates_kb(templates: list[dict]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for tpl in templates[:BROADCAST_TEMPLATE_LIBRARY_MAX]:
        tpl_id = _broadcast_template_safe_id(tpl.get("id"))
        if not tpl_id:
            continue
        title = _broadcast_template_button_title(tpl)
        rows.append([
            InlineKeyboardButton(f"▶️ {title}", callback_data=f"bc_tpl_use:{tpl_id}"),
            InlineKeyboardButton("🗑", callback_data=f"bc_tpl_delask:{tpl_id}"),
        ])
    rows.append([InlineKeyboardButton("🔄 Refresh", callback_data="bc_templates"),
                 InlineKeyboardButton("➕ Broadcast ថ្មី", callback_data="admin_broadcast")])
    rows.append([InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home"),
                 InlineKeyboardButton("❌ បិទ", callback_data="admin_close")])
    return InlineKeyboardMarkup(rows)


def get_broadcast_template_delete_confirm_kb(template_id: str) -> InlineKeyboardMarkup:
    tpl_id = _broadcast_template_safe_id(template_id)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ លុប Template នេះ", callback_data=f"bc_tpl_del:{tpl_id}")],
        [InlineKeyboardButton("↩️ មិនលុប / ត្រឡប់ក្រោយ", callback_data="bc_templates")],
    ])


def get_broadcast_sent_delete_kb(delete_job_id: str) -> InlineKeyboardMarkup:
    job_id = _broadcast_sent_delete_safe_id(delete_job_id)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗑️ លុបសារដែលបានផ្ញើ", callback_data=f"bc_del_sent_ask:{job_id}")],
        [InlineKeyboardButton("⬅️ Admin V8", callback_data="admin_home")],
    ])


def get_broadcast_sent_delete_confirm_kb(delete_job_id: str) -> InlineKeyboardMarkup:
    job_id = _broadcast_sent_delete_safe_id(delete_job_id)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ បញ្ជាក់លុបសារដែលបានផ្ញើ", callback_data=f"bc_del_sent_run:{job_id}")],
        [InlineKeyboardButton("↩️ ទុកសារនៅដដែល", callback_data="bc_del_sent_keep")],
    ])


def _broadcast_template_safe_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if re.fullmatch(r"[a-f0-9]{8,16}", text) else ""


def _broadcast_template_safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _broadcast_template_content(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    content = payload.get("caption") if payload.get("photo_file_id") else payload.get("text")
    return str(content or "")


def _broadcast_template_clean_preview(text: Any, *, max_len: int = BROADCAST_TEMPLATE_TITLE_MAX) -> str:
    clean = str(text or "").strip()
    clean = re.sub(r"^::(?:html|mdv2|md|plain)\s*\n", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        clean = "គ្មាន Caption"
    if len(clean) > max_len:
        clean = clean[: max(1, max_len - 1)].rstrip() + "…"
    return clean


def _broadcast_template_title_from_payload(payload: dict) -> str:
    prefix = "🖼️ " if payload.get("photo_file_id") else "📝 "
    return (prefix + _broadcast_template_clean_preview(_broadcast_template_content(payload))).strip()[:BROADCAST_TEMPLATE_TITLE_MAX + 3]


def _broadcast_template_button_title(tpl: dict) -> str:
    title = str(tpl.get("title") or "").strip()
    if not title:
        title = _broadcast_template_title_from_payload(tpl)
    return _broadcast_template_clean_preview(title, max_len=BROADCAST_TEMPLATE_BUTTON_TITLE_MAX)


def _broadcast_template_payload_from_template(tpl: dict) -> dict:
    return {
        "photo_file_id": str(tpl.get("photo_file_id") or "").strip() or None,
        "caption": str(tpl.get("caption") or "") if tpl.get("photo_file_id") else None,
        "text": None if tpl.get("photo_file_id") else str(tpl.get("text") or ""),
        "parse_mode": _broadcast_normalize_parse_mode(tpl.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO),
    }


def _broadcast_template_fingerprint(payload: dict) -> str:
    clean = {
        "photo_file_id": str(payload.get("photo_file_id") or ""),
        "caption": str(payload.get("caption") or ""),
        "text": str(payload.get("text") or ""),
        "parse_mode": _broadcast_normalize_parse_mode(payload.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO),
    }
    raw = _json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _broadcast_template_normalize(tpl: dict, *, fallback_id: str | None = None) -> dict | None:
    if not isinstance(tpl, dict):
        return None
    photo_file_id = str(tpl.get("photo_file_id") or "").strip()
    caption = str(tpl.get("caption") or "")[:1024]
    text = str(tpl.get("text") or "")[:TELE_MSG_LIMIT]
    parse_mode = _broadcast_normalize_parse_mode(tpl.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO)
    if not photo_file_id and not text.strip():
        return None
    if photo_file_id:
        text = ""
    else:
        caption = ""
    now_iso = _sched_iso()
    tpl_id = _broadcast_template_safe_id(tpl.get("id")) or _broadcast_template_safe_id(fallback_id) or secrets.token_hex(4)
    payload = {
        "id": tpl_id,
        "title": str(tpl.get("title") or "").strip()[:BROADCAST_TEMPLATE_TITLE_MAX],
        "photo_file_id": photo_file_id or None,
        "caption": caption if photo_file_id else None,
        "text": None if photo_file_id else text,
        "parse_mode": parse_mode,
        "created_at": str(tpl.get("created_at") or now_iso),
        "updated_at": str(tpl.get("updated_at") or now_iso),
        "created_by": _broadcast_template_safe_int(tpl.get("created_by"), 0),
    }
    if not payload["title"]:
        payload["title"] = _broadcast_template_title_from_payload(payload)
    payload["fingerprint"] = _broadcast_template_fingerprint(payload)
    return payload


def _broadcast_templates_parse(raw: Any) -> list[dict]:
    try:
        parsed = _json.loads(str(raw or "[]"))
    except Exception:
        parsed = []
    if isinstance(parsed, dict):
        parsed = parsed.get("templates") or []
    if not isinstance(parsed, list):
        return []
    templates: list[dict] = []
    seen: set[str] = set()
    for item in parsed:
        tpl = _broadcast_template_normalize(item)
        if not tpl:
            continue
        tpl_id = str(tpl.get("id"))
        if tpl_id in seen:
            continue
        seen.add(tpl_id)
        templates.append(tpl)
        if len(templates) >= BROADCAST_TEMPLATE_LIBRARY_MAX:
            break
    return templates


_BROADCAST_TEMPLATES_MEMORY_JSON = "[]"


def db_broadcast_templates_fetch() -> list[dict]:
    with _BROADCAST_TEMPLATES_LOCK:
        raw = _BROADCAST_TEMPLATES_MEMORY_JSON
    if supabase:
        try:
            res = (
                supabase.table("bot_settings")
                .select("value")
                .eq("key", BROADCAST_TEMPLATES_SETTING_KEY)
                .limit(1)
                .execute()
            )
            rows = getattr(res, "data", None) or []
            if rows:
                raw = str(rows[0].get("value") or "[]")
                # Keep memory fallback warm after a successful DB read.
                with _BROADCAST_TEMPLATES_LOCK:
                    globals()["_BROADCAST_TEMPLATES_MEMORY_JSON"] = raw
        except Exception as exc:
            logger.warning("broadcast template fetch fallback: %s", exc)
    return _broadcast_templates_parse(raw)


def db_broadcast_templates_save_all(templates: list[dict], admin_id: int) -> tuple[bool, str]:
    global _BROADCAST_TEMPLATES_MEMORY_JSON
    normalised = []
    seen_fingerprints: set[str] = set()
    for item in templates:
        tpl = _broadcast_template_normalize(item)
        if not tpl:
            continue
        fingerprint = str(tpl.get("fingerprint") or "")
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        normalised.append(tpl)
        if len(normalised) >= BROADCAST_TEMPLATE_LIBRARY_MAX:
            break
    raw = _json.dumps(normalised, ensure_ascii=False, separators=(",", ":"))
    with _BROADCAST_TEMPLATES_LOCK:
        _BROADCAST_TEMPLATES_MEMORY_JSON = raw
    if not supabase:
        return True, "saved in memory only"
    try:
        supabase.table("bot_settings").upsert({
            "key": BROADCAST_TEMPLATES_SETTING_KEY,
            "value": raw,
            "updated_by": int(admin_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="key").execute()
        return True, "saved"
    except Exception as exc:
        logger.warning("broadcast template save failed: %s", exc)
        return False, str(exc)


def db_broadcast_template_save(payload: dict, admin_id: int, title: str | None = None) -> tuple[bool, str, dict | None]:
    now_iso = _sched_iso()
    incoming = _broadcast_template_normalize({
        **dict(payload or {}),
        "id": secrets.token_hex(4),
        "title": str(title or "").strip(),
        "created_by": int(admin_id),
        "created_at": now_iso,
        "updated_at": now_iso,
    })
    if not incoming:
        return False, "Template content is empty.", None

    fingerprint = str(incoming.get("fingerprint") or "")
    existing_templates = db_broadcast_templates_fetch()
    updated_existing = False
    templates: list[dict] = []
    for existing in existing_templates:
        if str(existing.get("fingerprint") or "") == fingerprint:
            incoming["id"] = existing.get("id") or incoming["id"]
            incoming["created_at"] = existing.get("created_at") or incoming["created_at"]
            incoming["created_by"] = existing.get("created_by") or incoming["created_by"]
            # Keep a custom old title when the admin saves the same content again
            # without providing a new title.
            if not str(title or "").strip() and existing.get("title"):
                incoming["title"] = existing.get("title")
            incoming["updated_at"] = now_iso
            updated_existing = True
            continue
        templates.append(existing)
    templates.insert(0, incoming)
    ok, info = db_broadcast_templates_save_all(templates, admin_id)
    if not ok:
        return False, info, None
    return True, "updated existing template" if updated_existing else info, incoming


def db_broadcast_template_get(template_id: str) -> dict | None:
    template_id = _broadcast_template_safe_id(template_id)
    if not template_id:
        return None
    for tpl in db_broadcast_templates_fetch():
        if str(tpl.get("id")) == template_id:
            return tpl
    return None


def db_broadcast_template_delete(template_id: str, admin_id: int) -> tuple[bool, str]:
    template_id = _broadcast_template_safe_id(template_id)
    if not template_id:
        return False, "Invalid template id."
    templates = db_broadcast_templates_fetch()
    kept = [t for t in templates if str(t.get("id")) != template_id]
    if len(kept) == len(templates):
        return False, "Template not found."
    ok, info = db_broadcast_templates_save_all(kept, admin_id)
    return ok, info


def _broadcast_sent_delete_safe_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if re.fullmatch(r"[a-f0-9]{8,16}", text) else ""


def _broadcast_sent_delete_cleanup_locked(now: float | None = None) -> None:
    now = time.monotonic() if now is None else float(now)
    for job_id, item in list(_BROADCAST_SENT_DELETE_JOBS.items()):
        if float(item.get("expires_at") or 0.0) <= now:
            _BROADCAST_SENT_DELETE_JOBS.pop(job_id, None)
    while len(_BROADCAST_SENT_DELETE_JOBS) > BROADCAST_SENT_DELETE_MAX_JOBS:
        _BROADCAST_SENT_DELETE_JOBS.popitem(last=False)


def _broadcast_sent_delete_register(admin_id: int, label: str, sent_records: list[dict]) -> str:
    clean_records: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for item in sent_records or []:
        try:
            chat_id = int(item.get("chat_id"))
            message_id = int(item.get("message_id"))
        except Exception:
            continue
        if not chat_id or not message_id:
            continue
        key = (chat_id, message_id)
        if key in seen:
            continue
        seen.add(key)
        clean_records.append({"chat_id": chat_id, "message_id": message_id})
    if not clean_records:
        return ""
    job_id = secrets.token_hex(4)
    now = time.monotonic()
    with _BROADCAST_SENT_DELETE_LOCK:
        _broadcast_sent_delete_cleanup_locked(now)
        _BROADCAST_SENT_DELETE_JOBS[job_id] = {
            "id": job_id,
            "admin_id": int(admin_id),
            "label": str(label or "Broadcast")[:80],
            "records": clean_records,
            "created_at": _sched_iso(),
            "expires_at": now + BROADCAST_SENT_DELETE_TTL_S,
        }
        _broadcast_sent_delete_cleanup_locked(now)
    return job_id


def _broadcast_sent_delete_get(job_id: str, admin_id: int, *, pop: bool = False) -> dict | None:
    job_id = _broadcast_sent_delete_safe_id(job_id)
    if not job_id:
        return None
    with _BROADCAST_SENT_DELETE_LOCK:
        _broadcast_sent_delete_cleanup_locked()
        item = _BROADCAST_SENT_DELETE_JOBS.get(job_id)
        if not item or int(item.get("admin_id") or 0) != int(admin_id):
            return None
        if pop:
            _BROADCAST_SENT_DELETE_JOBS.pop(job_id, None)
        else:
            _BROADCAST_SENT_DELETE_JOBS.move_to_end(job_id)
        copied = dict(item)
        copied["records"] = [dict(r) for r in item.get("records") or []]
        return copied


def _broadcast_sent_delete_confirm_text(job: dict | None) -> str:
    if not job:
        return (
            "⚠️ <b>Delete job មិនមានទៀតទេ</b>\n\n"
            "វាអាចផុតកំណត់ ឬ Server បាន restart។"
        )
    records = job.get("records") or []
    label = html.escape(str(job.get("label") or "Broadcast"))
    return (
        "🗑️ <b>លុបសារដែល Broadcast បានផ្ញើ</b>\n\n"
        f"Broadcast: <b>{label}</b>\n"
        f"ចំនួនសារដែលអាចលុប: <b>{len(records)}</b>\n\n"
        "⚠️ Bot នឹងព្យាយាមលុបសារដែល bot បានផ្ញើទៅអ្នកប្រើ។ "
        "សារចាស់ពេក ឬ chat ខ្លះៗអាចលុបមិនបានតាមច្បាប់ Telegram។"
    )


async def _delete_broadcast_sent_messages(bot: Any, admin_id: int, delete_job: dict) -> tuple[int, int]:
    records = list(delete_job.get("records") or [])
    total = len(records)
    label = str(delete_job.get("label") or "Broadcast")[:80]
    if not records:
        await safe_send(lambda: bot.send_message(
            chat_id=admin_id,
            text="⚠️ មិនមានសារសម្រាប់លុបទេ។",
        ))
        return (0, 0)

    progress_msg = await safe_send(lambda: bot.send_message(
        chat_id=admin_id,
        text=f"🗑️ កំពុងលុបសារ {total} ពី {html.escape(label)}...",
        parse_mode="HTML",
    ))

    deleted = failed = 0
    for idx, item in enumerate(records, start=1):
        try:
            await bot.delete_message(
                chat_id=int(item.get("chat_id")),
                message_id=int(item.get("message_id")),
            )
            deleted += 1
        except RetryAfter as exc:
            await asyncio.sleep(float(getattr(exc, "retry_after", 1) or 1) + 1)
            try:
                await bot.delete_message(
                    chat_id=int(item.get("chat_id")),
                    message_id=int(item.get("message_id")),
                )
                deleted += 1
            except Exception as retry_exc:
                failed += 1
                logger.info("broadcast sent-delete retry failed chat=%s msg=%s: %s", item.get("chat_id"), item.get("message_id"), retry_exc)
        except Exception as exc:
            failed += 1
            logger.info("broadcast sent-delete failed chat=%s msg=%s: %s", item.get("chat_id"), item.get("message_id"), exc)

        if progress_msg and (idx == total or idx % max(10, BROADCAST_SENT_DELETE_BATCH_SIZE) == 0):
            with suppress(Exception):
                pct = int(idx / total * 100) if total else 100
                await progress_msg.edit_text(
                    f"🗑️ កំពុងលុប: {pct}% ({idx}/{total})\n"
                    f"✅ លុបបាន: {deleted}  ❌ មិនបាន: {failed}"
                )
        if idx < total and BROADCAST_SENT_DELETE_DELAY_S:
            await asyncio.sleep(BROADCAST_SENT_DELETE_DELAY_S)

    report = (
        "✅ <b>លុបសារ Broadcast រួចរាល់</b>\n\n"
        f"Broadcast: <b>{html.escape(label)}</b>\n"
        f"សរុប: <b>{total}</b>\n"
        f"🗑️ លុបបាន: <b>{deleted}</b>\n"
        f"❌ លុបមិនបាន: <b>{failed}</b>"
    )
    try:
        if progress_msg:
            await progress_msg.edit_text(report, parse_mode="HTML")
        else:
            await safe_send(lambda: bot.send_message(chat_id=admin_id, text=report, parse_mode="HTML"))
    except Exception as exc:
        logger.warning("broadcast sent-delete final report failed: %s", exc)
    return (deleted, failed)


def _broadcast_template_delete_confirm_text(tpl: dict) -> str:
    title = html.escape(_broadcast_template_button_title(tpl))
    kind = "🖼️ Photo" if tpl.get("photo_file_id") else "📝 Text"
    mode = html.escape(_broadcast_parse_mode_label(tpl.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO))
    preview = html.escape(_broadcast_template_clean_preview(_broadcast_template_content(tpl), max_len=180))
    return (
        "⚠️ <b>បញ្ជាក់ការលុប Template</b>\n\n"
        f"ឈ្មោះ: <b>{title}</b>\n"
        f"ប្រភេទ: <b>{kind}</b> · Format: <b>{mode}</b>\n"
        f"Preview: <code>{preview}</code>\n\n"
        "ការលុបនេះមិនអាច Undo បានទេ។"
    )


def _broadcast_templates_panel_text(templates: list[dict], notice: str = "") -> str:
    lines: list[str] = []
    if notice:
        lines.append(f"{html.escape(notice)}\n")
    lines.extend([
        "📚 <b>បណ្ណាល័យ Broadcast Template</b>",
        "",
        "រក្សាទុកសារ Broadcast ដែលប្រើញឹកញាប់ ដើម្បីយកមកប្រើម្ដងទៀតបានលឿន។",
        f"ចំនួន Template: <b>{len(templates)}/{BROADCAST_TEMPLATE_LIBRARY_MAX}</b>",
        "",
    ])
    if not templates:
        lines.append("📭 មិនទាន់មាន Template ទេ។")
        lines.append("បង្កើត Broadcast Preview រួចចុច <b>📝 រក្សាទុក Template</b>។")
    else:
        for idx, tpl in enumerate(templates, start=1):
            kind = "🖼️ Photo" if tpl.get("photo_file_id") else "📝 Text"
            title = html.escape(_broadcast_template_button_title(tpl))
            mode = html.escape(_broadcast_parse_mode_label(tpl.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO))
            preview = html.escape(_broadcast_template_clean_preview(
                _broadcast_template_content(tpl),
                max_len=min(120, BROADCAST_TEMPLATE_PREVIEW_MAX),
            ))
            lines.append(f"{idx}. <b>{title}</b> — {kind} · {mode}")
            lines.append(f"   <code>{preview}</code>")
    lines.extend([
        "",
        "▶️ ចុច Template ដើម្បី Preview មុនផ្ញើ។",
        "🗑 ចុចលុប នឹងមាន Confirm មុនលុបពិត។",
        "🔄 Refresh ប្រើពេលមាន Admin ច្រើនកំពុងកែ Template។",
    ])
    return "\n".join(lines)


def _broadcast_recipient_estimate_sync() -> dict[str, int]:
    """Return registered/active/blocked recipient counts for Safer Broadcast V2."""
    raw_user_ids = get_all_user_ids()
    registered = len(raw_user_ids)
    blocked_ids = _web_blocked_ids_for_users(raw_user_ids) if raw_user_ids else set()
    blocked = len(blocked_ids)
    active = max(0, registered - blocked)
    return {"registered": registered, "active": active, "blocked": blocked}


def _broadcast_preview_summary(payload: dict, estimate: dict[str, int]) -> str:
    kind = "រូបភាព + Caption" if payload.get("photo_file_id") else "អត្ថបទ"
    content = payload.get("caption") if payload.get("photo_file_id") else payload.get("text")
    content, mode = _broadcast_strip_format_directive(
        content,
        payload.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO,
    )
    content = str(content or "").strip()
    chars = len(content)
    return (
        "🛡️ <b>Preview Broadcast សុវត្ថិភាព V2</b>\n\n"
        "📦 <b>មាតិកា</b>\n"
        f"ប្រភេទ: <b>{html.escape(kind)}</b>\n"
        f"Format: <b>{html.escape(_broadcast_parse_mode_label(mode))}</b>\n"
        f"ចំនួនតួអក្សរ: <b>{chars}</b>\n\n"
        "📊 <b>អ្នកទទួល</b>\n"
        f"👥 អ្នកប្រើបានចុះឈ្មោះ: <b>{int(estimate.get('registered') or 0)}</b>\n"
        f"🎯 គោលដៅសកម្ម: <b>{int(estimate.get('active') or 0)}</b>\n"
        f"🚫 រំលង Blocked/Unreachable: <b>{int(estimate.get('blocked') or 0)}</b>\n"
        f"⚙️ Batch: <b>{_run_state_broadcast_batch_size()}</b> · ពន្យារ: <b>{_run_state_broadcast_delay():g}s</b>\n\n"
        "✅ សូមពិនិត្យ Preview ខាងក្រោមជាមុនសិន មុនចុចបញ្ជាក់។ "
        "អ្នកដែល Block/Unreachable នឹងត្រូវរំលងដោយស្វ័យប្រវត្តិ។"
    )


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
                 InlineKeyboardButton("❌ បិទ", callback_data="users_close")])
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
                 InlineKeyboardButton("❌ បិទ", callback_data="users_close")])
    return InlineKeyboardMarkup(rows)


def get_user_search_prompt_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👥 All Users", callback_data="users_page:0")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="users_close")],
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
                     InlineKeyboardButton("❌ បិទ", callback_data="history_close")])
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
         InlineKeyboardButton("❌ បិទ", callback_data="users_close")],
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
    for key in BOT_FEATURE_SETTING_KEYS:
        label = BOT_SETTING_LABELS.get(key, key)
        enabled = _setting_bool_from(settings, key, default=True)
        if key == "maintenance_mode":
            state = "ON 🛠️" if enabled else "OFF ✅"
        else:
            state = "ON ✅" if enabled else "OFF ⚠️"
        rows.append([InlineKeyboardButton(f"{label}: {state}", callback_data=f"admin_set:{key}")])

    rows.append([InlineKeyboardButton("⚡ Performance Settings", callback_data="admin_perf")])
    rows.extend([
        [InlineKeyboardButton("🔄 Refresh", callback_data="admin_settings_refresh"),
         InlineKeyboardButton("🧩 Setup SQL", callback_data="admin_settings_sql")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
    ])
    return InlineKeyboardMarkup(rows)


def get_bot_perf_settings_kb(settings: dict[str, str]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for key, spec in BOT_PERFORMANCE_SETTING_SPECS.items():
        label = str(spec.get("label", key))
        value = _setting_raw_from(settings, key, BOT_SETTING_DEFAULTS.get(key, ""))
        rows.append([InlineKeyboardButton(f"{label}: {_format_bot_setting_value(key, value)}", callback_data=f"admin_perf_set:{key}")])
    rows.extend([
        [InlineKeyboardButton("✅ Apply saved values now", callback_data="admin_perf_apply")],
        [InlineKeyboardButton("🔄 Reset to code defaults", callback_data="admin_perf_reset")],
        [InlineKeyboardButton("⬅️ Settings", callback_data="admin_settings"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
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
# ---------------------------------------------------------------------------
# Telegram broadcast formatting helpers
# ---------------------------------------------------------------------------
# Telegram supports three parse modes for Bot API messages:
#   HTML, MarkdownV2, and legacy Markdown.
#
# Older code escaped every broadcast as HTML, which made admin formatting show
# as literal text.  These helpers keep broadcasts safe while allowing:
#   - native Telegram formatting from admin messages (converted to HTML),
#   - explicit web/admin formatting mode selection,
#   - first-line format directives, e.g. ::html, ::mdv2, ::md, ::plain,
#   - automatic parse-mode retry with a plain-text fallback.
_BROADCAST_PARSE_MODE_AUTO = "auto"
_BROADCAST_PARSE_MODE_PLAIN = "plain"
_BROADCAST_PARSE_MODE_HTML = "html"
_BROADCAST_PARSE_MODE_MARKDOWN = "markdown"
_BROADCAST_PARSE_MODE_MARKDOWN_V2 = "markdownv2"

_BROADCAST_PARSE_MODE_ALIASES = {
    "": _BROADCAST_PARSE_MODE_AUTO,
    "auto": _BROADCAST_PARSE_MODE_AUTO,
    "detect": _BROADCAST_PARSE_MODE_AUTO,
    "plain": _BROADCAST_PARSE_MODE_PLAIN,
    "text": _BROADCAST_PARSE_MODE_PLAIN,
    "none": _BROADCAST_PARSE_MODE_PLAIN,
    "html": _BROADCAST_PARSE_MODE_HTML,
    "htm": _BROADCAST_PARSE_MODE_HTML,
    "md": _BROADCAST_PARSE_MODE_MARKDOWN,
    "markdown": _BROADCAST_PARSE_MODE_MARKDOWN,
    "markdown1": _BROADCAST_PARSE_MODE_MARKDOWN,
    "legacy_markdown": _BROADCAST_PARSE_MODE_MARKDOWN,
    "markdownv2": _BROADCAST_PARSE_MODE_MARKDOWN_V2,
    "markdown_v2": _BROADCAST_PARSE_MODE_MARKDOWN_V2,
    "mdv2": _BROADCAST_PARSE_MODE_MARKDOWN_V2,
    "v2": _BROADCAST_PARSE_MODE_MARKDOWN_V2,
}

_BROADCAST_PARSE_MODE_TO_TELEGRAM = {
    _BROADCAST_PARSE_MODE_HTML: "HTML",
    _BROADCAST_PARSE_MODE_MARKDOWN: "Markdown",
    _BROADCAST_PARSE_MODE_MARKDOWN_V2: "MarkdownV2",
    _BROADCAST_PARSE_MODE_PLAIN: None,
}

_BROADCAST_FORMAT_DIRECTIVE_RE = re.compile(
    r"^\s*(?:"
    r"(?:(?:::|//|#)\s*(?P<short>auto|plain|text|none|html|htm|markdownv2|markdown_v2|mdv2|v2|markdown|md))"
    r"|(?:(?:parse[_ -]?mode|format|mode)\s*[:=]\s*(?P<long>auto|plain|text|none|html|htm|markdownv2|markdown_v2|mdv2|v2|markdown|md))"
    r")\s*(?:-->)?\s*(?:\r?\n|$)",
    re.IGNORECASE,
)

_BROADCAST_HTML_HINT_RE = re.compile(
    r"</?(?:b|strong|i|em|u|ins|s|strike|del|a|code|pre|tg-spoiler|blockquote|tg-emoji)\b",
    re.IGNORECASE,
)
_BROADCAST_MD_V2_HINT_RE = re.compile(r"(\|\|.+?\|\||__.+?__|~.+?~)", re.DOTALL)
_BROADCAST_MD_HINT_RE = re.compile(r"(\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`|\[[^\]\n]+\]\([^)]+\))", re.DOTALL)


def _broadcast_normalize_parse_mode(mode: str | None, default: str = _BROADCAST_PARSE_MODE_AUTO) -> str:
    key = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _BROADCAST_PARSE_MODE_ALIASES:
        return _BROADCAST_PARSE_MODE_ALIASES[key]
    return _BROADCAST_PARSE_MODE_ALIASES.get(str(default or "").strip().lower(), _BROADCAST_PARSE_MODE_AUTO)


def _broadcast_strip_format_directive(
    text: str | None,
    default_mode: str | None = _BROADCAST_PARSE_MODE_AUTO,
) -> tuple[str, str]:
    raw = str(text or "")
    mode = _broadcast_normalize_parse_mode(default_mode)
    match = _BROADCAST_FORMAT_DIRECTIVE_RE.match(raw)
    if match:
        mode = _broadcast_normalize_parse_mode(match.group("short") or match.group("long"), mode)
        raw = raw[match.end():]
    return raw, mode


def _broadcast_apply_format_directive(text: str | None, parse_mode: str | None) -> str:
    """Encode a web-selected parse mode in saved scheduled-broadcast text.

    This avoids a DB migration for scheduled_broadcasts.  When the scheduler
    fires later, _broadcast_strip_format_directive removes the first line and
    uses it as the parse mode.
    """
    raw = str(text or "").strip()
    mode = _broadcast_normalize_parse_mode(parse_mode)
    if not raw or mode in {_BROADCAST_PARSE_MODE_AUTO, _BROADCAST_PARSE_MODE_PLAIN}:
        return raw
    existing_text, existing_mode = _broadcast_strip_format_directive(raw, _BROADCAST_PARSE_MODE_AUTO)
    if existing_mode != _BROADCAST_PARSE_MODE_AUTO and existing_text != raw:
        return raw
    directive = {
        _BROADCAST_PARSE_MODE_HTML: "::html",
        _BROADCAST_PARSE_MODE_MARKDOWN: "::md",
        _BROADCAST_PARSE_MODE_MARKDOWN_V2: "::mdv2",
    }.get(mode)
    return f"{directive}\n{raw}" if directive else raw


def _broadcast_parse_mode_label(mode: str | None) -> str:
    mode = _broadcast_normalize_parse_mode(mode)
    return {
        _BROADCAST_PARSE_MODE_AUTO: "Auto",
        _BROADCAST_PARSE_MODE_PLAIN: "Plain",
        _BROADCAST_PARSE_MODE_HTML: "HTML",
        _BROADCAST_PARSE_MODE_MARKDOWN: "Markdown",
        _BROADCAST_PARSE_MODE_MARKDOWN_V2: "MarkdownV2",
    }.get(mode, "Auto")


def _broadcast_parse_mode_select_html(name: str = "parse_mode", selected: str | None = "auto") -> str:
    selected = _broadcast_normalize_parse_mode(selected)
    options = [
        (_BROADCAST_PARSE_MODE_AUTO, "Auto detect"),
        (_BROADCAST_PARSE_MODE_HTML, "Telegram HTML"),
        (_BROADCAST_PARSE_MODE_MARKDOWN_V2, "MarkdownV2"),
        (_BROADCAST_PARSE_MODE_MARKDOWN, "Markdown"),
        (_BROADCAST_PARSE_MODE_PLAIN, "Plain text"),
    ]
    return "".join(
        f"<option value='{_web_h(value)}' {'selected' if value == selected else ''}>{_web_h(label)}</option>"
        for value, label in options
    )


def _broadcast_candidate_parse_modes(text: str, requested_mode: str | None) -> list[str | None]:
    """Return parse modes to try, ordered from richest to safest.

    Plain text is always the final fallback.  This protects large broadcasts
    from failing completely because one admin message contains malformed markup.
    """
    text = str(text or "")
    requested = _broadcast_normalize_parse_mode(requested_mode)
    candidates: list[str | None] = []

    if requested == _BROADCAST_PARSE_MODE_PLAIN:
        candidates = [None]
    elif requested in _BROADCAST_PARSE_MODE_TO_TELEGRAM:
        candidates = [_BROADCAST_PARSE_MODE_TO_TELEGRAM[requested], None]
    else:
        if _BROADCAST_HTML_HINT_RE.search(text):
            candidates.extend(["HTML"])
        if _BROADCAST_MD_V2_HINT_RE.search(text):
            candidates.extend(["MarkdownV2", "Markdown"])
        elif _BROADCAST_MD_HINT_RE.search(text):
            candidates.extend(["Markdown", "MarkdownV2"])
        candidates.append(None)

    deduped: list[str | None] = []
    for mode in candidates:
        if mode not in deduped:
            deduped.append(mode)
    return deduped or [None]


def _is_telegram_parse_error(exc_or_text: Any) -> bool:
    low = str(exc_or_text or "").lower()
    return (
        "can't parse entities" in low
        or "can't find end" in low
        or ("entity" in low and "parse" in low)
        or "unsupported start tag" in low
        or ("tag" in low and "not found" in low)
        or "can't parse" in low
    )


def _broadcast_visible_len(text: str, parse_mode: str | None) -> int:
    raw = str(text or "")
    mode = _broadcast_normalize_parse_mode(parse_mode)
    if mode == _BROADCAST_PARSE_MODE_HTML:
        without_tags = re.sub(r"<[^>]*>", "", raw)
        return len(html.unescape(without_tags))
    return len(raw)


def _broadcast_prepare_text(
    text: str | None,
    default_parse_mode: str | None = "auto",
    *,
    max_chars: int,
) -> tuple[str | None, str]:
    raw, mode = _broadcast_strip_format_directive(text, default_parse_mode)
    raw = raw.strip()
    if not raw:
        return None, mode
    if _broadcast_visible_len(raw, mode) > max_chars:
        raise ValueError(f"Broadcast content too long. Max {max_chars} Telegram-visible characters.")
    return raw, mode


def _broadcast_message_text_and_mode(msg: Any, *, caption: bool = False) -> tuple[str, str]:
    """Read admin Telegram message content while preserving native formatting.

    Telegram native formatting arrives as entities, not literal markdown.  PTB's
    text_html/caption_html properties convert those entities to Telegram HTML,
    so immediate and scheduled broadcasts keep bold/italic/code/links/etc.
    """
    raw = (getattr(msg, "caption", None) if caption else getattr(msg, "text", None)) or ""
    entities = (getattr(msg, "caption_entities", None) if caption else getattr(msg, "entities", None)) or []
    if entities:
        prop_name = "caption_html" if caption else "text_html"
        try:
            html_text = getattr(msg, prop_name, None)
            if html_text:
                return str(html_text), _BROADCAST_PARSE_MODE_HTML
        except Exception as exc:
            logger.warning("Could not convert Telegram entities to HTML for broadcast: %s", exc)
    clean, mode = _broadcast_strip_format_directive(raw, _BROADCAST_PARSE_MODE_AUTO)
    return clean, mode


async def _send_telegram_broadcast_message(
    bot: Any,
    *,
    chat_id: int,
    text: str,
    parse_mode: str | None,
    photo_file_id: str | None = None,
    reply_markup: Any | None = None,
) -> Any:
    """Send a text/photo broadcast with parse-mode fallback and return Telegram Message."""
    parse_candidates = _broadcast_candidate_parse_modes(text, parse_mode)
    last_parse_error: Exception | None = None

    for telegram_parse_mode in parse_candidates:
        try:
            if photo_file_id:
                kwargs = {
                    "chat_id": int(chat_id),
                    "photo": photo_file_id,
                    "caption": text if text else None,
                }
                if telegram_parse_mode and text:
                    kwargs["parse_mode"] = telegram_parse_mode
                if reply_markup is not None:
                    kwargs["reply_markup"] = reply_markup
                return await bot.send_photo(**kwargs)
            else:
                kwargs = {
                    "chat_id": int(chat_id),
                    "text": text or " ",
                    "disable_web_page_preview": True,
                }
                if telegram_parse_mode:
                    kwargs["parse_mode"] = telegram_parse_mode
                if reply_markup is not None:
                    kwargs["reply_markup"] = reply_markup
                return await bot.send_message(**kwargs)
        except BadRequest as exc:
            if telegram_parse_mode and _is_telegram_parse_error(exc):
                last_parse_error = exc
                logger.info(
                    "Telegram parse mode %s rejected for chat_id=%s; retrying fallback: %s",
                    telegram_parse_mode,
                    chat_id,
                    str(exc)[:180],
                )
                continue
            raise

    if last_parse_error:
        raise last_parse_error
    raise BadRequest("Broadcast send failed without a Telegram response.")


def _safe_broadcast_html(text: str | None, max_chars: int) -> str | None:
    """Escape broadcast content for admin previews only."""
    raw, _mode = _broadcast_strip_format_directive(text, _BROADCAST_PARSE_MODE_PLAIN)
    raw = raw.strip()
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

    loop = asyncio.get_running_loop()
    raw_user_ids = await loop.run_in_executor(_DB_EXECUTOR, get_all_user_ids)

    # Do not keep retrying users already known as unreachable/blocked.
    # This prevents scheduled broadcasts from repeatedly logging "Chat not found"
    # for the same invalid Telegram IDs.
    existing_blocked_ids = await loop.run_in_executor(
        _DB_EXECUTOR, functools.partial(_web_blocked_ids_for_users, raw_user_ids)
    )
    user_ids = [int(uid) for uid in raw_user_ids if int(uid) not in existing_blocked_ids]
    total_registered = len(raw_user_ids)
    total = len(user_ids)

    if total_registered == 0:
        await safe_send(lambda: bot.send_message(
            chat_id=admin_id,
            text=f"⚠️ {label}: មិនមានអ្នកប្រើប្រាស់ registered ណាមួយទេ។",
        ))
        return (0, 0, 0)

    sent = failed = 0
    sent_records: list[dict] = []
    blocked = max(0, total_registered - total)

    if total == 0:
        await safe_send(lambda: bot.send_message(
            chat_id=admin_id,
            text=f"⚠️ {label}: អ្នកប្រើប្រាស់ទាំងអស់ស្ថិតក្នុង blocked/unreachable list។",
        ))
        return (0, 0, blocked)
    photo_file_id = pending.get("photo_file_id")
    default_parse_mode = pending.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO
    try:
        if photo_file_id:
            send_text, send_parse_mode = _broadcast_prepare_text(
                pending.get("caption"),
                default_parse_mode,
                max_chars=1024,
            )
            send_text = send_text or ""
        else:
            send_text, send_parse_mode = _broadcast_prepare_text(
                pending.get("text"),
                default_parse_mode,
                max_chars=TELE_MSG_LIMIT,
            )
    except ValueError as exc:
        await safe_send(lambda: bot.send_message(
            chat_id=admin_id,
            text=f"❌ {label}: {html.escape(str(exc))}",
            parse_mode="HTML",
        ))
        return (0, 0, 0)

    if not photo_file_id and not send_text:
        await safe_send(lambda: bot.send_message(
            chat_id=admin_id, text=f"❌ {label}: Broadcast text is empty."
        ))
        return (0, 0, 0)

    progress_msg = await safe_send(lambda: bot.send_message(
        chat_id=admin_id,
        text=(
            f"📡 {label} — កំពុង Broadcast ទៅ {total} active user(s)...\n"
            f"👥 Registered: {total_registered}  🚫 Skipped blocked: {blocked}"
        )
    ))

    async def _send_one(uid: int) -> tuple[str, dict | None]:
        async with broadcast_semaphore:
            for attempt in range(2):
                try:
                    sent_message = await _send_telegram_broadcast_message(
                        bot,
                        chat_id=uid,
                        text=send_text or "",
                        parse_mode=send_parse_mode,
                        photo_file_id=photo_file_id,
                    )
                    message_id = int(getattr(sent_message, "message_id", 0) or 0)
                    record = {"chat_id": int(uid), "message_id": message_id} if message_id else None
                    return "sent", record
                except Forbidden as e:
                    await loop.run_in_executor(
                        None,
                        functools.partial(
                            db_user_set_blocked,
                            int(uid),
                            int(admin_id),
                            True,
                            f"Telegram Forbidden during broadcast: {str(e)[:180]}",
                        ),
                    )
                    return "blocked", None
                except RetryAfter as e:
                    await asyncio.sleep(e.retry_after + 1)
                    if attempt == 1:
                        return "failed", None
                except BadRequest as e:
                    low = str(e).lower()
                    if (
                        "chat not found" in low
                        or "user is deactivated" in low
                        or "bot can't initiate conversation" in low
                        or "bot can’t initiate conversation" in low
                    ):
                        logger.info(f"{label}: marking unreachable uid={uid} as blocked: {e}")
                        await loop.run_in_executor(
                            None,
                            functools.partial(
                                db_user_set_blocked,
                                int(uid),
                                int(admin_id),
                                True,
                                f"Telegram unreachable during broadcast: {str(e)[:180]}",
                            ),
                        )
                        return "blocked", None
                    logger.error(f"{label} Telegram BadRequest uid={uid}: {e}")
                    return "failed", None
                except Exception as e:
                    logger.error(f"{label} error uid={uid} attempt={attempt}: {e}")
                    if attempt == 1:
                        return "failed", None
                    await asyncio.sleep(0.5 * (attempt + 1))
        return "failed", None

    batch_size = max(1, _run_state_broadcast_batch_size())
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
                continue

            status = result
            record = None
            if isinstance(result, tuple):
                status = result[0]
                record = result[1] if len(result) > 1 else None

            if status == "sent":
                sent += 1
                if isinstance(record, dict):
                    sent_records.append(record)
            elif status == "blocked":
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
            await asyncio.sleep(max(0.0, _run_state_broadcast_delay()))

    delete_job_id = _broadcast_sent_delete_register(admin_id, label, sent_records)
    delete_note = (
        "\n\n🗑️ ចង់លុបសារដែលបានផ្ញើទៅអ្នកប្រើវិញ សូមចុចប៊ូតុងខាងក្រោម។"
        if delete_job_id else ""
    )
    report = (
        f"✅ <b>{html.escape(label)}</b> រួចរាល់!\n\n"
        f"👥 Registered: {total_registered}\n"
        f"🎯 Active target: {total}\n"
        f"📨 បានផ្ញើ: {sent}\n"
        f"🚫 Blocked/unreachable: {blocked}\n"
        f"❌ Failed: {failed}"
        f"{delete_note}"
    )
    reply_markup = get_broadcast_sent_delete_kb(delete_job_id) if delete_job_id else None
    try:
        if progress_msg:
            await safe_send(lambda: progress_msg.edit_text(report, parse_mode="HTML", reply_markup=reply_markup))
        else:
            await safe_send(lambda: bot.send_message(chat_id=admin_id, text=report, parse_mode="HTML", reply_markup=reply_markup))
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
        "🛡️ <b>Broadcast សុវត្ថិភាព V2</b>\n\n"
        "📨 <b>របៀបប្រើ</b>\n"
        "• ផ្ញើ <b>អត្ថបទ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Broadcast\n"
        "• Bot នឹងបង្ហាញ Preview ចុងក្រោយ មុនពេលផ្ញើពិត\n"
        "• គាំទ្រ Telegram formatting, HTML, MarkdownV2, Markdown និង Plain text\n"
        "• បើចង់បង្ខំ Format សូមដាក់ <code>::html</code>, <code>::mdv2</code>, <code>::md</code>, ឬ <code>::plain</code> នៅជួរទី១\n\n"
        "🔐 <b>ការពារ</b>\n"
        "✅ Preview confirmation មុនផ្ញើ\n"
        "✅ រំលងអ្នកដែល Block/Unreachable\n"
        "✅ គ្រប់គ្រងល្បឿនផ្ញើ និង RetryAfter\n\n"
        "វាយ /cancel ដើម្បីបោះបង់។",
        parse_mode="HTML",
        reply_markup=get_broadcast_entry_kb(),
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
            if supabase:
                res = db_call_sync(
                    "admin_total_users_count",
                    lambda: (
                        supabase.table("user_prefs")
                        .select("user_id", count="exact")
                        .limit(1)
                        .execute()
                    ),
                    default=None,
                    attempts=2,
                    critical=False,
                )
                counts["total_users"] = int(getattr(res, "count", None) or 0)
        except Exception as e:
            logger.warning(f"admin user count failed: {e}")
        try:
            if supabase:
                res = db_call_sync(
                    f"admin_pending_sched_count:{admin_id}",
                    lambda: (
                        supabase.table("scheduled_broadcasts")
                        .select("id", count="exact")
                        .eq("admin_id", int(admin_id))
                        .eq("status", SCHED_STATUS_PENDING)
                        .limit(1)
                        .execute()
                    ),
                    default=None,
                    attempts=2,
                    critical=False,
                )
                counts["pending_sched"] = int(getattr(res, "count", None) or 0)
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


def _strip_html_tags(value: Any) -> str:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    return html.unescape(text)


def _pdf_safe_line(value: Any, limit: int = 110) -> str:
    text = re.sub(r"\s+", " ", _strip_html_tags(value)).strip()
    text = text.encode("latin-1", "replace").decode("latin-1")
    return text[:limit]


def _minimal_pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_minimal_pdf(path: str, lines: list[str]) -> None:
    """Dependency-free fallback PDF writer used when reportlab is unavailable."""
    y = 800
    content_parts = ["BT", "/F1 11 Tf", "50 820 Td"]
    first = True
    for raw in lines:
        line = _pdf_safe_line(raw, 100)
        if not first:
            content_parts.append("0 -15 Td")
        first = False
        content_parts.append(f"({_minimal_pdf_escape(line)}) Tj")
        y -= 15
        if y < 60:
            break
    content_parts.append("ET")
    stream = "\n".join(content_parts).encode("latin-1", "replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(stream)).encode() + b" >> stream\n" + stream + b"\nendstream endobj\n",
    ]
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets = [0]
        for obj in objects:
            offsets.append(f.tell())
            f.write(obj)
        xref_at = f.tell()
        f.write(f"xref\n0 {len(objects)+1}\n".encode())
        f.write(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            f.write(f"{off:010d} 00000 n \n".encode())
        f.write(f"trailer << /Root 1 0 R /Size {len(objects)+1} >>\nstartxref\n{xref_at}\n%%EOF\n".encode())


def _admin_report_local_midnight(day_value: Any) -> datetime:
    """Return a timezone-aware Phnom Penh local midnight for a date-like value."""
    if isinstance(day_value, datetime):
        day_value = _to_local_time(day_value).date()
    return datetime(day_value.year, day_value.month, day_value.day, tzinfo=APP_TIMEZONE)


def _admin_report_range_from_key(range_key: str) -> dict[str, Any]:
    """Resolve Telegram/web report range keys to exact local and UTC bounds."""
    key = str(range_key or "today").strip().lower()
    today = _local_now().date()
    if key in {"yesterday", "prev", "previous"}:
        start_day = today - timedelta(days=1)
        end_day = today
        title = "Yesterday"
    elif key in {"last7", "7", "7d", "week"}:
        start_day = today - timedelta(days=6)
        end_day = today + timedelta(days=1)
        title = "Last 7 days"
    elif key in {"last30", "30", "30d", "month"}:
        start_day = today - timedelta(days=29)
        end_day = today + timedelta(days=1)
        title = "Last 30 days"
    else:
        start_day = today
        end_day = today + timedelta(days=1)
        title = "Today"

    start_local = _admin_report_local_midnight(start_day)
    end_local = _admin_report_local_midnight(end_day)
    return _admin_report_range_payload(title, start_local, end_local, key)


def _admin_report_range_payload(title: str, start_local: datetime, end_local: datetime, range_key: str = "custom") -> dict[str, Any]:
    start_local = _to_local_time(start_local)
    end_local = _to_local_time(end_local)
    if end_local <= start_local:
        end_local = start_local + timedelta(days=1)
    day_count = max(1, (end_local.date() - start_local.date()).days)
    if day_count == 1:
        range_label = f"{title} - {start_local.strftime('%Y-%m-%d')} {APP_TIMEZONE_ALIAS}"
    else:
        inclusive_end = (end_local - timedelta(seconds=1)).date()
        range_label = f"{title} - {start_local.date().isoformat()} to {inclusive_end.isoformat()} {APP_TIMEZONE_ALIAS}"
    return {
        "key": str(range_key or "custom"),
        "title": str(title or "Custom day"),
        "label": range_label,
        "days": day_count,
        "start_local": start_local,
        "end_local": end_local,
        "start_utc": _local_to_utc(start_local),
        "end_utc": _local_to_utc(end_local),
    }


def _admin_report_parse_day_text(text_value: Any) -> dict[str, Any] | None:
    """Parse admin typed report day/range text.

    Supported examples: today, yesterday, 7, 30, last 7, 2026-06-29,
    29/06/2026, and 29-06-2026.  Custom typed dates intentionally generate a
    one-day report so admins cannot accidentally request a huge DB scan.
    """
    raw = re.sub(r"\s+", " ", str(text_value or "").strip().lower())
    if not raw:
        return None
    normalized = raw.replace("days", "day").replace(" ", "")
    if normalized in {"today", "td"}:
        return _admin_report_range_from_key("today")
    if normalized in {"yesterday", "ys", "yday"}:
        return _admin_report_range_from_key("yesterday")
    if normalized in {"7", "7d", "last7", "last7day", "week"}:
        return _admin_report_range_from_key("last7")
    if normalized in {"30", "30d", "last30", "last30day", "month"}:
        return _admin_report_range_from_key("last30")

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(raw, fmt).date()
            start_local = _admin_report_local_midnight(parsed)
            return _admin_report_range_payload("Custom day", start_local, start_local + timedelta(days=1), "custom")
        except ValueError:
            continue
    return None


def _admin_report_day_counts_template(start_local: datetime, end_local: datetime) -> OrderedDict[str, dict]:
    start_day = _to_local_time(start_local).date()
    end_day = _to_local_time(end_local).date()
    days = max(1, min(366, (end_day - start_day).days))
    buckets: OrderedDict[str, dict] = OrderedDict()
    for idx in range(days):
        day_key = (start_day + timedelta(days=idx)).isoformat()
        buckets[day_key] = {"schedules": 0, "sent": 0, "failed": 0, "blocked": 0}
    return buckets


def _admin_report_activity_timeline_template(start_local: datetime, end_local: datetime) -> tuple[OrderedDict[str, dict], str]:
    """Return a chart-friendly report activity timeline.

    One-day reports use hourly buckets so the PDF graph looks like a real
    activity chart instead of a single bar/point. Multi-day reports use daily
    buckets. Values are kept simple dictionaries so the PDF renderer can draw
    a custom analytics card without adding another runtime dependency.
    """
    start_local = _to_local_time(start_local)
    end_local = _to_local_time(end_local)
    day_count = max(1, (end_local.date() - start_local.date()).days)
    buckets: OrderedDict[str, dict] = OrderedDict()

    if day_count <= 1:
        base = _admin_report_local_midnight(start_local.date())
        for hour in range(24):
            key = (base + timedelta(hours=hour)).strftime("%Y-%m-%dT%H")
            buckets[key] = {
                "label": f"{hour:02d}:00",
                "schedules": 0,
                "sent": 0,
                "failed": 0,
                "blocked": 0,
                "issues": 0,
            }
        return buckets, "hour"

    days = max(1, min(366, day_count))
    start_day = start_local.date()
    for idx in range(days):
        day = start_day + timedelta(days=idx)
        key = day.isoformat()
        buckets[key] = {
            "label": day.strftime("%b %d"),
            "schedules": 0,
            "sent": 0,
            "failed": 0,
            "blocked": 0,
            "issues": 0,
        }
    return buckets, "day"


def _collect_admin_report_analytics_sync(report_range: dict[str, Any]) -> dict[str, Any]:
    """Collect date-bounded scheduled-broadcast analytics for the PDF report."""
    start_local = report_range.get("start_local") or _admin_report_range_from_key("today")["start_local"]
    end_local = report_range.get("end_local") or (start_local + timedelta(days=1))
    start_utc = report_range.get("start_utc") or _local_to_utc(start_local)
    end_utc = report_range.get("end_utc") or _local_to_utc(end_local)
    day_counts = _admin_report_day_counts_template(start_local, end_local)
    activity_timeline, activity_timeline_mode = _admin_report_activity_timeline_template(start_local, end_local)

    empty = {
        "rows": [],
        "status_counts": {},
        "day_counts": day_counts,
        "activity_timeline": list(activity_timeline.values()),
        "activity_timeline_mode": activity_timeline_mode,
        "total_schedules": 0,
        "sent": 0,
        "failed": 0,
        "blocked": 0,
        "delivery_total": 0,
        "delivery_rate": 0,
        "upcoming": 0,
        "overdue": 0,
        "previews": 0,
        "photo": 0,
        "text": 0,
        "error": "",
    }
    if not supabase:
        empty["error"] = "Supabase is not configured."
        return empty

    try:
        rows = _web_sched_fetch_analytics_window(start_utc, until_utc=end_utc, limit=50000)
    except Exception as exc:
        logger.warning("PDF report analytics fetch failed: %s", exc)
        empty["error"] = str(exc)[:240]
        return empty

    now_utc = datetime.now(timezone.utc)
    status_counts: dict[str, int] = {}
    counted_rows: list[dict] = []
    sent = failed = blocked = upcoming = overdue = previews = photo = text_count = 0

    for row in rows:
        dt = _sched_parse_iso(row.get("broadcast_at"))
        if not dt or dt < start_utc or dt >= end_utc:
            continue
        local_key = _to_local_time(dt).date().isoformat()
        bucket = day_counts.get(local_key)
        if bucket is None:
            continue

        status = _web_schedule_status_key(row)
        sent_i = _web_safe_int(row.get("sent_count"))
        failed_i = _web_safe_int(row.get("failed_count"))
        blocked_i = _web_safe_int(row.get("blocked_count"))
        local_dt = _to_local_time(dt)
        timeline_key = local_dt.strftime("%Y-%m-%dT%H") if activity_timeline_mode == "hour" else local_dt.date().isoformat()
        timeline_bucket = activity_timeline.get(timeline_key)

        counted_rows.append(row)
        status_counts[status] = status_counts.get(status, 0) + 1
        sent += sent_i
        failed += failed_i
        blocked += blocked_i
        bucket["schedules"] += 1
        bucket["sent"] += sent_i
        bucket["failed"] += failed_i
        bucket["blocked"] += blocked_i
        if timeline_bucket is not None:
            timeline_bucket["schedules"] += 1
            timeline_bucket["sent"] += sent_i
            timeline_bucket["failed"] += failed_i
            timeline_bucket["blocked"] += blocked_i
            timeline_bucket["issues"] += failed_i + blocked_i

        if row.get("photo_file_id"):
            photo += 1
        else:
            text_count += 1
        if status == "preview":
            previews += 1
        if status == SCHED_STATUS_PENDING:
            if dt >= now_utc:
                upcoming += 1
            else:
                overdue += 1

    delivery_total = sent + failed + blocked
    return {
        "rows": counted_rows,
        "status_counts": status_counts,
        "day_counts": day_counts,
        "activity_timeline": list(activity_timeline.values()),
        "activity_timeline_mode": activity_timeline_mode,
        "total_schedules": len(counted_rows),
        "sent": sent,
        "failed": failed,
        "blocked": blocked,
        "delivery_total": delivery_total,
        "delivery_rate": _web_pct(sent, delivery_total),
        "upcoming": upcoming,
        "overdue": overdue,
        "previews": previews,
        "photo": photo,
        "text": text_count,
        "error": "",
    }


def _collect_admin_report_data_sync(admin_id: int, report_range: dict[str, Any] | None = None) -> dict[str, Any]:
    counts = {
        "users": 0,
        "blocked": 0,
        "schedules": 0,
        "pending": 0,
        "sending": 0,
        "failed": 0,
        "api_keys": 0,
    }
    with suppress(Exception):
        counts.update(_web_counts(force=True))

    settings: dict[str, str] = {}
    settings_status: dict[str, Any] = {"db_ok": False}
    with suppress(Exception):
        settings, settings_status = db_bot_settings_fetch_all()

    temp_dir = ""
    temp_ok = False
    temp_count = 0
    with suppress(Exception):
        temp_dir = _get_temp_dir()
        temp_ok = os.path.isdir(temp_dir) and os.access(temp_dir, os.W_OK)
        temp_count = sum(1 for entry in os.scandir(temp_dir) if entry.name.startswith(_TMP_PREFIX))

    ffmpeg_ok = bool(_FFMPEG_EXE and os.path.exists(_FFMPEG_EXE))
    leader = _telegram_leader_snapshot() if "_telegram_leader_snapshot" in globals() else {}
    run_state = globals().get("RUN_STATE") if isinstance(globals().get("RUN_STATE"), dict) else {}
    recent_errors = _admin_error_center_snapshot(8) if "_admin_error_center_snapshot" in globals() else []
    report_range = report_range or _admin_report_range_from_key("today")
    analytics = _collect_admin_report_analytics_sync(report_range)

    return {
        "admin_id": int(admin_id or 0),
        "generated_at": _fmt_local_dt(),
        "report_range": report_range,
        "report_range_label": report_range.get("label", "Today"),
        "analytics": analytics,
        "uptime": _format_uptime() if "_format_uptime" in globals() else "unknown",
        "counts": counts,
        "settings": settings,
        "settings_db_ok": bool(settings_status.get("db_ok")),
        "supabase_on": bool(supabase),
        "redis_on": bool(redis_client),
        "ffmpeg_ok": ffmpeg_ok,
        "temp_dir": temp_dir,
        "temp_ok": temp_ok,
        "temp_count": temp_count,
        "bot_mode": _run_state_bot_mode() if "_run_state_bot_mode" in globals() else globals().get("BOT_MODE", "unknown"),
        "webhook_ready": bool(globals().get("_TELEGRAM_APP_READY", False)),
        "leader": leader,
        "run_state": dict(run_state or {}),
        "metrics": dict(_RUNTIME_METRICS),
        "recent_errors": recent_errors,
    }


def _admin_report_lines(data: dict[str, Any]) -> list[str]:
    counts = data.get("counts") or {}
    metrics = data.get("metrics") or {}
    leader = data.get("leader") or {}
    settings = data.get("settings") or {}
    run_state = data.get("run_state") or {}
    lines = [
        "Telegram Bot Admin PDF Report",
        f"Generated: {data.get('generated_at')}",
        f"Admin ID: {data.get('admin_id')}",
        f"Report Range: {data.get('report_range_label')}",
        "",
        "System Status",
        f"Supabase: {'ON' if data.get('supabase_on') else 'OFF'}",
        f"Redis: {'ON' if data.get('redis_on') else 'OFF'}",
        f"FFmpeg: {'OK' if data.get('ffmpeg_ok') else 'ERROR'}",
        f"Temp: {'OK' if data.get('temp_ok') else 'ERROR'} ({data.get('temp_count')} files) {data.get('temp_dir')}",
        f"Bot mode: {data.get('bot_mode')}",
        f"Webhook ready: {'YES' if data.get('webhook_ready') else 'NO'}",
        f"Uptime: {data.get('uptime')}",
        f"Telegram leader owner: {leader.get('owner') or 'unknown'}",
        "",
        "Counts",
        f"Users: {counts.get('users', 0)}",
        f"Blocked users: {counts.get('blocked', 0)}",
        f"Schedules: {counts.get('schedules', 0)} | Pending: {counts.get('pending', 0)} | Sending: {counts.get('sending', 0)} | Failed: {counts.get('failed', 0)}",
        f"Active API keys: {counts.get('api_keys', 0)}",
        "",
        "Report Analytics",
        f"Schedules in range: {(data.get('analytics') or {}).get('total_schedules', 0)}",
        f"Sent: {(data.get('analytics') or {}).get('sent', 0)} | Failed: {(data.get('analytics') or {}).get('failed', 0)} | Blocked: {(data.get('analytics') or {}).get('blocked', 0)}",
        f"Delivery rate: {(data.get('analytics') or {}).get('delivery_rate', 0)}%",
        "",
        "Runtime Metrics Since Restart",
    ]
    for key in sorted(metrics):
        lines.append(f"{key}: {metrics.get(key)}")
    lines.extend(["", "Feature Settings"])
    for key in BOT_FEATURE_SETTING_KEYS:
        lines.append(f"{key}: {settings.get(key, BOT_SETTING_DEFAULTS.get(key, ''))}")
    lines.extend(["", "Performance Runtime Settings"])
    for key in BOT_PERFORMANCE_SETTING_SPECS:
        value = run_state.get(key, BOT_SETTING_DEFAULTS.get(key, _perf_default(key, "")))
        lines.append(f"{key}: {value}")
    lines.extend(["", "Recent Errors"])
    errors = data.get("recent_errors") or []
    if not errors:
        lines.append("No recent errors captured.")
    for item in errors[:8]:
        lines.append(f"{item.get('ts', '')} {item.get('level', '')} {item.get('source', '')}: {item.get('message', '')}")
    return lines


def _report_font_candidates() -> list[str]:
    """Common Unicode font paths for Linux/Render, Windows, and macOS.

    The bot does not bundle or expose font files.  If one of these fonts exists
    on the server, the PDF can render Khmer/Unicode text more cleanly; otherwise
    ReportLab safely falls back to built-in Helvetica.
    """
    return [
        os.environ.get("REPORT_PDF_FONT", ""),
        "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansKhmer-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/NotoSansKhmer-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]


def _register_report_pdf_fonts() -> tuple[str, str]:
    """Return (regular_font, bold_font) registered for the admin PDF report."""
    regular = "Helvetica"
    bold = "Helvetica-Bold"
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        for font_path in _report_font_candidates():
            font_path = str(font_path or "").strip()
            if not font_path or not os.path.exists(font_path):
                continue
            try:
                pdfmetrics.registerFont(TTFont("AdminReportUnicode", font_path))
                regular = "AdminReportUnicode"
                break
            except Exception:
                continue

        for font_path in [
            os.environ.get("REPORT_PDF_BOLD_FONT", ""),
            "/usr/share/fonts/truetype/noto/NotoSansKhmer-Bold.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansKhmer-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]:
            font_path = str(font_path or "").strip()
            if not font_path or not os.path.exists(font_path):
                continue
            try:
                pdfmetrics.registerFont(TTFont("AdminReportUnicodeBold", font_path))
                bold = "AdminReportUnicodeBold"
                break
            except Exception:
                continue
    except Exception:
        pass
    return regular, bold


def _report_pdf_text(value: Any, limit: int | None = None) -> str:
    """Clean text for ReportLab paragraphs while preserving Unicode when possible."""
    text = re.sub(r"\s+", " ", _strip_html_tags(value)).strip()
    text = text.replace("\x00", "")
    if limit and len(text) > limit:
        return text[: max(0, limit - 1)] + "…"
    return text


def _report_bool_label(ok: bool, true_text: str = "OK", false_text: str = "Attention") -> str:
    return true_text if bool(ok) else false_text


def _report_status_style(ok: bool) -> str:
    return "good" if bool(ok) else "bad"


def _write_admin_report_pdf_sync(path: str, data: dict[str, Any]) -> None:
    """Write a clean, branded, multi-page admin report PDF.

    The PDF uses ReportLab Platypus for proper wrapping, tables, page breaks,
    footer/page numbers, status cards, and a modern dashboard-like layout.  A
    tiny dependency-free PDF fallback remains available if ReportLab is missing.
    """
    lines = _admin_report_lines(data)
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.graphics.shapes import Circle, Drawing, Line, Path, Rect, String
        from reportlab.platypus import (
            BaseDocTemplate,
            Frame,
            KeepTogether,
            PageBreak,
            PageTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        regular_font, bold_font = _register_report_pdf_fonts()
        page_width, page_height = A4
        margin_x = 17 * mm
        margin_top = 22 * mm
        margin_bottom = 17 * mm
        primary = colors.HexColor("#2563EB")
        primary_dark = colors.HexColor("#1E3A8A")
        ink = colors.HexColor("#0F172A")
        muted = colors.HexColor("#64748B")
        line = colors.HexColor("#E2E8F0")
        soft = colors.HexColor("#F8FAFC")
        good = colors.HexColor("#16A34A")
        warn = colors.HexColor("#DC2626")
        amber = colors.HexColor("#D97706")

        doc = BaseDocTemplate(
            path,
            pagesize=A4,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=margin_top,
            bottomMargin=margin_bottom,
            title="Telegram Bot Admin Report",
            author="Telegram Bot Admin",
        )
        frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

        def _draw_page(canvas_obj, doc_obj):
            canvas_obj.saveState()
            canvas_obj.setFillColor(primary)
            canvas_obj.rect(0, page_height - 13 * mm, page_width, 13 * mm, fill=1, stroke=0)
            canvas_obj.setFillColor(colors.white)
            canvas_obj.setFont(bold_font, 9)
            canvas_obj.drawString(margin_x, page_height - 8.6 * mm, "Telegram Bot Admin Report")
            canvas_obj.setFont(regular_font, 8)
            canvas_obj.drawRightString(page_width - margin_x, page_height - 8.6 * mm, _report_pdf_text(data.get("generated_at"), 80))
            canvas_obj.setStrokeColor(line)
            canvas_obj.line(margin_x, 12 * mm, page_width - margin_x, 12 * mm)
            canvas_obj.setFillColor(muted)
            canvas_obj.setFont(regular_font, 8)
            footer_left = f"Admin ID: {_report_pdf_text(data.get('admin_id'), 32)} | Uptime: {_report_pdf_text(data.get('uptime'), 60)}"
            canvas_obj.drawString(margin_x, 7.5 * mm, footer_left)
            canvas_obj.drawRightString(page_width - margin_x, 7.5 * mm, f"Page {doc_obj.page}")
            canvas_obj.restoreState()

        doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_draw_page)])

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=24,
            leading=29,
            textColor=ink,
            spaceAfter=5,
        ))
        styles.add(ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["BodyText"],
            fontName=regular_font,
            fontSize=9.5,
            leading=13,
            textColor=muted,
            spaceAfter=12,
        ))
        styles.add(ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=13,
            leading=16,
            textColor=primary_dark,
            spaceBefore=12,
            spaceAfter=7,
        ))
        styles.add(ParagraphStyle(
            name="SmallMuted",
            parent=styles["BodyText"],
            fontName=regular_font,
            fontSize=8.2,
            leading=11,
            textColor=muted,
        ))
        styles.add(ParagraphStyle(
            name="TableText",
            parent=styles["BodyText"],
            fontName=regular_font,
            fontSize=8.5,
            leading=11,
            textColor=ink,
        ))
        styles.add(ParagraphStyle(
            name="TableHead",
            parent=styles["BodyText"],
            fontName=bold_font,
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        ))
        styles.add(ParagraphStyle(
            name="CardLabel",
            parent=styles["BodyText"],
            fontName=regular_font,
            fontSize=7.6,
            leading=9.5,
            textColor=muted,
            alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            name="CardValue",
            parent=styles["BodyText"],
            fontName=bold_font,
            fontSize=15,
            leading=18,
            textColor=ink,
            alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            name="StatusGood",
            parent=styles["BodyText"],
            fontName=bold_font,
            fontSize=8.5,
            leading=11,
            textColor=good,
        ))
        styles.add(ParagraphStyle(
            name="StatusBad",
            parent=styles["BodyText"],
            fontName=bold_font,
            fontSize=8.5,
            leading=11,
            textColor=warn,
        ))
        styles.add(ParagraphStyle(
            name="RightMuted",
            parent=styles["SmallMuted"],
            alignment=TA_RIGHT,
        ))

        def P(value: Any, style_name: str = "TableText", limit: int | None = None) -> Paragraph:
            return Paragraph(html.escape(_report_pdf_text(value, limit)), styles[style_name])

        def section(title: str):
            return Paragraph(html.escape(title), styles["SectionTitle"])

        def table(rows, col_widths=None, header=True, extra_style=None):
            tbl = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
            base = [
                ("BOX", (0, 0), (-1, -1), 0.55, line),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, line),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, soft]),
            ]
            if header:
                base.extend([
                    ("BACKGROUND", (0, 0), (-1, 0), primary),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ])
            if extra_style:
                base.extend(extra_style)
            tbl.setStyle(TableStyle(base))
            return tbl

        def activity_graph(analytics: dict[str, Any], report_label: str = ""):
            """Draw a modern line-chart activity card directly in the PDF.

            Style goal: similar to a clean analytics dashboard card with KPI
            values, legend dots, subtle grid lines, smooth blue/cyan lines, and
            compact range pills. This avoids image/network dependencies and is
            safe on Render because it uses only ReportLab vector shapes.
            """
            if not isinstance(analytics, dict):
                return None

            timeline = analytics.get("activity_timeline") or []
            if not timeline:
                day_counts = analytics.get("day_counts") or {}
                for day_key, bucket in list(day_counts.items())[:31]:
                    try:
                        label = datetime.strptime(str(day_key), "%Y-%m-%d").strftime("%b %d")
                    except Exception:
                        label = str(day_key)[5:] or str(day_key)
                    bucket = bucket or {}
                    timeline.append({
                        "label": label,
                        "schedules": max(0, _web_safe_int(bucket.get("schedules"))),
                        "sent": max(0, _web_safe_int(bucket.get("sent"))),
                        "failed": max(0, _web_safe_int(bucket.get("failed"))),
                        "blocked": max(0, _web_safe_int(bucket.get("blocked"))),
                        "issues": max(0, _web_safe_int(bucket.get("failed")) + _web_safe_int(bucket.get("blocked"))),
                    })
            if not timeline:
                return None

            timeline = list(timeline)[:60]
            series_a = [max(0, _web_safe_int(item.get("sent"))) for item in timeline]
            series_b = [max(0, _web_safe_int(item.get("schedules"))) for item in timeline]
            labels = [str(item.get("label") or "") for item in timeline]
            total_sent = max(0, _web_safe_int(analytics.get("sent")))
            total_schedules = max(0, _web_safe_int(analytics.get("total_schedules")))
            total_issues = max(0, _web_safe_int(analytics.get("failed")) + _web_safe_int(analytics.get("blocked")))
            if max(series_a + series_b + [total_sent, total_schedules, 1]) <= 0:
                # Keep the card visible even on quiet days, but draw flat zero lines.
                series_a = [0 for _ in timeline]
                series_b = [0 for _ in timeline]

            def compact_number(value: Any) -> str:
                n = max(0, _web_safe_int(value))
                if n >= 1_000_000:
                    return f"{n / 1_000_000:.2f}m".rstrip("0").rstrip(".")
                if n >= 1_000:
                    return f"{n / 1_000:.2f}k".rstrip("0").rstrip(".")
                return str(n)

            def nice_max(value: int) -> int:
                value = max(1, int(value or 1))
                if value <= 5:
                    return 5
                magnitude = 10 ** max(0, len(str(value)) - 1)
                for mult in (1, 2, 5, 10):
                    candidate = mult * magnitude
                    if candidate >= value:
                        return candidate
                return value

            def smooth_path(points: list[tuple[float, float]]) -> Path:
                path = Path()
                if not points:
                    return path
                path.moveTo(points[0][0], points[0][1])
                if len(points) == 1:
                    return path
                if len(points) == 2:
                    path.lineTo(points[1][0], points[1][1])
                    return path
                for i in range(len(points) - 1):
                    p0 = points[max(i - 1, 0)]
                    p1 = points[i]
                    p2 = points[i + 1]
                    p3 = points[min(i + 2, len(points) - 1)]
                    c1x = p1[0] + (p2[0] - p0[0]) / 6.0
                    c1y = p1[1] + (p2[1] - p0[1]) / 6.0
                    c2x = p2[0] - (p3[0] - p1[0]) / 6.0
                    c2y = p2[1] - (p3[1] - p1[1]) / 6.0
                    path.curveTo(c1x, c1y, c2x, c2y, p2[0], p2[1])
                return path

            width = max(440, min(540, int(doc.width)))
            height = 238
            drawing = Drawing(width, height)
            blue = colors.HexColor("#2563EB")
            cyan = colors.HexColor("#7DD3E0")
            card_bg = colors.HexColor("#FFFFFF")
            shadow = colors.HexColor("#E5EAF3")
            grid = colors.HexColor("#E5E7EB")
            axis = colors.HexColor("#D1D5DB")
            text_dark = colors.HexColor("#111827")
            text_muted = colors.HexColor("#6B7280")
            pill_border = colors.HexColor("#D7DBE2")
            pill_active_bg = colors.HexColor("#EFF6FF")
            issue_color = colors.HexColor("#F59E0B")

            # Soft dashboard card shell.
            drawing.add(Rect(3, 0, width - 6, height - 3, rx=13, ry=13, fillColor=shadow, strokeColor=None))
            drawing.add(Rect(0, 4, width - 6, height - 7, rx=13, ry=13, fillColor=card_bg, strokeColor=colors.HexColor("#EEF2F7"), strokeWidth=0.7))

            # KPI legend and numbers.
            top_y = height - 27
            drawing.add(Circle(26, top_y + 1, 3.6, fillColor=blue, strokeColor=blue))
            drawing.add(String(36, top_y - 2, "Sent", fontName=regular_font, fontSize=8.5, fillColor=text_muted))
            drawing.add(String(22, top_y - 31, compact_number(total_sent), fontName=bold_font, fontSize=25, fillColor=text_dark))

            drawing.add(Circle(144, top_y + 1, 3.6, fillColor=cyan, strokeColor=cyan))
            drawing.add(String(154, top_y - 2, "Schedules", fontName=regular_font, fontSize=8.5, fillColor=text_muted))
            drawing.add(String(140, top_y - 31, compact_number(total_schedules), fontName=bold_font, fontSize=25, fillColor=text_dark))

            # Compact range pills at top right, with selected range highlighted.
            pill_labels = ["Today", "7D", "30D"]
            label_lower = str(report_label or "").lower()
            active_idx = 0
            if "30" in label_lower:
                active_idx = 2
            elif "7" in label_lower:
                active_idx = 1
            pill_w = 35
            pill_h = 17
            start_x = width - 22 - (pill_w * len(pill_labels))
            for idx, label in enumerate(pill_labels):
                x = start_x + idx * pill_w
                active = idx == active_idx
                drawing.add(Rect(
                    x, height - 31, pill_w, pill_h,
                    rx=3.5, ry=3.5,
                    fillColor=pill_active_bg if active else colors.white,
                    strokeColor=blue if active else pill_border,
                    strokeWidth=0.8,
                ))
                drawing.add(String(x + 9, height - 25.2, label, fontName=regular_font, fontSize=7.3, fillColor=blue if active else text_muted))

            if total_issues:
                issue_text = f"Issues {compact_number(total_issues)}"
                drawing.add(Circle(width - 91, height - 49, 2.8, fillColor=issue_color, strokeColor=issue_color))
                drawing.add(String(width - 83, height - 52, issue_text, fontName=regular_font, fontSize=7.3, fillColor=text_muted))

            # Chart area.
            chart_x0 = 40
            chart_y0 = 42
            chart_w = width - 70
            chart_h = 122
            max_value = nice_max(max(series_a + series_b + [1]))
            steps = 4
            for i in range(steps + 1):
                y = chart_y0 + (chart_h / steps) * i
                drawing.add(Line(chart_x0, y, chart_x0 + chart_w, y, strokeColor=grid, strokeWidth=0.55))
                value = int(round((max_value / steps) * i))
                label = compact_number(value)
                drawing.add(String(12, y - 3, label, fontName=regular_font, fontSize=7.2, fillColor=text_muted))
            drawing.add(Line(chart_x0, chart_y0, chart_x0 + chart_w, chart_y0, strokeColor=axis, strokeWidth=0.75))

            def scale_points(values: list[int]) -> list[tuple[float, float]]:
                n = max(1, len(values) - 1)
                return [
                    (
                        chart_x0 + (chart_w * idx / n if len(values) > 1 else chart_w / 2),
                        chart_y0 + (chart_h * (max(0, float(value)) / max_value if max_value else 0)),
                    )
                    for idx, value in enumerate(values)
                ]

            blue_points = scale_points(series_a)
            cyan_points = scale_points(series_b)
            blue_path = smooth_path(blue_points)
            cyan_path = smooth_path(cyan_points)
            cyan_path.strokeColor = cyan
            cyan_path.strokeWidth = 2.15
            cyan_path.fillColor = None
            blue_path.strokeColor = blue
            blue_path.strokeWidth = 2.25
            blue_path.fillColor = None
            drawing.add(cyan_path)
            drawing.add(blue_path)

            # Minimal x-axis labels, kept readable for hourly/day timelines.
            count = len(labels)
            if count <= 1:
                tick_indexes = [0]
            elif count <= 8:
                tick_indexes = list(range(count))
            else:
                tick_indexes = sorted(set([0, count // 4, count // 2, (count * 3) // 4, count - 1]))
            n = max(1, count - 1)
            for idx in tick_indexes:
                x = chart_x0 + (chart_w * idx / n if count > 1 else chart_w / 2)
                label = labels[idx]
                if len(label) > 7:
                    label = label[:7]
                drawing.add(String(x - 10, chart_y0 - 16, label, fontName=regular_font, fontSize=7.2, fillColor=text_muted))

            # Small visual baseline ticks, like analytics dashboard activity cards.
            mini_y = 18
            tick_count = min(12, max(4, count))
            for idx in range(tick_count):
                x = chart_x0 + (chart_w * idx / max(1, tick_count - 1))
                drawing.add(Line(x - 3, mini_y, x + 3, mini_y, strokeColor=colors.HexColor("#D6D8DC"), strokeWidth=3, strokeLineCap=1))

            return drawing

        counts = data.get("counts") or {}
        metrics = data.get("metrics") or {}
        leader = data.get("leader") or {}
        settings = data.get("settings") or {}
        run_state = data.get("run_state") or {}
        recent_errors = data.get("recent_errors") or []
        analytics = data.get("analytics") or {}
        report_range_label = data.get("report_range_label") or "Today"

        story = []
        story.append(Paragraph("Telegram Bot Admin Report", styles["ReportTitle"]))
        story.append(Paragraph(
            html.escape(
                f"Generated {_report_pdf_text(data.get('generated_at'))} | "
                f"Mode: {_report_pdf_text(data.get('bot_mode'))} | "
                f"Admin ID: {_report_pdf_text(data.get('admin_id'))} | "
                f"Range: {_report_pdf_text(report_range_label, 120)}"
            ),
            styles["ReportSubtitle"],
        ))

        card_data = [[
            [P("Users", "CardLabel"), P(counts.get("users", 0), "CardValue")],
            [P("Schedules", "CardLabel"), P(counts.get("schedules", 0), "CardValue")],
            [P("Pending", "CardLabel"), P(counts.get("pending", 0), "CardValue")],
            [P("Failed", "CardLabel"), P(counts.get("failed", 0), "CardValue")],
            [P("API Keys", "CardLabel"), P(counts.get("api_keys", 0), "CardValue")],
        ]]
        card_table = Table(card_data, colWidths=[doc.width / 5.0] * 5, hAlign="LEFT")
        card_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, line),
            ("INNERGRID", (0, 0), (-1, -1), 0.6, line),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FBFF")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 8))

        graph = activity_graph(analytics, report_range_label)
        if graph is not None:
            story.append(KeepTogether([section("Activity Graph"), graph]))

        analytics_rows = [
            [P("Report Window", "TableHead"), P("Schedules", "TableHead"), P("Sent", "TableHead"), P("Failed", "TableHead"), P("Blocked", "TableHead"), P("Delivery Rate", "TableHead")],
            [
                P(report_range_label, limit=95),
                P(analytics.get("total_schedules", 0)),
                P(analytics.get("sent", 0)),
                P(analytics.get("failed", 0)),
                P(analytics.get("blocked", 0)),
                P(f"{analytics.get('delivery_rate', 0)}%"),
            ],
        ]
        if analytics.get("error"):
            analytics_rows.append([P("Data warning", "StatusBad"), P("-"), P("-"), P("-"), P("-"), P(analytics.get("error"), "StatusBad", limit=80)])
        story.append(table(analytics_rows, [doc.width * 0.34, doc.width * 0.13, doc.width * 0.13, doc.width * 0.12, doc.width * 0.12, doc.width * 0.16]))

        status_rows = [
            [P("Service", "TableHead"), P("Status", "TableHead"), P("Details", "TableHead")],
            [P("Supabase"), P(_report_bool_label(data.get("supabase_on"), "Connected", "Offline"), "StatusGood" if data.get("supabase_on") else "StatusBad"), P("Database-backed features and settings storage")],
            [P("Redis"), P(_report_bool_label(data.get("redis_on"), "Connected", "Offline"), "StatusGood" if data.get("redis_on") else "StatusBad"), P("Runtime state, locks, cache, and queue helpers")],
            [P("FFmpeg"), P(_report_bool_label(data.get("ffmpeg_ok"), "Available", "Missing"), "StatusGood" if data.get("ffmpeg_ok") else "StatusBad"), P("Video/audio processing dependency")],
            [P("Temp folder"), P(_report_bool_label(data.get("temp_ok"), "Writable", "Problem"), "StatusGood" if data.get("temp_ok") else "StatusBad"), P(f"{data.get('temp_count')} temp files | {_report_pdf_text(data.get('temp_dir'), 85)}")],
            [P("Webhook"), P(_report_bool_label(data.get("webhook_ready"), "Ready", "Not ready"), "StatusGood" if data.get("webhook_ready") else "StatusBad"), P(f"Bot mode: {_report_pdf_text(data.get('bot_mode'))} | Leader: {_report_pdf_text(leader.get('owner') or 'unknown', 70)}")],
        ]
        story.append(KeepTogether([section("System Health"), table(status_rows, [33 * mm, 31 * mm, doc.width - 64 * mm])]))

        schedule_rows = [
            [P("Metric", "TableHead"), P("Value", "TableHead"), P("Note", "TableHead")],
            [P("Blocked users"), P(counts.get("blocked", 0)), P("Users currently blocked by admin tools")],
            [P("Sending jobs"), P(counts.get("sending", 0)), P("Broadcasts or scheduled jobs currently marked sending")],
            [P("Failed jobs"), P(counts.get("failed", 0)), P("Jobs that need admin review or retry")],
            [P("Settings DB"), P("OK" if data.get("settings_db_ok") else "Fallback", "StatusGood" if data.get("settings_db_ok") else "StatusBad"), P("Whether bot settings loaded from Supabase")],
        ]
        story.append(KeepTogether([section("Operations Summary"), table(schedule_rows, [40 * mm, 28 * mm, doc.width - 68 * mm])]))

        metric_rows = [[P("Metric", "TableHead"), P("Value", "TableHead")]]
        for key in sorted(metrics):
            metric_rows.append([P(key, limit=70), P(metrics.get(key), limit=90)])
        if len(metric_rows) == 1:
            metric_rows.append([P("No metrics captured"), P("-")])
        story.append(section("Runtime Metrics Since Restart"))
        story.append(table(metric_rows, [doc.width * 0.58, doc.width * 0.42]))

        feature_rows = [[P("Feature", "TableHead"), P("Current Value", "TableHead")]]
        for key in BOT_FEATURE_SETTING_KEYS:
            feature_rows.append([P(key, limit=75), P(settings.get(key, BOT_SETTING_DEFAULTS.get(key, "")), limit=95)])
        story.append(section("Feature Settings"))
        story.append(table(feature_rows, [doc.width * 0.55, doc.width * 0.45]))

        perf_rows = [[P("Setting", "TableHead"), P("Current Value", "TableHead")]]
        for key in BOT_PERFORMANCE_SETTING_SPECS:
            value = run_state.get(key, BOT_SETTING_DEFAULTS.get(key, _perf_default(key, "")))
            perf_rows.append([P(key, limit=75), P(value, limit=95)])
        story.append(section("Performance Runtime Settings"))
        story.append(table(perf_rows, [doc.width * 0.62, doc.width * 0.38]))

        error_rows = [[P("Time", "TableHead"), P("Level", "TableHead"), P("Source", "TableHead"), P("Message", "TableHead")]]
        if not recent_errors:
            error_rows.append([P("-"), P("OK", "StatusGood"), P("System"), P("No recent errors captured.")])
        else:
            for item in recent_errors[:12]:
                level = _report_pdf_text(item.get("level", ""), 20)
                style = "StatusBad" if level.upper() in {"ERROR", "CRITICAL"} else "TableText"
                error_rows.append([
                    P(item.get("ts", ""), limit=42),
                    P(level, style),
                    P(item.get("source", ""), limit=45),
                    P(item.get("message", ""), limit=160),
                ])
        story.append(section("Recent Errors"))
        story.append(table(error_rows, [32 * mm, 21 * mm, 31 * mm, doc.width - 84 * mm]))

        story.append(Spacer(1, 10))
        story.append(Paragraph(
            html.escape("Generated automatically from the Telegram bot admin system. Keep this report private because it may include operational details."),
            styles["SmallMuted"],
        ))

        doc.build(story)
    except Exception as exc:
        logger.warning("reportlab styled PDF generation failed; using minimal PDF fallback: %s", exc)
        _write_minimal_pdf(path, lines)


def build_admin_report_pdf_sync(admin_id: int, report_range: dict[str, Any] | None = None) -> str:
    report_range = report_range or _admin_report_range_from_key("today")
    data = _collect_admin_report_data_sync(admin_id, report_range)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    range_key = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(report_range.get("key") or "report")).strip("-") or "report"
    path = os.path.join(tempfile.gettempdir(), f"bot-admin-report-{int(admin_id or 0)}-{range_key}-{stamp}.pdf")
    _write_admin_report_pdf_sync(path, data)
    return path


async def build_admin_report_pdf(admin_id: int, report_range: dict[str, Any] | None = None) -> str:
    return await asyncio.to_thread(build_admin_report_pdf_sync, int(admin_id or 0), report_range)


async def _admin_open_report_day_picker(query, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    context.user_data.pop("admin_report_state", None)
    await safe_send(lambda: query.message.edit_text(
        "📄 <b>PDF Report</b>\n\n"
        "ជ្រើសរើសថ្ងៃ ឬ Range ដែលអ្នកចង់ទាញយក Report។\n"
        "PDF នឹងមាន activity graph style ស្អាត សម្រាប់ schedules/sent/failed/blocked នៅក្នុងថ្ងៃដែលបានជ្រើស។\n\n"
        "Custom: ចុច <b>Type Date</b> ហើយវាយជា <code>YYYY-MM-DD</code> ឧ. <code>2026-06-29</code>។",
        parse_mode="HTML",
        reply_markup=get_admin_report_day_kb(),
        disable_web_page_preview=True,
    ))


async def _admin_send_report_pdf_for_range(query, context: ContextTypes.DEFAULT_TYPE, user_id: int, report_range: dict[str, Any]) -> None:
    context.user_data.pop("admin_report_state", None)
    label = str((report_range or {}).get("label") or "Selected range")
    path = ""
    progress = None
    try:
        progress = await safe_send(lambda: query.message.reply_text(
            f"⏳ Generating PDF report...\nRange: <b>{html.escape(label)}</b>",
            parse_mode="HTML",
        ))
        path = await build_admin_report_pdf(user_id, report_range)
        with open(path, "rb") as report_file:
            await context.bot.send_document(
                chat_id=user_id,
                document=report_file,
                filename=os.path.basename(path),
                caption=f"📄 Admin PDF report with activity graph\nRange: {label}",
            )
        if progress:
            with suppress(Exception):
                await progress.edit_text("✅ PDF report generated and sent.")
    except Exception as exc:
        logger.error("admin_report_pdf failed: %s", exc, exc_info=True)
        if progress:
            with suppress(Exception):
                await progress.edit_text(f"⚠️ Report generation failed: {html.escape(str(exc)[:300])}")
        else:
            await safe_send(lambda: query.message.reply_text(f"⚠️ Report generation failed: {html.escape(str(exc)[:300])}"))
    finally:
        if path:
            with suppress(Exception):
                os.remove(path)


async def _handle_admin_report_day_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    msg = update.message
    if not user or not msg or context.user_data.get("admin_report_state") != ADMIN_REPORT_WAIT_DAY:
        return False
    user_id = int(user.id)
    if not _is_admin(user_id):
        return False

    report_range = _admin_report_parse_day_text(msg.text)
    if not report_range:
        await safe_send(lambda: msg.reply_text(
            "⚠️ Date មិនត្រឹមត្រូវ។ សូមវាយជា <code>YYYY-MM-DD</code> ឧ. <code>2026-06-29</code>\n"
            "ឬវាយ <code>today</code>, <code>yesterday</code>, <code>7</code>, <code>30</code>។",
            parse_mode="HTML",
            reply_markup=get_admin_action_kb(),
        ))
        return True

    context.user_data.pop("admin_report_state", None)
    label = str(report_range.get("label") or "Selected day")
    path = ""
    progress = None
    try:
        progress = await safe_send(lambda: msg.reply_text(
            f"⏳ Generating PDF report...\nRange: <b>{html.escape(label)}</b>",
            parse_mode="HTML",
        ))
        path = await build_admin_report_pdf(user_id, report_range)
        with open(path, "rb") as report_file:
            await context.bot.send_document(
                chat_id=user_id,
                document=report_file,
                filename=os.path.basename(path),
                caption=f"📄 Admin PDF report with activity graph\nRange: {label}",
            )
        if progress:
            with suppress(Exception):
                await progress.edit_text("✅ PDF report generated and sent.")
    except Exception as exc:
        logger.error("admin_report_pdf_text_date failed: %s", exc, exc_info=True)
        if progress:
            with suppress(Exception):
                await progress.edit_text(f"⚠️ Report generation failed: {html.escape(str(exc)[:300])}")
        else:
            await safe_send(lambda: msg.reply_text(f"⚠️ Report generation failed: {html.escape(str(exc)[:300])}"))
    finally:
        if path:
            with suppress(Exception):
                os.remove(path)
    return True


async def _admin_home_text(admin_id: int, title: str = "🛠️ Admin System Center V7") -> str:
    counts = await _admin_summary_counts(admin_id)
    settings, settings_status = await get_bot_settings_async()
    ffmpeg_ok = bool(_FFMPEG_EXE and os.path.exists(_FFMPEG_EXE))
    temp_ok = False
    temp_info = ""
    try:
        temp_dir = _get_temp_dir()
        temp_ok = os.path.isdir(temp_dir) and os.access(temp_dir, os.W_OK)
        temp_count = sum(
            1
            for entry in os.scandir(temp_dir)
            if entry.name.startswith(_TMP_PREFIX)
        )
        temp_info = f"{temp_count} files"
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
    """Admin-facing whole-bot health snapshot.

    Keep this command read-only and safe to call often.  It intentionally uses
    existing runtime/cache state and short executor calls instead of changing
    external API contracts.
    """
    ffmpeg_ok = bool(_FFMPEG_EXE and os.path.exists(_FFMPEG_EXE))
    temp_dir = ""
    temp_ok = False
    try:
        temp_dir = _get_temp_dir()
        temp_ok = os.path.isdir(temp_dir) and os.access(temp_dir, os.W_OK)
    except Exception:
        temp_dir = "ERROR"

    loop = asyncio.get_running_loop()
    try:
        counts = await _admin_summary_counts(0)
    except Exception:
        counts = {}

    try:
        template_count = await loop.run_in_executor(_DB_EXECUTOR, lambda: len(_broadcast_template_load_sync()))
    except Exception:
        template_count = 0

    with _BROADCAST_SENT_DELETE_LOCK:
        delete_jobs = len(_BROADCAST_SENT_DELETE_JOBS)

    try:
        db_queue = _db_executor_queue_size()
    except Exception:
        db_queue = 0

    try:
        active_requests = int(globals().get("_WEB_ACTIVE_REQUESTS", 0) or 0)
    except Exception:
        active_requests = 0

    mode = _run_state_bot_mode() if "_run_state_bot_mode" in globals() else str(globals().get("BOT_MODE", "POLLING"))
    webhook_ready = bool(globals().get("_TELEGRAM_APP_READY"))
    redis_ok = bool(globals().get("redis_client"))
    db_ok = bool(globals().get("supabase"))
    ocr_ready = _ocr_configured()
    tts_provider = str(globals().get("TTS_PROVIDER", "auto") or "auto")

    return "\n".join([
        "🩺 <b>Whole Bot Health</b>",
        "",
        "<b>Core Services</b>",
        f"🤖 Telegram app: <b>{_ok_bad(webhook_ready, 'READY', 'BOOTING')}</b>",
        f"🌐 Runtime mode: <code>{html.escape(str(mode))}</code>",
        f"🗄️ Supabase: <b>{_ok_bad(db_ok)}</b>",
        f"🧰 Redis: <b>{_ok_bad(redis_ok)}</b>",
        f"🎧 FFmpeg: <b>{_ok_bad(ffmpeg_ok, 'OK', 'ERROR')}</b>",
        f"📁 Temp folder: <b>{_ok_bad(temp_ok, 'OK', 'ERROR')}</b> <code>{html.escape(str(temp_dir))}</code>",
        "",
        "<b>AI / TTS</b>",
        f"🗣️ TTS provider: <code>{html.escape(tts_provider)}</code>",
        f"🧠 Hugging Face: <b>{_ok_bad(bool(_hf_client))}</b>",
        f"🔍 OCR: <b>{_ok_bad(bool(ocr_ready), 'READY', 'OFF')}</b>",
        f"🔒 TTS concurrency: <b>{_run_state_max_concurrent_tts_users() if '_run_state_max_concurrent_tts_users' in globals() else MAX_CONCURRENT_TTS_USERS}</b>",
        "",
        "<b>Broadcast / Schedule</b>",
        f"👥 Users: <b>{int(counts.get('total_users') or 0)}</b> | 🚫 Blocked: <b>{int(counts.get('blocked_users') or 0)}</b>",
        f"⏰ Pending schedules: <b>{int(counts.get('pending_sched') or 0)}</b>",
        f"📚 Templates: <b>{int(template_count)}/{BROADCAST_TEMPLATE_LIBRARY_MAX}</b>",
        f"🗑️ Sent-delete jobs: <b>{int(delete_jobs)}</b>",
        f"📦 Batch: <b>{_run_state_broadcast_batch_size()}</b> · Delay: <b>{_run_state_broadcast_delay():g}s</b>",
        "",
        "<b>Runtime Pressure</b>",
        f"🌍 Active web requests: <b>{active_requests}</b>",
        f"🧵 DB executor queue: <b>{db_queue}</b>",
        f"🔐 Rate limit: <b>{_run_state_user_rate_limit()}</b> req/{_run_state_user_rate_window():g}s",
        f"📌 Scheduler lock: <b>{html.escape(str(_scheduler_lock_last_status))}</b>",
        f"🏃 Active schedule jobs: <b>{len(_scheduler_active_ids)}</b>",
        "",
        "<b>Since Restart</b>",
        f"🗣️ TTS: <b>{_RUNTIME_METRICS.get('tts', 0)}</b> | 🔍 OCR: <b>{_RUNTIME_METRICS.get('ocr', 0)}</b> | 🎙️ Voice: <b>{_RUNTIME_METRICS.get('voice', 0)}</b>",
        f"❌ Errors: <b>{_RUNTIME_METRICS.get('errors', 0)}</b> | ⛔ Blocked hits: <b>{_RUNTIME_METRICS.get('blocked_hits', 0)}</b>",
        f"⏱️ Uptime: <b>{html.escape(_format_uptime())}</b>",
    ])


async def _admin_stats_text(admin_id: int) -> str:
    counts = await _admin_summary_counts(admin_id)
    settings, status = await get_bot_settings_async()
    return (
        "📊 <b>Admin System Center V7 Stats</b>\n\n"
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


# ---------------------------------------------------------------------------
# Admin panels: TTS Provider Control, Schedule Calendar, Error Center
# ---------------------------------------------------------------------------
def _admin_back_close_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
    ])


def _tts_provider_control_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🤖 Auto", callback_data="admin_tts_set:auto"),
         InlineKeyboardButton("🌐 Edge Only", callback_data="admin_tts_set:edge")],
        [InlineKeyboardButton("🇰🇭 HF Khmer", callback_data="admin_tts_set:hf_space"),
         InlineKeyboardButton("🇰🇭 Khmer → Edge", callback_data="admin_tts_set:khmer_edge")],
        [InlineKeyboardButton("✅ Enable HF Now", callback_data="admin_tts_hf_enable"),
         InlineKeyboardButton("⏸ Disable HF 5m", callback_data="admin_tts_hf_5m")],
        [InlineKeyboardButton("🧹 Clear HF Client", callback_data="admin_tts_hf_clear"),
         InlineKeyboardButton("🔄 Refresh", callback_data="admin_tts")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
    ])


def _admin_tts_provider_text(notice: str = "") -> str:
    hf_remaining = _hf_tts_disabled_remaining_s() if "_hf_tts_disabled_remaining_s" in globals() else 0
    hf_disabled = hf_remaining > 0
    hf_client_ready = bool(globals().get("_HF_TTS_CLIENT"))
    lines = []
    if notice:
        lines.append(f"{html.escape(notice)}\n")
    lines.extend([
        "🎛 <b>TTS Provider Control</b>",
        "",
        f"Main provider: <code>{html.escape(str(TTS_PROVIDER or 'auto'))}</code>",
        f"Khmer provider: <code>{html.escape(str(KHMER_TTS_PROVIDER or 'hf_space'))}</code>",
        f"Default user model: <code>{html.escape(_normalize_tts_model(DEFAULT_TTS_MODEL))}</code>",
        f"HF Space: <code>{html.escape(str(HF_TTS_SPACE))}{html.escape(str(HF_TTS_API_NAME))}</code>",
        f"Gradio client: <b>{'READY' if GradioClient is not None else 'MISSING'}</b>",
        f"HF cached client: <b>{'YES' if hf_client_ready else 'NO'}</b>",
        f"HF cooldown: <b>{hf_remaining}s</b> {'⚠️' if hf_disabled else '✅'}",
        f"Edge fallback: <b>{'ON' if HF_TTS_EDGE_FALLBACK else 'OFF'}</b>",
        f"HF retries: <b>{HF_TTS_RETRIES}</b> · timeout: <b>{HF_TTS_TIMEOUT_S:g}s</b>",
        f"Edge retries: <b>{EDGE_TTS_RETRIES}</b> · stream timeout: <b>{EDGE_TTS_STREAM_TIMEOUT_S:g}s</b>",
        "",
        "Mode guide:",
        "• Auto: Khmer tries HF, then Edge fallback.",
        "• Edge Only: all TTS uses Microsoft Edge TTS.",
        "• HF Khmer: Khmer strongly prefers the HF Khmer Space.",
        "• Khmer → Edge: Khmer skips HF and uses Edge immediately.",
    ])
    return "\n".join(lines)


async def _admin_open_tts_provider_panel(query, notice: str = "") -> None:
    await safe_send(lambda: query.message.edit_text(
        _admin_tts_provider_text(notice),
        parse_mode="HTML",
        reply_markup=_tts_provider_control_kb(),
        disable_web_page_preview=True,
    ))


async def _admin_set_tts_provider(mode: str) -> str:
    """Apply a live TTS provider mode. Redis persistence is best-effort."""
    global TTS_PROVIDER, KHMER_TTS_PROVIDER
    mode = str(mode or "auto").strip().lower()
    if mode == "edge":
        TTS_PROVIDER = "edge"
        KHMER_TTS_PROVIDER = "edge"
        label = "Edge Only"
    elif mode == "hf_space":
        TTS_PROVIDER = "hf_space"
        KHMER_TTS_PROVIDER = "hf_space"
        label = "HF Khmer"
    elif mode == "khmer_edge":
        TTS_PROVIDER = "auto"
        KHMER_TTS_PROVIDER = "edge"
        label = "Khmer → Edge"
    else:
        TTS_PROVIDER = "auto"
        KHMER_TTS_PROVIDER = "hf_space"
        label = "Auto"

    if redis_client is not None:
        def _persist():
            pipe = redis_client.pipeline(transaction=False)
            pipe.set("RUN_STATE:TTS_PROVIDER", TTS_PROVIDER)
            pipe.set("RUN_STATE:KHMER_TTS_PROVIDER", KHMER_TTS_PROVIDER)
            pipe.execute()
            return True
        with suppress(Exception):
            await redis_call("tts_provider_runtime_persist", _persist, default=False, attempts=2, timeout=2.0, critical=False)
    logger.info("Admin TTS provider changed: provider=%s khmer=%s", TTS_PROVIDER, KHMER_TTS_PROVIDER)
    return label


def _restore_tts_provider_runtime_overrides_sync() -> None:
    """Restore live TTS provider choices from Redis if admin changed them before restart."""
    global TTS_PROVIDER, KHMER_TTS_PROVIDER
    if redis_client is None:
        return
    try:
        provider = redis_client.get("RUN_STATE:TTS_PROVIDER")
        khmer_provider = redis_client.get("RUN_STATE:KHMER_TTS_PROVIDER")
        if isinstance(provider, bytes):
            provider = provider.decode("utf-8", errors="ignore")
        if isinstance(khmer_provider, bytes):
            khmer_provider = khmer_provider.decode("utf-8", errors="ignore")
        provider = str(provider or "").strip().lower()
        khmer_provider = str(khmer_provider or "").strip().lower()
        if provider in {"auto", "edge", "edge_tts", "hf", "hf_space", "khmer_hf_space", "khmer-tts"}:
            TTS_PROVIDER = "edge" if provider == "edge_tts" else ("hf_space" if provider in {"hf", "khmer_hf_space", "khmer-tts"} else provider)
        if khmer_provider in {"edge", "edge_tts", "hf", "hf_space", "khmer_hf_space", "khmer-tts"}:
            KHMER_TTS_PROVIDER = "edge" if khmer_provider == "edge_tts" else ("hf_space" if khmer_provider in {"hf", "khmer_hf_space", "khmer-tts"} else khmer_provider)
    except Exception as exc:
        logger.warning("Could not restore TTS provider runtime override: %s", exc)


def _schedule_calendar_fetch_sync(admin_id: int, offset_days: int = 0, span_days: int = 14) -> list[dict]:
    start_local = (_local_now() + timedelta(days=int(offset_days or 0))).replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=max(1, min(31, int(span_days or 14))))
    rows = _web_sched_fetch_range(_local_to_utc(start_local), _local_to_utc(end_local), limit=1000)
    return [r for r in rows if int(r.get("admin_id") or 0) == int(admin_id)]


def _schedule_calendar_text(rows: list[dict], offset_days: int = 0, span_days: int = 14) -> str:
    start_local = (_local_now() + timedelta(days=int(offset_days or 0))).replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=max(1, min(31, int(span_days or 14))))
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for i in range((end_local.date() - start_local.date()).days):
        key = (start_local + timedelta(days=i)).date().isoformat()
        grouped[key] = []
    for row in rows:
        dt = _sched_parse_iso(row.get("broadcast_at"))
        if not dt:
            continue
        key = _to_local_time(dt).date().isoformat()
        grouped.setdefault(key, []).append(row)

    lines = [
        "🗓 <b>Scheduled Broadcast Calendar</b>",
        f"Window: <b>{html.escape(start_local.strftime('%Y-%m-%d'))}</b> → <b>{html.escape((end_local - timedelta(days=1)).strftime('%Y-%m-%d'))}</b>",
        f"Timezone: <b>{APP_TIMEZONE_ALIAS} ({APP_TIMEZONE_UTC_LABEL})</b>",
        "",
    ]
    shown = 0
    for day, items in grouped.items():
        if not items:
            lines.append(f"<b>{html.escape(day)}</b> · 0")
            continue
        lines.append(f"<b>{html.escape(day)}</b> · {len(items)}")
        for row in items[:5]:
            dt = _sched_parse_iso(row.get("broadcast_at"))
            local_time = _to_local_time(dt).strftime("%I:%M %p") if dt else "?"
            status = _sched_status_label(row)
            content = _sched_content_preview(row, limit=72)
            lines.append(
                f"  • <code>#{int(row.get('id') or 0)}</code> {html.escape(local_time)} "
                f"<b>{html.escape(status)}</b> — {html.escape(content)}"
            )
            shown += 1
        if len(items) > 5:
            lines.append(f"  … +{len(items) - 5} more")
    if shown == 0:
        lines.append("\nNo scheduled broadcasts in this window.")
    return "\n".join(lines)[:3900]


def _schedule_calendar_kb(offset_days: int = 0) -> InlineKeyboardMarkup:
    offset_days = int(offset_days or 0)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("← Previous 14d", callback_data=f"admin_calendar:{offset_days - 14}"),
         InlineKeyboardButton("Today", callback_data="admin_calendar:0"),
         InlineKeyboardButton("Next 14d →", callback_data=f"admin_calendar:{offset_days + 14}")],
        [InlineKeyboardButton("📅 New Schedule", callback_data="admin_schedule_new"),
         InlineKeyboardButton("📋 List", callback_data="admin_schedules")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
    ])


async def _admin_open_schedule_calendar_panel(query, admin_id: int, offset_days: int = 0) -> None:
    rows = await asyncio.get_running_loop().run_in_executor(
        _DB_EXECUTOR,
        lambda: _schedule_calendar_fetch_sync(admin_id, offset_days, 14),
    )
    await safe_send(lambda: query.message.edit_text(
        _schedule_calendar_text(rows, offset_days, 14),
        parse_mode="HTML",
        reply_markup=_schedule_calendar_kb(offset_days),
        disable_web_page_preview=True,
    ))


def _error_center_text() -> str:
    rows = _admin_error_center_snapshot(12)
    if not rows:
        return (
            "🚨 <b>Error Center</b>\n\n"
            "No captured runtime errors yet. New ERROR-level logs will appear here automatically.\n\n"
            f"Storage: in-memory ring buffer, max <b>{_ADMIN_ERROR_CENTER_MAX}</b> items."
        )
    lines = [
        "🚨 <b>Error Center</b>",
        f"Showing latest <b>{len(rows)}</b> error(s).",
        f"Total stored: <b>{len(_ADMIN_ERROR_CENTER)}</b> / {_ADMIN_ERROR_CENTER_MAX}",
        "",
    ]
    for idx, item in enumerate(rows, 1):
        dt = _sched_parse_iso(item.get("ts"))
        local_ts = _fmt_dt(dt) if dt else str(item.get("ts") or "")[:19]
        lines.append(
            f"{idx}. <b>{html.escape(str(item.get('level') or 'ERROR'))}</b> "
            f"<code>{html.escape(str(item.get('fingerprint') or ''))}</code>\n"
            f"   🕒 {html.escape(local_ts)}\n"
            f"   📍 {html.escape(str(item.get('source') or 'unknown'))}\n"
            f"   {html.escape(str(item.get('message') or '')[:450])}"
        )
    return "\n\n".join(lines)[:3900]


def _error_center_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh", callback_data="admin_errors"),
         InlineKeyboardButton("🧹 Clear", callback_data="admin_errors_clear")],
        [InlineKeyboardButton("🩺 Health", callback_data="admin_health"),
         InlineKeyboardButton("🎛 TTS Provider", callback_data="admin_tts")],
        [InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
         InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
    ])


async def _admin_open_error_center(query, notice: str = "") -> None:
    text = _error_center_text()
    if notice:
        text = f"{html.escape(notice)}\n\n{text}"
    await safe_send(lambda: query.message.edit_text(
        text,
        parse_mode="HTML",
        reply_markup=_error_center_kb(),
        disable_web_page_preview=True,
    ))


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
                 InlineKeyboardButton("❌ បិទ", callback_data="admin_close")],
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


async def _apply_bot_performance_setting(key: str, raw_value: Any, *, admin_id: int = 0, persist_runtime: bool = True) -> tuple[bool, str]:
    """Validate, save to bot_settings, and hot-apply supported performance knobs."""
    try:
        coerced = _coerce_bot_perf_setting(key, raw_value)
    except Exception as exc:
        return False, str(exc)

    stored = _bool_to_setting_value(coerced) if isinstance(coerced, bool) else str(coerced)
    ok, info = await asyncio.get_running_loop().run_in_executor(
        _DB_EXECUTOR,
        lambda: db_bot_setting_value_set(key, stored, admin_id),
    )
    if not ok:
        return False, info

    try:
        await _update_run_state(key, coerced, persist=persist_runtime and key in _RUN_STATE_REDIS_KEYS)
    except Exception:
        # Some startup-only settings are still saved and will apply on restart.
        globals()[key] = coerced
    _bot_settings_cache["ts"] = 0.0
    return True, "saved and applied"


async def _apply_all_bot_performance_settings(admin_id: int = 0, *, force: bool = True) -> list[str]:
    settings, _status = await get_bot_settings_async(force=force)
    changed: list[str] = []
    for key, spec in BOT_PERFORMANCE_SETTING_SPECS.items():
        raw = _setting_raw_from(settings, key, BOT_SETTING_DEFAULTS.get(key, ""))
        ok, info = await _apply_bot_performance_setting(key, raw, admin_id=admin_id)
        if ok:
            changed.append(key)
        else:
            logger.warning("Could not apply bot performance setting %s=%r: %s", key, raw, info)
    return changed


async def _admin_open_perf_settings_panel(query, force: bool = False, notice: str = "") -> None:
    settings, status = await get_bot_settings_async(force=force)
    lines = []
    if notice:
        lines.append(f"{html.escape(notice)}\n")
    lines.extend([
        "⚡ <b>Performance Settings</b>",
        "",
        "តម្លៃទាំងនេះ save ក្នុង <code>bot_settings</code> ហើយភាគច្រើនអាច apply ភ្លាមៗ។",
        f"Storage: <b>{_ok_bad(bool(status.get('db_ok')), 'Supabase', 'Memory / setup needed')}</b>",
        "",
    ])
    for key, spec in BOT_PERFORMANCE_SETTING_SPECS.items():
        label = html.escape(str(spec.get("label", key)))
        value = html.escape(_format_bot_setting_value(key, _setting_raw_from(settings, key, BOT_SETTING_DEFAULTS.get(key, ""))))
        help_text = html.escape(str(spec.get("help", "")))
        lines.append(f"{label}: <b>{value}</b> — {help_text}")
    lines.extend(["", "ចុច setting មួយ រួចផ្ញើតម្លៃថ្មី។ ឧទាហរណ៍: <code>2</code>, <code>30</code>, <code>on</code>។"])
    await safe_send(lambda: query.message.edit_text(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=get_bot_perf_settings_kb(settings),
        disable_web_page_preview=True,
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
    for key in BOT_FEATURE_SETTING_KEYS:
        label = BOT_SETTING_LABELS.get(key, key)
        enabled = _setting_bool_from(settings, key, True)
        if key == "maintenance_mode":
            state = "ON ⚠️" if enabled else "OFF ✅"
        else:
            state = "ON ✅" if enabled else "OFF ⚠️"
        desc = html.escape(BOT_SETTING_DESCRIPTIONS.get(key, ""))
        lines.append(f"{label}: <b>{state}</b> — {desc}")
    lines.extend(["", "ចុច setting ណាមួយដើម្បី ON/OFF ឬបើក ⚡ Performance Settings ដើម្បីកែ speed/cache/concurrency។"])
    await safe_send(lambda: query.message.edit_text(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=get_bot_settings_kb(settings),
        disable_web_page_preview=True,
    ))


async def _admin_send_settings_sql(message) -> None:
    for page in _paginate_pre_html(
        ADMIN_V2_TABLES_SQL,
        limit=3800,
        header="🧩 <b>Admin System Center V7 SQL</b>\n\n",
    ):
        await safe_send(lambda p=page: message.reply_text(
            p,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))


async def _admin_start_broadcast_from_button(query, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    _pending_broadcast.pop(user_id, None)
    context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
    estimate = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, _broadcast_recipient_estimate_sync)
    await safe_send(lambda: query.message.edit_text(
        "🛡️ <b>Broadcast សុវត្ថិភាព V2</b>\n\n"
        "📨 <b>របៀបប្រើ</b>\n"
        "• ផ្ញើ <b>អត្ថបទ</b> ឬ <b>រូបភាព + Caption</b>\n"
        "• Bot នឹងបង្ហាញ Preview ចុងក្រោយ មុនពេលផ្ញើពិត\n"
        "• គាំទ្រ Telegram formatting, HTML, MarkdownV2, Markdown និង Plain text\n"
        "• បើចង់បង្ខំ Format សូមដាក់ <code>::html</code>, <code>::mdv2</code>, <code>::md</code>, ឬ <code>::plain</code> នៅជួរទី១\n\n"
        "📊 <b>អ្នកទទួល</b>\n"
        f"👥 អ្នកប្រើបានចុះឈ្មោះ: <b>{int(estimate.get('registered') or 0)}</b>\n"
        f"🎯 គោលដៅសកម្ម: <b>{int(estimate.get('active') or 0)}</b>\n"
        f"🚫 រំលង Blocked/Unreachable: <b>{int(estimate.get('blocked') or 0)}</b>\n"
        f"⚙️ Batch: <b>{_run_state_broadcast_batch_size()}</b> · ពន្យារ: <b>{_run_state_broadcast_delay():g}s</b>\n\n"
        "🔐 <b>ការពារ</b>\n"
        "✅ Preview confirmation មុនផ្ញើ\n"
        "✅ រំលងអ្នកដែល Block/Unreachable\n"
        "✅ គ្រប់គ្រងល្បឿនផ្ញើ និង RetryAfter\n"
        "✅ Auto-block chat ដែលមិនអាចផ្ញើបាន\n\n"
        "ចុច <b>បោះបង់</b> ឬវាយ /cancel ដើម្បីបញ្ឈប់។",
        parse_mode="HTML",
        reply_markup=get_broadcast_entry_kb(),
    ))


async def _admin_open_broadcast_templates(query, context: ContextTypes.DEFAULT_TYPE, user_id: int, notice: str = "") -> None:
    templates = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, db_broadcast_templates_fetch)
    await safe_send(lambda: query.message.edit_text(
        _broadcast_templates_panel_text(templates, notice=notice),
        parse_mode="HTML",
        reply_markup=get_broadcast_templates_kb(templates),
        disable_web_page_preview=True,
    ))


async def _admin_show_broadcast_preview_message(message, bot, user_id: int, payload: dict) -> bool:
    estimate = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, _broadcast_recipient_estimate_sync)
    summary = _broadcast_preview_summary(payload, estimate)
    photo_file_id = payload.get("photo_file_id")
    caption_text = payload.get("caption")
    plain_text = payload.get("text")
    parse_mode = payload.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO

    preview_content, preview_mode = _broadcast_strip_format_directive(
        caption_text if photo_file_id else plain_text,
        parse_mode,
    )

    await safe_send(lambda: message.reply_text(
        f"{summary}\n\n"
        "👁️ <b>Preview មាតិកានឹងបង្ហាញខាងក្រោម</b>\n"
        "សូមពិនិត្យទម្រង់សារឲ្យបានច្បាស់ បន្ទាប់មកចុចប៊ូតុងខាងក្រោម Preview ដើម្បីផ្ញើ បញ្ចូល Template ឬបោះបង់។",
        parse_mode="HTML",
        disable_web_page_preview=True,
    ))
    try:
        await safe_send(lambda: _send_telegram_broadcast_message(
            bot,
            chat_id=message.chat_id,
            text=preview_content or "",
            parse_mode=preview_mode,
            photo_file_id=photo_file_id,
            reply_markup=get_broadcast_confirm_kb(),
        ))
        return True
    except Exception as exc:
        await safe_send(lambda: message.reply_text(
            "❌ Preview មិនជោគជ័យ ដូច្នេះ Broadcast មិនត្រូវបានដាក់ Queue ទេ។\n"
            f"មូលហេតុ: <code>{html.escape(str(exc)[:300])}</code>",
            parse_mode="HTML",
        ))
        return False


async def _admin_start_schedule_from_button(query, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    _sched_payload.pop(user_id, None)
    context.user_data["sched_state"] = SCHED_WAIT_MSG
    await safe_send(lambda: query.message.edit_text(
        "📅 <b>Scheduled Broadcast</b>\n\n"
        "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Schedule ។\n"
        "បន្ទាប់មក Bot នឹងសួរពេលវេលា Phnom Penh local time (ICT, UTC+7)។\n"
        "Formatting: Telegram native formatting, HTML, MarkdownV2, Markdown, or plain text. First-line directives: <code>::html</code>, <code>::mdv2</code>, <code>::md</code>, <code>::plain</code>.\n\n"
        "ចុច Cancel ឬវាយ /cancel ដើម្បីបោះបង់។",
        parse_mode="HTML",
        reply_markup=get_admin_action_kb(),
    ))


@admin_only
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Preserve the full Telegram Admin System Center V7 on /admin.
    # Runtime controls are now reachable from the "🛠️ Runtime" submenu button.
    user_id = update.effective_user.id if update.effective_user else 0
    text = await _admin_home_text(user_id)
    await safe_send(lambda: update.message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
        disable_web_page_preview=True,
    ))


@admin_only
async def cmd_runtime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Direct shortcut for runtime operations without replacing /admin.
    await safe_send(lambda: update.message.reply_text(
        _runtime_admin_text(),
        parse_mode="HTML",
        reply_markup=get_runtime_admin_kb(),
        disable_web_page_preview=True,
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

    parse_mode = _BROADCAST_PARSE_MODE_AUTO
    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text, parse_mode = _broadcast_message_text_and_mode(msg, caption=True)
    elif msg.text:
        plain_text, parse_mode = _broadcast_message_text_and_mode(msg, caption=False)
        if not plain_text.strip():
            await safe_send(lambda: msg.reply_text("⚠️ អត្ថបទមិនអាចទទេបាន។ សូមវាយសារ។"))
            return
    else:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return

    _pending_broadcast[user_id] = {
        "photo_file_id": photo_file_id,
        "caption":       caption_text,
        "text":          plain_text,
        "parse_mode":    parse_mode,
    }

    ok = await _admin_show_broadcast_preview_message(msg, context.bot, user_id, _pending_broadcast[user_id])
    if not ok:
        _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)


async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data or ""

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    with suppress(Exception):
        await query.answer()

    if data == "bc_templates":
        await _admin_open_broadcast_templates(query, context, user_id)
        return

    if data == "bc_save_template":
        pending = _pending_broadcast.get(user_id)
        if not pending:
            await safe_send(lambda: query.message.reply_text(
                "⚠️ មិនមាន Preview សម្រាប់ Save ទេ។ សូមផ្ញើ Broadcast មុនសិន។"
            ))
            return
        ok, info, tpl = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: db_broadcast_template_save(pending, user_id),
        )
        if ok and str(info).startswith("updated existing"):
            notice = "♻️ Template មានរួចហើយ — បាន Update និងដាក់ឡើងលើ។"
        else:
            notice = "✅ បាន Save Template។" if ok else f"⚠️ Save Template មិនជោគជ័យ: {info}"
        await _admin_open_broadcast_templates(query, context, user_id, notice=notice)
        return

    if data.startswith("bc_tpl_use:"):
        tpl_id = data.split(":", 1)[1]
        tpl = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_broadcast_template_get(tpl_id))
        if not tpl:
            await _admin_open_broadcast_templates(query, context, user_id, notice="⚠️ រក Template មិនឃើញ។")
            return
        pending = _broadcast_template_payload_from_template(tpl)
        _pending_broadcast[user_id] = pending
        context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f"📚 បានជ្រើស Template: <b>{html.escape(_broadcast_template_button_title(tpl))}</b>",
            parse_mode="HTML",
        ))
        ok = await _admin_show_broadcast_preview_message(query.message, context.bot, user_id, pending)
        if not ok:
            _pending_broadcast.pop(user_id, None)
            context.user_data.pop("bc_state", None)
        return

    if data.startswith("bc_tpl_delask:"):
        tpl_id = data.split(":", 1)[1]
        tpl = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_broadcast_template_get(tpl_id))
        if not tpl:
            await _admin_open_broadcast_templates(query, context, user_id, notice="⚠️ រក Template មិនឃើញ។")
            return
        await safe_send(lambda: query.message.edit_text(
            _broadcast_template_delete_confirm_text(tpl),
            parse_mode="HTML",
            reply_markup=get_broadcast_template_delete_confirm_kb(tpl_id),
            disable_web_page_preview=True,
        ))
        return

    if data.startswith("bc_tpl_del:"):
        tpl_id = data.split(":", 1)[1]
        ok, info = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: db_broadcast_template_delete(tpl_id, user_id),
        )
        notice = "🗑️ បានលុប Template។" if ok else f"⚠️ លុបមិនជោគជ័យ: {info}"
        await _admin_open_broadcast_templates(query, context, user_id, notice=notice)
        return

    if data.startswith("bc_del_sent_ask:"):
        job_id = data.split(":", 1)[1]
        job = _broadcast_sent_delete_get(job_id, user_id)
        await safe_send(lambda: query.message.reply_text(
            _broadcast_sent_delete_confirm_text(job),
            parse_mode="HTML",
            reply_markup=get_broadcast_sent_delete_confirm_kb(job_id) if job else None,
            disable_web_page_preview=True,
        ))
        return

    if data.startswith("bc_del_sent_run:"):
        job_id = data.split(":", 1)[1]
        job = _broadcast_sent_delete_get(job_id, user_id, pop=True)
        if not job:
            await safe_send(lambda: query.message.reply_text(
                "⚠️ Delete job មិនមានទៀតទេ។ វាអាចផុតកំណត់ ឬ server restart។"
            ))
            return
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        context.application.create_task(_delete_broadcast_sent_messages(context.bot, user_id, job))
        await safe_send(lambda: query.message.reply_text("🗑️ បានចាប់ផ្ដើមលុបសារ Broadcast ដែលបានផ្ញើ..."))
        return

    if data == "bc_del_sent_keep":
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text("✅ ទុកសារ Broadcast នៅដដែល។"))
        return

    if data == "bc_cancel":
        _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text("❌ Broadcast ត្រូវបានបោះបង់។"))
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
        "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Schedule ។\n"
        "✅ Supports Telegram native formatting, HTML, MarkdownV2, Markdown, and plain text.\n"
        "Prefix first line with <code>::html</code>, <code>::mdv2</code>, <code>::md</code>, or <code>::plain</code>.\n\n"
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

    parse_mode = _BROADCAST_PARSE_MODE_AUTO
    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text, parse_mode = _broadcast_message_text_and_mode(msg, caption=True)
    elif msg.text:
        plain_text, parse_mode = _broadcast_message_text_and_mode(msg, caption=False)
        if not plain_text.strip():
            await safe_send(lambda: msg.reply_text("⚠️ អត្ថបទមិនអាចទទេបាន។ សូមវាយសារ ឬ ផ្ញើរូបភាព។"))
            return True
    else:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return True

    _sched_payload[user_id] = {
        "photo_file_id": photo_file_id,
        "caption": caption_text,
        "text": plain_text,
        "parse_mode": parse_mode,
    }
    context.user_data["sched_state"] = SCHED_WAIT_TIME
    await safe_send(lambda: msg.reply_text(
        "🕐 <b>ពេលវេលា Broadcast — Phnom Penh local time</b>\n\n"
        "វាយកាលបរិច្ឆេទ និងម៉ោង ។\n"
        "ទម្រង់: <code>YYYY-MM-DD HH:MM AM/PM</code> ឬ <code>YYYY-MM-DD HH:MM</code>\n"
        "ម៉ោង: Phnom Penh, Cambodia — ICT (UTC+7)\n"
        "ឧទាហរណ៍: <code>2025-12-25 09:00 AM</code> ឬ <code>2025-12-25 21:00</code>",
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
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើ Text ពេលវេលា Phnom Penh local time (ICT, UTC+7)។"))
        return True

    broadcast_at = _parse_dt(msg.text)
    if broadcast_at is None:
        await safe_send(lambda: msg.reply_text(
            "❌ ទម្រង់ពេលវេលាខុស។\nឧទាហរណ៍ត្រឹមត្រូវ: <code>2025-12-25 09:00 AM</code> ឬ <code>2025-12-25 21:00</code>",
            parse_mode="HTML",
        ))
        return True

    now = datetime.now(timezone.utc)
    if broadcast_at <= now:
        await safe_send(lambda: msg.reply_text(
            "❌ ពេលវេលាត្រូវតែជាអនាគត តាមម៉ោង Phnom Penh (ICT, UTC+7)។\n"
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

    preview_content, preview_mode = _broadcast_strip_format_directive(
        payload.get("caption") if payload.get("photo_file_id") else payload.get("text"),
        payload.get("parse_mode") or _BROADCAST_PARSE_MODE_AUTO,
    )
    mode_line = f"Format: <b>{html.escape(_broadcast_parse_mode_label(preview_mode))}</b>\n"

    # Keep schedule details in a safe HTML message and render the actual
    # broadcast content separately with its selected Telegram parse mode.
    # This prevents HTML/Markdown from showing as raw tags in the preview.
    await safe_send(lambda: msg.reply_text(
        f"📅 <b>Preview Schedule #{row_id}</b>\n"
        f"⏰ {dt_str}\n"
        "ស្ថានភាព: <b>Preview — មិនទាន់បញ្ជាក់</b>\n"
        f"{mode_line}\n"
        "👁️ <b>Content preview is shown below.</b>",
        parse_mode="HTML",
        disable_web_page_preview=True,
    ))
    try:
        await safe_send(lambda: _send_telegram_broadcast_message(
            context.bot,
            chat_id=msg.chat_id,
            text=preview_content or "",
            parse_mode=preview_mode,
            photo_file_id=payload.get("photo_file_id"),
            reply_markup=get_sched_confirm_kb(row_id),
        ))
    except Exception as exc:
        await safe_send(lambda: msg.reply_text(
            "❌ Schedule was saved, but the rendered preview failed. You can cancel it from /schedules.\n"
            f"Reason: <code>{html.escape(str(exc)[:300])}</code>",
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
        "new_time_in_past": "❌ ពេលថ្មីត្រូវតែជាអនាគត តាមម៉ោង Phnom Penh (ICT, UTC+7)។",
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
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើ Text ពេលវេលា Phnom Penh local time (ICT, UTC+7)។"))
        return True
    new_dt = _parse_dt(msg.text)
    if new_dt is None:
        await safe_send(lambda: msg.reply_text(
            "❌ ទម្រង់ពេលវេលាខុស។\nឧទាហរណ៍: <code>2026-12-25 09:00 AM</code> ឬ <code>2026-12-25 21:00</code>",
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
        await _clear_admin_transient_state(context, user_id)
        with suppress(Exception):
            await query.message.delete()
        return

    if data in ("admin_home", "admin_refresh"):
        await _clear_admin_transient_state(context, user_id)
        text = await _admin_home_text(user_id)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return

    if data == "admin_cancel_state":
        cleared = await _clear_admin_transient_state(context, user_id)
        suffix = f" ({', '.join(cleared[:5])})" if cleared else ""
        text = await _admin_home_text(user_id, title=f"✅ Admin action cancelled{suffix}")
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

    if data == "admin_web_key":
        await _admin_generate_web_key(query, user_id, context)
        return

    if data == "admin_report":
        # Ask which day/range the admin wants before generating the PDF.
        await _admin_open_report_day_picker(query, context, user_id)
        return

    if data.startswith("admin_report_range:"):
        range_key = data.split(":", 1)[1].strip().lower()
        if range_key == "custom":
            context.user_data["admin_report_state"] = ADMIN_REPORT_WAIT_DAY
            await safe_send(lambda: query.message.edit_text(
                "✍️ <b>Custom PDF Report Date</b>\n\n"
                "វាយថ្ងៃដែលចង់បានជា <code>YYYY-MM-DD</code>។\n"
                "Example: <code>2026-06-29</code>\n\n"
                "You can also type: <code>today</code>, <code>yesterday</code>, <code>7</code>, or <code>30</code>.",
                parse_mode="HTML",
                reply_markup=get_admin_action_kb(),
                disable_web_page_preview=True,
            ))
            return
        report_range = _admin_report_range_from_key(range_key)
        await _admin_send_report_pdf_for_range(query, context, user_id, report_range)
        return

    if data == "admin_runtime":
        await safe_send(lambda: query.message.edit_text(
            _runtime_admin_text(),
            parse_mode="HTML",
            reply_markup=get_runtime_admin_kb(),
            disable_web_page_preview=True,
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

    if data in ("admin_perf", "admin_perf_refresh"):
        await _admin_open_perf_settings_panel(query, force=True)
        return

    if data == "admin_perf_apply":
        changed = await _apply_all_bot_performance_settings(user_id, force=True)
        await _admin_open_perf_settings_panel(query, force=True, notice=f"✅ Applied {len(changed)} performance settings.")
        return

    if data == "admin_perf_reset":
        for perf_key in BOT_PERFORMANCE_SETTING_SPECS:
            await _apply_bot_performance_setting(perf_key, _perf_default(perf_key, BOT_SETTING_DEFAULTS.get(perf_key, "")), admin_id=user_id)
        await _admin_open_perf_settings_panel(query, force=True, notice="✅ Reset to code defaults and applied.")
        return

    if data.startswith("admin_perf_set:"):
        key = data.split(":", 1)[1].strip()
        if key not in BOT_PERFORMANCE_SETTING_SPECS:
            await query.answer("Unknown setting", show_alert=True)
            return
        async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS[user_id] = {"state": "awaiting_bot_perf_value", "key": key, "ts": time.monotonic()}
        spec = BOT_PERFORMANCE_SETTING_SPECS[key]
        settings, _status = await get_bot_settings_async(force=True)
        await query.answer("Send new value", show_alert=False)
        await safe_send(lambda: query.message.edit_text(
            f"⚡ <b>{html.escape(str(spec.get('label', key)))}</b>\n\n"
            f"Current: <code>{html.escape(_setting_raw_from(settings, key, BOT_SETTING_DEFAULTS.get(key, '')))}</code>\n"
            f"Allowed: <code>{html.escape(str(spec.get('min', 'off/on')))} - {html.escape(str(spec.get('max', '')))}</code>\n"
            f"Help: {html.escape(str(spec.get('help', '')))}\n\n"
            "Send new value now. Use <code>/cancel</code> to cancel.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="admin_perf")]]),
            disable_web_page_preview=True,
        ))
        return

    if data.startswith("admin_set:"):
        key = data.split(":", 1)[1].strip()
        if key not in BOT_FEATURE_SETTING_KEYS:
            await query.answer("Use Performance Settings for this value.", show_alert=True)
            return
        settings, _status = await get_bot_settings_async(force=True)
        current = _setting_bool_from(settings, key, True)
        ok, info = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: db_bot_setting_set(key, not current, user_id),
        )
        notice = f"✅ {BOT_SETTING_LABELS.get(key, key)} updated." if ok else f"⚠️ Could not persist setting: {info}"
        await _admin_open_settings_panel(query, force=True, notice=notice)
        return

    if data == "admin_tts":
        await _admin_open_tts_provider_panel(query)
        return

    if data.startswith("admin_tts_set:"):
        mode = data.split(":", 1)[1]
        label = await _admin_set_tts_provider(mode)
        await _admin_open_tts_provider_panel(query, notice=f"✅ TTS provider changed to {label}.")
        return

    if data == "admin_tts_hf_enable":
        _reset_hf_tts_cooldown()
        await _admin_open_tts_provider_panel(query, notice="✅ HF Khmer TTS cooldown cleared.")
        return

    if data == "admin_tts_hf_5m":
        global _HF_TTS_DISABLED_UNTIL
        with _HF_TTS_STATE_LOCK:
            _HF_TTS_DISABLED_UNTIL = max(_HF_TTS_DISABLED_UNTIL, time.monotonic() + 300)
        await _admin_open_tts_provider_panel(query, notice="⏸ HF Khmer TTS disabled for 5 minutes.")
        return

    if data == "admin_tts_hf_clear":
        global _HF_TTS_CLIENT, _HF_TTS_CLIENT_KEY
        with _HF_TTS_CLIENT_LOCK:
            _HF_TTS_CLIENT = None
            _HF_TTS_CLIENT_KEY = None
        await _admin_open_tts_provider_panel(query, notice="🧹 HF cached client cleared.")
        return

    if data == "admin_calendar" or data.startswith("admin_calendar:"):
        offset = 0
        if ":" in data:
            with suppress(Exception):
                offset = int(data.split(":", 1)[1])
        await _admin_open_schedule_calendar_panel(query, user_id, offset)
        return

    if data == "admin_errors":
        await _admin_open_error_center(query)
        return

    if data == "admin_errors_clear":
        cleared = _admin_error_center_clear()
        await _admin_open_error_center(query, notice=f"🧹 Cleared {cleared} captured error(s).")
        return

    if data == "admin_api":
        await _cb_api_dashboard(query, user_id, context, "api_menu")
        return

    if data == "admin_crm" or data.startswith("admin_crm:"):
        segment = data.split(":", 1)[1] if ":" in data else "all"
        await _admin_open_crm_panel(query, segment)
        return

    if data == "admin_optimize":
        await _admin_open_optimize_panel(query)
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
            "ផ្ញើពេលវេលាថ្មីតាមម៉ោង Phnom Penh (ICT, UTC+7)។\n"
            "Format: <code>YYYY-MM-DD HH:MM AM/PM</code> or <code>YYYY-MM-DD HH:MM</code>\n"
            "Timezone: Phnom Penh, Cambodia — ICT (UTC+7)\n"
            "Example: <code>2026-12-25 09:00 AM</code> or <code>2026-12-25 21:00</code>\n\n"
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


async def _download_telegram_file_to_temp_path(tg_file, max_bytes: int, suffix: str = ".bin") -> str:
    """Download a Telegram file to disk with a hard size check.

    This avoids io.BytesIO/download_to_memory, which can OOM the worker when
    multiple users upload audio/documents at the same time.  Caller owns the
    returned temp path and must delete it with _cleanup(path).
    """
    max_bytes = int(max_bytes or 0)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    file_size = getattr(tg_file, "file_size", None)
    if file_size is not None and int(file_size) > max_bytes:
        raise ValueError(f"File too large. Max {max_bytes // 1024 // 1024} MB.")

    fd, path = tempfile.mkstemp(prefix="tg_download_", suffix=suffix or ".bin")
    os.close(fd)
    try:
        await tg_file.download_to_drive(path)
        actual_size = os.path.getsize(path)
        if actual_size > max_bytes:
            raise ValueError(f"File too large. Max {max_bytes // 1024 // 1024} MB.")
        return path
    except Exception:
        _cleanup(path)
        raise


async def _download_telegram_file_to_bytes(tg_file, max_bytes: int) -> bytes:
    """Compatibility wrapper that downloads via disk, then reads bounded bytes.

    Keep this only for small text/subtitle files that genuinely need bytes.
    Audio and larger media paths should use _download_telegram_file_to_temp_path()
    and pass the temp path to FFmpeg/AI processing directly.
    """
    path = await _download_telegram_file_to_temp_path(tg_file, max_bytes)
    try:
        chunks: list[bytes] = []
        total = 0
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"File too large. Max {max_bytes // 1024 // 1024} MB.")
                chunks.append(chunk)
        return b"".join(chunks)
    finally:
        _cleanup(path)


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
        get_text_cache_async(src_msg_id, chat_id),
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
    tts_model = prefs.get("tts_model", "auto")
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
                user_id=user_id, username=uname, tts_model=tts_model,
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
                critical=True,
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
                critical=True,
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
                lock_row = await loop.run_in_executor(None, db_lock_read, _SCHED_LOCK_KEY)
                owner_hint = ""
                if lock_row:
                    owner_hint = f" owner={str(lock_row.get('owner') or '')[:96]} until={_web_dt(lock_row.get('locked_until') or '')}"
                _log_once(
                    logging.INFO,
                    "sched_lock_not_owner",
                    "Scheduler tick skipped because another bot instance owns lock %s.%s",
                    _SCHED_LOCK_KEY,
                    owner_hint,
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
    query = update.callback_query
    if not query:
        return

    user_id = query.from_user.id if query.from_user else 0
    data = query.data or ""

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    if query.message is None:
        return

    def _int_part(parts: list[str], index: int, default: int = 0) -> int:
        try:
            return int(parts[index])
        except Exception:
            return int(default)

    async def _invalid_callback() -> None:
        await safe_send(lambda: query.message.reply_text("⚠️ Invalid or expired button data. Please refresh this panel."))

    try:
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
            page = _web_int(data.split(":", 1)[1], 0)
            await _admin_open_recent_history_panel(query, page=page)
            return

        if data.startswith("history_user:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            page = _int_part(parts, 2, 0)
            if target_id <= 0:
                await _invalid_callback()
                return
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
            page = _web_int(data.split(":", 1)[1], 0)
            await _show_user_search_results(query, context, page=page)
            return

        if data.startswith("users_page:"):
            page = _web_int(data.split(":", 1)[1], 0)
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
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
            ))
            return

        if data.startswith("user_history:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            page = _int_part(parts, 3, 0)
            if target_id <= 0:
                await _invalid_callback()
                return
            await _show_user_full_history(query, target_id, back_ref=back_ref, page=page)
            return

        if data.startswith("user_chat:"):
            target_id = _web_int(data.split(":", 1)[1], 0)
            if target_id <= 0:
                await _invalid_callback()
                return
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
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
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
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
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
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_history_clear(target_id))
            row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                "✅ User conversation history cleared.\n\n" + _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
            ))
            return

        logger.debug("users_page_callback: unhandled data=%r", data)

    except Exception as exc:
        logger.error("users_page_callback failed [data=%s]: %s", data, exc, exc_info=True)
        with suppress(Exception):
            await safe_send(lambda: query.message.reply_text("⚠️ Admin user panel error. Please refresh and try again."))

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
        pages = _paginate_pre_html(
            AI_API_KEYS_TABLE_SQL,
            limit=3800,
            header="🧩 <b>Supabase SQL for API keys</b>\n\n",
        )
        for page in pages:
            await safe_send(lambda p=page: msg.reply_text(
                p,
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
            pages = _paginate_pre_html(
                err,
                limit=3500,
                header=(
                    "❌ Cannot create API key.\n"
                    "If this is first setup, press <b>🧩 Setup SQL</b> or run <code>/api sql</code> "
                    "and execute it in Supabase.\n\n"
                ),
            )
            for page in pages:
                await safe_send(lambda p=page: msg.reply_text(
                    p,
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
        for page in _paginate_html(body, limit=3900, header="🔑 <b>AI API Keys</b>\n\n"):
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
    pages = _paginate_pre_html(
        AI_API_KEYS_TABLE_SQL,
        limit=3800,
        header="🧩 <b>Supabase SQL for API keys</b>\n\n",
    )
    for page in pages:
        await safe_send(lambda p=page: message.reply_text(
            p,
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
        loop.run_in_executor(_DB_EXECUTOR, get_all_user_ids),
        loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_admin_pending, update.effective_user.id),
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


@admin_only
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram admin shortcut for the full bot health panel."""
    text = await _admin_health_text()
    await safe_send(lambda: update.message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
        disable_web_page_preview=True,
    ))


async def _clear_admin_transient_state(context: Any, admin_id: int) -> list[str]:
    """Clear every short-lived admin conversation state safely.

    Admin flows are spread across broadcast, schedule editing, user search,
    report generation, chat bridge, and runtime/performance setting inputs.
    A single clear helper prevents stale states from hijacking the next admin
    text message after /cancel or Admin → Cancel.
    """
    cleared: list[str] = []
    user_data = getattr(context, "user_data", {}) or {}

    def _pop_state(key: str, label: str) -> None:
        if user_data.pop(key, None) is not None:
            cleared.append(label)

    if int(user_data.get("bc_state") or 0) == BROADCAST_WAIT_MESSAGE:
        cleared.append("broadcast")
    _pop_state("bc_state", "broadcast")
    if _pending_broadcast.pop(int(admin_id), None) is not None and "broadcast" not in cleared:
        cleared.append("broadcast")

    if int(user_data.get("sched_state") or 0) in (SCHED_WAIT_MSG, SCHED_WAIT_TIME, SCHED_EDIT_WAIT_TIME, SCHED_EDIT_WAIT_TEXT, SCHED_EDIT_WAIT_PHOTO):
        cleared.append("schedule")
    _pop_state("sched_state", "schedule")
    _pop_state("sched_edit_row_id", "schedule-edit")
    if _sched_payload.pop(int(admin_id), None) is not None and "schedule" not in cleared:
        cleared.append("schedule")

    _pop_state("admin_report_state", "report")
    _pop_state("user_search_state", "user-search")
    _pop_state("users_search_query", "user-search")
    _pop_state("users_search_results", "user-search")

    if int(user_data.get("chat_state") or 0) == CHAT_WAIT_MESSAGE:
        _pop_state("chat_state", "admin-chat")
        _close_session(int(admin_id))

    async with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
        if ACTIVE_ADMIN_CONVERSATIONS.pop(int(admin_id), None) is not None:
            cleared.append("runtime-input")

    # De-duplicate while preserving the first useful order.
    return list(dict.fromkeys(cleared))


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not _is_admin(uid):
        return

    # Preserve the old admin-chat notification behavior while still using the
    # unified cleanup helper for every other transient state.
    target_id = None
    with suppress(Exception):
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(uid)

    cleared = await _clear_admin_transient_state(context, uid)

    if target_id:
        with suppress(Exception):
            await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")

    if cleared:
        labels = ", ".join(cleared[:8])
        await safe_send(lambda: update.message.reply_text(
            f"✅ បានបោះបង់/សម្អាត state រួច: <code>{html.escape(labels)}</code>",
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return

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
    # Webhook mode must not drop old message timestamps. During cold starts,
    # this app returns 503 until Telegram is ready, and Telegram may retry the
    # same update later with the original message date.  Dropping it here would
    # lose user messages after a slow Render restart/deploy.
    if "_run_state_bot_mode" in globals() and _run_state_bot_mode() == "WEBHOOK":
        return

    if _BOT_START_TIME == 0.0:
        return

    # Only filter plain messages (not callbacks)
    msg = update.message or update.edited_message
    if msg and getattr(msg, "date", None):
        if msg.date.timestamp() < (_BOT_START_TIME - _STALE_GRACE_S):
            logger.debug(f"Dropping stale message update (id={update.update_id})")
            raise ApplicationHandlerStop


async def cmd_security(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.effective_message:
        return
    blocked = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_is_blocked(int(user.id)))
    status = "blocked" if blocked else "active"
    await safe_send(lambda: update.effective_message.reply_text(
        "🔐 <b>Security & Privacy</b>\n\n"
        f"User ID: <code>{int(user.id)}</code>\n"
        f"Status: <b>{html.escape(status)}</b>\n\n"
        "✅ Admin commands are protected.\n"
        "✅ Spam/flood protection is enabled.\n"
        "✅ API keys are accepted from headers, not URL query strings by default.\n"
        "✅ Use /clear to clear chat context.\n"
        "🗑️ Use /deleteme to delete your saved bot history and preferences.",
        parse_mode="HTML",
    ))


async def cmd_privacy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message:
        return
    await safe_send(lambda: update.effective_message.reply_text(
        "🔒 <b>Privacy</b>\n\n"
        "This bot stores only what it needs for preferences, cached text/audio workflow, "
        "conversation context, admin safety, and delivery logs.\n\n"
        "You can use /clear to remove current conversation context, or /deleteme to delete "
        "your saved bot history and preferences from cache/database where configured.",
        parse_mode="HTML",
    ))


async def _delete_user_personal_data(user_id: int) -> None:
    user_id = int(user_id)
    db_history_clear(user_id)
    _invalidate_prefs(user_id)

    def _run():
        with suppress(Exception):
            _hist_cache_clear(user_id)
        with suppress(Exception):
            if redis_client is not None:
                _redis_delete_sync(_prefs_redis_key(user_id))
                _redis_delete_sync(_hist_redis_key(user_id))
                _redis_delete_sync(_text_cache_user_history_redis_key(user_id))
        if supabase:
            db_call_sync(
                f"user_prefs_delete:{user_id}",
                lambda: supabase.table("user_prefs").delete().eq("user_id", user_id).execute(),
                default=None,
                attempts=3,
                critical=False,
            )
            db_call_sync(
                f"text_cache_delete:{user_id}",
                lambda: supabase.table("text_cache").delete().eq("user_id", user_id).execute(),
                default=None,
                attempts=3,
                critical=False,
            )
            db_call_sync(
                f"conversation_history_delete:{user_id}",
                lambda: supabase.table("conversation_history").delete().eq("user_id", user_id).execute(),
                default=None,
                attempts=3,
                critical=False,
            )

    await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, _run)


async def cmd_delete_my_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.effective_message:
        return
    if _is_admin(int(user.id)) and "--confirm-admin" not in context.args:
        await safe_send(lambda: update.effective_message.reply_text(
            "⚠️ Admin account detected. To delete your own admin-user data, run:\n"
            "<code>/deleteme --confirm-admin</code>",
            parse_mode="HTML",
        ))
        return
    await _delete_user_personal_data(int(user.id))
    await safe_send(lambda: update.effective_message.reply_text(
        "✅ Your saved bot history, text cache, and preferences were cleared.\n"
        "Note: security block records and required delivery/audit logs may be retained by admins for abuse prevention."
    ))


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
    model_label = _tts_model_label(prefs.get("tts_model", "auto"))
    await safe_send(lambda: update.message.reply_text(
        f"⚙️ <b>ការកំណត់របស់អ្នក</b>\n\n"
        f"🗣️ សំឡេង: <b>{gender_label}</b>\n"
        f"🎚️ ល្បឿន: <b>{speed_label}</b>\n"
        f"🤖 ម៉ូដែល TTS: <b>{html.escape(model_label)}</b>\n\n"
        "ផ្ញើ text ណាមួយ ហើយប្រើប៊ូតុងក្រោមសំឡេង ដើម្បីប្តូរ។",
        parse_mode="HTML",
        reply_markup=get_main_kb(prefs["gender"], prefs.get("tts_model", "auto")),
    ))


async def cmd_ttsmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prefs = await get_user_prefs_async(user_id)
    await safe_send(lambda: update.message.reply_text(
        "🤖 <b>ជ្រើសរើសម៉ូដែល TTS</b>\n\n"
        "Auto: Khmer HF Space សម្រាប់ខ្មែរ និង Edge fallback\n"
        "Khmer HF: ប្រើ mrrtmob/khmer-tts សម្រាប់អត្ថបទខ្មែរ\n"
        "Edge: ប្រើ Microsoft Edge TTS គ្រប់ភាសា",
        parse_mode="HTML",
        reply_markup=get_tts_model_kb(prefs.get("tts_model", "auto")),
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
        if await _handle_runtime_admin_text(update, context):
            return
        if await _handle_user_search_text(update, context):
            return
        if await _handle_admin_report_day_text(update, context):
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

    gender    = prefs["gender"]
    speed     = prefs["speed"]
    tts_model = prefs.get("tts_model", "auto")
    tts_text  = tts_text.strip() or stripped

    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(update.effective_chat.id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)

    async with lock:
        try:
            # Long text is safer as multiple smaller voice messages.  This avoids
            # one large MP3 -> OGG/Opus FFmpeg conversion timing out on Render.
            if len(tts_text) > TTS_SINGLE_VOICE_MAX_CHARS:
                uname = user.username or user.first_name or str(user_id)
                await _deliver_paged_tts(
                    chat_id=msg.chat_id,
                    bot=context.bot,
                    text=tts_text,
                    gender=gender,
                    speed=speed,
                    user_id=user_id,
                    username=uname,
                    tts_model=tts_model,
                )
                record_turn(user_id, "user", stripped)
                record_turn(user_id, "assistant", tts_text[:CONV_CONTEXT_MAX_CHARS])
                _set_last_tts(user_id)
                return

            audio_bytes = await generate_voice_limited(tts_text, gender, speed, file_path, tts_model)
            sent_msg    = await safe_send(
                lambda ab=audio_bytes: msg.reply_voice(
                    voice=io.BytesIO(ab),
                    caption=f"🗣️ {BOT_TAG}",
                    reply_markup=get_main_kb(gender, tts_model),
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
        reply_markup=get_main_kb(prefs["gender"], prefs.get("tts_model", "auto"))
    ))


async def _cb_show_tts_model(query, user_id: int, context):
    prefs = await get_user_prefs_async(user_id)
    await safe_send(lambda: query.message.edit_reply_markup(
        reply_markup=get_tts_model_kb(prefs.get("tts_model", "auto"))
    ))


async def _cb_hide_tts_model(query, user_id: int, context):
    prefs = await get_user_prefs_async(user_id)
    await safe_send(lambda: query.message.edit_reply_markup(
        reply_markup=get_main_kb(prefs["gender"], prefs.get("tts_model", "auto"))
    ))


async def _cb_tts_model(query, user_id: int, context, data: str):
    """Save the selected TTS model and regenerate the current voice message.

    UX rule:
      - When the user taps Auto / Khmer HF / Edge from the model menu, save it.
      - If the original text for this voice message can be resolved, immediately
        regenerate the same text with the selected model.
      - If the original text cannot be found, only save the preference and return
        to the main keyboard.
    """
    if query.message is None:
        return

    chat_id = query.message.chat.id
    requested_model = data.replace("ttsmodel_", "", 1)
    model = update_user_tts_model(user_id, requested_model)

    original_text, prefs = await asyncio.gather(
        get_callback_original_text(query, user_id),
        get_user_prefs_async(user_id),
    )
    prefs["tts_model"] = model

    gender = prefs["gender"]
    speed  = prefs["speed"]

    if not original_text:
        await safe_send(lambda: query.message.edit_reply_markup(
            reply_markup=get_main_kb(gender, model)
        ))
        await safe_send(lambda: query.message.reply_text(
            "✅ បានប្តូរម៉ូដែល TTS រួច។\n"
            "❌ រកអត្ថបទដើមមិនឃើញ ដូច្នេះមិនអាចបង្កើតសំឡេងឡើងវិញបានទេ។\n"
            "សូមផ្ញើអត្ថបទម្តងទៀត។"
        ))
        return

    # Model changes are an edit/regenerate action, not a new user message, so do
    # not block them with the normal text-message cooldown. The per-user lock and
    # global TTS semaphore still prevent duplicate concurrent generation.
    with suppress(Exception):
        await query.message.edit_reply_markup(reply_markup=get_main_kb(gender, model))

    status_msg = await safe_send(lambda: context.bot.send_message(
        chat_id=chat_id,
        text=f"🔄 កំពុងប្តូរម៉ូដែល TTS ទៅ {html.escape(TTS_MODEL_OPTIONS.get(model, TTS_MODEL_OPTIONS['auto'])[0])} ហើយបង្កើតសំឡេងឡើងវិញ...",
    ))

    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, context.bot, stop_event))
    lock       = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, gender, speed, file_path, model)
                with suppress(Exception):
                    await query.message.delete()
                new_msg = await safe_send(
                    lambda ab=audio_bytes: context.bot.send_voice(
                        chat_id=chat_id,
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(gender, model),
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
                with suppress(Exception):
                    if status_msg:
                        await status_msg.delete()
                _set_last_tts(user_id)
            except Exception as e:
                logger.error("tts model regenerate error: %s", e, exc_info=True)
                with suppress(Exception):
                    await safe_send(lambda: context.bot.send_message(
                        chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេងឡើងវិញ។"
                    ))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(file_path)


async def get_callback_original_text(query, user_id: int) -> str | None:
    """Resolve the TTS text for a callback button (3-tier fallback)."""
    if query.message is None:
        return get_last_tts_text(user_id)

    chat_id = query.message.chat.id
    msg_id  = query.message.message_id
    loop    = asyncio.get_running_loop()

    # 1. Exact text_cache
    try:
        original_text = await get_text_cache_async(msg_id, chat_id)
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
    tts_model = prefs.get("tts_model", "auto")
    update_user_speed(user_id, new_speed)

    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, context.bot, stop_event))
    lock       = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, gender, new_speed, file_path, tts_model)
                with suppress(Exception):
                    await query.message.delete()
                # FIX: was query.message.chat.send_voice() which is not a valid method.
                # Chat object has no send_voice. Must use bot.send_voice(chat_id=...).
                new_msg = await safe_send(
                    lambda ab=audio_bytes, g=gender: context.bot.send_voice(
                        chat_id=chat_id,
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(g, tts_model),
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
    tts_model = prefs.get("tts_model", "auto")
    update_user_gender(user_id, new_gender)

    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, context.bot, stop_event))
    lock       = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, new_gender, speed, file_path, tts_model)
                with suppress(Exception):
                    await query.message.delete()
                # FIX: was query.message.chat.send_voice() — Chat object has no send_voice.
                # Must use bot.send_voice(chat_id=...).
                new_msg = await safe_send(
                    lambda ab=audio_bytes, ng=new_gender: context.bot.send_voice(
                        chat_id=chat_id,
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(ng, tts_model),
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
    transcript_msg_id = _callback_int_arg(data, "tts_transcript:")
    if transcript_msg_id is None:
        await safe_send(lambda: query.message.reply_text("❌ Invalid transcript id."))
        return
    if query.message is None:
        return

    chat_id = query.message.chat.id
    loop    = asyncio.get_running_loop()
    original_text, prefs = await asyncio.gather(
        get_text_cache_async(transcript_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender     = prefs["gender"]
    speed      = prefs["speed"]
    tts_model  = prefs.get("tts_model", "auto")
    file_path  = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(send_status_timer(chat_id, context.bot, stop_event))
    lock       = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, gender, speed, file_path, tts_model)
                # FIX: was query.message.chat.send_voice() — Chat object has no send_voice.
                new_msg = await safe_send(
                    lambda ab=audio_bytes, g=gender: context.bot.send_voice(
                        chat_id=chat_id,
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(g, tts_model),
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
    if query.message is None:
        return
    src_msg_id = _callback_int_arg(data, "audio_tts:")
    if src_msg_id is None:
        await safe_send(lambda: query.message.reply_text("❌ Invalid audio transcript id."))
        return
    chat_id    = query.message.chat.id
    loop       = asyncio.get_running_loop()

    full_text, prefs = await asyncio.gather(
        get_text_cache_async(src_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )
    if not full_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ (cache expired)។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender    = prefs["gender"]
    speed     = prefs["speed"]
    tts_model = prefs.get("tts_model", "auto")
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
                    user_id=user_id, username=uname, tts_model=tts_model,
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
    query = update.callback_query
    if query is None:
        return

    user_id = query.from_user.id
    data    = (query.data or "").strip()

    if not data:
        with suppress(Exception):
            await query.answer()
        return

    if query.message is None:
        logger.debug(f"on_callback: no message for data={data!r}")
        # FIX: Still answer the query to prevent Telegram showing a loading spinner
        with suppress(Exception):
            await query.answer()
        return

    # FIX: These patterns are handled by dedicated CallbackQueryHandlers that
    # already call query.answer(). Do NOT answer here — it would cause a
    # "query is too old" double-answer error on Telegram's side.
    _HANDLED_EXACT = {"users_close", "noop"}
    _HANDLED_PREFIX = ("bc_", "sched_", "users_page:", "users_search", "user_", "history_", "rtadmin_")
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
        elif data == "show_tts_model":
            await _cb_show_tts_model(query, user_id, context)
        elif data == "hide_tts_model":
            await _cb_hide_tts_model(query, user_id, context)
        elif data.startswith("ttsmodel_"):
            await _cb_tts_model(query, user_id, context, data)
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
    err = getattr(context, "error", None)
    if err is None:
        return

    if isinstance(err, ApplicationHandlerStop) or _is_nonfatal_telegram_edit_error(err):
        logger.debug("Ignored non-fatal Telegram handler condition: %s", err)
        return

    _metric_inc("errors")
    _record_admin_error("telegram_error_handler", str(err), level="ERROR", context=type(update).__name__)
    logger.error(
        "Unhandled exception: %s",
        err,
        exc_info=(type(err), err, getattr(err, "__traceback__", None)),
    )

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
    global _prefs_cache_lock, _TTS_CHUNK_SEMAPHORE, _TELEGRAM_APP, _TELEGRAM_APP_READY, telegram_application

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    # This application lifecycle supports both webhook and polling.  The
    # getUpdates worker itself is guarded separately by ACTIVE_POLLING_TASK;
    # if mode is WEBHOOK, no polling updater is started.

    with suppress(Exception):
        applied = await _apply_all_bot_performance_settings(admin_id=0, force=True)
        if applied:
            logger.info("Applied %s persisted /admin performance settings before bot startup.", len(applied))

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

    builder = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
    )
    # Admin Performance V5: tune update processing and HTTP connection pool.
    # hasattr keeps compatibility with older python-telegram-bot versions.
    if hasattr(builder, "concurrent_updates"):
        builder = builder.concurrent_updates(TELEGRAM_CONCURRENT_UPDATES)
    if hasattr(builder, "connection_pool_size"):
        builder = builder.connection_pool_size(int(TELEGRAM_CONNECTION_POOL_SIZE))
    app = builder.build()
    _TELEGRAM_APP_READY = False
    _TELEGRAM_APP = app
    telegram_application = app

    # Anti-flood/security guards must run before stale-drop and expensive handlers.
    app.add_handler(TypeHandler(Update, _telegram_rate_limit_guard), group=-3)
    app.add_handler(TypeHandler(Update, _telegram_user_security_guard), group=-2)
    if _run_state_bot_mode() != "WEBHOOK":
        app.add_handler(TypeHandler(Update, _drop_stale_updates), group=-1)
    else:
        webhook_logger.info("Stale update drop handler skipped in WEBHOOK mode to preserve Telegram retries after cold starts.")

    # Commands
    app.add_handler(CommandHandler("start",           on_start))
    app.add_handler(CommandHandler("help",            on_help))
    app.add_handler(CommandHandler("myprefs",         cmd_myprefs))
    app.add_handler(CommandHandler("ttsmodel",        cmd_ttsmodel))
    app.add_handler(CommandHandler("clear",           cmd_clear))
    app.add_handler(CommandHandler("security",        cmd_security))
    app.add_handler(CommandHandler("privacy",         cmd_privacy))
    app.add_handler(CommandHandler("deleteme",        cmd_delete_my_data))
    app.add_handler(CommandHandler("broadcast",       broadcast_start))
    app.add_handler(CommandHandler("schedule",        cmd_schedule))
    app.add_handler(CommandHandler("schedules",       cmd_schedules))
    app.add_handler(CommandHandler("cancelschedule",  cmd_cancelschedule))
    app.add_handler(CommandHandler("cancel",          cmd_cancel))
    app.add_handler(CommandHandler("stats",           admin_stats))
    app.add_handler(CommandHandler("health",          cmd_health))
    app.add_handler(CommandHandler("admin",           cmd_admin))
    app.add_handler(CommandHandler("runtime",         cmd_runtime))
    app.add_handler(CommandHandler("api",             cmd_api))
    app.add_handler(CommandHandler("botsettings",     cmd_botsettings))
    app.add_handler(CommandHandler("users",           cmd_users))
    app.add_handler(CommandHandler("chat",            cmd_chat))
    app.add_handler(CommandHandler("endchat",         cmd_endchat))

    # Callback handlers (priority order matters)
    app.add_handler(CallbackQueryHandler(broadcast_callback,  pattern=r"^bc_"))
    app.add_handler(CallbackQueryHandler(
        users_page_callback,
        # Route every admin user/history button here, including paged callbacks like
        # user_history:<id>:p0:<page>. The callback function validates IDs safely.
        pattern=r"^(?:users_(?:page:\d+|search(?:_page:\d+)?|close)|noop|history_(?:page:\d+|refresh|close|user:\d+(?::\d+)?)|user_(?:view|chat|block|unblock|resetprefs|clearhist|history):\d+(?::[psh]\d+)?(?::\d+)?)$",
    ))
    app.add_handler(CallbackQueryHandler(sched_callback,      pattern=r"^sched_"))
    app.add_handler(CallbackQueryHandler(_runtime_admin_callback, pattern=r"^rtadmin_"))
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
        f"Bot starting in {_run_state_bot_mode()} mode. Admins: {ADMIN_IDS or 'none configured'} | "
        f"HF: {HF_MODEL} | OCR provider: {OCR_PROVIDER} | "
        f"HF OCR: {HF_OCR_MODEL} | Gemini: {_gemini is not None} | "
        f"TTS: {_tts_provider_summary()}"
    )

    sched_stop = asyncio.Event()
    sweep_stop = asyncio.Event()
    sched_task: asyncio.Task | None = None
    sweep_task: asyncio.Task | None = None

    async with app:
        # Application.__aenter__() already calls initialize(). Calling
        # initialize() again can raise "Application is already initialized"
        # on python-telegram-bot v20/v21 and break deploy restarts.
        await app.start()
        _TELEGRAM_APP_READY = True
        polling_started = False
        leader_refresh_due = 0.0
        active_owner = await _telegram_leader_refresh_once(app)
        leader_refresh_due = time.monotonic() + _telegram_leader_renew_interval_s()

        if not active_owner:
            webhook_logger.warning(
                "Telegram application is running in STANDBY mode. Web/Admin routes are online, but this Render service will not process Telegram updates until it owns the active lock."
            )
        elif _run_state_bot_mode() == "WEBHOOK":
            await _cancel_active_polling_task("startup_webhook_mode")
            webhook_logger.info("Telegram polling updater is disabled because BOT_MODE=WEBHOOK and this instance owns the active lock.")
        else:
            # Startup may follow a previous webhook deployment; delete it before
            # long polling to prevent Telegram 409 Conflict. Only the active
            # owner may do this in two-server mode.
            with suppress(Exception):
                await _delete_telegram_webhook_via_http(drop_pending=True)
            polling_started = await _telegram_start_polling_runtime(app)
            logger.info("Telegram polling started by active owner.")

        # Start background jobs only after the Telegram application is fully ready.
        # Scheduler itself has its own distributed lock, so both Render services
        # may run this loop safely; only one will fire due scheduled broadcasts.
        sched_task = asyncio.create_task(_scheduler_loop(app.bot, sched_stop))
        sweep_task = asyncio.create_task(_periodic_temp_sweep(sweep_stop))
        try:
            while True:
                now = time.monotonic()
                if now >= leader_refresh_due:
                    active_owner = await _telegram_leader_refresh_once(app)
                    leader_refresh_due = now + _telegram_leader_renew_interval_s()
                else:
                    active_owner = _telegram_should_run_telegram_workers()

                if not active_owner:
                    if polling_started:
                        webhook_logger.info("Stopping Telegram polling because this Render service is standby.")
                        await _telegram_stop_polling_runtime(app)
                        polling_started = False
                    await asyncio.sleep(1.0)
                    continue

                # Early polling-loop exit guard requested for runtime mode switches.
                # In webhook mode the Telegram application stays alive so the
                # FastAPI /telegram-webhook route can continue processing updates;
                # only the updater polling worker is stopped.
                mode = _run_state_bot_mode()
                if polling_started and mode != "POLLING":
                    webhook_logger.info("_run_bot observed BOT_MODE=%s while polling_started; stopping polling worker.", mode)
                    await _telegram_stop_polling_runtime(app)
                    polling_started = False
                elif not polling_started and mode == "POLLING":
                    with suppress(Exception):
                        await _delete_telegram_webhook_via_http(drop_pending=True)
                    polling_started = await _telegram_start_polling_runtime(app)

                await asyncio.sleep(1.0)
        finally:
            sched_stop.set()
            sweep_stop.set()
            if polling_started and app.updater is not None:
                with suppress(Exception):
                    await _telegram_stop_polling_runtime(app)
            with suppress(Exception):
                await asyncio.to_thread(_telegram_leader_release_sync)
            for task in (sched_task, sweep_task):
                if task is None:
                    continue
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            _TELEGRAM_APP_READY = False
            # start() is not automatically paired by the async context manager;
            # stop it explicitly before __aexit__ performs shutdown().
            with suppress(Exception):
                await app.stop()


# ---------------------------------------------------------------------------
# Startup background maintenance
# ---------------------------------------------------------------------------
async def _run_startup_background_checks() -> None:
    """Run slow/non-critical startup checks after web + bot tasks are launched.

    Keeping these checks in the background makes Render/Railway boot faster and
    lets /readyz respond while Supabase column probes and temp cleanup continue.
    """
    async def _run_one(label: str, fn: Callable[[], Any], timeout_s: float = 20.0) -> None:
        try:
            await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout_s)
            logger.info("Startup background check finished: %s", label)
        except asyncio.TimeoutError:
            logger.warning("Startup background check timed out: %s", label)
        except Exception as exc:
            logger.warning("Startup background check failed: %s: %s", label, exc)

    await asyncio.gather(
        _run_one("sweep stale temp files", _sweep_stale_temps, 30.0),
        _run_one("ensure user_prefs.speed column", ensure_speed_column, 25.0),
        _run_one("ensure user_prefs.tts_model column", ensure_tts_model_column, 25.0),
        _run_one("startup self check", startup_self_check, 10.0),
        return_exceptions=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _async_main_once():
    global ACTIVE_POLLING_TASK
    load_dotenv()
    _refresh_arch_runtime_settings()
    _init_clients()
    await _init_async_clients()
    try:
        await _restore_run_state_from_redis()
    except Exception as rexc:
        webhook_logger.warning("Redis fallback triggered. Runtime state restore failed during boot: %s", rexc)
    _runtime_admin_bootstrap_state()

    if not ADMIN_IDS:
        logger.warning(
            "No ADMIN_IDS configured. "
            "Set ADMIN_IDS=123456,789012 in your environment."
        )

    print(
        f"Bot + FastAPI are starting... (AI: {AI_PROVIDER} | HF: {HF_MODEL} | "
        f"OCR: {OCR_PROVIDER} | HF OCR: {HF_OCR_MODEL} | "
        f"TTS: {TTS_PROVIDER}/{KHMER_TTS_PROVIDER} | user_model_default: {_normalize_tts_model(DEFAULT_TTS_MODEL)} | "
        f"Redis: {'on' if redis_client is not None else 'off'} | "
        f"TG mode: {_run_state_bot_mode()} | TG leader-lock: {'on' if TELEGRAM_ACTIVE_LOCK_ENABLED else 'off'} | TG updates: {TELEGRAM_CONCURRENT_UPDATES} | TG pool: {_run_state_http_max_connections()} | "
        f"HTTP pool: {HTTP_MAX_CONNECTIONS}/{HTTP_MAX_KEEPALIVE_CONNECTIONS})"
    )

    _start_web_broadcast_queue_workers()
    keepalive_stop = asyncio.Event()
    telegram_app_task = asyncio.create_task(_run_bot(), name="telegram-bot")
    startup_checks_task = asyncio.create_task(_run_startup_background_checks(), name="startup-background-checks")
    tasks = [
        asyncio.create_task(run_fastapi(), name="fastapi-web"),
        telegram_app_task,
        startup_checks_task,
    ]
    if (os.environ.get("RENDER_EXTERNAL_URL") or getattr(SETTINGS, "RENDER_EXTERNAL_URL", "") or "").strip():
        tasks.append(asyncio.create_task(keep_alive_async(keepalive_stop), name="async-keep-alive"))

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            exc = task.exception()
            if exc:
                raise exc
    finally:
        keepalive_stop.set()
        await _stop_web_broadcast_queue_workers()
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task


def main():
    while True:
        try:
            asyncio.run(_async_main_once())
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
            break
        except Exception as e:
            logger.error(f"Runtime crashed: {e} — restarting in 5s...", exc_info=True)
            time.sleep(5)
        else:
            logger.warning("Runtime stopped — restarting in 5s...")
            time.sleep(5)


if __name__ == "__main__":
    main()
