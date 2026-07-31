"""Redis/Supabase-backed dynamic CORS policy and ASGI middleware."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

CORS_SUPABASE_SETTING_KEY = "frontend_allowed_origins"
DEFAULT_CORS_CACHE_TTL_SECONDS = 5.0
ALLOWED_CORS_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
ALLOWED_CORS_HEADERS = (
    "accept",
    "accept-language",
    "authorization",
    "content-language",
    "content-type",
    "x-admin-session-token",
    "x-api-key",
    "x-csrf-token",
    "x-requested-with",
    "x-telegram-init-data",
)
EXPOSED_CORS_HEADERS = ("X-Request-ID", "X-Response-Time-ms")
STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class DynamicCorsError(RuntimeError):
    """Base exception for dynamic CORS configuration failures."""


class InvalidOriginError(DynamicCorsError, ValueError):
    """Raised when an administrator submits something other than an origin."""


class DynamicCorsUnavailable(DynamicCorsError):
    """Raised when neither Redis nor Supabase can persist the policy."""


def normalize_origin(value: str) -> str:
    """Validate and canonicalize one exact HTTP(S) origin.

    Paths, credentials, queries, fragments, wildcards, and opaque ``null``
    origins are rejected.  Credentialed admin CORS must use explicit origins.
    """

    raw = str(value or "").strip()
    if not raw:
        raise InvalidOriginError("Origin must not be empty.")
    if len(raw) > 2048:
        raise InvalidOriginError("Origin is too long.")
    if raw == "*" or raw.lower() == "null":
        raise InvalidOriginError("Wildcard and null origins are not allowed.")

    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise InvalidOriginError("Origin contains an invalid host or port.") from exc

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if scheme not in {"http", "https"}:
        raise InvalidOriginError("Origin scheme must be http or https.")
    if not hostname:
        raise InvalidOriginError("Origin must include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise InvalidOriginError("Origin must not contain credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise InvalidOriginError("Origin must not contain a path, query, or fragment.")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            hostname = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise InvalidOriginError("Origin hostname is invalid.") from exc
        if len(hostname) > 253 or any(
            not _DNS_LABEL_RE.fullmatch(label)
            for label in hostname.split(".")
        ):
            raise InvalidOriginError("Origin hostname is invalid.")
        host = hostname
    else:
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    port_part = "" if port is None or default_port else f":{port}"
    return f"{scheme}://{host}{port_part}"


@dataclass(frozen=True)
class CorsSnapshot:
    origins: tuple[str, ...]
    source: str
    loaded_at_monotonic: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "origins": list(self.origins),
            "count": len(self.origins),
            "source": self.source,
        }


class DynamicCorsStore:
    """Exact-origin allowlist with Redis primary and Supabase fallback."""

    def __init__(
        self,
        *,
        redis_client: Any | None = None,
        redis_url: str = "",
        supabase_client: Any | None = None,
        redis_prefix: str = "tgbot",
        cache_ttl_seconds: float = DEFAULT_CORS_CACHE_TTL_SECONDS,
    ) -> None:
        prefix = str(redis_prefix or "tgbot").strip().strip(":") or "tgbot"
        self.redis_prefix = prefix
        self.redis_client = redis_client
        self.redis_url = str(redis_url or "").strip()
        self.supabase_client = supabase_client
        self.cache_ttl_seconds = max(0.25, min(60.0, float(cache_ttl_seconds)))
        self._owns_redis = False
        self._snapshot = CorsSnapshot((), "not-loaded", 0.0)
        self._cache_lock = threading.RLock()
        self._client_lock = threading.RLock()
        self._load_lock = threading.Lock()

    @property
    def redis_origins_key(self) -> str:
        return f"{self.redis_prefix}:security:cors:origins:v1"

    @property
    def redis_initialized_key(self) -> str:
        return f"{self.redis_prefix}:security:cors:initialized:v1"

    def configure(
        self,
        *,
        redis_client: Any | None = None,
        redis_url: str = "",
        supabase_client: Any | None = None,
    ) -> DynamicCorsStore:
        requested_url = str(redis_url).strip()
        with self._load_lock, self._client_lock:
            changed = False
            if redis_client is not None:
                if redis_client is not self.redis_client:
                    self._close_owned_redis_locked()
                    self.redis_client = redis_client
                    changed = True
                # An explicitly supplied client is caller-owned, even when it
                # happens to be the same object this store created earlier.
                self._owns_redis = False
            if requested_url and requested_url != self.redis_url:
                if self._owns_redis:
                    self._close_owned_redis_locked()
                self.redis_url = requested_url
                changed = True
            if (
                supabase_client is not None
                and supabase_client is not self.supabase_client
            ):
                self.supabase_client = supabase_client
                changed = True
            if changed:
                with self._cache_lock:
                    self._snapshot = CorsSnapshot((), "not-loaded", 0.0)
        return self

    def _close_owned_redis_locked(self) -> None:
        if self._owns_redis and self.redis_client is not None:
            try:
                self.redis_client.close()
            except Exception:
                logger.debug("Dynamic CORS Redis close failed.", exc_info=True)
        self.redis_client = None
        self._owns_redis = False

    def _connect_redis_sync(self) -> Any | None:
        with self._client_lock:
            if self.redis_client is not None:
                return self.redis_client
            if not self.redis_url:
                return None
            try:
                import redis

                client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30,
                    retry_on_timeout=True,
                    max_connections=8,
                )
                client.ping()
            except Exception as exc:  # noqa: BLE001 - Redis client boundary
                logger.warning("Dynamic CORS Redis connection failed: %s", exc)
                return None
            self.redis_client = client
            self._owns_redis = True
            return client

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        return str(value or "")

    def _redis_read_sync(self) -> tuple[list[str] | None, str]:
        client = self._connect_redis_sync()
        if client is None:
            return None, "redis-unavailable"
        try:
            initialized = bool(client.get(self.redis_initialized_key))
            if not initialized:
                return None, "redis-not-initialized"
            origins = sorted(
                normalize_origin(self._decode(value))
                for value in (client.smembers(self.redis_origins_key) or set())
            )
            return list(dict.fromkeys(origins)), "redis"
        except InvalidOriginError as exc:
            raise DynamicCorsError(f"Redis CORS data is invalid: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - Redis client boundary
            logger.warning("Dynamic CORS Redis read failed: %s", exc)
            return None, "redis-error"

    def _supabase_read_sync(self) -> tuple[list[str] | None, str]:
        client = self.supabase_client
        if client is None:
            return None, "supabase-unavailable"
        try:
            response = (
                client.table("bot_settings")
                .select("value")
                .eq("key", CORS_SUPABASE_SETTING_KEY)
                .limit(1)
                .execute()
            )
            rows = list(getattr(response, "data", None) or [])
            if not rows:
                return None, "supabase-not-initialized"
            raw = rows[0].get("value") if isinstance(rows[0], dict) else None
            parsed = json.loads(str(raw or "[]"))
            if not isinstance(parsed, list):
                raise DynamicCorsError("Supabase CORS value must be a JSON list.")
            origins = sorted({normalize_origin(item) for item in parsed})
            return origins, "supabase"
        except DynamicCorsError:
            raise
        except Exception as exc:  # noqa: BLE001 - Supabase SDK boundary
            logger.warning("Dynamic CORS Supabase read failed: %s", exc)
            return None, "supabase-error"

    def _redis_replace_sync(self, origins: list[str]) -> bool:
        client = self._connect_redis_sync()
        if client is None:
            return False
        normalized = sorted({normalize_origin(item) for item in origins})
        try:
            pipe = client.pipeline(transaction=True)
            pipe.delete(self.redis_origins_key)
            if normalized:
                pipe.sadd(self.redis_origins_key, *normalized)
            pipe.set(self.redis_initialized_key, "1")
            pipe.execute()
            return True
        except Exception as exc:  # noqa: BLE001 - Redis pipeline boundary
            logger.warning("Dynamic CORS Redis write failed: %s", exc)
            return False

    def _redis_mutate_sync(
        self,
        origin: str,
        *,
        add: bool,
    ) -> tuple[bool, bool, list[str]]:
        """Atomically add/remove one origin in the Redis source of truth."""

        client = self._connect_redis_sync()
        if client is None:
            return False, False, []
        normalized = normalize_origin(origin)
        try:
            pipe = client.pipeline(transaction=True)
            if add:
                pipe.sadd(self.redis_origins_key, normalized)
            else:
                pipe.srem(self.redis_origins_key, normalized)
            pipe.set(self.redis_initialized_key, "1")
            results = pipe.execute()
            changed = bool(results[0])
            origins = sorted(
                normalize_origin(self._decode(value))
                for value in (client.smembers(self.redis_origins_key) or set())
            )
            return True, changed, list(dict.fromkeys(origins))
        except Exception as exc:  # noqa: BLE001 - Redis pipeline boundary
            logger.warning("Dynamic CORS Redis mutation failed: %s", exc)
            return False, False, []

    def _supabase_replace_sync(self, origins: list[str], admin_id: int) -> bool:
        client = self.supabase_client
        if client is None:
            return False
        payload = {
            "key": CORS_SUPABASE_SETTING_KEY,
            "value": json.dumps(
                sorted({normalize_origin(item) for item in origins}),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "updated_by": int(admin_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            client.table("bot_settings").upsert(
                payload,
                on_conflict="key",
            ).execute()
            return True
        except Exception as exc:  # noqa: BLE001 - Supabase SDK boundary
            logger.warning("Dynamic CORS Supabase write failed: %s", exc)
            return False

    def _set_snapshot(self, origins: list[str], source: str) -> CorsSnapshot:
        snapshot = CorsSnapshot(
            tuple(sorted({normalize_origin(item) for item in origins})),
            source,
            time.monotonic(),
        )
        with self._cache_lock:
            self._snapshot = snapshot
        return snapshot

    def cached_snapshot(self) -> CorsSnapshot:
        with self._cache_lock:
            return self._snapshot

    def cached_origins(self) -> list[str]:
        return list(self.cached_snapshot().origins)

    def _load_sync(
        self,
        *,
        force: bool,
        observed_loaded_at: float,
    ) -> CorsSnapshot:
        """Load once when concurrent requests observe the same stale snapshot."""

        with self._load_lock:
            snapshot = self.cached_snapshot()
            refreshed_by_another_request = (
                snapshot.source != "not-loaded"
                and snapshot.loaded_at_monotonic != observed_loaded_at
            )
            if refreshed_by_another_request or (
                not force
                and snapshot.source != "not-loaded"
                and time.monotonic() - snapshot.loaded_at_monotonic
                < self.cache_ttl_seconds
            ):
                return snapshot

            redis_origins, redis_source = self._redis_read_sync()
            if redis_origins is not None:
                return self._set_snapshot(redis_origins, redis_source)

            supabase_origins, supabase_source = self._supabase_read_sync()
            if supabase_origins is not None:
                mirrored = self._redis_replace_sync(supabase_origins)
                source = "supabase+redis" if mirrored else supabase_source
                return self._set_snapshot(supabase_origins, source)

            # An empty list is the secure first-boot policy. Persist its
            # initialized marker so Redis remains authoritative instead of
            # querying Supabase on every cache refresh.
            initialized = self._redis_replace_sync([])
            if initialized:
                return self._set_snapshot([], "redis-initialized-empty")
            if self.supabase_client is not None:
                saved = self._supabase_replace_sync([], 0)
                if saved:
                    return self._set_snapshot([], "supabase-initialized-empty")
            raise DynamicCorsUnavailable(
                "Dynamic CORS requires an available Redis or Supabase connection."
            )

    async def load(self, *, force: bool = False) -> CorsSnapshot:
        snapshot = self.cached_snapshot()
        if (
            not force
            and snapshot.source != "not-loaded"
            and time.monotonic() - snapshot.loaded_at_monotonic < self.cache_ttl_seconds
        ):
            return snapshot

        return await asyncio.to_thread(
            self._load_sync,
            force=force,
            observed_loaded_at=snapshot.loaded_at_monotonic,
        )

    async def replace(self, origins: list[str], *, admin_id: int) -> CorsSnapshot:
        normalized = sorted({normalize_origin(item) for item in origins})
        redis_ok, supabase_ok = await asyncio.gather(
            asyncio.to_thread(self._redis_replace_sync, normalized),
            asyncio.to_thread(self._supabase_replace_sync, normalized, admin_id),
        )
        if not redis_ok and not supabase_ok:
            raise DynamicCorsUnavailable(
                "Could not persist CORS origins in Redis or Supabase."
            )
        source = (
            "redis+supabase"
            if redis_ok and supabase_ok
            else "redis"
            if redis_ok
            else "supabase"
        )
        return self._set_snapshot(normalized, source)

    async def add(self, origin: str, *, admin_id: int) -> tuple[CorsSnapshot, bool]:
        normalized = normalize_origin(origin)
        redis_ok, changed, redis_origins = await asyncio.to_thread(
            self._redis_mutate_sync,
            normalized,
            add=True,
        )
        if redis_ok:
            supabase_ok = await asyncio.to_thread(
                self._supabase_replace_sync,
                redis_origins,
                admin_id,
            )
            source = "redis+supabase" if supabase_ok else "redis"
            return self._set_snapshot(redis_origins, source), changed

        current = await self.load(force=True)
        origins = set(current.origins)
        changed = normalized not in origins
        origins.add(normalized)
        snapshot = await self.replace(sorted(origins), admin_id=admin_id)
        return snapshot, changed

    async def delete(self, origin: str, *, admin_id: int) -> tuple[CorsSnapshot, bool]:
        normalized = normalize_origin(origin)
        redis_ok, changed, redis_origins = await asyncio.to_thread(
            self._redis_mutate_sync,
            normalized,
            add=False,
        )
        if redis_ok:
            supabase_ok = await asyncio.to_thread(
                self._supabase_replace_sync,
                redis_origins,
                admin_id,
            )
            source = "redis+supabase" if supabase_ok else "redis"
            return self._set_snapshot(redis_origins, source), changed

        current = await self.load(force=True)
        origins = set(current.origins)
        changed = normalized in origins
        origins.discard(normalized)
        snapshot = await self.replace(sorted(origins), admin_id=admin_id)
        return snapshot, changed

    async def is_allowed(self, origin: str) -> bool:
        try:
            normalized = normalize_origin(origin)
        except InvalidOriginError:
            return False
        snapshot = await self.load(force=False)
        return normalized in snapshot.origins

    def close(self) -> None:
        with self._load_lock, self._client_lock:
            self._close_owned_redis_locked()
            with self._cache_lock:
                self._snapshot = CorsSnapshot((), "not-loaded", 0.0)


class DynamicCORSMiddleware:
    """Credentialed CORS middleware backed by :class:`DynamicCorsStore`."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        store: DynamicCorsStore,
        max_age: int = 60,
    ) -> None:
        self.app = app
        self.store = store
        self.max_age = max(0, min(600, int(max_age)))

    @staticmethod
    def _headers(scope: Scope) -> dict[str, str]:
        return {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }

    @staticmethod
    def _is_admin_write(scope: Scope) -> bool:
        method = str(scope.get("method") or "GET").upper()
        path = str(scope.get("path") or "")
        return method in STATE_CHANGING_METHODS and path.startswith(
            ("/api/admin", "/admin/")
        )

    def _preflight_headers(self, origin: str) -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": ", ".join(ALLOWED_CORS_METHODS),
            "Access-Control-Allow-Headers": ", ".join(ALLOWED_CORS_HEADERS),
            "Access-Control-Max-Age": str(self.max_age),
            "Vary": "Origin",
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = self._headers(scope)
        origin = headers.get("origin", "").strip()
        if not origin:
            await self.app(scope, receive, send)
            return

        allowed = await self.store.is_allowed(origin)
        is_preflight = (
            str(scope.get("method") or "").upper() == "OPTIONS"
            and bool(headers.get("access-control-request-method"))
        )
        if is_preflight:
            requested_method = headers.get("access-control-request-method", "").upper()
            requested_headers = {
                item.strip().lower()
                for item in headers.get("access-control-request-headers", "").split(",")
                if item.strip()
            }
            valid_request = (
                allowed
                and requested_method in ALLOWED_CORS_METHODS
                and requested_headers.issubset(set(ALLOWED_CORS_HEADERS))
            )
            response = PlainTextResponse(
                "OK" if valid_request else "CORS origin, method, or headers denied",
                status_code=204 if valid_request else 403,
                headers=self._preflight_headers(normalize_origin(origin)) if valid_request else None,
            )
            await response(scope, receive, send)
            return

        if not allowed and self._is_admin_write(scope):
            response = JSONResponse(
                {
                    "ok": False,
                    "code": "admin_origin_forbidden",
                    "error": "This admin API origin is not allowed.",
                },
                status_code=403,
            )
            await response(scope, receive, send)
            return

        if not allowed:
            await self.app(scope, receive, send)
            return

        normalized_origin = normalize_origin(origin)

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                mutable = MutableHeaders(scope=message)
                mutable["Access-Control-Allow-Origin"] = normalized_origin
                mutable["Access-Control-Allow-Credentials"] = "true"
                mutable["Access-Control-Expose-Headers"] = ", ".join(
                    EXPOSED_CORS_HEADERS
                )
                existing_vary = mutable.get("Vary", "")
                vary_values = {
                    item.strip()
                    for item in existing_vary.split(",")
                    if item.strip()
                }
                vary_values.add("Origin")
                mutable["Vary"] = ", ".join(sorted(vary_values))
            await send(message)

        await self.app(scope, receive, send_with_cors)


_DYNAMIC_CORS_STORE = DynamicCorsStore()


def configure_dynamic_cors_store(
    *,
    redis_client: Any | None = None,
    redis_url: str = "",
    supabase_client: Any | None = None,
) -> DynamicCorsStore:
    return _DYNAMIC_CORS_STORE.configure(
        redis_client=redis_client,
        redis_url=redis_url,
        supabase_client=supabase_client,
    )


def get_dynamic_cors_store() -> DynamicCorsStore:
    return _DYNAMIC_CORS_STORE


__all__ = [
    "ALLOWED_CORS_HEADERS",
    "ALLOWED_CORS_METHODS",
    "CORS_SUPABASE_SETTING_KEY",
    "CorsSnapshot",
    "DynamicCORSMiddleware",
    "DynamicCorsError",
    "DynamicCorsStore",
    "DynamicCorsUnavailable",
    "InvalidOriginError",
    "configure_dynamic_cors_store",
    "get_dynamic_cors_store",
    "normalize_origin",
]
