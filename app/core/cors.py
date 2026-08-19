"""Dynamic exact-origin CORS policy stored in Supabase bot_settings."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.services.settings.store import SettingsStore, get_settings_store

_CORS_KEY = "security:frontend_allowed_origins:v2"


class DynamicCorsError(RuntimeError):
    pass


class DynamicCorsUnavailable(DynamicCorsError):
    pass


class InvalidOriginError(DynamicCorsError):
    pass


def normalize_origin(origin: str) -> str:
    value = str(origin or "").strip().rstrip("/")
    if not value:
        raise InvalidOriginError("Origin is required.")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InvalidOriginError("Origin must be an absolute http:// or https:// URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise InvalidOriginError("Origin cannot contain credentials, query, or fragment.")
    if parsed.path not in {"", "/"}:
        raise InvalidOriginError("Origin must not contain a path.")
    hostname = parsed.hostname or ""
    if not hostname:
        raise InvalidOriginError("Origin hostname is invalid.")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme.lower()}://{hostname.lower()}{port}"


@dataclass(frozen=True)
class CorsSnapshot:
    origins: tuple[str, ...]
    source: str
    persistent: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "origins": list(self.origins),
            "source": self.source,
            "persistent": self.persistent,
        }


class DynamicCorsStore:
    def __init__(self, settings_store: SettingsStore | None = None, *, cache_ttl_seconds: float = 5.0) -> None:
        self.settings = settings_store or get_settings_store()
        self.cache_ttl_seconds = max(0.5, float(cache_ttl_seconds))
        self._snapshot = CorsSnapshot((), "memory", self.settings.status.persistent)
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    async def load(self, *, force: bool = False) -> CorsSnapshot:
        now = time.monotonic()
        if not force and self._loaded_at and now - self._loaded_at < self.cache_ttl_seconds:
            return self._snapshot
        async with self._lock:
            now = time.monotonic()
            if not force and self._loaded_at and now - self._loaded_at < self.cache_ttl_seconds:
                return self._snapshot
            payload = await self.settings.get_json(_CORS_KEY, [])
            origins: set[str] = set()
            if isinstance(payload, list):
                for value in payload:
                    try:
                        origins.add(normalize_origin(str(value)))
                    except InvalidOriginError:
                        continue
            status = self.settings.status
            self._snapshot = CorsSnapshot(
                tuple(sorted(origins)),
                status.backend,
                status.persistent,
            )
            self._loaded_at = now
            return self._snapshot

    async def add(self, origin: str, *, admin_id: int) -> tuple[CorsSnapshot, bool]:
        normalized = normalize_origin(origin)
        current = set((await self.load(force=True)).origins)
        changed = normalized not in current
        current.add(normalized)
        persistent = await self.settings.set_json(
            _CORS_KEY,
            sorted(current),
            updated_by=admin_id,
        )
        self._snapshot = CorsSnapshot(tuple(sorted(current)), self.settings.status.backend, persistent)
        self._loaded_at = time.monotonic()
        return self._snapshot, changed

    async def delete(self, origin: str, *, admin_id: int) -> tuple[CorsSnapshot, bool]:
        normalized = normalize_origin(origin)
        current = set((await self.load(force=True)).origins)
        changed = normalized in current
        current.discard(normalized)
        persistent = await self.settings.set_json(
            _CORS_KEY,
            sorted(current),
            updated_by=admin_id,
        )
        self._snapshot = CorsSnapshot(tuple(sorted(current)), self.settings.status.backend, persistent)
        self._loaded_at = time.monotonic()
        return self._snapshot, changed

    async def is_allowed(self, origin: str) -> bool:
        try:
            normalized = normalize_origin(origin)
        except InvalidOriginError:
            return False
        return normalized in (await self.load()).origins

    def close(self) -> None:
        return None


_STORE = DynamicCorsStore()


def configure_dynamic_cors_store(
    *,
    settings_store: SettingsStore | None = None,
    redis_client: Any | None = None,
    redis_url: str = "",
    supabase_client: Any | None = None,
    **_ignored: Any,
) -> DynamicCorsStore:
    # Compatibility parameters are accepted so older imports do not fail. Redis
    # is intentionally ignored. The shared settings store is configured by the
    # runtime after the Supabase client is initialized.
    del redis_client, redis_url, supabase_client
    global _STORE
    _STORE = DynamicCorsStore(settings_store or get_settings_store())
    return _STORE


def get_dynamic_cors_store() -> DynamicCorsStore:
    return _STORE


class DynamicCORSMiddleware:
    """Credentialed exact-origin CORS middleware with a small policy cache."""

    def __init__(self, app: ASGIApp, *, store: DynamicCorsStore | None = None, max_age: int = 60) -> None:
        self.app = app
        self.store = store
        self.max_age = max(0, int(max_age))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if not origin:
            await self.app(scope, receive, send)
            return
        policy_store = self.store or get_dynamic_cors_store()
        allowed = await policy_store.is_allowed(origin)
        is_preflight = scope.get("method") == "OPTIONS" and bool(headers.get("access-control-request-method"))
        if is_preflight:
            if not allowed:
                response = PlainTextResponse("CORS origin is not allowed.", status_code=403)
                await response(scope, receive, send)
                return
            requested_headers = headers.get("access-control-request-headers", "")
            response_headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
                "Access-Control-Max-Age": str(self.max_age),
                "Vary": "Origin",
            }
            if requested_headers:
                response_headers["Access-Control-Allow-Headers"] = requested_headers
            response = PlainTextResponse("", status_code=204, headers=response_headers)
            await response(scope, receive, send)
            return

        async def send_with_cors(message: Message) -> None:
            if allowed and message["type"] == "http.response.start":
                mutable = list(message.get("headers", []))
                mutable.extend(
                    [
                        (b"access-control-allow-origin", origin.encode("latin-1")),
                        (b"access-control-allow-credentials", b"true"),
                        (b"vary", b"Origin"),
                    ]
                )
                message["headers"] = mutable
            await send(message)

        await self.app(scope, receive, send_with_cors)


__all__ = [
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
