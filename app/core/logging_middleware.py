"""HTTP Server Request Lifecycle & Comprehensive Access Logging Middleware.

Features:
- Logs ALL incoming requests across all HTTP methods (GET, POST, OPTIONS, PUT, DELETE, etc.)
- Logs initial PENDING state with client IP, path, query params, and request ID
- Handles OPTIONS preflight requests automatically with standard CORS headers
- Logs completion with HTTP status code, status indicator, duration in ms, and active request count
- Captures and logs crashes/unhandled exceptions cleanly
- Injects standard telemetry headers (X-Request-ID, X-Response-Time-ms, X-Active-Requests)
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qsl, unquote, urlencode

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime

try:
    from fastapi import Request, Response
except (ImportError, ModuleNotFoundError):
    class Request:  # type: ignore[no-redef]
        pass

    class Response:  # type: ignore[no-redef]
        def __init__(self, content: Any = None, status_code: int = 200, headers: Any = None) -> None:
            self.content = content
            self.status_code = status_code
            self.headers = dict(headers or {})

logger = logging.getLogger("app.server")

_ACTIVE_REQUESTS = 0
_ACTIVE_LOCK = threading.RLock()


@dataclass
class RequestRecord:
    """Structured telemetry record for an HTTP request or Telegram update."""

    id: str
    timestamp: float
    time_str: str
    category: str  # "HTTP" or "TELEGRAM"
    method: str  # "GET", "POST", "MESSAGE", "CALLBACK", "COMMAND"
    path: str  # "/healthz", "/webhook", "/tiktok", "tt_mp3:123"
    client: str  # IP address or Telegram user "@username (123456)"
    status: str  # "PENDING", "200 OK", "404 NOT_FOUND", "SUCCESS", "FAILED"
    status_code: int = 200
    duration_ms: float = 0.0
    detail: str = ""
    active_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "timestamp": float(self.timestamp) if isinstance(self.timestamp, (int, float)) else 0.0,
            "time_str": str(self.time_str),
            "category": str(self.category),
            "method": str(self.method),
            "path": str(self.path),
            "client": str(self.client),
            "status": str(self.status),
            "status_code": int(self.status_code) if isinstance(self.status_code, (int, float)) else 200,
            "duration_ms": round(float(self.duration_ms), 2) if isinstance(self.duration_ms, (int, float)) else 0.0,
            "detail": str(self.detail),
            "active_requests": int(self.active_requests) if isinstance(self.active_requests, int) else 0,
        }


class RequestLogStore:
    """Thread-safe in-memory ring buffer and live event bus for real-time request tracking."""

    def __init__(self, maxlen: int = 500) -> None:
        self._maxlen = maxlen
        self._buffer: deque[RequestRecord] = deque(maxlen=maxlen)
        self._records_map: dict[str, RequestRecord] = {}
        self._lock = threading.RLock()
        self._subscribers: set[asyncio.Queue[RequestRecord]] = set()

        # Telemetry counters
        self._total_requests = 0
        self._total_http = 0
        self._total_telegram = 0
        self._total_errors = 0
        self._start_time = time.time()

    def record_start(
        self,
        request_id: str,
        category: str,
        method: str,
        path: str,
        client: str,
        detail: str = "",
    ) -> RequestRecord:
        """Record the initiation of an incoming HTTP or Telegram request."""
        now = time.time()
        time_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")

        with _ACTIVE_LOCK:
            active = _ACTIVE_REQUESTS

        record = RequestRecord(
            id=request_id,
            timestamp=now,
            time_str=time_str,
            category=category.upper(),
            method=method.upper(),
            path=path,
            client=client,
            status="PENDING",
            status_code=0,
            duration_ms=0.0,
            detail=detail,
            active_requests=active,
        )

        with self._lock:
            self._buffer.appendleft(record)
            self._records_map[request_id] = record
            self._total_requests += 1
            if category.upper() == "HTTP":
                self._total_http += 1
            elif category.upper() == "TELEGRAM":
                self._total_telegram += 1

            # Prune records_map if larger than twice maxlen
            if len(self._records_map) > self._maxlen * 2:
                valid_ids = {r.id for r in self._buffer}
                self._records_map = {k: v for k, v in self._records_map.items() if k in valid_ids}

        self._notify_subscribers(record)
        return record

    def record_complete(
        self,
        request_id: str,
        status: str,
        status_code: int = 200,
        duration_ms: float = 0.0,
        detail: str = "",
    ) -> RequestRecord | None:
        """Update request record with its completion metrics and notify live listeners."""
        with self._lock:
            record = self._records_map.get(request_id)
            if record is None:
                # If not found in map, find in buffer
                for r in self._buffer:
                    if r.id == request_id:
                        record = r
                        break

            if record is not None:
                was_pending = (record.status == "PENDING")
                record.status = status
                record.status_code = status_code
                record.duration_ms = duration_ms
                if detail:
                    record.detail = detail
                with _ACTIVE_LOCK:
                    record.active_requests = _ACTIVE_REQUESTS

                is_error = status_code >= 400 or "FAIL" in status.upper() or "ERROR" in status.upper()
                if is_error and was_pending:
                    self._total_errors += 1

        if record is not None:
            self._notify_subscribers(record)
        return record

    def get_recent(
        self,
        limit: int = 100,
        category: str | None = None,
        only_errors: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve recent request records as serialized dictionaries."""
        with self._lock:
            items = list(self._buffer)

        limit_val = max(1, min(int(limit or 100), 500))
        cat_filter = category.upper() if isinstance(category, str) and category else None

        results: list[dict[str, Any]] = []
        for rec in items:
            if cat_filter and rec.category != cat_filter:
                continue
            if only_errors and rec.status_code < 400 and "FAIL" not in rec.status and "ERROR" not in rec.status:
                continue
            results.append(rec.to_dict())
            if len(results) >= limit_val:
                break
        return results

    def get_metrics(self) -> dict[str, Any]:
        """Return high-level telemetry summary for live server monitoring."""
        uptime_s = max(1.0, time.time() - self._start_time)
        with self._lock:
            total = self._total_requests
            http = self._total_http
            tg = self._total_telegram
            errs = self._total_errors
            buffer_size = len(self._buffer)

        with _ACTIVE_LOCK:
            active = _ACTIVE_REQUESTS

        req_per_min = round((total / uptime_s) * 60.0, 1)

        # Average latency of completed records in current buffer
        with self._lock:
            durations = [r.duration_ms for r in self._buffer if r.duration_ms > 0.0]
        avg_latency = round(sum(durations) / len(durations), 2) if durations else 0.0

        return {
            "uptime_seconds": int(uptime_s),
            "total_requests": total,
            "total_http": http,
            "total_telegram": tg,
            "total_errors": errs,
            "active_requests": active,
            "requests_per_minute": req_per_min,
            "average_latency_ms": avg_latency,
            "buffer_capacity": self._maxlen,
            "buffered_records": buffer_size,
        }

    def subscribe(self) -> asyncio.Queue[RequestRecord]:
        """Subscribe an async queue to receive real-time request events (e.g. for SSE)."""
        queue: asyncio.Queue[RequestRecord] = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[RequestRecord]) -> None:
        """Unsubscribe an async queue."""
        with self._lock:
            self._subscribers.discard(queue)

    def _notify_subscribers(self, record: RequestRecord) -> None:
        """Push update record to all active subscribers without blocking."""
        with self._lock:
            subscribers = list(self._subscribers)

        for q in subscribers:
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                # Discard oldest to prevent lag
                try:
                    q.get_nowait()
                    q.put_nowait(record)
                except Exception:
                    logger.debug("Dropped telemetry record due to queue full/error")
            except Exception as e:
                logger.debug(f"Failed to put record in telemetry queue: {e}")


_GLOBAL_LOG_STORE: RequestLogStore | None = None
_STORE_LOCK = threading.RLock()


def get_request_log_store() -> RequestLogStore:
    """Return singleton instance of the real-time RequestLogStore."""
    global _GLOBAL_LOG_STORE
    with _STORE_LOCK:
        if _GLOBAL_LOG_STORE is None:
            _GLOBAL_LOG_STORE = RequestLogStore(maxlen=500)
        return _GLOBAL_LOG_STORE


def get_active_requests_count() -> int:
    """Return the current number of in-flight/pending HTTP requests."""
    with _ACTIVE_LOCK:
        return _ACTIVE_REQUESTS


def get_client_ip(request: Request) -> str:
    """Extract real client IP considering reverse proxy and CDN headers."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    client = getattr(request, "client", None)
    if client and getattr(client, "host", None):
        return client.host
    return "unknown"


def get_status_indicator(status_code: int) -> tuple[str, str]:
    """Return status icon and label for a given HTTP status code."""
    if 200 <= status_code < 300:
        return "✅", "OK"
    elif 300 <= status_code < 400:
        return "↪️", "REDIRECT"
    elif 400 <= status_code < 500:
        return "⚠️", "CLIENT_ERROR"
    else:
        return "❌", "SERVER_ERROR"


_SENSITIVE_PARAM_NAMES = {
    "token", "secret", "key", "api_key", "apikey", "password", "pass",
    "auth", "authorization", "bot_token", "gemini_key", "access_token",
    "refresh_token", "jwt", "session", "csrf", "csrf_token"
}

_TELEGRAM_TOKEN_PATTERN = re.compile(r"([0-9]{8,10}:[a-zA-Z0-9_-]{35})")


def sanitize_path_and_query(raw_path: str, raw_query: str) -> tuple[str, str]:
    """Sanitize sensitive tokens in path and query parameters to prevent leaks in logs."""
    sanitized_path = _TELEGRAM_TOKEN_PATTERN.sub("[REDACTED_TOKEN]", raw_path or "/")

    if not raw_query:
        return sanitized_path, ""

    try:
        parsed_params = parse_qsl(raw_query, keep_blank_values=True)
        sanitized_params = []
        for k, v in parsed_params:
            k_lower = k.lower()
            if k_lower in _SENSITIVE_PARAM_NAMES or any(sec in k_lower for sec in ("token", "secret", "key", "auth", "pass")):
                sanitized_params.append((k, "[REDACTED]"))
            else:
                v_san = _TELEGRAM_TOKEN_PATTERN.sub("[REDACTED_TOKEN]", v)
                sanitized_params.append((k, v_san))
        return sanitized_path, f"?{urlencode(sanitized_params, safe='[]')}"
    except Exception:
        return sanitized_path, "?[REDACTED_QUERY]"


async def request_lifecycle_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Any],
) -> Response:
    """ASGI Middleware to log all HTTP requests across their full lifecycle."""
    global _ACTIVE_REQUESTS

    request_id = (request.headers.get("x-request-id") or secrets.token_hex(4)).strip()[:32]
    method = request.method.upper() if getattr(request, "method", None) else "GET"
    raw_path = getattr(getattr(request, "url", None), "path", "/") or "/"
    query = getattr(getattr(request, "url", None), "query", "")
    safe_path, query_str = sanitize_path_and_query(raw_path, query)
    client_ip = get_client_ip(request)
    start_time = time.monotonic()

    with _ACTIVE_LOCK:
        _ACTIVE_REQUESTS += 1
        active_now = _ACTIVE_REQUESTS

    store = get_request_log_store()
    endpoint_display = f"{safe_path}{query_str}"
    is_tg_webhook = "/webhook" in safe_path
    initial_detail = "Telegram Webhook Ingest" if is_tg_webhook else ""

    # 1. Record start in store and log in PENDING state immediately
    store.record_start(
        request_id=request_id,
        category="HTTP",
        method=method,
        path=endpoint_display,
        client=client_ip,
        detail=initial_detail,
    )

    logger.info(
        "⏳ [PENDING] %s %s%s from %s (req_id=%s, active=%d)",
        method,
        safe_path,
        query_str,
        client_ip,
        request_id,
        active_now,
    )

    resp: Response | None = None
    try:
        # 2. Dedicated handling for OPTIONS (CORS preflight & checks)
        if method == "OPTIONS":
            try:
                resp = await call_next(request)
                if getattr(resp, "status_code", 200) == 405:
                    origin = request.headers.get("origin") or "*"
                    req_headers = request.headers.get("access-control-request-headers") or "*"
                    resp = Response(
                        content="",
                        status_code=200,
                        headers={
                            "Access-Control-Allow-Origin": origin,
                            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD",
                            "Access-Control-Allow-Headers": req_headers,
                            "Access-Control-Max-Age": "86400",
                            "Content-Length": "0",
                        },
                    )
            except Exception:
                origin = request.headers.get("origin") or "*"
                resp = Response(
                    content="",
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD",
                        "Access-Control-Allow-Headers": "*",
                        "Access-Control-Max-Age": "86400",
                        "Content-Length": "0",
                    },
                )
        else:
            resp = await call_next(request)

    except Exception as exc:
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        store.record_complete(
            request_id=request_id,
            status=f"500 SERVER_ERROR",
            status_code=500,
            duration_ms=elapsed_ms,
            detail=str(exc)[:100],
        )
        logger.error(
            "❌ [CRASH] %s %s%s from %s failed after %.2fms: %s (req_id=%s, active=%d)",
            method,
            safe_path,
            query_str,
            client_ip,
            elapsed_ms,
            exc,
            request_id,
            active_now,
            exc_info=True,
        )
        raise
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_REQUESTS = max(0, _ACTIVE_REQUESTS - 1)
            active_remaining = _ACTIVE_REQUESTS

    # 3. Log request completion with elapsed time and status
    elapsed_ms = (time.monotonic() - start_time) * 1000.0
    if resp is None:
        resp = Response(content="", status_code=200)
    status_code = getattr(resp, "status_code", 200) or 200
    icon, label = get_status_indicator(status_code)
    status_label = f"{status_code} {label}"

    store.record_complete(
        request_id=request_id,
        status=status_label,
        status_code=status_code,
        duration_ms=elapsed_ms,
        detail=initial_detail,
    )

    logger.info(
        "%s [%d %s] %s %s%s from %s in %.2fms (req_id=%s, active=%d)",
        icon,
        status_code,
        label,
        method,
        safe_path,
        query_str,
        client_ip,
        elapsed_ms,
        request_id,
        active_remaining,
    )

    # Attach telemetry headers to response
    if hasattr(resp, "headers"):
        resp.headers.setdefault("X-Request-ID", request_id)
        resp.headers.setdefault("X-Response-Time-ms", f"{elapsed_ms:.2f}")
        resp.headers.setdefault("X-Active-Requests", str(active_remaining))

    return resp


__all__ = [
    "RequestLogStore",
    "RequestRecord",
    "get_active_requests_count",
    "get_client_ip",
    "get_request_log_store",
    "get_status_indicator",
    "logger",
    "request_lifecycle_logging_middleware",
]
