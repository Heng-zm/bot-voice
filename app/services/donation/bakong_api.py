"""Bakong Open API client for real-time transaction verification and KHQR payment checking.

Integrates with the National Bank of Cambodia (NBC) Bakong Open API gateway
to automatically verify KHQR payments, enabling instant donation approval
and automated AI voice blessing delivery.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

# Fallback gateway URL
DEFAULT_BAKONG_BASE_URL = "https://api-bakong.nbc.gov.kh/v1"


def get_bakong_base_url() -> str:
    """Dynamically retrieve the Bakong Open API gateway base URL."""
    url = (
        os.getenv("BAKONG_BASE_URL")
        or os.getenv("BAKONG_GATEWAY_URL")
        or ""
    ).strip()

    if not url:
        with suppress(Exception):
            from app.core.config import SETTINGS

            url = (
                getattr(SETTINGS, "BAKONG_BASE_URL", None)
                or getattr(SETTINGS, "BAKONG_GATEWAY_URL", None)
                or ""
            ).strip()

    return (url or DEFAULT_BAKONG_BASE_URL).rstrip("/")


# Exported module-level constants for compatibility
BAKONG_BASE_URL = get_bakong_base_url()
CHECK_TRANSACTION_BY_MD5_URL = f"{BAKONG_BASE_URL}/check_transaction_by_md5"
GENERATE_DEEPLINK_URL = f"{BAKONG_BASE_URL}/generate_deeplink_by_qr"


def _clean_token(token: str) -> str:
    """Strip unnecessary whitespace and unintentional 'Bearer ' prefixes."""
    t = str(token or "").strip()
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t


def get_bakong_token() -> str:
    """Retrieve configured Bakong Open API JWT token from environment or SETTINGS.

    Supports BAKONG_OPEN_API_TOKEN (primary), BAKONG_DEVELOPER_TOKEN, and BAKONG_TOKEN.
    """
    token = (
        os.getenv("BAKONG_OPEN_API_TOKEN")
        or os.getenv("BAKONG_DEVELOPER_TOKEN")
        or os.getenv("BAKONG_TOKEN")
        or ""
    ).strip()

    if not token:
        with suppress(Exception):
            from app.core.config import SETTINGS

            token = (
                getattr(SETTINGS, "BAKONG_OPEN_API_TOKEN", None)
                or getattr(SETTINGS, "BAKONG_DEVELOPER_TOKEN", None)
                or getattr(SETTINGS, "BAKONG_TOKEN", None)
                or ""
            ).strip()

    return _clean_token(token)


def is_bakong_api_configured() -> bool:
    """Return True if a Bakong Open API token is configured."""
    return bool(get_bakong_token())


def decode_token_payload(token: str = "") -> dict[str, Any]:
    """Inspect and decode the JWT payload without verifying signature.

    Returns merchant metadata, issued date, expiry timestamp, and remaining validity.
    """
    token = _clean_token(token or get_bakong_token())
    if not token:
        return {"configured": False, "error": "No token provided"}

    parts = token.split(".")
    if len(parts) != 3:
        return {"configured": False, "error": "Malformed JWT token structure"}

    try:
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))

        raw_exp = payload.get("exp")
        raw_iat = payload.get("iat")
        exp_ts: float | None = None
        iat_ts: float | None = None

        if raw_exp is not None:
            with suppress(ValueError, TypeError):
                exp_ts = float(raw_exp)
        if raw_iat is not None:
            with suppress(ValueError, TypeError):
                iat_ts = float(raw_iat)

        data_block = payload.get("data", {})
        if not isinstance(data_block, dict):
            data_block = {}

        merchant_id = (
            data_block.get("id")
            or data_block.get("merchantId")
            or data_block.get("developerId")
            or payload.get("sub", "")
        )
        email = data_block.get("email") or payload.get("email", "")

        now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        now_ts = now_utc.timestamp()

        exp_dt = datetime.datetime.fromtimestamp(exp_ts, tz=datetime.timezone.utc) if exp_ts else None
        iat_dt = datetime.datetime.fromtimestamp(iat_ts, tz=datetime.timezone.utc) if iat_ts else None

        is_expired = now_ts > exp_ts if exp_ts else False
        seconds_left = max(0.0, exp_ts - now_ts) if exp_ts and not is_expired else 0.0
        days_left = round(seconds_left / 86400.0, 1) if exp_ts and not is_expired else 0.0

        return {
            "configured": True,
            "merchant_id": str(merchant_id or ""),
            "email": str(email or ""),
            "issued_at": iat_dt.strftime("%Y-%m-%d %H:%M:%S UTC") if iat_dt else "",
            "expires_at": exp_dt.strftime("%Y-%m-%d %H:%M:%S UTC") if exp_dt else "",
            "is_expired": is_expired,
            "seconds_left": round(seconds_left, 1),
            "days_left": days_left,
            "raw_payload": payload,
        }
    except Exception as exc:
        logger.warning("Failed to decode Bakong JWT payload: %s", exc)
        return {"configured": False, "error": f"Invalid token encoding: {exc}"}


# Friendly alias for token inspection
inspect_bakong_token = decode_token_payload

_SHARED_BAKONG_CLIENT: Any = None
_SHARED_BAKONG_LOCK = threading.Lock()
_CLIENT_LOOP: Any = None
_SHARED_SSL_CONTEXT: Any = None


def _get_ssl_context() -> Any:
    global _SHARED_SSL_CONTEXT
    if _SHARED_SSL_CONTEXT is None:
        import ssl

        _SHARED_SSL_CONTEXT = ssl.create_default_context()
    return _SHARED_SSL_CONTEXT


async def _get_bakong_client(timeout: float = 12.0) -> Any:
    """Retrieve or initialize an httpx.AsyncClient safely bound to the current event loop."""
    global _SHARED_BAKONG_CLIENT, _CLIENT_LOOP
    current_loop = asyncio.get_running_loop()

    with _SHARED_BAKONG_LOCK:
        if (
            _SHARED_BAKONG_CLIENT is None
            or getattr(_SHARED_BAKONG_CLIENT, "is_closed", True)
            or _CLIENT_LOOP != current_loop
        ):
            import httpx

            if _SHARED_BAKONG_CLIENT is not None and not getattr(_SHARED_BAKONG_CLIENT, "is_closed", True):
                with suppress(Exception):
                    await _SHARED_BAKONG_CLIENT.aclose()

            _SHARED_BAKONG_CLIENT = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            _CLIENT_LOOP = current_loop

        return _SHARED_BAKONG_CLIENT


async def close_bakong_client() -> None:
    """Cleanly close the pooled HTTP client on application shutdown."""
    global _SHARED_BAKONG_CLIENT, _CLIENT_LOOP
    with _SHARED_BAKONG_LOCK:
        client = _SHARED_BAKONG_CLIENT
        _SHARED_BAKONG_CLIENT = None
        _CLIENT_LOOP = None

    if client is not None and not getattr(client, "is_closed", True):
        with suppress(Exception):
            await client.aclose()


async def _execute_http_post(
    endpoint_path: str,
    payload: dict[str, Any],
    token: str,
    timeout: float = 12.0,
) -> tuple[int, dict[str, Any]]:
    """Execute an authenticated HTTP POST against the Bakong Open API gateway."""
    token = _clean_token(token)
    base_url = get_bakong_base_url()
    url = f"{base_url}/{endpoint_path.lstrip('/')}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "BotVoice-BakongOpenAPI/2.0",
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    # 1. Try pooled httpx client with explicit per-request timeout
    try:
        client = await _get_bakong_client(timeout=timeout)
        resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text}
        return resp.status_code, data
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as exc:
        logger.debug("Pooled httpx POST failed (%s), falling back to urllib", exc)

    # 2. Resilient standard library fallback
    def _sync_request() -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        try:
            ctx = _get_ssl_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
                status = response.status
                raw_body = response.read().decode("utf-8", errors="replace")
                try:
                    return status, json.loads(raw_body)
                except Exception:
                    return status, {"raw_text": raw_body}
        except urllib.error.HTTPError as err:
            raw_body = err.read().decode("utf-8", errors="replace")
            try:
                return err.code, json.loads(raw_body)
            except Exception:
                return err.code, {"raw_text": raw_body}
        except Exception as exc:
            return 500, {"error": str(exc)}

    return await asyncio.to_thread(_sync_request)


async def check_transaction_by_md5(
    md5_hash: str,
    token: str = "",
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Query Bakong Open API by MD5 hash of the generated KHQR payload.

    Returns:
    {
        "success": bool,          # True if responseCode == 0 (payment confirmed)
        "response_code": int,     # 0 = success, 1 = not found / pending
        "response_message": str,  # Gateway message
        "data": dict | None,      # Transaction details
        "http_status": int,
        "raw": dict,
    }
    """
    token = _clean_token(token or get_bakong_token())
    if not token:
        return {
            "success": False,
            "response_code": -1,
            "response_message": "Bakong Open API token is not configured.",
            "data": None,
            "http_status": 0,
            "raw": {},
        }

    clean_md5 = md5_hash.strip().lower()
    payload = {"md5": clean_md5}

    try:
        http_status, resp_data = await _execute_http_post(
            "check_transaction_by_md5",
            payload,
            token,
            timeout=timeout,
        )

        resp_code = resp_data.get("responseCode")
        if resp_code is None:
            resp_code = -1 if http_status != 200 else 1

        resp_msg = str(resp_data.get("responseMessage") or resp_data.get("message") or "")
        tx_data = resp_data.get("data")

        # responseCode 0 with data indicates confirmed payment
        is_success = (http_status == 200) and (resp_code == 0) and (tx_data is not None)

        return {
            "success": is_success,
            "response_code": resp_code,
            "response_message": resp_msg,
            "data": tx_data if is_success else None,
            "http_status": http_status,
            "raw": resp_data,
        }
    except Exception as exc:
        logger.error("Error querying Bakong Open API md5=%s: %s", clean_md5, exc)
        return {
            "success": False,
            "response_code": -1,
            "response_message": f"Network or gateway error: {exc}",
            "data": None,
            "http_status": 500,
            "raw": {},
        }


async def verify_khqr_payment(
    khqr_or_md5: str,
    token: str = "",
    timeout: float = 12.0,
) -> tuple[bool, dict[str, Any] | None, str]:
    """Verify payment for a given KHQR string or precomputed MD5 hash.

    Returns:
        (is_paid: bool, transaction_data: dict | None, message: str)
    """
    clean_input = str(khqr_or_md5 or "").strip()
    if not clean_input:
        return False, None, "Invalid empty KHQR input."

    # Avoid double-hashing if caller passed a 32-character hex MD5 hash directly
    if len(clean_input) == 32 and all(c in "0123456789abcdefABCDEF" for c in clean_input):
        md5_hash = clean_input.lower()
    else:
        md5_hash = hashlib.md5(clean_input.encode("utf-8")).hexdigest()

    result = await check_transaction_by_md5(md5_hash, token=token, timeout=timeout)

    is_paid = result.get("success", False)
    data = result.get("data")
    msg = result.get("response_message", "")

    return is_paid, data, msg


async def test_connection(token: str = "") -> dict[str, Any]:
    """Test connectivity to Bakong Open API gateway and validate token credentials."""
    token = _clean_token(token or get_bakong_token())
    token_meta = decode_token_payload(token)

    if not token_meta.get("configured"):
        return {
            "ok": False,
            "error": token_meta.get("error", "No token configured"),
            "latency_ms": 0.0,
            "merchant_id": "",
            "expires_at": "",
            "is_expired": True,
        }

    # Ping gateway with dummy MD5 — HTTP 200 with responseCode 1 confirms active handshake
    t0 = time.perf_counter()
    res = await check_transaction_by_md5("00000000000000000000000000000000", token=token, timeout=10.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    is_authenticated = (res.get("http_status") == 200) and not token_meta.get("is_expired", False)
    msg = res.get("response_message") or ""

    return {
        "ok": is_authenticated,
        "latency_ms": round(latency_ms, 1),
        "merchant_id": token_meta.get("merchant_id", ""),
        "expires_at": token_meta.get("expires_at", ""),
        "is_expired": token_meta.get("is_expired", False),
        "days_left": token_meta.get("days_left", 0.0),
        "message": msg if is_authenticated else f"HTTP {res.get('http_status')}: {msg}",
    }


# In-memory deep link cache with 2-hour TTL
_DEEPLINK_CACHE: dict[str, tuple[dict[str, str], float]] = {}
_DEEPLINK_CACHE_LOCK = threading.Lock()
_DEEPLINK_CACHE_TTL = 7200.0  # 2 hours


async def generate_deeplink_by_qr(
    khqr_text: str,
    *,
    app_name: str = "Bot Voice",
    callback_url: str = "https://t.me/khmer_voice_bot",
    app_icon_url: str = "https://bakong.nbc.gov.kh/images/logo.svg",
    token: str = "",
    timeout: float = 8.0,
) -> dict[str, Any] | None:
    """Generate 1-tap mobile banking payment deep link from NBC Bakong Open API.

    Returns dict with 'shortLink' and 'fullLink', or None if unavailable/failed.
    """
    clean_qr = str(khqr_text or "").strip()
    if not clean_qr:
        return None

    token = _clean_token(token or get_bakong_token())
    if not token:
        return None

    md5_hash = hashlib.md5(clean_qr.encode("utf-8")).hexdigest()
    now = time.monotonic()

    with _DEEPLINK_CACHE_LOCK:
        if md5_hash in _DEEPLINK_CACHE:
            cached_data, cached_at = _DEEPLINK_CACHE[md5_hash]
            if (now - cached_at) < _DEEPLINK_CACHE_TTL:
                return dict(cached_data)

    payload = {
        "qr": clean_qr,
        "sourceInfo": {
            "appIconUrl": app_icon_url,
            "appName": app_name,
            "appDeepLinkCallback": callback_url,
        },
    }

    try:
        status, resp_data = await _execute_http_post(
            "generate_deeplink_by_qr",
            payload=payload,
            token=token,
            timeout=timeout,
        )
        if status == 200 and isinstance(resp_data, dict) and resp_data.get("responseCode") == 0:
            data = resp_data.get("data")
            if isinstance(data, dict) and data.get("shortLink"):
                res = {
                    "shortLink": data["shortLink"],
                    "fullLink": data.get("fullLink", data["shortLink"]),
                }
                with _DEEPLINK_CACHE_LOCK:
                    if len(_DEEPLINK_CACHE) > 300:
                        _DEEPLINK_CACHE.clear()
                    _DEEPLINK_CACHE[md5_hash] = (res, now)
                return res
    except Exception as exc:
        logger.debug("Failed to generate NBC Bakong deeplink: %s", exc)

    return None


__all__ = [
    "BAKONG_BASE_URL",
    "CHECK_TRANSACTION_BY_MD5_URL",
    "DEFAULT_BAKONG_BASE_URL",
    "GENERATE_DEEPLINK_URL",
    "check_transaction_by_md5",
    "close_bakong_client",
    "decode_token_payload",
    "generate_deeplink_by_qr",
    "get_bakong_base_url",
    "get_bakong_token",
    "inspect_bakong_token",
    "is_bakong_api_configured",
    "test_connection",
    "verify_khqr_payment",
]