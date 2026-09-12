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
import time
import urllib.error
import urllib.request
import threading
from typing import Any

logger = logging.getLogger(__name__)

BAKONG_BASE_URL = os.getenv("BAKONG_BASE_URL", "https://api-bakong.nbc.gov.kh/v1").rstrip("/")
CHECK_TRANSACTION_BY_MD5_URL = f"{BAKONG_BASE_URL}/check_transaction_by_md5"
GENERATE_DEEPLINK_URL = f"{BAKONG_BASE_URL}/generate_deeplink_by_qr"


def get_bakong_token() -> str:
    """Retrieve configured Bakong Open API JWT token from environment."""
    return os.getenv("BAKONG_OPEN_API_TOKEN", "").strip()


def is_bakong_api_configured() -> bool:
    """Return True if a Bakong Open API token is configured."""
    return bool(get_bakong_token())


def decode_token_payload(token: str = "") -> dict[str, Any]:
    """Inspect and decode the JWT payload without verifying signature.

    Returns merchant id, issued at, expiration timestamp, and expiry status.
    """
    token = (token or get_bakong_token()).strip()
    if not token:
        return {"configured": False, "error": "No token provided"}

    parts = token.split(".")
    if len(parts) != 3:
        return {"configured": False, "error": "Malformed JWT token structure"}

    try:
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        exp_ts = payload.get("exp", 0)
        iat_ts = payload.get("iat", 0)
        data_block = payload.get("data", {})
        merchant_id = data_block.get("id", "") if isinstance(data_block, dict) else ""

        now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        exp_dt = datetime.datetime.fromtimestamp(exp_ts, tz=datetime.timezone.utc) if exp_ts else None
        iat_dt = datetime.datetime.fromtimestamp(iat_ts, tz=datetime.timezone.utc) if iat_ts else None
        is_expired = now_utc.timestamp() > exp_ts if exp_ts else False

        return {
            "configured": True,
            "merchant_id": merchant_id,
            "issued_at": iat_dt.strftime("%Y-%m-%d %H:%M:%S UTC") if iat_dt else "",
            "expires_at": exp_dt.strftime("%Y-%m-%d %H:%M:%S UTC") if exp_dt else "",
            "is_expired": is_expired,
            "raw_payload": payload,
        }
    except Exception as exc:
        logger.warning("Failed to decode Bakong JWT payload: %s", exc)
        return {"configured": False, "error": f"Invalid token encoding: {exc}"}


async def _execute_http_post(
    url: str,
    payload: dict[str, Any],
    token: str,
    timeout: float = 12.0,
) -> tuple[int, dict[str, Any]]:
    """Execute HTTP POST request against Bakong Open API gateway.

    Uses httpx when available, with standard library urllib fallback.
    """
    token = token.strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "BotVoice-BakongOpenAPI/2.0",
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    # 1. Try httpx if available
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": resp.text}
            return resp.status_code, data
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("httpx POST failed, attempting urllib fallback: %s", exc)

    # 2. Resilient standard library fallback
    def _sync_request() -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
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

    Returns a standardized dictionary:
    {
        "success": bool,          # True if responseCode == 0 (payment confirmed)
        "response_code": int,     # 0 = success, 1 = not found / pending
        "response_message": str,  # Gateway message
        "data": dict | None,      # Transaction details (hash, from, to, amount, etc.)
        "http_status": int,
    }
    """
    token = token.strip() or get_bakong_token()
    if not token:
        return {
            "success": False,
            "response_code": -1,
            "response_message": "Bakong Open API token is not configured.",
            "data": None,
            "http_status": 0,
        }

    clean_md5 = md5_hash.strip().lower()
    payload = {"md5": clean_md5}

    try:
        http_status, resp_data = await _execute_http_post(
            CHECK_TRANSACTION_BY_MD5_URL,
            payload,
            token,
            timeout=timeout,
        )

        resp_code = resp_data.get("responseCode")
        if resp_code is None:
            resp_code = -1 if http_status != 200 else 1

        resp_msg = resp_data.get("responseMessage") or resp_data.get("message") or ""
        tx_data = resp_data.get("data")

        # responseCode 0 indicates payment successfully received in Bakong
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
        }


async def verify_khqr_payment(
    khqr_text: str,
    token: str = "",
    timeout: float = 12.0,
) -> tuple[bool, dict[str, Any] | None, str]:
    """Verify payment for a given KHQR string.

    Computes MD5 hash of raw KHQR text according to Bakong specification
    and checks transaction status.

    Returns:
        (is_paid: bool, transaction_data: dict | None, message: str)
    """
    if not khqr_text:
        return False, None, "Invalid empty KHQR string."

    md5_hash = hashlib.md5(khqr_text.encode("utf-8")).hexdigest()
    result = await check_transaction_by_md5(md5_hash, token=token, timeout=timeout)

    is_paid = result.get("success", False)
    data = result.get("data")
    msg = result.get("response_message", "")

    return is_paid, data, msg


async def test_connection(token: str = "") -> dict[str, Any]:
    """Test connectivity to Bakong Open API gateway and validate token credentials."""
    token = token.strip() or get_bakong_token()
    token_meta = decode_token_payload(token)

    if not token_meta.get("configured"):
        return {
            "ok": False,
            "error": token_meta.get("error", "No token configured"),
            "latency_ms": 0.0,
            "merchant_id": "",
            "expires_at": "",
        }

    # Ping with dummy MD5 — HTTP 200 with responseCode 1 proves valid token & endpoint
    t0 = time.perf_counter()
    res = await check_transaction_by_md5("00000000000000000000000000000000", token=token, timeout=10.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # HTTP 200 or 404 (with valid JSON) indicates authenticated gateway handshake
    is_authenticated = res.get("http_status") == 200
    msg = res.get("response_message") or ""

    return {
        "ok": is_authenticated,
        "latency_ms": round(latency_ms, 1),
        "merchant_id": token_meta.get("merchant_id", ""),
        "expires_at": token_meta.get("expires_at", ""),
        "is_expired": token_meta.get("is_expired", False),
        "message": msg if is_authenticated else f"HTTP {res.get('http_status')}: {msg}",
    }


_DEEPLINK_CACHE: dict[str, dict[str, str]] = {}
_DEEPLINK_CACHE_LOCK = threading.Lock()


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
    if not khqr_text:
        return None

    token = (token or get_bakong_token()).strip()
    if not token:
        return None

    md5_hash = hashlib.md5(khqr_text.encode("utf-8")).hexdigest()
    with _DEEPLINK_CACHE_LOCK:
        if md5_hash in _DEEPLINK_CACHE:
            return dict(_DEEPLINK_CACHE[md5_hash])

    payload = {
        "qr": khqr_text,
        "sourceInfo": {
            "appIconUrl": app_icon_url,
            "appName": app_name,
            "appDeepLinkCallback": callback_url,
        },
    }

    try:
        status, resp_data = await _execute_http_post(
            GENERATE_DEEPLINK_URL,
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
                    _DEEPLINK_CACHE[md5_hash] = res
                return res
    except Exception as exc:
        logger.debug("Failed to generate NBC Bakong deeplink: %s", exc)

    return None

