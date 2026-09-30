"""Bakong KHQR (EMVCo) generation and QR code image rendering with high-speed caching."""

from __future__ import annotations

import asyncio
import collections
import hashlib
import io
import logging
import os
import random
import threading
import time
import urllib.parse
from contextlib import suppress
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Resolve Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Resolve Configuration Defaults (Checking Environment and SETTINGS)
_settings_account_id = None
_settings_merchant_name = None
_settings_merchant_city = None
_settings_currency = None
_settings_merchant_id = None
_settings_static_url = None
_settings_static_path = None

with suppress(Exception):
    from app.core.config import SETTINGS

    _settings_account_id = getattr(SETTINGS, "BAKONG_ACCOUNT_ID", None)
    _settings_merchant_name = getattr(SETTINGS, "BAKONG_MERCHANT_NAME", None)
    _settings_merchant_city = getattr(SETTINGS, "BAKONG_MERCHANT_CITY", None)
    _settings_currency = getattr(SETTINGS, "BAKONG_CURRENCY", None)
    _settings_merchant_id = getattr(SETTINGS, "BAKONG_MERCHANT_ID", None)
    _settings_static_url = getattr(SETTINGS, "KHQR_STATIC_IMAGE_URL", None)
    _settings_static_path = getattr(SETTINGS, "KHQR_STATIC_IMAGE_PATH", None)

DEFAULT_BAKONG_ACCOUNT_ID = (
    os.getenv("BAKONG_ACCOUNT_ID")
    or _settings_account_id
    or "chuo_kimheng@bkrt"
).strip()

DEFAULT_BAKONG_MERCHANT_NAME = (
    os.getenv("BAKONG_MERCHANT_NAME")
    or _settings_merchant_name
    or "KIMHENG CHUO"
).strip()

DEFAULT_BAKONG_MERCHANT_CITY = (
    os.getenv("BAKONG_MERCHANT_CITY")
    or _settings_merchant_city
    or "Phnom Penh"
).strip()

DEFAULT_BAKONG_CURRENCY = (
    os.getenv("BAKONG_CURRENCY")
    or _settings_currency
    or "USD"
).strip().upper()

DEFAULT_BAKONG_MERCHANT_ID = (
    os.getenv("BAKONG_MERCHANT_ID")
    or _settings_merchant_id
    or ""
).strip()

STATIC_QR_IMAGE_URL = (
    os.getenv("KHQR_STATIC_IMAGE_URL")
    or _settings_static_url
    or ""
).strip()

STATIC_QR_IMAGE_PATH = (
    os.getenv("KHQR_STATIC_IMAGE_PATH")
    or _settings_static_path
    or ""
).strip()

# -----------------------------------------------------------------------------
# Dynamic Runtime Configuration Store
# -----------------------------------------------------------------------------
_CONFIG_LOCK = threading.Lock()
_RUNTIME_CONFIG: dict[str, str] = {
    "account_id": DEFAULT_BAKONG_ACCOUNT_ID,
    "merchant_name": DEFAULT_BAKONG_MERCHANT_NAME,
    "merchant_city": DEFAULT_BAKONG_MERCHANT_CITY,
    "currency": DEFAULT_BAKONG_CURRENCY,
    "merchant_id": DEFAULT_BAKONG_MERCHANT_ID,
}


def get_khqr_config() -> dict[str, str]:
    """Retrieve currently active KHQR configuration."""
    with _CONFIG_LOCK:
        return dict(_RUNTIME_CONFIG)


def update_khqr_config(
    *,
    account_id: str | None = None,
    merchant_name: str | None = None,
    merchant_city: str | None = None,
    currency: str | None = None,
    merchant_id: str | None = None,
) -> dict[str, str]:
    """Update runtime KHQR configuration without restarting the application."""
    with _CONFIG_LOCK:
        if account_id is not None and account_id.strip():
            _RUNTIME_CONFIG["account_id"] = account_id.strip()
        if merchant_name is not None and merchant_name.strip():
            _RUNTIME_CONFIG["merchant_name"] = merchant_name.strip()[:25]
        if merchant_city is not None and merchant_city.strip():
            _RUNTIME_CONFIG["merchant_city"] = merchant_city.strip()[:15]
        if currency is not None and currency.strip():
            c = currency.strip().upper()
            if c in ("USD", "KHR"):
                _RUNTIME_CONFIG["currency"] = c
        if merchant_id is not None:
            _RUNTIME_CONFIG["merchant_id"] = merchant_id.strip()[:32]
        return dict(_RUNTIME_CONFIG)


# -----------------------------------------------------------------------------
# High-Performance CRC16-CCITT Lookup Table (256 entries)
# -----------------------------------------------------------------------------
_CRC_TABLE: list[int] = []
for _i in range(256):
    _curr = _i << 8
    for _ in range(8):
        _curr = ((_curr << 1) ^ 0x1021) & 0xFFFF if (_curr & 0x8000) else ((_curr << 1) & 0xFFFF)
    _CRC_TABLE.append(_curr)


def crc16_ccitt(data: str) -> str:
    """Compute standard EMVCo CRC16-CCITT checksum with precomputed lookup table."""
    crc = 0xFFFF
    for byte in data.encode("utf-8"):
        crc = ((crc << 8) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return f"{crc:04X}"


def verify_khqr_crc(khqr_text: str) -> bool:
    """Verify that the CRC16-CCITT checksum at the end of a KHQR string is valid."""
    clean = str(khqr_text or "").strip()
    if len(clean) < 8 or "6304" not in clean:
        return False
    idx = clean.rfind("6304")
    if idx == -1 or len(clean) != idx + 8:
        return False
    data_part = clean[: idx + 4]
    expected_crc = clean[idx + 4 :].upper()
    return crc16_ccitt(data_part) == expected_crc


def _format_tlv(tag: str, value: str) -> str:
    """Format Tag-Length-Value with EMVCo 2-digit length enforcement (<= 99 bytes)."""
    val_bytes = value.encode("utf-8")
    if len(val_bytes) > 99:
        val_bytes = val_bytes[:99]
        value = val_bytes.decode("utf-8", errors="ignore")
        val_bytes = value.encode("utf-8")
    return f"{tag}{len(val_bytes):02d}{value}"


def generate_khqr_string(
    *,
    account_id: str = "",
    merchant_name: str = "",
    merchant_city: str = "",
    merchant_id: str = "",
    amount: float | int | str | None = None,
    currency: str = "",
    bill_number: str = "",
    mobile_number: str = "",
    store_label: str = "",
    reference_label: str = "",
    terminal_label: str = "BOTVOICE",
    expiration_days: float = 1.0,
) -> str:
    """Generate official Bakong KHQR EMVCo payload string compliant with NBC standards."""
    cfg = get_khqr_config()
    account_id = (account_id.strip() or cfg["account_id"])
    merchant_name = (merchant_name.strip() or cfg["merchant_name"])[:25]
    merchant_city = (merchant_city.strip() or cfg["merchant_city"])[:15]
    merchant_id = (merchant_id.strip() or cfg.get("merchant_id", ""))[:32]
    currency = (currency.strip().upper() or cfg["currency"])

    # Safe amount conversion
    parsed_amount: float | None = None
    if amount is not None:
        with suppress(ValueError, TypeError):
            val = float(amount)
            if val > 0:
                parsed_amount = val

    is_dynamic = parsed_amount is not None

    # Tag 00: Payload Format Indicator (01)
    payload = _format_tlv("00", "01")

    # Tag 01: Point of Initiation Method (12 = dynamic with amount, 11 = static)
    payload += _format_tlv("01", "12" if is_dynamic else "11")

    # Tag 29 / Tag 30: Merchant Account Information
    if merchant_id:
        sub30_acc = _format_tlv("00", account_id)
        sub30_mid = _format_tlv("01", merchant_id)
        payload += _format_tlv("30", sub30_acc + sub30_mid)
    else:
        sub29_acc = _format_tlv("00", account_id)
        payload += _format_tlv("29", sub29_acc)

    # Tag 52: Merchant Category Code (5999 = Miscellaneous/Specialty Retail)
    payload += _format_tlv("52", "5999")

    # Tag 53: Transaction Currency (840 = USD, 116 = KHR)
    currency_code = "116" if currency == "KHR" else "840"
    payload += _format_tlv("53", currency_code)

    # Tag 54: Transaction Amount
    if is_dynamic and parsed_amount is not None:
        amount_str = f"{parsed_amount:.2f}" if currency == "USD" else f"{int(round(parsed_amount))}"
        payload += _format_tlv("54", amount_str)

    # Tag 58: Country Code
    payload += _format_tlv("58", "KH")

    # Tag 59: Merchant Name
    payload += _format_tlv("59", merchant_name)

    # Tag 60: Merchant City
    payload += _format_tlv("60", merchant_city)

    # Tag 62: Additional Data Field Template (Safe size bounded to <= 99 bytes)
    tag62_val = ""
    if bill_number:
        tag62_val += _format_tlv("01", str(bill_number)[:25])
    if mobile_number:
        tag62_val += _format_tlv("02", str(mobile_number)[:25])
    if store_label:
        tag62_val += _format_tlv("03", str(store_label)[:25])
    if reference_label:
        tag62_val += _format_tlv("05", str(reference_label)[:25])
    if terminal_label:
        tag62_val += _format_tlv("07", str(terminal_label)[:25])

    if tag62_val:
        if len(tag62_val.encode("utf-8")) > 99:
            # Drop terminal and store labels if size exceeds EMVCo 99-byte limit
            tag62_val = _format_tlv("01", str(bill_number)[:25])
            if reference_label:
                tag62_val += _format_tlv("05", str(reference_label)[:25])
        payload += _format_tlv("62", tag62_val)

    # Tag 99: Timestamp & Expiration (Bakong NBC Standard)
    now_ms = int(time.time() * 1000)
    tag99_val = _format_tlv("00", str(now_ms))
    if is_dynamic:
        exp_ms = now_ms + int(max(0.05, expiration_days) * 86400 * 1000)
        tag99_val += _format_tlv("01", str(exp_ms))
    payload += _format_tlv("99", tag99_val)

    # Tag 63: CRC16-CCITT Checksum
    payload_for_crc = payload + "6304"
    crc_hex = crc16_ccitt(payload_for_crc)
    return payload_for_crc + crc_hex


def decode_khqr(khqr_text: str) -> dict[str, Any]:
    """Parse and decode an EMVCo KHQR string into a structured dictionary."""
    clean = str(khqr_text or "").strip()
    if len(clean) < 12:
        return {"valid": False, "error": "Payload too short"}

    is_crc_valid = verify_khqr_crc(clean)
    tags: dict[str, str] = {}
    idx = 0
    clean_len = len(clean)

    while idx + 4 <= clean_len:
        tag = clean[idx : idx + 2]
        try:
            length = int(clean[idx + 2 : idx + 4])
        except ValueError:
            break
        val_start = idx + 4
        val_end = val_start + length
        if val_end > clean_len:
            break
        tags[tag] = clean[val_start:val_end]
        idx = val_end

    # Parse nested Tag 29 / Tag 30
    merchant_acc = ""
    merchant_id = ""
    account_block = tags.get("29") or tags.get("30") or ""
    if account_block:
        sub_idx = 0
        while sub_idx + 4 <= len(account_block):
            stag = account_block[sub_idx : sub_idx + 2]
            try:
                slen = int(account_block[sub_idx + 2 : sub_idx + 4])
            except ValueError:
                break
            sval = account_block[sub_idx + 4 : sub_idx + 4 + slen]
            if stag == "00":
                merchant_acc = sval
            elif stag == "01":
                merchant_id = sval
            sub_idx += 4 + slen

    # Parse nested Tag 62 (Additional Data)
    bill_number = ""
    ref_label = ""
    tag62 = tags.get("62", "")
    if tag62:
        sub_idx = 0
        while sub_idx + 4 <= len(tag62):
            stag = tag62[sub_idx : sub_idx + 2]
            try:
                slen = int(tag62[sub_idx + 2 : sub_idx + 4])
            except ValueError:
                break
            sval = tag62[sub_idx + 4 : sub_idx + 4 + slen]
            if stag == "01":
                bill_number = sval
            elif stag == "05":
                ref_label = sval
            sub_idx += 4 + slen

    currency_num = tags.get("53", "")
    currency_str = "KHR" if currency_num == "116" else ("USD" if currency_num == "840" else currency_num)

    amount_str = tags.get("54")
    amount_float: float | None = None
    if amount_str:
        with suppress(ValueError):
            amount_float = float(amount_str)

    return {
        "valid": is_crc_valid,
        "is_dynamic": tags.get("01") == "12",
        "account_id": merchant_acc,
        "merchant_id": merchant_id,
        "merchant_name": tags.get("59", ""),
        "merchant_city": tags.get("60", ""),
        "currency": currency_str,
        "amount": amount_float,
        "bill_number": bill_number,
        "reference_label": ref_label,
        "raw_tags": tags,
    }


# -----------------------------------------------------------------------------
# High-Speed In-Memory QR Image Cache & Connection Pool
# -----------------------------------------------------------------------------
_QR_IMAGE_CACHE: collections.OrderedDict[str, bytes] = collections.OrderedDict()
_QR_CACHE_LOCK = threading.Lock()
_MAX_CACHE_ENTRIES = 128

_BRANDED_CARD_CACHE: bytes | None = None
_BRANDED_CARD_LOCK = threading.Lock()
_BRANDED_CARD_ATTEMPTED = False


def invalidate_branded_card_cache() -> None:
    """Clear in-memory cached static branded KHQR card."""
    global _BRANDED_CARD_CACHE, _BRANDED_CARD_ATTEMPTED
    with _BRANDED_CARD_LOCK:
        _BRANDED_CARD_CACHE = None
        _BRANDED_CARD_ATTEMPTED = False
    with _QR_CACHE_LOCK:
        _QR_IMAGE_CACHE.clear()


def set_cached_branded_card(data: bytes | None) -> None:
    """Explicitly prime or update the in-memory static branded card cache."""
    global _BRANDED_CARD_CACHE, _BRANDED_CARD_ATTEMPTED
    with _BRANDED_CARD_LOCK:
        _BRANDED_CARD_CACHE = data
        _BRANDED_CARD_ATTEMPTED = True
    with _QR_CACHE_LOCK:
        _QR_IMAGE_CACHE.clear()


_HTTP_CLIENT: Any = None
_HTTP_CLIENT_LOCK = threading.Lock()
_CLIENT_LOOP: Any = None


async def _get_shared_http_client() -> Any:
    """Provide a thread-safe, loop-aware pooled HTTP client."""
    global _HTTP_CLIENT, _CLIENT_LOOP
    if httpx is None:
        return None

    current_loop = asyncio.get_running_loop()
    with _HTTP_CLIENT_LOCK:
        if (
            _HTTP_CLIENT is None
            or getattr(_HTTP_CLIENT, "is_closed", True)
            or _CLIENT_LOOP != current_loop
        ):
            if _HTTP_CLIENT is not None and not getattr(_HTTP_CLIENT, "is_closed", True):
                with suppress(Exception):
                    await _HTTP_CLIENT.aclose()

            _HTTP_CLIENT = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={"User-Agent": "BotVoice-KHQR/2.0"},
            )
            _CLIENT_LOOP = current_loop

        return _HTTP_CLIENT


async def close_khqr_http_client() -> None:
    """Cleanly close the pooled HTTP client."""
    global _HTTP_CLIENT, _CLIENT_LOOP
    with _HTTP_CLIENT_LOCK:
        client = _HTTP_CLIENT
        _HTTP_CLIENT = None
        _CLIENT_LOOP = None

    if client is not None and not getattr(client, "is_closed", True):
        with suppress(Exception):
            await client.aclose()


async def _fetch_url_bytes(url: str, timeout: float = 6.0) -> bytes | None:
    """Fetch bytes from URL with pooled httpx if available, falling back to urllib."""
    try:
        client = await _get_shared_http_client()
        if client is not None:
            resp = await client.get(url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                return resp.content
    except Exception as exc:
        logger.debug("httpx fetch failed for %s (%s), trying urllib", url, exc)

    def _sync_fetch() -> bytes | None:
        try:
            import ssl
            import urllib.request

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "BotVoice-KHQR/2.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
                status = getattr(response, "status", getattr(response, "code", 200))
                if status == 200:
                    data = response.read()
                    if data:
                        return data
        except Exception as err:
            logger.debug("urllib fetch failed for %s: %s", url, err)
        return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_fetch)


def _read_static_qr_file(paths: list[str]) -> bytes | None:
    seen = set()
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    data = f.read()
                    if data:
                        return data
            except Exception as e:
                logger.warning("Failed to read static QR image path %s: %s", p, e)
    return None


async def get_static_khqr_card() -> bytes | None:
    """Retrieve the static branded KHQR card with cached disk lookup."""
    global _BRANDED_CARD_ATTEMPTED

    with _BRANDED_CARD_LOCK:
        if _BRANDED_CARD_CACHE is not None:
            return _BRANDED_CARD_CACHE
        if _BRANDED_CARD_ATTEMPTED:
            return None

    candidate_paths = [
        STATIC_QR_IMAGE_PATH,
        os.path.join(PROJECT_ROOT, "assets", "my_khqr.webp"),
        os.path.join(PROJECT_ROOT, "assets", "my_khqr.png"),
        os.path.join(PROJECT_ROOT, "assets", "my_khqr.jpg"),
        os.path.join(PROJECT_ROOT, "static", "my_khqr.webp"),
        os.path.join(PROJECT_ROOT, "static", "khqr.png"),
        os.path.join(PROJECT_ROOT, "static", "khqr.jpg"),
        os.path.join(PROJECT_ROOT, "static", "khqr.jpeg"),
        os.path.join(os.getcwd(), "assets", "my_khqr.webp"),
        os.path.join(os.getcwd(), "assets", "my_khqr.png"),
        os.path.join(os.getcwd(), "assets", "my_khqr.jpg"),
        os.path.join(os.getcwd(), "static", "my_khqr.webp"),
        os.path.join(os.getcwd(), "static", "khqr.png"),
        os.path.join(os.getcwd(), "static", "khqr.jpg"),
        os.path.join(os.getcwd(), "static", "khqr.jpeg"),
        os.path.join(os.getcwd(), "static", "aba.png"),
        os.path.join(os.getcwd(), "static", "aba.jpg"),
        os.path.join(os.getcwd(), "static", "qr.png"),
        os.path.join(os.getcwd(), "static", "qr.jpg"),
    ]

    loop = asyncio.get_running_loop()
    static_bytes = await loop.run_in_executor(None, _read_static_qr_file, candidate_paths)
    if static_bytes:
        set_cached_branded_card(static_bytes)
        return static_bytes

    if STATIC_QR_IMAGE_URL:
        url_bytes = await _fetch_url_bytes(STATIC_QR_IMAGE_URL, timeout=8.0)
        if url_bytes:
            set_cached_branded_card(url_bytes)
            return url_bytes

    with _BRANDED_CARD_LOCK:
        _BRANDED_CARD_ATTEMPTED = True

    return None


def _render_local_qr_sync(khqr_text: str) -> bytes | None:
    """Synchronous worker for CPU-bound QR image generation."""
    try:
        import qrcode

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(khqr_text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#000000", back_color="#ffffff")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.debug("Local qrcode generation error: %s", exc)
        return None


async def get_khqr_qr_image(khqr_text: str) -> bytes | None:
    """Render or retrieve cached QR code image bytes for a given KHQR string."""
    clean_text = str(khqr_text or "").strip()
    if not clean_text:
        return await get_static_khqr_card()

    cache_key = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

    with _QR_CACHE_LOCK:
        if cache_key in _QR_IMAGE_CACHE:
            _QR_IMAGE_CACHE.move_to_end(cache_key)
            return _QR_IMAGE_CACHE[cache_key]

    # 1. Primary: Local qrcode generation offloaded to thread pool
    loop = asyncio.get_running_loop()
    local_bytes = await loop.run_in_executor(None, _render_local_qr_sync, clean_text)
    if local_bytes:
        with _QR_CACHE_LOCK:
            if len(_QR_IMAGE_CACHE) >= _MAX_CACHE_ENTRIES:
                _QR_IMAGE_CACHE.popitem(last=False)
            _QR_IMAGE_CACHE[cache_key] = local_bytes
        return local_bytes

    # 2. Secondary: Online QR generation API fallback
    encoded_data = urllib.parse.quote(clean_text, safe="")
    api_urls = [
        f"https://api.qrserver.com/v1/create-qr-code/?size=350x350&margin=10&data={encoded_data}",
        f"https://quickchart.io/qr?text={encoded_data}&size=350&margin=1",
    ]

    for api_url in api_urls:
        api_bytes = await _fetch_url_bytes(api_url, timeout=6.0)
        if api_bytes and len(api_bytes) > 20:
            with _QR_CACHE_LOCK:
                if len(_QR_IMAGE_CACHE) >= _MAX_CACHE_ENTRIES:
                    _QR_IMAGE_CACHE.popitem(last=False)
                _QR_IMAGE_CACHE[cache_key] = api_bytes
            return api_bytes

    # 3. Fallback: Static branded KHQR card
    static_bytes = await get_static_khqr_card()
    if static_bytes:
        with _QR_CACHE_LOCK:
            if len(_QR_IMAGE_CACHE) >= _MAX_CACHE_ENTRIES:
                _QR_IMAGE_CACHE.popitem(last=False)
            _QR_IMAGE_CACHE[cache_key] = static_bytes
        return static_bytes

    return None


class BakongKHQR:
    """Helper wrapper for Bakong KHQR operations."""

    @staticmethod
    def generate(
        amount: float | int | str | None = None,
        currency: str = "USD",
        user_id: int | str = "",
        tier: str = "coffee",
        merchant_id: str = "",
        expiration_days: float = 1.0,
    ) -> tuple[str, str]:
        """Generate collision-free KHQR string and bill reference. Returns (khqr_text, bill_no)."""
        tier_clean = tier[:4].upper()
        now_ms = int(time.time() * 1000)
        rand_entropy = f"{random.randint(0, 0xFFF):03x}"
        bill_no = f"{tier_clean}{now_ms:x}{rand_entropy}"[:16]

        khqr = generate_khqr_string(
            amount=amount,
            currency=currency,
            merchant_id=merchant_id,
            bill_number=bill_no,
            reference_label=str(user_id) if user_id else "",
            expiration_days=expiration_days,
        )
        return khqr, bill_no

    @staticmethod
    def get_md5(khqr_text: str) -> str:
        """Compute the MD5 hash of the KHQR payload required by Bakong Open API."""
        return hashlib.md5(khqr_text.encode("utf-8")).hexdigest()

    @staticmethod
    def verify_crc(khqr_text: str) -> bool:
        """Verify the CRC16-CCITT checksum of a KHQR string."""
        return verify_khqr_crc(khqr_text)

    @staticmethod
    def decode(khqr_text: str) -> dict[str, Any]:
        """Parse and decode an EMVCo KHQR string into structured attributes."""
        return decode_khqr(khqr_text)


def generate_khqr_payload(
    amount: float | int | str | None = None,
    currency: str = "USD",
    user_id: int | str = "",
    tier: str = "coffee",
    bill_number: str = "",
    merchant_id: str = "",
    expiration_days: float = 1.0,
) -> dict[str, Any]:
    """Generate KHQR payload dict containing qr_string, bill_no, and md5."""
    if bill_number:
        bill_no = str(bill_number)[:25]
        khqr = generate_khqr_string(
            amount=amount,
            currency=currency,
            merchant_id=merchant_id,
            bill_number=bill_no,
            reference_label=str(user_id) if user_id else "",
            expiration_days=expiration_days,
        )
    else:
        khqr, bill_no = BakongKHQR.generate(
            amount=amount,
            currency=currency,
            user_id=user_id,
            tier=tier,
            merchant_id=merchant_id,
            expiration_days=expiration_days,
        )

    return {
        "qr_string": khqr,
        "bill_no": bill_no,
        "md5": hashlib.md5(khqr.encode("utf-8")).hexdigest(),
    }


__all__ = [
    "DEFAULT_BAKONG_ACCOUNT_ID",
    "DEFAULT_BAKONG_CURRENCY",
    "DEFAULT_BAKONG_MERCHANT_CITY",
    "DEFAULT_BAKONG_MERCHANT_ID",
    "DEFAULT_BAKONG_MERCHANT_NAME",
    "PROJECT_ROOT",
    "STATIC_QR_IMAGE_PATH",
    "STATIC_QR_IMAGE_URL",
    "BakongKHQR",
    "close_khqr_http_client",
    "crc16_ccitt",
    "decode_khqr",
    "generate_khqr_payload",
    "generate_khqr_string",
    "get_khqr_config",
    "get_khqr_qr_image",
    "get_static_khqr_card",
    "invalidate_branded_card_cache",
    "set_cached_branded_card",
    "update_khqr_config",
    "verify_khqr_crc",
]