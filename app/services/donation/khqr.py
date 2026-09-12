"""Bakong KHQR (EMVCo) generation and QR code image rendering with high-speed caching."""

from __future__ import annotations

import asyncio
import collections
import hashlib
import io
import logging
import os
import threading
import time
import urllib.parse

import httpx

logger = logging.getLogger(__name__)

# Default Bakong configuration (configured for user chuo_kimheng@bkrt)
DEFAULT_BAKONG_ACCOUNT_ID = os.getenv("BAKONG_ACCOUNT_ID", "chuo_kimheng@bkrt").strip() or "chuo_kimheng@bkrt"
DEFAULT_BAKONG_MERCHANT_NAME = os.getenv("BAKONG_MERCHANT_NAME", "KIMHENG CHUO").strip() or "KIMHENG CHUO"
DEFAULT_BAKONG_MERCHANT_CITY = os.getenv("BAKONG_MERCHANT_CITY", "Phnom Penh").strip() or "Phnom Penh"
DEFAULT_BAKONG_CURRENCY = os.getenv("BAKONG_CURRENCY", "USD").strip().upper() or "USD"
DEFAULT_BAKONG_MERCHANT_ID = os.getenv("BAKONG_MERCHANT_ID", "").strip()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STATIC_QR_IMAGE_URL = os.getenv("KHQR_STATIC_IMAGE_URL", "").strip()
STATIC_QR_IMAGE_PATH = os.getenv("KHQR_STATIC_IMAGE_PATH", "").strip()

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
# 8x faster than bit-by-bit calculation; zero heap allocations during hashing
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


def _format_tlv(tag: str, value: str) -> str:
    """Format Tag-Length-Value according to EMVCo specification."""
    val_bytes = value.encode("utf-8")
    length = len(val_bytes)
    return f"{tag}{length:02d}{value}"


def generate_khqr_string(
    *,
    account_id: str = "",
    merchant_name: str = "",
    merchant_city: str = "",
    merchant_id: str = "",
    amount: float | None = None,
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

    # Tag 00: Payload Format Indicator (01)
    payload = _format_tlv("00", "01")

    # Tag 01: Point of Initiation Method (12 = dynamic with amount, 11 = static)
    is_dynamic = amount is not None and amount > 0
    payload += _format_tlv("01", "12" if is_dynamic else "11")

    # Tag 29 / Tag 30: Merchant Account Information
    # If merchant_id is specified -> Tag 30 (Merchant)
    # Otherwise -> Tag 29 (Individual) with subtag 00 containing account_id
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
    if is_dynamic and amount is not None:
        amount_str = f"{amount:.2f}" if currency == "USD" else f"{int(round(amount))}"
        payload += _format_tlv("54", amount_str)

    # Tag 58: Country Code
    payload += _format_tlv("58", "KH")

    # Tag 59: Merchant Name
    payload += _format_tlv("59", merchant_name)

    # Tag 60: Merchant City
    payload += _format_tlv("60", merchant_city)

    # Tag 62: Additional Data Field Template
    tag62_val = ""
    if bill_number:
        tag62_val += _format_tlv("01", bill_number[:25])
    if mobile_number:
        tag62_val += _format_tlv("02", mobile_number[:25])
    if store_label:
        tag62_val += _format_tlv("03", store_label[:25])
    if reference_label:
        tag62_val += _format_tlv("05", reference_label[:25])
    if terminal_label:
        tag62_val += _format_tlv("07", terminal_label[:25])

    if tag62_val:
        payload += _format_tlv("62", tag62_val)

    # Tag 99: Timestamp & Expiration (NBC EMVCo Standard)
    now_ms = int(time.time() * 1000)
    tag99_val = _format_tlv("00", str(now_ms))
    if is_dynamic:
        exp_ms = now_ms + int(max(0.1, expiration_days) * 86400 * 1000)
        tag99_val += _format_tlv("01", str(exp_ms))
    payload += _format_tlv("99", tag99_val)

    # Tag 63: CRC16-CCITT (Checksum of payload up to 6304)
    payload_for_crc = payload + "6304"
    crc_hex = crc16_ccitt(payload_for_crc)
    return payload_for_crc + crc_hex


# -----------------------------------------------------------------------------
# High-Speed In-Memory QR Image Cache & Connection Pool
# -----------------------------------------------------------------------------
_QR_IMAGE_CACHE: collections.OrderedDict[str, bytes] = collections.OrderedDict()
_QR_CACHE_LOCK = threading.Lock()
_MAX_CACHE_ENTRIES = 64

_HTTP_CLIENT: httpx.AsyncClient | None = None
_HTTP_LOCK = asyncio.Lock() if "asyncio" in globals() else threading.Lock()


async def _get_shared_http_client() -> httpx.AsyncClient:
    """Provide a persistent pooled HTTP client for QR image retrieval."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"User-Agent": "BotVoice-KHQR/2.0"},
        )
    return _HTTP_CLIENT


def _read_static_qr_file(paths: list[str]) -> bytes | None:
    for p in paths:
        if p and os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    data = f.read()
                    if data:
                        return data
            except Exception as e:
                logger.warning("Failed to read static QR image path %s: %s", p, e)
    return None


async def get_khqr_qr_image(khqr_text: str) -> bytes | None:
    """Render or retrieve cached QR code image bytes (PNG) for a given KHQR string."""
    cache_key = hashlib.sha256(khqr_text.encode("utf-8")).hexdigest()

    with _QR_CACHE_LOCK:
        if cache_key in _QR_IMAGE_CACHE:
            _QR_IMAGE_CACHE.move_to_end(cache_key)
            return _QR_IMAGE_CACHE[cache_key]

    is_dynamic_amount = "010212" in khqr_text

    candidate_paths = [
        STATIC_QR_IMAGE_PATH,
        os.path.join(PROJECT_ROOT, "asset", "my_khqr.webp"),
        os.path.join(PROJECT_ROOT, "asset", "my_khqr.png"),
        os.path.join(PROJECT_ROOT, "asset", "my_khqr.jpg"),
        os.path.join(os.getcwd(), "asset", "my_khqr.webp"),
        os.path.join(os.getcwd(), "asset", "my_khqr.png"),
        os.path.join(os.getcwd(), "asset", "my_khqr.jpg"),
        os.path.join(PROJECT_ROOT, "static", "my_khqr.webp"),
        os.path.join(PROJECT_ROOT, "static", "khqr.png"),
        os.path.join(PROJECT_ROOT, "static", "khqr.jpg"),
        os.path.join(PROJECT_ROOT, "static", "khqr.jpeg"),
        os.path.join(os.getcwd(), "static", "my_khqr.webp"),
        os.path.join(os.getcwd(), "static", "khqr.png"),
        os.path.join(os.getcwd(), "static", "khqr.jpg"),
        os.path.join(os.getcwd(), "static", "khqr.jpeg"),
        os.path.join(os.getcwd(), "static", "aba.png"),
        os.path.join(os.getcwd(), "static", "aba.jpg"),
        os.path.join(os.getcwd(), "static", "qr.png"),
        os.path.join(os.getcwd(), "static", "qr.jpg"),
    ]

    # 1. Primary priority: Branded static card asset (e.g. asset/my_khqr.webp)
    loop = asyncio.get_running_loop()
    static_bytes = await loop.run_in_executor(None, _read_static_qr_file, candidate_paths)
    if static_bytes:
        with _QR_CACHE_LOCK:
            _QR_IMAGE_CACHE[cache_key] = static_bytes
        return static_bytes

    # 2. Dynamic generation fallback if no static branded file exists: Try local qrcode
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
        result = buf.getvalue()
        if result:
            with _QR_CACHE_LOCK:
                if len(_QR_IMAGE_CACHE) >= _MAX_CACHE_ENTRIES:
                    _QR_IMAGE_CACHE.popitem(last=False)
                _QR_IMAGE_CACHE[cache_key] = result
            return result
    except ImportError as exc:
        logger.debug("Local qrcode library not installed: %s", exc)
    except Exception as e:
        logger.warning("Local qrcode library generation failed: %s", e)

    # 3. Clean compact public QR generation service fallback (350x350)
    try:
        encoded_data = urllib.parse.quote(khqr_text)
        api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=350x350&margin=10&data={encoded_data}"
        client = await _get_shared_http_client()
        resp = await client.get(api_url)
        if resp.status_code == 200 and resp.content:
            with _QR_CACHE_LOCK:
                if len(_QR_IMAGE_CACHE) >= _MAX_CACHE_ENTRIES:
                    _QR_IMAGE_CACHE.popitem(last=False)
                _QR_IMAGE_CACHE[cache_key] = resp.content
            return resp.content
    except Exception as e:
        logger.error("QR Code API generation failed: %s", e)

    # 3. Fallback to static QR URL if configured
    if STATIC_QR_IMAGE_URL:
        try:
            client = await _get_shared_http_client()
            resp = await client.get(STATIC_QR_IMAGE_URL)
            if resp.status_code == 200 and resp.content:
                with _QR_CACHE_LOCK:
                    _QR_IMAGE_CACHE[cache_key] = resp.content
                return resp.content
        except Exception as e:
            logger.warning("Failed to fetch static QR image URL: %s", e)

    # 4. Final fallback to static local file if dynamic generation and API both failed
    loop = asyncio.get_running_loop()
    static_bytes = await loop.run_in_executor(None, _read_static_qr_file, candidate_paths)
    if static_bytes:
        with _QR_CACHE_LOCK:
            _QR_IMAGE_CACHE[cache_key] = static_bytes
        return static_bytes

    return None


class BakongKHQR:
    """Helper wrapper for Bakong KHQR operations."""

    @staticmethod
    def generate(
        amount: float | None = None,
        currency: str = "USD",
        user_id: int | str = "",
        tier: str = "coffee",
        merchant_id: str = "",
        expiration_days: float = 1.0,
    ) -> tuple[str, str]:
        """Generate KHQR string and bill reference. Returns (khqr_text, bill_no)."""
        ts = int(time.time())
        tier_clean = tier[:4].upper()
        bill_no = f"{tier_clean}{ts:x}"[-12:]
        khqr = generate_khqr_string(
            amount=amount,
            currency=currency,
            merchant_id=merchant_id,
            bill_number=bill_no,
            reference_label=str(user_id),
            expiration_days=expiration_days,
        )
        return khqr, bill_no

    @staticmethod
    def get_md5(khqr_text: str) -> str:
        """Compute the MD5 hash of the KHQR payload required by Bakong Open API."""
        return hashlib.md5(khqr_text.encode("utf-8")).hexdigest()
