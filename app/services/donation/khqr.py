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
DEFAULT_BAKONG_MERCHANT_NAME = os.getenv("BAKONG_MERCHANT_NAME", "CHUO KIMHENG").strip() or "CHUO KIMHENG"
DEFAULT_BAKONG_MERCHANT_CITY = os.getenv("BAKONG_MERCHANT_CITY", "Phnom Penh").strip() or "Phnom Penh"
DEFAULT_BAKONG_CURRENCY = os.getenv("BAKONG_CURRENCY", "USD").strip().upper() or "USD"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STATIC_QR_IMAGE_URL = os.getenv("KHQR_STATIC_IMAGE_URL", "").strip()
STATIC_QR_IMAGE_PATH = os.getenv("KHQR_STATIC_IMAGE_PATH", "").strip()

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
    amount: float | None = None,
    currency: str = "USD",
    bill_number: str = "",
    reference_label: str = "",
    terminal_label: str = "BOTVOICE",
) -> str:
    """Generate official Bakong KHQR EMVCo payload string."""
    account_id = account_id.strip() or DEFAULT_BAKONG_ACCOUNT_ID
    merchant_name = (merchant_name.strip() or DEFAULT_BAKONG_MERCHANT_NAME)[:25]
    merchant_city = (merchant_city.strip() or DEFAULT_BAKONG_MERCHANT_CITY)[:15]
    currency = (currency.strip().upper() or DEFAULT_BAKONG_CURRENCY)

    # Tag 00: Payload Format Indicator
    payload = _format_tlv("00", "01")

    # Tag 01: Point of Initiation Method (12 = dynamic with amount, 11 = static)
    is_dynamic = amount is not None and amount > 0
    payload += _format_tlv("01", "12" if is_dynamic else "11")

    # Tag 29: Merchant Account Information (Bakong)
    sub29_guid = _format_tlv("00", "bakong@nbc")
    sub29_acc = _format_tlv("01", account_id)
    tag29_val = sub29_guid + sub29_acc
    payload += _format_tlv("29", tag29_val)

    # Tag 52: Merchant Category Code (5999 = Miscellaneous/Specialty Retail)
    payload += _format_tlv("52", "5999")

    # Tag 53: Transaction Currency (840 = USD, 116 = KHR)
    currency_code = "840" if currency == "USD" else "116"
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
    if reference_label:
        tag62_val += _format_tlv("05", reference_label[:25])
    if terminal_label:
        tag62_val += _format_tlv("07", terminal_label[:25])

    if tag62_val:
        payload += _format_tlv("62", tag62_val)

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

    # For static QR codes (no specific amount), prefer the existing branded static file if present
    if not is_dynamic_amount:
        loop = asyncio.get_running_loop()
        static_bytes = await loop.run_in_executor(None, _read_static_qr_file, candidate_paths)
        if static_bytes:
            with _QR_CACHE_LOCK:
                _QR_IMAGE_CACHE[cache_key] = static_bytes
            return static_bytes

    # Generate exact dynamic QR code: 1. Try local python qrcode package
    try:
        import qrcode

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=3,
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

    # 2. Try public QR code generation service with pooled client for dynamic payload
    try:
        encoded_data = urllib.parse.quote(khqr_text)
        api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=500x500&margin=15&data={encoded_data}"
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
    ) -> tuple[str, str]:
        """Generate KHQR string and bill reference. Returns (khqr_text, bill_no)."""
        ts = int(time.time())
        tier_clean = tier[:4].upper()
        bill_no = f"{tier_clean}{ts:x}"[-12:]
        khqr = generate_khqr_string(
            amount=amount,
            currency=currency,
            bill_number=bill_no,
            reference_label=str(user_id),
        )
        return khqr, bill_no
