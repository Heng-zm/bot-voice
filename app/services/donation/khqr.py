"""Bakong KHQR (EMVCo) generation and QR code image rendering."""

from __future__ import annotations

import io
import logging
import os
import urllib.parse
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default Bakong configuration (can be customized via environment variables)
DEFAULT_BAKONG_ACCOUNT_ID = os.getenv("BAKONG_ACCOUNT_ID", "chuo_kimheng@bkrt").strip() or "chuo_kimheng@bkrt"
DEFAULT_BAKONG_MERCHANT_NAME = os.getenv("BAKONG_MERCHANT_NAME", "CHUO KIMHENG").strip() or "CHUO KIMHENG"
DEFAULT_BAKONG_MERCHANT_CITY = os.getenv("BAKONG_MERCHANT_CITY", "Phnom Penh").strip() or "Phnom Penh"
DEFAULT_BAKONG_CURRENCY = os.getenv("BAKONG_CURRENCY", "USD").strip().upper() or "USD"
STATIC_QR_IMAGE_URL = os.getenv("KHQR_STATIC_IMAGE_URL", "").strip()
STATIC_QR_IMAGE_PATH = os.getenv("KHQR_STATIC_IMAGE_PATH", "").strip()


def crc16_ccitt(data: str) -> str:
    """Compute standard EMVCo CRC16-CCITT checksum (poly 0x1021, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data.encode("utf-8"):
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
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
        if currency == "USD":
            amount_str = f"{amount:.2f}"
        else:
            amount_str = f"{int(round(amount))}"
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


async def get_khqr_qr_image(khqr_text: str) -> bytes | None:
    """Render QR code image bytes (PNG) for a given KHQR string.

    Tries local `qrcode` library first, then custom static image URL/file,
    then remote QR generation API fallback.
    """
    # 1. If static file path is provided and exists
    candidate_paths = [
        STATIC_QR_IMAGE_PATH,
        os.path.join(os.getcwd(), "static", "khqr.png"),
        os.path.join(os.getcwd(), "static", "khqr.jpg"),
        os.path.join(os.getcwd(), "static", "khqr.jpeg"),
        os.path.join(os.getcwd(), "static", "aba.png"),
        os.path.join(os.getcwd(), "static", "aba.jpg"),
        os.path.join(os.getcwd(), "static", "qr.png"),
        os.path.join(os.getcwd(), "static", "qr.jpg"),
    ]
    for p in candidate_paths:
        if p and os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.warning("Failed to read static QR image path %s: %s", p, e)


    # 2. Try python qrcode package
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
        return buf.getvalue()
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Local qrcode library generation failed: %s", e)

    # 3. If static QR URL is provided
    if STATIC_QR_IMAGE_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(STATIC_QR_IMAGE_URL)
                if resp.status_code == 200 and resp.content:
                    return resp.content
        except Exception as e:
            logger.warning("Failed to fetch static QR image URL: %s", e)

    # 4. Fallback to public QR code generation service
    try:
        encoded_data = urllib.parse.quote(khqr_text)
        api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=500x500&margin=15&data={encoded_data}"
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(api_url)
            if resp.status_code == 200 and resp.content:
                return resp.content
    except Exception as e:
        logger.error("QR Code API generation failed: %s", e)

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
        import time

        ts = int(time.time())
        bill_no = f"{tier.upper()}-{ts}"[-20:]
        khqr = generate_khqr_string(
            amount=amount,
            currency=currency,
            bill_number=bill_no,
            reference_label=str(user_id),
        )
        return khqr, bill_no
