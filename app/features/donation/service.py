"""Donation service orchestrating Bakong KHQR payments, verification, and blessings."""

from __future__ import annotations

from typing import Any

from app.features.donation.bakong import (
    check_transaction_by_md5,
    decode_token_payload,
    generate_deeplink_by_qr,
    is_bakong_api_configured,
    test_connection as test_bakong_connection,
    verify_khqr_payment,
)
from app.features.donation.khqr import (
    BakongKHQR,
    decode_khqr,
    generate_khqr_payload,
    generate_khqr_string,
    get_khqr_config,
    get_khqr_qr_image,
    get_static_khqr_card,
    update_khqr_config,
    verify_khqr_crc,
)
from app.services.donation.blessing import (
    deliver_voice_blessing,
    generate_blessing_script,
    generate_voice_blessing,
)
from app.services.donation.store import (
    TIER_DETAILS,
    DonationStore,
    donation_store,
)


class DonationService:
    """Consolidated business service for donations, KHQR, and voice blessings."""

    def __init__(self, store: DonationStore | None = None) -> None:
        self.store = store or donation_store

    def generate_khqr(
        self,
        amount: float,
        currency: str = "USD",
        bill_no: str | None = None,
        store_label: str = "BotVoice",
        terminal_label: str = "Telegram",
    ) -> dict[str, Any]:
        """Generate EMVCo-compliant Bakong KHQR payload."""
        return generate_khqr_payload(
            amount=amount,
            currency=currency,
            bill_number=bill_no or "",
        )

    async def verify_payment(
        self,
        md5: str,
        expected_amount: float | None = None,
        expected_currency: str = "USD",
    ) -> dict[str, Any]:
        """Verify transaction against NBC Bakong Open API."""
        return await verify_khqr_payment(
            md5=md5,
            expected_amount=expected_amount,
            expected_currency=expected_currency,
        )

    def generate_blessing(
        self,
        donor_name: str,
        tier: str = "coffee",
        amount: float = 1.0,
        currency: str = "USD",
    ) -> str:
        """Generate personalized Khmer voice blessing script."""
        return generate_blessing_script(
            donor_name=donor_name,
            tier=tier,
            amount=amount,
            currency=currency,
        )

    async def deliver_voice_blessing(
        self,
        bot: Any,
        chat_id: int,
        donor_name: str,
        tier: str = "coffee",
        amount: float = 1.0,
        currency: str = "USD",
    ) -> bool:
        """Generate and deliver TTS voice blessing audio to user."""
        return await deliver_voice_blessing(
            bot=bot,
            chat_id=chat_id,
            donor_name=donor_name,
            tier=tier,
            amount=amount,
            currency=currency,
        )


donation_service = DonationService()

__all__ = [
    "BakongKHQR",
    "DonationService",
    "DonationStore",
    "TIER_DETAILS",
    "check_transaction_by_md5",
    "decode_khqr",
    "decode_token_payload",
    "deliver_voice_blessing",
    "donation_service",
    "donation_store",
    "generate_blessing_script",
    "generate_deeplink_by_qr",
    "generate_khqr_payload",
    "generate_khqr_string",
    "generate_voice_blessing",
    "get_khqr_config",
    "get_khqr_qr_image",
    "get_static_khqr_card",
    "is_bakong_api_configured",
    "test_bakong_connection",
    "update_khqr_config",
    "verify_khqr_crc",
    "verify_khqr_payment",
]
