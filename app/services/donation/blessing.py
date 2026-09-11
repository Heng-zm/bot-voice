"""Automated AI Khmer Voice Blessing generator and delivery."""

from __future__ import annotations

import html
import io
import logging
from typing import Any

from app.services.donation.store import TIER_DETAILS

logger = logging.getLogger(__name__)


def generate_blessing_script(donor_name: str, tier: str = "coffee", amount: float = 1.0) -> str:
    """Generate a warm, heartfelt Khmer voice blessing script tailored to the donor and tier."""
    name = (donor_name or "បង").strip()
    tier_lower = tier.lower()

    if tier_lower == "coffee" or amount <= 1.5:
        return (
            f"សូមអរគុណបង {name} យ៉ាងជ្រាលជ្រៅសម្រាប់ការឧបត្ថម្ភកាហ្វេ ១ កែវនេះ! "
            "សូមជូនពរបងមានសុខភាពល្អ រកទទួលទានមានបាន និងជោគជ័យគ្រប់ភារកិច្ច! "
            "អរគុណច្រើនបង!"
        )
    elif tier_lower == "milktea" or amount <= 2.5:
        return (
            f"សូមអរគុណបង {name} ដ៏ច្រើនលើសលប់សម្រាប់ការឧបត្ថម្ភតែទឹកដោះគោដ៏ផ្អែមឆ្ងាញ់នេះ! "
            "សូមជូនពរបងជួបតែសេចក្តីសុខ សំណាងល្អ និងមានស្នាមញញឹមស្រស់ស្រាយជានិច្ច! "
            "អរគុណច្រើនបង!"
        )
    elif tier_lower == "lunch" or amount <= 4.0:
        return (
            f"សូមអរគុណបង {name} យ៉ាងខ្លាំងសម្រាប់ការឧបត្ថម្ភអាហារថ្ងៃត្រង់ដ៏ឈ្ងុយឆ្ងាញ់! "
            "សូមជូនពរបង និងក្រុមគ្រួសារ ជួបប្រទះតែសុភមង្គល វិបុលសុខ និងចម្រើនរុងរឿងគ្រប់ពេលវេលា! "
            "អរគុណច្រើនបង!"
        )
    elif tier_lower == "server" or amount <= 8.0:
        return (
            f"សូមថ្លែងអំណរគុណយ៉ាងជ្រាលជ្រៅបំផុតជូនចំពោះបង {name} ដែលបានជួយឧបត្ថម្ភទ្រទ្រង់ដល់ដំណើរការ Server របស់ Bot Voice! "
            "ការគាំទ្ររបស់បងជាកម្លាំងចិត្តដ៏ធំធេងបំផុតសម្រាប់ពួកយើង! "
            "សូមជូនពរបងមានសុខភាពល្អបរិបូរណ៍ និងសម្រេចបានគ្រប់បំណងប្រាថ្នា! "
            "អរគុណច្រើនបង!"
        )
    else:  # Patron / Gold tier
        return (
            f"សូមគោរពថ្លែងអំណរគុណយ៉ាងជ្រាលជ្រៅបំផុតចំពោះបង {name} សម្រាប់ទឹកចិត្តសប្បុរសធម៌ និងការឧបត្ថម្ភដ៏ថ្លៃថ្លានេះ! "
            "សូមវត្ថុស័ក្តិសិទ្ធិក្នុងលោក តាមជួយបីបាច់ថែរក្សាបង និងក្រុមគ្រួសារ ឱ្យទទួលបាននូវសុខភាពមាំមួន ចម្រើនដោយលាភយស និងទ្រព្យសម្បត្តិហូរហៀរគ្រប់ពេលវេលា! "
            "អរគុណច្រើនបង!"
        )


async def generate_voice_blessing(
    donor_name: str,
    tier: str = "coffee",
    amount: float = 1.0,
    gender: str = "female",
    speed: float = 0.95,
) -> tuple[bytes, str]:
    """Synthesize Khmer speech audio for the blessing. Returns (audio_bytes, script_text)."""
    script = generate_blessing_script(donor_name, tier=tier, amount=amount)

    try:
        from app.legacy import generate_voice  # type: ignore

        audio_bytes = await generate_voice(
            text=script,
            gender=gender,
            speed=speed,
            output_path="",
            tts_model="edge",
        )
        return audio_bytes, script
    except Exception as e:
        logger.warning("Primary generate_voice failed for blessing, trying _generate_voice_edge: %s", e)

    try:
        from app.legacy import _generate_voice_edge  # type: ignore

        audio_bytes = await _generate_voice_edge(
            text=script,
            gender=gender,
            speed=speed,
            output_path="",
        )
        return audio_bytes, script
    except Exception as e:
        logger.error("Failed to synthesize voice blessing: %s", e)
        raise


async def deliver_voice_blessing(
    bot: Any,
    *,
    user_id: int,
    donor_name: str,
    tier: str = "coffee",
    amount: float = 1.0,
) -> bool:
    """Synthesize and send personalized Khmer Voice Note to donor via Telegram."""
    tier_info = TIER_DETAILS.get(tier.lower(), {})
    tier_title = tier_info.get("title", f"${amount:.2f}")
    escaped_name = html.escape(donor_name)

    caption = (
        f"🎙️ <b>សារសំឡេងអរគុណពិសេសពី Bot Voice</b> ❤️\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"ជូនចំពោះបង៖ <b>{escaped_name}</b>\n"
        f"កញ្ចប់ឧបត្ថម្ភ៖ <b>{tier_title}</b> (${amount:.2f} USD)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"<i>«សូមអរគុណបងយ៉ាងជ្រាលជ្រៅបំផុតសម្រាប់ការគាំទ្រ និងលើកទឹកចិត្ត! បងជាចំណែកដ៏សំខាន់ដែលជួយឱ្យ Bot Voice បន្តដំណើរការដោយឥតគិតថ្លៃសម្រាប់បងប្អូនខ្មែរទាំងអស់គ្នា!»</i> ☕💖\n\n"
        f"🏆 <i>ពិនិត្យមើលឈ្មោះបងក្នុងតារាងកិត្តិយស៖</i> /donors"
    )

    try:
        audio_bytes, script = await generate_voice_blessing(donor_name, tier=tier, amount=amount)
        if audio_bytes:
            voice_file = io.BytesIO(audio_bytes)
            voice_file.name = "blessing.ogg"
            await bot.send_voice(
                chat_id=user_id,
                voice=voice_file,
                caption=caption,
                parse_mode="HTML",
            )
            logger.info("Voice blessing delivered successfully to user %s (%s)", user_id, donor_name)
            return True
    except Exception as e:
        logger.error("Failed to send voice blessing to %s: %s; falling back to text", user_id, e)

    # Fallback to rich text message if voice synthesis / sending encountered an error
    try:
        script_text = generate_blessing_script(donor_name, tier, amount)
        text_message = (
            f"🎉 <b>សារអរគុណ និងជូនពរពិសេស!</b> ❤️\n\n"
            f"{caption}\n\n"
            f"📜 <b>ពាក្យជូនពរ៖</b>\n"
            f"<i>«{html.escape(script_text)}»</i>"
        )
        await bot.send_message(
            chat_id=user_id,
            text=text_message,
            parse_mode="HTML",
        )
        return True
    except Exception as e:
        logger.error("Could not deliver fallback blessing text to %s: %s", user_id, e)
        return False

