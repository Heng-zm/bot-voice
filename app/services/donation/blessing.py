"""Automated AI Khmer Voice Blessing generator and delivery."""

from __future__ import annotations

import html
import io
import logging
import os
from contextlib import suppress
from typing import Any

from app.services.donation.store import TIER_DETAILS

logger = logging.getLogger(__name__)


def generate_blessing_script(
    donor_name: str,
    tier: str = "coffee",
    amount: float = 1.0,
    currency: str = "USD",
) -> str:
    """Generate a warm, heartfelt Khmer voice blessing script tailored to donor, currency, and tier."""
    name = (donor_name or "បង").strip()
    tier_lower = (tier or "coffee").lower().strip()
    curr_upper = (currency or "USD").upper().strip()

    # Normalize KHR tier suffixes (e.g. 'coffee_khr' -> 'coffee')
    base_tier = tier_lower.replace("_khr", "")

    # Calculate effective USD equivalent to support both USD and KHR amounts
    is_khr = (curr_upper == "KHR") or tier_lower.endswith("_khr") or amount > 100.0
    usd_equiv = (amount / 4000.0) if is_khr else amount

    if base_tier == "patron":
        resolved_tier = "patron"
    elif base_tier == "server":
        resolved_tier = "server"
    elif base_tier == "lunch":
        resolved_tier = "lunch"
    elif base_tier == "milktea":
        resolved_tier = "milktea"
    elif base_tier == "coffee":
        resolved_tier = "coffee"
    else:
        if usd_equiv <= 1.5:
            resolved_tier = "coffee"
        elif usd_equiv <= 2.5:
            resolved_tier = "milktea"
        elif usd_equiv <= 4.0:
            resolved_tier = "lunch"
        elif usd_equiv <= 8.0:
            resolved_tier = "server"
        else:
            resolved_tier = "patron"

    if resolved_tier == "coffee":
        return (
            f"សូមអរគុណបង {name} យ៉ាងជ្រាលជ្រៅសម្រាប់ការឧបត្ថម្ភកាហ្វេ ១ កែវនេះ! "
            "សូមជូនពរបងមានសុខភាពល្អ រកទទួលទានមានបាន និងជោគជ័យគ្រប់ភារកិច្ច! "
            "អរគុណច្រើនបង!"
        )
    elif resolved_tier == "milktea":
        return (
            f"សូមអរគុណបង {name} ដ៏ច្រើនលើសលប់សម្រាប់ការឧបត្ថម្ភតែទឹកដោះគោដ៏ផ្អែមឆ្ងាញ់នេះ! "
            "សូមជូនពរបងជួបតែសេចក្តីសុខ សំណាងល្អ និងមានស្នាមញញឹមស្រស់ស្រាយជានិច្ច! "
            "អរគុណច្រើនបង!"
        )
    elif resolved_tier == "lunch":
        return (
            f"សូមអរគុណបង {name} យ៉ាងខ្លាំងសម្រាប់ការឧបត្ថម្ភអាហារថ្ងៃត្រង់ដ៏ឈ្ងុយឆ្ងាញ់! "
            "សូមជូនពរបង និងក្រុមគ្រួសារ ជួបប្រទះតែសុភមង្គល វិបុលសុខ និងចម្រើនរុងរឿងគ្រប់ពេលវេលា! "
            "អរគុណច្រើនបង!"
        )
    elif resolved_tier == "server":
        return (
            f"សូមថ្លែងអំណរគុណយ៉ាងជ្រាលជ្រៅបំផុតជូនចំពោះបង {name} ដែលបានជួយឧបត្ថម្ភទ្រទ្រង់ដល់ដំណើរការ Server របស់ Bot Voice! "
            "ការគាំទ្ររបស់បងជាកម្លាំងចិត្តដ៏ធំធេងបំផុតសម្រាប់ពួកយើង! "
            "សូមជូនពរបងមានសុខភាពល្អបរិបូរណ៍ និងសម្រេចបានគ្រប់បំណងប្រាថ្នា! "
            "អរគុណច្រើនបង!"
        )
    else:  # patron
        return (
            f"សូមគោរពថ្លែងអំណរគុណយ៉ាងជ្រាលជ្រៅបំផុតចំពោះបង {name} សម្រាប់ទឹកចិត្តសប្បុរសធម៌ និងការឧបត្ថម្ភដ៏ថ្លៃថ្លានេះ! "
            "សូមវត្ថុស័ក្តិសិទ្ធិក្នុងលោក តាមជួយបីបាច់ថែរក្សាបង និងក្រុមគ្រួសារ ឱ្យទទួលបាននូវសុខភាពមាំមួន ចម្រើនដោយលាភយស និងទ្រព្យសម្បត្តិហូរហៀរគ្រប់ពេលវេលា! "
            "អរគុណច្រើនបង!"
        )


async def _synthesize_edge_tts_direct(text: str, gender: str = "female", speed: float = 0.95) -> bytes:
    """Direct standalone Microsoft Edge TTS synthesizer fallback for Khmer language."""
    import edge_tts

    voice = "km-KH-PisethNeural" if str(gender).lower() == "male" else "km-KH-SreymomNeural"
    spd_pct = int(round((speed - 1.0) * 100))
    rate_str = f"{spd_pct:+d}%"

    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.extend(chunk["data"])
    return bytes(buffer)


async def generate_voice_blessing(
    donor_name: str,
    tier: str = "coffee",
    amount: float = 1.0,
    currency: str = "USD",
    gender: str = "female",
    speed: float = 0.95,
) -> tuple[bytes, str]:
    """Synthesize Khmer speech audio for the blessing using resilient multi-tier fallbacks."""
    script = generate_blessing_script(donor_name, tier=tier, amount=amount, currency=currency)

    # 1. Try modern modular TTS service if available
    with suppress(Exception):
        from app.services.tts.service import synthesize_speech

        audio_bytes = await synthesize_speech(
            text=script,
            gender=gender,
            speed=speed,
            model="edge",
        )
        if audio_bytes:
            return audio_bytes, script

    # 2. Try legacy generate_voice helpers
    with suppress(Exception):
        from app.legacy import generate_voice  # type: ignore

        audio_bytes = await generate_voice(
            text=script,
            gender=gender,
            speed=speed,
            output_path="",
            tts_model="edge",
        )
        if audio_bytes:
            return audio_bytes, script

    with suppress(Exception):
        from app.legacy import _generate_voice_edge  # type: ignore

        audio_bytes = await _generate_voice_edge(
            text=script,
            gender=gender,
            speed=speed,
            output_path="",
        )
        if audio_bytes:
            return audio_bytes, script

    # 3. Direct Edge TTS library fallback
    try:
        audio_bytes = await _synthesize_edge_tts_direct(script, gender=gender, speed=speed)
        if audio_bytes:
            return audio_bytes, script
    except Exception as exc:
        logger.error("Direct Edge TTS synthesis for blessing failed: %s", exc)

    raise RuntimeError("All TTS synthesis tiers failed to generate blessing audio.")


async def deliver_voice_blessing(
    bot: Any,
    *,
    user_id: int,
    donor_name: str,
    tier: str = "coffee",
    amount: float = 1.0,
    currency: str = "USD",
) -> bool:
    """Synthesize and send personalized Khmer Voice Note to donor via Telegram."""
    tier_info = TIER_DETAILS.get(tier.lower(), {})
    tier_title = tier_info.get("title", f"${amount:.2f}")
    escaped_name = html.escape(donor_name)
    is_khr = (currency.upper() == "KHR") or tier.lower().endswith("_khr") or amount > 100.0
    amt_display = f"{int(round(amount)):,} KHR (៛)" if is_khr else f"${amount:.2f} USD"

    # Query donor's personalized voice gender preference if available
    voice_gender = "female"
    with suppress(Exception):
        from app.legacy import get_user_prefs_async

        prefs = await get_user_prefs_async(user_id)
        if isinstance(prefs, dict) and prefs.get("gender"):
            voice_gender = prefs["gender"]

    voice_caption = (
        f"🎙️ <b>សារសំឡេងអរគុណពិសេសពី Bot Voice</b> ❤️\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"ជូនចំពោះបង៖ <b>{escaped_name}</b>\n"
        f"កញ្ចប់ឧបត្ថម្ភ៖ <b>{tier_title}</b> ({amt_display})\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"<i>«សូមអរគុណបងយ៉ាងជ្រាលជ្រៅបំផុតសម្រាប់ការគាំទ្រ និងលើកទឹកចិត្ត! "
        f"បងជាចំណែកដ៏សំខាន់ដែលជួយឱ្យ Bot Voice បន្តដំណើរការដោយឥតគិតថ្លៃសម្រាប់បងប្អូនខ្មែរទាំងអស់គ្នា!»</i> ☕💖\n\n"
        f"🏆 <i>ពិនិត្យមើលឈ្មោះបងក្នុងតារាងកិត្តិយស៖</i> /donors"
    )

    # 1. Attempt Voice Note Delivery (.ogg)
    try:
        audio_bytes, _ = await generate_voice_blessing(
            donor_name,
            tier=tier,
            amount=amount,
            currency=currency,
            gender=voice_gender,
        )
        if audio_bytes:
            voice_file = io.BytesIO(audio_bytes)
            voice_file.name = "blessing.ogg"
            try:
                await bot.send_voice(
                    chat_id=user_id,
                    voice=voice_file,
                    caption=voice_caption,
                    parse_mode="HTML",
                )
                logger.info("Voice blessing delivered successfully to user %s (%s)", user_id, donor_name)
                return True
            except Exception as voice_err:
                logger.warning("send_voice failed (%s), attempting send_audio fallback", voice_err)
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "blessing.mp3"
                await bot.send_audio(
                    chat_id=user_id,
                    audio=audio_file,
                    caption=voice_caption,
                    parse_mode="HTML",
                    title=f"ពរជ័យថ្លែងអំណរគុណ - {donor_name}",
                    performer="Bot Voice Cambodia",
                )
                return True
    except Exception as e:
        logger.error("Failed to generate or send voice blessing to %s: %s; falling back to text", user_id, e)

    # 2. Fallback to formatted text delivery if voice synthesis is unavailable
    try:
        script_text = generate_blessing_script(donor_name, tier=tier, amount=amount, currency=currency)
        text_message = (
            f"💌 <b>សារអរគុណ និងពរជ័យពិសេសពី Bot Voice</b> ❤️\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"ជូនចំពោះបង៖ <b>{escaped_name}</b>\n"
            f"កញ្ចប់ឧបត្ថម្ភ៖ <b>{tier_title}</b> ({amt_display})\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📜 <b>ពាក្យជូនពរ៖</b>\n"
            f"<i>«{html.escape(script_text)}»</i>\n\n"
            f"🏆 <i>ពិនិត្យមើលឈ្មោះបងក្នុងតារាងកិត្តិយស៖</i> /donors"
        )
        await bot.send_message(
            chat_id=user_id,
            text=text_message,
            parse_mode="HTML",
        )
        logger.info("Fallback text blessing delivered to user %s (%s)", user_id, donor_name)
        return True
    except Exception as e:
        logger.error("Could not deliver fallback blessing text to %s: %s", user_id, e)
        return False


async def cmd_testblessing(update: Any, context: Any) -> None:
    """Administrator command to test and preview the voice blessing system."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return

    admin_ids: set[int] = set()
    for env_aid in os.environ.get("ADMIN_IDS", "").split(","):
        if env_aid.strip().lstrip("-").isdigit():
            admin_ids.add(int(env_aid.strip()))

    if user.id not in admin_ids:
        with suppress(Exception):
            await msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML")
        return

    args = list(context.args or [])
    donor_name = user.first_name or "Admin"
    amount = 1.0
    tier = "coffee"

    if args:
        with suppress(ValueError):
            amount = float(args[0])
            tier = "lunch" if amount >= 5.0 else ("milktea" if amount >= 2.0 else "coffee")

    await msg.reply_text(f"⏳ <b>កំពុងបង្កើតសាកល្បង Voice Blessing (${amount:.2f})...</b>", parse_mode="HTML")
    success = await deliver_voice_blessing(
        context.bot,
        user_id=user.id,
        donor_name=donor_name,
        tier=tier,
        amount=amount,
        currency="USD",
    )
    if not success:
        await msg.reply_text("❌ បរាជ័យក្នុងការបង្កើត Voice Blessing សូមពិនិត្យមើល System Logs។")


__all__ = [
    "cmd_testblessing",
    "deliver_voice_blessing",
    "generate_blessing_script",
    "generate_voice_blessing",
]