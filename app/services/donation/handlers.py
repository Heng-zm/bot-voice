"""Telegram commands and callback handlers for Bakong KHQR donations and Hall of Fame."""

from __future__ import annotations

import html
import io
import logging
import os
import threading
import time
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from app.core.config import SETTINGS
from app.services.donation.blessing import deliver_voice_blessing, generate_voice_blessing
from app.services.donation.khqr import (
    DEFAULT_BAKONG_ACCOUNT_ID,
    DEFAULT_BAKONG_MERCHANT_NAME,
    BakongKHQR,
    generate_khqr_string,
    get_khqr_qr_image,
)
from app.services.donation.store import TIER_DETAILS, donation_store

logger = logging.getLogger(__name__)

# Concurrency & deduplication guards for admin approvals
_PROCESSED_APPROVALS: set[str] = set()
_APPROVALS_LOCK = threading.Lock()


def is_admin_user(user_id: int) -> bool:
    """Check whether a given user_id is an authorized administrator."""
    try:
        from app.legacy import _is_admin  # type: ignore

        if _is_admin(user_id):
            return True
    except Exception:
        pass

    admin_str = os.getenv("ADMIN_IDS", "") or getattr(SETTINGS, "ADMIN_IDS", "")
    for part in admin_str.split(","):
        part = part.strip()
        if part.isdigit() and int(part) == user_id:
            return True
    return False


def get_admin_ids() -> set[int]:
    """Retrieve all configured administrator Telegram IDs."""
    admin_ids: set[int] = set()
    try:
        from app.legacy import ADMIN_IDS  # type: ignore

        if isinstance(ADMIN_IDS, (set, list, tuple)):
            admin_ids.update(int(aid) for aid in ADMIN_IDS if str(aid).isdigit())
    except Exception:
        pass

    admin_str = os.getenv("ADMIN_IDS", "") or getattr(SETTINGS, "ADMIN_IDS", "")
    for part in admin_str.split(","):
        part = part.strip()
        if part.isdigit():
            admin_ids.add(int(part))
    return admin_ids


def _build_donation_menu_markup() -> InlineKeyboardMarkup:
    """Construct main donation tiers inline keyboard."""
    buttons = [
        [
            InlineKeyboardButton("☕ $1.00 កាហ្វេ ១ កែវ", callback_data="donate_tier:coffee"),
            InlineKeyboardButton("🧋 $2.00 តែទឹកដោះគោ", callback_data="donate_tier:milktea"),
        ],
        [
            InlineKeyboardButton("🍜 $3.00 គុយទាវ ១ ចាន", callback_data="donate_tier:lunch"),
            InlineKeyboardButton("🖥️ $5.00 ថ្លៃ Server", callback_data="donate_tier:server"),
        ],
        [
            InlineKeyboardButton("🌟 $10.00 ឧបត្ថម្ភពិសេស", callback_data="donate_tier:patron"),
            InlineKeyboardButton("💎 $20.00 អ្នកគាំទ្រឆ្នើម", callback_data="donate_tier:gold"),
        ],
        [
            InlineKeyboardButton("🏆 តារាងកិត្តិយស (Hall of Fame)", callback_data="donate_halloffame"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


async def _send_khqr_screen(
    *,
    chat_id: int,
    user_name: str,
    amount: float,
    tier_key: str,
    tier_title: str,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Reusable generator and sender for Bakong KHQR payment interface."""
    khqr_text, bill_no = BakongKHQR.generate(
        amount=amount,
        currency="USD",
        user_id=chat_id,
        tier=tier_key,
    )

    caption = (
        f"🇰🇭 <b>Bakong KHQR — {tier_title}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>ឈ្មោះគណនី:</b> <code>{DEFAULT_BAKONG_MERCHANT_NAME}</code>\n"
        f"🆔 <b>Bakong ID:</b> <code>{DEFAULT_BAKONG_ACCOUNT_ID}</code>\n"
        f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>${amount:.2f} USD</b>\n"
        f"🧾 <b>លេខវិក្កយបត្រ:</b> <code>{bill_no}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📲 <b>របៀបបង់ប្រាក់៖</b>\n"
        f"1. បើកកម្មវិធីធនាគាររបស់បង (ABA, ACLEDA, Wing, Canadia, Bakong...)\n"
        f"2. ស្កេនរូបភាព QR កូដនេះ\n"
        f"3. ផ្ទៀងផ្ទាត់ចំនួន <b>${amount:.2f}</b> រួចចុចផ្ទេរប្រាក់\n"
        f"4. បន្ទាប់ពីផ្ទេររួច សូមចុចប៊ូតុង <b>«✅ ខ្ញុំបានផ្ទេរប្រាក់រួចរាល់»</b> ខាងក្រោម\n\n"
        f"✨ <i>Bot នឹងផ្ញើសារសំឡេងអរគុណពិសេសជូនបងភ្លាមៗ!</i>"
    )

    action_buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ ខ្ញុំបានផ្ទេរប្រាក់រួចរាល់", callback_data=f"donate_paid:{tier_key}:{bill_no}:{amount:.2f}")],
        [InlineKeyboardButton("🔙 ជ្រើសរើសចំនួនផ្សេង", callback_data="donate_menu")],
    ])

    qr_bytes = await get_khqr_qr_image(khqr_text)
    if qr_bytes and context.bot:
        try:
            photo_file = io.BytesIO(qr_bytes)
            photo_file.name = "khqr.png"
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=photo_file,
                caption=caption,
                reply_markup=action_buttons,
                parse_mode="HTML",
            )
            return
        except Exception as e:
            logger.warning("Could not send QR as photo, falling back to message text: %s", e)

    # Fallback text with KHQR payload if photo transmission fails
    fallback_text = (
        f"{caption}\n\n"
        f"📋 <b>KHQR Payload String:</b>\n"
        f"<pre><code>{khqr_text}</code></pre>"
    )
    if context.bot:
        await context.bot.send_message(
            chat_id=chat_id,
            text=fallback_text,
            reply_markup=action_buttons,
            parse_mode="HTML",
        )


async def cmd_donate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display interactive donation tiers or direct custom amount QR code."""
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    user_name = html.escape(user.first_name or "បង")

    # Check for direct custom amount argument e.g. /donate 5 or /coffee 2.50
    args = context.args or []
    if args:
        try:
            custom_amount = float(args[0].replace("$", "").replace(",", ""))
            if 0.1 <= custom_amount <= 5000.0:
                tier_key = "custom"
                if custom_amount in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
                    for k, v in TIER_DETAILS.items():
                        if v["amount"] == custom_amount:
                            tier_key = k
                            break
                tier_title = f"ឧបត្ថម្ភ ${custom_amount:.2f} USD"
                await _send_khqr_screen(
                    chat_id=user.id,
                    user_name=user_name,
                    amount=custom_amount,
                    tier_key=tier_key,
                    tier_title=tier_title,
                    context=context,
                )
                return
        except ValueError:
            pass

    # Standard donation menu
    text = (
        f"☕ <b>សូមស្វាគមន៍មកកាន់ការឧបត្ថម្ភ Bot Voice!</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"សួស្តីបង <b>{user_name}</b>! Bot Voice ត្រូវបានបង្កើតឡើងដើម្បីបម្រើបងប្អូនខ្មែរក្នុងការបម្លែងអត្ថបទជាសំឡេង (Khmer TTS) "
        f"និងឆ្លើយសំណួរ AI ដោយឥតគិតថ្លៃ ១០០%។\n\n"
        f"ការឧបត្ថម្ភកាហ្វេ ១ កែវ ឬជួយថ្លៃ Server របស់បង គឺជាកម្លាំងចិត្តដ៏ធំធេង និងជួយទ្រទ្រង់ឱ្យ Bot ដំណើរការបានលឿន ឥតគាំង និងមានមុខងារថ្មីៗជានិច្ច! 💖\n\n"
        f"✨ <b>រាល់ការឧបត្ថម្ភ បងនឹងទទួលបាន៖</b>\n"
        f"🎙️ <b>សារសំឡេងអរគុណ និងជូនពរពិសេស</b> (AI Voice Blessing) ផ្ទាល់ខ្លួន\n"
        f"🏆 <b>ឈ្មោះក្នុងតារាងកិត្តិយស</b> (/donors — Hall of Fame)\n"
        f"🏅 <b>Badge កិត្តិយស</b> (🥇, 🥈, 🥉, ⭐ Supporter)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"👇 <b>សូមជ្រើសរើសចំនួនដែលបងចង់ឧបត្ថម្ភ៖</b>\n"
        f"<i>(ឬវាយ <code>/donate ចំនួនទឹកប្រាក់</code> ឧទាហរណ៍ <code>/donate 5</code>)</i>"
    )

    await msg.reply_text(
        text,
        reply_markup=_build_donation_menu_markup(),
        parse_mode="HTML",
    )


async def cmd_donors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display Hall of Fame (/donors) with top contributors and metrics."""
    stats = await donation_store.get_donation_stats()
    top_supporters = await donation_store.get_top_supporters(10)
    recent = await donation_store.get_recent_donations(5)

    total_cups = stats.get("total_cups", 0)
    total_usd = stats.get("total_usd", 0.0)
    total_donors = stats.get("total_donors", 0)

    lines: list[str] = [
        "🏆 <b>តារាងកិត្តិយសអ្នកឧបត្ថម្ភ (Hall of Fame)</b>",
        "━━━━━━━━━━━━━━━━━━━",
        f"☕ <b>កាហ្វេទទួលបាន:</b> {total_cups} កែវ",
        f"💰 <b>មូលនិធិគាំទ្រ:</b> ${total_usd:.2f} USD",
        f"👥 <b>សប្បុរសជន:</b> {total_donors} នាក់",
        "━━━━━━━━━━━━━━━━━━━\n",
    ]

    if top_supporters:
        lines.append("🌟 <b>កំពូលអ្នកឧបត្ថម្ភ (Top Supporters):</b>")
        for item in top_supporters:
            badge = item.get("badge", "⭐")
            name = html.escape(item.get("full_name") or f"User {item.get('user_id')}")
            amount = float(item.get("total_amount") or 0.0)
            cups = item.get("total_cups", 1)
            lines.append(f"{badge} <b>{name}</b> — ${amount:.2f} ({cups} កែវ)")
        lines.append("")
    else:
        lines.append("🌟 <i>មិនទាន់មានទិន្នន័យអ្នកឧបត្ថម្ភនៅឡើយទេ។ ចុចប៊ូតុងខាងក្រោមដើម្បីក្លាយជាអ្នកឧបត្ថម្ភដំបូងបង្អស់!</i>\n")

    if recent:
        lines.append("🕒 <b>អ្នកឧបត្ថម្ភថ្មីៗ (Recent Supporters):</b>")
        for r in recent:
            name = html.escape(r.get("full_name") or f"User {r.get('user_id')}")
            amt = float(r.get("amount") or 0.0)
            tier = r.get("tier", "coffee")
            tier_info = TIER_DETAILS.get(tier, {})
            tier_title = tier_info.get("title", f"${amt:.2f}")
            lines.append(f"• <b>{name}</b> — ${amt:.2f} ({tier_title})")
        lines.append("")

    lines.append("💖 <i>សូមថ្លែងអំណរគុណយ៉ាងជ្រាលជ្រៅដល់បងប្អូនទាំងអស់ដែលបានចូលរួមចំណែកគាំទ្រ Bot Voice!</i>")

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("☕ ចូលរួមឧបត្ថម្ភ / Buy Coffee", callback_data="donate_menu")],
    ])

    text = "\n".join(lines)
    msg = update.effective_message
    if msg:
        await msg.reply_text(text, reply_markup=keyboard, parse_mode="HTML")


async def cmd_adddonor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to manually credit a donor and trigger voice blessing.

    Usage: /adddonor <user_id> <amount> [tier] [custom_name]
    Example: /adddonor 1272791365 2.0 milktea Sokha
    """
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    if not is_admin_user(int(user.id)):
        await msg.reply_text("⛔ អ្នកមិនមានសិទ្ធិប្រើប្រាស់ពាក្យបញ្ជានេះទេ។")
        return

    args = context.args or []
    if len(args) < 2:
        await msg.reply_text(
            "ℹ️ <b>របៀបប្រើប្រាស់ពាក្យបញ្ជា /adddonor:</b>\n\n"
            "<code>/adddonor &lt;user_id&gt; &lt;amount&gt; [tier] [name]</code>\n\n"
            "• ឧទាហរណ៍៖ <code>/adddonor 1272791365 1.0 coffee Dara</code>\n"
            "• Tiers: <code>coffee</code> ($1), <code>milktea</code> ($2), <code>lunch</code> ($3), <code>server</code> ($5), <code>patron</code> ($10)",
            parse_mode="HTML",
        )
        return

    try:
        donor_uid = int(args[0])
        amount = float(args[1])
    except ValueError:
        await msg.reply_text("❌ user_id និង amount ត្រូវតែជាតួលេខ។")
        return

    tier = args[2].lower() if len(args) > 2 and args[2].lower() in TIER_DETAILS else "coffee"
    custom_name = " ".join(args[3:]) if len(args) > 3 else ""

    # Attempt to auto-fetch donor's actual name from Telegram if omitted
    if not custom_name and context.bot:
        try:
            chat = await context.bot.get_chat(donor_uid)
            if chat and chat.first_name:
                custom_name = chat.first_name
        except Exception:
            pass

    if not custom_name:
        custom_name = f"User {donor_uid}"

    # 1. Record in store
    record = await donation_store.record_donation(
        user_id=donor_uid,
        full_name=custom_name,
        amount=amount,
        tier=tier,
        note=f"Added by admin {user.id}",
        blessing_sent=True,
    )

    # 2. Synthesize and deliver Voice Blessing Note
    blessing_sent = False
    if context.bot:
        blessing_sent = await deliver_voice_blessing(
            context.bot,
            user_id=donor_uid,
            donor_name=custom_name,
            tier=tier,
            amount=amount,
        )

    status_blessing = "🎙️ បានផ្ញើសារសំឡេងជូនពររួចរាល់" if blessing_sent else "⚠️ មិនអាចផ្ញើសំឡេងទៅកាន់អ្នកប្រើបានទេ (អាច user មិនទាន់ /start bot)"
    await msg.reply_text(
        f"✅ <b>បានកត់ត្រាការឧបត្ថម្ភជោគជ័យ!</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>សប្បុរសជន:</b> {html.escape(custom_name)} (ID: <code>{donor_uid}</code>)\n"
        f"💵 <b>ចំនួន:</b> ${amount:.2f} USD ({tier})\n"
        f"✨ <b>ស្ថានភាព:</b> {status_blessing}\n"
        f"🏆 <b>តារាងកិត្តិយស:</b> បានធ្វើបច្ចុប្បន្នភាព (/donors)",
        parse_mode="HTML",
    )


async def cmd_testblessing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to test/preview the automated voice blessing note on themselves.

    Usage: /testblessing [name] [tier]
    """
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    if not is_admin_user(int(user.id)):
        await msg.reply_text("⛔ អ្នកមិនមានសិទ្ធិប្រើប្រាស់ពាក្យបញ្ជានេះទេ។")
        return

    args = context.args or []
    name = args[0] if len(args) > 0 else (user.first_name or "បង")
    tier = args[1].lower() if len(args) > 1 and args[1].lower() in TIER_DETAILS else "coffee"
    tier_info = TIER_DETAILS.get(tier, {})
    amount = tier_info.get("amount", 1.0)

    wait_msg = await msg.reply_text("⏳ កំពុងបង្កើតសំឡេងជូនពរ AI Voice Blessing...")

    try:
        audio_bytes, script = await generate_voice_blessing(name, tier=tier, amount=amount)
        if audio_bytes and context.bot:
            voice_file = io.BytesIO(audio_bytes)
            voice_file.name = "test_blessing.ogg"
            await context.bot.send_voice(
                chat_id=user.id,
                voice=voice_file,
                caption=(
                    f"🎙️ <b>AI Voice Blessing Preview</b> ({tier_info.get('title', tier)})\n\n"
                    f"📜 <b>អត្ថបទ៖</b> <i>«{script}»</i>"
                ),
                parse_mode="HTML",
            )
            await wait_msg.delete()
        else:
            await wait_msg.edit_text("❌ បរាជ័យក្នុងការបង្កើតសំឡេង។")
    except Exception as e:
        logger.error("Test blessing error: %s", e)
        await wait_msg.edit_text(f"❌ កំហុស៖ {e}")


async def donation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle interactive donation callbacks (tier selection, payment confirmation, admin approvals)."""
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data
    user = query.from_user
    user_name = html.escape(user.first_name or "បង") if user else "បង"
    user_id = user.id if user else 0

    # -------------------------------------------------------------------------
    # 1. Back to main donation menu
    # -------------------------------------------------------------------------
    if data == "donate_menu":
        await query.answer()
        text = (
            f"☕ <b>សូមស្វាគមន៍មកកាន់ការឧបត្ថម្ភ Bot Voice!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"ការឧបត្ថម្ភរបស់បង <b>{user_name}</b> ជួយឱ្យ Bot ដំណើរការបានលឿន ឥតគាំង និងឥតគិតថ្លៃសម្រាប់បងប្អូនខ្មែរទាំងអស់! 💖\n\n"
            f"👇 <b>សូមជ្រើសរើសចំនួនដែលបងចង់ឧបត្ថម្ភ៖</b>"
        )
        if query.message:
            try:
                await query.message.edit_text(
                    text,
                    reply_markup=_build_donation_menu_markup(),
                    parse_mode="HTML",
                )
            except Exception:
                await query.message.reply_text(
                    text,
                    reply_markup=_build_donation_menu_markup(),
                    parse_mode="HTML",
                )
        return

    # -------------------------------------------------------------------------
    # 2. View Hall of Fame from callback
    # -------------------------------------------------------------------------
    if data == "donate_halloffame":
        await query.answer()
        await cmd_donors(update, context)
        return

    # -------------------------------------------------------------------------
    # 3. User selects a tier: generate Bakong KHQR
    # -------------------------------------------------------------------------
    if data.startswith("donate_tier:"):
        tier_key = data.split(":", 1)[1].lower()
        tier_info = TIER_DETAILS.get(tier_key, TIER_DETAILS["coffee"])
        amount = tier_info["amount"]
        tier_title = tier_info["title"]

        await query.answer(f"កំពុងបង្កើត Bakong KHQR សម្រាប់ {tier_title}...")
        await _send_khqr_screen(
            chat_id=user_id,
            user_name=user_name,
            amount=amount,
            tier_key=tier_key,
            tier_title=tier_title,
            context=context,
        )
        return

    # -------------------------------------------------------------------------
    # 4. User clicks "I have paid"
    # -------------------------------------------------------------------------
    if data.startswith("donate_paid:"):
        parts = data.split(":")
        tier_key = parts[1] if len(parts) > 1 else "coffee"
        bill_no = parts[2] if len(parts) > 2 else ""
        try:
            amount = float(parts[3]) if len(parts) > 3 else TIER_DETAILS.get(tier_key, {}).get("amount", 1.0)
        except ValueError:
            amount = 1.0

        tier_info = TIER_DETAILS.get(tier_key, {})
        tier_title = tier_info.get("title", f"${amount:.2f}")

        await query.answer("អរគុណបង! ប្រព័ន្ធកំពុងផ្ទៀងផ្ទាត់...", show_alert=False)

        # Acknowledge to user
        if query.message:
            await query.message.reply_text(
                f"🙏 <b>សូមអរគុណបង {user_name}!</b>\n\n"
                f"ប្រព័ន្ធបានទទួលការជូនដំណឹងពីការឧបត្ថម្ភ <b>{tier_title} (${amount:.2f})</b> រួចរាល់ហើយ។\n"
                f"បន្ទាប់ពីការផ្ទៀងផ្ទាត់ Bot នឹងផ្ញើសារសំឡេងអរគុណ និងជូនពរពិសេស (AI Voice Blessing) ជូនបងភ្លាមៗ! ❤️☕\n\n"
                f"🏆 <i>ពិនិត្យមើលតារាងកិត្តិយស៖</i> /donors",
                parse_mode="HTML",
            )

        # Notify Administrators with 1-click Approval
        approval_token = f"{user_id}:{amount:.2f}:{tier_key}:{bill_no}"
        admin_markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "💖 អនុម័ត & ផ្ញើសំឡេងជូនពរ",
                    callback_data=f"donate_approve:{approval_token}",
                ),
                InlineKeyboardButton("❌ បដិសេធ", callback_data=f"donate_reject:{user_id}:{bill_no}"),
            ],
        ])

        admin_notification = (
            f"🎉 <b>មានការជូនដំណឹងឧបត្ថម្ភថ្មី!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>សប្បុរសជន:</b> {user_name} (@{user.username or 'N/A'})\n"
            f"🆔 <b>Telegram ID:</b> <code>{user_id}</code>\n"
            f"☕ <b>កញ្ចប់:</b> {tier_title}\n"
            f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>${amount:.2f} USD</b>\n"
            f"🧾 <b>វិក្កយបត្រ:</b> <code>{bill_no}</code>\n"
            f"⏰ <b>ម៉ោង:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"សូមពិនិត្យគណនីធនាគារ Bakong/ABA របស់បង រួចចុចប៊ូតុងខាងក្រោមដើម្បីអនុម័ត៖"
        )

        admin_ids = get_admin_ids()
        for aid in admin_ids:
            try:
                if context.bot:
                    await context.bot.send_message(
                        chat_id=aid,
                        text=admin_notification,
                        reply_markup=admin_markup,
                        parse_mode="HTML",
                    )
            except Exception as e:
                logger.warning("Failed to notify admin %s of donation: %s", aid, e)
        return

    # -------------------------------------------------------------------------
    # 5. Admin Approves Donation (with duplicate protection)
    # -------------------------------------------------------------------------
    if data.startswith("donate_approve:"):
        if not is_admin_user(user_id):
            await query.answer("⛔ មានតែ Admin ប៉ុណ្ណោះដែលអាចអនុម័តបាន!", show_alert=True)
            return

        payload_part = data[len("donate_approve:"):]
        parts = payload_part.split(":")
        donor_uid = int(parts[0])
        amount = float(parts[1])
        tier_key = parts[2] if len(parts) > 2 else "coffee"
        bill_no = parts[3] if len(parts) > 3 else f"{donor_uid}_{amount}"

        # Deduplication guard: ensure donation is processed only once
        dedup_key = f"{donor_uid}:{bill_no}"
        with _APPROVALS_LOCK:
            if dedup_key in _PROCESSED_APPROVALS:
                await query.answer("⚠️ ការឧបត្ថម្ភនេះត្រូវបានអនុម័តរួចរាល់ហើយ!", show_alert=True)
                if query.message:
                    await query.message.edit_text(
                        f"{query.message.text}\n\n"
                        f"ℹ️ <b>ការឧបត្ថម្ភនេះត្រូវបានអនុម័តរួចរាល់ជាស្ថាពរហើយ។</b>",
                        parse_mode="HTML",
                    )
                return
            _PROCESSED_APPROVALS.add(dedup_key)

        await query.answer("កំពុងអនុម័ត និងបង្កើតសំឡេងជូនពរ...")

        # Resolve donor's real name
        donor_name = "បង"
        if context.bot:
            try:
                chat = await context.bot.get_chat(donor_uid)
                if chat and chat.first_name:
                    donor_name = chat.first_name
            except Exception:
                pass

        # 1. Record donation
        record = await donation_store.record_donation(
            user_id=donor_uid,
            full_name=donor_name if donor_name != "បង" else f"User {donor_uid}",
            amount=amount,
            tier=tier_key,
            blessing_sent=True,
        )

        # 2. Synthesize & Send Voice Blessing
        blessing_ok = False
        if context.bot:
            blessing_ok = await deliver_voice_blessing(
                context.bot,
                user_id=donor_uid,
                donor_name=donor_name,
                tier=tier_key,
                amount=amount,
            )

        status_txt = "🎙️ បានផ្ញើសារសំឡេងជូនពររួចរាល់!" if blessing_ok else "⚠️ មិនអាចផ្ញើសំឡេងទៅ Telegram បានទេ"
        if query.message:
            await query.message.edit_text(
                f"{query.message.text}\n\n"
                f"✅ <b>បានអនុម័តជោគជ័យដោយ Admin!</b>\n"
                f"{status_txt}\n"
                f"🏆 បានបញ្ចូលក្នុងតារាងកិត្តិយស (/donors)!",
                parse_mode="HTML",
            )
        return

    # -------------------------------------------------------------------------
    # 6. Admin Rejects Donation Notification
    # -------------------------------------------------------------------------
    if data.startswith("donate_reject:"):
        if not is_admin_user(user_id):
            await query.answer("⛔ មានតែ Admin ប៉ុណ្ណោះដែលអាចបដិសេធបាន!", show_alert=True)
            return

        await query.answer("បានបដិសេធការជូនដំណឹងនេះ។")
        if query.message:
            await query.message.edit_text(
                f"{query.message.text}\n\n"
                f"❌ <b>ការជូនដំណឹងនេះត្រូវបានបដិសេធដោយ Admin។</b>",
                parse_mode="HTML",
            )
        return
