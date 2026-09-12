"""Telegram commands and callback handlers for Bakong KHQR donations and Hall of Fame."""

from __future__ import annotations

import html
import io
import json
import logging
import os
import threading
import time
from contextlib import suppress
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from app.core.config import SETTINGS
from app.services.donation.blessing import (
    deliver_voice_blessing,
    generate_voice_blessing,
)
from app.services.donation import bakong_api
from app.services.donation.khqr import (
    DEFAULT_BAKONG_ACCOUNT_ID,
    DEFAULT_BAKONG_MERCHANT_NAME,
    BakongKHQR,
    generate_khqr_string,
    get_khqr_qr_image,
)
from app.services.donation.store import TIER_DETAILS, donation_store

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(PROJECT_ROOT, "data")
PENDING_TICKETS_PATH = os.path.join(DATA_DIR, "pending_donations.json")

# Concurrency & deduplication guards for admin approvals
_PROCESSED_APPROVALS: set[str] = set()
_APPROVALS_LOCK = threading.Lock()

# Bounded store for pending approval tickets (guarantees callback_data <= 64 bytes)
_PENDING_DONATIONS: dict[str, dict[str, Any]] = {}
_PENDING_LOCK = threading.Lock()

# Bounded in-memory store for active KHQR bill payloads (for fast MD5 lookup)
_ACTIVE_BILLS: dict[str, dict[str, Any]] = {}
_ACTIVE_BILLS_LOCK = threading.Lock()


def _load_pending_tickets() -> None:
    """Load pending donation tickets from persistent store on startup."""
    global _PENDING_DONATIONS
    if not os.path.isfile(PENDING_TICKETS_PATH):
        return
    try:
        with open(PENDING_TICKETS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                with _PENDING_LOCK:
                    _PENDING_DONATIONS.update(data)
    except Exception as e:
        logger.warning("Failed to load pending donations from %s: %s", PENDING_TICKETS_PATH, e)


def _save_pending_tickets() -> None:
    """Safely persist pending donation tickets to disk with atomic write."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = f"{PENDING_TICKETS_PATH}.tmp"
        with _PENDING_LOCK:
            data = dict(_PENDING_DONATIONS)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PENDING_TICKETS_PATH)
    except Exception as e:
        logger.warning("Failed to save pending donations to %s: %s", PENDING_TICKETS_PATH, e)


# Initialize persisted tickets on module load
_load_pending_tickets()


def is_admin_user(user_id: int) -> bool:
    """Check whether a given user_id is an authorized administrator."""
    with suppress(Exception):
        from app.legacy import _is_admin  # type: ignore

        if _is_admin(user_id):
            return True

    admin_str = os.getenv("ADMIN_IDS", "") or getattr(SETTINGS, "ADMIN_IDS", "")
    for part in admin_str.split(","):
        part = part.strip()
        if part.isdigit() and int(part) == user_id:
            return True
    return False


def get_admin_ids() -> set[int]:
    """Retrieve all configured administrator Telegram IDs."""
    admin_ids: set[int] = set()
    with suppress(Exception):
        from app.legacy import ADMIN_IDS  # type: ignore

        if isinstance(ADMIN_IDS, (set, list, tuple)):
            admin_ids.update(int(aid) for aid in ADMIN_IDS if str(aid).isdigit())

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
        [
            InlineKeyboardButton("🔙 ត្រឡប់ទៅម៉ឺនុយដើម", callback_data="donate_close"),
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
        f"👤 <b>ឈ្មោះគណនី:</b> <code>{html.escape(DEFAULT_BAKONG_MERCHANT_NAME)}</code>\n"
        f"🆔 <b>Bakong ID:</b> <code>{html.escape(DEFAULT_BAKONG_ACCOUNT_ID)}</code>\n"
        f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>${amount:.2f} USD</b>\n"
        f"🧾 <b>លេខវិក្កយបត្រ:</b> <code>{html.escape(bill_no)}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📲 <b>របៀបបង់ប្រាក់៖</b>\n"
        f"1. បើកកម្មវិធីធនាគាររបស់បង (ABA, ACLEDA, Wing, Canadia, Bakong...)\n"
        f"2. ស្កេនរូបភាព QR កូដនេះ\n"
        f"3. ផ្ទៀងផ្ទាត់ចំនួន <b>${amount:.2f}</b> រួចចុចផ្ទេរប្រាក់\n"
        f"4. បន្ទាប់ពីផ្ទេររួច សូមចុចប៊ូតុង <b>«✅ ខ្ញុំបានផ្ទេរប្រាក់រួចរាល់»</b> ខាងក្រោម\n\n"
        f"✨ <i>Bot នឹងផ្ញើសារសំឡេងអរគុណពិសេសជូនបងភ្លាមៗ!</i>"
    )

    # Compact callback_data ensuring it stays well under the 64-byte Telegram limit
    tier_short = tier_key[:8]
    bill_short = bill_no[:10]
    paid_cb = f"donate_paid:{tier_short}:{bill_short}:{amount:.2f}"

    # Cache active bill payload and MD5 for instant Bakong Open API verification
    khqr_md5 = BakongKHQR.get_md5(khqr_text)
    with _ACTIVE_BILLS_LOCK:
        if len(_ACTIVE_BILLS) > 300:
            for k in list(_ACTIVE_BILLS.keys())[:100]:
                _ACTIVE_BILLS.pop(k, None)
        _ACTIVE_BILLS[bill_short] = {
            "khqr_text": khqr_text,
            "md5": khqr_md5,
            "amount": amount,
            "tier_key": tier_key,
            "bill_no": bill_no,
            "chat_id": chat_id,
            "created_at": time.time(),
        }

    action_buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ ខ្ញុំបានផ្ទេរប្រាក់រួចរាល់", callback_data=paid_cb)],
        [InlineKeyboardButton("🔙 ជ្រើសរើសចំនួនផ្សេង", callback_data="donate_menu")],
    ])

    qr_bytes = await get_khqr_qr_image(khqr_text)
    if qr_bytes and context.bot:
        try:
            photo_file = io.BytesIO(qr_bytes)
            ext = "webp" if qr_bytes.startswith(b"RIFF") else ("jpg" if qr_bytes.startswith(b"\xff\xd8") else "png")
            photo_file.name = f"khqr.{ext}"
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
                target_chat_id = update.effective_chat.id if update.effective_chat else user.id
                await _send_khqr_screen(
                    chat_id=target_chat_id,
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
    """Display Hall of Fame (/donors) with top contributors and metrics. Protects donor privacy."""
    user = update.effective_user
    chat = update.effective_chat
    target_chat_id = chat.id if chat else (user.id if user else 0)

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
            raw_name = str(item.get("full_name") or "").strip()
            # Privacy guard: mask raw user IDs from being exposed publicly
            if not raw_name or raw_name.lower().startswith("user ") or raw_name.isdigit():
                uid_str = str(item.get("user_id", ""))
                raw_name = f"Supporter *{uid_str[-4:]}" if len(uid_str) >= 4 else "សប្បុរសជន"
            name = html.escape(raw_name)
            amount = float(item.get("total_amount") or 0.0)
            cups = item.get("total_cups", 1)
            lines.append(f"{badge} <b>{name}</b> — ${amount:.2f} ({cups} កែវ)")
        lines.append("")
    else:
        lines.append("🌟 <i>មិនទាន់មានទិន្នន័យអ្នកឧបត្ថម្ភនៅឡើយទេ។ ចុចប៊ូតុងខាងក្រោមដើម្បីក្លាយជាអ្នកឧបត្ថម្ភដំបូងបង្អស់!</i>\n")

    if recent:
        lines.append("🕒 <b>អ្នកឧបត្ថម្ភថ្មីៗ (Recent Supporters):</b>")
        for r in recent:
            raw_name = str(r.get("full_name") or "").strip()
            if not raw_name or raw_name.lower().startswith("user ") or raw_name.isdigit():
                uid_str = str(r.get("user_id", ""))
                raw_name = f"Supporter *{uid_str[-4:]}" if len(uid_str) >= 4 else "សប្បុរសជន"
            name = html.escape(raw_name)
            amt = float(r.get("amount") or 0.0)
            tier = r.get("tier", "coffee")
            tier_info = TIER_DETAILS.get(tier, {})
            tier_title = tier_info.get("title", f"${amt:.2f}")
            lines.append(f"• <b>{name}</b> — ${amt:.2f} ({tier_title})")
        lines.append("")

    lines.append("💖 <i>សូមថ្លែងអំណរគុណយ៉ាងជ្រាលជ្រៅដល់បងប្អូនទាំងអស់ដែលបានចូលរួមចំណែកគាំទ្រ Bot Voice!</i>")

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("☕ ចូលរួមឧបត្ថម្ភ / Buy Coffee", callback_data="donate_menu"),
            InlineKeyboardButton("🔄 ធ្វើបច្ចុប្បន្នភាព / Refresh", callback_data="donate_halloffame_refresh"),
        ],
        [
            InlineKeyboardButton("🔙 ត្រឡប់ទៅម៉ឺនុយដើម", callback_data="donate_close"),
        ],
    ])

    text = "\n".join(lines)
    query = update.callback_query
    if query and query.message:
        if query.message.photo:
            with suppress(Exception):
                await query.message.delete()
            if context.bot and target_chat_id:
                await context.bot.send_message(
                    chat_id=target_chat_id,
                    text=text,
                    reply_markup=keyboard,
                    parse_mode="HTML",
                )
            return
        with suppress(Exception):
            await query.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
            return

    msg = update.effective_message
    if msg:
        await msg.reply_text(text, reply_markup=keyboard, parse_mode="HTML")


def _build_adddonor_amount_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("☕ $1.00 (Coffee)", callback_data="donate_add_tier:coffee:1.0"),
            InlineKeyboardButton("🧋 $2.00 (Milk Tea)", callback_data="donate_add_tier:milktea:2.0"),
        ],
        [
            InlineKeyboardButton("🍜 $3.00 (Lunch)", callback_data="donate_add_tier:lunch:3.0"),
            InlineKeyboardButton("🖥️ $5.00 (Server)", callback_data="donate_add_tier:server:5.0"),
        ],
        [
            InlineKeyboardButton("🌟 $10.00 (Patron)", callback_data="donate_add_tier:patron:10.0"),
            InlineKeyboardButton("💎 $20.00 (Gold)", callback_data="donate_add_tier:gold:20.0"),
        ],
        [
            InlineKeyboardButton("✍️ វាយចំនួនផ្សេង (Custom)", callback_data="donate_add_custom"),
            InlineKeyboardButton("❌ បោះបង់", callback_data="donate_add_cancel"),
        ],
    ])


async def _show_adddonor_step_amount(target_msg: Any, data: dict[str, Any], *, edit: bool = False) -> None:
    donor_uid = data.get("user_id", 0)
    auto_name = data.get("auto_name") or f"User {donor_uid}"
    text = (
        "➕ <b>បន្ថែមអ្នកឧបត្ថម្ភ (Add Donor) — ជំហានទី ២/៤</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>អ្នកឧបត្ថម្ភ:</b> {html.escape(auto_name)}\n"
        f"🆔 <b>Telegram ID:</b> <code>{donor_uid}</code>\n\n"
        "💵 <b>សូមជ្រើសរើសកម្រិតឧបត្ថម្ភ (Tier) ឬវាយបញ្ចូលចំនួនទឹកប្រាក់ ($ USD)៖</b>\n"
        "<i>(អ្នកអាចចុចប៊ូតុងខាងក្រោម ឬវាយលេខដូចជា 1.5, 5, 25 ផ្ញើមកទីនេះ)</i>"
    )
    markup = _build_adddonor_amount_markup()
    if edit and hasattr(target_msg, "edit_text"):
        with suppress(Exception):
            await target_msg.edit_text(text, parse_mode="HTML", reply_markup=markup)
            return
    await target_msg.reply_text(text, parse_mode="HTML", reply_markup=markup)


def _build_adddonor_name_markup(data: dict[str, Any]) -> InlineKeyboardMarkup:
    donor_uid = data.get("user_id", 0)
    auto_name = (data.get("auto_name") or "").strip()
    rows = []
    if auto_name and auto_name != f"User {donor_uid}":
        rows.append([InlineKeyboardButton(f"✅ ប្រើឈ្មោះ: {auto_name[:25]}", callback_data="donate_add_use_auto")])
    rows.append([InlineKeyboardButton(f"👤 ប្រើឈ្មោះ: User {donor_uid}", callback_data="donate_add_use_id")])
    rows.append([InlineKeyboardButton("❌ បោះបង់", callback_data="donate_add_cancel")])
    return InlineKeyboardMarkup(rows)


async def _show_adddonor_step_name(target_msg: Any, data: dict[str, Any], *, edit: bool = False) -> None:
    donor_uid = data.get("user_id", 0)
    amount = float(data.get("amount", 1.0))
    tier = data.get("tier", "coffee")
    tier_info = TIER_DETAILS.get(tier, {})
    tier_title = tier_info.get("title", tier)
    auto_name = data.get("auto_name") or f"User {donor_uid}"

    text = (
        "➕ <b>បន្ថែមអ្នកឧបត្ថម្ភ (Add Donor) — ជំហានទី ៣/៤</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 <b>Telegram ID:</b> <code>{donor_uid}</code>\n"
        f"💵 <b>ចំនួន:</b> ${amount:.2f} USD ({tier_title})\n\n"
        f"📝 <b>តើអ្នកចង់ដាក់ឈ្មោះអ្វីសម្រាប់បង្ហាញក្នុងតារាងកិត្តិយស?</b>\n"
        f"• ឈ្មោះ Telegram ស្វ័យប្រវត្តិ: <b>{html.escape(auto_name)}</b>\n\n"
        "<i>👉 ចុចប៊ូតុងខាងក្រោមដើម្បីជ្រើសឈ្មោះ ឬវាយឈ្មោះថ្មីផ្ញើមកទីនេះ (ឧ. Dara, លោកពូសុខ)៖</i>"
    )
    markup = _build_adddonor_name_markup(data)
    if edit and hasattr(target_msg, "edit_text"):
        with suppress(Exception):
            await target_msg.edit_text(text, parse_mode="HTML", reply_markup=markup)
            return
    await target_msg.reply_text(text, parse_mode="HTML", reply_markup=markup)


def _build_adddonor_confirm_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ បញ្ជាក់ & កត់ត្រា (Confirm)", callback_data="donate_add_confirm"),
            InlineKeyboardButton("❌ បោះបង់", callback_data="donate_add_cancel"),
        ]
    ])


async def _show_adddonor_step_confirm(target_msg: Any, data: dict[str, Any], *, edit: bool = False) -> None:
    donor_uid = data.get("user_id", 0)
    amount = float(data.get("amount", 1.0))
    tier = data.get("tier", "coffee")
    custom_name = data.get("custom_name") or f"User {donor_uid}"
    tier_info = TIER_DETAILS.get(tier, {})
    tier_title = tier_info.get("title", tier)
    tier_emoji = tier_info.get("emoji", "☕")

    text = (
        "📋 <b>ផ្ទៀងផ្ទាត់ព័ត៌មាន (ជំហានទី ៤/៤)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>សប្បុរសជន:</b> {html.escape(custom_name)}\n"
        f"🆔 <b>Telegram ID:</b> <code>{donor_uid}</code>\n"
        f"💵 <b>ចំនួនទឹកប្រាក់:</b> ${amount:.2f} USD\n"
        f"🎖️ <b>កម្រិត (Tier):</b> {tier_emoji} {tier_title}\n"
        "🎙️ <b>AI Voice Blessing:</b> បង្កើត និងផ្ញើសំឡេងជូនពរ\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "តើអ្នកពិតជាចង់កត់ត្រាការឧបត្ថម្ភនេះមែនទេ?"
    )
    markup = _build_adddonor_confirm_markup()
    if edit and hasattr(target_msg, "edit_text"):
        with suppress(Exception):
            await target_msg.edit_text(text, parse_mode="HTML", reply_markup=markup)
            return
    await target_msg.reply_text(text, parse_mode="HTML", reply_markup=markup)


async def handle_adddonor_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handles text input and forwarded messages during the interactive /adddonor wizard."""
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg or not is_admin_user(int(user.id)):
        return False

    state = context.user_data.get("adddonor_state")
    if not state:
        return False

    text = (msg.text or msg.caption or "").strip()
    if text.lower() in ("/cancel", "cancel", "បោះបង់"):
        context.user_data.pop("adddonor_state", None)
        context.user_data.pop("adddonor_data", None)
        await msg.reply_text("❌ <b>បានបោះបង់ការបន្ថែមអ្នកឧបត្ថម្ភ។</b>", parse_mode="HTML")
        return True

    data = context.user_data.setdefault("adddonor_data", {})

    # Step 1: Wait for User ID
    if state == "wait_user_id":
        donor_uid = None
        donor_auto_name = ""

        fwd_user = getattr(msg, "forward_from", None)
        if fwd_user and getattr(fwd_user, "id", None):
            donor_uid = int(fwd_user.id)
            donor_auto_name = fwd_user.first_name or fwd_user.full_name or ""
        elif getattr(msg, "forward_from_chat", None):
            donor_uid = int(msg.forward_from_chat.id)
            donor_auto_name = msg.forward_from_chat.title or ""

        if donor_uid is None and text.isdigit():
            donor_uid = int(text)

        if donor_uid is None:
            await msg.reply_text(
                "⚠️ <b>មិនស្គាល់ Telegram User ID ទេ។</b>\n\n"
                "សូមវាយលេខសម្គាល់ជាលេខសុទ្ធ (ឧ. <code>1272791365</code>) ឬ Forward សារពីគាត់មកទីនេះ។\n"
                "<i>(ផ្ញើ /cancel ដើម្បីបោះបង់)</i>",
                parse_mode="HTML",
            )
            return True

        if not donor_auto_name and context.bot:
            with suppress(Exception):
                chat = await context.bot.get_chat(donor_uid)
                if chat and (chat.first_name or chat.title):
                    donor_auto_name = chat.first_name or chat.title or ""

        if not donor_auto_name:
            donor_auto_name = f"User {donor_uid}"

        data["user_id"] = donor_uid
        data["auto_name"] = donor_auto_name
        data["custom_name"] = donor_auto_name
        context.user_data["adddonor_state"] = "wait_amount"
        await _show_adddonor_step_amount(msg, data)
        return True

    # Step 2: Wait for Amount
    if state == "wait_amount":
        clean_text = text.replace("$", "").strip()
        try:
            amt = float(clean_text)
            if amt <= 0 or amt > 10000.0:
                await msg.reply_text("❌ ចំនួនទឹកប្រាក់ត្រូវតែចន្លោះពី $0.01 ដល់ $10,000.00 USD។")
                return True
        except ValueError:
            await msg.reply_text(
                "❌ សូមវាយចំនួនទឹកប្រាក់ជាលេខ (ឧទាហរណ៍៖ <code>1.0</code>, <code>2.5</code>, <code>5</code>) ឬចុចប៊ូតុងខាងលើ៖",
                parse_mode="HTML",
            )
            return True

        tier = next((k for k, v in TIER_DETAILS.items() if abs(v.get("amount", 0.0) - amt) < 0.01), "coffee")
        data["amount"] = amt
        data["tier"] = tier
        context.user_data["adddonor_state"] = "wait_name"
        await _show_adddonor_step_name(msg, data)
        return True

    # Step 3: Wait for Name
    if state == "wait_name":
        data["custom_name"] = text[:60]
        context.user_data["adddonor_state"] = "confirm"
        await _show_adddonor_step_confirm(msg, data)
        return True

    return False


async def cmd_adddonor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin command to credit a donor and trigger voice blessing.

    Supports:
    1. Direct 1-line execution: /adddonor <user_id> <amount> [tier] [name]
    2. Interactive Step-by-Step wizard: /adddonor (with 0 arguments)
    """
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    if not is_admin_user(int(user.id)):
        await msg.reply_text("⛔ អ្នកមិនមានសិទ្ធិប្រើប្រាស់ពាក្យបញ្ជានេះទេ។")
        return

    help_message = (
        "ℹ️ <b>របៀបប្រើប្រាស់ពាក្យបញ្ជា /adddonor:</b>\n\n"
        "<code>/adddonor &lt;user_id&gt; &lt;amount&gt; [tier] [name]</code>\n\n"
        "• ឧទាហរណ៍៖ <code>/adddonor 1272791365 1.0 coffee Dara</code>\n"
        "• Tiers: <code>coffee</code> ($1), <code>milktea</code> ($2), <code>lunch</code> ($3), <code>server</code> ($5), <code>patron</code> ($10)\n\n"
        "💡 ឬវាយ <code>/adddonor</code> ដោយមិនដាក់ parameter ដើម្បីដំណើរការតាមជំហាន (Step by Step)!"
    )

    args = context.args or []

    # Case A: Explicit help requested
    if len(args) == 1 and args[0].lower() in {"help", "info", "?"}:
        await msg.reply_text(help_message, parse_mode="HTML")
        return

    # Case B: Launch Step-by-Step flow (no args or only user_id provided)
    if not args:
        context.user_data["adddonor_state"] = "wait_user_id"
        context.user_data["adddonor_data"] = {}
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ បោះបង់ (Cancel)", callback_data="donate_add_cancel")]])
        await msg.reply_text(
            "➕ <b>បន្ថែមអ្នកឧបត្ថម្ភ (Add Donor) — ជំហានទី ១/៤</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "🆔 <b>សូមបញ្ចូល Telegram User ID របស់អ្នកឧបត្ថម្ភ៖</b>\n\n"
            "• វាយលេខសម្គាល់ ឧទាហរណ៍៖ <code>1272791365</code>\n"
            "• ឬ <b>Forward</b> សារពីគាត់មកកាន់ទីនេះ\n\n"
            "<i>ផ្ញើ /cancel ដើម្បីបោះបង់</i>",
            parse_mode="HTML",
            reply_markup=kb,
        )
        return

    if len(args) == 1 and args[0].isdigit():
        donor_uid = int(args[0])
        donor_name = f"User {donor_uid}"
        if context.bot:
            with suppress(Exception):
                chat = await context.bot.get_chat(donor_uid)
                if chat and (chat.first_name or chat.title):
                    donor_name = chat.first_name or chat.title or donor_name
        data = {
            "user_id": donor_uid,
            "auto_name": donor_name,
            "custom_name": donor_name,
        }
        context.user_data["adddonor_data"] = data
        context.user_data["adddonor_state"] = "wait_amount"
        await _show_adddonor_step_amount(msg, data)
        return

    # Case C: Direct one-line execution
    try:
        donor_uid = int(args[0])
        amount = float(args[1])
        if amount <= 0 or amount > 10000.0:
            await msg.reply_text("❌ ចំនួនទឹកប្រាក់ត្រូវតែចន្លោះពី $0.01 ដល់ $10,000.00 USD។")
            return
    except ValueError:
        await msg.reply_text(
            f"❌ <b>user_id និង amount ត្រូវតែជាតួលេខ។</b>\n\n{help_message}",
            parse_mode="HTML",
        )
        return

    if len(args) > 2 and args[2].lower() in TIER_DETAILS:
        tier = args[2].lower()
        custom_name = " ".join(args[3:]) if len(args) > 3 else ""
    elif len(args) > 2:
        custom_name = " ".join(args[2:])
        tier = next((k for k, v in TIER_DETAILS.items() if abs(v.get("amount", 0.0) - amount) < 0.01), "coffee")
    else:
        custom_name = ""
        tier = next((k for k, v in TIER_DETAILS.items() if abs(v.get("amount", 0.0) - amount) < 0.01), "coffee")

    # Attempt to auto-fetch donor's actual name from Telegram if omitted
    if not custom_name and context.bot:
        with suppress(Exception):
            chat = await context.bot.get_chat(donor_uid)
            if chat and chat.first_name:
                custom_name = chat.first_name

    if not custom_name:
        custom_name = f"User {donor_uid}"

    # 1. Record in store
    await donation_store.record_donation(
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
                    f"📜 <b>អត្ថបទ៖</b> <i>«{html.escape(script)}»</i>"
                ),
                parse_mode="HTML",
            )
            await wait_msg.delete()
        else:
            await wait_msg.edit_text("❌ បរាជ័យក្នុងការបង្កើតសំឡេង។")
    except Exception as e:
        logger.error("Test blessing error: %s", e)
        await wait_msg.edit_text(f"❌ កំហុស៖ {e}")


async def _execute_donation_approval(
    *,
    donor_uid: int,
    amount: float,
    tier_key: str,
    bill_no: str,
    ticket_id: str,
    context: ContextTypes.DEFAULT_TYPE,
    approved_by: str = "Admin",
    notify_donor_text: str = "",
) -> tuple[bool, str]:
    """Execute shared donation approval, AI voice blessing delivery, and stats recording."""
    dedup_key = f"{donor_uid}:{bill_no or ticket_id or amount}"
    with _APPROVALS_LOCK:
        if dedup_key in _PROCESSED_APPROVALS:
            return False, "ALREADY_PROCESSED"
        _PROCESSED_APPROVALS.add(dedup_key)

    # Resolve donor's real name from Telegram
    donor_name = "បង"
    if context.bot:
        with suppress(Exception):
            chat = await context.bot.get_chat(donor_uid)
            if chat and chat.first_name:
                donor_name = chat.first_name

    # 1. Record donation
    await donation_store.record_donation(
        user_id=donor_uid,
        full_name=donor_name if donor_name != "បង" else f"Supporter *{str(donor_uid)[-4:]}",
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

    # 3. Deliver optional direct notification to donor
    if notify_donor_text and context.bot:
        with suppress(Exception):
            await context.bot.send_message(
                chat_id=donor_uid,
                text=notify_donor_text,
                parse_mode="HTML",
            )

    # 4. Remove approved ticket from pending
    if ticket_id:
        with _PENDING_LOCK:
            _PENDING_DONATIONS.pop(ticket_id, None)
        _save_pending_tickets()

    status_txt = "🎙️ បានផ្ញើសារសំឡេងជូនពររួចរាល់!" if blessing_ok else "⚠️ មិនអាចផ្ញើសំឡេងទៅ Telegram បានទេ"
    return True, status_txt


async def donation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle interactive donation callbacks (tier selection, payment confirmation, admin approvals)."""
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data
    user = query.from_user
    user_name = html.escape(user.first_name or "បង") if user else "បង"
    user_id = user.id if user else 0
    chat = update.effective_chat
    target_chat_id = chat.id if chat else user_id

    # -------------------------------------------------------------------------
    # 0. Add Donor Step-by-Step Callbacks
    # -------------------------------------------------------------------------
    if data == "donate_add_cancel":
        await query.answer()
        context.user_data.pop("adddonor_state", None)
        context.user_data.pop("adddonor_data", None)
        if query.message:
            with suppress(Exception):
                await query.message.edit_text("❌ <b>បានបោះបង់ការបន្ថែមអ្នកឧបត្ថម្ភ។</b>", parse_mode="HTML")
        return

    if data == "donate_add_start":
        await query.answer()
        if not is_admin_user(user_id):
            await query.answer("⛔ សម្រាប់អ្នកគ្រប់គ្រងប៉ុណ្ណោះ។", show_alert=True)
            return
        context.user_data["adddonor_state"] = "wait_user_id"
        context.user_data["adddonor_data"] = {}
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ បោះបង់", callback_data="donate_add_cancel")]])
        if query.message:
            await query.message.reply_text(
                "➕ <b>បន្ថែមអ្នកឧបត្ថម្ភ (Add Donor) — ជំហានទី ១/៤</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "🆔 <b>សូមបញ្ចូល Telegram User ID របស់អ្នកឧបត្ថម្ភ៖</b>\n\n"
                "• វាយលេខសម្គាល់ ឧទាហរណ៍៖ <code>1272791365</code>\n"
                "• ឬ <b>Forward</b> សារពីគាត់មកកាន់ទីនេះ\n\n"
                "<i>ផ្ញើ /cancel ដើម្បីបោះបង់</i>",
                parse_mode="HTML",
                reply_markup=kb,
            )
        return

    if data.startswith("donate_add_tier:"):
        await query.answer()
        parts = data.split(":")
        tier = parts[1] if len(parts) > 1 else "coffee"
        amt = float(parts[2]) if len(parts) > 2 else 1.0

        step_data = context.user_data.setdefault("adddonor_data", {})
        step_data["tier"] = tier
        step_data["amount"] = amt
        context.user_data["adddonor_state"] = "wait_name"
        if query.message:
            await _show_adddonor_step_name(query.message, step_data, edit=True)
        return

    if data == "donate_add_custom":
        await query.answer()
        context.user_data["adddonor_state"] = "wait_amount"
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("❌ បោះបង់", callback_data="donate_add_cancel")]])
        if query.message:
            await query.message.reply_text(
                "💵 <b>សូមវាយចំនួនទឹកប្រាក់ ($ USD) ដែលចង់កត់ត្រា៖</b>\n\n"
                "ឧទាហរណ៍៖ <code>1.0</code>, <code>2.5</code>, <code>15.0</code>",
                parse_mode="HTML",
                reply_markup=kb,
            )
        return

    if data in ("donate_add_use_auto", "donate_add_use_id"):
        await query.answer()
        step_data = context.user_data.setdefault("adddonor_data", {})
        donor_uid = step_data.get("user_id", 0)
        if data == "donate_add_use_auto":
            step_data["custom_name"] = step_data.get("auto_name") or f"User {donor_uid}"
        else:
            step_data["custom_name"] = f"User {donor_uid}"
        context.user_data["adddonor_state"] = "confirm"
        if query.message:
            await _show_adddonor_step_confirm(query.message, step_data, edit=True)
        return

    if data == "donate_add_confirm":
        await query.answer()
        if not is_admin_user(user_id):
            await query.answer("⛔ សម្រាប់អ្នកគ្រប់គ្រងប៉ុណ្ណោះ។", show_alert=True)
            return

        step_data = context.user_data.pop("adddonor_data", {}) or {}
        context.user_data.pop("adddonor_state", None)

        donor_uid = int(step_data.get("user_id") or 0)
        amount = float(step_data.get("amount", 1.0))
        tier = step_data.get("tier", "coffee")
        custom_name = step_data.get("custom_name") or f"User {donor_uid}"
        tier_info = TIER_DETAILS.get(tier, {})

        if not donor_uid:
            if query.message:
                with suppress(Exception):
                    await query.message.edit_text("❌ ព័ត៌មានមិនត្រឹមត្រូវ។ សូមចាប់ផ្ដើមម្ដងទៀត /adddonor")
            return

        if query.message:
            with suppress(Exception):
                await query.message.edit_text("⏳ <b>កំពុងកត់ត្រា និងបង្កើតសំឡេងជូនពរ AI Voice Blessing...</b>", parse_mode="HTML")

        await donation_store.record_donation(
            user_id=donor_uid,
            full_name=custom_name,
            amount=amount,
            tier=tier,
            note=f"Added via wizard by admin {user_id}",
            blessing_sent=True,
        )

        blessing_sent = False
        if context.bot:
            blessing_sent = await deliver_voice_blessing(
                context.bot,
                user_id=donor_uid,
                donor_name=custom_name,
                tier=tier,
                amount=amount,
            )

        status_blessing = "🎙️ បានផ្ញើសារសំឡេងជូនពររួចរាល់" if blessing_sent else "⚠️ មិនអាចផ្ញើសំឡេងបានទេ (User មិនទាន់ /start)"

        success_text = (
            "🎉 <b>បានកត់ត្រាការឧបត្ថម្ភជោគជ័យ!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>សប្បុរសជន:</b> {html.escape(custom_name)} (ID: <code>{donor_uid}</code>)\n"
            f"💵 <b>ចំនួន:</b> ${amount:.2f} USD ({tier_info.get('title', tier)})\n"
            f"✨ <b>ស្ថានភាព:</b> {status_blessing}\n"
            f"🏆 <b>តារាងកិត្តិយស:</b> បានធ្វើបច្ចុប្បន្នភាព (/donors)"
        )
        success_kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🏆 តារាងកិត្តិយស (/donors)", callback_data="donate_halloffame"),
                InlineKeyboardButton("➕ បន្ថែមអ្នកថ្មី", callback_data="donate_add_start"),
            ],
            [
                InlineKeyboardButton("❌ បិទ (Close)", callback_data="donate_close")
            ]
        ])
        if query.message:
            with suppress(Exception):
                await query.message.edit_text(success_text, parse_mode="HTML", reply_markup=success_kb)
        return

    # -------------------------------------------------------------------------
    # 0. Close/dismiss donation menu
    # -------------------------------------------------------------------------
    if data in ("donate_close", "donate_back"):
        await query.answer()
        if query.message:
            with suppress(Exception):
                await query.message.delete()
        return

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
            if query.message.photo:
                with suppress(Exception):
                    await query.message.delete()
                if context.bot:
                    await context.bot.send_message(
                        chat_id=target_chat_id,
                        text=text,
                        reply_markup=_build_donation_menu_markup(),
                        parse_mode="HTML",
                    )
                return
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
    if data in ("donate_halloffame", "donate_halloffame_refresh"):
        if data == "donate_halloffame_refresh":
            donation_store._invalidate_cache()
            await query.answer("🔄 បានធ្វើបច្ចុប្បន្នភាពតារាងកិត្តិយសរួចរាល់!")
        else:
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
            chat_id=target_chat_id,
            user_name=user_name,
            amount=amount,
            tier_key=tier_key,
            tier_title=tier_title,
            context=context,
        )
        return

    # -------------------------------------------------------------------------
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

        # Lookup cached bill metadata or reconstruct
        bill_info = None
        with _ACTIVE_BILLS_LOCK:
            bill_info = _ACTIVE_BILLS.get(bill_no)

        if bill_info:
            khqr_text = bill_info.get("khqr_text", "")
            md5_hash = bill_info.get("md5", "")
        else:
            khqr_text = generate_khqr_string(
                amount=amount,
                currency="USD",
                bill_number=bill_no,
                reference_label=str(user_id),
            )
            md5_hash = BakongKHQR.get_md5(khqr_text)

        # Register ticket in memory
        ticket_id = f"t{int(time.time() * 1000) % 100000000:08x}"
        with _PENDING_LOCK:
            if len(_PENDING_DONATIONS) > 200:
                for k in list(_PENDING_DONATIONS.keys())[:50]:
                    _PENDING_DONATIONS.pop(k, None)
            _PENDING_DONATIONS[ticket_id] = {
                "user_id": user_id,
                "amount": amount,
                "tier_key": tier_key,
                "bill_no": bill_no,
                "user_name": user_name,
                "md5": md5_hash,
                "khqr_text": khqr_text,
                "created_at": time.time(),
            }
        _save_pending_tickets()

        # Check real-time via Bakong Open API if configured
        is_paid = False
        api_msg = ""
        if bakong_api.is_bakong_api_configured() and md5_hash:
            with suppress(Exception):
                await query.answer("🔍 កំពុងផ្ទៀងផ្ទាត់ជាមួយ Bakong Open API...")
            try:
                res = await bakong_api.check_transaction_by_md5(md5_hash)
                is_paid = bool(res.get("success"))
                api_msg = res.get("response_message", "")
            except Exception as exc:
                logger.error("Bakong Open API check error for md5=%s: %s", md5_hash, exc)

        if is_paid:
            # AUTO-APPROVED VIA BAKONG OPEN API!
            ok, status_txt = await _execute_donation_approval(
                donor_uid=user_id,
                amount=amount,
                tier_key=tier_key,
                bill_no=bill_no,
                ticket_id=ticket_id,
                context=context,
                approved_by="Bakong Open API (Auto)",
            )
            success_caption = (
                f"🎉 <b>ការផ្ទេរប្រាក់ត្រូវបានផ្ទៀងផ្ទាត់ជោគជ័យ!</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"🙏 <b>សូមអរគុណបង {user_name}!</b>\n\n"
                f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>${amount:.2f} USD</b> ({tier_title})\n"
                f"🧾 <b>លេខវិក្កយបត្រ:</b> <code>{html.escape(bill_no)}</code>\n"
                f"⚡ <b>ផ្ទៀងផ្ទាត់ដោយ:</b> <b>Bakong Open API (ស្វ័យប្រវត្តិ 100%)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"🎙️ <i>Bot បានផ្ញើសារសំឡេងអរគុណ និងជូនពរពិសេស (AI Voice Blessing) ជូនបងរួចរាល់ហើយ! ❤️☕</i>\n\n"
                f"🏆 <i>ពិនិត្យមើលតារាងកិត្តិយស៖</i> /donors"
            )
            success_kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🏆 តារាងកិត្តិយស (/donors)", callback_data="donate_halloffame")],
                [InlineKeyboardButton("☕ ឧបត្ថម្ភបន្ថែម", callback_data="donate_menu")],
            ])
            if query.message:
                with suppress(Exception):
                    await query.message.reply_text(success_caption, parse_mode="HTML", reply_markup=success_kb)

            # Notify admins of automated verification
            username_text = f"@{html.escape(user.username)}" if user and user.username else "N/A"
            admin_auto_msg = (
                f"⚡ <b>[Bakong Open API] ការឧបត្ថម្ភត្រូវបានផ្ទៀងផ្ទាត់ដោយស្វ័យប្រវត្តិ!</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"👤 <b>សប្បុរសជន:</b> {user_name} ({username_text})\n"
                f"🆔 <b>Telegram ID:</b> <code>{user_id}</code>\n"
                f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>${amount:.2f} USD</b> ({tier_title})\n"
                f"🧾 <b>លេខវិក្កយបត្រ:</b> <code>{html.escape(bill_no)}</code>\n"
                f"⏰ <b>ម៉ោង:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"✅ <i>{status_txt} និងបានកត់ត្រាចូលក្នុងប្រព័ន្ធរួចរាល់។</i>"
            )
            for aid in get_admin_ids():
                if context.bot:
                    with suppress(Exception):
                        await context.bot.send_message(chat_id=aid, text=admin_auto_msg, parse_mode="HTML")
            return

        # If not confirmed yet: notify user with retry button & notify admins with API check button
        with suppress(Exception):
            await query.answer("ប្រព័ន្ធកំពុងដំណើរការការផ្ទៀងផ្ទាត់...", show_alert=False)

        user_pending_text = (
            f"⏳ <b>មិនទាន់ឃើញប្រតិបត្តិការផ្ទេរប្រាក់នៅឡើយទេ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"បង <b>{user_name}</b> ប្រសិនបើបងទើបតែបានផ្ទេរប្រាក់តាម App ធនាគារ សូមរង់ចាំប្រហែល 5–10 វិនាទី រួចចុចប៊ូតុង <b>«🔄 ផ្ទៀងផ្ទាត់ម្តងទៀត»</b> ខាងក្រោម។\n\n"
            f"💵 <b>កញ្ចប់:</b> {tier_title} (${amount:.2f})\n"
            f"🧾 <b>វិក្កយបត្រ:</b> <code>{html.escape(bill_no)}</code>\n\n"
            f"💡 <i>ឬបងអាចរង់ចាំ Admin ពិនិត្យដោយផ្ទាល់។ នៅពេលផ្ទៀងផ្ទាត់រួចរាល់ Bot នឹងផ្ញើសំឡេងជូនពរជូនភ្លាមៗ! ❤️</i>"
        )
        user_pending_kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 ផ្ទៀងផ្ទាត់ម្តងទៀត (Check Again)", callback_data=f"donate_recheck:{ticket_id}")],
            [InlineKeyboardButton("🔙 ត្រឡប់ទៅម៉ឺនុយដើម", callback_data="donate_menu")],
        ])
        if query.message:
            with suppress(Exception):
                await query.message.reply_text(user_pending_text, parse_mode="HTML", reply_markup=user_pending_kb)

        admin_markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔍 ផ្ទៀងផ្ទាត់តាម Bakong API", callback_data=f"donate_api_check:{ticket_id}"),
            ],
            [
                InlineKeyboardButton("💖 អនុម័តដោយដៃ (Manual)", callback_data=f"donate_appr:{ticket_id}"),
                InlineKeyboardButton("❌ បដិសេធ", callback_data=f"donate_rej:{ticket_id}"),
            ],
        ])
        username_text = f"@{html.escape(user.username)}" if user and user.username else "N/A"
        admin_notification = (
            f"🎉 <b>មានការជូនដំណឹងឧបត្ថម្ភថ្មី!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"👤 <b>សប្បុរសជន:</b> {user_name} ({username_text})\n"
            f"🆔 <b>Telegram ID:</b> <code>{user_id}</code>\n"
            f"☕ <b>កញ្ចប់:</b> {tier_title}\n"
            f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>${amount:.2f} USD</b>\n"
            f"🧾 <b>វិក្កយបត្រ:</b> <code>{html.escape(bill_no)}</code>\n"
            f"⏰ <b>ម៉ោង:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីពិនិត្យតាម Bakong API ឬអនុម័តដោយផ្ទាល់៖</i>"
        )
        for aid in get_admin_ids():
            if context.bot:
                with suppress(Exception):
                    await context.bot.send_message(
                        chat_id=aid,
                        text=admin_notification,
                        reply_markup=admin_markup,
                        parse_mode="HTML",
                    )
        return

    # -------------------------------------------------------------------------
    # 4b. Donor retries verification via Bakong Open API
    # -------------------------------------------------------------------------
    if data.startswith("donate_recheck:"):
        ticket_id = data.split(":", 1)[1].strip()
        with _PENDING_LOCK:
            info = _PENDING_DONATIONS.get(ticket_id)
        if not info:
            await query.answer("⚠️ ព័ត៌មាននេះផុតកំណត់ ឬត្រូវបានអនុម័តរួចរាល់ហើយ!", show_alert=True)
            return

        donor_uid = int(info["user_id"])
        amount = float(info["amount"])
        tier_key = str(info["tier_key"])
        bill_no = str(info["bill_no"])
        md5_hash = str(info.get("md5") or "")
        khqr_text = str(info.get("khqr_text") or "")
        tier_info = TIER_DETAILS.get(tier_key, {})
        tier_title = tier_info.get("title", f"${amount:.2f}")

        if not md5_hash and khqr_text:
            md5_hash = BakongKHQR.get_md5(khqr_text)

        await query.answer("🔍 កំពុងផ្ទៀងផ្ទាត់ជាមួយ Bakong Open API...")

        is_paid = False
        api_msg = ""
        if md5_hash and bakong_api.is_bakong_api_configured():
            try:
                res = await bakong_api.check_transaction_by_md5(md5_hash)
                is_paid = bool(res.get("success"))
                api_msg = res.get("response_message", "")
            except Exception as exc:
                logger.error("Bakong Open API recheck error for md5=%s: %s", md5_hash, exc)

        if is_paid:
            ok, status_txt = await _execute_donation_approval(
                donor_uid=donor_uid,
                amount=amount,
                tier_key=tier_key,
                bill_no=bill_no,
                ticket_id=ticket_id,
                context=context,
                approved_by="Bakong Open API (Donor Retry)",
            )
            success_caption = (
                f"🎉 <b>ការផ្ទេរប្រាក់ត្រូវបានផ្ទៀងផ្ទាត់ជោគជ័យ!</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"🙏 <b>សូមអរគុណបង {user_name}!</b>\n\n"
                f"💵 <b>ចំនួនទឹកប្រាក់:</b> <b>${amount:.2f} USD</b> ({tier_title})\n"
                f"🧾 <b>លេខវិក្កយបត្រ:</b> <code>{html.escape(bill_no)}</code>\n"
                f"⚡ <b>ផ្ទៀងផ្ទាត់ដោយ:</b> <b>Bakong Open API (ស្វ័យប្រវត្តិ 100%)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"🎙️ <i>Bot បានផ្ញើសារសំឡេងអរគុណ និងជូនពរពិសេស (AI Voice Blessing) ជូនបងរួចរាល់ហើយ! ❤️☕</i>\n\n"
                f"🏆 <i>ពិនិត្យមើលតារាងកិត្តិយស៖</i> /donors"
            )
            success_kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🏆 តារាងកិត្តិយស (/donors)", callback_data="donate_halloffame")],
                [InlineKeyboardButton("☕ ឧបត្ថម្ភបន្ថែម", callback_data="donate_menu")],
            ])
            if query.message:
                with suppress(Exception):
                    await query.message.edit_text(success_caption, parse_mode="HTML", reply_markup=success_kb)
            return
        else:
            await query.answer(
                "⚠️ ប្រព័ន្ធ Bakong នៅតែមិនទាន់ឃើញប្រតិបត្តិការនេះទេ។ សូមរង់ចាំបន្តិច ឬរង់ចាំ Admin ពិនិត្យដោយដៃ។",
                show_alert=True,
            )
            return

    # -------------------------------------------------------------------------
    # 4c. Admin checks transaction via Bakong Open API
    # -------------------------------------------------------------------------
    if data.startswith("donate_api_check:"):
        if not is_admin_user(user_id):
            await query.answer("⛔ មានតែ Admin ប៉ុណ្ណោះដែលអាចប្រើបាន!", show_alert=True)
            return
        ticket_id = data.split(":", 1)[1].strip()
        with _PENDING_LOCK:
            info = _PENDING_DONATIONS.get(ticket_id)
        if not info:
            await query.answer("⚠️ សំណើនេះផុតកំណត់ ឬត្រូវបានអនុម័តរួចរាល់ហើយ!", show_alert=True)
            return

        donor_uid = int(info["user_id"])
        amount = float(info["amount"])
        tier_key = str(info["tier_key"])
        bill_no = str(info["bill_no"])
        md5_hash = str(info.get("md5") or "")
        khqr_text = str(info.get("khqr_text") or "")
        tier_info = TIER_DETAILS.get(tier_key, {})
        tier_title = tier_info.get("title", f"${amount:.2f}")

        if not md5_hash and khqr_text:
            md5_hash = BakongKHQR.get_md5(khqr_text)

        await query.answer("🔍 កំពុងពិនិត្យតាម Bakong Open API...")

        res = await bakong_api.check_transaction_by_md5(md5_hash) if md5_hash else {"success": False, "response_message": "No MD5"}
        if res.get("success"):
            ok, status_txt = await _execute_donation_approval(
                donor_uid=donor_uid,
                amount=amount,
                tier_key=tier_key,
                bill_no=bill_no,
                ticket_id=ticket_id,
                context=context,
                approved_by=f"Admin {user_id} via Bakong API",
                notify_donor_text=(
                    f"🎉 <b>ការផ្ទេរប្រាក់ត្រូវបានផ្ទៀងផ្ទាត់ជោគជ័យតាម Bakong Open API!</b>\n\n"
                    f"💵 <b>ចំនួន:</b> ${amount:.2f} USD ({tier_title})\n"
                    f"🧾 <b>វិក្កយបត្រ:</b> <code>{html.escape(bill_no)}</code>\n\n"
                    f"🎙️ <i>Bot បានផ្ញើសារសំឡេងអរគុណ និងជូនពរពិសេសជូនបងរួចរាល់ហើយ! ❤️☕</i>"
                ),
            )
            if query.message:
                orig_text = html.escape(query.message.text or "")
                with suppress(Exception):
                    await query.message.edit_text(
                        f"{orig_text}\n\n"
                        f"⚡ <b>បានផ្ទៀងផ្ទាត់ជោគជ័យតាម Bakong Open API!</b>\n"
                        f"{status_txt}\n"
                        f"🏆 បានបញ្ចូលក្នុងតារាងកិត្តិយស (/donors)!",
                        parse_mode="HTML",
                    )
            return
        else:
            err_msg = res.get("response_message") or "Transaction not found"
            await query.answer(f"⚠️ មិនទាន់ឃើញប្រតិបត្តិការក្នុង Bakong ទេ: {err_msg}", show_alert=True)
            return

    # -------------------------------------------------------------------------
    # 5. Admin Approves Donation Manually (supports both donate_appr: and donate_approve:)
    # -------------------------------------------------------------------------
    if data.startswith("donate_appr:") or data.startswith("donate_approve:"):
        if not is_admin_user(user_id):
            await query.answer("⛔ មានតែ Admin ប៉ុណ្ណោះដែលអាចអនុម័តបាន!", show_alert=True)
            return

        donor_uid = 0
        amount = 1.0
        tier_key = "coffee"
        bill_no = ""
        ticket_id = ""

        if data.startswith("donate_appr:"):
            ticket_id = data[len("donate_appr:"):]
            with _PENDING_LOCK:
                info = _PENDING_DONATIONS.get(ticket_id)
            if info:
                donor_uid = int(info["user_id"])
                amount = float(info["amount"])
                tier_key = str(info["tier_key"])
                bill_no = str(info["bill_no"])
            else:
                await query.answer("⚠️ ព័ត៌មានសំណើនេះផុតកំណត់ ឬត្រូវបានអនុម័តរួចហើយ!", show_alert=True)
                return
        else:
            payload_part = data[len("donate_approve:"):]
            parts = payload_part.split(":")
            donor_uid = int(parts[0])
            amount = float(parts[1])
            tier_key = parts[2] if len(parts) > 2 else "coffee"
            bill_no = parts[3] if len(parts) > 3 else f"{donor_uid}_{amount}"

        await query.answer("កំពុងអនុម័ត និងបង្កើតសំឡេងជូនពរ...")

        ok, status_txt = await _execute_donation_approval(
            donor_uid=donor_uid,
            amount=amount,
            tier_key=tier_key,
            bill_no=bill_no,
            ticket_id=ticket_id,
            context=context,
            approved_by=f"Admin {user_id} (Manual)",
        )

        if not ok and status_txt == "ALREADY_PROCESSED":
            await query.answer("⚠️ ការឧបត្ថម្ភនេះត្រូវបានអនុម័តរួចរាល់ហើយ!", show_alert=True)
            if query.message:
                orig_text = html.escape(query.message.text or "")
                with suppress(Exception):
                    await query.message.edit_text(
                        f"{orig_text}\n\n"
                        f"ℹ️ <b>ការឧបត្ថម្ភនេះត្រូវបានអនុម័តរួចរាល់ជាស្ថាពរហើយ។</b>",
                        parse_mode="HTML",
                    )
            return

        if query.message:
            orig_text = html.escape(query.message.text or "")
            with suppress(Exception):
                await query.message.edit_text(
                    f"{orig_text}\n\n"
                    f"✅ <b>បានអនុម័តជោគជ័យដោយ Admin!</b>\n"
                    f"{status_txt}\n"
                    f"🏆 បានបញ្ចូលក្នុងតារាងកិត្តិយស (/donors)!",
                    parse_mode="HTML",
                )
        return

    # -------------------------------------------------------------------------
    # 6. Admin Rejects Donation Notification (supports donate_rej: & donate_reject:)
    # -------------------------------------------------------------------------
    if data.startswith("donate_rej:") or data.startswith("donate_reject:"):
        if not is_admin_user(user_id):
            await query.answer("⛔ មានតែ Admin ប៉ុណ្ណោះដែលអាចបដិសេធបាន!", show_alert=True)
            return

        if data.startswith("donate_rej:"):
            rej_ticket_id = data[len("donate_rej:"):]
            with _PENDING_LOCK:
                _PENDING_DONATIONS.pop(rej_ticket_id, None)
            _save_pending_tickets()

        await query.answer("បានបដិសេធការជូនដំណឹងនេះ។")
        if query.message:
            orig_text = html.escape(query.message.text or "")
            with suppress(Exception):
                await query.message.edit_text(
                    f"{orig_text}\n\n"
                    f"❌ <b>ការជូនដំណឹងនេះត្រូវបានបដិសេធដោយ Admin។</b>",
                    parse_mode="HTML",
                )
        return
