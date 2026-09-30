"""Callback handlers and action dispatchers for the Full Option Admin Bot Controller."""

from __future__ import annotations

import asyncio
import datetime
import html
import io
import logging
import os
from contextlib import suppress
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from app.services.admin.dashboard import (
    build_admin_bakong_text,
    build_admin_bot_mode_text,
    build_admin_home_full_text,
    build_admin_live_logs_text,
    build_admin_podcast_text,
    build_admin_quick_actions_text,
    build_admin_ui_hub_text,
    build_admin_web_news_text,
    get_admin_bakong_kb,
    get_admin_bot_mode_kb,
    get_admin_dashboard_full_kb,
    get_admin_live_logs_kb,
    get_admin_podcast_kb,
    get_admin_quick_actions_kb,
    get_admin_ui_hub_kb,
    get_admin_web_news_kb,
)
from app.services.telegram._legacy_runtime import safe_send

logger = logging.getLogger(__name__)


def _is_authorized_admin(user_id: int) -> bool:
    """Verify administrator identity against legacy and modern settings."""
    if not user_id:
        return False

    with suppress(Exception):
        from app import legacy

        is_admin_fn = getattr(legacy, "_is_admin", getattr(legacy, "is_admin", None))
        if callable(is_admin_fn) and is_admin_fn(user_id):
            return True

    # Fallback: check environment and core settings
    admin_ids: set[int] = set()
    with suppress(Exception):
        from app.core.config import SETTINGS

        for src in (getattr(SETTINGS, "ADMIN_IDS", None),):
            if isinstance(src, (set, list, tuple)):
                admin_ids.update(int(a) for a in src if str(a).lstrip("-").isdigit())
            elif isinstance(src, str):
                admin_ids.update(int(a.strip()) for a in src.split(",") if a.strip().lstrip("-").isdigit())

    for env_a in os.environ.get("ADMIN_IDS", "").split(","):
        if env_a.strip().lstrip("-").isdigit():
            admin_ids.add(int(env_a.strip()))

    return user_id in admin_ids


async def _resolve_text(target: Any, *args: Any, **kwargs: Any) -> str:
    """Safely resolve dynamic dashboard text builders regardless of sync or async nature."""
    if callable(target):
        val = target(*args, **kwargs)
    else:
        val = target
    if asyncio.iscoroutine(val):
        return await val
    return str(val)


async def handle_admin_callback(
    query: Any,
    user_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    data: str,
) -> bool:
    """Handle administrative sub-controller callbacks.

    Returns True if the callback was processed by this handler, False otherwise.
    """
    # Guard: Must be admin
    if not _is_authorized_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ សម្រាប់តែអ្នកគ្រប់គ្រងប៉ុណ្ណោះ (Admin only)។", show_alert=True)
        if getattr(query, "message", None) and hasattr(query.message, "reply_text"):
            with suppress(Exception):
                await query.message.reply_text("⛔ សម្រាប់តែអ្នកគ្រប់គ្រងប៉ុណ្ណោះ (Admin only)។", parse_mode="HTML")
        return True

    # ── 0. ADMIN HOME & MAIN DASHBOARD ─────────────────────────────────────
    if data in ("admin_home", "admin_dashboard", "admin_dashboard_full"):
        with suppress(Exception):
            await query.answer()

        text = await _resolve_text(build_admin_home_full_text, user_id=user_id)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_full_kb(),
            disable_web_page_preview=True,
        ))
        return True

    # ── 1. PODCAST CONTROLLER ──────────────────────────────────────────────
    if data == "admin_podcast":
        with suppress(Exception):
            await query.answer()

        text = await _resolve_text(build_admin_podcast_text)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_podcast_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_podcast_test":
        with suppress(Exception):
            await query.answer("⏳ កំពុងរៀបចំ និងផ្ញើតេស្ត...")

        from app.services.podcast.generator import generate_morning_podcast
        from app.services.podcast.handlers import get_podcast_kb, send_podcast_card, send_podcast_voice

        html_text, speech_text = await generate_morning_podcast()
        kb = get_podcast_kb(True, getattr(context.bot, "username", ""))

        await send_podcast_card(
            None,
            f"🧪 <b>[Admin Test Preview]</b>\n\n{html_text}",
            reply_markup=kb,
            bot=context.bot,
            chat_id=user_id,
        )
        try:
            await send_podcast_voice(
                None,
                speech_text,
                bot=context.bot,
                chat_id=user_id,
            )
        except Exception as exc:
            logger.warning("Admin podcast test voice delivery failed: %s", exc)

        with suppress(Exception):
            await query.answer("✅ បានផ្ញើតេស្ត Morning Podcast រួចរាល់!", show_alert=True)
        return True

    if data == "admin_podcast_refresh":
        with suppress(Exception):
            await query.answer("🔄 កំពុងទាញយក និងបង្កើតព័ត៌មានថ្មី...")

        from app.services.podcast.generator import generate_morning_podcast
        from app.services.podcast.handlers import (
            fetch_source_banner_bytes,
            get_or_synthesize_podcast_voice,
        )

        with suppress(Exception):
            await asyncio.wait_for(fetch_source_banner_bytes(), timeout=3.0)
        _, speech_text = await generate_morning_podcast(force_refresh=True)
        with suppress(Exception):
            await get_or_synthesize_podcast_voice(speech_text, force_refresh=True)

        with suppress(Exception):
            await query.answer("✅ បានទាញយក និងផ្ទុកព័ត៌មានថ្មីរួចរាល់!", show_alert=True)

        pod_text = await _resolve_text(build_admin_podcast_text)
        await safe_send(lambda: query.message.edit_text(
            f"🔄 <b>បាន Refresh និងបង្កើត Podcast ថ្មីជោគជ័យ!</b>\n\n{pod_text}",
            parse_mode="HTML",
            reply_markup=get_admin_podcast_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data in ("admin_podcast_voice_f", "admin_podcast_voice_m"):
        pres = "male" if data == "admin_podcast_voice_m" else "female"
        pres_label = "សំឡេងប្រុស" if pres == "male" else "សំឡេងស្រី"
        with suppress(Exception):
            await query.answer(f"🎙️ កំពុងផ្ញើ {pres_label}...")

        from app.services.podcast.generator import generate_morning_podcast
        from app.services.podcast.handlers import send_podcast_voice

        _, speech_text = await generate_morning_podcast()
        try:
            await send_podcast_voice(
                None,
                speech_text,
                presenter=pres,
                bot=context.bot,
                chat_id=user_id,
            )
        except Exception as exc:
            logger.warning("Admin podcast voice test (%s) failed: %s", pres, exc)
        return True

    if data == "admin_podcast_mp3":
        with suppress(Exception):
            await query.answer("🎵 កំពុងផ្ញើ MP3 Track...")

        from app.services.podcast.generator import generate_morning_podcast
        from app.services.podcast.handlers import send_podcast_mp3

        _, speech_text = await generate_morning_podcast()
        try:
            await send_podcast_mp3(
                None,
                speech_text,
                presenter="female",
                bot=context.bot,
                chat_id=user_id,
            )
        except Exception as exc:
            logger.warning("Admin podcast MP3 test failed: %s", exc)
        return True

    if data == "admin_podcast_broadcast":
        with suppress(Exception):
            await query.answer("🚀 កំពុងចាប់ផ្តើមផ្សាយ Podcast...", show_alert=False)

        from app.services.podcast.generator import generate_morning_podcast
        from app.services.podcast.handlers import (
            fetch_source_banner_bytes,
            get_or_synthesize_podcast_voice,
            get_podcast_kb,
            send_podcast_card,
            send_podcast_voice,
        )
        from app.services.podcast.store import podcast_store

        subscribers = podcast_store.get_all_subscribers()
        if not subscribers:
            with suppress(Exception):
                await query.answer("⚠️ មិនទាន់មាន Subscribers ណាម្នាក់នៅឡើយទេ!", show_alert=True)
            return True

        with suppress(Exception):
            await asyncio.wait_for(fetch_source_banner_bytes(), timeout=3.0)
        html_text, speech_text = await generate_morning_podcast(force_refresh=True)
        with suppress(Exception):
            await get_or_synthesize_podcast_voice(speech_text, force_refresh=True)

        success_count = 0
        fail_count = 0
        kb = get_podcast_kb(True, getattr(context.bot, "username", ""))
        dead_subscribers: set[int] = set()

        for chat_id in subscribers:
            try:
                await send_podcast_card(
                    None,
                    html_text,
                    reply_markup=kb,
                    bot=context.bot,
                    chat_id=chat_id,
                )
                await send_podcast_voice(
                    None,
                    speech_text,
                    bot=context.bot,
                    chat_id=chat_id,
                )
                success_count += 1
            except Exception as exc:
                fail_count += 1
                logger.warning("Podcast broadcast failed to chat_id=%s: %s", chat_id, exc)

                retry_after = getattr(exc, "retry_after", None)
                if retry_after:
                    await asyncio.sleep(float(retry_after) + 0.5)

                low_exc = str(exc).lower()
                if any(k in low_exc for k in ("blocked", "deactivated", "chat not found", "user is deactivated")):
                    dead_subscribers.add(chat_id)

            await asyncio.sleep(0.04)

        if dead_subscribers:
            removed = podcast_store.unsubscribe_batch(dead_subscribers)
            logger.info("Auto-unsubscribed %d dead/blocked podcast subscribers", removed)

        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        podcast_store.set_last_broadcast_date(today_str)

        pod_text = await _resolve_text(build_admin_podcast_text)
        notice = f"✅ <b>ផ្សាយជោគជ័យទៅកាន់ {success_count} នាក់!</b> (បរាជ័យ {fail_count})\n\n"
        await safe_send(lambda: query.message.edit_text(
            notice + pod_text,
            parse_mode="HTML",
            reply_markup=get_admin_podcast_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_podcast_subs":
        from app.services.podcast.store import podcast_store

        subscribers = podcast_store.get_all_subscribers()
        if not subscribers:
            sub_list_str = "<i>មិនទាន់មានអ្នកជាវ (No subscribers)</i>"
        else:
            sub_list_str = "\n".join(f"• <code>{cid}</code>" for cid in subscribers[:50])
            if len(subscribers) > 50:
                sub_list_str += f"\n<i>...និង {len(subscribers) - 50} នាក់ទៀត</i>"

        text = (
            f"👥 <b>បញ្ជីអ្នកជាវ Daily Morning Podcast ({len(subscribers)} នាក់)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{sub_list_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Podcast Menu", callback_data="admin_podcast"),
                InlineKeyboardButton("🏠 Admin Home", callback_data="admin_home"),
            ]]),
            disable_web_page_preview=True,
        ))
        return True

    # ── 2. BAKONG & KHQR CONTROLLER ─────────────────────────────────────────
    if data == "admin_bakong":
        with suppress(Exception):
            await query.answer()

        text = await _resolve_text(build_admin_bakong_text)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_bakong_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_bakong_ping":
        with suppress(Exception):
            await query.answer("⏳ កំពុងតេស្តការតភ្ជាប់...")

        from app.services.donation import bakong_api

        res = await bakong_api.test_connection()
        if res.get("ok"):
            status_text = f"🟢 <b>ភ្ជាប់ជោគជ័យ (Connected):</b> Latency <b>{res.get('latency_ms', 0)}ms</b>"
        else:
            status_text = f"🔴 <b>បរាជ័យ (Failed):</b> {html.escape(str(res.get('error') or res.get('message')))}"

        base_text = await _resolve_text(build_admin_bakong_text)
        text = f"{status_text}\n\n{base_text}"
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_bakong_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_bakong_preview_qr":
        with suppress(Exception):
            await query.answer("⏳ កំពុងបង្កើតរូបភាព QR...")

        from app.services.donation.khqr import (
            DEFAULT_BAKONG_ACCOUNT_ID,
            DEFAULT_BAKONG_MERCHANT_NAME,
            generate_khqr_payload,
            get_khqr_config,
            get_khqr_qr_image,
        )

        cfg = get_khqr_config()
        acc_id = cfg.get("account_id") or DEFAULT_BAKONG_ACCOUNT_ID
        m_name = cfg.get("merchant_name") or DEFAULT_BAKONG_MERCHANT_NAME

        khqr_data = generate_khqr_payload(1.00, "USD", bill_number="TEST0001")
        qr_bytes = await get_khqr_qr_image(khqr_data.get("qr_string", ""))

        if qr_bytes:
            bio = io.BytesIO(qr_bytes)
            bio.name = "bakong_preview.png"
            bio.seek(0)
            await safe_send(lambda: query.message.reply_photo(
                photo=bio,
                caption=(
                    f"💳 <b>Bakong KHQR Preview (Test $1.00)</b>\n\n"
                    f"• <b>Merchant:</b> <code>{html.escape(m_name)}</code>\n"
                    f"• <b>Account:</b> <code>{html.escape(acc_id)}</code>\n"
                    f"• <b>Payload MD5:</b> <code>{khqr_data.get('md5', 'N/A')}</code>\n\n"
                    f"💡 <i>ស្កេនដើម្បីតេស្តការបង្កើត KHQR ជាក់ស្តែង។</i>"
                ),
                parse_mode="HTML",
            ))
            with suppress(Exception):
                await query.answer("✅ បានផ្ញើរូបភាព QR Code Preview រួចរាល់!", show_alert=False)
        else:
            with suppress(Exception):
                await query.answer("❌ មិនអាចបង្កើត QR image បានទេ!", show_alert=True)
        return True

    if data == "admin_bakong_pending":
        pending_items: list[tuple[str, dict[str, Any]]] = []
        with suppress(Exception):
            from app.services.donation.handlers import _PENDING_DONATIONS, _PENDING_LOCK

            with _PENDING_LOCK:
                pending_items = list(_PENDING_DONATIONS.items())

        if not pending_items:
            text = (
                "⏳ <b>សំណើកំពុងរង់ចាំពិនិត្យ (Pending Donations)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>គ្មានសំណើកំពុងរង់ចាំពិនិត្យទេ (No pending tickets)។</i>\n"
                "━━━━━━━━━━━━━━━━━━━━━━"
            )
        else:
            lines = [
                f"⏳ <b>សំណើកំពុងរង់ចាំពិនិត្យ ({len(pending_items)}):</b>",
                "━━━━━━━━━━━━━━━━━━━━━━",
            ]
            for tid, tinfo in pending_items[:10]:
                amt = tinfo.get("amount", 0)
                curr = tinfo.get("currency", "USD")
                uid = tinfo.get("user_id", 0)
                bill = tinfo.get("bill_number", "N/A")
                lines.append(f"• Ticket: <code>{tid}</code> | <b>{amt} {curr}</b> | User: <code>{uid}</code> | Bill: <code>{bill}</code>")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━\n💡 ប្រើ <code>/checkpay &lt;Ticket_ID&gt;</code> ដើម្បីពិនិត្យ។")
            text = "\n".join(lines)

        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Bakong Menu", callback_data="admin_bakong"),
                InlineKeyboardButton("🏠 Admin Home", callback_data="admin_home"),
            ]]),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_bakong_donors":
        from app.services.donation.store import donation_store

        donors = donation_store.get_hall_of_fame(limit=10)
        if not donors:
            donor_text = "<i>មិនទាន់មានទិន្នន័យអ្នកឧបត្ថម្ភនៅឡើយទេ</i>"
        else:
            donor_text = "\n".join(
                f"{idx+1}. <b>{html.escape(d.get('name', 'Anonymous'))}</b> — <b>${d.get('total_usd', 0):.2f}</b>"
                for idx, d in enumerate(donors)
            )

        text = (
            "🏆 <b>តារាងអ្នកឧបត្ថម្ភកំពូល (Hall of Fame)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{donor_text}\n"
            "━━━━━━━━━━━━━━━━━━━━━━"
        )
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Bakong Menu", callback_data="admin_bakong"),
                InlineKeyboardButton("🏠 Admin Home", callback_data="admin_home"),
            ]]),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_bakong_config":
        from app.services.donation.khqr import (
            DEFAULT_BAKONG_ACCOUNT_ID,
            DEFAULT_BAKONG_MERCHANT_NAME,
            get_khqr_config,
        )

        cfg = get_khqr_config()
        acc_id = cfg.get("account_id") or DEFAULT_BAKONG_ACCOUNT_ID
        m_name = cfg.get("merchant_name") or DEFAULT_BAKONG_MERCHANT_NAME

        text = (
            "⚙️ <b>ការកំណត់ Bakong KHQR Gateway</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Merchant Name:</b> <code>{html.escape(m_name)}</code>\n"
            f"• <b>Account ID:</b> <code>{html.escape(acc_id)}</code>\n"
            f"• <b>Static Fallback File:</b> <code>assets/my_khqr.webp</code>\n\n"
            "💡 <b>របៀបផ្លាស់ប្តូរ៖</b>\n"
            "• ប្រើប្រាស់ Environment Variables:\n"
            "  <code>BAKONG_ACCOUNT_ID</code>\n"
            "  <code>BAKONG_MERCHANT_NAME</code>\n"
            "  <code>BAKONG_OPEN_API_TOKEN</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━"
        )
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Bakong Menu", callback_data="admin_bakong"),
                InlineKeyboardButton("🏠 Admin Home", callback_data="admin_home"),
            ]]),
            disable_web_page_preview=True,
        ))
        return True

    # ── 3. BOT INTERACTION MODE CONTROLLER ─────────────────────────────────
    if data == "admin_bot_mode":
        with suppress(Exception):
            await query.answer()

        from app import legacy

        settings, _ = await legacy.get_bot_settings_async()
        curr = legacy._setting_raw_from(settings, "DEFAULT_BOT_MODE", "auto")
        text = build_admin_bot_mode_text(curr)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_bot_mode_kb(curr),
            disable_web_page_preview=True,
        ))
        return True

    if data.startswith("admin_mode_set:"):
        new_mode = data.split(":", 1)[1].strip().lower()
        if new_mode in ("auto", "tts", "ai_chat"):
            from app import legacy

            executor = getattr(legacy, "_DB_EXECUTOR", None)
            set_fn = getattr(legacy, "db_bot_setting_value_set", None)

            if callable(set_fn):
                ok, info = await asyncio.get_running_loop().run_in_executor(
                    executor,
                    lambda: set_fn("DEFAULT_BOT_MODE", new_mode, user_id),
                )
            else:
                from app.services.settings.store import get_settings_store

                await get_settings_store().set_text("DEFAULT_BOT_MODE", new_mode)
                ok, info = True, ""

            notice = f"✅ បានកំណត់ Bot Mode លំនាំដើមទៅ៖ <b>{new_mode.upper()}</b>" if ok else f"⚠️ កំហុស: {info}"
            text = f"{notice}\n\n" + build_admin_bot_mode_text(new_mode)
            await safe_send(lambda: query.message.edit_text(
                text,
                parse_mode="HTML",
                reply_markup=get_admin_bot_mode_kb(new_mode),
                disable_web_page_preview=True,
            ))
            with suppress(Exception):
                await query.answer(f"✅ Bot Mode: {new_mode.upper()}")
        return True

    # ── 4. UI & CUSTOMIZATION HUB ──────────────────────────────────────────
    if data == "admin_ui_hub":
        with suppress(Exception):
            await query.answer()

        text = await _resolve_text(build_admin_ui_hub_text)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_ui_hub_kb(),
            disable_web_page_preview=True,
        ))
        return True

    # ── 5. QUICK ACTIONS & EMERGENCY HUB ───────────────────────────────────
    if data == "admin_quick_actions":
        with suppress(Exception):
            await query.answer()

        text = await _resolve_text(build_admin_quick_actions_text)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_quick_actions_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_quick_optimize":
        with suppress(Exception):
            await query.answer("⏳ កំពុង Optimize & សម្អាតធនធានប្រព័ន្ធ...")

        from app.services.admin.optimization import run_system_optimization_async

        stats = await run_system_optimization_async(admin_id=user_id, prune_db=True)
        applied = stats.get("perf_knobs_applied", [])
        swept = stats.get("temp_files_swept", 0)
        gc_freed = stats.get("gc_objects_freed", 0)
        trimmed = stats.get("audio_cache_trimmed", 0)
        hist_pruned = stats.get("db_history_pruned", 0)

        with suppress(Exception):
            await query.answer(
                f"✅ Optimized! Swept {swept} files, freed {gc_freed} GC objects, {len(applied)} knobs verified.",
                show_alert=True,
            )

        base_qa = await _resolve_text(build_admin_quick_actions_text)
        text = (
            "✅ <b>បាន Optimize ប្រព័ន្ធ និងសម្អាតធនធានជោគជ័យ!</b>\n\n"
            f"• 🧹 <b>Temp Files:</b> បានលុប <code>{swept}</code> ឯកសារបណ្ដោះអាសន្ន\n"
            f"• 🧠 <b>Garbage Collection:</b> បានរំដោះ <code>{gc_freed:,}</code> objects ក្នុង RAM\n"
            f"• 💾 <b>Audio Cache:</b> បាន Trim <code>{trimmed}</code> ឯកសារហួសកំណត់\n"
            f"• 🗄️ <b>Database Pruning:</b> សម្អាត <code>{hist_pruned}</code> កំណត់ត្រាចាស់ៗ\n"
            f"• ⚙️ <b>Performance Knobs:</b> <code>{len(applied)}</code> settings ត្រួតពិនិត្យរួចរាល់\n\n"
            + base_qa
        )
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_quick_actions_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_quick_flush_cache":
        with suppress(Exception):
            await query.answer("⏳ កំពុងសម្អាត Audio, Button & AI Response Cache...")

        import gc
        res: dict[str, Any] = {}
        gemini_cleared = 0

        with suppress(Exception):
            from app.services.tts.cache import clear_all_tts_caches
            res = clear_all_tts_caches()

        with suppress(Exception):
            from app.services.ai.gemini import clear_gemini_response_cache
            gemini_cleared = clear_gemini_response_cache()

        with suppress(Exception):
            from app.services.telegram.buttons import clear_button_labels_cache
            clear_button_labels_cache()

        gc.collect()

        notice = (
            f"✅ <b>បានសម្អាត Caches រួចរាល់!</b>\n"
            f"• Audio Mem: {res.get('audio_items_cleared', 0)} | CDN IDs: {res.get('file_ids_cleared', 0)}\n"
            f"• AI Responses: {gemini_cleared} items | Buttons & GC Memory Compacted\n\n"
        )
        base_qa = await _resolve_text(build_admin_quick_actions_text)
        await safe_send(lambda: query.message.edit_text(
            notice + base_qa,
            parse_mode="HTML",
            reply_markup=get_admin_quick_actions_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_quick_clear_errors":
        with suppress(Exception):
            await query.answer("⏳ កំពុងសម្អាត Error logs...")

        from app import legacy

        clear_err_fn = getattr(legacy, "_admin_error_center_clear", None)
        cleared = clear_err_fn() if callable(clear_err_fn) else 0

        notice = f"✅ <b>បានសម្អាត Error Inbox ({cleared} errors removed)!</b>\n\n"
        base_qa = await _resolve_text(build_admin_quick_actions_text)
        await safe_send(lambda: query.message.edit_text(
            notice + base_qa,
            parse_mode="HTML",
            reply_markup=get_admin_quick_actions_kb(),
            disable_web_page_preview=True,
        ))
        return True

    # ── 6. REAL-TIME SERVER REQUEST LOGS VIEWER ───────────────────────────
    if data in ("admin_live_logs", "admin_live_logs:errors", "admin_live_logs:refresh"):
        only_errors = (data == "admin_live_logs:errors")
        with suppress(Exception):
            await query.answer("🔄 កំពុងទាញយក Live Logs...")

        from app.core.config import get_detected_webhook_url

        base_url = get_detected_webhook_url()
        dashboard_url = f"{base_url.rstrip('/')}/logs" if base_url else None
        text = await _resolve_text(build_admin_live_logs_text, only_errors=only_errors)
        kb = get_admin_live_logs_kb(dashboard_url=dashboard_url, only_errors=only_errors)

        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=kb,
            disable_web_page_preview=True,
        ))
        return True

    # ── 7. WEB NEWS SCANNER & MONITORED SOURCES ───────────────────────────
    if data == "admin_web_news":
        with suppress(Exception):
            await query.answer()

        text = await _resolve_text(build_admin_web_news_text)
        await safe_send(lambda: query.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_web_news_kb(),
            disable_web_page_preview=True,
        ))
        return True

    if data == "admin_web_news_scan":
        with suppress(Exception):
            await query.answer("🔄 កំពុងស្កេន... (Scanning)")

        await safe_send(lambda: query.message.reply_text(
            "🔄 កំពុងស្វែងរកព័ត៌មានពីប្រភពទាំងអស់... (Scanning sources and notifying admins)"
        ))
        from app.services.ai.article_monitor import scan_sources_and_notify_admin

        discovered = await scan_sources_and_notify_admin(bot=context.bot)
        if discovered:
            await safe_send(lambda: query.message.reply_text(
                f"✅ <b>ស្វែងរកជោគជ័យ!</b>\n\n🎉 បានរកឃើញ និងរង់ចាំការយល់ព្រម <b>{len(discovered)}</b> ព័ត៌មាន! (Sent {len(discovered)} articles to admin queue)",
                parse_mode="HTML",
            ))
        else:
            await safe_send(lambda: query.message.reply_text(
                "✅ <b>ស្វែងរកជោគជ័យ!</b>\n\n🤷‍♂️ មិនមានព័ត៌មានថ្មីទេ (No new articles found)",
                parse_mode="HTML",
            ))
        return True

    if data == "admin_web_news_sources":
        with suppress(Exception):
            await query.answer()

        from app.services.ai.article_storage import get_article_sources

        sources = await get_article_sources()
        back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ (Back)", callback_data="admin_web_news")]])

        if not sources:
            await safe_send(lambda: query.message.edit_text(
                "📋 <b>ប្រភពព័ត៌មាន:</b>\n\n<i>មិនមានប្រភពព័ត៌មានត្រូវបានកំណត់រចនាសម្ព័ន្ធទេ (No sources configured)។</i>",
                parse_mode="HTML",
                reply_markup=back_kb,
            ))
            return True

        pages = ["📋 <b>ប្រភពព័ត៌មានទាំងអស់:</b>\n"]
        for idx, src in enumerate(sources, 1):
            name = src.get("name") or "Unnamed"
            url = src.get("url") or ""
            active = "✅" if src.get("is_active") else "❌"
            line = f"{idx}. {active} <b>{name}</b>\n   └ {url}\n"
            if len(pages[-1]) + len(line) > 3500:
                pages.append("")
            pages[-1] += line

        first_page = pages[0]
        await safe_send(lambda: query.message.edit_text(
            first_page,
            parse_mode="HTML",
            reply_markup=back_kb if len(pages) == 1 else None,
            disable_web_page_preview=True,
        ))
        for i, page in enumerate(pages[1:], 2):
            footer_kb = back_kb if i == len(pages) else None
            await safe_send(lambda: query.message.reply_text(
                page,
                parse_mode="HTML",
                reply_markup=footer_kb,
                disable_web_page_preview=True,
            ))
        return True

    return False


__all__ = ["handle_admin_callback"]