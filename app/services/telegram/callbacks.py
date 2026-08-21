"""Extracted Telegram handler implementations.

These are live runtime handlers; app.legacy now contains compatibility wrappers only.
"""

from __future__ import annotations

# Transitional V4.1 modules bind remaining legacy helpers at runtime.
# ruff: noqa: F821
from app.services.telegram._legacy_runtime import legacy_bound_handler


@legacy_bound_handler
async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    user_id = query.from_user.id
    data    = query.data or ""

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    with suppress(Exception):
        await query.answer()

    if data == "bc_templates":
        await _admin_open_broadcast_templates(query, context, user_id)
        return

    if data == "bc_save_template":
        pending = _pending_broadcast.get(user_id)
        if not pending:
            await safe_send(lambda: query.message.reply_text(
                '⚠️ មិនមានសារមើលជាមុនសម្រាប់រក្សាទុកទេ។ សូមបង្កើតការផ្សាយសារជាមុនសិន។'
            ))
            return
        ok, info, tpl = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: db_broadcast_template_save(pending, user_id),
        )
        if ok and str(info).startswith("updated existing"):
            notice = "♻️ Template មានរួចហើយ — បាន Update និងដាក់ឡើងលើ។"
        else:
            notice = "✅ បាន Save Template។" if ok else f"⚠️ Save Template មិនជោគជ័យ: {info}"
        await _admin_open_broadcast_templates(query, context, user_id, notice=notice)
        return

    if data.startswith("bc_tpl_use:"):
        tpl_id = data.split(":", 1)[1]
        tpl = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_broadcast_template_get(tpl_id))
        if not tpl:
            await _admin_open_broadcast_templates(query, context, user_id, notice="⚠️ រក Template មិនឃើញ។")
            return
        pending = _broadcast_template_payload_from_template(tpl)
        _pending_broadcast[user_id] = pending
        context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f'📚 បានជ្រើសគំរូ៖ <b>{html.escape(_broadcast_template_button_title(tpl))}</b>',
            parse_mode="HTML",
        ))
        ok = await _admin_show_broadcast_preview_message(query.message, context.bot, user_id, pending)
        if not ok:
            _pending_broadcast.pop(user_id, None)
            context.user_data.pop("bc_state", None)
        return

    if data.startswith("bc_tpl_delask:"):
        tpl_id = data.split(":", 1)[1]
        tpl = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_broadcast_template_get(tpl_id))
        if not tpl:
            await _admin_open_broadcast_templates(query, context, user_id, notice="⚠️ រក Template មិនឃើញ។")
            return
        await safe_send(lambda: query.message.edit_text(
            _broadcast_template_delete_confirm_text(tpl),
            parse_mode="HTML",
            reply_markup=get_broadcast_template_delete_confirm_kb(tpl_id),
            disable_web_page_preview=True,
        ))
        return

    if data.startswith("bc_tpl_del:"):
        tpl_id = data.split(":", 1)[1]
        ok, info = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: db_broadcast_template_delete(tpl_id, user_id),
        )
        notice = "🗑️ បានលុប Template។" if ok else f"⚠️ លុបមិនជោគជ័យ: {info}"
        await _admin_open_broadcast_templates(query, context, user_id, notice=notice)
        return

    if data.startswith("bc_del_sent_ask:"):
        job_id = data.split(":", 1)[1]
        job = _broadcast_sent_delete_get(job_id, user_id)
        await safe_send(lambda: query.message.reply_text(
            _broadcast_sent_delete_confirm_text(job),
            parse_mode="HTML",
            reply_markup=get_broadcast_sent_delete_confirm_kb(job_id) if job else None,
            disable_web_page_preview=True,
        ))
        return

    if data.startswith("bc_del_sent_run:"):
        job_id = data.split(":", 1)[1]
        job = _broadcast_sent_delete_get(job_id, user_id, pop=True)
        if not job:
            await safe_send(lambda: query.message.reply_text(
                '⚠️ ការងារលុបនេះមិនមានទៀតទេ។ វាអាចផុតកំណត់ ឬម៉ាស៊ីនមេបានចាប់ផ្ដើមឡើងវិញ។'
            ))
            return
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        context.application.create_task(_delete_broadcast_sent_messages(context.bot, user_id, job))
        await safe_send(lambda: query.message.reply_text("🗑️ បានចាប់ផ្ដើមលុបសារ Broadcast ដែលបានផ្ញើ..."))
        return

    if data == "bc_del_sent_keep":
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text('✅ បានរក្សាទុកសារផ្សាយនៅដដែល។'))
        return

    if data == "bc_cancel":
        _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text('❌ បានបោះបង់ការផ្សាយសារ។'))
        return

    if data == "bc_confirm":
        pending = _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        if not pending:
            await safe_send(lambda: query.message.reply_text("⚠️ រកទិន្នន័យ Broadcast មិនឃើញ។ សូមចាប់ផ្ដើមថ្មី។"))
            return
        context.application.create_task(
            _run_broadcast_to_all(context.bot, user_id, pending, label="ការផ្សាយសារ")
        )
        return

    await safe_send(lambda: query.message.reply_text(
        "This broadcast button is no longer available. Please reopen the menu."
    ))


@legacy_bound_handler
async def users_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return

    user_id = query.from_user.id if query.from_user else 0
    data = query.data or ""

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    if query.message is None:
        return

    def _int_part(parts: list[str], index: int, default: int = 0) -> int:
        try:
            return int(parts[index])
        except Exception:
            return int(default)

    async def _invalid_callback() -> None:
        await safe_send(lambda: query.message.reply_text('⚠️ ទិន្នន័យប៊ូតុងមិនត្រឹមត្រូវ ឬផុតកំណត់។ សូមផ្ទុកផ្ទាំងនេះឡើងវិញ។'))

    try:
        if data == "users_close":
            context.user_data.pop("user_search_state", None)
            with suppress(Exception):
                await query.message.delete()
            return

        if data == "noop":
            return

        if data in ("history_refresh", "history_page:0"):
            await _admin_open_recent_history_panel(query, page=0)
            return

        if data == "history_close":
            with suppress(Exception):
                await query.message.delete()
            return

        if data.startswith("history_page:"):
            page = _web_int(data.split(":", 1)[1], 0)
            await _admin_open_recent_history_panel(query, page=page)
            return

        if data.startswith("history_user:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            page = _int_part(parts, 2, 0)
            if target_id <= 0:
                await _invalid_callback()
                return
            row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=f"h{page}"),
            ))
            return

        if data == "users_search":
            context.user_data["user_search_state"] = USER_SEARCH_WAIT_QUERY
            await safe_send(lambda: query.message.edit_text(
                '🔎 <b>ស្វែងរកអ្នកប្រើប្រាស់</b>\n\nសូមផ្ញើលេខសម្គាល់អ្នកប្រើប្រាស់ Telegram ឬឈ្មោះអ្នកប្រើប្រាស់។\n\nឧទាហរណ៍៖\n<code>1272791365</code>\n<code>heng</code>\n<code>@username</code>\n\nប្រើ /cancel ដើម្បីបញ្ឈប់ការស្វែងរក។',
                parse_mode="HTML",
                reply_markup=get_user_search_prompt_kb(),
            ))
            return

        if data.startswith("users_search_page:"):
            page = _web_int(data.split(":", 1)[1], 0)
            await _show_user_search_results(query, context, page=page)
            return

        if data.startswith("users_page:"):
            page = _web_int(data.split(":", 1)[1], 0)
            users = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, get_all_users_with_names)
            page = _clamp_users_page(users, page)
            await safe_send(lambda: query.message.edit_text(
                f'👥 <b>ការគ្រប់គ្រងអ្នកប្រើប្រាស់ ({len(users)} នាក់)</b>\nសូមជ្រើសរើសអ្នកប្រើប្រាស់ ឬចុច 🔎 ស្វែងរកអ្នកប្រើប្រាស់ ដើម្បីស្វែងរកតាមលេខសម្គាល់ ឬឈ្មោះអ្នកប្រើប្រាស់។',
                parse_mode="HTML",
                reply_markup=get_users_page_kb(users, page=page),
            ))
            return

        if data.startswith("user_view:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
            ))
            return

        if data.startswith("user_history:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            page = _int_part(parts, 3, 0)
            if target_id <= 0:
                await _invalid_callback()
                return
            await _show_user_full_history(query, target_id, back_ref=back_ref, page=page)
            return

        if data.startswith("user_chat:"):
            target_id = _web_int(data.split(":", 1)[1], 0)
            if target_id <= 0:
                await _invalid_callback()
                return
            exists = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: user_exists_in_db(target_id))
            if not exists:
                await safe_send(lambda: query.message.edit_text(
                    f'❌ អ្នកប្រើប្រាស់ <code>{target_id}</code> មិនមាននៅក្នុងមូលដ្ឋានទិន្នន័យទេ។',
                    parse_mode="HTML",
                    reply_markup=get_admin_dashboard_kb(),
                ))
                return
            await _open_chat_session(context.bot, user_id, target_id, context)
            await safe_send(lambda: query.message.edit_text(
                f"💬 <b>Chat Mode បើក</b>\n\nកំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
                "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
                "វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("⬅️ Admin", callback_data="admin_home"),
                    InlineKeyboardButton("❌ End Chat", callback_data="admin_cancel_state"),
                ]]),
            ))
            return

        if data.startswith(("user_block:", "user_unblock:")):
            parts = data.split(":")
            action = parts[0]
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            blocked = action == "user_block"
            ok, info = await asyncio.get_running_loop().run_in_executor(
                _DB_EXECUTOR,
                lambda: db_user_set_blocked(target_id, user_id, blocked),
            )
            row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
            notice = "✅ User blocked." if blocked else "✅ User unblocked."
            if not ok:
                notice = f"⚠️ Saved memory only / DB issue: {info[:500]}"
            await safe_send(lambda: query.message.edit_text(
                notice + "\n\n" + _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
            ))
            return

        if data.startswith("user_resetprefs:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            ok, info = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_reset_prefs(target_id))
            row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
            notice = "✅ User preferences reset." if ok else f"❌ Reset failed: {info[:500]}"
            await safe_send(lambda: query.message.edit_text(
                notice + "\n\n" + _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
            ))
            return

        if data.startswith("user_clearhist:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_history_clear(target_id))
            row = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                "✅ User conversation history cleared.\n\n" + _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, bool(row.get("blocked")), back_ref=back_ref),
            ))
            return

        logger.debug("users_page_callback: unhandled data=%r", data)

    except Exception as exc:
        logger.error("users_page_callback failed [data=%s]: %s", data, exc, exc_info=True)
        with suppress(Exception):
            await safe_send(lambda: query.message.reply_text('⚠️ ផ្ទាំងអ្នកប្រើប្រាស់មានបញ្ហា។ សូមផ្ទុកឡើងវិញ ហើយព្យាយាមម្ដងទៀត។'))


@legacy_bound_handler
async def sched_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    user_id = query.from_user.id
    data = query.data or ""

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    with suppress(Exception):
        await query.answer()

    if query.message is None:
        return

    loop = asyncio.get_running_loop()

    if data.startswith(("sched_repeat_once:", "sched_repeat_daily:")):
        recurrence = (
            SCHED_RECURRENCE_DAILY
            if data.startswith("sched_repeat_daily:")
            else SCHED_RECURRENCE_ONCE
        )
        try:
            row_id = int(data.rsplit(":", 1)[1])
        except (TypeError, ValueError, IndexError):
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule ID."))
            return
        ok, reason, saved = await loop.run_in_executor(
            None,
            db_sched_update_recurrence,
            row_id,
            user_id,
            recurrence,
        )
        if not ok:
            await safe_send(lambda: query.message.reply_text(
                _sched_edit_error_text(row_id, reason),
                parse_mode="HTML",
            ))
            return
        await safe_send(lambda: query.message.reply_text(
            f"🔁 Schedule <b>#{row_id}</b> repeat changed to "
            f"<b>{html.escape(_sched_recurrence_label(_sched_row_recurrence(saved)))}</b>.",
            parse_mode="HTML",
            reply_markup=get_sched_detail_kb(saved or {}),
        ))
        return

    if data.startswith("sched_ok:"):
        row_id = _callback_int_arg(data, "sched_ok:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return

        ok, reason, row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_confirm, row_id, user_id)
        if not ok:
            if reason == "not_found":
                text = "❌ រកមិនឃើញ Schedule ។"
            elif reason == "not_owner":
                text = "⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"
            elif reason == "expired":
                text = f"⚠️ Schedule #{row_id} ផុតពេលមុនពេលបញ្ជាក់ ដូច្នេះបានបោះបង់។"
            else:
                text = f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{html.escape(str(reason))}</b> — មិនអាចបញ្ជាក់ទេ។"
            with suppress(Exception):
                await query.message.edit_reply_markup(reply_markup=None)
            await safe_send(lambda: query.message.reply_text(text, parse_mode="HTML"))
            return

        try:
            dt_str = _fmt_dt(datetime.fromisoformat(str(row["broadcast_at"]).replace("Z", "+00:00")))
        except Exception:
            dt_str = str(row.get("broadcast_at", "?")) if row else "?"
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        status_note = "បានបញ្ជាក់រួចហើយ" if reason == "already_confirmed" else "បានបញ្ជាក់"
        await safe_send(lambda: query.message.reply_text(
            f'✅ <b>កាលវិភាគ #{row_id} {status_note}!</b>\n'
            f'⏰ នឹងផ្សាយសារនៅ {dt_str}\n'
            f'🔁 Repeat: <b>{html.escape(_sched_recurrence_label(_sched_row_recurrence(row)))}</b>',
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_no:"):
        row_id = _callback_int_arg(data, "sched_no:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text('❌ រកកាលវិភាគមិនឃើញ។'))
            return
        if int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
            return
        if str(row.get("status")) in (SCHED_STATUS_DRAFT, SCHED_STATUS_PENDING):
            await loop.run_in_executor(_DB_EXECUTOR, db_sched_set_status, row_id, SCHED_STATUS_CANCELLED)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f"❌ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
        ))
        return

    if data == "sched_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "sched_noop":
        return

    if data.startswith("sched_page:"):
        page = _callback_int_arg(data, "sched_page:")
        if page is None:
            return
        rows = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_admin_pending, user_id)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=get_schedules_list_kb(rows, page=page))
        return

    if data.startswith("sched_view:"):
        row_id = _callback_int_arg(data, "sched_view:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text('❌ រកកាលវិភាគមិនឃើញ។'))
            return
        if int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
            return
        await safe_send(lambda: query.message.reply_text(
            _sched_detail_text(row),
            parse_mode="HTML",
            reply_markup=get_sched_detail_kb(row),
        ))
        return

    if data.startswith("sched_edit_time:"):
        row_id = _callback_int_arg(data, "sched_edit_time:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = SCHED_EDIT_WAIT_TIME
        context.user_data["sched_edit_row_id"] = row_id
        await safe_send(lambda: query.message.reply_text(
            f'✏️ <b>កែម៉ោងកាលវិភាគ #{row_id}</b>\n\nសូមផ្ញើពេលវេលាថ្មីតាមម៉ោងភ្នំពេញ (ICT, UTC+7)។\nទម្រង់៖ <code>YYYY-MM-DD HH:MM AM/PM</code> ឬ <code>YYYY-MM-DD HH:MM</code>\nតំបន់ម៉ោង៖ ភ្នំពេញ កម្ពុជា — ICT (UTC+7)\nឧទាហរណ៍៖ <code>2026-12-25 09:00 AM</code> ឬ <code>2026-12-25 21:00</code>\n\nវាយ /cancel ដើម្បីបោះបង់ការកែសម្រួល។',
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_edit_text:"):
        row_id = _callback_int_arg(data, "sched_edit_text:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = SCHED_EDIT_WAIT_TEXT
        context.user_data["sched_edit_row_id"] = row_id
        target = "caption" if row.get("photo_file_id") else "text"
        await safe_send(lambda: query.message.reply_text(
            f"📝 <b>Edit Schedule #{row_id} {target}</b>\n\n"
            "ផ្ញើអត្ថបទថ្មី។ វាយ /cancel ដើម្បីបោះបង់ edit។",
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_edit_photo:"):
        row_id = _callback_int_arg(data, "sched_edit_photo:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = SCHED_EDIT_WAIT_PHOTO
        context.user_data["sched_edit_row_id"] = row_id
        await safe_send(lambda: query.message.reply_text(
            f'🖼 <b>ប្ដូររូបភាពកាលវិភាគ #{row_id}</b>\n\nសូមផ្ញើរូបភាពថ្មី + ចំណងជើង (មិនចាំបាច់)។ វាយ /cancel ដើម្បីបោះបង់ការកែសម្រួល។',
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_cancel_confirm:"):
        row_id = _callback_int_arg(data, "sched_cancel_confirm:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_one, row_id)
        if not row or int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text('⛔ អ្នកមិនមានសិទ្ធិបោះបង់កាលវិភាគនេះទេ។'))
            return
        if row.get("status") not in (SCHED_STATUS_DRAFT, SCHED_STATUS_PENDING):
            st = html.escape(str(row.get("status") or "?"))
            await safe_send(lambda: query.message.reply_text(
                f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
                parse_mode="HTML",
            ))
            return
        await loop.run_in_executor(_DB_EXECUTOR, db_sched_set_status, row_id, SCHED_STATUS_CANCELLED)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f'✅ កាលវិភាគ <b>#{row_id}</b> បានបោះបង់។', parse_mode="HTML"
        ))
        return

    await safe_send(lambda: query.message.reply_text(
        "This schedule button is no longer available. Please reopen the menu."
    ))


@legacy_bound_handler
async def _runtime_admin_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    if query is None:
        return
    admin_id = query.from_user.id if query.from_user else 0
    data = query.data or ""
    if not _is_admin(admin_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    if query.message is None:
        with suppress(Exception):
            await query.answer()
        return

    if data == "rtadmin_close":
        with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS.pop(admin_id, None)
        with suppress(Exception):
            await query.answer('បានបិទ')
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "rtadmin_rate":
        with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS[admin_id] = {"state": "awaiting_rate_limit", "ts": time.monotonic()}
        with suppress(Exception):
            await query.answer('សូមផ្ញើជាលេខ', show_alert=False)
        await safe_send(lambda: query.message.edit_text(
            "⚡ <b>កែប្រែ Rate Limit</b>\n\n"
            f"តម្លៃបច្ចុប្បន្ន: <b>{_run_state_user_rate_limit()} req/{_run_state_user_rate_window():g}s</b>\n\n"
            "សូមផ្ញើលេខគត់ថ្មី ឧទាហរណ៍ <code>3</code> ឬ <code>5</code>។\n"
            "បើចង់កែ HTTP pool សូមផ្ញើ <code>http 120</code>។",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="rtadmin_cancel")]]),
        ))
        return

    if data == "rtadmin_cancel":
        with ACTIVE_ADMIN_CONVERSATIONS_LOCK:
            ACTIVE_ADMIN_CONVERSATIONS.pop(admin_id, None)
        with suppress(Exception):
            await query.answer('បានបោះបង់')
        await safe_send(lambda: query.message.edit_text(
            _runtime_admin_text(),
            parse_mode="HTML",
            reply_markup=_refresh_runtime_admin_markup(),
        ))
        return

    if data == "rtadmin_rotate_secret":
        with suppress(Exception):
            await query.answer('កំពុងប្ដូរសោសម្ងាត់…', show_alert=False)

        async with _webhook_rotate_lock():
            remaining = _webhook_rotate_begin_or_remaining()
            if remaining < 0:
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text()
                    + "\n\n⚠️ Webhook secret rotation is already running. Please wait.",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
                return
            if remaining > 0:
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text()
                    + f"\n\n⚠️ Please wait {int(remaining) + 1}s before rotating the webhook secret again.",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
                return

            success = False
            new_token = generate_new_webhook_token()
            try:
                # Important order: ask Telegram to accept the new webhook first.
                # Only persist RUN_STATE after setWebhook succeeds.
                if _run_state_bot_mode() == "WEBHOOK":
                    await _configure_telegram_webhook_via_http_for_secret(new_token)

                await _update_run_state("TELEGRAM_WEBHOOK_SECRET_TOKEN", new_token, persist=True)
                success = True

                logger.info("Admin %s rotated Webhook Secret Token.", admin_id)
                webhook_logger.info("Webhook secret rotated by admin_id=%s mode=%s", admin_id, _run_state_bot_mode())

                new_path = f"/tg-webhook-{new_token}"
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text()
                    + "\n\n✅ Webhook secret updated!"
                    + f"\nNew URL path: <code>{html.escape(new_path)}</code>",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
            except Exception as exc:
                webhook_logger.error("Webhook secret rotation failed admin_id=%s: %s", admin_id, exc, exc_info=True)
                error_text = html.escape(str(exc)[:800])
                await safe_send(lambda: query.message.edit_text(
                    _runtime_admin_text() + f"\n\n❌ Rotate secret failed: <code>{error_text}</code>",
                    parse_mode="HTML",
                    reply_markup=_refresh_runtime_admin_markup(),
                    disable_web_page_preview=True,
                ))
            finally:
                _webhook_rotate_finish(success)
        return

    if data.startswith("rtadmin_switch:"):
        target = data.split(":", 1)[1].strip().upper()
        with suppress(Exception):
            await query.answer('កំពុងប្ដូរ…', show_alert=False)
        try:
            mode = await _switch_telegram_runtime_mode(target, admin_id=admin_id)
            await safe_send(lambda: query.message.edit_text(
                _runtime_admin_text() + f"\n\n✅ បានប្ដូរទៅ <b>{html.escape(mode)}</b> រួចរាល់។",
                parse_mode="HTML",
                reply_markup=_refresh_runtime_admin_markup(),
                disable_web_page_preview=True,
            ))
        except Exception as exc:
            webhook_logger.error("Runtime mode switch failed admin_id=%s target=%s: %s", admin_id, target, exc, exc_info=True)
            error_text = html.escape(str(exc)[:800])
            await safe_send(lambda: query.message.edit_text(
                _runtime_admin_text() + f"\n\n❌ ប្ដូរ Mode មិនបាន: <code>{error_text}</code>",
                parse_mode="HTML",
                reply_markup=_refresh_runtime_admin_markup(),
                disable_web_page_preview=True,
            ))
        return

    with suppress(Exception):
        await query.answer(
            "This runtime button is no longer available. Please reopen the menu.",
            show_alert=False,
        )


@legacy_bound_handler
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query is None:
        return

    user_id = query.from_user.id
    data    = (query.data or "").strip()

    if not data:
        with suppress(Exception):
            await query.answer()
        return

    if query.message is None:
        logger.debug(f"on_callback: no message for data={data!r}")
        with suppress(Exception):
            await query.answer()
        return

    action = classify_callback(data, speed_callbacks=SPEED_OPTIONS)
    if action is None:
        logger.info("Ignored unknown or expired callback data=%r user=%s", data, user_id)
        with suppress(Exception):
            await query.answer(
                "This button is no longer available. Please reopen the menu.",
                show_alert=False,
            )
        return

    with suppress(Exception):
        await query.answer()

    try:
        if callback_requires_tts_access(action, data) and not await _ensure_user_allowed(
            update,
            context,
            "tts_enabled",
            "បម្លែងអត្ថបទទៅជាសំឡេង",
        ):
            return

        if action == "welcome_profile":
            from app.services.telegram.commands import send_user_profile

            await send_user_profile(query.message, user_id)
        elif action == "show_speed":
            await _cb_show_speed(query, user_id, context)
        elif action == "hide_speed":
            await _cb_hide_speed(query, user_id, context)
        elif action == "show_tts_model":
            await _cb_show_tts_model(query, user_id, context)
        elif action == "hide_tts_model":
            await _cb_hide_tts_model(query, user_id, context)
        elif action == "tts_model":
            await _cb_tts_model(query, user_id, context, data)
        elif action == "speed":
            await _cb_speed(query, user_id, context, data)
        elif action == "gender":
            await _cb_gender(query, user_id, context, data)
        elif action == "tts_transcript":
            await _cb_tts_transcript(query, user_id, context, data)
        elif action == "delete":
            with suppress(Exception):
                await query.message.delete()
        elif action == "doc_read":
            await _cb_doc_read(query, user_id, context, data)
        elif action == "audio_tts":
            await _cb_audio_tts(query, user_id, context, data)
        elif action == "needs_admin":
            await _cb_user_needs_admin(query, user_id, context, data)
        elif action == "api_admin":
            await _cb_api_dashboard(query, user_id, context, data)
        elif action == "admin":
            await _cb_admin_dashboard(query, user_id, context, data)
    except Exception as exc:
        _metric_inc("errors")
        logger.error("on_callback failed action=%s data=%r: %s", action, data, exc, exc_info=True)
        await safe_send(lambda: query.message.reply_text(
            "⚠️ Something went wrong while processing this button. Please try again."
        ))


__all__ = [
    'broadcast_callback',
    'users_page_callback',
    'sched_callback',
    '_runtime_admin_callback',
    'on_callback'
]
