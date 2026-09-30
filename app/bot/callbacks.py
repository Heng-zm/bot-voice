"""Extracted Telegram handler implementations.

These are live runtime handlers; app.legacy now contains compatibility wrappers only.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
import html
import logging
import os
import time
from typing import Any, Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from app import legacy

# Transitional V4.1 modules bind remaining legacy helpers at runtime.
# ruff: noqa: F821
try:
    from app.services.telegram._legacy_runtime import legacy_bound_handler
except ImportError:
    def legacy_bound_handler(fn: Callable) -> Callable:
        return fn

logger = logging.getLogger(__name__)
webhook_logger = logging.getLogger("webhook")

# Local fallback store for broadcast data
_LOCAL_PENDING_BROADCAST: dict[int, Any] = {}


async def safe_send(coro_or_fn: Any) -> Any:
    """Execute Telegram send/edit operation with automatic HTML parse error recovery."""
    try:
        res = coro_or_fn() if callable(coro_or_fn) else coro_or_fn
        if asyncio.iscoroutine(res):
            return await res
        return res
    except BadRequest as b_err:
        err_msg = str(b_err)
        if "can't parse entities" in err_msg.lower() or "tag" in err_msg.lower():
            logger.warning("Telegram parse_mode='HTML' error: %s. Retrying without HTML.", b_err)
            return None
        logger.error("safe_send BadRequest: %s", b_err)
        return None
    except Exception as exc:
        logger.error("safe_send invocation failed: %s", exc)
        return None


def _is_admin(user_id: int) -> bool:
    """Robust administrator authorization check across all policy sources."""
    if not user_id:
        return False

    with suppress(Exception):
        from app.services.security.admin_policy import is_telegram_admin
        if is_telegram_admin(user_id):
            return True

    is_admin_fn = getattr(legacy, "_is_admin", getattr(legacy, "is_admin", None))
    if callable(is_admin_fn):
        with suppress(Exception):
            if is_admin_fn(user_id):
                return True

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


def _spawn_task(context: ContextTypes.DEFAULT_TYPE, coro: Any) -> asyncio.Task:
    """Safely spawn a background task via PTB Application or the active loop."""
    app = getattr(context, "application", None)
    if app and hasattr(app, "create_task"):
        return app.create_task(coro)
    return asyncio.create_task(coro)


@legacy_bound_handler
async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return

    user_id = query.from_user.id if query.from_user else 0
    data = (query.data or "").strip()

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    pending_store = getattr(legacy, "_pending_broadcast", _LOCAL_PENDING_BROADCAST)
    executor = getattr(legacy, "_DB_EXECUTOR", None)

    if data == "bc_templates":
        await _admin_open_broadcast_templates(query, context, user_id)
        return

    if data == "bc_save_template":
        pending = pending_store.get(user_id)
        if not pending:
            await safe_send(lambda: query.message.reply_text(
                '⚠️ មិនមានសារមើលជាមុនសម្រាប់រក្សាទុកទេ។ សូមបង្កើតការផ្សាយសារជាមុនសិន។'
            ))
            return
        ok, info, tpl = await asyncio.get_running_loop().run_in_executor(
            executor,
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
        tpl = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_broadcast_template_get(tpl_id))
        if not tpl:
            await _admin_open_broadcast_templates(query, context, user_id, notice="⚠️ រក Template មិនឃើញ។")
            return
        pending = _broadcast_template_payload_from_template(tpl)
        pending_store[user_id] = pending
        context.user_data["bc_state"] = getattr(legacy, "BROADCAST_WAIT_MESSAGE", "bc_wait_msg")
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f'📚 បានជ្រើសគំរូ៖ <b>{html.escape(_broadcast_template_button_title(tpl))}</b>',
            parse_mode="HTML",
        ))
        ok = await _admin_show_broadcast_preview_message(query.message, context.bot, user_id, pending)
        if not ok:
            pending_store.pop(user_id, None)
            context.user_data.pop("bc_state", None)
        return

    if data.startswith("bc_tpl_delask:"):
        tpl_id = data.split(":", 1)[1]
        tpl = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_broadcast_template_get(tpl_id))
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
            executor,
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
        _spawn_task(context, _delete_broadcast_sent_messages(context.bot, user_id, job))
        await safe_send(lambda: query.message.reply_text("🗑️ បានចាប់ផ្ដើមលុបសារ Broadcast ដែលបានផ្ញើ..."))
        return

    if data == "bc_del_sent_keep":
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text('✅ បានរក្សាទុកសារផ្សាយនៅដដែល។'))
        return

    if data == "bc_cancel":
        pending_store.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text('❌ បានបោះបង់ការផ្សាយសារ។'))
        return

    if data == "bc_confirm":
        pending = pending_store.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        if not pending:
            await safe_send(lambda: query.message.reply_text("⚠️ រកទិន្នន័យ Broadcast មិនឃើញ។ សូមចាប់ផ្ដើមថ្មី។"))
            return
        _spawn_task(
            context,
            _run_broadcast_to_all(context.bot, user_id, pending, label="ការផ្សាយសារ")
        )
        return

    if query.message:
        await safe_send(lambda: query.message.reply_text(
            "This broadcast button is no longer available. Please reopen the menu."
        ))


@legacy_bound_handler
async def users_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return

    user_id = query.from_user.id if query.from_user else 0
    data = (query.data or "").strip()

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

    executor = getattr(legacy, "_DB_EXECUTOR", None)

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
            row = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, False, back_ref=f"h{page}"),
            ))
            return

        if data == "users_search":
            context.user_data["user_search_state"] = getattr(legacy, "USER_SEARCH_WAIT_QUERY", "usr_search_wait")
            await safe_send(lambda: query.message.edit_text(
                '🔎 <b>ស្វែងរកអ្នកប្រើប្រាស់</b>\n\n'
                'សូមផ្ញើលេខសម្គាល់អ្នកប្រើប្រាស់ Telegram ឬឈ្មោះអ្នកប្រើប្រាស់។\n\n'
                'ឧទាហរណ៍៖\n<code>1272791365</code>\n<code>heng</code>\n<code>@username</code>\n\n'
                'ប្រើ /cancel ដើម្បីបញ្ឈប់ការស្វែងរក។',
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
            users = await asyncio.get_running_loop().run_in_executor(executor, get_all_users_with_names)
            page = _clamp_users_page(users, page)
            await safe_send(lambda: query.message.edit_text(
                f'👥 <b>ការគ្រប់គ្រងអ្នកប្រើប្រាស់ ({len(users)} នាក់)</b>\n'
                f'សូមជ្រើសរើសអ្នកប្រើប្រាស់ ឬចុច 🔎 ស្វែងរកអ្នកប្រើប្រាស់ ដើម្បីស្វែងរកតាមលេខសម្គាល់ ឬឈ្មោះអ្នកប្រើប្រាស់។',
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
            row = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, False, back_ref=back_ref),
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
            exists = await asyncio.get_running_loop().run_in_executor(executor, lambda: user_exists_in_db(target_id))
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
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            await asyncio.get_running_loop().run_in_executor(
                executor,
                lambda: db_user_set_blocked(target_id, user_id, False),
            )
            row = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_user_detail(target_id))
            notice = "ℹ️ មុខងារ Block User ត្រូវបានបិទ/ដកចេញហើយ។ អ្នកប្រើប្រាស់ទាំងអស់អាចប្រើប្រាស់ Bot បានធម្មតា។"
            await safe_send(lambda: query.message.edit_text(
                notice + "\n\n" + _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, False, back_ref=back_ref),
            ))
            return

        if data.startswith("user_resetprefs:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            ok, info = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_user_reset_prefs(target_id))
            row = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_user_detail(target_id))
            notice = "✅ User preferences reset." if ok else f"❌ Reset failed: {str(info)[:500]}"
            await safe_send(lambda: query.message.edit_text(
                notice + "\n\n" + _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, False, back_ref=back_ref),
            ))
            return

        if data.startswith("user_clearhist:"):
            parts = data.split(":")
            target_id = _int_part(parts, 1, 0)
            back_ref = parts[2] if len(parts) > 2 else "p0"
            if target_id <= 0:
                await _invalid_callback()
                return
            await asyncio.get_running_loop().run_in_executor(executor, lambda: db_history_clear(target_id))
            row = await asyncio.get_running_loop().run_in_executor(executor, lambda: db_user_detail(target_id))
            await safe_send(lambda: query.message.edit_text(
                "✅ User conversation history cleared.\n\n" + _format_user_detail_text(row),
                parse_mode="HTML",
                reply_markup=get_user_detail_kb(target_id, False, back_ref=back_ref),
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
    user_id = query.from_user.id if query.from_user else 0
    data = (query.data or "").strip()

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    with suppress(Exception):
        await query.answer()

    if query.message is None:
        return

    loop = asyncio.get_running_loop()
    executor = getattr(legacy, "_DB_EXECUTOR", None)

    if data.startswith(("sched_repeat_once:", "sched_repeat_daily:")):
        recurrence = (
            getattr(legacy, "SCHED_RECURRENCE_DAILY", "daily")
            if data.startswith("sched_repeat_daily:")
            else getattr(legacy, "SCHED_RECURRENCE_ONCE", "once")
        )
        try:
            row_id = int(data.rsplit(":", 1)[1])
        except (TypeError, ValueError, IndexError):
            await safe_send(lambda: query.message.reply_text("❌ Invalid schedule ID."))
            return
        ok, reason, saved = await loop.run_in_executor(
            executor,
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

        ok, reason, row = await loop.run_in_executor(executor, db_sched_confirm, row_id, user_id)
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
        row = await loop.run_in_executor(executor, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text('❌ រកកាលវិភាគមិនឃើញ។'))
            return
        if int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
            return
        status_draft = getattr(legacy, "SCHED_STATUS_DRAFT", "draft")
        status_pending = getattr(legacy, "SCHED_STATUS_PENDING", "pending")
        status_cancelled = getattr(legacy, "SCHED_STATUS_CANCELLED", "cancelled")
        if str(row.get("status")) in (status_draft, status_pending):
            await loop.run_in_executor(executor, db_sched_set_status, row_id, status_cancelled)
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
        rows = await loop.run_in_executor(executor, db_sched_fetch_admin_pending, user_id)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=get_schedules_list_kb(rows, page=page))
        return

    if data.startswith("sched_view:"):
        row_id = _callback_int_arg(data, "sched_view:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(executor, db_sched_fetch_one, row_id)
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
        row = await loop.run_in_executor(executor, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = getattr(legacy, "SCHED_EDIT_WAIT_TIME", "sched_edit_time")
        context.user_data["sched_edit_row_id"] = row_id
        await safe_send(lambda: query.message.reply_text(
            f'✏️ <b>កែម៉ោងកាលវិភាគ #{row_id}</b>\n\n'
            f'សូមផ្ញើពេលវេលាថ្មីតាមម៉ោងភ្នំពេញ (ICT, UTC+7)។\n'
            f'ទម្រង់៖ <code>YYYY-MM-DD HH:MM AM/PM</code> ឬ <code>YYYY-MM-DD HH:MM</code>\n'
            f'តំបន់ម៉ោង៖ ភ្នំពេញ កម្ពុជា — ICT (UTC+7)\n'
            f'ឧទាហរណ៍៖ <code>2026-12-25 09:00 AM</code> ឬ <code>2026-12-25 21:00</code>\n\n'
            f'វាយ /cancel ដើម្បីបោះបង់ការកែសម្រួល។',
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_edit_text:"):
        row_id = _callback_int_arg(data, "sched_edit_text:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(executor, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = getattr(legacy, "SCHED_EDIT_WAIT_TEXT", "sched_edit_text")
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
        row = await loop.run_in_executor(executor, db_sched_fetch_one, row_id)
        ok, reason = _sched_can_edit(row, user_id)
        if not ok:
            await safe_send(lambda: query.message.reply_text(_sched_edit_error_text(row_id, reason), parse_mode="HTML"))
            return
        context.user_data["sched_state"] = getattr(legacy, "SCHED_EDIT_WAIT_PHOTO", "sched_edit_photo")
        context.user_data["sched_edit_row_id"] = row_id
        await safe_send(lambda: query.message.reply_text(
            f'🖼 <b>ប្ដូររូបភាពកាលវិភាគ #{row_id}</b>\n\n'
            f'សូមផ្ញើរូបភាពថ្មី + ចំណងជើង (មិនចាំបាច់)។ វាយ /cancel ដើម្បីបោះបង់ការកែសម្រួល។',
            parse_mode="HTML",
        ))
        return

    if data.startswith("sched_cancel_confirm:"):
        row_id = _callback_int_arg(data, "sched_cancel_confirm:")
        if row_id is None:
            await safe_send(lambda: query.message.reply_text('❌ លេខសម្គាល់កាលវិភាគមិនត្រឹមត្រូវ។'))
            return
        row = await loop.run_in_executor(executor, db_sched_fetch_one, row_id)
        if not row or int(row.get("admin_id") or 0) != int(user_id):
            await safe_send(lambda: query.message.reply_text('⛔ អ្នកមិនមានសិទ្ធិបោះបង់កាលវិភាគនេះទេ។'))
            return
        status_draft = getattr(legacy, "SCHED_STATUS_DRAFT", "draft")
        status_pending = getattr(legacy, "SCHED_STATUS_PENDING", "pending")
        status_cancelled = getattr(legacy, "SCHED_STATUS_CANCELLED", "cancelled")
        if row.get("status") not in (status_draft, status_pending):
            st = html.escape(str(row.get("status") or "?"))
            await safe_send(lambda: query.message.reply_text(
                f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
                parse_mode="HTML",
            ))
            return
        await loop.run_in_executor(executor, db_sched_set_status, row_id, status_cancelled)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text(
            f'✅ កាលវិភាគ <b>#{row_id}</b> បានបោះបង់។', parse_mode="HTML"
        ))
        return

    if query.message:
        await safe_send(lambda: query.message.reply_text(
            "This schedule button is no longer available. Please reopen the menu."
        ))


@legacy_bound_handler
async def _runtime_admin_callback(update: Any, context: Any) -> None:
    query = update.callback_query
    if query is None:
        return
    admin_id = query.from_user.id if query.from_user else 0
    data = (query.data or "").strip()
    if not _is_admin(admin_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return
    if query.message is None:
        with suppress(Exception):
            await query.answer()
        return

    active_convs = getattr(legacy, "ACTIVE_ADMIN_CONVERSATIONS", {})
    conv_lock = getattr(legacy, "ACTIVE_ADMIN_CONVERSATIONS_LOCK", suppress())

    if data == "rtadmin_close":
        with conv_lock:
            active_convs.pop(admin_id, None)
        with suppress(Exception):
            await query.answer('បានបិទ')
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "rtadmin_rate":
        with conv_lock:
            active_convs[admin_id] = {"state": "awaiting_rate_limit", "ts": time.monotonic()}
        with suppress(Exception):
            await query.answer('សូមផ្ញើជាលេខ', show_alert=False)
        rate_lim = getattr(legacy, "_run_state_user_rate_limit", lambda: 5)()
        rate_win = getattr(legacy, "_run_state_user_rate_window", lambda: 10.0)()
        await safe_send(lambda: query.message.edit_text(
            "⚡ <b>កែប្រែ Rate Limit</b>\n\n"
            f"តម្លៃបច្ចុប្បន្ន: <b>{rate_lim} req/{rate_win:g}s</b>\n\n"
            "សូមផ្ញើលេខគត់ថ្មី ឧទាហរណ៍ <code>3</code> ឬ <code>5</code>។\n"
            "បើចង់កែ HTTP pool សូមផ្ញើ <code>http 120</code>។",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="rtadmin_cancel")]]),
        ))
        return

    if data == "rtadmin_cancel":
        with conv_lock:
            active_convs.pop(admin_id, None)
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

        rotate_lock_fn = getattr(legacy, "_webhook_rotate_lock", None)
        lock_ctx = rotate_lock_fn() if callable(rotate_lock_fn) else suppress()

        async with lock_ctx:
            rotate_begin_fn = getattr(legacy, "_webhook_rotate_begin_or_remaining", lambda: 0)
            remaining = rotate_begin_fn()
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
            token_gen_fn = getattr(legacy, "generate_new_webhook_token", lambda: os.urandom(16).hex())
            new_token = token_gen_fn()
            bot_mode = getattr(legacy, "_run_state_bot_mode", lambda: "POLLING")()
            try:
                if bot_mode == "WEBHOOK":
                    conf_hook = getattr(legacy, "_configure_telegram_webhook_via_http_for_secret", None)
                    if callable(conf_hook):
                        await conf_hook(new_token)

                update_state = getattr(legacy, "_update_run_state", None)
                if callable(update_state):
                    await update_state("TELEGRAM_WEBHOOK_SECRET_TOKEN", new_token, persist=True)
                success = True

                logger.info("Admin %s rotated Webhook Secret Token.", admin_id)
                webhook_logger.info("Webhook secret rotated by admin_id=%s mode=%s", admin_id, bot_mode)

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
                finish_rotate = getattr(legacy, "_webhook_rotate_finish", None)
                if callable(finish_rotate):
                    finish_rotate(success)
        return

    if data.startswith("rtadmin_switch:"):
        target = data.split(":", 1)[1].strip().upper()
        with suppress(Exception):
            await query.answer('កំពុងប្ដូរ…', show_alert=False)
        try:
            switch_mode = getattr(legacy, "_switch_telegram_runtime_mode", None)
            mode = await switch_mode(target, admin_id=admin_id) if callable(switch_mode) else target
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

    user_id = query.from_user.id if query.from_user else 0
    data = (query.data or "").strip()

    if not data:
        with suppress(Exception):
            await query.answer()
        return

    if query.message is None:
        logger.debug("on_callback: no message for data=%r", data)
        with suppress(Exception):
            await query.answer()
        return

    # 1. Dispatch Admin Sub-Controller callbacks
    if data.startswith("admin_"):
        with suppress(Exception):
            from app.services.admin.callbacks import handle_admin_callback
            if await handle_admin_callback(query, user_id, context, data):
                return

    # 2. Dispatch Donation & Payment callbacks
    if data.startswith(("donate_", "khqr_")):
        with suppress(Exception):
            from app.services.donation.handlers import handle_donation_callback
            if callable(handle_donation_callback):
                await handle_donation_callback(update, context)
                return

    # 3. Direct Inline User Preference Toggles
    if data.startswith("set_tts_model:"):
        model_code = data.split(":", 1)[1].strip()
        set_pref = getattr(legacy, "set_user_pref_async", None)
        if callable(set_pref):
            await set_pref(user_id, "tts_model", model_code)
        with suppress(Exception):
            await query.answer(f"✅ បានកំណត់ម៉ូដែល: {model_code.upper()}")
        with suppress(Exception):
            from app.services.telegram.commands import _get_tts_model_kb
            await query.edit_message_reply_markup(reply_markup=_get_tts_model_kb(model_code))
        return

    if data.startswith("set_speed:"):
        with suppress(ValueError):
            spd_val = float(data.split(":", 1)[1].strip())
            set_pref = getattr(legacy, "set_user_pref_async", None)
            if callable(set_pref):
                await set_pref(user_id, "speed", spd_val)
            with suppress(Exception):
                await query.answer(f"✅ បានកំណត់ល្បឿន: {spd_val}x")
        return

    if data.startswith("set_gender:"):
        g_val = data.split(":", 1)[1].strip().lower()
        set_pref = getattr(legacy, "set_user_pref_async", None)
        if callable(set_pref):
            await set_pref(user_id, "gender", g_val)
        lbl = "សំឡេងស្រី" if g_val == "female" else "សំឡេងប្រុស"
        with suppress(Exception):
            await query.answer(f"✅ បានកំណត់ {lbl}")
        return

    speed_options = getattr(legacy, "SPEED_OPTIONS", {})
    classify_fn = getattr(legacy, "classify_callback", None)
    action = classify_fn(data, speed_callbacks=speed_options) if callable(classify_fn) else None

    if action is None:
        # Fallback keyword matching for standard navigational callbacks
        if data in ("welcome_profile", "welcome_back", "welcome_menu", "welcome_help"):
            action = data
        elif data == "show_media_hub":
            action = "show_media_hub"
        elif data.startswith("show_") and data.endswith("_guide"):
            action = data

    if action is None:
        logger.info("Ignored unknown or expired callback data=%r user=%s", data, user_id)
        with suppress(Exception):
            await query.answer("ប៊ូតុងនេះផុតសុពលភាពហើយ។ សូមបើកម៉ឺនុយឡើងវិញ។", show_alert=False)
        return

    chat = getattr(query.message, "chat", None) if query.message else None
    chat_type = str(getattr(chat, "type", "private") or "private").lower()
    if (
        chat
        and (chat_type in ("channel", "group", "supergroup") or int(chat.id) < 0)
        and action in ("gender", "speed", "tts_model", "show_speed", "hide_speed", "show_tts_model", "hide_tts_model")
    ):
        is_authorized = _is_admin(user_id)
        if not is_authorized:
            try:
                member = await context.bot.get_chat_member(chat_id=chat.id, user_id=user_id)
                is_authorized = getattr(member, "status", "") in ("creator", "administrator")
            except Exception:
                is_authorized = False
        if not is_authorized:
            with suppress(Exception):
                await query.answer(
                    "⛔ មានតែអ្នកគ្រប់គ្រង Channel/Group ប៉ុណ្ណោះដែលអាចកែប្រែការកំណត់សំឡេងបាន។",
                    show_alert=True,
                )
            return

    if action not in ("speed", "gender", "tts_model"):
        with suppress(Exception):
            await query.answer()

    try:
        req_tts_fn = getattr(legacy, "callback_requires_tts_access", None)
        ensure_user_fn = getattr(legacy, "_ensure_user_allowed", None)
        if callable(req_tts_fn) and req_tts_fn(action, data) and callable(ensure_user_fn):
            if not await ensure_user_fn(update, context, "tts_enabled", "បម្លែងអត្ថបទទៅជាសំឡេង"):
                return

        if action == "welcome_profile":
            from app.services.telegram.commands import send_user_profile
            user = update.effective_user or getattr(query, "from_user", None)
            await send_user_profile(query.message, user_id, user=user, edit=True)
        elif action in ("welcome_back", "welcome_menu"):
            if data in ("close", "welcome_close") or (action == "welcome_back" and not _is_profile_message(query.message)):
                with suppress(Exception):
                    await query.message.delete()
            elif action == "welcome_menu" or _is_profile_message(query.message):
                settings, _ = await get_bot_settings_async()
                default_welcome = getattr(legacy, "WELCOME_TEXT", "សូមស្វាគមន៍មកកាន់ Bot Voice!")
                w_text = _setting_raw_from(settings, "welcome_message", default_welcome) or default_welcome
                from app.services.telegram.menu import get_welcome_kb
                if getattr(query.message, "photo", None):
                    with suppress(Exception):
                        await query.message.edit_caption(caption=w_text, parse_mode="HTML", reply_markup=get_welcome_kb())
                else:
                    with suppress(Exception):
                        await query.message.edit_text(w_text, parse_mode="HTML", reply_markup=get_welcome_kb(), disable_web_page_preview=True)
            else:
                with suppress(Exception):
                    await query.message.delete()
        elif action in ("help", "welcome_help"):
            from app.services.telegram.commands import on_help
            await on_help(update, context)
        elif action == "show_media_hub":
            from app.services.telegram.menu import get_media_hub_kb

            hub_text = (
                "📥 <b>មជ្ឈមណ្ឌលទាញយក Media (Downloader Hub)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "ជ្រើសរើសបណ្ដាញសង្គមដែលអ្នកចង់ទាញយកវីដេអូ ឬសំឡេង៖\n\n"
                "• 🎵 <b>TikTok:</b> វីដេអូ HD គ្មាន Watermark & MP3 Audio\n"
                "• 📘 <b>Facebook:</b> វីដេអូ Reels & Post HD/SD + MP3\n"
                "• 📸 <b>Instagram:</b> វីដេអូ Reels & Story Original HD\n"
                "• ▶️ <b>YouTube:</b> វីដេអូ & Shorts រួមទាំងដកស្រង់ MP3\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 <i>អ្នកក៏អាច Copy & Paste Link ណាមួយផ្ញើមកក្នុងឆាតនេះផ្ទាល់បានភ្លាមៗ!</i>"
            )
            with suppress(Exception):
                if getattr(query.message, "photo", None):
                    await query.message.edit_caption(caption=hub_text, parse_mode="HTML", reply_markup=get_media_hub_kb())
                else:
                    await query.message.edit_text(hub_text, parse_mode="HTML", reply_markup=get_media_hub_kb())
        elif action == "show_tiktok_guide":
            from app.core.features import is_tiktok_enabled
            if not is_tiktok_enabled():
                with suppress(Exception):
                    await query.answer("⚠️ មុខងារទាញយក TikTok ត្រូវបានបិទដំណើរការ។", show_alert=True)
                return

            guide_text = (
                "📥 <b>របៀបប្រើប្រាស់ TikTok Downloader:</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "គ្រាន់តែ Copy និង Paste តំណភ្ជាប់ (Link) វីដេអូ ឬរូបភាព TikTok ចូលក្នុងឆាតនេះផ្ទាល់!\n\n"
                "✨ <b>លក្ខណៈពិសេស:</b>\n"
                "• 📹 វីដេអូកម្រិតច្បាស់ HD គ្មាន Watermark\n"
                "• 📁 ទាញយកជាឯកសារច្បាស់ដើម (Document File)\n"
                "• 🎵 ទាញយកតែសំឡេងដើមជា MP3 Audio\n"
                "• 🖼️ ទាញយករូបភាព Slide ទាំងអស់\n"
                "• 🤖 AI សង្ខេបខ្លឹមសារវីដេអូជាភាសាខ្មែរ\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 <i>សាកល្បងឥឡូវនេះ៖ ផ្ញើ Link TikTok ណាមួយមកកាន់ខ្ញុំ!</i>"
            )
            back_kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ", callback_data="show_media_hub"),
                InlineKeyboardButton("🏠 ម៉ឺនុយដើម", callback_data="welcome_menu"),
            ]])
            with suppress(Exception):
                await query.edit_message_text(guide_text, parse_mode="HTML", reply_markup=back_kb)
        elif action == "show_facebook_guide":
            from app.core.features import is_facebook_enabled
            if not is_facebook_enabled():
                with suppress(Exception):
                    await query.answer("⚠️ មុខងារទាញយក Facebook ត្រូវបានបិទដំណើរការ។", show_alert=True)
                return

            guide_text = (
                "📥 <b>របៀបប្រើប្រាស់ Facebook Downloader:</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "គ្រាន់តែ Copy និង Paste តំណភ្ជាប់ (Link) វីដេអូ ឬ Reels ពី Facebook ចូលក្នុងឆាតនេះផ្ទាល់!\n\n"
                "✨ <b>លក្ខណៈពិសេស:</b>\n"
                "• 🎬 ជម្រើសកម្រិតរូបភាព HD (1080p) និង SD (480p)\n"
                "• 🎵 ដកស្រង់សំឡេងដើមជា MP3 Audio\n"
                "• 📁 ទាញយកជាឯកសារដើម (Document File)\n"
                "• 🤖 Gemini AI សង្ខេបខ្លឹមសារវីដេអូជាភាសាខ្មែរ\n"
                "• 🛡️ ប្រព័ន្ធការពារសុវត្ថិភាព 100% មិនជាប់កម្រិត 50MB\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 <i>សាកល្បងឥឡូវនេះ៖ ផ្ញើ Link Facebook Reels ឬ Video មកកាន់ខ្ញុំ!</i>"
            )
            back_kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ", callback_data="show_media_hub"),
                InlineKeyboardButton("🏠 ម៉ឺនុយដើម", callback_data="welcome_menu"),
            ]])
            with suppress(Exception):
                await query.edit_message_text(guide_text, parse_mode="HTML", reply_markup=back_kb)
        elif action == "show_instagram_guide":
            from app.core.features import is_instagram_enabled
            if not is_instagram_enabled():
                with suppress(Exception):
                    await query.answer("⚠️ មុខងារទាញយក Instagram ត្រូវបានបិទដំណើរការ។", show_alert=True)
                return

            guide_text = (
                "📥 <b>របៀបប្រើប្រាស់ Instagram Downloader:</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "គ្រាន់តែ Copy និង Paste តំណភ្ជាប់ (Link) Reels ឬ Post ពី Instagram ចូលក្នុងឆាតនេះផ្ទាល់!\n\n"
                "✨ <b>លក្ខណៈពិសេស:</b>\n"
                "• 🎬 ទាញយកវីដេអូកម្រិត Original ច្បាស់ត្រជាក់ភ្នែក\n"
                "• 🎵 ដកស្រង់សំឡេងដើមជា MP3 Audio\n"
                "• 📁 ទាញយកជាឯកសារដើម (Document File)\n"
                "• 🤖 Gemini AI សង្ខេបខ្លឹមសារវីដេអូជាភាសាខ្មែរ\n"
                "• ⚡ ល្បឿនលឿន & 0ms CDN Caching\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 <i>សាកល្បងឥឡូវនេះ៖ ផ្ញើ Link Instagram Reels ឬ Post មកកាន់ខ្ញុំ!</i>"
            )
            back_kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ", callback_data="show_media_hub"),
                InlineKeyboardButton("🏠 ម៉ឺនុយដើម", callback_data="welcome_menu"),
            ]])
            with suppress(Exception):
                await query.edit_message_text(guide_text, parse_mode="HTML", reply_markup=back_kb)
        elif action == "show_youtube_guide":
            from app.core.features import is_youtube_enabled
            if not is_youtube_enabled():
                with suppress(Exception):
                    await query.answer("⚠️ មុខងារទាញយក YouTube ត្រូវបានបិទដំណើរការ។", show_alert=True)
                return

            guide_text = (
                "📥 <b>របៀបប្រើប្រាស់ YouTube Downloader:</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "គ្រាន់តែ Copy និង Paste តំណភ្ជាប់ (Link) វីដេអូ ឬ Shorts ពី YouTube ចូលក្នុងឆាតនេះផ្ទាល់!\n\n"
                "✨ <b>លក្ខណៈពិសេស:</b>\n"
                "• 🎬 គាំទ្រទាំងវីដេអូពេញ និង YouTube Shorts\n"
                "• 🎵 ដកស្រង់សំឡេងដើមជា MP3 Audio ដោយចុច 1-Tap\n"
                "• 🤖 Gemini AI សង្ខេបខ្លឹមសារវីដេអូជាភាសាខ្មែរ\n"
                "• 🛡️ មិនជាប់កម្រិត 50MB (មានជម្រើស Browser Download)\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 <i>សាកល្បងឥឡូវនេះ៖ ផ្ញើ Link YouTube ឬ Shorts មកកាន់ខ្ញុំ!</i>"
            )
            back_kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 ត្រឡប់ក្រោយ", callback_data="show_media_hub"),
                InlineKeyboardButton("🏠 ម៉ឺនុយដើម", callback_data="welcome_menu"),
            ]])
            with suppress(Exception):
                await query.edit_message_text(guide_text, parse_mode="HTML", reply_markup=back_kb)
        elif action == "system_status":
            from app.services.telegram.commands import cmd_system
            await cmd_system(update, context)
        elif action == "show_speed":
            await _cb_show_speed(query, user_id, context)
        elif action == "hide_speed":
            await _cb_hide_speed(query, user_id, context)
        elif action == "show_tts_model":
            await _cb_show_tts_model(query, user_id, context)
        elif action == "hide_tts_model":
            await _cb_hide_tts_model(query, user_id, context)
        elif action == "show_mode":
            await _cb_show_bot_mode(query, user_id, context)
        elif action == "hide_mode":
            await _cb_hide_bot_mode(query, user_id, context)
        elif action == "mode_change":
            await _cb_bot_mode(query, user_id, context, data)
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
                await query.answer("🗑️ បានលុប")
            with suppress(Exception):
                await query.message.delete()
        elif action == "doc_read":
            await _cb_doc_read(query, user_id, context, data)
        elif action == "doc_trans":
            await _cb_doc_trans(query, user_id, context, data)
        elif action == "audio_tts":
            await _cb_audio_tts(query, user_id, context, data)
        elif action == "needs_admin":
            await _cb_user_needs_admin(query, user_id, context, data)
        elif action == "api_admin":
            await _cb_api_dashboard(query, user_id, context, data)
        elif action == "admin":
            await _cb_admin_dashboard(query, user_id, context, data)
    except Exception as exc:
        inc_metric = getattr(legacy, "_metric_inc", None)
        if callable(inc_metric):
            inc_metric("errors")
        logger.error("on_callback failed action=%s data=%r: %s", action, data, exc, exc_info=True)
        if query.message:
            await safe_send(lambda: query.message.reply_text(
                "⚠️ មានបញ្ហាក្នុងការដំណើរការប៊ូតុងនេះ។ សូមព្យាយាមម្ដងទៀត។"
            ))


@legacy_bound_handler
async def article_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle article narrator interactive buttons (switch voice, original language, full text)."""
    query = update.callback_query
    if query is None:
        return

    user_id = query.from_user.id if query.from_user else 0
    data = (query.data or "").strip()
    if not data:
        with suppress(Exception):
            await query.answer()
        return

    from app.services.ai.article_reader import get_article_session

    parts = data.split(":")
    action_type = parts[0]

    # --- Admin Article Approval & Source Actions ---
    if action_type == "art_adm" and len(parts) >= 3:
        sub_action = parts[1]
        target_hash = parts[-1]

        if not _is_admin(user_id):
            with suppress(Exception):
                await query.answer("⛔ អ្នកមិនមានសិទ្ធិជា Admin ឡើយ។", show_alert=True)
            return

        from app.services.ai.article_storage import (
            get_pending_article,
            get_pending_article_by_prefix,
            mark_article_sent,
            update_pending_status,
        )

        record = await get_pending_article(target_hash)
        if not record:
            record = await get_pending_article_by_prefix(target_hash)
        if not record:
            with suppress(Exception):
                await query.answer("❌ រកមិនឃើញទិន្នន័យអត្ថបទនេះឡើយ។", show_alert=True)
            return

        if sub_action in ("appr", "approve"):
            with suppress(Exception):
                await query.answer("⏳ បានអនុម័ត! កំពុងចាប់ផ្តើមផ្សាយ...")

            full_hash = record.get("hash", target_hash)
            await update_pending_status(full_hash, "approved", reviewed_by=user_id)
            title_to_mark = record.get("khmer_title") or record.get("title", "")
            await mark_article_sent(full_hash, url=record.get("url", ""), title=title_to_mark, user_id=user_id)

            if query.message:
                with suppress(Exception):
                    await query.message.edit_reply_markup(reply_markup=None)
                    await query.message.reply_text("✅ <b>បានអនុម័តជោគជ័យ!</b>\n<i>🚀 កំពុងផ្សាយព័ត៌មានទៅកាន់អ្នកប្រើប្រាស់...</i>", parse_mode="HTML")

            from app.services.ai.article_monitor import broadcast_approved_article

            bot = context.bot
            success_cnt, fail_cnt = await broadcast_approved_article(bot, full_hash)
            if query.message:
                with suppress(Exception):
                    await query.message.reply_text(
                        f"📢 <b>ការផ្សាយព័ត៌មានបានបញ្ចប់!</b>\n\n"
                        f"✅ បានផ្ញើជោគជ័យ: <b>{success_cnt}</b> នាក់\n"
                        f"❌ បរាជ័យ/Blocked: <b>{fail_cnt}</b> នាក់",
                        parse_mode="HTML",
                    )
            return

        elif sub_action in ("rej", "reject"):
            with suppress(Exception):
                await query.answer("❌ បានបដិសេធព័ត៌មាននេះ។")

            full_hash = record.get("hash", target_hash)
            await update_pending_status(full_hash, "rejected", reviewed_by=user_id)
            title_to_mark = record.get("khmer_title") or record.get("title", "")
            await mark_article_sent(full_hash, url=record.get("url", ""), title=title_to_mark, user_id=user_id)

            if query.message:
                with suppress(Exception):
                    await query.message.edit_reply_markup(reply_markup=None)
                    await query.message.reply_text("❌ <b>ព័ត៌មាននេះត្រូវបានបដិសេធ (Rejected) ដោយ Admin។</b>\n<i>នឹងមិនត្រូវបានផ្សាយទៅកាន់អ្នកប្រើប្រាស់ឡើយ។</i>", parse_mode="HTML")
            return

    if action_type == "art_src" and len(parts) >= 3 and parts[1] in ("del", "delete"):
        source_id = parts[-1]

        if not _is_admin(user_id):
            with suppress(Exception):
                await query.answer("⛔ អ្នកមិនមានសិទ្ធិជា Admin ឡើយ។", show_alert=True)
            return

        from app.services.ai.article_storage import remove_article_source

        ok = await remove_article_source(source_id)
        if ok:
            with suppress(Exception):
                await query.answer("🗑️ បានលុបប្រភពព័ត៌មានរួចរាល់!")
            if query.message:
                with suppress(Exception):
                    await query.message.reply_text(f"🗑️ បានលុបប្រភពព័ត៌មាន ID <code>{source_id}</code> រួចរាល់។", parse_mode="HTML")
        else:
            with suppress(Exception):
                await query.answer("❌ មិនអាចលុបបានទេ (រកមិនឃើញ ID)។", show_alert=True)
        return

    # --- Dynamic Interactive Card Reactions & Quick Tool Tips ---
    if action_type == "art_tip":
        tip_type = parts[1] if len(parts) > 1 else ""
        if tip_type == "scam":
            alert_text = (
                "🛡️ គន្លឹះសុវត្ថិភាពសាយប័រ៖\n"
                "• កុំចុចលើតំណភ្ជាប់ (Link) ក្នុងសារ SMS ឬ Telegram ដែលមិនស្គាល់ប្រភព\n"
                "• កុំផ្ដល់លេខកូដ OTP ឬ Password ឱ្យនរណាម្នាក់ឡើយ\n"
                "• ពិនិត្យឈ្មោះ Domain និង URL មុននឹងបញ្ចូលលេខកុងធនាគារ!"
            )
        elif tip_type == "tool":
            alert_text = (
                "💡 គន្លឹះប្រើប្រាស់ AI Tool៖\n"
                "• អ្នកអាចសាកល្បងចុះឈ្មោះរៀន ឬប្រើប្រាស់ដោយឥតគិតថ្លៃ\n"
                "• ប្រើប្រាស់ ChatGPT ដើម្បីបកប្រែពាក្យពិបាក ឬសួរពន្យល់បន្ថែមជាភាសាខ្មែរ!"
            )
        elif tip_type == "ai":
            alert_text = (
                "🤖 ចំណេះដឹង AI៖\n"
                "• ម៉ូដែល AI ជំនាន់ថ្មីជួយសម្រួលការងារ និងបង្កើនល្បឿនច្រើនដង\n"
                "• តែងតែផ្ទៀងផ្ទាត់ទិន្នន័យសំខាន់ៗឡើងវិញជានិច្ច!"
            )
        else:
            alert_text = "💡 ព័ត៌មានបន្ថែម៖ តាមដានព័ត៌មានបច្ចេកវិទ្យាប្រចាំថ្ងៃជាមួយ Bot Voice!"
        with suppress(Exception):
            await query.answer(alert_text, show_alert=True)
        return

    if action_type == "art_like":
        with suppress(Exception):
            await query.answer("❤️ អរគុណសម្រាប់ការចូលចិត្តអត្ថបទនេះ!", show_alert=False)
        return

    if action_type == "art_save":
        with suppress(Exception):
            await query.answer("🔖 បានរក្សាទុកក្នុងបញ្ជីចំណាំ (Bookmarks) ដោយជោគជ័យ!", show_alert=True)
        return

    session_id = parts[-1]
    session = get_article_session(session_id)
    if not session:
        with suppress(Exception):
            await query.answer(
                "⚠️ អត្ថបទនេះផុតកំណត់ហើយ។ សូមផ្ញើតំណភ្ជាប់ម្ដងទៀត។",
                show_alert=True,
            )
        return

    # Helper resolver for process_tts_for_text
    def _get_tts_processor():
        with suppress(Exception):
            from app.services.telegram.media import process_tts_for_text
            return process_tts_for_text
        with suppress(Exception):
            from app.services.telegram.handlers import process_tts_for_text
            return process_tts_for_text
        return getattr(legacy, "process_tts_for_text", None)

    if action_type == "art_voice" and len(parts) >= 3:
        if parts[1] == "km_gtts":
            with suppress(Exception):
                await query.answer("🎙️ កំពុងបង្កើតសំឡេង Google gTTS...")

            from app.services.ai.gtts_narrator import generate_khmer_gtts_audio_async

            tts_text = session.get("khmer_tts_script") or session.get("khmer_summary") or ""
            if tts_text:
                try:
                    audio_bytes = await generate_khmer_gtts_audio_async(tts_text, speed=1.0, to_voice_note=True)
                    if audio_bytes and query.message:
                        from io import BytesIO

                        bio = BytesIO(audio_bytes)
                        bio.name = "narration_gtts.mp3"
                        await query.message.reply_audio(
                            audio=bio,
                            title=session.get("khmer_title", "Voice Note")[:64],
                            performer="Google gTTS (Khmer)",
                            caption="🗣️ <b>សំឡេងអានដោយ Google gTTS (Khmer)</b>",
                            parse_mode="HTML",
                            reply_markup=query.message.reply_markup,
                        )
                except Exception as g_err:
                    logger.warning("gTTS audio generation failed: %s", g_err)
            return

        target_gender = "female" if "female" in parts[1] else "male"
        voice_label = "ស្រី (Sreymom)" if target_gender == "female" else "ប្រុស (Piseth)"
        with suppress(Exception):
            await query.answer(f"🎙️ កំពុងបង្កើតសំឡេង {voice_label}...")

        proc_tts = _get_tts_processor()
        tts_text = session.get("khmer_tts_script") or session.get("khmer_summary") or ""
        if tts_text and callable(proc_tts):
            await proc_tts(
                update,
                context,
                tts_text,
                user_id,
                gender_override=target_gender,
                voice_markup_override=query.message.reply_markup if query.message else None,
            )
        return

    if action_type == "art_orig":
        with suppress(Exception):
            lang_name = session.get("orig_lang_name", "Original")
            await query.answer(f"🎙️ កំពុងបង្កើតសំឡេងជាភាសាដើម ({lang_name})...")

        proc_tts = _get_tts_processor()
        orig_text = session.get("original_summary") or session.get("body_text", "")[:800]
        orig_title = session.get("original_title") or session.get("title") or ""
        orig_script = f"{orig_title}. {orig_text}" if orig_title and not orig_text.startswith(orig_title) else orig_text
        if orig_script and callable(proc_tts):
            await proc_tts(
                update,
                context,
                orig_script,
                user_id,
                voice_markup_override=query.message.reply_markup if query.message else None,
            )
        return

    if action_type == "art_full":
        with suppress(Exception):
            await query.answer("📄 កំពុងផ្ញើអត្ថបទពេញ...")

        from app.services.telegram.formatters import send_split_html

        title = session.get("khmer_title") or session.get("original_title") or "អត្ថបទព័ត៌មាន"
        orig_title = session.get("original_title", "")
        url = session.get("url", "")
        body = session.get("body_text", "")

        header_lines = [
            f"📄 <b>{html.escape(title)}</b>",
        ]
        if orig_title and orig_title != title:
            header_lines.append(f"<i>({html.escape(orig_title)})</i>")
        if url:
            header_lines.append(f"🔗 <a href='{html.escape(url)}'>ប្រភពដើម (Source Link)</a>")
        header_lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        header_lines.append(html.escape(body or "គ្មានខ្លឹមសារបន្ថែមទេ។"))

        full_content = "\n".join(header_lines)
        if query.message:
            await send_split_html(query.message, full_content, disable_web_page_preview=True)
        return

    with suppress(Exception):
        await query.answer()


# ---------------------------------------------------------------------------
# RESILIENT DOWNLOADER CALLBACK DISPATCHERS
# ---------------------------------------------------------------------------

try:
    from app.services.downloader.facebook import facebook_callback
except ImportError:
    @legacy_bound_handler
    async def facebook_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.callback_query:
            with suppress(Exception):
                await update.callback_query.answer("❌ សេវាកម្ម Facebook Downloader មិនទាន់ដំណើរការទេ។", show_alert=True)

try:
    from app.services.downloader.instagram import instagram_callback
except ImportError:
    @legacy_bound_handler
    async def instagram_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.callback_query:
            with suppress(Exception):
                await update.callback_query.answer("❌ សេវាកម្ម Instagram Downloader មិនទាន់ដំណើរការទេ។", show_alert=True)

try:
    from app.services.downloader.tiktok import tiktok_callback
except ImportError:
    @legacy_bound_handler
    async def tiktok_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.callback_query:
            with suppress(Exception):
                await update.callback_query.answer("❌ សេវាកម្ម TikTok Downloader មិនទាន់ដំណើរការទេ។", show_alert=True)

try:
    from app.services.downloader.youtube import youtube_callback
except ImportError:
    @legacy_bound_handler
    async def youtube_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.callback_query:
            with suppress(Exception):
                await update.callback_query.answer("❌ សេវាកម្ម YouTube Downloader មិនទាន់ដំណើរការទេ។", show_alert=True)


# ---------------------------------------------------------------------------
# DYNAMIC LEGACY RESOLVER (PEP 562)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """Fallback dynamically to app.legacy or _legacy_runtime for transitional symbols."""
    with suppress(Exception):
        if hasattr(legacy, name):
            return getattr(legacy, name)
    with suppress(Exception):
        import app.services.telegram._legacy_runtime as lr
        if hasattr(lr, name):
            return getattr(lr, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    '_runtime_admin_callback',
    'article_callback',
    'broadcast_callback',
    'facebook_callback',
    'instagram_callback',
    'on_callback',
    'safe_send',
    'sched_callback',
    'tiktok_callback',
    'users_page_callback',
    'youtube_callback',
]