"""Extracted Telegram handler implementations.

These are live runtime handlers; app.legacy now contains compatibility wrappers only.
"""

from __future__ import annotations

import asyncio
import time

# Transitional V4.1 modules bind remaining legacy helpers at runtime.
# ruff: noqa: F821
from app.services.telegram._legacy_runtime import legacy_bound_handler
from app.services.telegram.workloads import WorkloadBusy, run_telegram_workload


@legacy_bound_handler
async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None or msg is None:
        return

    if _is_admin(user_id):
        if await _handle_admin_welcome_photo(update, context):
            return
        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_EDIT_WAIT_PHOTO:
            await _handle_sched_edit_photo(update, context)
            return
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                reply = (
                    f"✅ បានផ្ញើរូបភាពទៅអ្នកប្រើប្រាស់ <code>{target_id}</code>។"
                    if ok else
                    f"❌ អ្នកប្រើប្រាស់ <code>{target_id}</code> បានបិទបូត។"
                )
                await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
                if not ok:
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ បានផ្ញើរូបភាពទៅអ្នកគ្រប់គ្រង។"))
        return

    if not await _ensure_user_allowed(update, context, "ocr_enabled", "អានអត្ថបទពីរូបភាព"):
        return
    if not _ocr_configured():
        await safe_send(lambda: msg.reply_text(_ocr_status_for_user()))
        return
    if await _check_cooldown(msg, user_id):
        return

    _metric_inc("ocr")
    sync_user_data(user)
    uname = user.username or user.first_name or str(user_id)
    progress = await TelegramProgress.start(
        bot=context.bot,
        chat_id=msg.chat_id,
        reply_target=msg,
        title="កំពុងអានអត្ថបទពីរូបភាព",
        percent=5,
        stage="កំពុងពិនិត្យរូបភាព",
        detail="កំពុងរៀបចំឯកសាររូបភាព។",
    )
    img_path: str | None = None
    try:
        await progress.update(12, "កំពុងទាញយករូបភាព", "កំពុងទទួលរូបភាពគុណភាពខ្ពស់ពី Telegram។", force=True)
        img_path = _make_temp_img(suffix=".jpg")
        tg_file = await safe_send(lambda: context.bot.get_file(msg.photo[-1].file_id))
        if not tg_file:
            raise RuntimeError("Could not download photo.")
        await tg_file.download_to_drive(img_path)

        await progress.update(35, "បានទាញយករូបភាព", "កំពុងស្គាល់ប្រភេទរូបភាព។", force=True)
        mime_type = _detect_image_mime(img_path)

        await progress.update(50, "កំពុងស្វែងរកអត្ថបទ", "រូបភាពកំពុងត្រូវបានផ្ញើទៅម៉ាស៊ីន OCR។", force=True)
        ocr_text = await run_telegram_workload(
            "ocr", lambda: ocr_image(img_path, mime_type=mime_type)
        )
        if not ocr_text or ocr_text.upper() == "NOTEXT":
            await progress.finish("🖼️ រូបភាពនេះមិនមានអត្ថបទដែលអាចអានបានទេ។")
            return

        await progress.update(85, "បានអានអត្ថបទរួច", f"រកឃើញ {len(ocr_text)} តួអក្សរ។", force=True)
        record_turn(user_id, "user", f"[Image OCR]: {ocr_text[:500]}")
        lang_key = _detect_lang(ocr_text)
        lang_flag, lang_name = _language_display(lang_key)
        header = f"🔍 <b>អត្ថបទពីរូបភាព {lang_flag} {html.escape(lang_name)}</b>\n\n"
        plain_pages = _paginate_plain(ocr_text, limit=max(500, TELE_MSG_LIMIT - len(header) - 64))
        if not plain_pages:
            await progress.fail("❌ មិនអាចរៀបចំអត្ថបទដែលបានអានទេ។")
            return

        first_page = header + html.escape(plain_pages[0])
        await progress.finish(first_page, parse_mode="HTML")
        result_id = progress.message_id or int(msg.message_id)
        save_text_cache(
            result_id,
            ocr_text,
            chat_id=msg.chat_id,
            user_id=user_id,
            username=uname,
        )
        if progress.message is not None:
            await safe_send(lambda: progress.message.edit_reply_markup(
                reply_markup=get_ocr_confirm_kb(result_id)
            ))

        total_pages = len(plain_pages)
        for idx, plain_page in enumerate(plain_pages[1:], 2):
            page_body = (
                f"🔍 <b>អត្ថបទពីរូបភាព — ទំព័រ {idx}/{total_pages}</b>\n\n"
                + html.escape(plain_page)
            )
            await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            await asyncio.sleep(0.15)
    except WorkloadBusy:
        _metric_inc("busy_rejected")
        await progress.fail("⏳ សេវា OCR កំពុងរវល់។ សូមសាកម្ដងទៀតបន្តិចក្រោយ។")
    except Exception as exc:
        err_msg = str(exc) or repr(exc)
        if _is_expected_ocr_outage_error(err_msg):
            logger.warning("on_photo OCR unavailable: %s: %s", type(exc).__name__, err_msg[:700])
        else:
            logger.error("on_photo OCR error: %s: %r", type(exc).__name__, exc, exc_info=True)
        if _is_dns_or_network_error(err_msg):
            user_msg = "❌ មិនអាចភ្ជាប់ទៅសេវា OCR បានទេ។ សូមសាកម្ដងទៀតបន្តិចក្រោយ។"
        elif "temporarily disabled" in err_msg.lower():
            user_msg = "⚠️ សេវា OCR ត្រូវបានផ្អាកបណ្ដោះអាសន្ន។ សូមសាកម្ដងទៀតក្រោយពេលខ្លី។"
        else:
            user_msg = "❌ មិនអាចអានអត្ថបទពីរូបភាពនេះបានទេ។ សូមប្រើរូបភាពច្បាស់ជាងនេះ ហើយសាកម្ដងទៀត។"
        await progress.fail(user_msg)
    finally:
        if img_path:
            _cleanup(img_path)


@legacy_bound_handler
async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    if msg is None or user is None or msg.voice is None:
        return
    user_id = int(user.id)

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ បានផ្ញើសារសំឡេងទៅអ្នកប្រើប្រាស់ <code>{target_id}</code>។"
                if ok else
                f"❌ អ្នកប្រើប្រាស់ <code>{target_id}</code> បានបិទបូត។"
            )
            await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)
                context.user_data.pop("chat_state", None)
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ បានផ្ញើសារសំឡេងទៅអ្នកគ្រប់គ្រង។"))
        return

    if not await _ensure_user_allowed(update, context, "voice_transcribe_enabled", "បម្លែងសំឡេងទៅជាអត្ថបទ"):
        return
    if not _gemini:
        await safe_send(lambda: msg.reply_text("❌ សេវាបម្លែងសំឡេងទៅអត្ថបទមិនទាន់បានបើកទេ។"))
        return
    if msg.voice.file_size and msg.voice.file_size > MAX_VOICE_BYTES:
        await safe_send(lambda: msg.reply_text("❌ ឯកសារសំឡេងធំពេក។ អតិបរមា 20 MB។"))
        return
    if await _check_cooldown(msg, user_id):
        return

    _metric_inc("voice")
    sync_user_data(user)
    progress = await TelegramProgress.start(
        bot=context.bot,
        chat_id=msg.chat_id,
        reply_target=msg,
        title="កំពុងបម្លែងសារសំឡេងទៅជាអត្ថបទ",
        percent=5,
        stage="កំពុងពិនិត្យសារសំឡេង",
        detail=f"រយៈពេល {float(msg.voice.duration or 0):g} វិនាទី។",
    )
    ogg_path = _make_temp_ogg()
    try:
        await progress.update(15, "កំពុងទាញយកសារសំឡេង", "កំពុងទទួលឯកសារពី Telegram។", force=True)
        voice_file = await safe_send(lambda: context.bot.get_file(msg.voice.file_id))
        if not voice_file:
            raise RuntimeError("Could not get voice file")
        await voice_file.download_to_drive(ogg_path)

        await progress.update(40, "បានទាញយកសារសំឡេង", "កំពុងផ្ញើទៅម៉ាស៊ីនស្គាល់សំឡេង។", force=True)
        transcript = await run_telegram_workload(
            "transcribe", lambda: transcribe_voice(ogg_path)
        )
        if not transcript:
            await progress.fail("❌ មិនអាចស្គាល់អត្ថបទនៅក្នុងសារសំឡេងនេះបានទេ។")
            return

        await progress.update(85, "បានស្គាល់អត្ថបទ", f"រកឃើញ {len(transcript)} តួអក្សរ។", force=True)
        record_turn(user_id, "user", f"[Voice Transcript]: {transcript[:500]}")
        detected_lang = _detect_lang(transcript)
        lang_flag, lang_name = _language_display(detected_lang)
        header = (
            f"📝 <b>អត្ថបទពីសារសំឡេង</b> {lang_flag} "
            f"{html.escape(lang_name)}\n\n"
        )
        pages = _paginate_plain(transcript, limit=max(500, TELE_MSG_LIMIT - len(header) - 64))
        if not pages:
            raise RuntimeError("Could not paginate transcript")
        await progress.finish(header + html.escape(pages[0]), parse_mode="HTML")
        result_id = progress.message_id or int(msg.message_id)
        save_text_cache(
            result_id,
            transcript,
            chat_id=msg.chat_id,
            user_id=user_id,
            username=user.username or user.first_name,
        )
        if progress.message is not None:
            await safe_send(lambda: progress.message.edit_reply_markup(
                reply_markup=get_transcription_kb(result_id)
            ))
        total_pages = len(pages)
        for idx, page in enumerate(pages[1:], 2):
            body = f"📝 <b>អត្ថបទពីសារសំឡេង — ទំព័រ {idx}/{total_pages}</b>\n\n{html.escape(page)}"
            await safe_send(lambda b=body: msg.reply_text(b, parse_mode="HTML"))
            await asyncio.sleep(0.15)
    except WorkloadBusy:
        _metric_inc("busy_rejected")
        await progress.fail("⏳ សេវាបម្លែងសំឡេងកំពុងរវល់។ សូមសាកម្ដងទៀតបន្តិចក្រោយ។")
    except Exception as exc:
        logger.error("on_voice error: %s", exc, exc_info=True)
        await progress.fail("❌ មិនអាចបម្លែងសារសំឡេងទៅជាអត្ថបទបានទេ។ សូមសាកម្ដងទៀត។")
    finally:
        _cleanup(ogg_path)


@legacy_bound_handler
async def on_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Convert and/or transcribe audio using one progress message."""
    msg = update.message
    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None or msg is None:
        return

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ បានផ្ញើឯកសារទៅអ្នកប្រើប្រាស់ <code>{target_id}</code>។"
                if ok else
                f"❌ អ្នកប្រើប្រាស់ <code>{target_id}</code> បានបិទបូត។"
            )
            await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)
                context.user_data.pop("chat_state", None)
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ បានផ្ញើឯកសារទៅអ្នកគ្រប់គ្រង។"))
        return

    doc = msg.document
    audio = msg.audio
    if doc is not None:
        filename = doc.file_name or ""
        mime_type = doc.mime_type or ""
        file_id = doc.file_id
        file_size = int(doc.file_size or 0)
        if _is_subtitle_file(filename) or not _is_audio_file(filename, mime_type):
            await on_document(update, context)
            return
    elif audio is not None:
        filename = audio.file_name or ""
        mime_type = audio.mime_type or ""
        file_id = audio.file_id
        file_size = int(audio.file_size or 0)
    else:
        return

    if not await _ensure_user_allowed(update, context):
        return
    settings, _settings_status = await get_bot_settings_async()
    convert_enabled = _setting_bool_from(settings, "audio_to_voice_enabled", True)
    transcribe_enabled = _setting_bool_from(settings, "audio_transcribe_enabled", True)
    if not convert_enabled and not transcribe_enabled:
        _metric_inc("disabled_hits")
        await safe_send(lambda: msg.reply_text(
            "⚠️ មុខងារបម្លែងឯកសារអូឌីយ៉ូត្រូវបានបិទបណ្ដោះអាសន្នដោយអ្នកគ្រប់គ្រង។"
        ))
        return
    if file_size > MAX_AUDIO_FILE_BYTES:
        await safe_send(lambda: msg.reply_text(
            f"❌ ឯកសារអូឌីយ៉ូធំពេក។ អតិបរមា {MAX_AUDIO_FILE_BYTES // 1024 // 1024} MB។"
        ))
        return
    if await _check_cooldown(msg, user_id):
        return

    sync_user_data(user)
    uname = user.username or user.first_name or str(user_id)
    ext = os.path.splitext(filename)[1].lower() if filename else ".mp3"
    if ext not in _AUDIO_EXTENSIONS:
        ext = ".mp3"
    gemini_mime = _audio_mime_for_gemini(filename, mime_type)
    audio_path: str | None = None
    voice_path: str | None = None
    voice_sent = False
    transcript = ""
    conversion_error: Exception | None = None
    transcription_error: Exception | None = None

    progress = await TelegramProgress.start(
        bot=context.bot,
        chat_id=msg.chat_id,
        reply_target=msg,
        title="កំពុងដំណើរការឯកសារអូឌីយ៉ូ",
        percent=5,
        stage="កំពុងពិនិត្យឯកសារ",
        detail=filename or "ឯកសារអូឌីយ៉ូ",
    )
    try:
        await progress.update(12, "កំពុងទាញយកឯកសារ", "កំពុងទទួលទិន្នន័យពី Telegram។", force=True)
        tg_file = await safe_send(lambda: context.bot.get_file(file_id))
        if not tg_file:
            raise RuntimeError("Could not download audio file.")
        audio_path = await _download_telegram_file_to_temp_path(
            tg_file,
            MAX_AUDIO_FILE_BYTES,
            suffix=ext,
        )
        await progress.update(30, "បានទាញយកឯកសារ", "កំពុងរៀបចំប្រតិបត្តិការដែលបានបើក។", force=True)

        if convert_enabled:
            voice_path = _make_temp_ogg()
            try:
                await progress.update(38, "កំពុងបម្លែងទៅសារសំឡេង", "កំពុងបម្លែងទៅទម្រង់ OGG/Opus។", force=True)
                voice_bytes = await run_telegram_workload(
                    "audio",
                    lambda: _convert_uploaded_audio_to_telegram_voice(audio_path, voice_path),
                )
                await progress.update(52, "បានបម្លែងសំឡេង", "កំពុងផ្ញើសារសំឡេង។", force=True)
                display_name = html.escape((filename or "audio")[:80])
                sent_voice = await safe_send(lambda vb=voice_bytes, dn=display_name: msg.reply_voice(
                    voice=vb,
                    caption=f"🎙️ <b>សារសំឡេង</b> — <code>{dn}</code>",
                    parse_mode="HTML",
                ))
                voice_sent = sent_voice is not None
                if voice_sent:
                    _metric_inc("audio_to_voice")
                    await progress.update(60, "បានផ្ញើសារសំឡេង", "ការបម្លែងទៅសារសំឡេងបានជោគជ័យ។", force=True)
            except WorkloadBusy as exc:
                conversion_error = exc
                _metric_inc("busy_rejected")
                await progress.update(
                    60,
                    "សេវាបម្លែងសំឡេងកំពុងរវល់",
                    "កំពុងបន្តទៅការបម្លែងអត្ថបទ ប្រសិនបើមុខងារនេះបានបើក។",
                    force=True,
                )
            except Exception as exc:
                conversion_error = exc
                logger.error("Audio-to-voice conversion failed: %s", exc, exc_info=True)
                await progress.update(60, "មិនអាចបម្លែងទៅសារសំឡេង", "កំពុងព្យាយាមបម្លែងទៅអត្ថបទ ប្រសិនបើមុខងារនេះបានបើក។", force=True)


        if transcribe_enabled:
            if _gemini is None:
                transcription_error = RuntimeError("Gemini API is not active.")
            else:
                try:
                    await progress.update(68, "កំពុងបម្លែងសំឡេងទៅជាអត្ថបទ", "កំពុងផ្ញើអូឌីយ៉ូទៅម៉ាស៊ីនស្គាល់សំឡេង។", force=True)
                    transcript = await run_telegram_workload(
                        "transcribe",
                        lambda: transcribe_audio_file(audio_path, gemini_mime),
                    )
                    if not transcript:
                        raise RuntimeError("No transcript was found.")
                    _metric_inc("audio")
                    record_turn(user_id, "user", f"[Audio File Transcript]: {transcript[:500]}")
                    await progress.update(88, "បានបម្លែងទៅអត្ថបទ", f"រកឃើញ {len(transcript)} តួអក្សរ។", force=True)
                except WorkloadBusy as exc:
                    transcription_error = exc
                    _metric_inc("busy_rejected")
                    logger.info("Audio transcription admission rejected: %s", exc)
                except Exception as exc:
                    transcription_error = exc
                    logger.error("Audio transcription failed: %s", exc, exc_info=True)

        if transcript:
            detected_lang = _detect_lang(transcript)
            lang_flag, lang_name = _language_display(detected_lang)
            fname_display = html.escape(filename[:50]) if filename else "audio"
            conversion_note = (
                "✅ បានបង្កើតសារសំឡេងរួចរាល់។"
                if voice_sent else
                "⚠️ មិនអាចបង្កើតសារសំឡេងបាន ប៉ុន្តែបានបម្លែងទៅអត្ថបទ។"
            )
            header = (
                f"🎵 <b>អត្ថបទពីឯកសារអូឌីយ៉ូ</b> {lang_flag} "
                f"{html.escape(lang_name)} — <code>{fname_display}</code>\n"
                f"{conversion_note}\n\n"
            )
            pages = _paginate_plain(transcript, limit=max(500, TELE_MSG_LIMIT - len(header) - 64))
            if not pages:
                raise RuntimeError("Could not paginate transcript.")
            await progress.finish(header + html.escape(pages[0]), parse_mode="HTML")
            result_id = progress.message_id or int(msg.message_id)
            save_text_cache(
                result_id,
                transcript,
                chat_id=msg.chat_id,
                user_id=user_id,
                username=uname,
            )
            if progress.message is not None:
                await safe_send(lambda: progress.message.edit_reply_markup(
                    reply_markup=get_audio_file_kb(result_id)
                ))
            total_pages = len(pages)
            for idx, page in enumerate(pages[1:], 2):
                body = f"🎵 <b>អត្ថបទពីអូឌីយ៉ូ — ទំព័រ {idx}/{total_pages}</b>\n\n{html.escape(page)}"
                await safe_send(lambda b=body: msg.reply_text(b, parse_mode="HTML"))
                await asyncio.sleep(0.15)
            return

        if voice_sent:
            if transcribe_enabled and transcription_error is not None:
                await progress.finish(
                    "⚠️ បានបង្កើតសារសំឡេងរួចរាល់ ប៉ុន្តែមិនអាចបម្លែងអូឌីយ៉ូទៅជាអត្ថបទបានទេ។"
                )
            else:
                await progress.finish("✅ បានបម្លែង និងផ្ញើសារសំឡេងរួចរាល់។", delete_after_s=5.0)
            return

        logger.warning(
            "Audio processing produced no output conversion_error=%s transcription_error=%s",
            conversion_error,
            transcription_error,
        )
        await progress.fail(
            "❌ មិនអាចដំណើរការឯកសារអូឌីយ៉ូនេះបានទេ។ សូមពិនិត្យថាឯកសារមានសំឡេង និងប្រើទម្រង់ MP3, WAV, OGG ឬ FLAC។"
        )
    except WorkloadBusy:
        _metric_inc("busy_rejected")
        await progress.fail("⏳ សេវាអូឌីយ៉ូកំពុងរវល់។ សូមសាកម្ដងទៀតបន្តិចក្រោយ។")
    except ValueError as exc:
        logger.warning("Audio upload rejected: %s", exc)
        await progress.fail("❌ ឯកសារអូឌីយ៉ូមិនត្រឹមត្រូវ ឬធំពេក។ សូមជ្រើសឯកសារថ្មី។")
    except Exception as exc:
        logger.error("on_audio_file error: %s", exc, exc_info=True)
        await progress.fail("❌ មានបញ្ហាក្នុងការទាញយក ឬដំណើរការឯកសារអូឌីយ៉ូ។ សូមសាកម្ដងទៀត។")
    finally:
        if audio_path:
            _cleanup(audio_path)
        if voice_path:
            _cleanup(voice_path)


@legacy_bound_handler
async def on_any_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg     = update.message
    user    = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok    = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok else
                f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)
                context.user_data.pop("chat_state", None)
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text('✅ បានផ្ញើទៅអ្នកគ្រប់គ្រង។'))


@legacy_bound_handler
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return
    text = msg.text
    user = update.effective_user
    if not user:
        return
    user_id = int(user.id)

    if _is_admin(user_id):
        if await _handle_admin_welcome_text(update, context):
            return
        if await _handle_admin_button_text(update, context):
            return
        if await _handle_feature_request_admin_reply_text(update, context):

            return
        if await _handle_runtime_admin_text(update, context):
            return
        if await _handle_user_search_text(update, context):
            return
        if await _handle_admin_report_day_text(update, context):
            return
        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        if sched_state == SCHED_WAIT_TIME:
            await _handle_sched_datetime(update, context)
            return
        if sched_state == SCHED_EDIT_WAIT_TIME:
            await _handle_sched_edit_time(update, context)
            return
        if sched_state == SCHED_EDIT_WAIT_TEXT:
            await _handle_sched_edit_text(update, context)
            return
        if sched_state == SCHED_EDIT_WAIT_PHOTO:
            await safe_send(lambda: msg.reply_text("⚠️ សូមផ្ញើរូបភាពថ្មី ឬ /cancel។"))
            return
        if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                if ok:
                    await safe_send(lambda: msg.reply_text(
                        f"✅ បានផ្ញើទៅអ្នកប្រើប្រាស់ <code>{target_id}</code>។",
                        parse_mode="HTML",
                    ))
                else:
                    await safe_send(lambda: msg.reply_text(
                        f"❌ អ្នកប្រើប្រាស់ <code>{target_id}</code> បានបិទបូត។ វគ្គជជែកត្រូវបានបញ្ចប់។",
                        parse_mode="HTML",
                    ))
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    if await _handle_feature_request_user_text(update, context):
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ សាររបស់អ្នកបានផ្ញើទៅអ្នកគ្រប់គ្រង។"))
        return

    if not await _ensure_user_allowed(update, context, "tts_enabled", "បម្លែងអត្ថបទទៅជាសំឡេង"):
        return
    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return
    stripped = text.strip()
    if not stripped:
        return
    if len(stripped) > MAX_INPUT_CHARS:
        await safe_send(lambda: msg.reply_text(
            f"❌ អត្ថបទវែងពេក។ អតិបរមា {MAX_INPUT_CHARS} តួអក្សរ។\n"
            f"អ្នកបានផ្ញើ {len(stripped)} តួអក្សរ។"
        ))
        return
    if await _check_cooldown(msg, user_id):
        return
    if not _reserve_tts_request(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return

    try:
        await process_tts_for_text(update, context, stripped, user_id)
    except Exception as err:
        logger.warning("process_tts_for_text failed for user %s: %s", user_id, err)


@legacy_bound_handler
async def process_tts_for_text(update: Update, context: ContextTypes.DEFAULT_TYPE, stripped: str, user_id: int):
    """Core TTS processing extracted from on_text so it can be called programmatically."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    try:
        _metric_inc("tts", user_id=user_id)
        sync_user_data(user)
        progress = await TelegramProgress.start(
            bot=context.bot,
            chat_id=msg.chat_id,
            reply_target=msg,
            title="កំពុងបម្លែងអត្ថបទទៅជាសំឡេង",
            percent=5,
            stage="កំពុងពិនិត្យអត្ថបទ",
            detail=f"មាន {len(stripped)} តួអក្សរ។",
            minimal=True,
        )
    except BaseException:
        _release_tts_request(user_id)
        raise
    file_path: str | None = None
    try:
        await progress.update(12, "កំពុងអានការកំណត់របស់អ្នក", "កំពុងជ្រើសសំឡេង ល្បឿន និងម៉ូដែល។", force=True)
        loop = asyncio.get_running_loop()
        prefs, tts_text = await asyncio.gather(
            get_user_prefs_async(user_id),
            resolve_tts_text(user_id, stripped, loop),
        )
        gender = prefs["gender"]
        speed = prefs["speed"]
        tts_model = prefs.get("tts_model", "auto")
        tts_text = tts_text.strip() or stripped
        model_key = _normalize_tts_model(tts_model)
        model_label = TTS_MODEL_OPTIONS.get(model_key, TTS_MODEL_OPTIONS["auto"])[0]
        await progress.update(
            25,
            "បានរៀបចំអត្ថបទ និងការកំណត់",
            f"ម៉ូដែល៖ {model_label} • អត្ថបទ {len(tts_text)} តួអក្សរ។",
            force=True,
        )

        lock = _get_user_lock(user_id)
        async with lock:
            if len(tts_text) > TTS_SINGLE_VOICE_MAX_CHARS:
                uname = user.username or user.first_name or str(user_id)
                try:
                    sent_count, failed_count = await asyncio.wait_for(
                        _deliver_paged_tts(
                            chat_id=msg.chat_id,
                            bot=context.bot,
                            text=tts_text,
                            gender=gender,
                            speed=speed,
                            user_id=user_id,
                            username=uname,
                            tts_model=tts_model,
                            progress=progress,
                            progress_start=25,
                            progress_end=95,
                        ),
                        timeout=900.0, # 15 minutes for long documents
                    )
                except TimeoutError as exc:
                    raise RuntimeError("ប្រតិបត្តិការចំណាយពេលយូរពេក (Timeout)។") from exc
                record_turn(user_id, "user", stripped)
                record_turn(user_id, "assistant", tts_text[:CONV_CONTEXT_MAX_CHARS])
                _set_last_tts(user_id)
                if failed_count:
                    await progress.finish(
                        f"⚠️ បានផ្ញើសំឡេង {sent_count} ផ្នែក ប៉ុន្តែមាន {failed_count} ផ្នែកមិនបានជោគជ័យ។"
                    )
                else:
                    await progress.finish(
                        f"✅ បានបង្កើត និងផ្ញើសំឡេងរួចរាល់ ({sent_count} ផ្នែក)។",
                        delete_after_s=5.0,
                    )
                return

            file_path = _make_temp_ogg()
            await progress.update(35, "កំពុងបង្កើតសំឡេង", f"កំពុងប្រើ {model_label}។", force=True)
            generation_started = time.perf_counter()
            try:
                audio_bytes = await asyncio.wait_for(
                    generate_user_voice_limited(
                        tts_text,
                        gender,
                        speed,
                        file_path,
                        tts_model,
                        user_id=user_id,
                        bot=context.bot,
                        chat_id=msg.chat_id,
                        progress=progress,
                    ),
                    timeout=180.0,
                )
            except TimeoutError as exc:
                raise RuntimeError("ការបង្កើតសំឡេងចំណាយពេលយូរពេក សូមព្យាយាមម្ដងទៀត។") from exc
            _record_admin_usage(
                user_id,
                "tts_generation",
                duration_ms=(time.perf_counter() - generation_started) * 1_000,
            )
            await progress.update(88, "បានបង្កើតសំឡេង", "កំពុងផ្ញើសារសំឡេងទៅអ្នក។", force=True)
            sent_msg = await safe_send(lambda ab=audio_bytes: msg.reply_voice(
                voice=io.BytesIO(ab),
                caption=f"🗣️ {BOT_TAG}",
                reply_markup=get_main_kb(gender, tts_model),
            ))
            if sent_msg is None:
                raise RuntimeError("Telegram មិនអាចផ្ញើសារសំឡេងបាន។")
            save_text_cache(
                sent_msg.message_id,
                tts_text,
                chat_id=msg.chat_id,
                user_id=user_id,
                username=user.username or user.first_name,
            )
            set_last_tts_text(user_id, tts_text)
            record_turn(user_id, "user", stripped)
            record_turn(user_id, "assistant", tts_text)
            _set_last_tts(user_id)
            await progress.finish("✅ បានបង្កើត និងផ្ញើសំឡេងរួចរាល់។", delete_after_s=5.0)
    except Exception as exc:
        logger.error("on_text TTS error: %s", exc, exc_info=True)
        await progress.fail(_tts_user_error_message(exc))
    finally:
        _release_tts_request(user_id)
        if file_path:
            _cleanup(file_path)


__all__ = [
    'on_photo',
    'on_voice',
    'on_audio_file',
    'on_any_media',
    'on_text',
    'process_tts_for_text'
]
