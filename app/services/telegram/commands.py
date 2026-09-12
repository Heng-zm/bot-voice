"""Extracted Telegram handler implementations.

These are live runtime handlers; app.legacy now contains compatibility wrappers only.
"""

from __future__ import annotations

import html
import re
import threading

# Transitional V4.1 modules bind remaining legacy helpers at runtime.
# ruff: noqa: F821
from app.services.telegram._legacy_runtime import legacy_bound_handler, safe_send


@legacy_bound_handler
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sync_user_data(update.effective_user)
        if not await _ensure_user_allowed(update, context):
            return
        await _send_welcome_message(update.message)
    except Exception as e:
        logger.error(f"on_start error: {e}")


@legacy_bound_handler
async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return
    if getattr(update, "callback_query", None):
        with suppress(Exception):
            await update.callback_query.answer()
    help_text = (
        "📖 <b>សៀវភៅណែនាំរបៀបប្រើប្រាស់ Bot Voice</b> 🎙️\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "1️⃣ <b>បម្លែងអត្ថបទទៅជាសំឡេង (TTS):</b>\n"
        "• គ្រាន់តែវាយអត្ថបទខ្មែរ ឬអន្តរជាតិ រួចផ្ញើមកកាន់ Bot\n"
        "• បូតនឹងបង្កើត Voice Note ជូនភ្លាមៗ\n\n"
        "2️⃣ <b>សួរ AI Assistant:</b>\n"
        "• វាយ <code>/ask សំណួររបស់អ្នក</code>\n"
        "• ឧទាហរណ៍៖ <code>/ask តើភ្នំពេញជារាជធានីនៃប្រទេសណា?</code>\n\n"
        "3️⃣ <b>បកប្រែជាភាសាខ្មែរ:</b>\n"
        "• វាយ <code>/translate អត្ថបទ</code>\n"
        "• ឧទាហរណ៍៖ <code>/translate Good morning, how are you?</code>\n\n"
        "4️⃣ <b>សង្ខេបអត្ថបទវែងៗ:</b>\n"
        "• វាយ <code>/summary អត្ថបទរបស់អ្នក</code>\n\n"
        "5️⃣ <b>អានអក្សរពីរូបភាព (OCR):</b>\n"
        "• ផ្ញើរូបភាពសៀវភៅ ឬឯកសារ មកកាន់ Bot\n\n"
        "6️⃣ <b>ការកំណត់សំឡេង & ម៉ូដែល:</b>\n"
        "• <code>/myprefs</code> — មើល និងកែប្រែការកំណត់សំឡេង\n"
        "• <code>/ttsmodel</code> — ជ្រើសរើសម៉ូដែលសំឡេង (Kiri, Gemini AI, Edge)\n"
        "• <code>/unlock</code> — ដោះសោររង់ចាំ\n"
        "• <code>/clear</code> — សម្អាតប្រវត្តិសន្ទនា\n\n"
        "7️⃣ <b>ឧបត្ថម្ភ & តារាងកិត្តិយស:</b>\n"
        "• <code>/donate</code> — ឧបត្ថម្ភកាហ្វេជួយទ្រទ្រង់ដំណើរការ Bot\n"
        "• <code>/donors</code> — មើលតារាងកិត្តិយសអ្នកឧបត្ថម្ភ\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "💬 <i>ផ្ញើសារ ឬសំណួររបស់អ្នកមកឥឡូវនេះបាន!</i>"
    )
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("⚙️ ការកំណត់ / Settings", callback_data="welcome_profile"),
         InlineKeyboardButton("🤖 ម៉ូដែល TTS", callback_data="show_tts_model")],
        [InlineKeyboardButton("📢 Channel", url="https://t.me/m11mmm112"),
         InlineKeyboardButton("☕ ឧបត្ថម្ភកាហ្វេ", callback_data="donate_menu")],
        [InlineKeyboardButton("🏆 តារាងកិត្តិយស (/donors)", callback_data="donate_halloffame")],
        [InlineKeyboardButton("🔙 ត្រឡប់ / Close", callback_data="welcome_back")],
    ])
    await safe_send(lambda: msg.reply_text(help_text, parse_mode="HTML", reply_markup=kb))


@legacy_bound_handler
async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Directly ask Gemini AI a question and hear the answer in voice."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    text = re.sub(r"^/ask(?:@\w+)?\s*", "", msg.text or "", flags=re.IGNORECASE).strip()
    if not text:
        await safe_send(lambda: msg.reply_text("💡 សូមសរសេរសំណួររបស់អ្នកតាមក្រោយ /ask (ឧ. /ask តើ AI គឺជាអ្វី?)"))
        return
    if await _check_cooldown(msg, user_id):
        return
    if not _reserve_tts_request(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return
    from app import legacy
    if getattr(legacy, "_gemini", None) is None:
        _release_tts_request(user_id)
        await safe_send(lambda: msg.reply_text("❌ Gemini API មិនទាន់បានបើកទេ។"))
        return
    await safe_send(lambda: msg.reply_chat_action("typing"))
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        from app.services.ai.gemini import (
            extract_gemini_text,
            generate_content_with_fallback,
        )
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")
        def _call_ai():
            return generate_content_with_fallback(
                legacy._gemini,
                contents=text,
                preferred_model=preferred,
            )
        resp = await loop.run_in_executor(None, _call_ai)
        ai_text = extract_gemini_text(resp)
        if not ai_text:
            _release_tts_request(user_id)
            await safe_send(lambda: msg.reply_text("⚠️ មិនអាចឆ្លើយបានទេ (អាចដោយសារគោលការណ៍សុវត្ថិភាព AI ឬគ្មានចម្លើយ)។"))
            return
        
        ai_header = f"🤖 <b>AI Assistant:</b>\n\n{html.escape(ai_text)}"
        await safe_send(lambda: msg.reply_text(ai_header, parse_mode="HTML"))
        await safe_send(lambda: context.bot.send_chat_action(chat_id=msg.chat_id, action="record_voice"))

        from app.services.telegram.media import process_tts_for_text
        await process_tts_for_text(update, context, ai_text, user_id)
    except Exception as exc:
        _release_tts_request(user_id)
        err_msg = f"❌ បរាជ័យ: {exc}"
        await safe_send(lambda: msg.reply_text(err_msg))


@legacy_bound_handler
async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Translate text to Khmer and hear it out loud."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    text = re.sub(r"^/translate(?:@\w+)?\s*", "", msg.text or "", flags=re.IGNORECASE).strip()
    if not text:
        await safe_send(lambda: msg.reply_text("💡 សូមបញ្ចូលអត្ថបទដើម្បីបកប្រែ (ឧ. /translate Hello world)"))
        return
    if await _check_cooldown(msg, user_id):
        return
    if not _reserve_tts_request(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return
    from app import legacy
    if getattr(legacy, "_gemini", None) is None:
        _release_tts_request(user_id)
        await safe_send(lambda: msg.reply_text("❌ Gemini API មិនទាន់បានបើកទេ។"))
        return
    await safe_send(lambda: msg.reply_chat_action("typing"))
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        from app.services.ai.gemini import (
            extract_gemini_text,
            generate_content_with_fallback,
        )
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")
        def _call_ai():
            return generate_content_with_fallback(
                legacy._gemini,
                contents=f"Translate the following text accurately and naturally into Khmer. Return only the translated text without extra explanation:\n\n{text}",
                preferred_model=preferred,
            )
        resp = await loop.run_in_executor(None, _call_ai)
        khmer_text = extract_gemini_text(resp)
        if not khmer_text:
            _release_tts_request(user_id)
            await safe_send(lambda: msg.reply_text("⚠️ មិនអាចបកប្រែបានទេ (អាចដោយសារគោលការណ៍សុវត្ថិភាព AI ឬគ្មានចម្លើយ)។"))
            return

        trans_header = f"🌐 <b>បកប្រែជាភាសាខ្មែរ:</b>\n\n{html.escape(khmer_text)}"
        await safe_send(lambda: msg.reply_text(trans_header, parse_mode="HTML"))
        await safe_send(lambda: context.bot.send_chat_action(chat_id=msg.chat_id, action="record_voice"))

        from app.services.telegram.media import process_tts_for_text
        await process_tts_for_text(update, context, khmer_text, user_id)
    except Exception as exc:
        _release_tts_request(user_id)
        err_msg = f"❌ បរាជ័យ: {exc}"
        await safe_send(lambda: msg.reply_text(err_msg))


@legacy_bound_handler
async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Summarize long text into bullet points and hear them out loud."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    text = re.sub(r"^/summary(?:@\w+)?\s*", "", msg.text or "", flags=re.IGNORECASE).strip()
    if not text:
        await safe_send(lambda: msg.reply_text("💡 សូមបញ្ចូលអត្ថបទវែងៗដើម្បីសង្ខេប (ឧ. /summary ...អត្ថបទ...)"))
        return
    if await _check_cooldown(msg, user_id):
        return
    if not _reserve_tts_request(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return
    from app import legacy
    if getattr(legacy, "_gemini", None) is None:
        _release_tts_request(user_id)
        await safe_send(lambda: msg.reply_text("❌ Gemini API មិនទាន់បានបើកទេ។"))
        return
    await safe_send(lambda: msg.reply_chat_action("typing"))
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        from app.services.ai.gemini import (
            extract_gemini_text,
            generate_content_with_fallback,
        )
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")
        def _call_ai():
            return generate_content_with_fallback(
                legacy._gemini,
                contents=f"Summarize the following text into clear, concise bullet points in Khmer:\n\n{text}",
                preferred_model=preferred,
            )
        resp = await loop.run_in_executor(None, _call_ai)
        summary_text = extract_gemini_text(resp)
        if not summary_text:
            _release_tts_request(user_id)
            await safe_send(lambda: msg.reply_text("⚠️ មិនអាចសង្ខេបបានទេ (អាចដោយសារគោលការណ៍សុវត្ថិភាព AI ឬគ្មានចម្លើយ)។"))
            return

        summary_header = f"📝 <b>សង្ខេបអត្ថបទ (Summary):</b>\n\n{html.escape(summary_text)}"
        await safe_send(lambda: msg.reply_text(summary_header, parse_mode="HTML"))
        await safe_send(lambda: context.bot.send_chat_action(chat_id=msg.chat_id, action="record_voice"))

        from app.services.telegram.media import process_tts_for_text
        await process_tts_for_text(update, context, summary_text, user_id)
    except Exception as exc:
        _release_tts_request(user_id)
        err_msg = f"❌ បរាជ័យ: {exc}"
        await safe_send(lambda: msg.reply_text(err_msg))


@legacy_bound_handler
async def cmd_narrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Extract and narrate article text from a web link with voice and AI summary."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    user_id = int(user.id)
    raw_text = msg.text or ""
    url = re.sub(r"^/(?:narrate|read)(?:@\w+)?\s*", "", raw_text, flags=re.IGNORECASE).strip()
    if not url:
        match = re.search(r"https?://[^\s]+", raw_text)
        if match:
            url = match.group(0)

    if not url or not (url.startswith("http://") or url.startswith("https://")):
        await safe_send(lambda: msg.reply_text(
            "📰 <b>របៀបប្រើប្រាស់ Web Link Article Narrator</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "• វាយ <code>/narrate https://news-site.com/article</code>\n"
            "• ឬគ្រាន់តែផ្ញើតំណភ្ជាប់ (Link) ព័ត៌មាន ឬអត្ថបទមកកាន់ Bot ដោយផ្ទាល់\n\n"
            "💡 <i>បូតនឹងទាញយកអត្ថបទ សង្ខេបចំណុចសំខាន់ៗ និងបង្កើតជាសំឡេង Voice Note ជូនភ្លាមៗ!</i>",
            parse_mode="HTML",
        ))
        return

    if await _check_cooldown(msg, user_id):
        return
    if not _reserve_tts_request(user_id):
        await safe_send(lambda: msg.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ..."))
        return

    progress = await TelegramProgress.start(
        bot=context.bot,
        chat_id=msg.chat_id,
        reply_target=msg,
        title="📰 កំពុងដំណើរការ Web Article Narrator",
        percent=10,
        stage="កំពុងទាញយកទំព័រវិបសាយ",
        detail="កំពុងភ្ជាប់ទៅកាន់តំណភ្ជាប់...",
        minimal=True,
    )

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        from app import legacy
        gemini_client = getattr(legacy, "_gemini", None)
        preferred = getattr(legacy, "GEMINI_MODEL", "gemini-2.5-flash")

        from app.services.ai.article_reader import (
            MIN_ARTICLE_CHARS,
            extract_article_content,
            fetch_article_html,
            is_safe_public_url,
            summarize_article_with_ai,
            summarize_url_with_ai,
        )

        safe, reason = is_safe_public_url(url)
        if not safe:
            _release_tts_request(user_id)
            await progress.fail(f"❌ តំណភ្ជាប់នេះមិនត្រូវបានអនុញ្ញាតទេ ({reason})")
            return

        await progress.update(25, "កំពុងទាញយកទំព័រ", "កំពុងទទួលទិន្នន័យពីគេហទំព័រ...", force=True)
        title = ""
        spoken_text = ""
        body_text = ""
        fetch_success = False

        try:
            html_data = await fetch_article_html(url)
            await progress.update(45, "កំពុងសម្រង់អត្ថបទ", "កំពុងសម្អាតផ្ទាំងពាណិជ្ជកម្ម និងកូដគេហទំព័រ...", force=True)
            title, body_text = extract_article_content(html_data)
            if len(body_text) >= MIN_ARTICLE_CHARS:
                fetch_success = True
        except Exception as net_err:
            logger.info("Direct article scraping failed (%s); switching to AI URL extraction", net_err)

        if not fetch_success:
            if gemini_client is not None:
                await progress.update(
                    50,
                    "កំពុងស្រាវជ្រាវព័ត៌មាន",
                    "គេហទំព័រមានប្រព័ន្ធការពារ Anti-Bot — កំពុងប្រើ Gemini AI ទាញយកខ្លឹមសារ...",
                    force=True,
                )

                def _ai_url_extract():
                    return summarize_url_with_ai(url, gemini_client, preferred)

                try:
                    title, spoken_text = await loop.run_in_executor(None, _ai_url_extract)
                except Exception as ai_err:
                    _release_tts_request(user_id)
                    logger.warning("AI URL fallback failed for %s: %s", url, ai_err)
                    await progress.fail("❌ មិនអាចទាញយកព័ត៌មានពី Link នេះបានទេ (គេហទំព័រការពារដោយ Anti-Bot)។")
                    return
            else:
                _release_tts_request(user_id)
                await progress.fail("❌ មិនអាចទាញយកព័ត៌មានពី Link នេះបានទេ (គេហទំព័រការពារដោយ Anti-Bot)។")
                return

        if not spoken_text:
            await progress.update(65, "កំពុងរៀបចំខ្លឹមសារ", f"រកឃើញ {len(body_text)} តួអក្សរ។ កំពុងសង្ខេបសម្រាប់អានជាសំឡេង...", force=True)
            if len(body_text) > 800 and gemini_client is not None:
                def _summarize():
                    return summarize_article_with_ai(title, body_text, gemini_client, preferred)

                spoken_text = await loop.run_in_executor(None, _summarize)
            else:
                spoken_text = body_text[:800]

        article_header = f"📰 <b>{html.escape(title or 'អត្ថបទព័ត៌មាន')}</b>\n🔗 <a href='{html.escape(url)}'>ប្រភពដើម (Original Link)</a>\n\n"
        preview_text = article_header + html.escape(spoken_text)
        await safe_send(lambda: msg.reply_text(preview_text, parse_mode="HTML", disable_web_page_preview=True))

        await progress.update(85, "កំពុងបង្កើតសំឡេង", "កំពុងបម្លែងសេចក្ដីសង្ខេបទៅជា Voice Note...", force=True)
        tts_script = f"{title}. {spoken_text}" if title and not spoken_text.startswith(title) else spoken_text

        from app.services.telegram.media import process_tts_for_text
        await progress.finish("🎙️ កំពុងផ្ញើសារសំឡេងជូនអ្នក...", delete_after_s=3.0)
        await process_tts_for_text(update, context, tts_script, user_id)
    except Exception as exc:
        _release_tts_request(user_id)
        logger.error("cmd_narrate failed for user %s: %s", user_id, exc, exc_info=True)
        await progress.fail(f"❌ បរាជ័យក្នុងការអានអត្ថបទ: {exc}")


@legacy_bound_handler
async def send_user_profile(message, user_id: int, user: Any = None):
    prefs = await get_user_prefs_async(user_id)
    
    # Extract user display name & username
    first_name = getattr(user, "first_name", None) or prefs.get("first_name") or ""
    last_name = getattr(user, "last_name", None) or ""
    full_name = f"{first_name} {last_name}".strip() or first_name or "អ្នកប្រើប្រាស់ (User)"
    username = getattr(user, "username", None) or prefs.get("username") or ""
    username_line = f"\n🔗 <b>Username:</b> @{html.escape(username)}" if username else ""

    gender_label = "👩 សំឡេងស្រី (Female)" if prefs.get("gender") == "female" else "👨 សំឡេងប្រុស (Male)"
    speed_label = next(
        (lbl for _, (lbl, val) in SPEED_OPTIONS.items() if abs(val - prefs.get("speed", 1.0)) < 0.01),
        f"{prefs.get('speed', 1.0)}x",
    )
    model_label = _tts_model_label(prefs.get("tts_model", "auto"))
    
    text = (
        f"⚙️ <b>កម្រងព័ត៌មាន & ការកំណត់សំឡេងរបស់អ្នក</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>ឈ្មោះ:</b> <b>{html.escape(full_name)}</b>\n"
        f"🆔 <b>User ID:</b> <code>{user_id}</code>"
        f"{username_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🗣️ <b>ប្រភេទសំឡេង:</b> <b>{gender_label}</b>\n"
        f"🎚️ <b>ល្បឿនអាន:</b> <code>{speed_label}</code>\n"
        f"🤖 <b>ម៉ូដែល TTS:</b> <b>{html.escape(model_label)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <i>ចុចប៊ូតុងខាងក្រោមដើម្បីកែប្រែការកំណត់បានភ្លាមៗ៖</i>"
    )

    await safe_send(lambda: message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_main_kb(
            prefs.get("gender", "female"),
            prefs.get("tts_model", "auto"),
            speed=prefs.get("speed", 1.0),
            include_back=True,
        ),
    ))


@legacy_bound_handler
async def cmd_myprefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    await send_user_profile(msg, user.id, user=user)


@legacy_bound_handler
async def cmd_ttsmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _ensure_user_allowed(update, context, "tts_enabled", "Text to voice"):
        return
    user_id = update.effective_user.id
    prefs = await get_user_prefs_async(user_id)
    await safe_send(lambda: update.message.reply_text(
        '🤖 <b>ជ្រើសរើសម៉ូដែល TTS</b>\n\n'
        '• <b>ស្វ័យប្រវត្តិ៖</b> ប្រើ Khmer HF សម្រាប់ភាសាខ្មែរ និង Edge សម្រាប់ភាសាផ្សេងៗ\n'
        '• <b>Gemini AI៖</b> ប្រើសំឡេង Google Gemini AI សំឡេងបែបធម្មជាតិ\n'
        '• <b>Khmer Kiri៖</b> ប្រើ mrrtmob/khmer-tts សម្រាប់អត្ថបទខ្មែរ\n'
        '• <b>Edge TTS៖</b> ប្រើ Microsoft Edge TTS សម្រាប់គ្រប់ភាសា',
        parse_mode="HTML",
        reply_markup=get_tts_model_kb(prefs.get("tts_model", "auto")),
    ))


@legacy_bound_handler
async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _hist_cache_clear(user_id)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_DB_EXECUTOR, db_history_clear, user_id)
    await safe_send(lambda: update.message.reply_text(
        "🗑️ ប្រវត្តិការសន្ទនារបស់អ្នកបានលុបចេញហើយ។\nBot នឹងចាប់ផ្ដើមការសន្ទនាថ្មី។"
    ))


@legacy_bound_handler
async def cmd_unlock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force release any stuck TTS locks or waiting queues for the user."""
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return
    user_id = int(user.id)
    from app import legacy
    if hasattr(legacy, "_release_tts_request"):
        legacy._release_tts_request(user_id)
    
    # Also release user lock and clear user state if needed
    with getattr(legacy, "_user_locks_guard", threading.RLock()):
        user_locks = getattr(legacy, "_user_locks", {})
        if user_id in user_locks:
            user_locks.pop(user_id, None)

    if getattr(context, "user_data", None) is not None:
        context.user_data.clear()

    await safe_send(lambda: msg.reply_text(
        "🔓 <b>ដោះសោរជោគជ័យ!</b>\n\nរាល់ការរង់ចាំ និងសោរដំណើរការចាស់របស់អ្នកត្រូវបានសម្អាតរួចរាល់។ អ្នកអាចផ្ញើសារថ្មីបានឥឡូវនេះ។",
        parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_security(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.effective_message:
        return
    status = "active"
    await safe_send(lambda: update.effective_message.reply_text(
        f'🔐 <b>សុវត្ថិភាព និងឯកជនភាព</b>\n\nលេខសម្គាល់អ្នកប្រើប្រាស់៖ <code>{int(user.id)}</code>\nស្ថានភាព៖ <b>{html.escape(status)}</b>\n\n✅ ពាក្យបញ្ជារបស់អ្នកគ្រប់គ្រងត្រូវបានការពារ។\n✅ ការការពារ Spam និងការផ្ញើសារច្រើនពេកត្រូវបានបើក។\n✅ តាមលំនាំដើម សោ API ត្រូវទទួលពី Header មិនមែនពី URL Query String ទេ។\n✅ ប្រើ /clear ដើម្បីសម្អាតបរិបទការជជែក។\n🗑️ ប្រើ /deleteme ដើម្បីលុបប្រវត្តិបូត និងចំណូលចិត្តដែលបានរក្សាទុក។',
        parse_mode="HTML",
    ))


@legacy_bound_handler
async def cmd_privacy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message:
        return
    await safe_send(lambda: update.effective_message.reply_text(
        '🔒 <b>ឯកជនភាព</b>\n\nបូតនេះរក្សាទុកតែទិន្នន័យដែលចាំបាច់សម្រាប់ចំណូលចិត្ត ដំណើរការ Cache អត្ថបទ/សំឡេង បរិបទការសន្ទនា សុវត្ថិភាពអ្នកគ្រប់គ្រង និងកំណត់ហេតុការផ្ញើ។\n\nអ្នកអាចប្រើ /clear ដើម្បីលុបបរិបទការសន្ទនាបច្ចុប្បន្ន ឬ /deleteme ដើម្បីលុបប្រវត្តិបូត និងចំណូលចិត្តដែលបានរក្សាទុកពី Cache/មូលដ្ឋានទិន្នន័យ តាមការកំណត់របស់ប្រព័ន្ធ។',
        parse_mode="HTML",
    ))


@legacy_bound_handler
async def cmd_delete_my_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user or not update.effective_message:
        return
    args = context.args or []
    if _is_admin(int(user.id)) and "--confirm-admin" not in args:
        await safe_send(lambda: update.effective_message.reply_text(
            '⚠️ បានរកឃើញគណនីអ្នកគ្រប់គ្រង។ ដើម្បីលុបទិន្នន័យអ្នកប្រើប្រាស់របស់ខ្លួន សូមប្រើ៖\n<code>/deleteme --confirm-admin</code>',
            parse_mode="HTML",
        ))
        return
    if getattr(context, "user_data", None) is not None:
        context.user_data.clear()
    await _delete_user_personal_data(int(user.id))
    await safe_send(lambda: update.effective_message.reply_text(
        '✅ បានសម្អាតប្រវត្តិបូត Cache អត្ថបទ និងចំណូលចិត្តរបស់អ្នក។\nសម្គាល់៖ កំណត់ត្រាបិទសិទ្ធិសុវត្ថិភាព និងកំណត់ហេតុការផ្ញើ/សវនកម្មចាំបាច់ អាចត្រូវរក្សាទុកដោយអ្នកគ្រប់គ្រង ដើម្បីការពារការប្រើប្រាស់ខុសគោលបំណង។'
    ))


@legacy_bound_handler
async def broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    _pending_broadcast.pop(user.id, None)
    context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
    await safe_send(lambda: msg.reply_text(
        '🛡️ <b>សុវត្ថិភាពការផ្សាយសារ V2</b>\n\n📨 <b>របៀបប្រើ</b>\n• ផ្ញើ <b>អត្ថបទ</b> ឬ <b>រូបភាព + ចំណងជើង</b> ដែលចង់ផ្សាយ\n• បូតនឹងបង្ហាញសារមើលជាមុនចុងក្រោយ មុនពេលផ្ញើពិត\n• គាំទ្រទម្រង់សារ Telegram, HTML, MarkdownV2, Markdown និងអត្ថបទធម្មតា\n• ដើម្បីបង្ខំទម្រង់ សូមដាក់ <code>::html</code>, <code>::mdv2</code>, <code>::md</code> ឬ <code>::plain</code> នៅជួរទី១\n\n🔐 <b>ការការពារ</b>\n✅ បញ្ជាក់សារមើលជាមុន មុនពេលផ្ញើ\n✅ រំលងអ្នកប្រើប្រាស់ដែលបានបិទបូត ឬមិនអាចទាក់ទងបាន\n✅ គ្រប់គ្រងល្បឿនផ្ញើ និង RetryAfter\n\nវាយ /cancel ដើម្បីបោះបង់។',
        parse_mode="HTML",
        reply_markup=get_broadcast_entry_kb(),
    ))


@legacy_bound_handler
async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    admin_id = user.id
    _sched_payload.pop(admin_id, None)
    context.user_data["sched_state"] = SCHED_WAIT_MSG
    await safe_send(lambda: msg.reply_text(
        '📅 <b>ការផ្សាយតាមកាលវិភាគ</b>\n\nសូមផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + ចំណងជើង</b> ដែលចង់កំណត់ពេលផ្ញើ។\n✅ គាំទ្រទម្រង់ដើមរបស់ Telegram, HTML, MarkdownV2, Markdown និងអត្ថបទធម្មតា។\nដើម្បីបង្ខំទម្រង់ សូមដាក់ <code>::html</code>, <code>::mdv2</code>, <code>::md</code> ឬ <code>::plain</code> នៅជួរទី១។\n\nវាយ /cancel ដើម្បីបោះបង់។',
        parse_mode="HTML",
    ))


@legacy_bound_handler
async def cmd_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    admin_id = user.id
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_admin_pending, admin_id)
    if not rows:
        await safe_send(lambda: msg.reply_text('📭 មិនមានការផ្សាយតាមកាលវិភាគទេ។'))
        return
    await safe_send(lambda: msg.reply_text(
        f'📋 <b>ការផ្សាយតាមកាលវិភាគ ({len(rows)} កំពុងរង់ចាំ)</b>\nចុចលើកាលវិភាគ ដើម្បីមើលព័ត៌មានលម្អិត ឬបោះបង់។',
        parse_mode="HTML",
        reply_markup=get_schedules_list_kb(rows, page=0),
    ))


@legacy_bound_handler
async def cmd_cancelschedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    admin_id = user.id
    args = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(lambda: msg.reply_text(
            '❌ របៀបប្រើ៖ /cancelschedule &lt;id&gt;\nឬប្រើ /schedules ដើម្បីជ្រើស។',
            parse_mode="HTML",
        ))
        return
    row_id = int(args[0])
    loop = asyncio.get_running_loop()
    row = await loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_one, row_id)
    if not row:
        await safe_send(lambda: msg.reply_text(f"❌ រកមិនឃើញ Schedule #{row_id}។"))
        return
    if row["admin_id"] != admin_id:
        await safe_send(lambda: msg.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
        return
    if row["status"] != "pending":
        st = row["status"]
        await safe_send(lambda: msg.reply_text(
            f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
            parse_mode="HTML",
        ))
        return
    await loop.run_in_executor(_DB_EXECUTOR, db_sched_set_status, row_id, "cancelled")
    await safe_send(lambda: msg.reply_text(
        f'✅ កាលវិភាគ <b>#{row_id}</b> បានបោះបង់។', parse_mode="HTML"
    ))


@legacy_bound_handler
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    uid = user.id
    if not _is_admin(uid):
        await safe_send(lambda: msg.reply_text('ℹ️ មិនមានប្រតិបត្តិការដែលត្រូវបោះបង់ទេ។'))
        return

    # Preserve the old admin-chat notification behavior while still using the
    # unified cleanup helper for every other transient state.
    target_id = None
    with suppress(Exception):
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(uid)

    cleared = await _clear_admin_transient_state(context, uid)

    if target_id:
        with suppress(Exception):
            await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")

    if cleared:
        labels = ", ".join(cleared[:8])
        await safe_send(lambda: msg.reply_text(
            f"✅ បានបោះបង់/សម្អាត state រួច: <code>{html.escape(labels)}</code>",
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        ))
        return

    await safe_send(lambda: msg.reply_text('ℹ️ មិនមានប្រតិបត្តិការដែលត្រូវបោះបង់ទេ។'))


@legacy_bound_handler
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return
    loop = asyncio.get_running_loop()
    user_ids, pending_scheds = await asyncio.gather(
        loop.run_in_executor(_DB_EXECUTOR, get_all_user_ids),
        loop.run_in_executor(_DB_EXECUTOR, db_sched_fetch_admin_pending, update.effective_user.id),
    )
    await safe_send(lambda: msg.reply_text(
        f"📊 <b>ស្ថិតិបូត</b>\n\n👥 អ្នកប្រើប្រាស់សរុប៖ <b>{len(user_ids)}</b>\n💬 ការជជែកសកម្មរបស់អ្នកគ្រប់គ្រង៖ <b>{len(_admin_chat_target)}</b>\n📅 កាលវិភាគកំពុងរង់ចាំ៖ <b>{len(pending_scheds)}</b>\n🔒 សោអ្នកប្រើប្រាស់សកម្ម៖ <b>{len(_user_locks)}</b>\n💭 ចំនួនធាតុប្រវត្តិក្នុង Cache៖ <b>{len(_hist_cache)}</b>\n🔑 ការផ្ទៀងផ្ទាត់ API បែប Dynamic៖ <b>{('ON' if _dynamic_ai_auth_configured() else 'OFF')}</b>\n🤗 ម៉ូដែល HF៖ <b>{HF_MODEL}</b>\nOCR៖ <b>{HF_OCR_MODEL}</b>",
        parse_mode="HTML",
    ))


@legacy_bound_handler
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram admin shortcut for the full bot health panel."""
    msg = update.effective_message
    user = update.effective_user
    if not msg or not user:
        return
    if not _is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ ពាក្យបញ្ជានេះសម្រាប់ Admin ប៉ុណ្ណោះ។"))
        return
    text = await _admin_health_text()
    await safe_send(lambda: msg.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
        disable_web_page_preview=True,
    ))


@legacy_bound_handler
async def cmd_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Live System Metrics & Telemetry report for Admins, and friendly status for users."""
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    from app import legacy
    snapshot = legacy._system_metrics_snapshot() if hasattr(legacy, "_system_metrics_snapshot") else {}

    if not _is_admin(int(user.id)):
        status_text = (
            "🟢 <b>ស្ថានភាពប្រព័ន្ធ Bot Voice (System Status)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚡ <b>ស្ថានភាពទូទៅ:</b> កំពុងដំណើរការយ៉ាងរលូន (Online 24/7)\n"
            "🎙️ <b>ម៉ាស៊ីន TTS:</b> ដំណើរការធម្មតា (Kiri, Gemini AI, Edge)\n"
            "📸 <b>ម៉ាស៊ីន OCR:</b> ដំណើរការធម្មតា (Google Gemini Vision)\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "💡 <i>ប្រសិនបើជួបបញ្ហា សូមប្រើ /unlock ឬទាក់ទងមកកាន់ @m11mmm112</i>"
        )
        await safe_send(lambda: msg.reply_text(status_text, parse_mode="HTML"))
        return

    storage = snapshot.get("storage", {})
    cache = snapshot.get("tts_audio_cache", {})
    anti_spam = snapshot.get("anti_spam", {})

    status_icon = "🟢" if snapshot.get("status") == "healthy" else "🟡"
    redis_status = f"✅ Connected ({storage.get('redis_ping_ms')}ms)" if storage.get("redis_connected") else "❌ Offline"
    supabase_status = "✅ Connected" if storage.get("supabase_connected") else "⚠️ Degraded"

    uptime_m = round(float(snapshot.get("uptime_seconds", 0.0)) / 60.0, 1)

    text = (
        f"{status_icon} <b>ប្រព័ន្ធ Telemetry & Metrics (v{snapshot.get('version', '4.2.0')})</b>\n\n"
        f"⏱️ Uptime: <b>{uptime_m} នាទី</b>\n"
        f"🤖 Bot Mode: <b>{snapshot.get('bot_mode', 'POLLING')}</b>\n\n"
        f"🗄️ <b>Storage Layer:</b>\n"
        f"• Redis L2 Cache: <b>{redis_status}</b>\n"
        f"• Supabase Database: <b>{supabase_status}</b>\n\n"
        f"🎵 <b>Audio Cache (L1/L2):</b>\n"
        f"• Memory Items: <b>{cache.get('l1_memory_items', 0)}</b> ({round(cache.get('l1_memory_bytes', 0)/1024, 1)} KB)\n"
        f"• Binary TTL: <b>{cache.get('ttl_seconds', 0)}s</b>\n\n"
        f"🛡️ <b>Anti-Spam & Rate Limiter:</b>\n"
        f"• Tracked Users: <b>{anti_spam.get('tracked_users', 0)}</b>\n"
        f"• Active Cooldowns: <b>{anti_spam.get('active_cooldowns', 0)}</b>\n\n"
        f"🌐 <i>REST API: <code>/system</code>, <code>/metrics</code></i>"
    )

    await safe_send(lambda: msg.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
        disable_web_page_preview=True,
    ))


@legacy_bound_handler
async def cmd_dbstatus(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin inspection of Supabase tables, row counts, and database health."""
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    if not _is_admin(int(user.id)):
        status_text = (
            "🟢 <b>ស្ថានភាពទិន្នន័យ (Database Status)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚡ <b>ស្ថានភាព:</b> ដំណើរការធម្មតា (Online 24/7)\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "💡 <i>សម្រាប់តែអ្នកគ្រប់គ្រងបូត (Admin only)</i>"
        )
        await safe_send(lambda: msg.reply_text(status_text, parse_mode="HTML"))
        return

    from app import legacy
    text = await legacy._get_admin_db_text()
    kb = legacy.get_admin_db_kb()
    await safe_send(lambda: msg.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=kb,
        disable_web_page_preview=True,
    ))


@legacy_bound_handler
async def cmd_dbbackup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin on-demand database backup trigger."""
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg:
        return

    if not _is_admin(int(user.id)):
        await safe_send(lambda: msg.reply_text("⛔ <b>សិទ្ធិត្រូវបានបដិសេធ (Admin only)</b>", parse_mode="HTML"))
        return

    import asyncio
    from app import legacy
    asyncio.create_task(legacy._admin_trigger_backup(msg, user.id))


@legacy_bound_handler
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Telegram /admin entry point with mobile shortcuts.

    Supported shortcuts:
      /admin compact, /admin health, /admin errors, /admin broadcast,
      /admin report, /admin optimize, /admin users, /admin settings, /admin runtime, /admin needs,
      /admin db, /admin backup
    """
    user_id = update.effective_user.id if update.effective_user else 0
    msg = update.effective_message
    if not msg:
        return

    arg = ""
    with suppress(Exception):
        arg = str((context.args or [""])[0]).strip().lower()

    if arg in {"needs", "userneeds", "user_needs", "feedback"}:
        text = await _user_needs_home_text(user_id)
        await safe_send(lambda: msg.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=get_user_needs_home_kb(),
            disable_web_page_preview=True,
        ))
        return

    if arg in {"compact", "mobile", "mini"}:
        text = await _admin_compact_text(user_id)
        await safe_send(lambda: msg.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_compact_kb(),
            disable_web_page_preview=True,
        ))
        return
    if arg in {"health", "status"}:
        text = await _admin_health_text()
        from app import legacy
        await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=legacy.get_admin_health_kb(), disable_web_page_preview=True))
        return
    if arg in {"stats", "stat", "telemetry"}:
        from app import legacy
        text = await legacy._admin_stats_text(user_id)
        await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=legacy.get_admin_stats_kb(), disable_web_page_preview=True))
        return
    if arg in {"errors", "error"}:
        await safe_send(lambda: msg.reply_text(_error_center_text(), parse_mode="HTML", reply_markup=_error_center_kb(), disable_web_page_preview=True))
        return
    if arg in {"optimize", "perf", "performance"}:
        await safe_send(lambda: msg.reply_text(_admin_optimize_text(), parse_mode="HTML", reply_markup=get_admin_optimize_kb(), disable_web_page_preview=True))
        return
    if arg in {"cache", "audiocache", "audio_cache", "cdn"}:
        from app import legacy
        text = legacy._admin_audio_cache_text()
        await safe_send(lambda: msg.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=legacy.get_admin_audio_cache_kb(),
            disable_web_page_preview=True,
        ))
        return
    if arg in {"report", "pdf"}:
        await safe_send(lambda: msg.reply_text('📄 <b>របាយការណ៍ PDF</b>\n\nសូមជ្រើសរើសចន្លោះពេលរបាយការណ៍៖', parse_mode="HTML", reply_markup=get_admin_report_day_kb(), disable_web_page_preview=True))
        return
    if arg in {"users", "user"}:
        await cmd_users(update, context)
        return
    if arg in {"crm"}:
        text = await _admin_crm_text("all")
        await safe_send(lambda: msg.reply_text(text, parse_mode="HTML", reply_markup=get_admin_crm_kb("all"), disable_web_page_preview=True))
        return
    if arg in {"system", "metrics"}:
        await cmd_system(update, context)
        return
    if arg in {"db", "database", "supabase", "dbstatus"}:
        await cmd_dbstatus(update, context)
        return
    if arg in {"backup", "dbbackup"}:
        await cmd_dbbackup(update, context)
        return
    if arg in {"api", "apikeys"}:
        await cmd_api(update, context)
        return
    if arg in {"broadcast", "bc"}:
        context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
        await safe_send(lambda: msg.reply_text(
            '📢 <b>ដំណើរការផ្សាយសារដោយសុវត្ថិភាព</b>\n\nសូមផ្ញើអត្ថបទ ឬរូបភាព + ចំណងជើងឥឡូវនេះ។ បូតនឹងបង្ហាញសារមើលជាមុន មុនពេលផ្ញើពិត។\n\nប្រើ /cancel ដើម្បីបោះបង់។',
            parse_mode="HTML",
            reply_markup=get_admin_action_kb(),
        ))
        return
    if arg in {"config", "botconfig", "bot_config", "configuration"}:
        from app import legacy
        await legacy._admin_open_bot_config_panel(msg, user_id, force=True)
        return
    if arg in {"settings", "setting"}:
        settings, _status = await get_bot_settings_async(force=True)
        await safe_send(lambda: msg.reply_text('⚙️ <b>ការកំណត់បូត</b>', parse_mode="HTML", reply_markup=get_bot_settings_kb(settings)))
        return
    if arg in {"runtime", "run"}:
        await safe_send(lambda: msg.reply_text(_runtime_admin_text(), parse_mode="HTML", reply_markup=get_runtime_admin_kb(), disable_web_page_preview=True))
        return
    if arg in {"geminiaudio", "gemini_audio", "gemini_audio_model", "geminimodel"}:
        args = context.args or []
        if len(args) > 1:
            new_model = args[1].strip()
            from app import legacy
            await legacy._update_run_state("GEMINI_AUDIO_MODEL", new_model, persist=True)
            await safe_send(lambda: msg.reply_text(
                f"✅ <b>បានប្តូរ Gemini Audio TTS Model ជោគជ័យ!</b>\n\n"
                f"🤖 <b>Active Model:</b> <code>{html.escape(new_model)}</code>\n"
                f"💾 <i>បានរក្សាទុកក្នុង Supabase & Redis រួចរាល់។</i>",
                parse_mode="HTML"
            ))
            return
        from app import legacy
        current = legacy._run_state_gemini_audio_model()
        await safe_send(lambda: msg.reply_text(
            f"🎙️ <b>ការកំណត់ម៉ូដែល Gemini Audio TTS</b>\n\n"
            f"🤖 <b>ម៉ូដែលបច្ចុប្បន្ន:</b> <code>{html.escape(current)}</code>\n\n"
            f"💡 <b>របៀបប្តូរម៉ូដែលថ្មី:</b>\n"
            f"• <code>/admin geminiaudio gemini-2.5-flash-preview-tts</code>\n"
            f"• <code>/admin geminiaudio gemini-2.5-pro-preview-tts</code>\n"
            f"• <code>/admin geminiaudio gemini-3.1-flash-tts-preview</code>",
            parse_mode="HTML"
        ))
        return

    text = await _admin_home_text(user_id)
    await safe_send(lambda: msg.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_admin_dashboard_kb(),
        disable_web_page_preview=True,
    ))


@legacy_bound_handler
async def cmd_feature_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User command: /need, /feedback, /request_feature."""
    user = update.effective_user
    msg = update.message
    if not user or not msg:
        return
    if not await _ensure_user_allowed(update, context):
        return
    detail = " ".join(context.args or []).strip()
    if detail:
        await _save_user_feature_request(update, context, detail)
        return
    context.user_data[FEATURE_REQUEST_WAIT_TEXT] = True
    await safe_send(lambda: msg.reply_text(
        "💬 <b>Open Answer</b>\n\n"
        "សូមសរសេរ Feature ថ្មី ឬការកែលម្អដែលអ្នកចង់បាន។\n\n"
        "ឧទាហរណ៍:\n"
        "• ចង់បានសង្ខេប PDF ជាខ្មែរ\n"
        "• ចង់បាន Khmer female voice ច្បាស់ជាងមុន\n"
        "• ចង់ឲ្យ OCR អានអក្សរខ្មែរពីរូបភាពបានល្អ\n\n"
        "Safety: អត្ថបទអតិបរមា 500 តួអក្សរ។ វាយ <code>cancel</code> ដើម្បីបោះបង់។",
        parse_mode="HTML",
        disable_web_page_preview=True,
    ))


@legacy_bound_handler
async def cmd_runtime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Direct shortcut for runtime operations without replacing /admin.
    msg = update.effective_message
    if not msg:
        return
    await safe_send(lambda: msg.reply_text(
        _runtime_admin_text(),
        parse_mode="HTML",
        reply_markup=get_runtime_admin_kb(),
        disable_web_page_preview=True,
    ))


@legacy_bound_handler
async def cmd_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to create/list/revoke API keys for /ai-assistant."""
    msg = update.effective_message
    if not msg:
        return
    admin_id = update.effective_user.id
    args = list(context.args or [])

    if not args or args[0].lower() in ("help", "-h", "--help"):
        await safe_send(lambda: msg.reply_text(
            _api_help_text(),
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
            disable_web_page_preview=True,
        ))
        return

    action = args[0].lower().strip()
    loop = asyncio.get_running_loop()

    if action == "sql":
        pages = _paginate_pre_html(
            AI_API_KEYS_TABLE_SQL,
            limit=3800,
            header="🧩 <b>Supabase SQL for API keys</b>\n\n",
        )
        for page in pages:
            await safe_send(lambda p=page: msg.reply_text(
                p,
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
        return

    if action == "create":
        note = " ".join(args[1:]).strip()
        try:
            raw_key, row, storage = await loop.run_in_executor(
                _DB_EXECUTOR,
                lambda: db_ai_api_key_create(admin_id=admin_id, note=note),
            )
        except Exception as e:
            logger.error(f"/api create failed: {e}", exc_info=True)
            err = str(e)
            pages = _paginate_pre_html(
                err,
                limit=3500,
                header=(
                    "❌ Cannot create API key.\n"
                    "If this is first setup, press <b>🧩 Setup SQL</b> or run <code>/api sql</code> "
                    "and execute it in Supabase.\n\n"
                ),
            )
            for page in pages:
                await safe_send(lambda p=page: msg.reply_text(
                    p,
                    parse_mode="HTML",
                    reply_markup=get_api_admin_kb(),
                ))
            return

        warning = ""
        if storage == "memory":
            warning = (
                "\n\n⚠️ Supabase is not configured, so this key is stored in memory only "
                "and will stop working after restart/deploy."
            )

        await safe_send(lambda: msg.reply_text(
            f"""✅ <b>បានបង្កើតសោ API សម្រាប់ AI ថ្មី</b>\n\nសូមចម្លងសោនេះឥឡូវនេះ។ វានឹងមិនត្រូវបានបង្ហាញម្ដងទៀតទេ។\n\n<code>{html.escape(raw_key)}</code>\n\nបុព្វបទ៖ <code>{html.escape(str(row.get('key_prefix') or _api_key_prefix(raw_key)))}</code>\nកន្លែងរក្សាទុក៖ <b>{html.escape(storage)}</b>\n\nឧទាហរណ៍៖\n<pre>curl -X POST https://YOUR-APP.onrender.com/ai-assistant \\\n  -H 'Content-Type: application/json' \\\n  -H 'X-Api-Key: {html.escape(raw_key)}' \\\n  -d '{{"message":"Hello"}}'</pre>{warning}""",
            parse_mode="HTML",
            reply_markup=get_api_admin_kb(),
            disable_web_page_preview=True,
        ))
        return

    if action == "list":
        try:
            rows = await loop.run_in_executor(_DB_EXECUTOR, lambda: db_ai_api_key_list(limit=20))
        except Exception as exc:
            logger.error("/api list failed: %s", exc, exc_info=True)
            error_text = html.escape(str(exc)[:3500])
            await safe_send(lambda: msg.reply_text(
                f'❌ មិនអាចបង្ហាញបញ្ជីសោ API បានទេ។\n<pre>{error_text}</pre>',
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        if not rows:
            await safe_send(lambda: msg.reply_text(
                'ℹ️ មិនមានសោ API ទេ។ សូមចុច <b>➕ បង្កើតសោ API</b> ឬប្រើ <code>/api create</code>។',
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        body = "\n\n".join(_format_api_key_row(r) for r in rows)
        for page in _paginate_html(body, limit=3900, header="🔑 <b>AI API Keys</b>\n\n"):
            await safe_send(lambda p=page: msg.reply_text(
                p,
                parse_mode="HTML",
                reply_markup=get_api_list_kb(rows),
                disable_web_page_preview=True,
            ))
        return

    if action == "revoke":
        if len(args) < 2:
            await safe_send(lambda: msg.reply_text(
                '⚠️ របៀបប្រើ៖ <code>/api revoke KEY_PREFIX_OR_ID</code>\n\nឬចុច <b>📋 បញ្ជីសោ API</b> ហើយដកសិទ្ធិតាមប៊ូតុង។',
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        identifier = args[1].strip()
        try:
            ok, info = await loop.run_in_executor(
                _DB_EXECUTOR,
                lambda: db_ai_api_key_revoke(identifier),
            )
        except Exception as exc:
            logger.error("/api revoke failed: %s", exc, exc_info=True)
            error_text = html.escape(str(exc)[:3500])
            await safe_send(lambda: msg.reply_text(
                f'❌ មិនអាចដកសិទ្ធិសោ API បានទេ។\n<pre>{error_text}</pre>',
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
            return

        if ok:
            await safe_send(lambda: msg.reply_text(
                f'✅ បានដកសិទ្ធិសោ API៖ <code>{html.escape(info)}</code>',
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
        else:
            await safe_send(lambda: msg.reply_text(
                f"❌ {html.escape(info)}",
                parse_mode="HTML",
                reply_markup=get_api_admin_kb(),
            ))
        return

    await safe_send(lambda: msg.reply_text(
        _api_help_text(),
        parse_mode="HTML",
        reply_markup=get_api_admin_kb(),
        disable_web_page_preview=True,
    ))


@legacy_bound_handler
async def cmd_botsettings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    settings, status = await get_bot_settings_async(force=True)
    text = (
        "⚙️ <b>Bot Settings Panel</b>\n\n"
        f"Storage: <b>{_ok_bad(bool(status.get('db_ok')), 'Supabase', 'Memory / setup needed')}</b>\n"
        "Use /admin → ⚙️ Settings for button controls."
    )
    await safe_send(lambda: update.message.reply_text(
        text,
        parse_mode="HTML",
        reply_markup=get_bot_settings_kb(settings),
    ))


@legacy_bound_handler
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if args:
        query_text = " ".join(args).strip()
        results = await asyncio.get_running_loop().run_in_executor(
            _DB_EXECUTOR,
            lambda: search_users_by_query(query_text),
        )
        context.user_data["users_search_query"] = query_text
        context.user_data["users_search_results"] = results
        if not results:
            await safe_send(lambda: update.message.reply_text(
                f'🔎 <b>ស្វែងរកអ្នកប្រើប្រាស់</b>\n\nរកមិនឃើញអ្នកប្រើប្រាស់សម្រាប់៖ <code>{html.escape(query_text)}</code>',
                parse_mode="HTML",
                reply_markup=get_user_search_prompt_kb(),
            ))
            return
        await safe_send(lambda: update.message.reply_text(
            f'🔎 <b>លទ្ធផលស្វែងរកអ្នកប្រើប្រាស់</b>\n\nពាក្យស្វែងរក៖ <code>{html.escape(query_text)}</code>\nរកឃើញ៖ <b>{len(results)}</b> នាក់\n\nសូមជ្រើសរើសអ្នកប្រើប្រាស់ ដើម្បីមើលព័ត៌មានលម្អិត។',
            parse_mode="HTML",
            reply_markup=get_user_search_page_kb(results, page=0),
        ))
        return

    users = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, get_all_users_with_names)
    if not users:
        await safe_send(lambda: update.message.reply_text("❌ គ្មានអ្នកប្រើប្រាស់ registered ទេ។"))
        return
    await safe_send(lambda: update.message.reply_text(
        f"👥 <b>អ្នកប្រើប្រាស់ ({len(users)} នាក់)</b>\nចុចលើឈ្មោះ ដើម្បីមើល Detail ឬប្រើ 🔎 Search User ។",
        parse_mode="HTML",
        reply_markup=get_users_page_kb(users, page=0),
    ))


@legacy_bound_handler
async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    args     = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(lambda: update.message.reply_text(
            '❌ របៀបប្រើ៖ /chat <user_id>\nឬប្រើ /users ដើម្បីជ្រើសអ្នកប្រើប្រាស់។'
        ))
        return
    target_id = int(args[0])
    exists    = await asyncio.get_running_loop().run_in_executor(_DB_EXECUTOR, user_exists_in_db, target_id)
    if not exists:
        await safe_send(lambda: update.message.reply_text(
            f'❌ អ្នកប្រើប្រាស់ <code>{target_id}</code> មិនមាននៅក្នុងមូលដ្ឋានទិន្នន័យទេ។', parse_mode="HTML"
        ))
        return
    await _open_chat_session(context.bot, admin_id, target_id, context)
    await safe_send(lambda: update.message.reply_text(
        f"💬 <b>Chat Mode បើក</b>\n\nកំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
        "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\nវាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
        parse_mode="HTML",
    ))


@legacy_bound_handler
async def cmd_endchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id  = update.effective_user.id
    target_id = _close_session(admin_id)
    context.user_data.pop("chat_state", None)
    if target_id is None:
        await safe_send(lambda: update.message.reply_text("ℹ️ អ្នកមិនទាន់ open Chat ណាមួយទេ។"))
        return
    await safe_send(lambda: update.message.reply_text(
        f'✅ បានបញ្ចប់ការជជែកជាមួយអ្នកប្រើប្រាស់ <code>{target_id}</code>។', parse_mode="HTML"
    ))
    with suppress(Exception):
        await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")


__all__ = [
    'admin_stats',
    'broadcast_start',
    'cmd_admin',
    'cmd_api',
    'cmd_ask',
    'cmd_botsettings',
    'cmd_cancel',
    'cmd_cancelschedule',
    'cmd_chat',
    'cmd_clear',
    'cmd_dbbackup',
    'cmd_dbstatus',
    'cmd_delete_my_data',
    'cmd_endchat',
    'cmd_feature_request',
    'cmd_health',
    'cmd_myprefs',
    'cmd_narrate',
    'cmd_privacy',
    'cmd_runtime',
    'cmd_schedule',
    'cmd_schedules',
    'cmd_security',
    'cmd_summary',
    'cmd_system',
    'cmd_translate',
    'cmd_ttsmodel',
    'cmd_unlock',
    'cmd_users',
    'on_help',
    'on_start'
]
