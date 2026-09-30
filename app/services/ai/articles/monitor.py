"""Background article monitoring, admin review queue, and broadcasting engine."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
import html
import inspect
import io
import logging
import os
import time
from typing import Any
import urllib.parse

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from app.services.ai.article_reader import (
    fetch_image_bytes,
    generate_article_hash,
    generate_smart_article_summary,
    store_article_session,
)
from app.services.ai.article_storage import (
    get_article_sources,
    get_pending_article,
    is_article_handled,
    is_article_pending,
    is_article_sent,
    mark_article_sent,
    save_pending_article,
    update_pending_status,
)
from app.services.ai.article_translator import translate_text, translate_text_async
from app.services.ai.categorizer import analyze_article_metadata
from app.services.ai.deduplicator import find_cross_source_duplicate
from app.services.ai.extractor import get_new_articles, verify_article_sources
from app.services.ai.formatter import format_article_message
from app.services.ai.telegraph_service import get_telegraph_url

logger = logging.getLogger(__name__)


def _get_active_bot() -> Bot | None:
    """Retrieve active Telegram Bot instance across modern runner and legacy runtime."""
    with suppress(Exception):
        from app.bot import get_global_telegram_app

        app = get_global_telegram_app()
        if app is not None and getattr(app, "bot", None) is not None:
            return app.bot

    with suppress(Exception):
        from app import legacy

        for attr in ("telegram_application", "_TELEGRAM_APP", "_bot_app"):
            app = getattr(legacy, attr, None)
            if app is not None and getattr(app, "bot", None) is not None:
                return app.bot

    return None


async def _get_admin_user_ids() -> list[int]:
    """Retrieve all authorized admin user IDs across policy authorizer, SETTINGS, and env."""
    admin_ids: set[int] = set()

    # 1. Check modern admin policy authorizer
    with suppress(Exception):
        from app.services.security.admin_policy import get_telegram_admin_authorizer

        authorizer = get_telegram_admin_authorizer()
        ids = await authorizer.load_ids()
        if ids:
            admin_ids.update(ids)

    # 2. Check legacy authorizer module if present
    if not admin_ids:
        with suppress(Exception):
            from app.core.telegram_auth import get_telegram_admin_authorizer

            authorizer = get_telegram_admin_authorizer()
            ids = await authorizer.load_ids()
            if ids:
                admin_ids.update(ids)

    # 3. Check SETTINGS and Environment Variables
    with suppress(Exception):
        from app.core.config import SETTINGS

        for attr in ("ADMIN_IDS", "ADMIN_USER_ID"):
            val = getattr(SETTINGS, attr, None)
            if isinstance(val, (set, list, tuple)):
                admin_ids.update(int(x) for x in val if str(x).lstrip("-").isdigit())
            elif isinstance(val, str):
                admin_ids.update(int(x.strip()) for x in val.split(",") if x.strip().lstrip("-").isdigit())

    for env_key in ("ADMIN_IDS", "ADMIN_USER_ID", "TELEGRAM_ADMIN_IDS"):
        env_val = os.getenv(env_key, "").strip()
        if env_val:
            for item in env_val.split(","):
                if item.strip().lstrip("-").isdigit():
                    admin_ids.add(int(item.strip()))

    return sorted(admin_ids)


def _safe_format_article_message(**kwargs: Any) -> tuple[str, InlineKeyboardMarkup]:
    """Proxy formatter call to only pass parameters accepted by the active format_article_message signature."""
    sig = inspect.signature(format_article_message)
    accepted_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return format_article_message(**accepted_kwargs)


async def _safe_verify_sources(query: str) -> dict[str, Any]:
    """Execute news source verification safely across sync or async implementations."""
    try:
        if asyncio.iscoroutinefunction(verify_article_sources):
            res = await verify_article_sources(query)
        else:
            res = await asyncio.to_thread(verify_article_sources, query)

        if asyncio.iscoroutine(res):
            res = await res
        elif callable(res):
            res = res()
            if asyncio.iscoroutine(res):
                res = await res

        if isinstance(res, dict):
            return res
        return {"verified": False, "is_verified": False, "sources": 0, "publishers": []}
    except Exception as exc:
        logger.debug("Source verification fallback: %s", exc)
        return {"verified": False, "is_verified": False, "sources": 0, "publishers": []}


async def scan_sources_and_notify_admin(bot: Bot | None = None) -> list[dict[str, Any]]:
    """Scan active base URLs for new articles and notify admins for approval."""
    if bot is None:
        bot = _get_active_bot()

    active_sources = await get_article_sources(active_only=True)
    if not active_sources:
        logger.info("No active article base URL sources configured to scan.")
        return []

    admin_ids = await _get_admin_user_ids()
    discovered: list[dict[str, Any]] = []
    discovered_lock = asyncio.Lock()
    session_seen_hashes: set[str] = set()
    semaphore = asyncio.Semaphore(3)

    async def _scan_single_source(src: dict[str, Any]) -> None:
        base_url = src.get("url")
        if not base_url:
            return

        async with semaphore:
            try:
                articles = (await get_new_articles(base_url))[:2]
                for art in articles:
                    art_url = art.get("url", "")
                    if not art_url:
                        continue

                    url_hash = art.get("hash") or generate_article_hash(art_url)
                    raw_title = art.get("title", "") or "ព័ត៌មានថ្មី"

                    async with discovered_lock:
                        if url_hash in session_seen_hashes:
                            continue
                        session_seen_hashes.add(url_hash)

                    # Gatekeeper 1: All-time handled check across hash, normalized URL, and raw title
                    if await is_article_handled(url_hash=url_hash, url=art_url, title=raw_title):
                        logger.info("Article already handled (URL/Hash/Title): '%s' (%s)", raw_title[:45], art_url)
                        continue

                    raw_text = art.get("text", "")
                    img_url = art.get("image_url")
                    url_slug = art_url.strip("/").split("/")[-1].replace("-", " ")
                    en_title = f"{raw_title} {url_slug}".strip()

                    # Gatekeeper 2: Cross-source duplicate check on raw title (7-day lookback)
                    is_dup_raw = False
                    match_raw: Any = None
                    sim_raw = 0.0
                    try:
                        dup_res = await find_cross_source_duplicate(
                            candidate_title=raw_title,
                            lookback_hours=168.0,
                            threshold=0.60,
                        )
                        if isinstance(dup_res, tuple):
                            if len(dup_res) == 3:
                                is_dup_raw, match_raw, sim_raw = dup_res
                            elif len(dup_res) == 2:
                                is_dup_raw, match_raw = dup_res
                                sim_raw = 1.0 if is_dup_raw else 0.0
                    except Exception as e:
                        logger.debug("Raw duplicate check failed: %s", e)

                    if is_dup_raw and match_raw:
                        logger.info(
                            "Cross-source duplicate skipped (raw title): '%s' is %.1f%% similar to existing '%s'",
                            raw_title[:45],
                            sim_raw * 100,
                            (match_raw.get("khmer_title") or match_raw.get("title", ""))[:45],
                        )
                        await mark_article_sent(url_hash, url=art_url, title=raw_title)
                        continue

                    # 1. Parallel editorial summary, source verification (by title), and body translation
                    smart_task = generate_smart_article_summary(
                        title=raw_title,
                        body_text=raw_text,
                        url=art_url,
                        source_name=src.get("name") or "",
                    )
                    verify_task = _safe_verify_sources(raw_title)
                    body_trans_task = translate_text_async(raw_text[:2500])

                    smart_res, verification, khmer_body_text = await asyncio.gather(
                        smart_task,
                        verify_task,
                        body_trans_task,
                        return_exceptions=True,
                    )

                    if isinstance(smart_res, Exception):
                        logger.warning("Summary generation failed: %s", smart_res)
                        smart_res = {}
                    if isinstance(verification, Exception):
                        verification = {"verified": False, "sources": 0, "publishers": []}
                    if isinstance(khmer_body_text, Exception):
                        khmer_body_text = ""

                    khmer_title = smart_res.get("badged_title") or smart_res.get("khmer_title") or raw_title
                    clean_km_title = smart_res.get("khmer_title") or raw_title
                    khmer_summary = smart_res.get("sections") or smart_res.get("khmer_summary") or ""
                    hook = smart_res.get("hook") or ""
                    takeaway = smart_res.get("takeaway") or ""
                    category_name = smart_res.get("category") or ""

                    # Gatekeeper 3: Check translated Khmer title against handled database
                    if clean_km_title and await is_article_handled(title=clean_km_title, khmer_title=clean_km_title):
                        logger.info("Article already handled (Khmer title match): '%s'", clean_km_title[:45])
                        await mark_article_sent(url_hash, url=art_url, title=clean_km_title)
                        continue

                    # Gatekeeper 4: Cross-source duplicate check on translated Khmer title
                    if clean_km_title:
                        is_dup_km = False
                        match_km: Any = None
                        sim_km = 0.0
                        try:
                            dup_km_res = await find_cross_source_duplicate(
                                candidate_title="",
                                candidate_khmer_title=clean_km_title,
                                lookback_hours=168.0,
                                threshold=0.62,
                            )
                            if isinstance(dup_km_res, tuple):
                                if len(dup_km_res) == 3:
                                    is_dup_km, match_km, sim_km = dup_km_res
                                elif len(dup_km_res) == 2:
                                    is_dup_km, match_km = dup_km_res
                                    sim_km = 1.0 if is_dup_km else 0.0
                        except Exception as e:
                            logger.debug("Khmer duplicate check failed: %s", e)

                        if is_dup_km and match_km:
                            logger.info(
                                "Cross-source duplicate skipped (Khmer title): '%s' is %.1f%% similar to existing '%s'",
                                clean_km_title[:45],
                                sim_km * 100,
                                (match_km.get("khmer_title") or match_km.get("title", ""))[:45],
                            )
                            await mark_article_sent(url_hash, url=art_url, title=clean_km_title)
                            continue

                    # 3. Categorizer NLP metadata analysis
                    analysis = {}
                    with suppress(Exception):
                        analysis = analyze_article_metadata(
                            km_title=clean_km_title or raw_title,
                            km_text=khmer_summary,
                            en_title=en_title,
                        )

                    # 4. Generate Telegra.ph Instant View
                    telegraph_url = ""
                    with suppress(Exception):
                        telegraph_url = await get_telegraph_url(
                            title=clean_km_title or raw_title,
                            content_text=khmer_body_text or khmer_summary,
                            image_url=img_url,
                            source_url=art_url,
                        )

                    # 5. Build TTS script
                    tts_lines = [
                        l.strip().lstrip("•-*▪►→").strip()
                        for l in khmer_summary.split("\n")
                        if l.strip() and not l.strip().endswith("៖")
                    ]
                    tts_script = f"{clean_km_title}\n" + "\n".join(tts_lines)

                    # 6. Save payload to pending approval queue
                    payload = {
                        "hash": url_hash,
                        "url": art_url,
                        "title": raw_title,
                        "khmer_title": khmer_title,
                        "clean_khmer_title": clean_km_title,
                        "khmer_summary": khmer_summary,
                        "hook": hook,
                        "takeaway": takeaway,
                        "category_name": category_name,
                        "source_name": src.get("name") or "",
                        "khmer_tts_script": tts_script,
                        "original_summary": smart_res.get("original_summary", ""),
                        "body_text": raw_text,
                        "image_url": img_url,
                        "telegraph_url": telegraph_url,
                        "analysis": analysis,
                        "verification": verification,
                        "status": "pending",
                        "created_at": time.time(),
                    }
                    await save_pending_article(payload)
                    store_article_session(url_hash, payload)
                    store_article_session(url_hash[:16], payload)

                    async with discovered_lock:
                        discovered.append(payload)

                    # 7. Notify admin for review
                    if bot and admin_ids:
                        logger.info("Notifying %d admins for new article: %s", len(admin_ids), raw_title[:40])
                        await _send_review_card_to_admins(bot, admin_ids, payload)

            except Exception as scan_err:
                logger.warning("Error scanning source %s: %s", base_url, scan_err)

    await asyncio.gather(*[_scan_single_source(src) for src in active_sources], return_exceptions=True)

    if discovered:
        logger.info("Scan completed: %d new articles submitted to admin review.", len(discovered))
    return discovered


async def _send_review_card_to_admins(bot: Bot, admin_ids: list[int], article: dict[str, Any]) -> None:
    """Send formatted article notification to admins with interactive approval buttons."""
    url_hash = article["hash"]
    khmer_title = article.get("khmer_title") or article.get("title", "")
    khmer_summary = article.get("khmer_summary", "")
    art_url = article.get("url", "")
    telegraph_url = article.get("telegraph_url", "")
    img_url = article.get("image_url")
    analysis = article.get("analysis", {})
    verification = article.get("verification")

    # Safe formatter invocation
    caption, base_markup = _safe_format_article_message(
        km_title=khmer_title,
        km_text=khmer_summary,
        url=art_url,
        analysis=analysis,
        verification=verification,
        telegraph_url=telegraph_url,
        date_str=datetime.now().strftime("%d/%m/%Y"),
        as_html=True,
    )

    # Prepend admin approval action buttons
    short_hash = url_hash[:16]
    buttons: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("✅ យល់ព្រម (Approve)", callback_data=f"art_adm:approve:{short_hash}"),
            InlineKeyboardButton("❌ បដិសេធ (Reject)", callback_data=f"art_adm:reject:{short_hash}"),
        ]
    ]
    if base_markup and getattr(base_markup, "inline_keyboard", None):
        buttons.extend(base_markup.inline_keyboard)

    markup = InlineKeyboardMarkup(buttons)

    # Retrieve cover image or generate dynamic banner
    img_bytes = None
    if img_url:
        with suppress(Exception):
            img_bytes = await fetch_image_bytes(img_url, timeout_s=4.0)

    if not img_bytes:
        with suppress(Exception):
            from app.services.ai.dynamic_card import detect_card_archetype, generate_dynamic_card_banner

            archetype = detect_card_archetype(khmer_title, khmer_summary)
            img_bytes = generate_dynamic_card_banner(
                title=article.get("title") or khmer_title,
                category=article.get("category_name") or "TECH DISPATCH",
                source_name=article.get("source_name") or "ONLINE",
                archetype=archetype,
            )

    for admin_id in admin_ids:
        try:
            if img_bytes and len(caption) <= 1024:
                bio = io.BytesIO(img_bytes)
                bio.name = "cover.jpg"
                await bot.send_photo(
                    chat_id=admin_id,
                    photo=bio,
                    caption=caption,
                    parse_mode="HTML",
                    reply_markup=markup,
                )
            else:
                await bot.send_message(
                    chat_id=admin_id,
                    text=caption,
                    parse_mode="HTML",
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )
        except Exception as send_err:
            logger.warning("Failed sending review card to admin %s: %s; trying plain text fallback", admin_id, send_err)
            with suppress(Exception):
                await bot.send_message(
                    chat_id=admin_id,
                    text=caption,
                    parse_mode="HTML",
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )


async def broadcast_approved_article(bot: Bot | None, article_hash: str) -> tuple[int, int]:
    """Broadcast an admin-approved article publication card to all users and subscribers.

    Returns:
        (success_count, failed_count)
    """
    if bot is None:
        bot = _get_active_bot()
    if not bot:
        logger.error("No active bot available for article broadcast.")
        return 0, 0

    record = await get_pending_article(article_hash)
    if not record:
        logger.error("Pending article %s not found for broadcast.", article_hash)
        return 0, 0

    # Ensure idempotency
    art_url = record.get("url", "")
    khmer_title = record.get("khmer_title") or record.get("title", "")
    await mark_article_sent(article_hash, url=art_url, title=khmer_title)

    # Collect recipient user IDs
    recipients: set[int] = set()
    try:
        from app.legacy import get_all_user_ids

        loop = asyncio.get_running_loop()
        user_ids = await loop.run_in_executor(None, get_all_user_ids)
        recipients.update(user_ids)
    except Exception as u_err:
        logger.debug("Failed getting all user IDs: %s", u_err)

    try:
        from app.services.podcast.store import podcast_store

        subs = podcast_store.get_all_subscribers()
        recipients.update(subs)
    except Exception as s_err:
        logger.debug("Failed getting podcast subscribers: %s", s_err)

    if not recipients:
        logger.info("No recipients available for article broadcast.")
        return 0, 0

    khmer_summary = record.get("khmer_summary", "")
    telegraph_url = record.get("telegraph_url", "")
    img_url = record.get("image_url")
    analysis = record.get("analysis", {})

    caption, base_markup = _safe_format_article_message(
        km_title=khmer_title,
        km_text=khmer_summary,
        url=art_url,
        analysis=analysis,
        verification=record.get("verification"),
        telegraph_url=telegraph_url,
        date_str=datetime.now().strftime("%d/%m/%Y"),
        as_html=True,
    )

    img_bytes = None
    if img_url:
        with suppress(Exception):
            img_bytes = await fetch_image_bytes(img_url, timeout_s=4.0)

    markup = base_markup
    with suppress(Exception):
        from app.services.ai.dynamic_card import (
            build_dynamic_card_keyboard,
            detect_card_archetype,
            generate_dynamic_card_banner,
        )

        archetype = detect_card_archetype(khmer_title, khmer_summary)
        if img_url and not img_bytes:
            img_bytes = generate_dynamic_card_banner(
                title=record.get("title") or khmer_title,
                category=record.get("category_name") or "TECH DISPATCH",
                source_name=record.get("source_name") or "ONLINE",
                archetype=archetype,
            )

        markup = build_dynamic_card_keyboard(
            url=art_url,
            session_id=article_hash,
            archetype=archetype,
            telegraph_url=telegraph_url,
            is_khmer=True,
        )

    success_count = 0
    failed_count = 0
    cached_photo_file_id: str | None = None
    dead_subscribers: set[int] = set()

    logger.info("Broadcasting approved article '%s' to %d recipients...", khmer_title[:30], len(recipients))

    for uid in recipients:
        try:
            if cached_photo_file_id:
                await bot.send_photo(
                    chat_id=uid,
                    photo=cached_photo_file_id,
                    caption=caption,
                    parse_mode="HTML",
                    reply_markup=markup,
                )
            elif img_bytes and len(caption) <= 1024:
                bio = io.BytesIO(img_bytes)
                bio.name = "news.jpg"
                sent_photo = await bot.send_photo(
                    chat_id=uid,
                    photo=bio,
                    caption=caption,
                    parse_mode="HTML",
                    reply_markup=markup,
                )
                if sent_photo and getattr(sent_photo, "photo", None):
                    cached_photo_file_id = sent_photo.photo[-1].file_id
            else:
                await bot.send_message(
                    chat_id=uid,
                    text=caption,
                    parse_mode="HTML",
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )

            success_count += 1
            await asyncio.sleep(0.04)  # ~25 msg/sec rate-limit spacing
        except Exception as exc:
            failed_count += 1
            retry_after = getattr(exc, "retry_after", None)
            if retry_after and isinstance(retry_after, (int, float)):
                await asyncio.sleep(float(retry_after) + 0.5)

            err_str = str(exc).lower()
            if any(k in err_str for k in ("blocked", "deactivated", "chat not found", "user is deactivated")):
                dead_subscribers.add(uid)

    if dead_subscribers:
        with suppress(Exception):
            from app.services.podcast.store import podcast_store

            removed = podcast_store.unsubscribe_batch(dead_subscribers)
            logger.info("Auto-pruned %d inactive broadcast subscribers.", removed)

    logger.info("Article broadcast completed: %d succeeded, %d failed", success_count, failed_count)
    return success_count, failed_count


async def periodic_article_monitor_scheduler(poll_interval: float = 180.0) -> None:
    """Background task to continuously monitor article sources every 3 minutes."""
    effective_interval = float(os.getenv("ARTICLE_MONITOR_INTERVAL_S", poll_interval))
    logger.info("Article base URL monitor scheduler started (interval: %.1fs).", effective_interval)

    # Initial grace delay to allow FastAPI, webhooks, and DB pools to initialize
    await asyncio.sleep(20.0)

    while True:
        try:
            from app.core.features import is_article_reader_enabled

            if is_article_reader_enabled():
                bot = _get_active_bot()
                if bot:
                    await scan_sources_and_notify_admin(bot)
                else:
                    logger.debug("Article monitor: Bot instance not ready yet, skipping cycle.")
        except asyncio.CancelledError:
            logger.info("Article base URL monitor scheduler cancelled.")
            break
        except Exception as exc:
            logger.error("Error in periodic_article_monitor_scheduler: %s", exc, exc_info=True)

        await asyncio.sleep(effective_interval)


send_pending_article_for_review = _send_review_card_to_admins

__all__ = [
    "broadcast_approved_article",
    "periodic_article_monitor_scheduler",
    "scan_sources_and_notify_admin",
    "send_pending_article_for_review",
]