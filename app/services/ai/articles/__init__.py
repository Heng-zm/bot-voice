"""News & Article Intelligence Package.

Provides automated monitoring, web crawling, Khmer script detection,
AI summarization, translation, and admin review broadcasting.
"""

from __future__ import annotations

from app.services.ai.articles.categorizer import analyze_article_metadata
from app.services.ai.articles.deduplicator import find_cross_source_duplicate
from app.services.ai.articles.dynamic_card import (
    build_dynamic_card_keyboard,
    detect_card_archetype,
    generate_dynamic_card_banner,
)
from app.services.ai.articles.extractor import get_new_articles, verify_article_sources
from app.services.ai.articles.monitor import (
    broadcast_approved_article,
    scan_sources_and_notify_admin,
    send_pending_article_for_review,
)
from app.services.ai.articles.reader import (
    fetch_article_html,
    fetch_image_bytes,
    generate_article_hash,
    generate_smart_article_summary,
    get_article_session,
    is_safe_public_url,
    make_article_session_id,
    store_article_session,
    summarize_and_translate_for_khmer,
    summarize_article_with_ai,
)
from app.services.ai.articles.storage import (
    add_article_source,
    get_article_sources,
    get_pending_article,
    is_article_handled,
    is_article_pending,
    is_article_sent,
    mark_article_sent,
    remove_article_source,
    save_pending_article,
    update_pending_status,
)
from app.services.ai.articles.summarizer import extractive_summary
from app.services.ai.articles.telegraph import get_telegraph_url
from app.services.ai.articles.translator import (
    is_khmer,
    translate_chunk,
    translate_text,
    translate_text_sync,
)

__all__ = [
    "add_article_source",
    "analyze_article_metadata",
    "broadcast_approved_article",
    "build_dynamic_card_keyboard",
    "detect_card_archetype",
    "extractive_summary",
    "fetch_article_html",
    "fetch_image_bytes",
    "find_cross_source_duplicate",
    "generate_article_hash",
    "generate_dynamic_card_banner",
    "generate_smart_article_summary",
    "get_article_session",
    "get_article_sources",
    "get_new_articles",
    "get_pending_article",
    "get_telegraph_url",
    "is_article_handled",
    "is_article_pending",
    "is_article_sent",
    "is_khmer",
    "is_safe_public_url",
    "make_article_session_id",
    "mark_article_sent",
    "remove_article_source",
    "save_pending_article",
    "scan_sources_and_notify_admin",
    "send_pending_article_for_review",
    "store_article_session",
    "summarize_and_translate_for_khmer",
    "summarize_article_with_ai",
    "translate_chunk",
    "translate_text",
    "translate_text_sync",
    "update_pending_status",
    "verify_article_sources",
]
