"""News articles narration and digest service."""

from __future__ import annotations

import logging
from typing import Any

from app.services.ai.articles.reader import fetch_and_clean_article
from app.services.ai.articles.summarizer import summarize_article_khmer

logger = logging.getLogger("app.features.podcast.news")


async def get_article_narrated_digest(url: str) -> dict[str, Any] | None:
    """Fetch article, summarize into key points, and prepare narration data."""
    try:
        article = await fetch_and_clean_article(url)
        if not article or not article.get("text"):
            return None
        summary = summarize_article_khmer(article["text"])
        return {
            "title": article.get("title", ""),
            "summary": summary,
            "url": url,
        }
    except Exception as exc:
        logger.warning("Article digest generation failed for %s: %s", url, exc)
        return None


__all__ = ["get_article_narrated_digest"]
