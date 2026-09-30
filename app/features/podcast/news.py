"""News articles narration and digest service."""

from __future__ import annotations

import logging
from typing import Any

from app.services.ai.articles.reader import extract_article_content_with_image, fetch_article_html
from app.services.ai.articles.summarizer import extractive_summary

logger = logging.getLogger("app.features.podcast.news")


async def get_article_narrated_digest(url: str) -> dict[str, Any] | None:
    """Fetch article, summarize into key points, and prepare narration data."""
    try:
        html = await fetch_article_html(url)
        if not html:
            return None
        title, text, image_url = extract_article_content_with_image(html, url)
        if not text:
            return None
        summary = extractive_summary(text)
        return {
            "title": title,
            "summary": summary,
            "image_url": image_url,
            "url": url,
        }
    except Exception as exc:
        logger.warning("Article digest generation failed for %s: %s", url, exc)
        return None


__all__ = ["get_article_narrated_digest"]
