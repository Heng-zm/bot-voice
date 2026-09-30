"""Cross-Source Smart Deduplication Engine.

Detects and filters duplicate news stories across multiple different media sources
using:
  1. Khmer zero-width space and punctuation normalization
  2. Character 3-gram (tri-gram) shingling (optimized for Khmer script)
  3. Token Jaccard overlap and containment similarity
  4. Cross-source historical lookback against sent and pending articles (default 48h)
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Common news publisher branding suffixes and prefixes to clean
_NOISE_PATTERNS = [
    r"[-–|•]\s*(?:fresh\s*news|vod|koh\s*santepheap|rfi|rasmei|thmey\s*thmey|kampuchea\s*thmey|reuters|ap|bbc|cnn|the\s*verge|techcrunch|wired|securityweek).*",
    r"^\s*\[(?:breaking|update|urgent|hot)\]\s*",
    r"^\s*(?:ព័ត៌មានបឋម|ព័ត៌មានទាន់ហេតុការណ៍|ទាន់ហេតុការណ៍|update|breaking)\s*[:：\-–]\s*",
]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]


import functools

@functools.lru_cache(maxsize=2000)
def normalize_title(title: str) -> str:
    """Clean and normalize news title for similarity matching."""
    if not title:
        return ""
    text = str(title).strip()

    # Strip zero-width spaces and control characters
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", text)

    # Strip publisher noise suffixes / prefixes
    for pat in _NOISE_RE:
        text = pat.sub("", text)

    # Normalize whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

@functools.lru_cache(maxsize=2000)
def _extract_word_tokens(text: str) -> frozenset[str]:
    """Extract alphanumeric and Khmer word tokens."""
    tokens = re.findall(r"[\w\u1780-\u17ff]+", text.lower())
    # Filter out very short tokens (<2 chars)
    return frozenset(t for t in tokens if len(t) >= 2)

@functools.lru_cache(maxsize=2000)
def _extract_char_shingles(text: str, n: int = 3) -> frozenset[str]:
    """Extract character n-grams without spaces (ideal for Khmer language)."""
    clean = re.sub(r"\s+", "", text.lower())
    if len(clean) < n:
        return frozenset({clean}) if clean else frozenset()
    return frozenset(clean[i : i + n] for i in range(len(clean) - n + 1))


def compute_headline_similarity(title1: str, title2: str) -> float:
    """Calculate multi-algorithm similarity score between two headlines (0.0 to 1.0).
    
    Combines:
      - Word-level Jaccard similarity
      - Character tri-gram shingle similarity (very strong for Khmer script)
      - Word containment ratio (when one title is a subset of another)
    """
    n1 = normalize_title(title1)
    n2 = normalize_title(title2)

    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    # 1. Word token Jaccard
    w1 = _extract_word_tokens(n1)
    w2 = _extract_word_tokens(n2)
    w_union = len(w1 | w2)
    w_jaccard = (len(w1 & w2) / w_union) if w_union > 0 else 0.0

    # 2. Character tri-gram Jaccard (handles Khmer compound words and morphological variants)
    c1 = _extract_char_shingles(n1, 3)
    c2 = _extract_char_shingles(n2, 3)
    c_union = len(c1 | c2)
    c_jaccard = (len(c1 & c2) / c_union) if c_union > 0 else 0.0

    # 3. Containment (if shorter title is mostly contained in longer title)
    containment = 0.0
    if len(w1) >= 3 and len(w2) >= 3:
        min_len = min(len(w1), len(w2))
        containment = (len(w1 & w2) / min_len) * 0.85

    # Return highest metric
    return max(w_jaccard, c_jaccard, containment)


def find_cross_source_duplicate_sync(
    candidate_title: str,
    candidate_khmer_title: str = "",
    lookback_hours: float = 168.0,
    threshold: float = 0.60,
) -> tuple[bool, dict[str, Any] | None, float]:
    """Check if the candidate article is a duplicate of a recently sent or pending story.
    
    Uses dual-window lookback:
      - Within lookback_hours (default 168h / 7 days): blocks stories with similarity >= threshold (default 0.60)
      - Older historical records (up to 500 records): blocks stories with high similarity >= 0.75
    
    Returns:
        (is_duplicate, matched_record_dict, similarity_score)
    """
    from app.services.ai.article_storage import _db

    if not candidate_title and not candidate_khmer_title:
        return False, None, 0.0

    cutoff = time.time() - (lookback_hours * 3600.0)
    best_score = 0.0
    best_match: dict[str, Any] | None = None

    try:
        with _db() as conn:
            cur = conn.cursor()

            # 1. Fetch recent sent articles (up to 500 records)
            cur.execute(
                """
                SELECT hash, url, title, clean_title, created_at, 'sent' AS record_type
                FROM sent_articles
                ORDER BY created_at DESC
                LIMIT 500
                """
            )
            sent_rows = [dict(r) for r in cur.fetchall()]

            # 2. Fetch pending / reviewed articles (up to 500 records)
            cur.execute(
                """
                SELECT hash, url, title, clean_title, khmer_title, clean_khmer_title, created_at, 'pending' AS record_type
                FROM pending_articles
                ORDER BY created_at DESC
                LIMIT 500
                """
            )
            pending_rows = [dict(r) for r in cur.fetchall()]

        all_records = sent_rows + pending_rows

        for item in all_records:
            stored_title = item.get("title") or ""
            stored_km_title = item.get("khmer_title") or ""
            stored_clean_title = item.get("clean_title") or ""
            stored_clean_km_title = item.get("clean_khmer_title") or ""
            item_created = float(item.get("created_at") or 0.0)

            # Determine required threshold for this item based on age
            # If within lookback window, use standard threshold (e.g. 0.60)
            # If older than lookback window, require high-confidence similarity (>= 0.75)
            item_threshold = threshold if item_created >= cutoff else max(threshold, 0.75)

            current_item_best = 0.0

            # Check candidate original title against stored title & clean_title
            if candidate_title:
                for target_t in (stored_title, stored_clean_title):
                    if target_t:
                        score_raw = compute_headline_similarity(candidate_title, target_t)
                        if score_raw > current_item_best:
                            current_item_best = score_raw

            # Check candidate Khmer title against stored Khmer title & clean_khmer_title
            if candidate_khmer_title:
                for target_kt in (stored_km_title, stored_clean_km_title, stored_title):
                    if target_kt:
                        score_km = compute_headline_similarity(candidate_khmer_title, target_kt)
                        if score_km > current_item_best:
                            current_item_best = score_km

            if current_item_best >= item_threshold and current_item_best > best_score:
                best_score = current_item_best
                best_match = item

            if best_score >= 0.90:
                # Early exit on near-exact match
                break

    except Exception as exc:
        logger.warning("find_cross_source_duplicate_sync error: %s", exc)
        return False, None, 0.0

    is_duplicate = best_match is not None and best_score >= (threshold if (best_match.get("created_at") or 0) >= cutoff else 0.75)
    return is_duplicate, best_match, best_score


async def find_cross_source_duplicate(
    candidate_title: str,
    candidate_khmer_title: str = "",
    lookback_hours: float = 168.0,
    threshold: float = 0.60,
) -> tuple[bool, dict[str, Any] | None, float]:
    """Asynchronous wrapper for cross-source smart deduplication."""
    return await asyncio.to_thread(
        find_cross_source_duplicate_sync,
        candidate_title,
        candidate_khmer_title,
        lookback_hours,
        threshold,
    )


__all__ = [
    "compute_headline_similarity",
    "find_cross_source_duplicate",
    "find_cross_source_duplicate_sync",
    "normalize_title",
]
