"""High-Performance Multi-Tier News Search & Source Verification Engine.

Multi-Tier Architecture:
  1. Google News RSS Search (Zero rate limits, bilingual US/KH localization, authentic publisher metadata)
  2. DuckDuckGo News & Text Search (DDGS) fallback
  3. Bing News RSS fallback
  + Thread-safe in-memory caching with TTL
  + Query optimization & noise keyword stripping
  + Global, Tech, and Regional trusted news publisher verification
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import hashlib
import logging
import re
import ssl
import threading
import time
from typing import Any
import urllib.parse
import urllib.request
import warnings

import feedparser

# Suppress DuckDuckGo package rename warning if duckduckgo_search is loaded
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*renamed to `ddgs`.*")

logger = logging.getLogger(__name__)

# In-memory search cache: hash -> (timestamp, result_dict)
_SEARCH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_TTL_SECONDS = 7200.0  # 2 hours
_MAX_CACHE_SIZE = 1000

# Trusted news publisher domains (Global Wire, Tech/Cybersecurity, Regional & Cambodian)
TRUSTED_DOMAINS: set[str] = {
    # Global Wire & Major Outlets
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "cnn.com",
    "bloomberg.com", "aljazeera.com", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "cnbc.com", "forbes.com", "afp.com", "ft.com",
    "wsj.com", "france24.com", "dw.com", "cna.asia", "channelnewsasia.com",
    "straitstimes.com", "scmp.com", "nikkei.com", "japantimes.co.jp",
    "abc.net.au", "voanews.com", "rfa.org",
    # Tech & Cybersecurity News
    "securityweek.com", "thehackernews.com", "bleepingcomputer.com",
    "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
    "darkreading.com", "zdnet.com", "tomshardware.com", "9to5mac.com",
    "theregister.com", "krebsonsecurity.com", "engadget.com", "venturebeat.com",
    # Regional & Cambodian Media
    "freshnewsasia.com", "kohsantepheapdaily.com.kh", "rfi.fr/km",
    "vodenglish.news", "kampucheathmey.com", "thmeythmey.com",
    "phnompenhpost.com", "postkhmer.com", "khmertimeskh.com",
    "dap-news.com", "voacambodia.com",
}

# Trusted publisher name tokens (for Google News RSS attribution matching)
TRUSTED_PUBLISHER_NAMES: set[str] = {
    "reuters", "associated press", "ap news", "ap", "bbc", "bbc news", "cnn",
    "bloomberg", "al jazeera", "the new york times", "new york times",
    "the washington post", "washington post", "the guardian", "cnbc", "forbes",
    "afp", "agence france-presse", "financial times", "wall street journal",
    "techcrunch", "the verge", "wired", "ars technica", "bleepingcomputer",
    "securityweek", "the hacker news", "zdnet", "tom's hardware", "9to5mac",
    "the register", "krebs on security", "venturebeat", "engadget",
    # Cambodian & Regional Publishers
    "fresh news", "koh santepheap", "rfi", "vod", "kampuchea thmey",
    "thmey thmey", "phnom penh post", "the phnom penh post", "khmer times",
    "voa", "rfa", "radio free asia", "cna",
}

# Resilient DDGS import (prefer modern 'ddgs' package over legacy 'duckduckgo_search')
_HAS_DDGS = False
DDGS: Any = None

try:
    from ddgs import DDGS as _DDGS  # type: ignore[assignment]
    DDGS = _DDGS
    _HAS_DDGS = True
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            from duckduckgo_search import DDGS as _DDGS  # type: ignore[assignment]
        DDGS = _DDGS
        _HAS_DDGS = True
    except ImportError:
        DDGS = None
        _HAS_DDGS = False


def _get_ssl_context() -> ssl.SSLContext:
    """Create a permissive SSL context for resilient RSS fetching in containers."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _is_khmer_text(text: str) -> bool:
    """Check if the text contains Khmer script characters."""
    return bool(re.search(r"[\u1780-\u17FF]", text))


def clean_search_query(title: str) -> str:
    """Prepare optimized search keywords from article title."""
    if not title:
        return ""
    text = str(title).strip()

    # Strip publisher suffixes (e.g. - Fresh News, | SecurityWeek, — Reuters)
    text = re.sub(r"\s*[-–—|•]\s*[\w\s.]+$", "", text)

    # Strip punctuation and brackets
    text = re.sub(r"[\[\](){}\"':;!?«»]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Bound word count for English queries
    if not _is_khmer_text(text):
        words = text.split()
        if len(words) > 9:
            text = " ".join(words[:9])
    else:
        # Khmer titles: truncate if excessively long
        if len(text) > 120:
            text = text[:120].strip()

    return text


def _search_google_news_rss(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Tier 1: Google News RSS Search — Fast, free, authentic publisher metadata."""
    clean_q = clean_search_query(query)
    if not clean_q:
        return []

    encoded = urllib.parse.quote(clean_q)
    # Localize query parameters based on language script
    if _is_khmer_text(clean_q):
        url = f"https://news.google.com/rss/search?q={encoded}&hl=km-KH&gl=KH&ceid=KH:km"
    else:
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        },
    )

    try:
        ctx = _get_ssl_context()
        with urllib.request.urlopen(req, timeout=6.0, context=ctx) as resp:
            data = resp.read()

        feed = feedparser.parse(data)
        results: list[dict[str, Any]] = []

        for entry in feed.entries[:max_results]:
            source_info = getattr(entry, "source", None) or {}
            publisher = str(source_info.get("title") or "Unknown").strip()
            source_url = str(source_info.get("url") or "").strip()
            link = getattr(entry, "link", "")
            title = getattr(entry, "title", "")

            results.append({
                "title": title,
                "url": link,
                "source_url": source_url,
                "publisher": publisher,
                "source_engine": "google_news",
            })
        return results
    except Exception as exc:
        logger.debug("Google News RSS search failed for '%s': %s", clean_q, exc)
        return []


def _search_ddgs(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Tier 2: DuckDuckGo news & web search fallback."""
    if not _HAS_DDGS or not DDGS:
        return []

    clean_q = clean_search_query(query)
    if not clean_q:
        return []

    results: list[dict[str, Any]] = []

    # Attempt 1: DDGS News search
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.news(clean_q, max_results=max_results))
            for hit in hits:
                url = hit.get("url") or hit.get("href", "")
                title = hit.get("title", "")
                pub = hit.get("source") or urllib.parse.urlparse(url).netloc.replace("www.", "")
                results.append({
                    "title": title,
                    "url": url,
                    "publisher": pub,
                    "source_engine": "ddgs_news",
                })
        if results:
            return results
    except Exception as exc:
        logger.debug("DDGS news search failed: %s; trying text search", exc)

    # Attempt 2: DDGS Text search fallback
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(clean_q, max_results=max_results))
            for hit in hits:
                url = hit.get("href", "")
                title = hit.get("title", "")
                domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
                results.append({
                    "title": title,
                    "url": url,
                    "publisher": domain,
                    "source_engine": "ddgs",
                })
        return results
    except Exception as exc:
        logger.debug("DDGS text search failed for '%s': %s", clean_q, exc)
        return []


def _search_bing_news_rss(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Tier 3: Bing News RSS fallback."""
    clean_q = clean_search_query(query)
    if not clean_q:
        return []

    encoded = urllib.parse.quote(clean_q)
    url = f"https://www.bing.com/news/search?q={encoded}&format=rss"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )

    try:
        ctx = _get_ssl_context()
        with urllib.request.urlopen(req, timeout=6.0, context=ctx) as resp:
            data = resp.read()

        feed = feedparser.parse(data)
        results: list[dict[str, Any]] = []

        for entry in feed.entries[:max_results]:
            link = getattr(entry, "link", "")
            domain = urllib.parse.urlparse(link).netloc.replace("www.", "")
            source_info = getattr(entry, "source", None) or {}
            publisher = str(source_info.get("title") or getattr(entry, "author", "") or domain or "Bing News").strip()

            results.append({
                "title": getattr(entry, "title", ""),
                "url": link,
                "publisher": publisher,
                "source_engine": "bing_news",
            })
        return results
    except Exception as exc:
        logger.debug("Bing News RSS search failed for '%s': %s", clean_q, exc)
        return []


def search_web_sync(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Multi-tier search execution with automatic failover."""
    # 1. Try Google News RSS
    res = _search_google_news_rss(query, max_results)
    if res:
        return res

    # 2. Try DuckDuckGo
    res = _search_ddgs(query, max_results)
    if res:
        return res

    # 3. Try Bing News RSS
    return _search_bing_news_rss(query, max_results)


async def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Asynchronous wrapper for search_web_sync."""
    return await asyncio.to_thread(search_web_sync, query, max_results)


def _is_publisher_trusted(pub_name: str, url: str, source_url: str = "") -> bool:
    """Evaluate whether a news publisher matches recognized trusted outlets."""
    clean_pub = pub_name.lower().strip()
    clean_url = url.lower().strip()
    clean_source_url = source_url.lower().strip()

    # 1. Match publisher name tokens
    for trusted_name in TRUSTED_PUBLISHER_NAMES:
        if trusted_name in clean_pub or clean_pub == trusted_name:
            return True

    # 2. Match target and origin source URLs against trusted domains
    for domain in TRUSTED_DOMAINS:
        if domain in clean_url or (clean_source_url and domain in clean_source_url):
            return True

    return False


def verify_article_sources_sync(title: str) -> dict[str, Any]:
    """Verify news story across global and tech publishers with thread-safe caching.

    Returns:
        {
            "verified": bool,
            "is_verified": bool,
            "sources": int,
            "publishers": list[str],
            "matched_urls": list[str],
            "engine": str,
        }
    """
    clean_title = str(title or "").strip()
    if len(clean_title) < 5:
        return {
            "verified": False,
            "is_verified": False,
            "sources": 0,
            "publishers": [],
            "matched_urls": [],
            "engine": "none",
        }

    cache_key = hashlib.sha256(clean_title.lower().encode("utf-8")).hexdigest()[:16]
    now = time.time()

    # Thread-safe cache check
    with _CACHE_LOCK:
        if cache_key in _SEARCH_CACHE:
            cached_time, cached_res = _SEARCH_CACHE[cache_key]
            if (now - cached_time) < _CACHE_TTL_SECONDS:
                return dict(cached_res)

    results = search_web_sync(clean_title, max_results=6)
    found_trusted_publishers: set[str] = set()
    found_urls: list[str] = []

    for item in results:
        url = item.get("url", "")
        source_url = item.get("source_url", "")
        pub = item.get("publisher", "Unknown")

        if _is_publisher_trusted(pub, url, source_url):
            found_trusted_publishers.add(pub if pub != "Unknown" else "Verified Source")
            found_urls.append(url)

    verified_count = len(found_trusted_publishers)
    res: dict[str, Any] = {
        "verified": verified_count > 0,
        "is_verified": verified_count > 0,
        "sources": verified_count,
        "publishers": sorted(found_trusted_publishers),
        "matched_urls": found_urls[:3],
        "engine": results[0].get("source_engine", "none") if results else "none",
    }

    # Thread-safe cache eviction & update
    with _CACHE_LOCK:
        if len(_SEARCH_CACHE) >= _MAX_CACHE_SIZE:
            for k in list(_SEARCH_CACHE.keys())[:150]:
                _SEARCH_CACHE.pop(k, None)
        _SEARCH_CACHE[cache_key] = (now, res)

    return res


async def verify_article_sources(title: str) -> dict[str, Any]:
    """Asynchronous verification running in background thread pool."""
    return await asyncio.to_thread(verify_article_sources_sync, title)


def clear_search_cache() -> int:
    """Clear all in-memory search verification cache entries."""
    with _CACHE_LOCK:
        count = len(_SEARCH_CACHE)
        _SEARCH_CACHE.clear()
        return count


def get_search_cache_stats() -> dict[str, Any]:
    """Return metrics on active search verification cache."""
    with _CACHE_LOCK:
        return {
            "cached_entries": len(_SEARCH_CACHE),
            "max_size": _MAX_CACHE_SIZE,
            "ttl_seconds": _CACHE_TTL_SECONDS,
        }


__all__ = [
    "TRUSTED_DOMAINS",
    "TRUSTED_PUBLISHER_NAMES",
    "clean_search_query",
    "clear_search_cache",
    "get_search_cache_stats",
    "search_web",
    "search_web_sync",
    "verify_article_sources",
    "verify_article_sources_sync",
]