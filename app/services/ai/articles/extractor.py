"""Deep Article Extraction & Intelligence Engine.

Multi-tier deep extraction pipeline:
  1. JSON-LD NewsArticle / Article schema parsing (100% clean body extraction)
  2. Trafilatura deep extraction (recall-optimized with table & link filtering)
  3. ReadabilityDocument fallback
  4. Semantic CSS content selector scoring (targeting standard news containers)
  5. Multi-source metadata extraction (Title, high-res Image, Description)
  6. Deep news link crawler with anti-crawler noise exclusion
  7. Resilient CloudScraper + HTTPX with User-Agent rotation
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import hashlib
import json
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse
import warnings

from bs4 import BeautifulSoup
import cloudscraper
import feedparser
import requests
from requests.adapters import HTTPAdapter

# Suppress runtime and package warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# Optional heavy dependencies — graceful fallback if not installed
try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    trafilatura = None
    _HAS_TRAFILATURA = False

try:
    from readability import Document as ReadabilityDocument
    _HAS_READABILITY = True
except ImportError:
    ReadabilityDocument = None
    _HAS_READABILITY = False

try:
    from ddgs import DDGS
    _HAS_DDGS = True
except ImportError:
    DDGS = None
    _HAS_DDGS = False


# ── Scraping & Networking Utilities ──────────────────────────────────────────

_FALLBACK_USER_AGENTS = [
    # Chrome Desktop Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    # Edge Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    # Firefox Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    # Chrome macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    # Safari macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]


def get_scraper():
    """Return a fresh CloudScraper instance with ultra-realistic browser headers."""
    import random

    browsers = ["chrome", "firefox"]
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": random.choice(browsers),
            "platform": "windows",
            "desktop": True,
        }
    )
    scraper.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,km;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
        "DNT": "1",
        "Cache-Control": "max-age=0",
        "Referer": "https://www.google.com/",
    })
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
    scraper.mount("http://", adapter)
    scraper.mount("https://", adapter)
    return scraper


def generate_hash(text: str) -> str:
    """Generate deterministic SHA-256 hash for article URL."""
    from app.services.ai.article_reader import generate_article_hash
    return generate_article_hash(text)


def fetch_url_sync(url: str) -> str | None:
    """Fetch URL with CloudScraper, retrying with rotated User-Agents on failures."""
    attempts = [None] + _FALLBACK_USER_AGENTS
    for ua in attempts:
        try:
            with get_scraper() as scraper:
                if ua:
                    scraper.headers["User-Agent"] = ua
                response = scraper.get(url, timeout=15)
                response.raise_for_status()
                return response.text
        except requests.exceptions.RequestException:
            continue
    return None


async def fetch_url(url: str) -> str | None:
    """Asynchronously fetch URL HTML using httpx first, then cloudscraper fallback."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9,km;q=0.8",
            }
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200 and resp.text:
                return resp.text
            elif resp.status_code not in (403, 503):
                logger.debug("httpx status %s for %s, falling back to cloudscraper", resp.status_code, url)
    except Exception as exc:
        logger.debug("httpx fast path failed for %s: %s; using cloudscraper...", url, exc)

    try:
        return await asyncio.to_thread(fetch_url_sync, url)
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def sanitize_html(html: str) -> str:
    """Remove NULL bytes and XML-incompatible control characters."""
    if not html:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", html)


# ── Deep Article Text Extraction ─────────────────────────────────────────────

# Standard news article body containers in major CMSs (WordPress, Drupal, custom news sites)
_ARTICLE_CONTAINER_SELECTORS = [
    "article",
    "main article",
    "[role='main'] article",
    ".article-content",
    ".entry-content",
    ".post-content",
    ".story-body",
    ".article__body",
    "#article-body",
    ".content-detail",
    ".detail-content",
    ".td-post-content",
    ".post_content",
    ".news-detail",
    ".news-content",
    ".story-content",
    "#content-main",
    ".main-content",
]

# Noise elements to decompose before text extraction
_NOISE_TAGS = [
    "script", "style", "noscript", "svg", "iframe", "form", "button",
    "nav", "header", "footer", "aside", "menu", "dialog",
]

_NOISE_CLASS_KEYWORDS = (
    "ad-", "-ad", "advert", "banner", "related", "share", "sharing",
    "social", "widget", "comment", "breadcrumb", "author", "subscribe",
    "newsletter", "disclaimer", "cookie", "modal", "popup", "meta-info",
)


def _is_noise_element(tag) -> bool:
    """Identify advertisement, social sharing, and widget noise tags."""
    if not tag or not hasattr(tag, "get"):
        return False
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    tag_id = str(tag.get("id", "")).lower()
    for c in classes:
        low_c = str(c).lower()
        if any(k in low_c for k in _NOISE_CLASS_KEYWORDS):
            return True
    if any(k in tag_id for k in _NOISE_CLASS_KEYWORDS):
        return True
    return False


def _extract_from_json_ld(soup: BeautifulSoup) -> tuple[str | None, str | None, str | None]:
    """Attempt deep extraction from Schema.org JSON-LD NewsArticle structured data.
    
    Returns (headline, articleBody, image_url)
    """
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
            candidates = []
            if isinstance(data, list):
                candidates.extend(data)
            elif isinstance(data, dict):
                if "@graph" in data and isinstance(data["@graph"], list):
                    candidates.extend(data["@graph"])
                else:
                    candidates.append(data)

            for item in candidates:
                if not isinstance(item, dict):
                    continue
                type_val = str(item.get("@type", ""))
                if any(t in type_val for t in ("NewsArticle", "Article", "BlogPosting", "ReportageNewsArticle")):
                    body = item.get("articleBody")
                    title = item.get("headline") or item.get("name")
                    image = item.get("image")
                    img_url = None
                    if isinstance(image, str):
                        img_url = image
                    elif isinstance(image, list) and image:
                        img_url = image[0] if isinstance(image[0], str) else image[0].get("url")
                    elif isinstance(image, dict):
                        img_url = image.get("url")

                    if body and len(str(body).strip()) > 100:
                        return (str(title).strip() if title else None, str(body).strip(), img_url)
        except Exception:
            continue
    return None, None, None


def extract_article_text(html: str) -> str:
    """Deep extraction of pure article text from HTML.
    
    Pipeline:
      1. JSON-LD articleBody (purest text, 0% ads)
      2. Trafilatura deep extraction (recall-optimized)
      3. ReadabilityDocument fallback
      4. Semantic CSS content container extraction with noise decomposition
      5. Density-based paragraph extraction
    """
    if not html:
        return ""

    try:
        html = sanitize_html(html)
        soup = BeautifulSoup(html, "lxml")

        # 1. Tier 1: JSON-LD NewsArticle Body
        _, json_ld_body, _ = _extract_from_json_ld(soup)
        if json_ld_body and len(json_ld_body) >= 120:
            return json_ld_body

        # Pre-clean DOM: strip noise tags and noise classes before extraction
        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        for junk in soup.find_all(_is_noise_element):
            junk.decompose()

        cleaned_html = str(soup)

        # 2. Tier 2: Trafilatura Deep Extraction on cleaned HTML
        if _HAS_TRAFILATURA and trafilatura:
            with suppress(Exception):
                text = trafilatura.extract(
                    cleaned_html,
                    include_links=False,
                    include_images=False,
                    include_comments=False,
                    include_tables=False,
                    favor_precision=False,
                    favor_recall=True,
                    deduplicate=True,
                )
                if text and len(text.strip()) >= 120:
                    return text.strip()

        # 3. Tier 3: Readability Document
        if _HAS_READABILITY and ReadabilityDocument:
            with suppress(Exception):
                doc = ReadabilityDocument(html)
                summary_html = doc.summary()
                if summary_html:
                    r_soup = BeautifulSoup(summary_html, "lxml")
                    for t in r_soup(_NOISE_TAGS):
                        t.decompose()
                    text = r_soup.get_text(separator="\n\n").strip()
                    if len(text) >= 120:
                        return text

        # 4. Tier 4: Semantic CSS Container Extraction
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        for sel in _ARTICLE_CONTAINER_SELECTORS:
            container = soup.select_one(sel)
            if container:
                # Decompose noise elements inside container
                for junk in container.find_all(_is_noise_element):
                    junk.decompose()
                paras = [p.get_text(strip=True) for p in container.find_all(["p", "blockquote"])]
                valid_paras = [p for p in paras if len(p) >= 25 and not _is_noise_element(p)]
                if valid_paras and sum(len(p) for p in valid_paras) >= 80:
                    return "\n\n".join(valid_paras)

        # 5. Tier 5: Paragraph Clustering Fallback
        body_tag = soup.body or soup
        paras = [p.get_text(strip=True) for p in body_tag.find_all("p")]
        meaningful = [p for p in paras if len(p) >= 30 and not _is_noise_element(p)]
        if meaningful:
            return "\n\n".join(meaningful)

        # Last resort: raw text
        return soup.get_text(separator="\n").strip()

    except Exception as exc:
        logger.warning("Deep extract_article_text failed: %s", exc)
        return ""


# ── Deep Metadata Extraction ─────────────────────────────────────────────────

_SITE_NAME_CLEAN_RE = re.compile(
    r"\s*[-–|•]\s*(?:fresh\s*news|vod|koh\s*santepheap|rfi|thmey\s*thmey|kampuchea\s*thmey|reuters|ap|bbc|cnn|techcrunch|the\s*verge|wired|securityweek).*",
    re.IGNORECASE,
)


def extract_metadata(html: str, base_url: str) -> tuple[str, str | None]:
    """Extract clean article title and highest-resolution lead image.
    
    Returns:
        (title: str, image_url: str | None)
    """
    title = "ព័ត៌មានថ្មី"
    image_url: str | None = None

    if not html:
        return title, image_url

    try:
        soup = BeautifulSoup(html, "lxml")

        # 1. Check JSON-LD metadata first
        ld_title, _, ld_img = _extract_from_json_ld(soup)
        if ld_title:
            title = ld_title
        if ld_img:
            image_url = urljoin(base_url, ld_img)

        # 2. OpenGraph / Twitter Title
        if title == "ព័ត៌មានថ្មី" or not title:
            og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
            if og_title and og_title.get("content"):
                title = str(og_title["content"]).strip()
            elif soup.find("h1"):
                h1 = soup.find("h1")
                if h1 and h1.get_text(strip=True):
                    title = h1.get_text(strip=True)
            elif soup.title and soup.title.string:
                title = str(soup.title.string).strip()

        # Clean publisher suffixes from title
        if title:
            title = _SITE_NAME_CLEAN_RE.sub("", title).strip()

        # 3. High-Res Image Extraction
        if not image_url:
            og_img = (
                soup.find("meta", property="og:image:secure_url")
                or soup.find("meta", property="og:image")
                or soup.find("meta", attrs={"name": "twitter:image"})
                or soup.find("meta", attrs={"name": "twitter:image:src"})
            )
            if og_img and og_img.get("content"):
                candidate = str(og_img["content"]).strip()
                if candidate and not candidate.startswith("data:") and not candidate.endswith(".svg"):
                    image_url = urljoin(base_url, candidate)

        # Fallback to lead image in content container
        if not image_url:
            for sel in ("article img", ".entry-content img", ".post-content img", ".featured-image img"):
                img = soup.select_one(sel)
                if img and img.get("src"):
                    src = str(img["src"]).strip()
                    if (
                        src
                        and not src.startswith("data:")
                        and any(ext in src.lower() for ext in (".jpg", ".jpeg", ".png", ".webp"))
                        and not any(bad in src.lower() for bad in ("avatar", "logo", "icon", "banner", "pixel", "1x1"))
                    ):
                        image_url = urljoin(base_url, src)
                        break

    except Exception as exc:
        logger.warning("extract_metadata error: %s", exc)

    return title or "ព័ត៌មានថ្មី", image_url


# ── Deep Article Listing Crawler ─────────────────────────────────────────────

_EXCLUDE_PATHS = (
    "/about", "/contact", "/privacy", "/terms", "/login", "/register",
    "/search", "/tag/", "/category/", "/author/", "/page/", "/feed/",
    "/cart", "/checkout", "/wp-admin", "/wp-login", "/archive",
)

_ARTICLE_URL_PATTERNS = [
    re.compile(r"/\d{4}/\d{2}/"),             # Year/Month (e.g. /2026/09/)
    re.compile(r"-\d+\.html?$"),               # ID.html (e.g. news-12345.html)
    re.compile(r"/\d{5,}(?:/|$)"),             # Long numeric ID (e.g. /123456/)
    re.compile(r"/(?:article|news|story|post|detail|report)/", re.IGNORECASE),
]


def is_listing_page(html: str, base_url: str) -> list[str]:
    """Deep heuristic crawler extracting genuine fresh article links from a page."""
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=True)
    base_domain = urlparse(base_url).netloc.replace("www.", "")

    article_links: list[str] = []
    seen: set[str] = set()

    for link in links:
        href = link.get("href", "").strip()
        text = link.get_text(strip=True)

        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        full_url = urljoin(base_url, href).split("?")[0].split("#")[0].rstrip("/")
        if full_url in seen or full_url == base_url.rstrip("/"):
            continue

        link_domain = urlparse(full_url).netloc.replace("www.", "")
        if base_domain not in link_domain:
            continue

        path = urlparse(full_url).path.lower()
        if any(ex in path for ex in _EXCLUDE_PATHS):
            continue

        # Check article signals
        has_news_pattern = any(p.search(path) for p in _ARTICLE_URL_PATTERNS)
        text_len = len(text)
        word_count = len(text.split())

        # Genuine news headline link signals
        if (has_news_pattern or word_count >= 4 or text_len >= 16) and text_len >= 8:
            seen.add(full_url)
            article_links.append(full_url)

    return article_links[:20]


async def get_new_articles(base_url: str) -> list[dict[str, Any]]:
    """Main entry point for deep real-time article discovery and extraction.
    
    Pipeline:
      1. RSS/Atom Feed Discovery & Extraction
      2. Listing Page Crawler with Deep Content Extraction
      3. Single Page Deep Extraction Fallback
    """
    from app.services.ai.article_storage import is_article_handled

    html = await fetch_url(base_url)
    if not html:
        return []

    try:
        # 1. RSS Feed Discovery
        rss_candidates: list[str] = []
        soup = BeautifulSoup(html, "lxml")
        for link_tag in soup.find_all("link", type=lambda t: t and ("rss" in t or "atom" in t)):
            if link_tag.get("href"):
                rss_candidates.append(urljoin(base_url, link_tag["href"]))

        rss_candidates.append(base_url.rstrip("/") + "/feed")
        rss_candidates.append(base_url.rstrip("/") + "/rss")
        rss_candidates.append(base_url.rstrip("/") + "/feed.xml")

        feed_data = None
        for r_url in rss_candidates[:3]:
            feed_xml = await fetch_url(r_url)
            if feed_xml and ("<rss" in feed_xml or "<feed" in feed_xml):
                feed_data = feedparser.parse(feed_xml)
                if feed_data and getattr(feed_data, "entries", None):
                    break

        if feed_data and feed_data.entries:
            articles = []
            for entry in feed_data.entries[:5]:
                link = getattr(entry, "link", None)
                if not link:
                    continue
                clean_link = link.split("?")[0]
                url_hash = generate_hash(clean_link)
                e_title = getattr(entry, "title", "")
                if await is_article_handled(url_hash=url_hash, url=link, title=e_title):
                    continue

                page_html = await fetch_url(link)
                desc = getattr(entry, "description", "") or ""

                if page_html:
                    title, image_url = extract_metadata(page_html, base_url)
                    article_text = extract_article_text(page_html)
                else:
                    title = getattr(entry, "title", "ព័ត៌មានថ្មី")
                    image_url = None
                    article_text = desc

                final_title = title or getattr(entry, "title", "ព័ត៌មានថ្មី")
                if await is_article_handled(url_hash=url_hash, url=link, title=final_title):
                    continue

                articles.append({
                    "url": link,
                    "title": final_title,
                    "text": article_text if article_text and len(article_text) >= 80 else desc,
                    "hash": url_hash,
                    "image_url": image_url,
                })

            if articles:
                return articles

        # 2. Listing Page Crawler
        links = is_listing_page(html, base_url)
        if links:
            unsent_links: list[str] = []
            for link in links:
                h = generate_hash(link.split("?")[0])
                if not await is_article_handled(url_hash=h, url=link):
                    unsent_links.append(link)

            if not unsent_links:
                return []

            logger.info("[%s] Found %d new candidate article links to extract.", base_url, len(unsent_links))

            async def fetch_and_deep_extract(link: str) -> dict[str, Any] | None:
                art_html = await fetch_url(link)
                if art_html:
                    title, image_url = extract_metadata(art_html, link)
                    h = generate_hash(link.split("?")[0])
                    if await is_article_handled(url_hash=h, url=link, title=title):
                        return None
                    text = extract_article_text(art_html)
                    if text and len(text) >= 80:
                        return {
                            "url": link,
                            "title": title,
                            "text": text,
                            "hash": h,
                            "image_url": image_url,
                        }
                return None

            # Concurrently extract up to 3 articles
            tasks = [fetch_and_deep_extract(link) for link in unsent_links[:3]]
            extracted = await asyncio.gather(*tasks, return_exceptions=True)

            results: list[dict[str, Any]] = []
            for item in extracted:
                if isinstance(item, dict) and item.get("text"):
                    results.append(item)
            return results

        # 3. Direct Single Article URL Fallback
        single_hash = generate_hash(base_url.split("?")[0])
        if await is_article_handled(url_hash=single_hash, url=base_url):
            return []

        text = extract_article_text(html)
        title, image_url = extract_metadata(html, base_url)
        if await is_article_handled(url_hash=single_hash, url=base_url, title=title):
            return []

        if text and len(text) >= 80:
            return [{
                "url": base_url,
                "title": title,
                "text": text,
                "hash": single_hash,
                "image_url": image_url,
            }]

        return []

    except Exception as exc:
        logger.error("get_new_articles error for %s: %s", base_url, exc)
        return []


# ── Source Verification via Search Engine ────────────────────────────────────

def verify_article_sources(english_title: str) -> dict:
    """Verify English title against top global news domains using multi-tier search engine."""
    from app.services.ai.search_engine import verify_article_sources_sync
    return verify_article_sources_sync(english_title)


__all__ = [
    "extract_article_text",
    "extract_metadata",
    "fetch_url",
    "fetch_url_sync",
    "generate_hash",
    "get_new_articles",
    "get_scraper",
    "is_listing_page",
    "sanitize_html",
    "verify_article_sources",
]
