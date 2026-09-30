"""Web Link Article Narrator service.

Provides SSRF-protected URL fetching, clean HTML article text extraction,
and AI-powered news summarization for voice audio synthesis.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import html
import ipaddress
import logging
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)


def _get_article_ssl_context(verify: bool = True) -> ssl.SSLContext:
    """Create robust SSL context supporting SECLEVEL=1 for regional Cambodian news sites."""
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    except Exception:
        pass
    return ctx


BROWSER_USER_AGENTS = [
    # Modern Chrome Desktop
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    # Social crawler (whitelisted by Cloudflare, Incapsula, and publisher firewalls)
    "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
    # Telegram link preview bot
    "Mozilla/5.0 (compatible; TelegramBot/1.0; +https://core.telegram.org/bots/webpages)",
    # Googlebot crawler fallback
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
]

USER_AGENT = BROWSER_USER_AGENTS[0]

MAX_ARTICLE_BYTES = 2 * 1024 * 1024  # 2MB max download
MAX_ARTICLE_CHARS = 15_000  # Cap extracted text length
MIN_ARTICLE_CHARS = 60  # Minimum readable text to be considered an article

_async_dns_lock: asyncio.Lock | None = None
_async_dns_loop: asyncio.AbstractEventLoop | None = None
_thread_dns_lock = threading.Lock()


def _get_async_dns_lock() -> asyncio.Lock:
    """Retrieve loop-aware asyncio lock for DNS-pinning context."""
    global _async_dns_lock, _async_dns_loop
    current_loop = asyncio.get_running_loop()
    if _async_dns_lock is None or _async_dns_loop != current_loop:
        _async_dns_lock = asyncio.Lock()
        _async_dns_loop = current_loop
    return _async_dns_lock


_UNSAFE_HOSTNAMES = frozenset({
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "instance-data",
})


def _is_unsafe_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """SSRF detection: checks private, loopback, multicast, link-local, and cloud metadata IPs."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    ip_str = str(ip)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or ip_str.startswith("169.254.")
        or ip_str == "100.100.100.200"  # Alibaba Cloud metadata IP
    )


def _resolve_and_validate_host(hostname: str) -> list[str]:
    """Resolve `hostname` and return validated public IP strings."""
    lower_host = hostname.lower().strip(".")
    if lower_host in _UNSAFE_HOSTNAMES:
        raise ValueError("Access to private or loopback host is forbidden.")

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as err:
        raise ValueError(f"Could not resolve host '{hostname}': {err}") from err

    safe_ips: list[str] = []
    for item in addr_info:
        ip_str = item[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_unsafe_ip(ip):
            raise ValueError(f"Access to private IP '{ip_str}' is forbidden.")
        if ip_str not in safe_ips:
            safe_ips.append(ip_str)

    if not safe_ips:
        raise ValueError(f"Host '{hostname}' did not resolve to any usable address.")

    return safe_ips


def is_safe_public_url(url: str) -> tuple[bool, str]:
    """Validate that a URL uses HTTP/HTTPS and does not target private networks."""
    clean_url = str(url or "").strip()
    if not clean_url:
        return False, "URL is empty."

    try:
        parsed = urllib.parse.urlparse(clean_url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False, "Only HTTP and HTTPS URLs are allowed."

        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL host."

        _resolve_and_validate_host(hostname)
        return True, ""
    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"URL validation failed: {exc}"


@contextlib.asynccontextmanager
async def _pinned_dns_async(hostname: str, validated_ips: list[str]):
    """Temporarily pin socket.getaddrinfo to validated public IPs during async fetch."""
    lock = _get_async_dns_lock()
    async with lock:
        original_getaddrinfo = socket.getaddrinfo
        target_host = hostname

        def _pinned_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            if host == target_host:
                results = []
                for ip_str in validated_ips:
                    ip_obj = ipaddress.ip_address(ip_str)
                    if ip_obj.version == 6:
                        if family in (socket.AF_UNSPEC, socket.AF_INET6):
                            results.append(
                                (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip_str, port, 0, 0))
                            )
                    else:
                        if family in (socket.AF_UNSPEC, socket.AF_INET):
                            results.append(
                                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip_str, port))
                            )
                if results:
                    return results
            return original_getaddrinfo(host, port, family, type, proto, flags)

        socket.getaddrinfo = _pinned_getaddrinfo
        try:
            yield
        finally:
            socket.getaddrinfo = original_getaddrinfo


@contextlib.contextmanager
def _pinned_dns_sync(hostname: str, validated_ips: list[str]):
    """Thread-safe synchronous context manager for worker thread fallbacks."""
    with _thread_dns_lock:
        original_getaddrinfo = socket.getaddrinfo
        target_host = hostname

        def _pinned_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            if host == target_host:
                results = []
                for ip_str in validated_ips:
                    ip_obj = ipaddress.ip_address(ip_str)
                    if ip_obj.version == 6:
                        if family in (socket.AF_UNSPEC, socket.AF_INET6):
                            results.append(
                                (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip_str, port, 0, 0))
                            )
                    else:
                        if family in (socket.AF_UNSPEC, socket.AF_INET):
                            results.append(
                                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip_str, port))
                            )
                if results:
                    return results
            return original_getaddrinfo(host, port, family, type, proto, flags)

        socket.getaddrinfo = _pinned_getaddrinfo
        try:
            yield
        finally:
            socket.getaddrinfo = original_getaddrinfo


_pinned_dns = _pinned_dns_sync


class ArticleHTMLParser(HTMLParser):
    """Clean HTML parser that extracts title, content, and lead image."""

    SKIP_TAGS: frozenset[str] = frozenset({
        "script", "style", "noscript", "nav", "footer", "header",
        "aside", "form", "svg", "button", "iframe", "dialog", "select", "input",
    })

    BLOCK_TAGS: frozenset[str] = frozenset({
        "p", "div", "article", "section", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "blockquote", "tr", "br",
    })

    BOILERPLATE_PATTERNS: tuple[str, ...] = (
        "share", "social", "related", "comment", "sidebar", "widget",
        "breadcrumb", "newsletter", "advertisement", "ad-box", "ads",
        "banner", "author-bio", "pop-up", "modal", "fn-news-related",
        "content-related", "sabay-news-related",
    )

    BOILERPLATE_PRESERVE_HINTS: tuple[str, ...] = (
        "article-body", "post-body", "article-content", "post-content",
        "entry-content", "main-content", "content-detail",
    )

    KHMER_BOILERPLATE_PREFIXES: tuple[str, ...] = (
        "ចុច like", "ចុច subscribe", "join telegram", "តាមដាន telegram",
        "អានអត្ថបទពាក់ព័ន្ធ", "អត្ថបទពេញនិយម", "អានបន្ត:", "share on facebook",
        "follow us on", "រក្សាសិទ្ធិដោយ", "all rights reserved", "សូមទស្សនាវីដេអូ",
    )

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth: int = 0
        self._skip_tags: list[str] = []
        self._in_title: bool = False
        self.title: str = ""
        self.og_title: str = ""
        self.og_description: str = ""
        self.og_image: str = ""
        self.lead_image: str = ""
        self.paragraphs: list[str] = []
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low_tag = tag.lower()
        if low_tag in self.SKIP_TAGS:
            self._skip_depth += 1
            self._skip_tags.append(low_tag)
            return

        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        classes_and_id = f"{attr_dict.get('class', '')} {attr_dict.get('id', '')}".lower()
        if classes_and_id and any(pat in classes_and_id for pat in self.BOILERPLATE_PATTERNS):
            if not any(hint in classes_and_id for hint in self.BOILERPLATE_PRESERVE_HINTS):
                self._skip_depth += 1
                self._skip_tags.append(low_tag)
                return

        if low_tag == "title":
            self._in_title = True
            return

        if low_tag == "meta":
            prop = attr_dict.get("property", "").lower()
            name = attr_dict.get("name", "").lower()
            content = attr_dict.get("content", "").strip()
            if content:
                if (prop == "og:title" or name == "twitter:title") and not self.og_title:
                    self.og_title = content
                elif (
                    prop in ("og:description", "description") or name in ("description", "twitter:description")
                ) and not self.og_description:
                    self.og_description = content
                elif (
                    prop in ("og:image", "twitter:image", "image")
                    or name in ("og:image", "twitter:image", "image", "thumbnail", "twitter:image:src")
                ) and not self.og_image:
                    self.og_image = content
            return

        if low_tag == "link":
            rel = attr_dict.get("rel", "").lower()
            href = attr_dict.get("href", "").strip()
            if rel in ("image_src", "apple-touch-icon") and href and not self.og_image:
                self.og_image = href
            return

        if low_tag == "img" and not self.lead_image:
            src = attr_dict.get("src", "").strip()
            if src and not src.startswith("data:") and any(ext in src.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
                self.lead_image = src
            return

        if low_tag in self.BLOCK_TAGS:
            self._flush_current()

    def handle_endtag(self, tag: str) -> None:
        low_tag = tag.lower()
        if self._skip_tags and self._skip_tags[-1] == low_tag:
            self._skip_tags.pop()
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if low_tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if low_tag == "title":
            self._in_title = False
            return

        if low_tag in self.BLOCK_TAGS:
            self._flush_current()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
            return
        text = data.strip()
        if text:
            self._current_text.append(data)

    def _flush_current(self) -> None:
        if self._current_text:
            text = " ".join("".join(self._current_text).split())
            if len(text) >= 20:
                self.paragraphs.append(text)
            self._current_text.clear()

    def get_clean_content(self) -> tuple[str, str]:
        self._flush_current()
        title = self.og_title or self.title or ""
        title = " ".join(html.unescape(title).split())

        clean_paras: list[str] = []
        for p in self.paragraphs:
            cleaned = html.unescape(p).strip()
            low_p = cleaned.lower()
            if any(low_p.startswith(bp) for bp in self.KHMER_BOILERPLATE_PREFIXES):
                continue
            if cleaned and cleaned not in clean_paras:
                clean_paras.append(cleaned)

        body = "\n\n".join(clean_paras)
        if not body and self.og_description:
            body = html.unescape(self.og_description).strip()

        return title, body[:MAX_ARTICLE_CHARS]

    def get_lead_image(self) -> str | None:
        return self.og_image or self.lead_image or None


def sanitize_html(html_text: str) -> str:
    """Remove NULL bytes and control characters from HTML."""
    if not html_text:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", html_text)


def generate_article_hash(url: str) -> str:
    """Generate SHA-256 hash for article URL to ensure idempotency."""
    clean = (url or "").split("?")[0].split("#")[0].rstrip("/").strip().lower()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def extract_article_content(html_content: str) -> tuple[str, str]:
    """Parse HTML string and extract clean title and body text."""
    title, body, _ = extract_article_content_with_image(html_content)
    return title, body


def extract_article_content_with_image(html_content: str, base_url: str = "") -> tuple[str, str, str | None]:
    """Parse HTML string and extract clean title, body text, and lead image URL."""
    parser = ArticleHTMLParser()
    try:
        parser.feed(html_content)
    except Exception as exc:
        logger.debug("HTML parser error: %s", exc)
    title, fallback_body = parser.get_clean_content()
    img_url = parser.get_lead_image()
    if img_url and base_url:
        img_url = urllib.parse.urljoin(base_url, img_url)

    smart_body = ""
    clean_html = sanitize_html(html_content)

    # 1. Primary: trafilatura
    try:
        import trafilatura

        extracted = trafilatura.extract(
            clean_html,
            include_links=False,
            include_images=False,
            include_comments=False,
        )
        if extracted and len(extracted.strip()) >= 80:
            smart_body = extracted.strip()
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)

    # 2. Secondary: BeautifulSoup fallback
    if not smart_body:
        try:
            summary_html = clean_html
            with contextlib.suppress(Exception):
                from readability import Document

                doc = Document(clean_html)
                summary_html = doc.summary()

            from bs4 import BeautifulSoup

            try:
                soup = BeautifulSoup(summary_html, "lxml")
            except Exception:
                soup = BeautifulSoup(summary_html, "html.parser")

            for s in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                s.decompose()
            text = soup.get_text(separator="\n").strip()
            if text and len(text) >= 80:
                smart_body = text
        except Exception as exc:
            logger.debug("secondary extraction failed: %s", exc)

    chosen_body = smart_body if (smart_body and len(smart_body) > len(fallback_body)) else fallback_body

    clean_paras: list[str] = []
    for line in chosen_body.split("\n"):
        c_line = html.unescape(line).strip()
        low_line = c_line.lower()
        if any(low_line.startswith(bp) for bp in ArticleHTMLParser.KHMER_BOILERPLATE_PREFIXES):
            continue
        if c_line and c_line not in clean_paras:
            clean_paras.append(c_line)

    final_body = "\n\n".join(clean_paras)
    if not final_body and parser.og_description:
        final_body = html.unescape(parser.og_description).strip()

    return title, final_body[:MAX_ARTICLE_CHARS], img_url


def is_listing_page(html_text: str, base_url: str) -> list[str]:
    """Extract article links from a news listing page or homepage."""
    if not html_text or not base_url:
        return []
    try:
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            soup = BeautifulSoup(html_text, "html.parser")

        links = soup.find_all("a", href=True)
        base_domain = urllib.parse.urlparse(base_url).netloc.replace("www.", "").lower()
        if not base_domain:
            return []

        article_links: list[str] = []
        seen: set[str] = set()
        excluded = (
            "/about", "/contact", "/privacy", "/terms", "/login", "/register",
            "/search", "/tag/", "/category/", "/author/", "/video", "/media",
            "/photos", "/live", "/signin", "/signup", "/account", "/subscribe",
            "/feed", "/rss",
        )

        for link in links:
            href = (link.get("href") or "").strip()
            text = (link.get_text() or "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = urllib.parse.urljoin(base_url, href).split("?")[0].split("#")[0].rstrip("/")
            if full_url in seen or full_url == base_url.rstrip("/"):
                continue

            parsed_full = urllib.parse.urlparse(full_url)
            link_domain = parsed_full.netloc.replace("www.", "").lower()
            if base_domain not in link_domain:
                continue

            path = parsed_full.path.lower()
            if any(ex in path for ex in excluded):
                continue

            word_count = len(text.split())
            char_count = len(text)
            has_news_slug = bool(re.search(r"/(news|article|story|post|detail|\d{4}/\d{2}|\d+)/?", path))

            if (word_count >= 3 or char_count >= 12 or has_news_slug) and char_count >= 6:
                seen.add(full_url)
                article_links.append(full_url)

        return article_links[:15]
    except Exception as exc:
        logger.warning("is_listing_page error for %s: %s", base_url, exc)
        return []


def extract_feed_links(html_text: str, base_url: str) -> list[str]:
    """Find RSS/Atom feed links in HTML or return standard candidates."""
    candidates = []
    if html_text:
        try:
            from bs4 import BeautifulSoup

            try:
                soup = BeautifulSoup(html_text, "lxml")
            except Exception:
                soup = BeautifulSoup(html_text, "html.parser")

            for link_tag in soup.find_all("link", type=True):
                t = (link_tag.get("type") or "").lower()
                h = (link_tag.get("href") or "").strip()
                if ("rss" in t or "atom" in t or "xml" in t) and h:
                    candidates.append(urllib.parse.urljoin(base_url, h))
        except Exception:
            pass

    clean_base = base_url.rstrip("/")
    for default_path in ("/feed", "/rss", "/rss.xml", "/feed.xml"):
        candidates.append(f"{clean_base}{default_path}")
    return candidates


async def get_new_articles_from_url(base_url: str, limit: int = 3) -> list[dict[str, Any]]:
    """Scan base_url for new unsent articles via RSS or listing page."""
    from app.services.ai.article_storage import is_article_handled

    try:
        html_data = await fetch_article_html(base_url)
    except Exception as exc:
        logger.warning("Failed to fetch %s for listing check: %s", base_url, exc)
        return []

    # 1. Try RSS feed parsing
    feed_candidates = extract_feed_links(html_data, base_url)
    for feed_url in feed_candidates[:3]:
        try:
            feed_xml = await fetch_article_html(feed_url, timeout_s=6.0)
            if feed_xml and ("<rss" in feed_xml or "<feed" in feed_xml):
                import feedparser

                feed = feedparser.parse(feed_xml)
                if feed and getattr(feed, "entries", None):
                    articles = []
                    for entry in feed.entries[:8]:
                        link = getattr(entry, "link", None)
                        if not link:
                            continue
                        clean_link = link.split("?")[0].split("#")[0].rstrip("/")
                        url_hash = generate_article_hash(clean_link)
                        e_title = getattr(entry, "title", "ព័ត៌មានថ្មី")
                        if await is_article_handled(url_hash=url_hash, url=clean_link, title=e_title):
                            continue
                        desc = getattr(entry, "description", "")
                        try:
                            p_html = await fetch_article_html(clean_link, timeout_s=8.0)
                            p_title, p_text, p_img = extract_article_content_with_image(p_html, base_url=clean_link)
                        except Exception:
                            p_title, p_text, p_img = e_title, desc, None

                        final_title = p_title or e_title
                        if await is_article_handled(url_hash=url_hash, url=clean_link, title=final_title):
                            continue

                        articles.append({
                            "url": clean_link,
                            "title": final_title,
                            "text": p_text if (p_text and len(p_text) >= MIN_ARTICLE_CHARS) else desc,
                            "hash": url_hash,
                            "image_url": p_img,
                            "is_listing": True,
                        })
                        if len(articles) >= limit:
                            break
                    if articles:
                        return articles
        except Exception as f_err:
            logger.debug("RSS check failed for %s: %s", feed_url, f_err)

    # 2. Check if page is an HTML listing page
    links = is_listing_page(html_data, base_url)
    if links:
        unsent_links = []
        for l in links:
            h = generate_article_hash(l)
            if not await is_article_handled(url_hash=h, url=l):
                unsent_links.append(l)

        if not unsent_links:
            logger.info("Listing page %s has no unsent articles", base_url)
            return []

        articles = []
        for l in unsent_links[:limit]:
            try:
                page_html = await fetch_article_html(l, timeout_s=8.0)
                p_title, p_text, p_img = extract_article_content_with_image(page_html, base_url=l)
                final_t = p_title or "ព័ត៌មានថ្មី"
                h_l = generate_article_hash(l)
                if await is_article_handled(url_hash=h_l, url=l, title=final_t):
                    continue
                if p_text and len(p_text) >= MIN_ARTICLE_CHARS:
                    articles.append({
                        "url": l,
                        "title": final_t,
                        "text": p_text,
                        "hash": h_l,
                        "image_url": p_img,
                        "is_listing": True,
                    })
            except Exception as item_err:
                logger.debug("Failed fetching listing item %s: %s", l, item_err)

        if articles:
            return articles

    # 3. Not a listing page; treat as single article
    single_hash = generate_article_hash(base_url)
    if await is_article_handled(url_hash=single_hash, url=base_url):
        return []

    title, body, img = extract_article_content_with_image(html_data, base_url=base_url)
    if await is_article_handled(url_hash=single_hash, url=base_url, title=title):
        return []

    if body and len(body) >= MIN_ARTICLE_CHARS:
        return [{
            "url": base_url,
            "title": title or "ព័ត៌មានថ្មី",
            "text": body,
            "hash": single_hash,
            "image_url": img,
            "is_listing": False,
        }]

    return []


async def fetch_image_bytes(image_url: str, timeout_s: float = 8.0, max_bytes: int = 4 * 1024 * 1024) -> bytes | None:
    """Download image bytes safely from a public HTTP/HTTPS URL with SSRF protection."""
    clean_url = str(image_url or "").strip()
    if not clean_url:
        return None
    safe, _ = is_safe_public_url(clean_url)
    if not safe:
        return None

    parsed = urllib.parse.urlparse(clean_url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        return None

    try:
        validated_ips = _resolve_and_validate_host(parsed.hostname)
    except Exception:
        return None

    loop = asyncio.get_running_loop()

    class _SafeImageRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            is_safe, _ = is_safe_public_url(newurl)
            if not is_safe:
                return None
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    def _sync_fetch() -> bytes | None:
        with _pinned_dns_sync(parsed.hostname, validated_ips):
            opener = urllib.request.build_opener(_SafeImageRedirect)
            req = urllib.request.Request(
                clean_url,
                headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*"},
            )
            try:
                with opener.open(req, timeout=timeout_s) as resp:
                    data = resp.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        return None
                    if data.startswith((b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8")):
                        return data
                    content_type = resp.headers.get("Content-Type", "").lower()
                    if content_type.startswith("image/"):
                        return data
            except Exception as exc:
                logger.debug("fetch_image_bytes failed: %s", exc)
                return None
        return None

    try:
        return await loop.run_in_executor(None, _sync_fetch)
    except Exception:
        return None


async def fetch_article_html(url: str, timeout_s: float = 12.0, max_redirects: int = 5) -> str:
    """Fetch webpage HTML safely with strict SSRF validation and multi-agent WAF bypass fallbacks."""
    current_url = str(url or "").strip()
    parsed_initial = urllib.parse.urlparse(current_url)
    if parsed_initial.scheme.lower() not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")
    if not parsed_initial.hostname:
        raise ValueError("Invalid URL host.")

    try:
        validated_ips = _resolve_and_validate_host(parsed_initial.hostname)
    except ValueError as exc:
        raise ValueError(f"Security validation error: {exc}") from exc

    headers_profiles = [
        {
            "User-Agent": BROWSER_USER_AGENTS[0],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,km;q=0.8",
            "Sec-Ch-Ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        },
        {
            "User-Agent": BROWSER_USER_AGENTS[1],
            "Accept": "*/*",
        },
        {
            "User-Agent": BROWSER_USER_AGENTS[2],
            "Accept": "*/*",
        },
        {
            "User-Agent": BROWSER_USER_AGENTS[3],
            "Accept": "*/*",
        },
    ]

    last_error: Exception | None = None

    try:
        import httpx

        ssl_ctx = _get_article_ssl_context(verify=True)
        for profile in headers_profiles:
            try:
                async with httpx.AsyncClient(
                    follow_redirects=False,
                    timeout=timeout_s,
                    verify=ssl_ctx,
                    headers=profile,
                ) as client:
                    hops = 0
                    req_url = current_url
                    req_host = parsed_initial.hostname
                    req_ips = validated_ips
                    while True:
                        async with _pinned_dns_async(req_host, req_ips):
                            resp = await client.get(req_url)
                        if resp.is_redirect:
                            hops += 1
                            if hops > max_redirects:
                                raise ValueError(f"Exceeded maximum allowed redirects ({max_redirects}).")
                            location = resp.headers.get("location")
                            if not location:
                                raise ValueError("Redirect response missing Location header.")
                            req_url = urllib.parse.urljoin(req_url, str(location or ""))
                            parsed_next = urllib.parse.urlparse(req_url)
                            if parsed_next.scheme.lower() not in ("http", "https") or not parsed_next.hostname:
                                raise ValueError(f"Invalid redirect target '{req_url}'.")
                            try:
                                req_ips = _resolve_and_validate_host(parsed_next.hostname)
                            except ValueError as exc:
                                raise ValueError(
                                    f"Security validation error on redirect to '{req_url}': {exc}"
                                ) from exc
                            req_host = parsed_next.hostname
                            continue

                        if resp.status_code in (401, 403, 503):
                            logger.info(
                                "HTTP %d received with agent '%s'; trying fallback crawler",
                                resp.status_code,
                                profile.get("User-Agent", "")[:35],
                            )
                            last_error = getattr(httpx, "HTTPStatusError", Exception)(
                                f"HTTP {resp.status_code}", request=resp.request, response=resp
                            )
                            break

                        resp.raise_for_status()
                        raw_bytes = resp.content
                        encoding = resp.encoding or "utf-8"
                        return raw_bytes.decode(encoding, errors="replace")[:MAX_ARTICLE_BYTES]
            except Exception as err:
                err_str = str(err).lower()
                if (
                    "certificate verify failed" in err_str
                    or "certificate key too weak" in err_str
                    or "ssl" in type(err).__name__.lower()
                ):
                    logger.info("SSL legacy key/cipher for %s; retrying with permissive SSL context", req_host)
                    try:
                        async with httpx.AsyncClient(
                            follow_redirects=False,
                            timeout=timeout_s,
                            verify=_get_article_ssl_context(verify=False),
                            headers=profile,
                        ) as unverified_client:
                            async with _pinned_dns_async(req_host, req_ips):
                                retry_resp = await unverified_client.get(req_url)
                            if not retry_resp.is_redirect:
                                retry_resp.raise_for_status()
                                return retry_resp.content.decode(
                                    retry_resp.encoding or "utf-8", errors="replace"
                                )[:MAX_ARTICLE_BYTES]
                    except Exception as retry_err:
                        last_error = retry_err
                        continue
                last_error = err
                continue

        # Final fallback: cloudscraper if available
        try:
            from app.services.ai.extractor import fetch_url

            logger.info("All httpx crawler profiles failed for %s. Attempting cloudscraper fallback.", url)
            fallback_html = await fetch_url(url)
            if fallback_html:
                return fallback_html[:MAX_ARTICLE_BYTES]
        except Exception as cs_err:
            logger.debug("Cloudscraper fallback failed: %s", cs_err)

        if last_error:
            raise last_error
        raise RuntimeError("Failed to retrieve article content after trying all crawler profiles.")

    except (ImportError, ModuleNotFoundError):

        class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
            def __init__(self) -> None:
                self.hops = 0
                super().__init__()

            def redirect_request(self, req, fp, code, msg, headers, newurl):
                self.hops += 1
                if self.hops > max_redirects:
                    raise urllib.error.HTTPError(
                        req.full_url, code, f"Exceeded max redirects ({max_redirects})", headers, fp
                    )
                target = urllib.parse.urljoin(req.full_url, newurl)
                parsed_target = urllib.parse.urlparse(target)
                if parsed_target.scheme.lower() not in ("http", "https") or not parsed_target.hostname:
                    raise urllib.error.HTTPError(target, 403, "SSRF blocked: invalid redirect target", headers, fp)
                try:
                    _resolve_and_validate_host(parsed_target.hostname)
                except ValueError as err_reason:
                    raise urllib.error.HTTPError(target, 403, f"SSRF blocked: {err_reason}", headers, fp) from err_reason
                return super().redirect_request(req, fp, code, msg, headers, target)

        def _fetch_urllib() -> str:
            for ua in BROWSER_USER_AGENTS:
                for verify_ssl in (True, False):
                    try:
                        ctx = _get_article_ssl_context(verify=verify_ssl)
                        https_handler = urllib.request.HTTPSHandler(context=ctx)
                        opener = urllib.request.build_opener(_SafeRedirectHandler(), https_handler)
                        req = urllib.request.Request(
                            current_url,
                            headers={"User-Agent": ua, "Accept": "text/html,*/*"},
                        )
                        with _pinned_dns(parsed_initial.hostname, validated_ips), opener.open(req, timeout=timeout_s) as response:
                            data = response.read(MAX_ARTICLE_BYTES)
                            encoding = response.headers.get_content_charset() or "utf-8"
                            return data.decode(encoding, errors="replace")
                    except urllib.error.HTTPError as h_err:
                        if h_err.code in (401, 403, 503):
                            break
                        raise
                    except Exception as u_err:
                        u_str = str(u_err).lower()
                        if ("certificate verify failed" in u_str or "certificate key too weak" in u_str) and verify_ssl:
                            continue
                        break
            raise RuntimeError("Urllib failed to retrieve article content.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _fetch_urllib)


# In-memory thread-safe article session cache
_ARTICLE_SESSIONS: dict[str, dict[str, Any]] = {}
_ARTICLE_SESSION_TTL: float = 3600.0  # 1 hour
_article_session_lock = threading.Lock()


def store_article_session(session_id: str, data: dict[str, Any]) -> None:
    """Store extracted and translated article data in memory for interactive callbacks."""
    if not session_id:
        return
    now = time.time()
    with _article_session_lock:
        expired = [
            k for k, v in list(_ARTICLE_SESSIONS.items())
            if now - v.get("_ts", 0) > _ARTICLE_SESSION_TTL
        ]
        for k in expired:
            _ARTICLE_SESSIONS.pop(k, None)
        if len(_ARTICLE_SESSIONS) > 300:
            oldest = sorted(_ARTICLE_SESSIONS.items(), key=lambda x: x[1].get("_ts", 0))[:60]
            for k, _ in oldest:
                _ARTICLE_SESSIONS.pop(k, None)
        payload = dict(data)
        payload["_ts"] = now
        _ARTICLE_SESSIONS[session_id] = payload


def get_article_session(session_id: str) -> dict[str, Any] | None:
    """Retrieve cached article session data by session ID."""
    if not session_id:
        return None
    with _article_session_lock:
        session = _ARTICLE_SESSIONS.get(session_id)
        if not session:
            return None
        if time.time() - session.get("_ts", 0) > _ARTICLE_SESSION_TTL:
            _ARTICLE_SESSIONS.pop(session_id, None)
            return None
        return dict(session)


def make_article_session_id(user_id: int, url: str) -> str:
    """Generate deterministic short session ID for Telegram callback data."""
    raw = f"{user_id}:{url.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def get_conflict_map_info(text: str = "", url: str = "") -> dict[str, str] | None:
    """Inspect article content and return active war/conflict live tracking map if relevant."""
    sample = f"{text} {url}".lower()

    # 1. Russia-Ukraine War
    if any(k in sample for k in ("ukraine", "russia", "kyiv", "kharkiv", "donetsk", "kursk", "អ៊ុយក្រែន", "រុស្ស៊ី", "សង្គ្រាមរុស្ស៊ី")):
        return {
            "label": "🗺️ ផែនទីសមរភូមិអ៊ុយក្រែន (Live Map)",
            "url": "https://deepstatemap.live/en",
        }

    # 2. Middle East / Israel / Gaza / Lebanon / Iran
    if any(k in sample for k in ("gaza", "israel", "lebanon", "hezbollah", "hamas", "iran", "beirut", "អ៊ីស្រាអែល", "ហ្កាហ្សា", "លីបង់", "ហេស្បូឡា")):
        return {
            "label": "🗺️ ផែនទីជម្លោះមជ្ឈិមបូព៌ា (Live Map)",
            "url": "https://israelpalestine.liveuamap.com",
        }

    # 3. Myanmar Conflict
    if any(k in sample for k in ("myanmar", "burma", "junta", "tatmadaw", "ភូមា", "មីយ៉ាន់ម៉ា")):
        return {
            "label": "🗺️ ផែនទីជម្លោះមីយ៉ាន់ម៉ា (Live Map)",
            "url": "https://myanmar.liveuamap.com",
        }

    # 4. Sudan Conflict
    if any(k in sample for k in ("sudan", "khartoum", "rsf", "saf", "ស៊ូដង់")):
        return {
            "label": "🗺️ ផែនទីជម្លោះស៊ូដង់ (Live Map)",
            "url": "https://sudan.liveuamap.com",
        }

    return None


class AwaitableDict(dict):
    """Dict subclass that is directly awaitable, bridging sync and async callers seamlessly."""

    def __await__(self):
        async def _coro():
            return dict(self)

        return _coro().__await__()


def generate_smart_article_summary_sync(
    title: str,
    body_text: str,
    url: str = "",
    source_name: str = "",
) -> dict[str, Any]:
    """Generate structured studio-grade Khmer article summary matching modern media design."""
    if not body_text:
        return {
            "category": "ព័ត៌មាន",
            "badged_title": f"🗞️ {title}",
            "khmer_title": title,
            "hook": "",
            "sections": "",
            "takeaway": "",
            "khmer_summary": "",
            "original_summary": "",
        }

    # 1. Hugging Face Qwen 2.5 Serverless Primary if configured
    with contextlib.suppress(Exception):
        from app.services.ai.hf_client import is_hf_available, summarize_with_qwen_sync

        if is_hf_available():
            qwen_res = summarize_with_qwen_sync(body_text, title=title)
            if qwen_res and qwen_res.get("badged_title") and qwen_res.get("sections"):
                logger.info("Generated smart article summary using Hugging Face Qwen 2.5")
                return {
                    "category": qwen_res.get("category", "ព័ត៌មានទូទៅ"),
                    "badged_title": qwen_res["badged_title"],
                    "khmer_title": qwen_res.get("khmer_title", title),
                    "hook": qwen_res.get("hook", ""),
                    "sections": qwen_res.get("sections", ""),
                    "takeaway": qwen_res.get("takeaway", ""),
                    "khmer_summary": qwen_res.get("khmer_summary", ""),
                    "original_summary": body_text[:400],
                }

    # 2. Local fallback using extractive summarizer and article translator
    from app.services.ai.summarizer import extractive_summary

    translate_fn = None
    with contextlib.suppress(Exception):
        from app.services.ai.article_translator import translate_text_sync
        translate_fn = translate_text_sync

    if not translate_fn:
        with contextlib.suppress(Exception):
            from app.services.ai.article_translator import translate_text
            translate_fn = lambda t, **kw: str(translate_text(t, **kw))

    paras = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    paragraph_excerpt = "\n\n".join(paras[:2]) if len(paras) >= 2 else body_text[:800]

    try:
        summary_excerpt = extractive_summary(body_text, sentences_count=3, as_bullets=False)
        if not summary_excerpt:
            summary_excerpt = paragraph_excerpt
    except Exception:
        summary_excerpt = paragraph_excerpt

    try:
        km_title = translate_fn(title, target_lang="km") if translate_fn else title
        km_summary_raw = translate_fn(summary_excerpt, target_lang="km") if translate_fn else summary_excerpt
    except Exception:
        km_title = title
        km_summary_raw = summary_excerpt

    # Detect archetype for badge formatting
    lower_t = f"{title} {km_title}".lower()
    if any(w in lower_t for w in ("scam", "បោក", "លួច", "fake", "phishing", "hack")):
        badged_title = f"🛡️ [ប្រយ័ត្នបោក]៖ {km_title} ⚠️"
        category = "សន្តិសុខ & បច្ចេកវិទ្យា"
        takeaway = "សូមពិនិត្យប្រភព និងឈ្មោះអ្នកទទួលឱ្យបានច្បាស់លាស់មុនពេលចុច ឬផ្ទេរប្រាក់។"
    elif any(w in lower_t for w in ("breaking", "war", "attack", "president", "minister", "alert", "ទាន់ហេតុការណ៍")):
        badged_title = f"🚨 [BREAKING]៖ {km_title} 🔥"
        category = "ព័ត៌មានទាន់ហេតុការណ៍"
        takeaway = "ស្ថានភាពនេះកំពុងត្រូវបានតាមដានយ៉ាងយកចិត្តទុកដាក់ដោយស្ថាប័នពាក់ព័ន្ធ។"
    elif any(w in lower_t for w in ("tool", "app", "course", "ai", "openai", "gpt", "gemini")):
        badged_title = f"🤖 [AI in 60s]៖ {km_title} ⚡"
        category = "បច្ចេកវិទ្យាថ្ងៃនេះ"
        takeaway = "បច្ចេកវិទ្យាថ្មីនេះអាចជួយបង្កើនប្រសិទ្ធភាពការងារ និងការសិក្សាប្រចាំថ្ងៃរបស់អ្នក។"
    else:
        badged_title = f"📰 [ព័ត៌មានថ្មី]៖ {km_title} 📌"
        category = "ព័ត៌មានទូទៅ"
        takeaway = "តាមដានព័ត៌មានលម្អិតបន្ថែមតាមរយៈតំណភ្ជាប់ដើមនៃអត្ថបទ។"

    lines = [
        l.strip().lstrip("•-*▪►→").strip()
        for l in km_summary_raw.replace("|||", "\n").split("\n")
        if l.strip()
    ]
    hook = lines[0] if lines else km_title
    bullet_lines = [f"• {line}" for line in lines[1:]] if len(lines) > 1 else [f"• {hook}"]
    sections = "ចំណុចសំខាន់ៗ៖\n" + "\n".join(bullet_lines)

    return {
        "category": category,
        "badged_title": badged_title,
        "khmer_title": km_title,
        "hook": hook,
        "sections": sections,
        "takeaway": takeaway,
        "khmer_summary": f"{hook}\n\n{sections}",
        "original_summary": paragraph_excerpt,
    }


async def generate_smart_article_summary(
    title: str,
    body_text: str,
    url: str = "",
    source_name: str = "",
) -> dict[str, Any]:
    """Async wrapper for generate_smart_article_summary_sync."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        generate_smart_article_summary_sync,
        title,
        body_text,
        url,
        source_name,
    )


def summarize_and_translate_for_khmer(
    title: str,
    body_text: str,
    gemini_client: Any = None,
    **kwargs: Any,
) -> AwaitableDict:
    """Analyze news article, detect language, and produce broadcast-ready Khmer summary and audio script."""
    if not body_text:
        return AwaitableDict({
            "orig_lang": "en",
            "orig_lang_name": "English",
            "orig_lang_flag": "🇺🇳",
            "is_khmer": False,
            "khmer_title": title,
            "original_title": title,
            "khmer_summary": "",
            "khmer_tts_script": "",
            "original_summary": "",
            "body_text": "",
        })

    is_km = any(ord(c) >= 0x1780 and ord(c) <= 0x17FF for c in body_text[:500])

    if gemini_client is not None:
        raw_text = ""
        prompt = (
            "Summarize and translate this article into Khmer.\n"
            f"Title: {title}\n"
            f"Content: {body_text}"
        )
        try:
            if hasattr(gemini_client, "models") and hasattr(gemini_client.models, "generate_content"):
                resp = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            else:
                from app.services.ai.gemini import generate_content_with_fallback
                resp = generate_content_with_fallback(gemini_client, contents=prompt)
            raw_text = getattr(resp, "text", "") or ""
        except Exception:
            raw_text = ""

        if raw_text:
            km_title = title
            km_summary = ""
            orig_summary = ""
            current_sec = None
            summary_lines = []
            orig_lines = []
            for line in raw_text.strip().splitlines():
                l = line.strip()
                if l.startswith("KHMER_TITLE:"):
                    km_title = l.split("KHMER_TITLE:", 1)[1].strip()
                    current_sec = None
                elif l.startswith("KHMER_SUMMARY:"):
                    current_sec = "khmer_summary"
                elif l.startswith("ORIGINAL_SUMMARY:"):
                    current_sec = "orig_summary"
                elif current_sec == "khmer_summary":
                    if l:
                        summary_lines.append(l)
                elif current_sec == "orig_summary":
                    if l:
                        orig_lines.append(l)
            if summary_lines:
                km_summary = "\n".join(summary_lines).strip()
            if orig_lines:
                orig_summary = "\n".join(orig_lines).strip()

            tts_lines = [
                l.strip().lstrip("•-*▪►→").strip()
                for l in km_summary.split("\n")
                if l.strip() and not l.strip().endswith("៖")
            ]
            khmer_tts_script = f"{km_title}\n" + "\n".join(tts_lines)

            return AwaitableDict({
                "orig_lang": "km" if is_km else "en",
                "orig_lang_name": "Khmer" if is_km else "English",
                "orig_lang_flag": "🇰🇭" if is_km else "🇺🇳",
                "is_khmer": is_km,
                "khmer_title": km_title,
                "original_title": title,
                "khmer_summary": km_summary,
                "khmer_tts_script": khmer_tts_script,
                "khmer_body_text": body_text,
                "original_summary": orig_summary or body_text[:300],
                "body_text": body_text,
                "category": "ព័ត៌មាន",
                "hook": tts_lines[0] if tts_lines else km_title,
                "sections": km_summary,
                "takeaway": "",
            })

    smart = generate_smart_article_summary_sync(
        title=title,
        body_text=body_text,
    )

    km_title = smart.get("badged_title") or smart.get("khmer_title") or title
    km_summary = smart.get("khmer_summary") or ""
    hook = smart.get("hook") or ""
    sections = smart.get("sections") or ""
    takeaway = smart.get("takeaway") or ""
    category = smart.get("category") or "ព័ត៌មាន"

    tts_lines = [
        l.strip().lstrip("•-*▪►→").strip()
        for l in km_summary.split("\n")
        if l.strip() and not l.strip().endswith("៖")
    ]
    khmer_tts_script = f"{smart.get('khmer_title') or title}\n" + "\n".join(tts_lines)

    return AwaitableDict({
        "orig_lang": "km" if any(ord(c) >= 0x1780 and ord(c) <= 0x17FF for c in body_text[:200]) else "en",
        "orig_lang_name": "Khmer" if any(ord(c) >= 0x1780 and ord(c) <= 0x17FF for c in body_text[:200]) else "English",
        "orig_lang_flag": "🇰🇭" if any(ord(c) >= 0x1780 and ord(c) <= 0x17FF for c in body_text[:200]) else "🇺🇳",
        "is_khmer": any(ord(c) >= 0x1780 and ord(c) <= 0x17FF for c in body_text[:200]),
        "khmer_title": km_title,
        "original_title": title,
        "khmer_summary": km_summary,
        "khmer_tts_script": khmer_tts_script,
        "khmer_body_text": body_text,
        "original_summary": smart.get("original_summary") or body_text[:300],
        "body_text": body_text,
        "category": category,
        "hook": hook,
        "sections": sections,
        "takeaway": takeaway,
    })


def summarize_article_with_ai(
    title: str,
    body_text: str,
    gemini_client: Any = None,
    preferred_model: str = "gemini-2.5-flash",
    *args: Any,
    **kwargs: Any,
) -> str:
    """Generate spoken audio-friendly news bulletin summary from article text, optionally using Gemini."""
    if gemini_client is not None:
        with contextlib.suppress(Exception):
            prompt = (
                "You are an executive newsreader for Bot Voice Cambodia. "
                "Summarize this news article into 3 clear, natural spoken bullet points in Khmer:\n\n"
                f"Title: {title}\nContent:\n{body_text[:3000]}"
            )
            if hasattr(gemini_client, "models") and hasattr(gemini_client.models, "generate_content"):
                resp = gemini_client.models.generate_content(model=preferred_model, contents=prompt)
            else:
                from app.services.ai.gemini import generate_content_with_fallback
                resp = generate_content_with_fallback(gemini_client, contents=prompt, preferred_model=preferred_model)
            from app.services.ai.gemini import extract_gemini_text
            ai_text = extract_gemini_text(resp)
            if not ai_text and hasattr(resp, "text"):
                ai_text = resp.text
            if ai_text and len(ai_text.strip()) > 10:
                return ai_text.strip()

    paras = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    if paras:
        return "\n\n".join(paras[:2])
    return body_text[:600]


def summarize_url_with_ai(
    url: str,
    gemini_client: Any = None,
    preferred_model: str = "gemini-2.5-flash",
    *args: Any,
    **kwargs: Any,
) -> tuple[str, str]:
    """Summarize an article URL directly with Gemini Grounding if available, or fetch and summarize."""
    clean_url = str(url or "").strip()
    if gemini_client is not None:
        with contextlib.suppress(Exception):
            from app.services.ai.gemini import extract_gemini_text, generate_content_with_fallback

            prompt = (
                "Read the news article at this URL and provide a clean Khmer summary. "
                "Return the Khmer Title on Line 1, followed by a concise 3-bullet summary in Khmer:\n"
                f"{clean_url}"
            )
            resp = generate_content_with_fallback(gemini_client, contents=prompt, preferred_model=preferred_model)
            ai_text = extract_gemini_text(resp)
            if ai_text and "\n" in ai_text:
                parts = ai_text.strip().split("\n", 1)
                return parts[0].strip(), parts[1].strip()

    return "", ""


__all__ = [
    "BROWSER_USER_AGENTS",
    "MAX_ARTICLE_BYTES",
    "MAX_ARTICLE_CHARS",
    "MIN_ARTICLE_CHARS",
    "USER_AGENT",
    "extract_article_content",
    "extract_article_content_with_image",
    "extract_feed_links",
    "fetch_article_html",
    "fetch_image_bytes",
    "generate_article_hash",
    "generate_smart_article_summary",
    "generate_smart_article_summary_sync",
    "get_article_session",
    "get_conflict_map_info",
    "get_new_articles_from_url",
    "is_listing_page",
    "is_safe_public_url",
    "make_article_session_id",
    "sanitize_html",
    "store_article_session",
    "summarize_and_translate_for_khmer",
    "summarize_article_with_ai",
    "summarize_url_with_ai",
]