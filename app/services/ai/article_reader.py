"""Web Link Article Narrator service.

Provides SSRF-protected URL fetching, clean HTML article text extraction,
and AI-powered news summarization for voice audio synthesis.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import ipaddress
import logging
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)

BROWSER_USER_AGENTS = [
    # Modern Chrome Desktop
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    # Social crawler (whitelisted by Cloudflare, Incapsula, and major publisher firewalls)
    "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
    # Telegram link preview bot (widely whitelisted by news publishers)
    "Mozilla/5.0 (compatible; TelegramBot/1.0; +https://core.telegram.org/bots/webpages)",
    # Googlebot crawler fallback
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
]

USER_AGENT = BROWSER_USER_AGENTS[0]

MAX_ARTICLE_BYTES = 2 * 1024 * 1024  # 2MB max download
MAX_ARTICLE_CHARS = 15_000  # Cap extracted text length
MIN_ARTICLE_CHARS = 60  # Minimum readable text to be considered an article

# Dedicated locks for DNS-pinning (see `_pinned_dns_async` and `_pinned_dns_sync`).
# Async requests use asyncio.Lock to avoid blocking the event loop thread;
# worker threads use threading.Lock.
_async_dns_lock: asyncio.Lock | None = None
_thread_dns_lock = threading.Lock()


def _get_async_dns_lock() -> asyncio.Lock:
    global _async_dns_lock
    if _async_dns_lock is None:
        _async_dns_lock = asyncio.Lock()
    return _async_dns_lock

_UNSAFE_HOSTNAMES = frozenset({
    "localhost",
    "127.0.0.1",
    "0.0.0.0",  # noqa: S104
    "::1",
    "metadata.google.internal",
    "instance-data",
})


def _is_unsafe_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or str(ip) == "169.254.169.254"
    )


def _resolve_and_validate_host(hostname: str) -> list[str]:
    """Resolve `hostname` and return the list of safe, public IP strings it maps to.

    Raises ValueError if the hostname is disallowed, fails to resolve, or every
    resolved address is private/internal. This is the single source of truth
    for "is this host safe" — both the public `is_safe_public_url` check and
    the actual outbound request use IPs produced by this function so there is
    no gap between validation-time and request-time DNS resolution.
    """
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
    """Validate that a URL uses HTTP/HTTPS and does not target private/internal networks (SSRF defense).

    Note: this performs its own DNS resolution for a quick pre-flight check.
    Because DNS can change between this check and the actual request (DNS
    rebinding), callers that go on to fetch the URL must not rely on this
    function alone — use `_resolve_and_validate_host` + `_pinned_dns` (see
    `fetch_article_html`) so the exact IP that was validated is the exact IP
    that gets connected to.
    """
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
    """Temporarily force `socket.getaddrinfo(hostname, ...)` to return only
    `validated_ips` during an async fetch without blocking the asyncio event loop.
    """
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
                        results.append(
                            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip_str, port, 0, 0))
                        )
                    else:
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
                        results.append(
                            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip_str, port, 0, 0))
                        )
                    else:
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
    """Clean HTML parser that extracts title and main textual content while dropping scripts, styles, and chrome."""

    SKIP_TAGS: frozenset[str] = frozenset({
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "svg",
        "button",
        "iframe",
        "dialog",
        "select",
        "input",
    })

    BLOCK_TAGS: frozenset[str] = frozenset({
        "p",
        "div",
        "article",
        "section",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "tr",
        "br",
    })

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth: int = 0
        self._in_title: bool = False
        self.title: str = ""
        self.og_title: str = ""
        self.og_description: str = ""
        self.paragraphs: list[str] = []
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low_tag = tag.lower()
        if low_tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if low_tag == "title":
            self._in_title = True
            return

        if low_tag == "meta":
            attr_dict = {k.lower(): (v or "") for k, v in attrs}
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
            return

        if low_tag in self.BLOCK_TAGS:
            self._flush_current()

    def handle_endtag(self, tag: str) -> None:
        low_tag = tag.lower()
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
            if len(text) >= 20:  # Skip trivial micro-fragments
                self.paragraphs.append(text)
            self._current_text.clear()

    def get_clean_content(self) -> tuple[str, str]:
        self._flush_current()
        title = self.og_title or self.title or ""
        title = " ".join(html.unescape(title).split())

        # Clean paragraphs
        clean_paras: list[str] = []
        for p in self.paragraphs:
            cleaned = html.unescape(p).strip()
            if cleaned and cleaned not in clean_paras:
                clean_paras.append(cleaned)

        body = "\n\n".join(clean_paras)
        if not body and self.og_description:
            body = html.unescape(self.og_description).strip()

        return title, body[:MAX_ARTICLE_CHARS]


def extract_article_content(html_content: str) -> tuple[str, str]:
    """Parse HTML string and extract clean title and body text."""
    parser = ArticleHTMLParser()
    try:
        parser.feed(html_content)
    except Exception as exc:
        logger.debug("HTML parser error (continuing with partial content): %s", exc)
    return parser.get_clean_content()


async def fetch_article_html(url: str, timeout_s: float = 12.0, max_redirects: int = 5) -> str:
    """Fetch webpage HTML safely with strict SSRF validation and multi-agent WAF bypass fallbacks.

    Every hostname involved (the original URL and any redirect target) is
    resolved and validated exactly once via `_resolve_and_validate_host`, and
    the resulting IP list is pinned with `_pinned_dns` before the actual
    network call is made. This guarantees the connection lands on one of the
    IPs we validated, even if the attacker's DNS server would answer a later,
    independent lookup differently (DNS rebinding).
    """
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
            "User-Agent": BROWSER_USER_AGENTS[1],  # facebookexternalhit (Cloudflare & news WAF bypass)
            "Accept": "*/*",
        },
        {
            "User-Agent": BROWSER_USER_AGENTS[2],  # Telegram link preview bot
            "Accept": "*/*",
        },
        {
            "User-Agent": BROWSER_USER_AGENTS[3],  # Googlebot crawler fallback
            "Accept": "*/*",
        },
    ]

    last_error: Exception | None = None

    try:
        import httpx

        for profile in headers_profiles:
            try:
                async with httpx.AsyncClient(
                    follow_redirects=False,
                    timeout=timeout_s,
                    verify=True,
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
                            req_url = urllib.parse.urljoin(req_url, location)
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

                        # If blocked by WAF/Cloudflare (401, 403, 503), try next crawler profile
                        if resp.status_code in (401, 403, 503):
                            logger.info(
                                "HTTP %d received with agent '%s'; trying fallback crawler",
                                resp.status_code,
                                profile.get("User-Agent", "")[:35],
                            )
                            last_error = httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                            break

                        resp.raise_for_status()
                        content_type = resp.headers.get("content-type", "").lower()
                        if "text/html" not in content_type and "text/plain" not in content_type and "xhtml" not in content_type:
                            logger.debug("Content-Type '%s' not HTML/text, attempting parse anyway", content_type)
                        return resp.text[:MAX_ARTICLE_BYTES]
            except (httpx.HTTPStatusError, httpx.RequestError) as err:
                last_error = err
                continue

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
                    raise urllib.error.HTTPError(req.full_url, code, f"Exceeded max redirects ({max_redirects})", headers, fp)
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
            # Runs in a worker thread via run_in_executor; the DNS pin lock is a
            # threading.Lock so it correctly serializes against the async path too.
            for ua in BROWSER_USER_AGENTS:
                try:
                    opener = urllib.request.build_opener(_SafeRedirectHandler())
                    req = urllib.request.Request(  # noqa: S310
                        current_url,
                        headers={"User-Agent": ua, "Accept": "text/html,*/*"},
                    )
                    with _pinned_dns(parsed_initial.hostname, validated_ips):
                        with opener.open(req, timeout=timeout_s) as response:  # noqa: S310
                            data = response.read(MAX_ARTICLE_BYTES)
                            encoding = response.headers.get_content_charset() or "utf-8"
                            return data.decode(encoding, errors="replace")
                except urllib.error.HTTPError as h_err:
                    if h_err.code in (401, 403, 503):
                        continue
                    raise
            raise RuntimeError("Urllib failed to retrieve article content.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _fetch_urllib)


def summarize_article_with_ai(
    title: str,
    body_text: str,
    gemini_client: Any = None,
    preferred_model: str = "gemini-3.6-flash",
) -> str:
    """Generate an executive, spoken audio-friendly news bulletin summary from article text."""
    if not body_text:
        return ""

    if gemini_client is None:
        try:
            from app import legacy

            gemini_client = getattr(legacy, "_gemini", None)
        except (ImportError, ModuleNotFoundError):
            gemini_client = None

    if gemini_client is None:
        # Fallback to smart paragraph truncation if AI client not configured
        paras = body_text.split("\n\n")
        return "\n\n".join(paras[:3])[:800]

    from app.services.ai.gemini import (
        extract_gemini_text,
        generate_content_with_fallback,
    )

    prompt = (
        f"You are a professional news editor and audio broadcaster. "
        f"Read this news article titled '{title}' and write a clear, engaging, and well-structured spoken summary. "
        f"Guidelines:\n"
        f"1. Summarize the main story into 3 to 5 clear bullet points or 2 short spoken paragraphs.\n"
        f"2. Write in the exact same language as the article (if in Khmer, reply in natural, fluent Khmer; if in English, reply in English).\n"
        f"3. Make it natural for text-to-speech audio narration.\n"
        f"4. Do not include markdown headers (#), formatting tokens, or preambles.\n\n"
        f"ARTICLE CONTENT:\n{body_text[:10_000]}"
    )

    try:
        response = generate_content_with_fallback(
            client=gemini_client,
            contents=prompt,
            preferred_model=preferred_model,
        )
        text = extract_gemini_text(response)
        return text.strip() if text else body_text[:800]
    except Exception as exc:
        logger.warning("AI article summarization failed (%s); using direct excerpt fallback.", exc)
        return body_text[:800]


def summarize_url_with_ai(
    url: str,
    gemini_client: Any = None,
    preferred_model: str = "gemini-3.6-flash",
) -> tuple[str, str]:
    """Extract and summarize a news story directly using Gemini when web scraping is blocked by anti-bot/WAF."""
    if not url:
        return "", ""

    if gemini_client is None:
        try:
            from app import legacy

            gemini_client = getattr(legacy, "_gemini", None)
        except (ImportError, ModuleNotFoundError):
            gemini_client = None

    if gemini_client is None:
        raise RuntimeError("Gemini AI client is not configured for URL extraction.")

    from app.services.ai.gemini import (
        extract_gemini_text,
        generate_content_with_fallback,
    )

    prompt = (
        f"You are a professional audio news broadcaster and journalist.\n"
        f"A user wants to listen to the news article at this URL:\n{url}\n\n"
        f"Task:\n"
        f"1. Identify the news event, key facts, figures, and developments reported at this link.\n"
        f"2. Write a clear, engaging, and professional spoken news summary in fluent, natural Khmer.\n"
        f"3. Make it natural for text-to-speech audio narration (2 to 3 concise spoken paragraphs).\n"
        f"4. Format your output strictly as:\n"
        f"TITLE: <News Title in Khmer>\n"
        f"SUMMARY: <Spoken news story in natural Khmer>\n"
        f"Do not include markdown headers (#) or extra notes."
    )

    search_config = None
    try:
        from google.genai import types as genai_types
        search_config = genai_types.GenerateContentConfig(tools=[{"google_search": {}}])
    except Exception:
        search_config = None

    try:
        try:
            response = generate_content_with_fallback(
                client=gemini_client,
                contents=prompt,
                preferred_model=preferred_model,
                config=search_config,
            )
        except Exception as search_err:
            if search_config is not None:
                logger.debug("Gemini with search config failed (%s); retrying without config", search_err)
                response = generate_content_with_fallback(
                    client=gemini_client,
                    contents=prompt,
                    preferred_model=preferred_model,
                    config=None,
                )
            else:
                raise
        text = extract_gemini_text(response).strip()
        if not text:
            raise RuntimeError("Gemini returned empty text for URL summarization.")

        title = "ព័ត៌មានជាតិ និងអន្តរជាតិ"
        summary = text
        if "TITLE:" in text and "SUMMARY:" in text:
            parts = text.split("SUMMARY:", 1)
            raw_title = parts[0].replace("TITLE:", "").strip()
            if raw_title:
                title = raw_title
            summary = parts[1].strip()
        elif "\n" in text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines and len(lines[0]) < 120:
                title = lines[0].lstrip("#*- ").strip()
                summary = "\n\n".join(lines[1:]) if len(lines) > 1 else text

        return title, summary
    except Exception as exc:
        logger.warning("AI URL direct summarization failed: %s", exc)
        raise


__all__ = [
    "BROWSER_USER_AGENTS",
    "MAX_ARTICLE_BYTES",
    "MAX_ARTICLE_CHARS",
    "MIN_ARTICLE_CHARS",
    "USER_AGENT",
    "extract_article_content",
    "fetch_article_html",
    "is_safe_public_url",
    "summarize_article_with_ai",
    "summarize_url_with_ai",
]
