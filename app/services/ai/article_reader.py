"""Web Link Article Narrator service.

Provides SSRF-protected URL fetching, clean HTML article text extraction,
and AI-powered news summarization for voice audio synthesis.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import socket
import urllib.parse
from html.parser import HTMLParser
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36 (TelegramVoiceBot/4.3; +https://t.me/m11mmm112)"
)

MAX_ARTICLE_BYTES = 2 * 1024 * 1024  # 2MB max download
MAX_ARTICLE_CHARS = 15_000  # Cap extracted text length
MIN_ARTICLE_CHARS = 60  # Minimum readable text to be considered an article


def is_safe_public_url(url: str) -> tuple[bool, str]:
    """Validate that a URL uses HTTP/HTTPS and does not target private/internal networks (SSRF defense)."""
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

        lower_host = hostname.lower().strip(".")
        if lower_host in (
            "localhost",
            "127.0.0.1",
            "0.0.0.0",  # noqa: S104
            "::1",
            "metadata.google.internal",
            "instance-data",
        ):
            return False, "Access to private or loopback host is forbidden."

        # Verify resolved IP addresses
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as err:
            return False, f"Could not resolve host '{hostname}': {err}"

        for item in addr_info:
            ip_str = item[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_multicast
                    or ip.is_reserved
                    or ip.is_unspecified
                ):
                    return False, f"Access to private IP '{ip_str}' is forbidden."
                if str(ip) == "169.254.169.254":
                    return False, "Access to cloud metadata IP is forbidden."
            except ValueError:
                continue

        return True, ""
    except Exception as exc:
        return False, f"URL validation failed: {exc}"


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
        "head",
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


async def fetch_article_html(url: str, timeout_s: float = 12.0) -> str:
    """Fetch webpage HTML safely using HTTPX or standard urllib with size bounds."""
    safe, reason = is_safe_public_url(url)
    if not safe:
        raise ValueError(f"Security validation error: {reason}")

    try:
        import httpx

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout_s,
            verify=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type and "xhtml" not in content_type:
                logger.debug("Content-Type '%s' not HTML/text, attempting parse anyway", content_type)
            # Bound payload size to prevent OOM
            return resp.text[:MAX_ARTICLE_BYTES]
    except (ImportError, ModuleNotFoundError):
        import urllib.request

        def _fetch_urllib() -> str:
            req = urllib.request.Request(  # noqa: S310
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310
                data = response.read(MAX_ARTICLE_BYTES)
                encoding = response.headers.get_content_charset() or "utf-8"
                return data.decode(encoding, errors="replace")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _fetch_urllib)


def summarize_article_with_ai(
    title: str,
    body_text: str,
    gemini_client: Any = None,
    preferred_model: str = "gemini-2.5-flash",
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


__all__ = [
    "MAX_ARTICLE_BYTES",
    "MAX_ARTICLE_CHARS",
    "MIN_ARTICLE_CHARS",
    "USER_AGENT",
    "extract_article_content",
    "fetch_article_html",
    "is_safe_public_url",
    "summarize_article_with_ai",
]
