"""Gemini client and multimodal inference helpers."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

def normalize_gemini_model(model: str | None) -> str:
    """Normalize model identifier, resolving discontinued or fictitious model aliases to real production models."""
    if not model or not isinstance(model, str):
        return "gemini-2.5-flash"
    m = model.strip()
    if not m:
        return "gemini-2.5-flash"
    lower = m.lower()
    if lower in (
        "gemini-3.6-flash",
        "gemini-3.6",
        "gemini-3-flash",
        "gemini-3.0-flash",
        "gemini-3.5-flash",
        "models/gemini-3.6-flash",
    ):
        return "gemini-2.5-flash"
    if lower in ("gemini-3.1-pro-preview", "gemini-3-pro", "models/gemini-3.1-pro-preview"):
        return "gemini-2.5-pro"
    return m


GEMINI_MODEL_DEFAULT = normalize_gemini_model(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash")

_GEMINI_CLIENT_POOL: list[Any] = []
_ACTIVE_POOL_INDEX: int = 0


def parse_gemini_api_keys(raw_key: str | None) -> list[str]:
    """Parse comma, semicolon, or newline-separated API keys."""
    if not raw_key or not isinstance(raw_key, str):
        return []
    import re
    tokens = [k.strip() for k in re.split(r"[,;\n\r\t]+", raw_key) if k.strip()]
    seen = set()
    result = []
    for k in tokens:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result


def get_primary_gemini_api_key(raw_key: str | None = None) -> str:
    """Get first valid Gemini API key from parameter or environment."""
    raw = raw_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    keys = parse_gemini_api_keys(raw)
    return keys[0] if keys else ""


def register_gemini_client(client: Any) -> None:
    """Register a client into the pool for key-rotation failover."""
    if client is not None and client not in _GEMINI_CLIENT_POOL:
        _GEMINI_CLIENT_POOL.append(client)


def get_next_gemini_client(current_client: Any = None) -> Any | None:
    """Rotate to the next registered client in pool if available."""
    global _ACTIVE_POOL_INDEX
    if len(_GEMINI_CLIENT_POOL) <= 1:
        return None
    _ACTIVE_POOL_INDEX = (_ACTIVE_POOL_INDEX + 1) % len(_GEMINI_CLIENT_POOL)
    next_client = _GEMINI_CLIENT_POOL[_ACTIVE_POOL_INDEX]
    if next_client is current_client and len(_GEMINI_CLIENT_POOL) > 1:
        _ACTIVE_POOL_INDEX = (_ACTIVE_POOL_INDEX + 1) % len(_GEMINI_CLIENT_POOL)
        next_client = _GEMINI_CLIENT_POOL[_ACTIVE_POOL_INDEX]
    return next_client


def clear_gemini_client_pool() -> None:
    """Clear pool (useful for testing or re-initialization)."""
    global _GEMINI_CLIENT_POOL, _ACTIVE_POOL_INDEX
    _GEMINI_CLIENT_POOL.clear()
    _ACTIVE_POOL_INDEX = 0


_GEMINI_RESPONSE_CACHE: OrderedDict[str, tuple[Any, float]] = OrderedDict()
_GEMINI_RESPONSE_CACHE_LOCK = threading.RLock()
_GEMINI_RESPONSE_CACHE_MAX_ITEMS = 500
_GEMINI_RESPONSE_CACHE_TTL_S = 1800.0  # 30 minutes


def _make_gemini_cache_key(contents: Any, model: str) -> str | None:
    """Generate deterministic cache key for text prompts only."""
    if not isinstance(contents, str) or not contents.strip():
        return None
    raw = f"{model}:{contents.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_gemini_content(contents: Any, model: str) -> Any | None:
    """Lookup cached Gemini response if available and not expired."""
    key = _make_gemini_cache_key(contents, model)
    if not key:
        return None
    now = time.monotonic()
    with _GEMINI_RESPONSE_CACHE_LOCK:
        item = _GEMINI_RESPONSE_CACHE.get(key)
        if item is not None:
            resp, ts = item
            if now - ts <= _GEMINI_RESPONSE_CACHE_TTL_S:
                _GEMINI_RESPONSE_CACHE.move_to_end(key)
                return resp
            _GEMINI_RESPONSE_CACHE.pop(key, None)
    return None


def set_cached_gemini_content(contents: Any, model: str, response: Any) -> None:
    """Cache successful Gemini response."""
    key = _make_gemini_cache_key(contents, model)
    if not key or response is None:
        return
    now = time.monotonic()
    with _GEMINI_RESPONSE_CACHE_LOCK:
        _GEMINI_RESPONSE_CACHE.pop(key, None)
        _GEMINI_RESPONSE_CACHE[key] = (response, now)
        while len(_GEMINI_RESPONSE_CACHE) > _GEMINI_RESPONSE_CACHE_MAX_ITEMS:
            _GEMINI_RESPONSE_CACHE.popitem(last=False)


def clear_gemini_response_cache() -> int:
    """Clear all in-memory Gemini responses."""
    with _GEMINI_RESPONSE_CACHE_LOCK:
        count = len(_GEMINI_RESPONSE_CACHE)
        _GEMINI_RESPONSE_CACHE.clear()
        return count


def is_retryable_gemini_error(exc: BaseException | str) -> bool:
    """Determine if a Gemini API failure is transient and can be retried."""
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "unavailable",
            "high demand",
            "resource exhausted",
            "temporarily overloaded",
            "service unavailable",
            "quota exceeded",
            "rate limit",
            "ratelimit",
            "deadline exceeded",
            "overloaded",
            "connection error",
            "connect error",
            "read timeout",
            "socket error",
        )
    )


def extract_gemini_text(response: Any) -> str:
    """Safely extract text from Gemini response without ValueError on safety blocks or empty candidates."""
    if response is None:
        return ""
    try:
        val = getattr(response, "text", "")
        if val:
            return str(val).strip()
    except (ValueError, AttributeError) as exc:
        logger.debug("Gemini response.text unavailable: %s", exc)

    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            cand_text = "".join(getattr(p, "text", "") or "" for p in parts if getattr(p, "text", None))
            if cand_text.strip():
                return cand_text.strip()
    except Exception as exc:
        logger.debug("Gemini candidates text extraction error: %s", exc)
    return ""


def generate_content_with_fallback(
    client: Any,
    contents: Any,
    preferred_model: str = GEMINI_MODEL_DEFAULT,
    config: Any = None,
) -> Any:
    """Generate content with automatic fallback across models and API key rotation if quota or transient error."""
    if client is None:
        raise RuntimeError("Gemini client is not configured.")

    register_gemini_client(client)
    active_client = client

    norm_preferred = normalize_gemini_model(preferred_model)
    # Check in-memory response cache for standard text prompts
    if config is None:
        cached = get_cached_gemini_content(contents, norm_preferred)
        if cached is not None:
            logger.debug("Gemini response served from in-memory cache for %s", norm_preferred)
            return cached

    candidates = [
        norm_preferred,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    unique_models: list[str] = []
    for m in candidates:
        if m and m not in unique_models:
            unique_models.append(m)

    last_exc: Exception | None = None
    for model_name in unique_models:
        try:
            kwargs: dict[str, Any] = {"model": model_name, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            res = active_client.models.generate_content(**kwargs)
            if config is None:
                set_cached_gemini_content(contents, model_name, res)
            return res
        except Exception as exc:
            last_exc = exc
            err_text = str(exc)
            err_lower = err_text.lower()
            is_quota = any(
                q in err_lower
                for q in ("429", "resource_exhausted", "quota", "rate limit", "ratelimit")
            )

            # If quota exhausted (429), attempt rotating API key client if pool has alternatives
            if is_quota and len(_GEMINI_CLIENT_POOL) > 1:
                alt_client = get_next_gemini_client(active_client)
                if alt_client is not None and alt_client is not active_client:
                    logger.warning(
                        "Gemini model %s quota hit (429); rotating to alternate API key in pool...",
                        model_name,
                    )
                    active_client = alt_client
                    try:
                        kwargs = {"model": model_name, "contents": contents}
                        if config is not None:
                            kwargs["config"] = config
                        res = active_client.models.generate_content(**kwargs)
                        if config is None:
                            set_cached_gemini_content(contents, model_name, res)
                        return res
                    except Exception as alt_exc:
                        last_exc = alt_exc
                        err_text = str(alt_exc)
                        err_lower = err_text.lower()

            if is_retryable_gemini_error(last_exc) or any(
                k in err_lower
                for k in (
                    "404",
                    "not found",
                    "not supported",
                    "unsupported",
                    "no longer available",
                    "not available",
                    "is not found",
                    "quota",
                    "resource_exhausted",
                )
            ):
                logger.warning(
                    "Gemini model %s failed (%s); falling back to next available model...",
                    model_name,
                    last_exc,
                )
                continue
            raise last_exc from None

    if last_exc:
        raise last_exc
    raise RuntimeError("No Gemini models succeeded.")


def detect_image_mime_from_bytes(header: bytes) -> str:
    """Detect image MIME type from initial header bytes."""
    if len(header) >= 8 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if len(header) >= 2 and header[:2] == b"\xff\xd8":
        return "image/jpeg"
    if len(header) >= 6 and header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def detect_image_mime(path: str) -> str:
    """Read magic bytes from file and return MIME type."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(12)
        return detect_image_mime_from_bytes(header)
    except OSError:
        return "image/jpeg"


__all__ = [
    "GEMINI_MODEL_DEFAULT",
    "clear_gemini_client_pool",
    "clear_gemini_response_cache",
    "detect_image_mime",
    "detect_image_mime_from_bytes",
    "extract_gemini_text",
    "generate_content_with_fallback",
    "get_cached_gemini_content",
    "get_next_gemini_client",
    "get_primary_gemini_api_key",
    "is_retryable_gemini_error",
    "normalize_gemini_model",
    "parse_gemini_api_keys",
    "register_gemini_client",
    "set_cached_gemini_content",
]
