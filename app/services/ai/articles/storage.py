"""SQLite-backed idempotency, article sources, and admin approval queue storage."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any
import urllib.parse

logger = logging.getLogger(__name__)

# Determine stable absolute database path
_DATA_DIR_ENV = os.getenv("DATA_DIR")
if _DATA_DIR_ENV:
    _DB_DIR = Path(_DATA_DIR_ENV)
else:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    _DB_DIR = _PROJECT_ROOT / "data"

_DB_PATH = _DB_DIR / "articles_sent.db"

# Thread locks for write serialization and hot in-memory cache
_WRITE_LOCK = threading.RLock()
_HANDLED_CACHE: set[str] = set()
_HANDLED_CACHE_LOCK = threading.RLock()
_MAX_HANDLED_CACHE_SIZE = 5000

# News publisher branding suffixes and prefixes to clean for title idempotency
_TITLE_NOISE_PATTERNS = [
    r"[-–—|•]\s*(?:fresh\s*news|vod|koh\s*santepheap|rfi|rasmei|thmey\s*thmey|kampuchea\s*thmey|reuters|ap|bbc|cnn|the\s*verge|techcrunch|wired|securityweek).*",
    r"^\s*\[(?:breaking|update|urgent|hot|ព័ត៌មានថ្មី)\]\s*[:：\-–]?\s*",
    r"^\s*(?:ព័ត៌មានបឋម|ព័ត៌មានទាន់ហេតុការណ៍|ទាន់ហេតុការណ៍|update|breaking)\s*[:：\-–]\s*",
]
_TITLE_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in _TITLE_NOISE_PATTERNS]


def normalize_url(url: str) -> str:
    """Normalize URL by stripping protocol, www, trailing slashes, and tracking query parameters."""
    if not url:
        return ""
    u = str(url).strip()
    if not u.startswith(("http://", "https://")):
        u = f"https://{u}"
    try:
        parsed = urllib.parse.urlparse(u)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")

        # Filter tracking query parameters
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        clean_pairs: list[tuple[str, str]] = []
        for k, v in query_pairs:
            k_low = k.lower()
            if (
                k_low.startswith("utm_")
                or k_low in ("fbclid", "gclid", "_hsenc", "ref", "source", "share", "from", "igshid", "mibextid", "mc_cid")
            ):
                continue
            clean_pairs.append((k, v))
        clean_pairs.sort()
        new_query = urllib.parse.urlencode(clean_pairs)

        clean_url = f"{netloc}{path}"
        if new_query:
            clean_url = f"{clean_url}?{new_query}"
        return clean_url
    except Exception:
        clean = u.lower().replace("https://", "").replace("http://", "").replace("www.", "")
        return clean.split("?")[0].split("#")[0].rstrip("/")


def normalize_title(title: str) -> str:
    """Clean and normalize news title for similarity and idempotency matching."""
    if not title:
        return ""
    text = str(title).strip()

    # Strip zero-width spaces and control characters
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", text)

    # Strip quotation marks and brackets
    text = re.sub(r"[«»\"'“”‘’\[\](){}<>]", " ", text)

    # Strip publisher noise suffixes / prefixes
    for pat in _TITLE_NOISE_RE:
        text = pat.sub("", text)

    # Normalize whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _get_supabase_client() -> Any | None:
    """Retrieve Supabase client across modern services and legacy runtime."""
    with suppress(Exception):
        from app.core.supabase import get_supabase_client
        client = get_supabase_client()
        if client:
            return client
    with suppress(Exception):
        from app import legacy
        client = getattr(legacy, "_supabase", None) or getattr(legacy, "supabase", None)
        if client:
            return client
    return None


def _get_connection() -> sqlite3.Connection:
    """Obtain a SQLite connection with 30s busy timeout."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


@contextlib.contextmanager
def _db():
    """Context manager providing an auto-closing SQLite connection."""
    conn = _get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _init_db_sync() -> None:
    """Initialize SQLite database tables, indexes, and configure WAL mode once."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, _db() as conn:
        with suppress(Exception):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")

        # 1. Sent articles (Idempotency)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_articles (
                hash TEXT PRIMARY KEY,
                url TEXT,
                clean_url TEXT,
                title TEXT,
                clean_title TEXT,
                created_at REAL,
                user_id INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sent_articles_created ON sent_articles(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sent_articles_clean_url ON sent_articles(clean_url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sent_articles_clean_title ON sent_articles(clean_title)")

        # 2. Monitored base URL sources
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS article_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                name TEXT,
                is_active INTEGER DEFAULT 1,
                created_at REAL,
                added_by INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_article_sources_active ON article_sources(is_active)")

        # 3. Pending articles (Admin review & approval queue)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_articles (
                hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                clean_url TEXT,
                title TEXT,
                clean_title TEXT,
                khmer_title TEXT,
                clean_khmer_title TEXT,
                khmer_summary TEXT,
                khmer_tts_script TEXT,
                original_summary TEXT,
                body_text TEXT,
                image_url TEXT,
                telegraph_url TEXT,
                audio_file_id TEXT,
                audio_bytes BLOB,
                status TEXT DEFAULT 'pending',
                created_at REAL,
                reviewed_at REAL,
                reviewed_by INTEGER,
                analysis TEXT,
                verification TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_articles_status ON pending_articles(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_articles_created ON pending_articles(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_articles_clean_url ON pending_articles(clean_url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_articles_clean_title ON pending_articles(clean_title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_articles_clean_km_title ON pending_articles(clean_khmer_title)")

        # Verify and add columns if upgrading existing database
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(pending_articles)")
        existing_cols = {row["name"] for row in cur.fetchall()}
        for col, col_type in (
            ("clean_url", "TEXT"),
            ("clean_title", "TEXT"),
            ("clean_khmer_title", "TEXT"),
            ("analysis", "TEXT"),
            ("verification", "TEXT"),
        ):
            if col not in existing_cols:
                with suppress(Exception):
                    conn.execute(f"ALTER TABLE pending_articles ADD COLUMN {col} {col_type}")

        # Seed initial default sources if empty
        cur.execute("SELECT COUNT(*) AS count FROM article_sources")
        row = cur.fetchone()
        if row and row["count"] == 0:
            now = time.time()
            conn.execute(
                """
                INSERT OR IGNORE INTO article_sources (url, name, is_active, created_at, added_by)
                VALUES (?, ?, 1, ?, NULL)
                """,
                ("https://freshnewsasia.com", "Fresh News Asia", now),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO article_sources (url, name, is_active, created_at, added_by)
                VALUES (?, ?, 1, ?, NULL)
                """,
                ("https://vodenglish.news", "VOD English", now),
            )

        conn.commit()


# Run schema initialization once on import
try:
    _init_db_sync()
except Exception as e:
    logger.warning("Failed initializing article SQLite database: %s", e)


# ==========================================
# 1. Idempotency Methods
# ==========================================

def is_article_handled_sync(
    url_hash: str = "",
    url: str = "",
    title: str = "",
    khmer_title: str = "",
) -> bool:
    """Check synchronously if an article has already been handled (sent, pending, or approved)."""
    clean_hash = url_hash.strip() if url_hash else ""
    norm_url = normalize_url(url) if url else ""

    # Fast in-memory cache lookup
    with _HANDLED_CACHE_LOCK:
        if clean_hash and clean_hash in _HANDLED_CACHE:
            return True
        if norm_url and norm_url in _HANDLED_CACHE:
            return True

    candidate_hashes: set[str] = set()
    if clean_hash:
        candidate_hashes.add(clean_hash)

    if url:
        u_clean = url.split("?")[0].split("#")[0].rstrip("/").strip().lower()
        candidate_hashes.add(hashlib.sha256(u_clean.encode("utf-8")).hexdigest())
        if norm_url:
            candidate_hashes.add(hashlib.sha256(norm_url.encode("utf-8")).hexdigest())

    clean_t = normalize_title(title) if title else ""
    clean_kt = normalize_title(khmer_title) if khmer_title else ""

    try:
        with _db() as conn:
            cur = conn.cursor()

            # 1. Fast Hash Lookup
            if candidate_hashes:
                placeholders = ",".join("?" for _ in candidate_hashes)
                hash_list = list(candidate_hashes)
                cur.execute(
                    f"""
                    SELECT 1 FROM sent_articles WHERE hash IN ({placeholders})
                    UNION
                    SELECT 1 FROM pending_articles WHERE hash IN ({placeholders})
                    LIMIT 1
                    """,
                    hash_list + hash_list,
                )
                if cur.fetchone() is not None:
                    with _HANDLED_CACHE_LOCK:
                        _HANDLED_CACHE.update(candidate_hashes)
                    return True

            # 2. Fast Clean URL Lookup
            if norm_url:
                cur.execute(
                    """
                    SELECT 1 FROM sent_articles WHERE clean_url = ? OR url = ?
                    UNION
                    SELECT 1 FROM pending_articles WHERE clean_url = ? OR url = ?
                    LIMIT 1
                    """,
                    (norm_url, url, norm_url, url),
                )
                if cur.fetchone() is not None:
                    with _HANDLED_CACHE_LOCK:
                        _HANDLED_CACHE.add(norm_url)
                    return True

            # 3. Clean Title / Khmer Title Lookup
            titles_to_check: set[str] = set()
            for t in (clean_t, clean_kt):
                if t and len(t) >= 8 and t not in ("ព័ត៌មានថ្មី", "ព័ត៌មានជាតិ", "breaking news", "news"):
                    titles_to_check.add(t)

            if titles_to_check:
                t_list = list(titles_to_check)
                placeholders = ",".join("?" for _ in t_list)
                cur.execute(
                    f"""
                    SELECT 1 FROM sent_articles WHERE clean_title IN ({placeholders})
                    UNION
                    SELECT 1 FROM pending_articles WHERE clean_title IN ({placeholders}) OR clean_khmer_title IN ({placeholders})
                    LIMIT 1
                    """,
                    t_list + t_list + t_list,
                )
                if cur.fetchone() is not None:
                    if clean_hash:
                        with _HANDLED_CACHE_LOCK:
                            _HANDLED_CACHE.add(clean_hash)
                    return True

    except Exception as exc:
        logger.warning("is_article_handled_sync error: %s", exc)

    return False


async def is_article_handled(
    url_hash: str = "",
    url: str = "",
    title: str = "",
    khmer_title: str = "",
) -> bool:
    """Check asynchronously if an article has already been handled."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_article_handled_sync, url_hash, url, title, khmer_title)


def is_article_sent_sync(url_hash: str, url: str = "", title: str = "") -> bool:
    """Check synchronously if an article hash has already been marked as sent."""
    if not url_hash and not url and not title:
        return False
    try:
        with _db() as conn:
            cur = conn.cursor()
            if url_hash:
                cur.execute("SELECT 1 FROM sent_articles WHERE hash = ? LIMIT 1", (url_hash,))
                if cur.fetchone() is not None:
                    return True
            if url:
                norm_u = normalize_url(url)
                cur.execute("SELECT 1 FROM sent_articles WHERE clean_url = ? OR url = ? LIMIT 1", (norm_u, url))
                if cur.fetchone() is not None:
                    return True
            if title:
                norm_t = normalize_title(title)
                if norm_t and len(norm_t) >= 8:
                    cur.execute("SELECT 1 FROM sent_articles WHERE clean_title = ? LIMIT 1", (norm_t,))
                    if cur.fetchone() is not None:
                        return True
            return False
    except Exception as exc:
        logger.warning("is_article_sent_sync error: %s", exc)
        return False


def mark_article_sent_sync(
    url_hash: str,
    url: str = "",
    title: str = "",
    user_id: int | None = None,
) -> None:
    """Mark an article as sent in SQLite synchronously and update hot cache."""
    if not url_hash:
        return
    norm_u = normalize_url(url) if url else ""
    norm_t = normalize_title(title) if title else ""

    with _WRITE_LOCK, _db() as conn:
        try:
            # Backfill from pending_articles if metadata missing
            if not url or not title:
                cur = conn.cursor()
                cur.execute(
                    "SELECT url, title, khmer_title FROM pending_articles WHERE hash = ? LIMIT 1",
                    (url_hash,),
                )
                p_row = cur.fetchone()
                if p_row:
                    if not url and p_row["url"]:
                        url = p_row["url"]
                        norm_u = normalize_url(url)
                    if not title:
                        title = p_row["khmer_title"] or p_row["title"] or ""
                        norm_t = normalize_title(title)

            conn.execute(
                """
                INSERT OR REPLACE INTO sent_articles (hash, url, clean_url, title, clean_title, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url_hash, url, norm_u or None, title, norm_t or None, time.time(), user_id),
            )
            conn.commit()

            with _HANDLED_CACHE_LOCK:
                _HANDLED_CACHE.add(url_hash)
                if norm_u:
                    _HANDLED_CACHE.add(norm_u)
                if len(_HANDLED_CACHE) > _MAX_HANDLED_CACHE_SIZE:
                    _HANDLED_CACHE.clear()
        except Exception as exc:
            logger.warning("mark_article_sent_sync error for hash %s: %s", url_hash, exc)


def get_sent_article_sync(url_hash: str) -> dict[str, Any] | None:
    """Fetch sent article record by hash synchronously."""
    if not url_hash:
        return None
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT hash, url, clean_url, title, clean_title, created_at, user_id FROM sent_articles WHERE hash = ? LIMIT 1",
                (url_hash,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as exc:
        logger.warning("get_sent_article_sync error for %s: %s", url_hash, exc)
    return None


async def is_article_sent(url_hash: str) -> bool:
    """Check asynchronously if an article hash has already been sent."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_article_sent_sync, url_hash)


async def mark_article_sent(
    url_hash: str,
    url: str = "",
    title: str = "",
    user_id: int | None = None,
) -> None:
    """Mark an article as sent asynchronously."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, mark_article_sent_sync, url_hash, url, title, user_id)


async def get_sent_article(url_hash: str) -> dict[str, Any] | None:
    """Get sent article record asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_sent_article_sync, url_hash)


# ==========================================
# 2. Monitored Article Sources Methods
# ==========================================

def add_article_source_sync(url: str, name: str = "", added_by: int | None = None) -> tuple[bool, str, int]:
    """Add a new base URL source to SQLite and sync to Supabase synchronously."""
    clean_url = (url or "").strip().rstrip("/")
    if not clean_url or not clean_url.startswith(("http://", "https://")):
        return False, "Invalid URL schema (must start with http:// or https://)", 0

    if not name:
        parsed = urllib.parse.urlparse(clean_url)
        name = parsed.netloc.replace("www.", "")

    supabase = _get_supabase_client()
    if supabase:
        with suppress(Exception):
            supabase.table("base_urls").upsert({"url": clean_url}).execute()

    try:
        with _WRITE_LOCK, _db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO article_sources (url, name, is_active, created_at, added_by)
                VALUES (?, ?, 1, ?, ?)
                """,
                (clean_url, name, time.time(), added_by),
            )
            conn.commit()
            source_id = cur.lastrowid or 0
            return True, f"Added source '{name}' ({clean_url})", source_id
    except sqlite3.IntegrityError:
        return False, f"Source URL '{clean_url}' already exists.", 0
    except Exception as exc:
        logger.warning("add_article_source_sync error: %s", exc)
        return False, str(exc), 0


def remove_article_source_sync(source_id_or_url: str | int) -> bool:
    """Remove a base URL source by persistent integer ID or URL string synchronously."""
    supabase = _get_supabase_client()
    target_url: str | None = None

    try:
        with _WRITE_LOCK, _db() as conn:
            cur = conn.cursor()
            if isinstance(source_id_or_url, int) or str(source_id_or_url).isdigit():
                source_id = int(source_id_or_url)
                cur.execute("SELECT url FROM article_sources WHERE id = ?", (source_id,))
                row = cur.fetchone()
                if row:
                    target_url = row["url"]
                cur.execute("DELETE FROM article_sources WHERE id = ?", (source_id,))
            else:
                target_url = str(source_id_or_url).strip().rstrip("/")
                cur.execute("DELETE FROM article_sources WHERE url = ? OR url = ?", (target_url, target_url + "/"))

            conn.commit()
            rowcount = cur.rowcount

            if supabase and target_url:
                with suppress(Exception):
                    supabase.table("base_urls").delete().eq("url", target_url).execute()
                    supabase.table("base_urls").delete().eq("url", target_url + "/").execute()

            return rowcount > 0
    except Exception as exc:
        logger.warning("remove_article_source_sync error: %s", exc)
        return False


def get_article_sources_sync(active_only: bool = True) -> list[dict[str, Any]]:
    """Retrieve all configured article base URL sources with stable IDs."""
    # 1. Sync any new URLs from Supabase base_urls into SQLite
    supabase = _get_supabase_client()
    if supabase:
        try:
            res = supabase.table("base_urls").select("url, created_at").execute()
            if res and res.data:
                now = time.time()
                with _WRITE_LOCK, _db() as conn:
                    for row in res.data:
                        sb_url = (row.get("url") or "").strip().rstrip("/")
                        if sb_url:
                            parsed = urllib.parse.urlparse(sb_url)
                            name = parsed.netloc.replace("www.", "")
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO article_sources (url, name, is_active, created_at, added_by)
                                VALUES (?, ?, 1, ?, 0)
                                """,
                                (sb_url, name, now),
                            )
                    conn.commit()
        except Exception as sb_err:
            logger.debug("Failed syncing Supabase base_urls: %s", sb_err)

    # 2. Query SQLite for guaranteed persistent integer IDs
    try:
        with _db() as conn:
            cur = conn.cursor()
            query = (
                "SELECT id, url, name, is_active, created_at, added_by FROM article_sources WHERE is_active = 1 ORDER BY id ASC"
                if active_only
                else "SELECT id, url, name, is_active, created_at, added_by FROM article_sources ORDER BY id ASC"
            )
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        logger.warning("get_article_sources_sync error: %s", exc)
        return []


def toggle_article_source_sync(source_id: int, is_active: bool) -> bool:
    """Enable or disable an article source synchronously."""
    try:
        with _WRITE_LOCK, _db() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE article_sources SET is_active = ? WHERE id = ?", (1 if is_active else 0, source_id))
            conn.commit()
            return cur.rowcount > 0
    except Exception as exc:
        logger.warning("toggle_article_source_sync error: %s", exc)
        return False


async def add_article_source(url: str, name: str = "", added_by: int | None = None) -> tuple[bool, str, int]:
    """Add an article base URL source asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, add_article_source_sync, url, name, added_by)


async def remove_article_source(source_id_or_url: str | int) -> bool:
    """Remove an article source asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, remove_article_source_sync, source_id_or_url)


async def get_article_sources(active_only: bool = True) -> list[dict[str, Any]]:
    """Fetch article sources asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_article_sources_sync, active_only)


async def toggle_article_source(source_id: int, is_active: bool) -> bool:
    """Toggle article source active state asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, toggle_article_source_sync, source_id, is_active)


# ==========================================
# 3. Pending Articles & Admin Queue Methods
# ==========================================

def _unpack_pending_dict(row: Any) -> dict[str, Any]:
    """Unpack SQLite row into article dict, expanding nested analysis JSON."""
    d = dict(row)
    analysis_raw = d.get("analysis")
    if analysis_raw and isinstance(analysis_raw, str):
        try:
            parsed = json.loads(analysis_raw)
            if isinstance(parsed, dict):
                d["analysis"] = parsed
                d.setdefault("hook", parsed.get("hook"))
                d.setdefault("takeaway", parsed.get("takeaway"))
                d.setdefault("category_name", parsed.get("category_name"))
                d.setdefault("source_name", parsed.get("source_name"))
        except Exception:
            pass

    verif_raw = d.get("verification")
    if verif_raw and isinstance(verif_raw, str):
        try:
            d["verification"] = json.loads(verif_raw)
        except Exception:
            pass

    return d


def save_pending_article_sync(payload: dict[str, Any]) -> None:
    """Store an article awaiting admin approval synchronously."""
    url_hash = payload.get("hash")
    url = payload.get("url")
    if not url_hash or not url:
        return

    raw_title = payload.get("title", "")
    khmer_title = payload.get("khmer_title", "")
    norm_u = normalize_url(url)
    norm_t = normalize_title(raw_title)
    norm_kt = normalize_title(khmer_title)

    analysis_dict = dict(payload.get("analysis") or {})
    for k in ("hook", "takeaway", "category_name", "source_name"):
        if payload.get(k):
            analysis_dict[k] = payload[k]

    with _WRITE_LOCK, _db() as conn:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_articles (
                    hash, url, clean_url, title, clean_title, khmer_title, clean_khmer_title,
                    khmer_summary, khmer_tts_script, original_summary, body_text,
                    image_url, telegraph_url, audio_file_id, audio_bytes,
                    status, created_at, analysis, verification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url_hash,
                    url,
                    norm_u or None,
                    raw_title,
                    norm_t or None,
                    khmer_title,
                    norm_kt or None,
                    payload.get("khmer_summary", ""),
                    payload.get("khmer_tts_script", ""),
                    payload.get("original_summary", ""),
                    payload.get("body_text", ""),
                    payload.get("image_url"),
                    payload.get("telegraph_url"),
                    payload.get("audio_file_id"),
                    payload.get("audio_bytes"),
                    payload.get("status", "pending"),
                    payload.get("created_at", time.time()),
                    json.dumps(analysis_dict, ensure_ascii=False) if analysis_dict else None,
                    json.dumps(payload.get("verification", {}), ensure_ascii=False) if payload.get("verification") else None,
                ),
            )
            conn.commit()

            with _HANDLED_CACHE_LOCK:
                _HANDLED_CACHE.add(url_hash)
                if norm_u:
                    _HANDLED_CACHE.add(norm_u)
        except Exception as exc:
            logger.warning("save_pending_article_sync error for %s: %s", url_hash, exc)


def get_pending_article_sync(url_hash: str) -> dict[str, Any] | None:
    """Retrieve pending article by hash synchronously."""
    if not url_hash:
        return None
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT hash, url, title, khmer_title, khmer_summary,
                       khmer_tts_script, original_summary, body_text,
                       image_url, telegraph_url, audio_file_id, audio_bytes,
                       status, created_at, reviewed_at, reviewed_by,
                       analysis, verification
                FROM pending_articles WHERE hash = ? LIMIT 1
                """,
                (url_hash,),
            )
            row = cur.fetchone()
            return _unpack_pending_dict(row) if row else None
    except Exception as exc:
        logger.warning("get_pending_article_sync error for %s: %s", url_hash, exc)
    return None


def get_pending_article_by_prefix_sync(hash_prefix: str) -> dict[str, Any] | None:
    """Find a pending article whose hash starts with the given prefix."""
    if not hash_prefix:
        return None
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT hash, url, title, khmer_title, khmer_summary,
                       khmer_tts_script, original_summary, body_text,
                       image_url, telegraph_url, audio_file_id, audio_bytes,
                       status, created_at, reviewed_at, reviewed_by,
                       analysis, verification
                FROM pending_articles WHERE hash LIKE ? LIMIT 1
                """,
                (f"{hash_prefix}%",),
            )
            row = cur.fetchone()
            return _unpack_pending_dict(row) if row else None
    except Exception as exc:
        logger.warning("get_pending_article_by_prefix_sync error for %s: %s", hash_prefix, exc)
    return None


async def get_pending_article_by_prefix(hash_prefix: str) -> dict[str, Any] | None:
    """Find pending article by hash prefix asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_pending_article_by_prefix_sync, hash_prefix)


def update_pending_status_sync(url_hash: str, status: str, reviewed_by: int | None = None) -> bool:
    """Update status ('approved', 'rejected') and reviewer ID synchronously."""
    if not url_hash or not status:
        return False
    with _WRITE_LOCK, _db() as conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE pending_articles
                SET status = ?, reviewed_at = ?, reviewed_by = ?
                WHERE hash = ?
                """,
                (status, time.time(), reviewed_by, url_hash),
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as exc:
            logger.warning("update_pending_status_sync error for %s: %s", url_hash, exc)
            return False


def is_article_pending_sync(url_hash: str) -> bool:
    """Check if an article hash is in the pending approval queue."""
    if not url_hash:
        return False
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pending_articles WHERE hash = ? LIMIT 1", (url_hash,))
            return cur.fetchone() is not None
    except Exception as exc:
        logger.warning("is_article_pending_sync error for %s: %s", url_hash, exc)
        return False


def get_pending_articles_by_status_sync(status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    """Fetch pending articles by status synchronously."""
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT hash, url, title, khmer_title, khmer_summary,
                       khmer_tts_script, original_summary, body_text,
                       image_url, telegraph_url, audio_file_id,
                       status, created_at, reviewed_at, reviewed_by,
                       analysis, verification
                FROM pending_articles WHERE status = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (status, max(1, limit)),
            )
            return [_unpack_pending_dict(r) for r in cur.fetchall()]
    except Exception as exc:
        logger.warning("get_pending_articles_by_status_sync error: %s", exc)
        return []


def get_article_storage_stats_sync() -> dict[str, int]:
    """Retrieve high-level counts for Web News Scanner Panel."""
    stats = {
        "sources_total": 0,
        "sources_active": 0,
        "pending_articles": 0,
        "approved_articles": 0,
        "rejected_articles": 0,
        "sent_articles": 0,
    }
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM article_sources")
            stats["sources_total"] = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM article_sources WHERE is_active = 1")
            stats["sources_active"] = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM pending_articles WHERE status = 'pending'")
            stats["pending_articles"] = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM pending_articles WHERE status = 'approved'")
            stats["approved_articles"] = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM pending_articles WHERE status = 'rejected'")
            stats["rejected_articles"] = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM sent_articles")
            stats["sent_articles"] = int(cur.fetchone()[0] or 0)
    except Exception as exc:
        logger.warning("get_article_storage_stats_sync error: %s", exc)
    return stats


def prune_old_articles_sync(days: int = 90) -> int:
    """Delete sent_articles records older than given days to keep SQLite lightweight."""
    cutoff = time.time() - (max(1, days) * 86400)
    with _WRITE_LOCK, _db() as conn:
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM sent_articles WHERE created_at < ?", (cutoff,))
            conn.commit()
            deleted = cur.rowcount
            if deleted > 0:
                logger.info("Pruned %d sent article records older than %d days.", deleted, days)
            return deleted
        except Exception as exc:
            logger.warning("prune_old_articles_sync error: %s", exc)
            return 0


async def prune_old_articles(days: int = 90) -> int:
    """Prune old articles asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, prune_old_articles_sync, days)


async def get_article_storage_stats() -> dict[str, int]:
    """Retrieve article storage stats asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_article_storage_stats_sync)


async def save_pending_article(payload: dict[str, Any]) -> None:
    """Store pending article asynchronously."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, save_pending_article_sync, payload)


async def get_pending_article(url_hash: str) -> dict[str, Any] | None:
    """Get pending article record asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_pending_article_sync, url_hash)


async def update_pending_status(url_hash: str, status: str, reviewed_by: int | None = None) -> bool:
    """Update pending article status asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, update_pending_status_sync, url_hash, status, reviewed_by)


async def is_article_pending(url_hash: str) -> bool:
    """Check if article is pending asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_article_pending_sync, url_hash)


async def get_pending_articles_by_status(status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    """Get pending articles by status asynchronously."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_pending_articles_by_status_sync, status, limit)


__all__ = [
    "add_article_source",
    "add_article_source_sync",
    "get_article_sources",
    "get_article_sources_sync",
    "get_article_storage_stats",
    "get_article_storage_stats_sync",
    "get_pending_article",
    "get_pending_article_by_prefix",
    "get_pending_article_by_prefix_sync",
    "get_pending_article_sync",
    "get_pending_articles_by_status",
    "get_pending_articles_by_status_sync",
    "get_sent_article",
    "get_sent_article_sync",
    "is_article_handled",
    "is_article_handled_sync",
    "is_article_pending",
    "is_article_pending_sync",
    "is_article_sent",
    "is_article_sent_sync",
    "mark_article_sent",
    "mark_article_sent_sync",
    "normalize_title",
    "normalize_url",
    "prune_old_articles",
    "prune_old_articles_sync",
    "remove_article_source",
    "remove_article_source_sync",
    "save_pending_article",
    "save_pending_article_sync",
    "toggle_article_source",
    "toggle_article_source_sync",
    "update_pending_status",
    "update_pending_status_sync",
]