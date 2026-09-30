"""Database connection clients and session factories (Supabase & SQLite)."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from app.config.settings import SETTINGS

logger = logging.getLogger("app.database.client")

_SUPABASE_CLIENT: Any | None = None


class SupabaseClient:
    """Wrapper or client representation for Supabase PostgREST access."""

    def __init__(self, url: str = "", key: str = "") -> None:
        self.url = url or (os.environ.get("SUPABASE_URL") or getattr(SETTINGS, "SUPABASE_URL", "")).strip()
        self.key = key or (os.environ.get("SUPABASE_KEY") or getattr(SETTINGS, "SUPABASE_KEY", "")).strip()
        self.is_configured = bool(self.url and self.key)

    def table(self, name: str) -> Any:
        client = get_supabase_client()
        if client and hasattr(client, "table"):
            return client.table(name)
        raise RuntimeError("Supabase client is not configured.")


def get_supabase_client() -> Any | None:
    """Initialize or return the global Supabase client singleton."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    url = (os.environ.get("SUPABASE_URL") or getattr(SETTINGS, "SUPABASE_URL", "")).strip()
    key = (os.environ.get("SUPABASE_KEY") or getattr(SETTINGS, "SUPABASE_KEY", "")).strip()

    if not url or not key:
        return None

    try:
        from supabase import Client, create_client

        _SUPABASE_CLIENT = create_client(url, key)
        logger.info("Supabase client initialized successfully.")
        return _SUPABASE_CLIENT
    except Exception as exc:
        logger.warning("Could not initialize Supabase client: %s", exc)
        return None


def get_db_client() -> SupabaseClient:
    """Return default SupabaseClient wrapper."""
    return SupabaseClient()


def get_sqlite_connection(db_path: str = "data/articles_sent.db") -> sqlite3.Connection:
    """Return a thread-safe connection to a local SQLite database with WAL mode enabled."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


__all__ = [
    "SupabaseClient",
    "get_db_client",
    "get_sqlite_connection",
    "get_supabase_client",
]
