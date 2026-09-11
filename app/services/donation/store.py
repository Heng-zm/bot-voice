"""Donation data store with Supabase persistence, resilient local JSON backup, and TTL cache."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(PROJECT_ROOT, "data")
LOCAL_STORE_PATH = os.path.join(DATA_DIR, "donations.json")

# Tier definitions and coffee cup equivalents
TIER_DETAILS: dict[str, dict[str, Any]] = {
    "coffee": {
        "title": "☕ កាហ្វេ ១ កែវ",
        "title_en": "1 Cup of Coffee",
        "amount": 1.0,
        "cups": 1,
        "emoji": "☕",
    },
    "milktea": {
        "title": "🧋 តែទឹកដោះគោ",
        "title_en": "Bubble Milk Tea",
        "amount": 2.0,
        "cups": 2,
        "emoji": "🧋",
    },
    "lunch": {
        "title": "🍜 គុយទាវ ១ ចាន",
        "title_en": "Delicious Lunch",
        "amount": 3.0,
        "cups": 3,
        "emoji": "🍜",
    },
    "server": {
        "title": "🖥️ ថ្លៃ Server",
        "title_en": "Server Maintenance",
        "amount": 5.0,
        "cups": 5,
        "emoji": "🖥️",
    },
    "patron": {
        "title": "🌟 ឧបត្ថម្ភពិសេស",
        "title_en": "Gold Supporter",
        "amount": 10.0,
        "cups": 10,
        "emoji": "🌟",
    },
    "gold": {
        "title": "💎 អ្នកគាំទ្រឆ្នើម",
        "title_en": "Platinum Supporter",
        "amount": 20.0,
        "cups": 20,
        "emoji": "💎",
    },
}

CACHE_TTL_SECONDS = 60.0


class DonationStore:
    """Thread-safe store for donor recognition and contribution stats with in-memory TTL caching."""

    def __init__(self, file_path: str = LOCAL_STORE_PATH) -> None:
        self._file_path = file_path
        self._lock = threading.Lock()
        self._donations: list[dict[str, Any]] = []

        # In-memory TTL caches for instant query responses (<0.05ms)
        self._cache_stats: dict[str, Any] | None = None
        self._cache_stats_ts: float = 0.0
        self._cache_top: list[dict[str, Any]] | None = None
        self._cache_top_ts: float = 0.0
        self._cache_recent: list[dict[str, Any]] | None = None
        self._cache_recent_ts: float = 0.0

        self._load_local_store()

    def _invalidate_cache(self) -> None:
        """Invalidate TTL caches upon recording a new donation."""
        self._cache_stats = None
        self._cache_stats_ts = 0.0
        self._cache_top = None
        self._cache_top_ts = 0.0
        self._cache_recent = None
        self._cache_recent_ts = 0.0

    def _load_local_store(self) -> None:
        """Load donations from local JSON file."""
        if not os.path.exists(self._file_path):
            try:
                os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
                with open(self._file_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("Could not initialize local donations file %s: %s", self._file_path, e)
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._donations = data
        except Exception as e:
            logger.warning("Could not load local donations from %s: %s", self._file_path, e)
            self._donations = []

    def _save_local_store(self) -> None:
        """Persist in-memory donations to local JSON file safely with atomic rename."""
        try:
            os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
            tmp_path = f"{self._file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._donations, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._file_path)
        except Exception as e:
            logger.error("Failed to save donations to %s: %s", self._file_path, e)

    def _get_supabase_client(self) -> Any:
        """Attempt to retrieve initialized Supabase client."""
        try:
            from app.legacy import supabase  # type: ignore

            return supabase
        except Exception:
            return None

    async def record_donation(
        self,
        *,
        user_id: int,
        username: str = "",
        full_name: str = "",
        amount: float = 1.0,
        currency: str = "USD",
        tier: str = "coffee",
        note: str = "",
        blessing_sent: bool = False,
    ) -> dict[str, Any]:
        """Record a successful donation into Supabase and local cache."""
        tier_info = TIER_DETAILS.get(tier.lower(), {})
        cups = tier_info.get("cups", max(1, int(round(amount))))
        now_iso = datetime.now(timezone.utc).isoformat()

        record: dict[str, Any] = {
            "id": int(time.time() * 1000),
            "user_id": int(user_id),
            "username": (username or "").strip().lstrip("@"),
            "full_name": (full_name or "").strip() or f"User {user_id}",
            "amount": float(amount),
            "currency": currency.upper(),
            "tier": tier.lower(),
            "cups": cups,
            "note": (note or "").strip(),
            "blessing_sent": bool(blessing_sent),
            "status": "completed",
            "created_at": now_iso,
        }

        # 1. Update local cache and invalidate TTL caches under lock
        with self._lock:
            self._donations.append(record)
            self._invalidate_cache()
            self._save_local_store()

        # 2. Asynchronously insert to Supabase if configured
        sb = self._get_supabase_client()
        if sb is not None:
            try:
                def _sb_insert() -> None:
                    sb.table("donations").insert({
                        "user_id": record["user_id"],
                        "username": record["username"],
                        "full_name": record["full_name"],
                        "amount": record["amount"],
                        "currency": record["currency"],
                        "tier": record["tier"],
                        "note": record["note"],
                        "blessing_sent": record["blessing_sent"],
                        "status": record["status"],
                    }).execute()

                await asyncio.to_thread(_sb_insert)
            except Exception as e:
                logger.warning("Supabase donation insert failed (saved locally): %s", e)

        return record

    async def get_top_supporters(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get aggregated list of top donors ranked by total contribution with TTL cache."""
        now = time.monotonic()
        with self._lock:
            if self._cache_top is not None and (now - self._cache_top_ts) < CACHE_TTL_SECONDS:
                return self._cache_top[:limit]

        # Try fetching from Supabase first if available
        sb = self._get_supabase_client()
        if sb is not None:
            try:
                def _sb_query() -> list[dict[str, Any]]:
                    res = sb.table("donations").select("*").eq("status", "completed").execute()
                    return res.data or []

                sb_data = await asyncio.to_thread(_sb_query)
                if sb_data:
                    top_list = self._aggregate_top(sb_data, limit)
                    with self._lock:
                        self._cache_top = top_list
                        self._cache_top_ts = now
                    return top_list
            except Exception as e:
                logger.debug("Supabase top supporters query failed; using local store: %s", e)

        with self._lock:
            top_list = self._aggregate_top(self._donations, limit)
            self._cache_top = top_list
            self._cache_top_ts = now
            return top_list

    def _aggregate_top(self, donations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Aggregate donation records by user_id and sort descending."""
        user_map: dict[int, dict[str, Any]] = {}

        for d in donations:
            uid = int(d.get("user_id") or 0)
            if not uid:
                continue
            amt = float(d.get("amount") or 0.0)
            tier = str(d.get("tier") or "coffee").lower()
            tier_info = TIER_DETAILS.get(tier, {})
            cups = d.get("cups") or tier_info.get("cups", max(1, int(round(amt))))

            if uid not in user_map:
                user_map[uid] = {
                    "user_id": uid,
                    "username": d.get("username") or "",
                    "full_name": d.get("full_name") or f"User {uid}",
                    "total_amount": 0.0,
                    "total_cups": 0,
                    "donation_count": 0,
                    "last_donation": d.get("created_at") or "",
                }

            user_map[uid]["total_amount"] += amt
            user_map[uid]["total_cups"] += int(cups)
            user_map[uid]["donation_count"] += 1
            if d.get("full_name"):
                user_map[uid]["full_name"] = d["full_name"]
            if d.get("username"):
                user_map[uid]["username"] = d["username"]

        ranked = sorted(user_map.values(), key=lambda x: x["total_amount"], reverse=True)

        badges = ["🥇", "🥈", "🥉"]
        for idx, item in enumerate(ranked):
            item["rank"] = idx + 1
            item["badge"] = badges[idx] if idx < 3 else "⭐"

        return ranked[:limit]

    async def get_recent_donations(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get chronological list of recent donations with TTL cache."""
        now = time.monotonic()
        with self._lock:
            if self._cache_recent is not None and (now - self._cache_recent_ts) < CACHE_TTL_SECONDS:
                return self._cache_recent[:limit]

            sorted_donations = sorted(
                self._donations,
                key=lambda x: str(x.get("created_at") or ""),
                reverse=True,
            )
            recent_slice = sorted_donations[:limit]
            self._cache_recent = recent_slice
            self._cache_recent_ts = now
            return recent_slice

    async def get_donation_stats(self) -> dict[str, Any]:
        """Get summary metrics for total raised, coffee cups, and donors with TTL cache."""
        now = time.monotonic()
        with self._lock:
            if self._cache_stats is not None and (now - self._cache_stats_ts) < CACHE_TTL_SECONDS:
                return self._cache_stats

            total_usd = sum(float(d.get("amount") or 0.0) for d in self._donations)
            total_cups = sum(
                int(d.get("cups") or TIER_DETAILS.get(str(d.get("tier", "")).lower(), {}).get("cups", 1))
                for d in self._donations
            )
            unique_users = len({int(d.get("user_id")) for d in self._donations if d.get("user_id")})
            stats = {
                "total_usd": round(total_usd, 2),
                "total_cups": total_cups,
                "total_donors": unique_users,
                "total_transactions": len(self._donations),
            }
            self._cache_stats = stats
            self._cache_stats_ts = now
            return stats


# Global shared instance
donation_store = DonationStore()
