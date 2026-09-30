"""Persistent subscriber store for Daily Morning Podcast."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STORE_FILE = os.path.join(PROJECT_ROOT, "data", "podcast_subscribers.json")


class PodcastSubscriberStore:
    """Store and manage Daily Morning Podcast subscriptions."""

    def __init__(self, file_path: str = STORE_FILE) -> None:
        self._file_path = file_path
        self._lock = threading.Lock()
        self._subscribers: set[int] = set()
        self._last_broadcast_date: str = ""
        self._load()

    def _load(self) -> None:
        try:
            if os.path.isfile(self._file_path):
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
                    self._subscribers = {int(x) for x in data.get("subscribers", []) if str(x).isdigit() or str(x).lstrip("-").isdigit()}
                    self._last_broadcast_date = str(data.get("last_broadcast_date", "")).strip()
        except Exception as exc:
            logger.warning("Failed to load podcast subscriber store %s: %s", self._file_path, exc)
            self._subscribers = set()
            self._last_broadcast_date = ""

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
            tmp = f"{self._file_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "subscribers": sorted(list(self._subscribers)),
                        "last_broadcast_date": self._last_broadcast_date,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(tmp, self._file_path)
        except Exception as exc:
            logger.error("Failed to save podcast subscriber store: %s", exc)

    def subscribe(self, chat_id: int) -> bool:
        """Add chat_id to daily morning podcast subscribers. Returns True if newly added."""
        with self._lock:
            cid = int(chat_id)
            if cid in self._subscribers:
                return False
            self._subscribers.add(cid)
            self._save()
            return True

    def unsubscribe(self, chat_id: int) -> bool:
        """Remove chat_id from daily morning podcast subscribers. Returns True if removed."""
        with self._lock:
            cid = int(chat_id)
            if cid not in self._subscribers:
                return False
            self._subscribers.remove(cid)
            self._save()
            return True

    def unsubscribe_batch(self, chat_ids: list[int] | set[int]) -> int:
        """Remove multiple chat_ids in a single batch with a single atomic disk write."""
        with self._lock:
            removed = 0
            for cid in chat_ids:
                try:
                    c = int(cid)
                    if c in self._subscribers:
                        self._subscribers.remove(c)
                        removed += 1
                except Exception:
                    pass
            if removed > 0:
                self._save()
            return removed

    def is_subscribed(self, chat_id: int) -> bool:
        with self._lock:
            return int(chat_id) in self._subscribers

    def get_all_subscribers(self) -> list[int]:
        with self._lock:
            return sorted(list(self._subscribers))

    def count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def get_last_broadcast_date(self) -> str:
        with self._lock:
            return self._last_broadcast_date

    def set_last_broadcast_date(self, date_str: str) -> None:
        with self._lock:
            self._last_broadcast_date = str(date_str).strip()
            self._save()


podcast_store = PodcastSubscriberStore()
