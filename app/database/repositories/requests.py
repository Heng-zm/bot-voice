"""Request logs persistence repository."""

from __future__ import annotations

import logging
from typing import Any

from app.core.telemetry.logger import RequestRecord, get_request_log_store

logger = logging.getLogger("app.database.repositories.requests")


class RequestLogRepository:
    """Repository accessing runtime in-memory and database request logs."""

    def __init__(self) -> None:
        self.store = get_request_log_store()

    def get_recent_requests(self, limit: int = 50, category: str | None = None) -> list[RequestRecord]:
        return self.store.get_recent(limit=limit, category=category)

    def get_metrics(self) -> dict[str, Any]:
        return self.store.get_metrics()


_REPO = RequestLogRepository()


def get_request_repository() -> RequestLogRepository:
    return _REPO


__all__ = ["RequestLogRepository", "get_request_repository"]
