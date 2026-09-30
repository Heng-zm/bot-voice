"""Backward compatibility shim for app.services.tasks.batcher."""

from __future__ import annotations

from app.database.batcher import (
    DatabaseBatcher,
    get_database_batcher,
)

__all__ = [
    "DatabaseBatcher",
    "get_database_batcher",
]
