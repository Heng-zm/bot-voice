"""Telegram bot application package.

The production runtime currently lives in :mod:`app.legacy`.  The surrounding
packages are stable module boundaries that allow the monolith to be migrated
incrementally without changing deployed behavior.
"""

__all__ = ["main"]
