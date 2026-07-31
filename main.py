"""Backward-compatible root entry point.

New code should import :mod:`app.main` or run ``python -m app.main``.
"""

from __future__ import annotations

from app import legacy as _legacy
from app.main import app, create_app, main

__all__ = ["app", "create_app", "main"]


def __getattr__(name: str):
    """Expose legacy symbols for callers of the former root module."""

    return getattr(_legacy, name)


if __name__ == "__main__":
    main()
