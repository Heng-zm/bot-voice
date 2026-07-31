"""Backward-compatible root entry point.

New code should import ``app.main`` or run ``python -m app.main``.
"""

from __future__ import annotations

from app import legacy as _legacy
from app.main import app, create_app, main

__all__ = ["app", "create_app", "main"]


def __getattr__(name: str):
    return getattr(_legacy, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy)))


if __name__ == "__main__":
    main()
