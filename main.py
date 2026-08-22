"""Compatibility launcher for deployment panels with varying working directories."""

from __future__ import annotations

import sys
from pathlib import Path

# ``python -m deploy.main`` starts with /home/container on sys.path instead of
# /home/container/deploy. Add the directory containing this launcher so the
# application's absolute ``app.*`` imports resolve in either launch mode.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.main import main  # noqa: E402 - import follows path bootstrap

__all__ = ["main"]


if __name__ == "__main__":
    main()
