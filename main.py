"""Compatibility launcher for deployment panels with varying working directories."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root so absolute ``app.*`` imports resolve in either launch mode.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.main import app, main  # noqa: E402

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
