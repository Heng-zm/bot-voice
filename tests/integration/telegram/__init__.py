"""Integration tests for Telegram functionality."""

from __future__ import annotations

import sys
import os

# If unittest discovers with -s tests/integration, this directory can shadow
# the top-level 'telegram' package (python-telegram-bot).
# Self-heal by forwarding to the real site-packages telegram module if shadowed.
if __name__ == "telegram":
    this_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if os.path.abspath(p) != this_parent]
        sys.modules.pop("telegram", None)
        import telegram as _real_telegram
        sys.modules["telegram"] = _real_telegram
        globals().update(_real_telegram.__dict__)
    finally:
        sys.path = old_path
