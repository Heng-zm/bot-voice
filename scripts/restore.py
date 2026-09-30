#!/usr/bin/env python3
"""Database restore CLI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

DB_SCRIPTS_DIR = Path(__file__).resolve().parent / "database"
if str(DB_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(DB_SCRIPTS_DIR))

if __name__ == "__main__":
    from restore_data import main
    main()
