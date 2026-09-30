#!/usr/bin/env python3
"""Database migration CLI entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add scripts/database to sys.path so it can find _migration_core
DB_SCRIPTS_DIR = Path(__file__).resolve().parent / "database"
if str(DB_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(DB_SCRIPTS_DIR))

if __name__ == "__main__":
    from migrate_data import main
    main()
