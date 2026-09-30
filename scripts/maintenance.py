#!/usr/bin/env python3
"""System maintenance CLI: temporary file cleanup, cache trimming, and DB vacuuming."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("maintenance")


async def run_maintenance() -> None:
    """Run all periodic maintenance routines."""
    from app.features.admin.service import admin_service

    logger.info("Starting maintenance routine...")
    opt_result = await admin_service.run_optimization()
    logger.info("Optimization result: %s", opt_result)

    cleanup_result = admin_service.run_cleanup()
    logger.info("Cleanup result: %s", cleanup_result)
    logger.info("All maintenance tasks finished.")


def main() -> None:
    try:
        asyncio.run(run_maintenance())
    except KeyboardInterrupt:
        logger.info("Maintenance cancelled.")
    except Exception as exc:
        logger.error("Maintenance failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
