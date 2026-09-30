#!/usr/bin/env python3
"""Database seeding utility for default configurations and initial bot states."""

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
logger = logging.getLogger("seed")

DEFAULT_SETTINGS = {
    "maintenance_mode": "false",
    "channel_narrator_enabled": "true",
    "channel_narrator_gender": "female",
    "channel_narrator_speed": "1.0",
    "default_tts_model": "edge",
    "default_tts_gender": "female",
    "rate_limit_per_minute": "30",
}


async def seed_defaults() -> None:
    """Seed initial bot settings into repository store."""
    from app.database.client import get_db_client
    from app.database.repositories.settings import SettingsStore

    logger.info("Connecting to database...")
    client = get_db_client()
    if not client.is_configured:
        logger.warning("Database client is not configured; using local memory/fallback store.")

    store = SettingsStore(client)
    for key, val in DEFAULT_SETTINGS.items():
        existing = await store.get(key)
        if existing is None:
            logger.info("Seeding default '%s' = '%s'", key, val)
            await store.set(key, val)
        else:
            logger.info("Setting '%s' already exists (value='%s')", key, existing)

    logger.info("Database seeding completed successfully.")


def main() -> None:
    try:
        asyncio.run(seed_defaults())
    except KeyboardInterrupt:
        logger.info("Seeding interrupted.")
    except Exception as exc:
        logger.error("Seeding failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
