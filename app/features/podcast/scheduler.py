"""Podcast background scheduler and automated broadcast loop."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("app.features.podcast.scheduler")


async def periodic_podcast_scheduler(poll_interval: float = 30.0) -> None:
    from app.services.podcast.handlers import periodic_podcast_scheduler as _handler

    return await _handler(poll_interval=poll_interval)


__all__ = ["periodic_podcast_scheduler"]
