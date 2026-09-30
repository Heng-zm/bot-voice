"""Unified System Optimization and Resource Cleanup Service.

Coordinates temp file purging, memory reclamation, cache trimming,
database pruning, and runtime performance knob application.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import gc
import logging
import time
from typing import Any

from app.utils.file_io import sweep_stale_temp_files

logger = logging.getLogger("app.optimization")


def run_system_cleanup_sync(*, prune_db: bool = False) -> dict[str, Any]:
    """Execute synchronous system cleanup tasks (Temp files, in-memory caches, GC, optional DB)."""
    stats: dict[str, Any] = {
        "timestamp": time.time(),
        "temp_files_swept": 0,
        "audio_cache_trimmed": 0,
        "gemini_cache_cleared": 0,
        "gc_objects_freed": 0,
        "db_history_pruned": 0,
        "db_cache_pruned": 0,
    }

    # 1. Sweep stale temporary media files older than 30 minutes (1800s)
    try:
        stats["temp_files_swept"] = sweep_stale_temp_files(max_age_seconds=1800.0)
    except Exception as exc:
        logger.warning("Temp file sweep error: %s", exc)

    # 2. Trim expired TTS in-memory audio entries
    try:
        from app.services.tts.cache import get_tts_cache

        stats["audio_cache_trimmed"] = get_tts_cache().trim_expired()
    except Exception as exc:
        logger.debug("Audio cache trim error: %s", exc)

    # 3. Trim or clear Gemini AI response cache
    try:
        from app.services.ai.gemini import clear_gemini_response_cache

        stats["gemini_cache_cleared"] = clear_gemini_response_cache()
    except Exception as exc:
        logger.debug("Gemini response cache clear error: %s", exc)

    # 4. Trim TikTok downloader in-memory cache if oversized
    try:
        from app.services.downloader.tiktok import _TIKTOK_CACHE, _TIKTOK_CACHE_LOCK

        with _TIKTOK_CACHE_LOCK:
            if len(_TIKTOK_CACHE) > 200:
                trimmed = 0
                while len(_TIKTOK_CACHE) > 100:
                    _TIKTOK_CACHE.popitem(last=False)
                    trimmed += 1
                stats["tiktok_cache_trimmed"] = trimmed
    except Exception as exc:
        logger.debug("TikTok cache trim error: %s", exc)

    # 5. Database bounded pruning (if requested and Supabase is configured)
    if prune_db:
        try:
            from app import legacy

            prune_fn = getattr(legacy, "db_run_periodic_pruning", None)
            if callable(prune_fn):
                db_res = prune_fn(max_batches=4, batch_size=500)
                if isinstance(db_res, dict):
                    stats["db_history_pruned"] = int(db_res.get("pruned_history", 0) or 0)
                    stats["db_cache_pruned"] = int(db_res.get("pruned_text_cache", 0) or 0)
        except Exception as exc:
            logger.warning("Database pruning error during cleanup: %s", exc)

    # 5. Force Python cyclic garbage collection
    try:
        stats["gc_objects_freed"] = gc.collect()
    except Exception as exc:
        logger.debug("GC error: %s", exc)

    return stats


async def run_system_optimization_async(
    *,
    admin_id: int = 0,
    prune_db: bool = True,
) -> dict[str, Any]:
    """Execute complete asynchronous system optimization."""
    loop = asyncio.get_running_loop()

    # Run synchronous IO / GC / DB pruning in thread executor to avoid blocking event loop
    cleanup_stats = await loop.run_in_executor(
        None,
        lambda: run_system_cleanup_sync(prune_db=prune_db),
    )

    # Apply all bot performance knobs
    applied_knobs: list[str] = []
    try:
        from app import legacy

        perf_fn = getattr(legacy, "_apply_all_bot_performance_settings", None)
        if callable(perf_fn):
            applied_knobs = await perf_fn(
                admin_id=admin_id,
                force=True,
                save_to_db=False,
                persist_runtime=False,
            )
    except Exception as exc:
        logger.warning("Performance settings application error: %s", exc)

    cleanup_stats["perf_knobs_applied"] = applied_knobs
    logger.info(
        "System optimization complete: swept %d temp files, freed %d GC objects, trimmed %d audio cache entries, %d perf knobs verified",
        cleanup_stats.get("temp_files_swept", 0),
        cleanup_stats.get("gc_objects_freed", 0),
        cleanup_stats.get("audio_cache_trimmed", 0),
        len(applied_knobs),
    )
    return cleanup_stats


__all__ = [
    "run_system_cleanup_sync",
    "run_system_optimization_async",
]
