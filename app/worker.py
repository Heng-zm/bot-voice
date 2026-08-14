"""Dedicated durable job worker process.

Run with ``python -m app.worker``. This process does not bind an HTTP port or
start Telegram polling/webhook ingestion; it only consumes Redis jobs.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress

from app.runtime import get_runtime_context
from app.services.build_info import get_build_info
from app.services.jobs.runtime import job_worker_snapshot

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    runtime = get_runtime_context()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        with suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(signum, request_stop)

    await runtime.start(None, owner="worker-process", role="worker")
    snapshot = runtime.snapshot()
    build = get_build_info(role="worker", started_at=snapshot.get("started_at"))
    logger.info(
        "Durable worker ready version=%s commit=%s count=%s artifact_backend=%s shared=%s",
        build["version"],
        build["commit_short"] or "-",
        snapshot["workers"].get("count", 0),
        snapshot["artifacts"].get("backend"),
        snapshot["artifacts"].get("shared"),
    )
    try:
        while not stop_event.is_set():
            workers = job_worker_snapshot()
            if workers.get("count", 0) and not workers.get("healthy"):
                raise RuntimeError("A durable worker task stopped unexpectedly.")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=5.0)
    finally:
        await runtime.stop(owner="worker-process")


def main() -> None:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker shutdown requested.")


if __name__ == "__main__":
    main()
