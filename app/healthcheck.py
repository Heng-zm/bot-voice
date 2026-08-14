"""Role-aware container health check for web and worker processes."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from app.core.network import web_server_port


def _role() -> str:
    configured = str(os.getenv("PROCESS_ROLE", "") or "").strip().lower()
    if configured:
        return configured
    try:
        command = Path("/proc/1/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        command = ""
    return "worker" if "app.worker" in command else "web"


def _check_worker() -> None:
    redis_url = str(os.getenv("REDIS_URL", "") or "").strip()
    if not redis_url:
        raise RuntimeError("REDIS_URL is not configured.")
    import redis

    client = redis.from_url(
        redis_url,
        socket_connect_timeout=3,
        socket_timeout=3,
        max_connections=1,
    )
    try:
        if not client.ping():
            raise RuntimeError("Redis ping failed.")
    finally:
        client.close()


def _check_web() -> None:
    port = web_server_port()
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/readyz",
        timeout=4,
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"Readiness returned HTTP {response.status}.")


def main() -> None:
    try:
        if _role() == "worker":
            _check_worker()
        else:
            _check_web()
    except Exception as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
