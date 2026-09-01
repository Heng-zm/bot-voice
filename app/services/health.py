"""Lightweight health check HTTP server for Render/Cloud Web Services.

Binds to $PORT to satisfy cloud platform port checks without heavy web frameworks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

logger = logging.getLogger("app.health")


async def _handle_http_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
        request_line = data.decode("utf-8", errors="ignore").split("\r\n")[0] if data else ""

        body = b'{"status":"ok","service":"telegram-bot-voice"}\n'
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
            b"Connection: close\r\n\r\n" + body
        )
        writer.write(response)
        await writer.drain()
    except Exception:
        pass
    finally:
        with suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def start_health_server(port: int | None = None) -> asyncio.Server | None:
    """Start listening on the configured cloud platform port."""
    target_port = int(port or os.environ.get("PORT", "8080"))
    try:
        server = await asyncio.start_server(_handle_http_client, "0.0.0.0", target_port)
        logger.info("Health check server listening on port %s", target_port)
        return server
    except Exception as exc:
        logger.warning("Could not bind health check server on port %s: %s", target_port, exc)
        return None


__all__ = ["start_health_server"]
