"""Network binding configuration shared by runtime and health checks."""

from __future__ import annotations

import os
from collections.abc import Mapping


def web_server_port(
    environ: Mapping[str, str] | None = None,
    *,
    default: int = 8080,
) -> int:
    """Resolve the externally allocated web port.

    Pterodactyl-based hosts, including Wispbyte deployments, expose the
    allocation as ``SERVER_PORT``. It must take precedence over a generic
    ``PORT`` default bundled with application settings.
    """

    source = os.environ if environ is None else environ
    for name in ("SERVER_PORT", "WISPBYTE_PORT", "PORT"):
        raw = str(source.get(name, "") or "").strip()
        if not raw:
            continue
        try:
            port = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid TCP port.") from exc
        if not 1 <= port <= 65_535:
            raise ValueError(f"{name} must be between 1 and 65535.")
        return port

    fallback = int(default)
    if not 1 <= fallback <= 65_535:
        raise ValueError("Default web port must be between 1 and 65535.")
    return fallback


__all__ = ["web_server_port"]
