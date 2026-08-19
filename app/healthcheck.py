"""Container health check for the single web/Telegram process."""

from __future__ import annotations

import os
import sys
import urllib.request


def main() -> None:
    port = str(os.getenv("PORT", "8080") or "8080")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=4) as response:
            if response.status != 200:
                raise RuntimeError(f"Readiness returned HTTP {response.status}.")
    except Exception as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
