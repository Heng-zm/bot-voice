"""Telegram-only bot entry point."""

from __future__ import annotations

# Support deployment panels that execute ``python app/main.py`` directly.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app import legacy


def main() -> None:
    """Run the Telegram polling bot and its internal schedulers."""

    legacy.main()


if __name__ == "__main__":
    main()
