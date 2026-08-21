"""Compatibility launcher for deployment panels that run ``python main.py``."""

from app.main import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
