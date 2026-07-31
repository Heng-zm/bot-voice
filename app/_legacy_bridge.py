"""Lazy access helpers for the staged monolith migration."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Iterable


def legacy_module() -> ModuleType:
    """Import and return the preserved runtime module on first use."""
    return import_module("app.legacy")


def exported_getattr(module_name: str, names: Iterable[str]) -> Callable[[str], Any]:
    """Build a PEP 562 module ``__getattr__`` for selected legacy exports."""
    allowed = frozenset(names)

    def _getattr(name: str) -> Any:
        if name not in allowed:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        return getattr(legacy_module(), name)

    return _getattr


def exported_dir(current: Iterable[str], names: Iterable[str]) -> list[str]:
    """Return a useful ``dir(module)`` result for lazy compatibility modules."""
    return sorted(set(current) | set(names))
