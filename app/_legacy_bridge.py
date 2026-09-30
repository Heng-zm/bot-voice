"""Small compatibility bridge for modules not yet extracted from ``app.legacy``.

New code should import native services directly. This bridge exists only for the
remaining staged-migration accessors and can be deleted once ``legacy.py`` is
fully retired.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from importlib import import_module
from types import ModuleType
from typing import Any


def legacy_module() -> ModuleType:
    """Return the live legacy module without caching a second state container."""

    return import_module("app.legacy")


def exported_getattr(module_name: str, exported_names: Collection[str]) -> Callable[[str], Any]:
    """Create a strict module ``__getattr__`` forwarding approved names only."""

    allowed = frozenset(str(name) for name in exported_names)

    def _getattr(name: str) -> Any:
        if name not in allowed:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            return getattr(legacy_module(), name)
        except AttributeError as exc:
            raise AttributeError(
                f"module {module_name!r} has no exported legacy attribute {name!r}"
            ) from exc

    return _getattr


def exported_dir(namespace: Mapping[str, Any], exported_names: Collection[str]) -> list[str]:
    """Return deterministic names for compatibility modules."""

    return sorted(set(namespace).union(str(name) for name in exported_names))


__all__ = ["exported_dir", "exported_getattr", "legacy_module"]
