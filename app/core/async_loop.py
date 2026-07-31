"""Dedicated background event-loop and awaitable resolution helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_await_sync_value",
    "_get_async_resolver_loop",
    "_resolve_maybe_awaitable_sync",
    "_shutdown_async_resolver_loop",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
