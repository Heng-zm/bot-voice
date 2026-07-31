"""Redis initialization, retry, JSON cache, and locking helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_redis_delete",
    "_redis_get_json",
    "_redis_key",
    "_redis_set_json",
    "redis_call",
    "redis_call_sync",
    "redis_client",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
