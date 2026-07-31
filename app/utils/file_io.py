"""Atomic synchronous and asynchronous binary file utilities."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_read_file_bytes_async",
    "_read_file_bytes_sync",
    "_write_file_bytes_sync",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
