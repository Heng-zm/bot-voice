"""Application settings and environment fallback helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "AppSettings",
    "SETTINGS",
    "_env_bool",
    "_env_float",
    "_env_int",
    "_env_str",
    "_perf_default",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
