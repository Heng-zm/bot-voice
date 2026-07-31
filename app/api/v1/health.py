"""Health, readiness, and administrative diagnostics handlers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_admin_diagnostics_payload",
    "health_check",
    "ping",
    "readyz",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
