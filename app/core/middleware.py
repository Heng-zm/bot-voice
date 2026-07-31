"""HTTP timing, rate limiting, origin guards, and security headers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_admin_origin_guard_middleware",
    "_backend_only_legacy_dashboard_guard",
    "_hot_api_rate_limit_middleware",
    "_web_request_id_and_timing",
    "_web_security_headers",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
