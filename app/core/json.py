"""Fast JSON codecs and Starlette response integration."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_FastJSONResponse",
    "_json_dumps_fast",
    "_json_loads_fast",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
