"""Direct AI assistant HTTP handlers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_authorize_ai_api_request",
    "_normalise_ai_history",
    "_normalise_ai_message",
    "ai_assistant",
    "ai_info",
    "ai_transcribe",
    "ai_vision",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
