"""Telegram update parsing, commands, callbacks, and message handlers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "cmd_version",
    "_run_bot",
    "error_handler",
    "on_any_media",
    "on_audio_file",
    "on_callback",
    "on_help",
    "on_photo",
    "on_start",
    "on_text",
    "on_voice",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
