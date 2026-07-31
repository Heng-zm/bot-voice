"""Redis-backed Telegram webhook replay protection."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_telegram_webhook_replay_key",
    "_telegram_webhook_update_claim",
    "_telegram_webhook_update_complete",
    "_telegram_webhook_update_id",
    "_telegram_webhook_update_release",
    "_trim_webhook_memory_locked",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
