"""Telegram webhook routes, payload validation, and update claims."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_process_telegram_webhook_request",
    "_read_limited_webhook_body",
    "_telegram_webhook_update_claim",
    "_telegram_webhook_update_complete",
    "_telegram_webhook_update_release",
    "telegram_webhook",
    "telegram_webhook_ingest",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
