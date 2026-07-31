"""Telegram HTTP client, retry, polling, and webhook registration helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_configure_telegram_webhook_via_http",
    "_delete_telegram_webhook_via_http",
    "_telegram_start_polling_runtime",
    "_telegram_stop_polling_runtime",
    "safe_send",
    "set_telegram_webhook",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
