"""Telegram Bot API webhook registration and deletion transport."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger("telegram_webhook")

Provider = Callable[[], Any]


def build_webhook_target_url(base_url: str, secret_token: str) -> str:
    """Build the canonical secret-bearing webhook URL."""

    base = str(base_url or "").strip().rstrip("/")
    secret = str(secret_token or "").strip()
    if not base:
        raise RuntimeError("BOT_MODE=WEBHOOK requires TELEGRAM_WEBHOOK_URL.")
    if not secret:
        raise RuntimeError(
            "BOT_MODE=WEBHOOK requires TELEGRAM_WEBHOOK_SECRET_TOKEN."
        )
    return f"{base}/tg-webhook-{quote(secret, safe='')}"


def parse_allowed_updates(value: str | Iterable[str] | None) -> list[str]:
    """Normalize Telegram ``allowed_updates`` without accepting invalid names."""

    items = value.split(",") if isinstance(value, str) else list(value or [])
    allowed: list[str] = []
    for raw_item in items:
        item = str(raw_item or "").strip()
        if item and re.fullmatch(r"[a-z_]+", item) and item not in allowed:
            allowed.append(item)
    return allowed or ["message", "callback_query"]


class TelegramWebhookClient:
    """Configure, delete, and verify Telegram Bot API webhooks."""

    def __init__(
        self,
        *,
        bot_token_provider: Provider | None = None,
        target_url_builder: Callable[[str], str] | None = None,
        current_secret_provider: Provider | None = None,
        allowed_updates_provider: Provider | None = None,
        drop_pending_provider: Provider | None = None,
        limits_provider: Provider | None = None,
        set_max_attempts_provider: Provider | None = None,
        pool_snapshot_provider: Provider | None = None,
        json_loader: Callable[[bytes], Any] | None = None,
        client_factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self._bot_token_provider: Provider = bot_token_provider or (lambda: "")
        self._target_url_builder = target_url_builder or (
            lambda secret: build_webhook_target_url("", secret)
        )
        self._current_secret_provider: Provider = current_secret_provider or (
            lambda: ""
        )
        self._allowed_updates_provider: Provider = allowed_updates_provider or (
            lambda: ["message", "callback_query"]
        )
        self._drop_pending_provider: Provider = drop_pending_provider or (
            lambda: False
        )
        self._limits_provider: Provider = limits_provider or httpx.Limits
        self._set_max_attempts_provider: Provider = (
            set_max_attempts_provider or (lambda: 3)
        )
        self._pool_snapshot_provider: Provider = pool_snapshot_provider or (
            lambda: {}
        )
        self._json_loader = json_loader or self._default_json_loader
        self._client_factory = client_factory or (
            lambda **kwargs: httpx.AsyncClient(**kwargs)
        )
        self._sleep = sleep or asyncio.sleep

    def configure(self, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            if value is not None and hasattr(self, f"_{name}"):
                setattr(self, f"_{name}", value)

    @staticmethod
    def _default_json_loader(raw: bytes) -> Any:
        import json

        return json.loads(raw)

    def _bot_token(self) -> str:
        token = str(self._bot_token_provider() or "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")
        return token

    def _decode_response(self, response: Any) -> dict[str, Any]:
        try:
            data = self._json_loader(response.content)
            return data if isinstance(data, dict) else {"ok": False}
        except Exception:
            return {
                "ok": False,
                "description": str(getattr(response, "text", ""))[:500],
            }

    async def configure_for_secret(self, secret_token: str) -> None:
        """Register an explicit secret, retrying Telegram HTTP 429 responses."""

        token = self._bot_token()
        secret = str(secret_token or "").strip()
        target_url = self._target_url_builder(secret)
        api_url = f"https://api.telegram.org/bot{token}/setWebhook"
        payload = {
            "url": target_url,
            "allowed_updates": parse_allowed_updates(
                self._allowed_updates_provider()
            ),
            "drop_pending_updates": bool(self._drop_pending_provider()),
            "secret_token": secret,
        }
        timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)
        try:
            configured_attempts = int(self._set_max_attempts_provider())
        except (TypeError, ValueError):
            configured_attempts = 3
        max_attempts = max(1, min(10, configured_attempts))
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            async with self._client_factory(
                timeout=timeout,
                limits=self._limits_provider(),
            ) as client:
                response = await client.post(api_url, json=payload)
            data = self._decode_response(response)

            if response.status_code == 429:
                try:
                    retry_after = int(
                        (data.get("parameters") or {}).get("retry_after") or 1
                    )
                except (TypeError, ValueError):
                    retry_after = 1
                last_error = (
                    "Telegram setWebhook rate-limited status=429 "
                    f"response={str(data)[:500]}"
                )
                if attempt < max_attempts:
                    logger.warning(
                        "Telegram setWebhook rate-limited; retrying after %ss "
                        "attempt=%s/%s",
                        retry_after,
                        attempt,
                        max_attempts,
                    )
                    await self._sleep(max(1, retry_after))
                    continue

            if response.status_code >= 400 or not bool(data.get("ok")):
                raise RuntimeError(
                    "Telegram setWebhook failed "
                    f"status={response.status_code} response={str(data)[:500]}"
                )

            pool = dict(self._pool_snapshot_provider() or {})
            logger.info(
                "Telegram webhook configured via HTTPX url=%s "
                "max_connections=%s keepalive=%s",
                target_url,
                pool.get("max_connections", "unknown"),
                pool.get("keepalive_connections", "unknown"),
            )
            return

        raise RuntimeError(
            last_error or "Telegram setWebhook rate-limited after retries."
        )

    async def configure_current(self) -> None:
        await self.configure_for_secret(str(self._current_secret_provider() or ""))

    async def delete(self, drop_pending: bool = True) -> None:
        """Delete and verify the webhook before polling is allowed."""

        token = self._bot_token()
        delete_url = f"https://api.telegram.org/bot{token}/deleteWebhook"
        info_url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
        timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)
        last_error = "Telegram did not confirm webhook deletion."

        async with self._client_factory(
            timeout=timeout,
            limits=self._limits_provider(),
        ) as client:
            for attempt in range(1, 6):
                try:
                    response = await client.post(
                        delete_url,
                        json={"drop_pending_updates": bool(drop_pending)},
                    )
                    data = self._decode_response(response)
                    if response.status_code >= 400 or not bool(data.get("ok")):
                        raise RuntimeError(
                            "Telegram deleteWebhook failed "
                            f"status={response.status_code} "
                            f"response={str(data)[:500]}"
                        )

                    info_response = await client.get(info_url)
                    info = self._decode_response(info_response)
                    if info_response.status_code >= 400 or not bool(info.get("ok")):
                        raise RuntimeError(
                            "Telegram getWebhookInfo failed "
                            f"status={info_response.status_code} "
                            f"response={str(info)[:500]}"
                        )

                    webhook_url = str(
                        (info.get("result") or {}).get("url") or ""
                    ).strip()
                    if not webhook_url:
                        logger.info(
                            "Telegram webhook deletion confirmed; polling may start."
                        )
                        return
                    last_error = (
                        "Telegram still reports an active webhook after deleteWebhook."
                    )
                except (httpx.HTTPError, RuntimeError) as exc:
                    last_error = str(exc)

                if attempt < 5:
                    logger.warning(
                        "Webhook deletion not confirmed; retrying attempt=%s/5.",
                        attempt,
                    )
                    await self._sleep(min(3.0, float(attempt)))

        raise RuntimeError(last_error)


_DEFAULT_CLIENT = TelegramWebhookClient()


def configure_telegram_webhook_client(**kwargs: Any) -> TelegramWebhookClient:
    _DEFAULT_CLIENT.configure(**kwargs)
    return _DEFAULT_CLIENT


async def _configure_telegram_webhook_via_http_for_secret(
    secret_token: str,
) -> None:
    await _DEFAULT_CLIENT.configure_for_secret(secret_token)


async def _configure_telegram_webhook_via_http() -> None:
    await _DEFAULT_CLIENT.configure_current()


async def set_telegram_webhook() -> None:
    await _configure_telegram_webhook_via_http()


async def _delete_telegram_webhook_via_http(
    drop_pending: bool = True,
) -> None:
    await _DEFAULT_CLIENT.delete(drop_pending=drop_pending)


async def safe_send(call, retries: int = 3, delay: float = 2.0):
    """Compatibility import retained until Telegram delivery extraction."""

    from app._legacy_bridge import legacy_module

    return await legacy_module().safe_send(call, retries=retries, delay=delay)


def __getattr__(name: str) -> Any:
    if name in {"_telegram_start_polling_runtime", "_telegram_stop_polling_runtime"}:
        from app._legacy_bridge import legacy_module

        return getattr(legacy_module(), name)
    raise AttributeError(name)


__all__ = [
    "TelegramWebhookClient",
    "_configure_telegram_webhook_via_http",
    "_configure_telegram_webhook_via_http_for_secret",
    "_delete_telegram_webhook_via_http",
    "_telegram_start_polling_runtime",
    "_telegram_stop_polling_runtime",
    "build_webhook_target_url",
    "configure_telegram_webhook_client",
    "parse_allowed_updates",
    "safe_send",
    "set_telegram_webhook",
]
