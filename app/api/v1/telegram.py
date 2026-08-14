"""Telegram webhook HTTP transport and request validation."""

from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response
from telegram import Update

from app.services.telegram.deduplication import (
    _telegram_webhook_update_claim,
    _telegram_webhook_update_complete,
    _telegram_webhook_update_release,
)

logger = logging.getLogger("telegram_webhook")

Provider = Callable[[], Any]
ResponseFactory = Callable[[dict[str, Any], int], Response]
ClaimUpdate = Callable[..., Awaitable[str | tuple[str, str | None]]]
FinishUpdate = Callable[..., Awaitable[bool]]


async def _read_limited_webhook_body(req: Request, max_body: int) -> bytes:
    """Read a request stream with strict Content-Length and streamed limits."""

    content_length = req.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_body:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Max {max_body} bytes.",
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid Content-Length header.",
            ) from exc

    chunks: list[bytes] = []
    total = 0
    async for chunk in req.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_body:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large. Max {max_body} bytes.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


class TelegramWebhookTransport:
    """Validate and dispatch Telegram webhook updates for one runtime."""

    def __init__(
        self,
        *,
        bot_mode_provider: Provider | None = None,
        secret_provider: Provider | None = None,
        application_provider: Provider | None = None,
        ready_provider: Provider | None = None,
        active_owner_provider: Provider | None = None,
        owner_snapshot_provider: Provider | None = None,
        max_body_provider: Provider | None = None,
        json_loader: Callable[[bytes], Any] | None = None,
        response_factory: ResponseFactory | None = None,
        metric_callback: Callable[[str], None] | None = None,
        claim_update: ClaimUpdate = _telegram_webhook_update_claim,
        complete_update: FinishUpdate = _telegram_webhook_update_complete,
        release_update: FinishUpdate = _telegram_webhook_update_release,
        processing_timeout_seconds: float = 55.0,
    ) -> None:
        self.configure(
            bot_mode_provider=bot_mode_provider,
            secret_provider=secret_provider,
            application_provider=application_provider,
            ready_provider=ready_provider,
            active_owner_provider=active_owner_provider,
            owner_snapshot_provider=owner_snapshot_provider,
            max_body_provider=max_body_provider,
            json_loader=json_loader,
            response_factory=response_factory,
            metric_callback=metric_callback,
        )
        self._claim_update = claim_update
        self._complete_update = complete_update
        self._release_update = release_update
        self._processing_timeout_seconds = max(
            1.0, float(processing_timeout_seconds)
        )
        self._background_tasks: set[asyncio.Task[None]] = set()

    def configure(
        self,
        *,
        bot_mode_provider: Provider | None = None,
        secret_provider: Provider | None = None,
        application_provider: Provider | None = None,
        ready_provider: Provider | None = None,
        active_owner_provider: Provider | None = None,
        owner_snapshot_provider: Provider | None = None,
        max_body_provider: Provider | None = None,
        json_loader: Callable[[bytes], Any] | None = None,
        response_factory: ResponseFactory | None = None,
        metric_callback: Callable[[str], None] | None = None,
    ) -> None:
        if not hasattr(self, "_bot_mode_provider"):
            self._bot_mode_provider: Provider = lambda: "POLLING"
            self._secret_provider: Provider = lambda: ""
            self._application_provider: Provider = lambda: None
            self._ready_provider: Provider = lambda: False
            self._active_owner_provider: Provider = lambda: False
            self._owner_snapshot_provider: Provider = lambda: {}
            self._max_body_provider: Provider = lambda: 2 * 1024 * 1024
            self._json_loader: Callable[[bytes], Any] = self._default_json_loader
            self._response_factory: ResponseFactory = self._default_response
            self._metric_callback: Callable[[str], None] = lambda _name: None
        if bot_mode_provider is not None:
            self._bot_mode_provider = bot_mode_provider
        if secret_provider is not None:
            self._secret_provider = secret_provider
        if application_provider is not None:
            self._application_provider = application_provider
        if ready_provider is not None:
            self._ready_provider = ready_provider
        if active_owner_provider is not None:
            self._active_owner_provider = active_owner_provider
        if owner_snapshot_provider is not None:
            self._owner_snapshot_provider = owner_snapshot_provider
        if max_body_provider is not None:
            self._max_body_provider = max_body_provider
        if json_loader is not None:
            self._json_loader = json_loader
        if response_factory is not None:
            self._response_factory = response_factory
        if metric_callback is not None:
            self._metric_callback = metric_callback

    @staticmethod
    def _default_json_loader(raw: bytes) -> Any:
        import json

        return json.loads(raw)

    @staticmethod
    def _default_response(payload: dict[str, Any], status_code: int) -> Response:
        return JSONResponse(payload, status_code=status_code)

    def _response(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
    ) -> Response:
        return self._response_factory(payload, status_code)

    @staticmethod
    def _client_host(req: Request) -> str:
        return req.client.host if req.client else "unknown"

    async def _run_update_background(
        self,
        application: Any,
        update: Update,
        update_id: int,
        claim_token: str | None,
    ) -> None:
        try:
            await asyncio.wait_for(
                application.process_update(update),
                timeout=self._processing_timeout_seconds,
            )
            await self._complete_update(update_id, claim_token=claim_token)
        except TimeoutError:
            logger.warning(
                "Telegram update processing timed out update_id=%s", update_id
            )
            await self._release_update(update_id, claim_token=claim_token)
        except Exception as exc:
            logger.error(
                "Telegram update processing failed update_id=%s: %s",
                update_id,
                exc,
                exc_info=True,
            )
            await self._release_update(update_id, claim_token=claim_token)

    async def process(
        self,
        req: Request,
        path_secret_token: str | None = None,
    ) -> Response:
        """Validate, parse, claim, and dispatch a Telegram webhook update."""

        mode = str(self._bot_mode_provider() or "POLLING").upper()
        if mode != "WEBHOOK":
            logger.info("Webhook update ignored because BOT_MODE=%s.", mode)
            return self._response(
                {"status": "ignored", "reason": "not_webhook_mode"}
            )

        expected_secret = str(self._secret_provider() or "").strip()
        if not expected_secret:
            logger.error(
                "Rejected Telegram webhook request because the secret is missing."
            )
            raise HTTPException(
                status_code=503,
                detail="Telegram webhook secret is not configured.",
            )
        if path_secret_token is not None and not hmac.compare_digest(
            str(path_secret_token), expected_secret
        ):
            logger.warning(
                "Rejected Telegram webhook request with invalid path secret from %s",
                self._client_host(req),
            )
            raise HTTPException(status_code=403, detail="Invalid webhook path secret.")

        got_secret = (
            req.headers.get("X-Telegram-Bot-Api-Secret-Token") or ""
        ).strip()
        if not hmac.compare_digest(got_secret, expected_secret):
            logger.warning(
                "Rejected Telegram webhook request with invalid secret header from %s",
                self._client_host(req),
            )
            raise HTTPException(status_code=403, detail="Invalid webhook secret.")

        application = self._application_provider()
        if application is None or not bool(self._ready_provider()):
            logger.warning(
                "Telegram webhook rejected with 503 because application is not ready."
            )
            raise HTTPException(
                status_code=503,
                detail="Telegram application is starting. Please retry.",
            )
        if not bool(self._active_owner_provider()):
            owner = dict(self._owner_snapshot_provider() or {}).get("owner")
            logger.info(
                "Webhook update acknowledged by standby instance; active owner=%s",
                owner,
            )
            return self._response(
                {"status": "ignored", "reason": "standby_instance"}
            )

        max_body = min(max(1, int(self._max_body_provider())), 2 * 1024 * 1024)
        try:
            raw = await _read_limited_webhook_body(req, max_body)
            data = self._json_loader(raw)
            if not isinstance(data, dict):
                raise ValueError("Telegram webhook JSON must be an object.")
            update = Update.de_json(data, application.bot)
        except HTTPException:
            raise
        except Exception as exc:
            error_id = secrets.token_hex(6)
            logger.warning(
                "Invalid Telegram webhook payload ignored error_id=%s: %s",
                error_id,
                exc,
                exc_info=True,
            )
            return self._response(
                {
                    "status": "ignored",
                    "reason": "invalid_payload",
                    "reference": error_id,
                }
            )

        update_id = getattr(update, "update_id", None)
        try:
            claim_state, claim_token = await self._claim_update(
                update_id,
                include_token=True,
            )
            if claim_state == "completed":
                logger.info(
                    "Completed Telegram webhook update ignored update_id=%s",
                    update_id,
                )
                self._metric_callback("replay_dropped")
                return self._response({"status": "ok", "duplicate": True})
            if claim_state == "processing":
                return self._response(
                    {
                        "status": "ok",
                        "duplicate": True,
                        "reason": "already_processing",
                    }
                )
            if claim_state != "claimed" or update_id is None:
                raise ValueError(f"Invalid Telegram update id: {update_id!r}")

            task = asyncio.create_task(
                self._run_update_background(
                    application,
                    update,
                    int(update_id),
                    claim_token,
                ),
                name=f"tg-update-{update_id}",
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return self._response({"status": "ok", "update_id": update_id})
        except Exception as exc:
            error_id = secrets.token_hex(6)
            logger.error(
                "Telegram webhook ingest failed error_id=%s: %s",
                error_id,
                exc,
                exc_info=True,
            )
            return self._response(
                {"status": "error", "reference": error_id}
            )


_DEFAULT_TRANSPORT = TelegramWebhookTransport()


def configure_telegram_webhook_transport(**kwargs: Any) -> TelegramWebhookTransport:
    _DEFAULT_TRANSPORT.configure(**kwargs)
    return _DEFAULT_TRANSPORT


async def _process_telegram_webhook_request(
    req: Request,
    path_secret_token: str | None = None,
) -> Response:
    return await _DEFAULT_TRANSPORT.process(req, path_secret_token)


async def telegram_webhook_ingest(secret_token: str, req: Request) -> Response:
    return await _process_telegram_webhook_request(req, secret_token)


async def telegram_webhook(req: Request) -> Response:
    return await _process_telegram_webhook_request(req, None)


__all__ = [
    "TelegramWebhookTransport",
    "_process_telegram_webhook_request",
    "_read_limited_webhook_body",
    "_telegram_webhook_update_claim",
    "_telegram_webhook_update_complete",
    "_telegram_webhook_update_release",
    "configure_telegram_webhook_transport",
    "telegram_webhook",
    "telegram_webhook_ingest",
]
