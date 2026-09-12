"""Modern Telegram Update Dispatcher for Webhook and Background Ingestion.

Features:
- Timing-safe HMAC webhook secret verification (checked first to prevent info leaks).
- Size-capped streaming payload ingestion (protects against OOM).
- Non-blocking asynchronous update dispatch (eliminates Telegram webhook retry storms).
- Bounded concurrency with asyncio.Semaphore and backpressure queue depth limits.
- Per-chat sequential ordering locks (prevents race conditions from rapid multi-message bursts).
- Atomic update deduplication and lease management with CancelledError recovery.
- Active in-flight worker tracking with cancellation-safe lease release on drain.
- Comprehensive telemetry counters and execution latency metrics.
- Graceful shutdown drain with timeout protection.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from telegram import Update

from app.services.telegram.deduplication import (
    _telegram_webhook_update_claim,
    _telegram_webhook_update_complete,
    _telegram_webhook_update_release,
)

logger = logging.getLogger("app.dispatcher")


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


class TelegramDispatcher:
    """Core update ingestion, deduplication, concurrency control, and worker dispatch engine."""

    def __init__(self) -> None:
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._updates_received = 0
        self._updates_dispatched = 0
        self._updates_completed = 0
        self._updates_failed = 0
        self._replays_dropped = 0
        self._standby_dropped = 0
        self._queue_rejections = 0
        self._latency_sum_s = 0.0
        self._latency_samples = 0

        # Concurrency & Backpressure Bounds
        self._concurrency_limit = _env_int("DISPATCHER_MAX_CONCURRENCY", 32, minimum=4, maximum=256)
        self._max_queue_depth = _env_int("DISPATCHER_MAX_QUEUE_DEPTH", 100, minimum=10, maximum=1000)
        self._semaphore = asyncio.Semaphore(self._concurrency_limit)

        # Per-chat sequencing locks (preserves FIFO message ordering per chat)
        self._chat_locks: OrderedDict[int, asyncio.Lock] = OrderedDict()
        self._chat_locks_guard = threading.Lock()

        # Operational Mode
        self._async_dispatch = os.environ.get("DISPATCHER_ASYNC", "true").lower() in ("1", "true", "yes")

    @property
    def async_dispatch(self) -> bool:
        return self._async_dispatch

    @async_dispatch.setter
    def async_dispatch(self, enabled: bool) -> None:
        self._async_dispatch = bool(enabled)

    def _get_expected_secret(self) -> str:
        """Resolve active webhook secret token from environment or runtime state."""
        try:
            from app import legacy

            if hasattr(legacy, "_runtime_webhook_secret_token"):
                token = legacy._runtime_webhook_secret_token()
                if token:
                    return str(token).strip()
        except Exception as exc:
            logger.debug("Failed resolving secret token via legacy runtime: %s", exc)

        return (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or "").strip()

    def _get_app_instance(self) -> Any:
        """Resolve current telegram Application instance."""
        try:
            from app.bot import get_global_telegram_app

            app = get_global_telegram_app()
            if app is not None:
                return app
        except Exception as exc:
            logger.debug("Failed resolving app via get_global_telegram_app: %s", exc)

        try:
            from app import legacy

            return getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
        except Exception as exc:
            logger.debug("Failed resolving app via legacy: %s", exc)
            return None

    def _is_app_ready(self) -> bool:
        """Check if telegram application has completed startup."""
        try:
            from app import legacy

            return bool(getattr(legacy, "_TELEGRAM_APP_READY", True))
        except Exception as exc:
            logger.debug("Failed checking app ready via legacy: %s", exc)
            return True

    def _should_process_update(self) -> bool:
        """Verify this process is the active cluster leader (not a standby node)."""
        try:
            from app import legacy

            if hasattr(legacy, "_telegram_should_process_webhook_update"):
                return bool(legacy._telegram_should_process_webhook_update())
        except Exception as exc:
            logger.debug("Failed checking leader status via legacy: %s", exc)
        return True

    def _is_webhook_mode(self) -> bool:
        """Verify bot is configured to ingest via Webhook."""
        try:
            from app import legacy

            if hasattr(legacy, "_run_state_bot_mode"):
                return legacy._run_state_bot_mode() == "WEBHOOK"
        except Exception as exc:
            logger.debug("Failed checking bot mode via legacy: %s", exc)
        return os.environ.get("BOT_MODE", "POLLING").upper() == "WEBHOOK"

    @staticmethod
    def _extract_chat_id(update: Update) -> int | None:
        """Extract chat_id from effective_chat, message, callback_query, or channel_post."""
        with suppress(Exception):
            if update.effective_chat and update.effective_chat.id is not None:
                return int(update.effective_chat.id)
            if update.message and update.message.chat_id is not None:
                return int(update.message.chat_id)
            if update.callback_query and update.callback_query.message and update.callback_query.message.chat_id is not None:
                return int(update.callback_query.message.chat_id)
            if update.channel_post and update.channel_post.chat_id is not None:
                return int(update.channel_post.chat_id)
        return None

    def _get_chat_lock(self, chat_id: int) -> asyncio.Lock:
        """Bounded LRU per-chat lock to preserve sequential message ordering per chat."""
        with self._chat_locks_guard:
            lock = self._chat_locks.get(chat_id)
            if lock is not None:
                self._chat_locks.move_to_end(chat_id)
                return lock

            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock
            if len(self._chat_locks) > 2000:
                to_evict = []
                for cid, clk in self._chat_locks.items():
                    if not clk.locked():
                        to_evict.append(cid)
                    if len(to_evict) >= 200:
                        break
                for cid in to_evict:
                    self._chat_locks.pop(cid, None)
            return lock

    async def _read_limited_body(self, req: Request, max_body: int = 2 * 1024 * 1024) -> bytes:
        """Read incoming request body with strict length validation."""
        content_length = req.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_body:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Webhook payload too large. Maximum {max_body} bytes.",
                    )
            except ValueError:
                pass

        chunks: list[bytes] = []
        received = 0
        async for chunk in req.stream():
            received += len(chunk)
            if received > max_body:
                raise HTTPException(
                    status_code=413,
                    detail=f"Webhook payload stream exceeded {max_body} bytes.",
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _record_completion(self, latency_s: float, *, success: bool) -> None:
        """Record update completion latency and outcome counters."""
        if success:
            self._updates_completed += 1
        else:
            self._updates_failed += 1

        self._latency_sum_s += latency_s
        self._latency_samples += 1
        # Half-life decay approximation to keep rolling average responsive over long uptimes
        if self._latency_samples > 100_000:
            self._latency_sum_s = self._latency_sum_s / 2
            self._latency_samples = self._latency_samples // 2

    async def _execute_update_handler(
        self,
        app_obj: Any,
        update: Update,
        update_id: int | None,
        claim_token: str | None,
        chat_id: int | None = None,
    ) -> None:
        """Execute the handler pipeline for an update with per-chat ordering and lease protection."""
        t0 = time.monotonic()
        try:
            if chat_id is not None:
                chat_lock = self._get_chat_lock(chat_id)
                async with chat_lock:
                    await app_obj.process_update(update)
            else:
                await app_obj.process_update(update)

            if update_id is not None:
                await _telegram_webhook_update_complete(
                    update_id,
                    claim_token=claim_token,
                )
            self._record_completion(time.monotonic() - t0, success=True)
        except asyncio.CancelledError:
            # Handle task cancellation during server shutdown drain cleanly.
            # BaseException in Python 3.8+ must be caught explicitly to release dedup lease.
            self._record_completion(time.monotonic() - t0, success=False)
            if update_id is not None and claim_token is not None:
                with suppress(Exception):
                    await _telegram_webhook_update_release(
                        update_id,
                        claim_token=claim_token,
                    )
            logger.warning("Update handler cancelled for update_id=%s (dedup lease released)", update_id)
            raise
        except Exception as exc:
            self._record_completion(time.monotonic() - t0, success=False)
            if update_id is not None and claim_token is not None:
                with suppress(Exception):
                    await _telegram_webhook_update_release(
                        update_id,
                        claim_token=claim_token,
                    )
            logger.error(
                "Telegram update processing error update_id=%s: %s",
                update_id,
                exc,
                exc_info=True,
            )

    async def dispatch_webhook_request(
        self,
        req: Request,
        path_secret_token: str | None = None,
    ) -> JSONResponse:
        """Validate, authenticate, deduplicate, and dispatch a Telegram webhook request."""
        self._updates_received += 1

        # 1. Timing-Safe Secret Token Verification FIRST (Prevents mode/server info leakage)
        expected_secret = self._get_expected_secret()
        if not expected_secret:
            logger.error("Telegram webhook rejected: TELEGRAM_WEBHOOK_SECRET_TOKEN is unconfigured.")
            raise HTTPException(
                status_code=503,
                detail="Telegram webhook secret token is not configured on this server.",
            )

        header_secret = (req.headers.get("X-Telegram-Bot-Api-Secret-Token") or "").strip()
        path_valid = (
            path_secret_token is not None
            and hmac.compare_digest(str(path_secret_token), expected_secret)
        )
        header_valid = (
            bool(header_secret)
            and hmac.compare_digest(header_secret, expected_secret)
        )

        if not (path_valid or header_valid):
            client_ip = req.client.host if req.client else "unknown"
            logger.warning(
                "Rejected webhook with invalid secret token (path_valid=%s, header_valid=%s) from %s",
                path_valid,
                header_valid,
                client_ip,
            )
            raise HTTPException(status_code=403, detail="Invalid webhook secret token.")

        # 2. Mode Validation (Now safe after authentication)
        if not self._is_webhook_mode():
            logger.info("Webhook update ignored: BOT_MODE is not set to WEBHOOK.")
            return JSONResponse(
                {"status": "ignored", "reason": "not_webhook_mode"},
                status_code=200,
            )

        # 3. Application Lifecycle Readiness
        app_obj = self._get_app_instance()
        if app_obj is None or not self._is_app_ready():
            logger.warning("Webhook 503: Telegram Application is not ready yet.")
            raise HTTPException(
                status_code=503,
                detail="Telegram Application is initializing. Please retry in a few moments.",
            )

        # 4. Standby Cluster Coordination
        if not self._should_process_update():
            self._standby_dropped += 1
            logger.info("Webhook update acknowledged on standby instance without execution.")
            return JSONResponse(
                {"status": "ignored", "reason": "standby_instance"},
                status_code=200,
            )

        # 5. Payload Ingestion & De-serialization
        try:
            raw_body = await self._read_limited_body(req)
            data = json.loads(raw_body)
            if not isinstance(data, dict):
                raise ValueError("Payload root must be a JSON object.")
            bot = getattr(app_obj, "bot", None)
            try:
                update = Update.de_json(data, bot)
            except TypeError:
                update = Update.de_json(data, None)
        except HTTPException:
            raise
        except Exception as exc:
            err_ref = secrets.token_hex(6)
            logger.warning("Invalid webhook payload ignored (ref=%s): %s", err_ref, exc)
            return JSONResponse(
                {
                    "status": "ignored",
                    "reason": "invalid_payload",
                    "reference": err_ref,
                },
                status_code=200,
            )

        # 6. Deduplication & Replay Lease
        update_id = getattr(update, "update_id", None)
        claim_token: str | None = None
        if update_id is not None:
            claim_state, claim_token = await _telegram_webhook_update_claim(
                update_id,
                include_token=True,
            )
            if claim_state == "completed":
                self._replays_dropped += 1
                logger.info("Replay dropped: Telegram update_id=%s already completed.", update_id)
                return JSONResponse(
                    {"status": "ok", "duplicate": True},
                    status_code=200,
                )
            if claim_state == "processing":
                logger.info("Update update_id=%s already processing by another lease.", update_id)
                response = JSONResponse(
                    {"status": "retry", "reason": "already_processing"},
                    status_code=503,
                )
                response.headers["Retry-After"] = "2"
                return response

        # 7. Concurrency Bound & Backpressure Queue Check
        if len(self._active_tasks) >= (self._concurrency_limit + self._max_queue_depth):
            self._queue_rejections += 1
            logger.warning(
                "Dispatcher queue saturated (active=%s, max_concurrency=%s, max_queue=%s). Shedding load.",
                len(self._active_tasks),
                self._concurrency_limit,
                self._max_queue_depth,
            )
            if update_id is not None and claim_token is not None:
                await _telegram_webhook_update_release(update_id, claim_token=claim_token)
            response = JSONResponse(
                {"status": "retry", "reason": "queue_saturated"},
                status_code=503,
            )
            response.headers["Retry-After"] = "2"
            return response

        self._updates_dispatched += 1
        chat_id = self._extract_chat_id(update)

        # 8. Adaptive Dispatch Execution (Async bounded worker or sync fallback)
        if self._async_dispatch:
            async def _bounded_worker() -> None:
                async with self._semaphore:
                    await self._execute_update_handler(
                        app_obj, update, update_id, claim_token, chat_id=chat_id
                    )

            worker_task = asyncio.create_task(
                _bounded_worker(),
                name=f"tg-update-{update_id}",
            )
            self._active_tasks.add(worker_task)
            worker_task.add_done_callback(self._active_tasks.discard)
            return JSONResponse({"status": "ok", "dispatched": True}, status_code=200)

        # Synchronous inline execution (fallback mode):
        async with self._semaphore:
            await self._execute_update_handler(
                app_obj, update, update_id, claim_token, chat_id=chat_id
            )
        return JSONResponse({"status": "ok"}, status_code=200)

    async def drain(self, timeout: float = 10.0) -> None:  # noqa: ASYNC109
        """Gracefully wait for all in-flight update handlers to finish before shutdown.

        If timeout expires, tasks are cancelled and awaited so their CancelledError
        handler safely releases deduplication leases before process exit.
        """
        if not self._active_tasks:
            return

        logger.info("Draining %s active update worker tasks (timeout=%ss)...", len(self._active_tasks), timeout)
        start_time = time.monotonic()
        pending = set(self._active_tasks)
        try:
            done, pending = await asyncio.wait(pending, timeout=timeout)
            logger.info("Drained %s tasks in %.2fs (%s remaining).", len(done), time.monotonic() - start_time, len(pending))
        except Exception as exc:
            logger.warning("Error during initial drain wait: %s", exc)

        for task in pending:
            if not task.done():
                task.cancel()

        if pending:
            # Await cancelled tasks so their CancelledError exception handler executes and releases dedup leases
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            except Exception as exc:
                logger.debug("Exception while awaiting cancelled drain tasks: %s", exc)

    def get_metrics(self) -> dict[str, Any]:
        """Return real-time dispatcher telemetry, concurrency, and health counters."""
        avg_latency_ms = (
            (self._latency_sum_s / self._latency_samples * 1000.0)
            if self._latency_samples > 0
            else 0.0
        )
        # Approximate current semaphore utilization
        concurrency_in_use = max(0, self._concurrency_limit - self._semaphore._value)
        return {
            "updates_received": self._updates_received,
            "updates_dispatched": self._updates_dispatched,
            "updates_completed": self._updates_completed,
            "updates_failed": self._updates_failed,
            "replays_dropped": self._replays_dropped,
            "standby_dropped": self._standby_dropped,
            "queue_rejections": self._queue_rejections,
            "active_in_flight": len(self._active_tasks),
            "concurrency_limit": self._concurrency_limit,
            "concurrency_in_use": concurrency_in_use,
            "max_queue_depth": self._max_queue_depth,
            "average_latency_ms": round(avg_latency_ms, 2),
            "mode": "async" if self._async_dispatch else "sync",
        }


_GLOBAL_DISPATCHER: TelegramDispatcher | None = None


def get_telegram_dispatcher() -> TelegramDispatcher:
    """Return the global TelegramDispatcher singleton."""
    global _GLOBAL_DISPATCHER
    if _GLOBAL_DISPATCHER is None:
        _GLOBAL_DISPATCHER = TelegramDispatcher()
    return _GLOBAL_DISPATCHER


__all__ = [
    "TelegramDispatcher",
    "get_telegram_dispatcher",
]
