"""Production handlers for durable Telegram bot workloads.

Payloads contain Telegram file IDs, chat IDs, text, and other durable values.
They never depend on request-local upload objects or temporary paths created by
the process that enqueued the job.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from telegram import Bot

from app.services.jobs.queue import JobContext, JobHandler


class BotJobPayloadError(ValueError):
    """Raised when a durable workload payload is invalid."""


def _required_int(payload: Mapping[str, Any], name: str) -> int:
    try:
        value = int(payload.get(name) or 0)
    except (TypeError, ValueError) as exc:
        raise BotJobPayloadError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise BotJobPayloadError(f"{name} is required.")
    return value


def _required_text(
    payload: Mapping[str, Any],
    name: str,
    *,
    max_chars: int,
) -> str:
    value = str(payload.get(name) or "").strip()
    if not value:
        raise BotJobPayloadError(f"{name} is required.")
    if len(value) > max_chars:
        raise BotJobPayloadError(f"{name} exceeds {max_chars} characters.")
    return value


def _optional_reply_id(payload: Mapping[str, Any]) -> int | None:
    raw = payload.get("reply_to_message_id")
    if raw in (None, "", 0):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BotJobPayloadError("reply_to_message_id must be an integer.") from exc
    return value if value > 0 else None


class BotJobHandlers:
    """Own a lightweight Telegram Bot client and durable workload handlers."""

    def __init__(self, legacy: Any) -> None:
        self.legacy = legacy
        self._bot: Bot | None = None
        self._bot_lock = asyncio.Lock()

    async def bot(self) -> Bot:
        existing_app = getattr(self.legacy, "_TELEGRAM_APP", None)
        existing_bot = getattr(existing_app, "bot", None)
        if existing_bot is not None:
            return existing_bot
        if self._bot is not None:
            return self._bot
        async with self._bot_lock:
            if self._bot is not None:
                return self._bot
            token = str(
                getattr(self.legacy, "TELEGRAM_BOT_TOKEN", "")
                or getattr(getattr(self.legacy, "SETTINGS", None), "TELEGRAM_BOT_TOKEN", "")
                or ""
            ).strip()
            if not token:
                raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
            client = Bot(token=token)
            await client.initialize()
            self._bot = client
            return client

    async def close(self) -> None:
        async with self._bot_lock:
            client, self._bot = self._bot, None
        if client is not None:
            with suppress(Exception):
                await client.shutdown()

    async def _download(
        self,
        file_id: str,
        *,
        suffix: str,
        max_bytes: int,
    ) -> str:
        bot = await self.bot()
        telegram_file = await bot.get_file(file_id)
        helper = getattr(self.legacy, "_download_telegram_file_to_temp_path", None)
        if callable(helper):
            return await helper(telegram_file, max_bytes, suffix=suffix)

        fd, path = tempfile.mkstemp(prefix="durable-job-", suffix=suffix)
        os.close(fd)
        try:
            await telegram_file.download_to_drive(custom_path=path)
            size = await asyncio.to_thread(os.path.getsize, path)
            if size > max_bytes:
                raise BotJobPayloadError(
                    f"Telegram file exceeds the {max_bytes}-byte limit."
                )
            return path
        except BaseException:
            with suppress(OSError):
                os.unlink(path)
            raise

    async def _notify_terminal_error(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
        error: BaseException,
    ) -> None:
        if context.job.attempts < context.job.max_attempts:
            return
        chat_id = payload.get("chat_id")
        try:
            target = int(chat_id or 0)
        except (TypeError, ValueError):
            return
        if target <= 0:
            return
        bot = await self.bot()
        message = (
            "❌ The background task could not be completed after retries. "
            f"Job: {context.job.id[:12]} · {type(error).__name__}."
        )
        with suppress(Exception):
            await bot.send_message(chat_id=target, text=message)

    async def tts(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
        *,
        force_model: str = "",
    ) -> dict[str, Any]:
        chat_id = _required_int(payload, "chat_id")
        user_id = int(payload.get("user_id") or chat_id)
        text = _required_text(payload, "text", max_chars=20_000)
        gender = str(payload.get("gender") or "female").strip().lower()
        if gender not in {"female", "male"}:
            raise BotJobPayloadError("gender must be female or male.")
        try:
            speed = float(payload.get("speed") or 1.0)
        except (TypeError, ValueError) as exc:
            raise BotJobPayloadError("speed must be numeric.") from exc
        speed = max(0.5, min(2.0, speed))
        model = str(force_model or payload.get("tts_model") or "auto").strip().lower()
        reply_to = _optional_reply_id(payload)
        bot = await self.bot()
        fd, output_path = tempfile.mkstemp(prefix="durable-tts-", suffix=".ogg")
        os.close(fd)
        try:
            audio = await self.legacy.generate_user_voice_limited(
                text,
                gender,
                speed,
                output_path,
                model,
                user_id=user_id,
                bot=bot,
                chat_id=chat_id,
            )
            if await context.cancelled():
                return {"cancelled": True, "bytes": len(audio or b"")}
            kwargs: dict[str, Any] = {"chat_id": chat_id}
            if reply_to is not None:
                kwargs["reply_to_message_id"] = reply_to
            with open(output_path, "rb") as handle:
                sent = await bot.send_voice(voice=handle, **kwargs)
            return {
                "chat_id": chat_id,
                "message_id": int(getattr(sent, "message_id", 0) or 0),
                "bytes": len(audio or b""),
                "model": model,
            }
        except Exception as exc:
            await self._notify_terminal_error(payload, context, exc)
            raise
        finally:
            with suppress(OSError):
                os.unlink(output_path)

    async def tts_job(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
    ) -> dict[str, Any]:
        return await self.tts(payload, context)

    async def voxcpm2_job(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
    ) -> dict[str, Any]:
        return await self.tts(payload, context, force_model="voxcpm2")

    async def ocr_job(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
    ) -> dict[str, Any]:
        chat_id = _required_int(payload, "chat_id")
        file_id = _required_text(payload, "file_id", max_chars=512)
        mime_type = str(payload.get("mime_type") or "image/jpeg").strip().lower()
        suffix = str(payload.get("suffix") or ".jpg").strip()
        if not suffix.startswith(".") or len(suffix) > 10:
            suffix = ".jpg"
        max_bytes = int(getattr(self.legacy, "MAX_IMAGE_FILE_BYTES", 20_000_000))
        path = await self._download(
            file_id,
            suffix=suffix,
            max_bytes=max_bytes,
        )
        try:
            text = await self.legacy.ocr_image(path, mime_type)
            if await context.cancelled():
                return {"cancelled": True}
            bot = await self.bot()
            sent = await bot.send_message(
                chat_id=chat_id,
                text=text or "No text was detected.",
                reply_to_message_id=_optional_reply_id(payload),
            )
            return {
                "chat_id": chat_id,
                "message_id": int(getattr(sent, "message_id", 0) or 0),
                "characters": len(text or ""),
            }
        except Exception as exc:
            await self._notify_terminal_error(payload, context, exc)
            raise
        finally:
            with suppress(OSError):
                os.unlink(path)

    async def transcription_job(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
    ) -> dict[str, Any]:
        chat_id = _required_int(payload, "chat_id")
        file_id = _required_text(payload, "file_id", max_chars=512)
        mime_type = str(payload.get("mime_type") or "audio/ogg").strip().lower()
        suffix = str(payload.get("suffix") or ".ogg").strip()
        if not suffix.startswith(".") or len(suffix) > 10:
            suffix = ".ogg"
        max_bytes = int(getattr(self.legacy, "MAX_AUDIO_FILE_BYTES", 50_000_000))
        path = await self._download(
            file_id,
            suffix=suffix,
            max_bytes=max_bytes,
        )
        try:
            if mime_type == "audio/ogg":
                text = await self.legacy.transcribe_voice(path)
            else:
                text = await self.legacy.transcribe_audio_file(path, mime_type)
            if await context.cancelled():
                return {"cancelled": True}
            bot = await self.bot()
            sent = await bot.send_message(
                chat_id=chat_id,
                text=text or "No speech was detected.",
                reply_to_message_id=_optional_reply_id(payload),
            )
            return {
                "chat_id": chat_id,
                "message_id": int(getattr(sent, "message_id", 0) or 0),
                "characters": len(text or ""),
            }
        except Exception as exc:
            await self._notify_terminal_error(payload, context, exc)
            raise
        finally:
            with suppress(OSError):
                os.unlink(path)

    async def broadcast_job(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
    ) -> dict[str, Any]:
        raw_recipients = payload.get("recipient_ids")
        if not isinstance(raw_recipients, list) or not raw_recipients:
            raise BotJobPayloadError("recipient_ids must be a non-empty list.")
        if len(raw_recipients) > 10_000:
            raise BotJobPayloadError("recipient_ids exceeds 10,000 entries.")
        recipients: list[int] = []
        for value in raw_recipients:
            try:
                recipient = int(value)
            except (TypeError, ValueError) as exc:
                raise BotJobPayloadError("recipient_ids contains an invalid ID.") from exc
            if recipient > 0:
                recipients.append(recipient)
        text = _required_text(payload, "text", max_chars=4096)
        parse_mode = str(payload.get("parse_mode") or "auto").strip().lower()
        photo_file_id = str(payload.get("photo_file_id") or "").strip() or None
        link_preview = bool(payload.get("link_preview", True))
        bot = await self.bot()
        sender = getattr(self.legacy, "_send_telegram_broadcast_message")
        concurrency = max(1, min(10, int(payload.get("concurrency") or 3)))
        semaphore = asyncio.Semaphore(concurrency)
        sent = failed = 0

        async def send_one(chat_id: int) -> bool:
            if await context.cancelled():
                return False
            async with semaphore:
                await sender(
                    bot,
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    photo_file_id=photo_file_id,
                    link_preview=link_preview,
                )
                return True

        for start in range(0, len(recipients), 100):
            if await context.cancelled():
                break
            batch = recipients[start : start + 100]
            results = await asyncio.gather(
                *(send_one(chat_id) for chat_id in batch),
                return_exceptions=True,
            )
            for result in results:
                if result is True:
                    sent += 1
                else:
                    failed += 1
        return {
            "recipients": len(recipients),
            "sent": sent,
            "failed": failed,
            "cancelled": await context.cancelled(),
        }

    def mapping(self) -> dict[str, JobHandler]:
        return {
            "tts": self.tts_job,
            "voxcpm2": self.voxcpm2_job,
            "ocr": self.ocr_job,
            "transcription": self.transcription_job,
            "broadcast": self.broadcast_job,
        }


def build_bot_job_handlers(legacy: Any) -> BotJobHandlers:
    return BotJobHandlers(legacy)


__all__ = [
    "BotJobHandlers",
    "BotJobPayloadError",
    "build_bot_job_handlers",
]
