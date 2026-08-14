"""Production handlers for durable Telegram bot workloads.

Payloads contain Telegram file IDs, chat IDs, text, and other durable values.
They never depend on request-local upload objects or temporary paths created by
the process that enqueued the job.
"""

from __future__ import annotations

import asyncio
import html
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from telegram import Bot

from app.services.ai.ocr import normalize_media_suffix, normalize_ocr_result
from app.services.ai.tts import normalize_tts_request
from app.services.artifacts.storage import ArtifactService
from app.services.jobs.queue import JobContext, JobHandler
from app.services.telegram.broadcast import BroadcastRequest
from app.services.telegram.delivery import IdempotentTelegramDelivery


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

    def __init__(
        self,
        legacy: Any,
        *,
        artifacts: ArtifactService,
        delivery: IdempotentTelegramDelivery,
    ) -> None:
        self.legacy = legacy
        self.artifacts = artifacts
        self.delivery = delivery
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

    @staticmethod
    def _progress_message_id(payload: Mapping[str, Any]) -> int | None:
        raw = payload.get("progress_message_id")
        try:
            value = int(raw or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    async def _notify_terminal_error(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
        error: BaseException,
    ) -> None:
        if context.job.attempts < context.job.max_attempts:
            await self._update_telegram_progress(
                payload,
                "⚠️ ការងារមិនទាន់បានជោគជ័យ។ Bot នឹងសាកល្បងម្ដងទៀតដោយស្វ័យប្រវត្តិ។\n\n"
                f"Job: {context.job.id[:12]} · "
                f"Attempt {context.job.attempts}/{context.job.max_attempts}",
            )
            return
        try:
            target = int(payload.get("chat_id") or 0)
        except (TypeError, ValueError):
            return
        if target <= 0:
            return
        message = (
            "❌ ការងារផ្ទៃក្រោយមិនអាចបញ្ចប់បានបន្ទាប់ពីសាកល្បងឡើងវិញ។ "
            f"Job: {context.job.id[:12]}."
        )
        with suppress(Exception):
            await self.delivery.deliver_text(
                bot=await self.bot(),
                idempotency_key=f"job:{context.job.id}:terminal-error",
                chat_id=target,
                text=message,
                progress_message_id=self._progress_message_id(payload),
                reply_to_message_id=_optional_reply_id(payload),
            )

    def _result_page(
        self,
        text: str,
        *,
        header: str,
    ) -> str:
        """Render the first page, budgeting on the escaped length.

        Telegram counts the characters it receives, which is the escaped text.
        Paginating the raw text first can overflow the limit by up to 6x (a
        message of apostrophes becomes ``&#x27;`` each), so the page is cut to
        fit *after* escaping.
        """

        suffix = (
            "\n\n<i>លទ្ធផលពេញត្រូវបានរក្សាទុកដោយសុវត្ថិភាពសម្រាប់សកម្មភាពបន្ទាប់។</i>"
        )
        message_limit = int(getattr(self.legacy, "TELE_MSG_LIMIT", 4096))
        budget = max(500, message_limit - len(header) - len(suffix) - 32)

        take_prefix = getattr(self.legacy, "_take_escaped_prefix", None)
        if callable(take_prefix):
            first, remainder = take_prefix(str(text or "").strip(), budget)
        else:  # pragma: no cover - the legacy helper is always present
            first, remainder = str(text or "").strip()[:budget], ""
        return header + html.escape(first) + (suffix if remainder else "")

    async def _deliver_text_result(
        self,
        *,
        payload: Mapping[str, Any],
        context: JobContext,
        text: str,
        kind: str,
    ) -> dict[str, Any]:
        artifact = await self.artifacts.put_text(
            job_id=context.job.id,
            name="ocr.txt" if kind == "ocr" else "transcript.txt",
            text=text,
            ttl_seconds=int(payload.get("artifact_ttl_seconds") or 604_800),
        )
        chat_id = _required_int(payload, "chat_id")
        user_id = int(payload.get("user_id") or chat_id)
        username = str(payload.get("username") or user_id)[:128]
        lang_key = self.legacy._detect_lang(text)
        lang_flag, lang_name = self.legacy._language_display(lang_key)
        progress_id = self._progress_message_id(payload)
        result_id = progress_id or int(payload.get("reply_to_message_id") or 0)

        if kind == "ocr":
            header = (
                f"🔍 <b>អត្ថបទពីរូបភាព {lang_flag} "
                f"{html.escape(lang_name)}</b>\n\n"
            )
            markup = self.legacy.get_ocr_confirm_kb(result_id) if result_id else None
            cache_prefix = "[Image OCR]"
        else:
            source_kind = str(payload.get("source_kind") or "voice")
            if source_kind == "audio_file":
                filename = html.escape(str(payload.get("filename") or "audio")[:50])
                header = (
                    f"🎵 <b>អត្ថបទពីឯកសារអូឌីយ៉ូ</b> {lang_flag} "
                    f"{html.escape(lang_name)} — <code>{filename}</code>\n\n"
                )
                markup = self.legacy.get_audio_file_kb(result_id) if result_id else None
                cache_prefix = "[Audio File Transcript]"
            else:
                header = (
                    f"📝 <b>អត្ថបទពីសារសំឡេង</b> {lang_flag} "
                    f"{html.escape(lang_name)}\n\n"
                )
                markup = self.legacy.get_transcription_kb(result_id) if result_id else None
                cache_prefix = "[Voice Transcript]"

        rendered = self._result_page(text, header=header)
        delivered = await self.delivery.deliver_text(
            bot=await self.bot(),
            idempotency_key=f"job:{context.job.id}:telegram-result",
            chat_id=chat_id,
            text=rendered,
            progress_message_id=progress_id,
            reply_to_message_id=_optional_reply_id(payload),
            parse_mode="HTML",
            reply_markup=markup,
        )
        resolved_id = int(delivered.get("message_id") or result_id or 0)
        if resolved_id:
            with suppress(Exception):
                self.legacy.save_text_cache(
                    resolved_id,
                    text,
                    chat_id=chat_id,
                    user_id=user_id,
                    username=username,
                )
        with suppress(Exception):
            await asyncio.to_thread(
                self.legacy.record_turn,
                user_id,
                "user",
                f"{cache_prefix}: {text[:500]}",
            )
        return {
            "artifact": artifact.as_dict(),
            "delivery": delivered,
            "characters": len(text),
            "language": lang_key,
        }

    async def _progress(
        self,
        context: JobContext,
        percent: int,
        stage: str,
        detail: str = "",
    ) -> None:
        with suppress(Exception):
            await context.progress(percent, stage, detail)

    async def _update_telegram_progress(
        self,
        payload: Mapping[str, Any],
        text: str,
    ) -> None:
        """Update the request's original progress message when available."""

        message_id = self._progress_message_id(payload)
        if message_id is None:
            return
        try:
            chat_id = _required_int(payload, "chat_id")
            bot = await self.bot()
            edit = getattr(bot, "edit_message_text", None)
            if callable(edit):
                await edit(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=str(text),
                )
        except Exception:
            # Telegram progress is best-effort and must never fail the audio job.
            return

    async def tts(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
    ) -> dict[str, Any]:
        chat_id = _required_int(payload, "chat_id")
        user_id = int(payload.get("user_id") or chat_id)
        try:
            request = normalize_tts_request(payload)
        except ValueError as exc:
            raise BotJobPayloadError(str(exc)) from exc
        reply_to = _optional_reply_id(payload)
        await self._progress(context, 10, "preparing", f"model={request.model}")
        await self._update_telegram_progress(
            payload,
            "⏳ កំពុងបម្លែងអត្ថបទទៅជាសំឡេង…\n\n"
            "██░░░░░░░░░░░░░░░░░░  10%\n"
            "📌 Worker បានទទួលការងារ និងកំពុងរៀបចំ",
        )
        bot = await self.bot()
        fd, output_path = tempfile.mkstemp(prefix="durable-tts-", suffix=".ogg")
        os.close(fd)
        try:
            await self._progress(context, 30, "generating_voice")
            await self._update_telegram_progress(
                payload,
                "⏳ កំពុងបម្លែងអត្ថបទទៅជាសំឡេង…\n\n"
                "██████░░░░░░░░░░░░░░  30%\n"
                "📌 កំពុងបង្កើតសំឡេង",
            )
            audio = await self.legacy.generate_user_voice_limited(
                request.text,
                request.gender,
                request.speed,
                output_path,
                request.model,
                user_id=user_id,
                bot=bot,
                chat_id=chat_id,
            )
            if await context.cancelled():
                return {"cancelled": True, "bytes": len(audio or b"")}
            await self._progress(context, 85, "sending_voice")
            await self._update_telegram_progress(
                payload,
                "⏳ កំពុងបម្លែងអត្ថបទទៅជាសំឡេង…\n\n"
                "█████████████████░░░  85%\n"
                "📌 កំពុងផ្ញើសំឡេងទៅ Telegram",
            )
            kwargs: dict[str, Any] = {"chat_id": chat_id}
            if reply_to is not None:
                kwargs["reply_to_message_id"] = reply_to
            handle = await asyncio.to_thread(open, output_path, "rb")
            try:
                bot_tag = str(getattr(self.legacy, "BOT_TAG", "") or "").strip()
                keyboard_factory = getattr(self.legacy, "get_main_kb", None)
                reply_markup = (
                    keyboard_factory(request.gender, request.model)
                    if callable(keyboard_factory)
                    else None
                )
                delivered = await self.delivery.deliver_voice(
                    bot=bot,
                    idempotency_key=f"job:{context.job.id}:telegram-voice",
                    voice=handle,
                    caption=f"🗣️ {bot_tag}".strip(),
                    reply_markup=reply_markup,
                    **kwargs,
                )
            finally:
                await asyncio.to_thread(handle.close)
            await self._update_telegram_progress(
                payload,
                "✅ បានបម្លែង និងផ្ញើសំឡេងដោយជោគជ័យ។",
            )
            delivered_message_id = int(delivered.get("message_id") or 0)
            username = str(payload.get("username") or user_id)[:128]
            if delivered_message_id:
                with suppress(Exception):
                    self.legacy.save_text_cache(
                        delivered_message_id,
                        request.text,
                        chat_id=chat_id,
                        user_id=user_id,
                        username=username,
                    )

            def _record_tts_success() -> None:
                original_text = str(payload.get("original_text") or request.text)
                self.legacy.record_turn(user_id, "user", original_text)
                self.legacy.record_turn(
                    user_id,
                    "assistant",
                    request.text[
                        : int(getattr(self.legacy, "CONV_CONTEXT_MAX_CHARS", 6000))
                    ],
                )

            with suppress(Exception):
                await asyncio.to_thread(_record_tts_success)
            set_last_tts = getattr(self.legacy, "_set_last_tts", None)
            if callable(set_last_tts):
                with suppress(Exception):
                    set_last_tts(user_id)
            return {
                "chat_id": chat_id,
                "message_id": int(delivered.get("message_id") or 0),
                "bytes": len(audio or b""),
                "model": request.model,
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

    async def ocr_job(
        self,
        payload: Mapping[str, Any],
        context: JobContext,
    ) -> dict[str, Any]:
        file_id = _required_text(payload, "file_id", max_chars=512)
        mime_type = str(payload.get("mime_type") or "image/jpeg").strip().lower()
        suffix = normalize_media_suffix(str(payload.get("suffix") or ""), default=".jpg")
        max_bytes = int(getattr(self.legacy, "MAX_IMAGE_FILE_BYTES", 20_000_000))
        await self._progress(context, 10, "downloading_image")
        path = await self._download(file_id, suffix=suffix, max_bytes=max_bytes)
        try:
            if await context.cancelled():
                return {"cancelled": True}
            await self._progress(context, 45, "recognizing_text")
            text = await self.legacy.ocr_image(path, mime_type)
            if await context.cancelled():
                return {"cancelled": True}
            text = normalize_ocr_result(
                text,
                no_text_message="រូបភាពនេះមិនមានអត្ថបទដែលអាចអានបានទេ។",
            )
            await self._progress(context, 80, "storing_result")
            result = await self._deliver_text_result(
                payload=payload,
                context=context,
                text=str(text),
                kind="ocr",
            )
            await self._progress(context, 100, "completed")
            return result
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
        file_id = _required_text(payload, "file_id", max_chars=512)
        mime_type = str(payload.get("mime_type") or "audio/ogg").strip().lower()
        suffix = normalize_media_suffix(str(payload.get("suffix") or ""), default=".ogg")
        max_bytes = int(getattr(self.legacy, "MAX_AUDIO_FILE_BYTES", 50_000_000))
        await self._progress(context, 10, "downloading_audio")
        path = await self._download(file_id, suffix=suffix, max_bytes=max_bytes)
        try:
            if await context.cancelled():
                return {"cancelled": True}
            await self._progress(context, 45, "transcribing_audio")
            if mime_type == "audio/ogg":
                text = await self.legacy.transcribe_voice(path)
            else:
                text = await self.legacy.transcribe_audio_file(path, mime_type)
            if await context.cancelled():
                return {"cancelled": True}
            if not text:
                text = "មិនអាចស្គាល់អត្ថបទនៅក្នុងសំឡេងនេះបានទេ។"
            await self._progress(context, 80, "storing_result")
            result = await self._deliver_text_result(
                payload=payload,
                context=context,
                text=str(text),
                kind="transcription",
            )
            await self._progress(context, 100, "completed")
            return result
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
        try:
            request = BroadcastRequest.from_payload(payload)
        except ValueError as exc:
            raise BotJobPayloadError(str(exc)) from exc
        bot = await self.bot()
        sender = self.legacy._send_telegram_broadcast_message
        semaphore = asyncio.Semaphore(request.concurrency)
        sent = failed = 0
        await self._progress(context, 5, "broadcast_starting")

        async def send_one(chat_id: int) -> bool:
            if await context.cancelled():
                return False
            async with semaphore:
                await sender(
                    bot,
                    chat_id=chat_id,
                    text=request.text,
                    parse_mode=request.parse_mode,
                    photo_file_id=request.photo_file_id,
                    link_preview=request.link_preview,
                )
                return True

        for start in range(0, len(request.recipients), 100):
            if await context.cancelled():
                break
            batch = request.recipients[start : start + 100]
            results = await asyncio.gather(
                *(send_one(chat_id) for chat_id in batch),
                return_exceptions=True,
            )
            for result in results:
                if result is True:
                    sent += 1
                else:
                    failed += 1
            processed = min(start + len(batch), len(request.recipients))
            percent = 5 + int((processed / len(request.recipients)) * 90)
            await self._progress(
                context,
                percent,
                "broadcasting",
                f"processed={processed} sent={sent} failed={failed}",
            )
        return {
            "recipients": len(request.recipients),
            "sent": sent,
            "failed": failed,
            "cancelled": await context.cancelled(),
        }

    def mapping(self) -> dict[str, JobHandler]:
        return {
            "tts": self.tts_job,
            "ocr": self.ocr_job,
            "transcription": self.transcription_job,
            "broadcast": self.broadcast_job,
        }


def build_bot_job_handlers(
    legacy: Any,
    *,
    artifacts: ArtifactService,
    delivery: IdempotentTelegramDelivery,
) -> BotJobHandlers:
    return BotJobHandlers(legacy, artifacts=artifacts, delivery=delivery)


__all__ = [
    "BotJobHandlers",
    "BotJobPayloadError",
    "build_bot_job_handlers",
]
