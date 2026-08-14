"""Typed submission helpers for durable bot workloads."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.services.jobs.runtime import enqueue_bot_job


async def submit_tts_job(
    *,
    chat_id: int,
    user_id: int,
    text: str,
    gender: str = "female",
    speed: float = 1.0,
    tts_model: str = "auto",
    progress_message_id: int | None = None,
    reply_to_message_id: int | None = None,
    idempotency_key: str,
):
    model = str(tts_model or "auto").strip().lower().replace("-", "_")
    if model not in {"auto", "hf_space", "edge"}:
        model = "auto"
    return await enqueue_bot_job(
        "tts",
        {
            "chat_id": int(chat_id),
            "user_id": int(user_id),
            "text": str(text),
            "gender": str(gender),
            "speed": float(speed),
            "tts_model": model,
            "progress_message_id": progress_message_id,
            "reply_to_message_id": reply_to_message_id,
        },
        idempotency_key=idempotency_key,
        timeout_seconds=600,
        max_attempts=3,
    )


async def submit_ocr_job(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    file_id: str,
    mime_type: str,
    suffix: str = ".jpg",
    progress_message_id: int | None = None,
    reply_to_message_id: int | None = None,
    idempotency_key: str,
):
    return await enqueue_bot_job(
        "ocr",
        {
            "chat_id": int(chat_id),
            "user_id": int(user_id),
            "username": str(username),
            "file_id": str(file_id),
            "mime_type": str(mime_type),
            "suffix": str(suffix),
            "progress_message_id": progress_message_id,
            "reply_to_message_id": reply_to_message_id,
            "artifact_ttl_seconds": 604_800,
        },
        idempotency_key=idempotency_key,
        timeout_seconds=180,
        max_attempts=3,
    )


async def submit_transcription_job(
    *,
    chat_id: int,
    user_id: int,
    username: str,
    file_id: str,
    mime_type: str,
    suffix: str = ".ogg",
    source_kind: str = "voice",
    filename: str = "",
    progress_message_id: int | None = None,
    reply_to_message_id: int | None = None,
    idempotency_key: str,
):
    clean_source = str(source_kind or "voice").strip().lower()
    if clean_source not in {"voice", "audio_file"}:
        raise ValueError("source_kind must be voice or audio_file.")
    return await enqueue_bot_job(
        "transcription",
        {
            "chat_id": int(chat_id),
            "user_id": int(user_id),
            "username": str(username),
            "file_id": str(file_id),
            "mime_type": str(mime_type),
            "suffix": str(suffix),
            "source_kind": clean_source,
            "filename": str(filename),
            "progress_message_id": progress_message_id,
            "reply_to_message_id": reply_to_message_id,
            "artifact_ttl_seconds": 604_800,
        },
        idempotency_key=idempotency_key,
        timeout_seconds=240,
        max_attempts=3,
    )


async def submit_broadcast_job(
    *,
    recipient_ids: Iterable[int],
    text: str,
    parse_mode: str = "auto",
    photo_file_id: str = "",
    link_preview: bool = True,
    idempotency_key: str,
    extra: dict[str, Any] | None = None,
):
    payload: dict[str, Any] = {
        "recipient_ids": [int(value) for value in recipient_ids],
        "text": str(text),
        "parse_mode": str(parse_mode),
        "photo_file_id": str(photo_file_id),
        "link_preview": bool(link_preview),
    }
    payload.update(extra or {})
    return await enqueue_bot_job(
        "broadcast",
        payload,
        idempotency_key=idempotency_key,
        timeout_seconds=3_600,
        max_attempts=2,
    )


__all__ = [
    "submit_broadcast_job",
    "submit_ocr_job",
    "submit_transcription_job",
    "submit_tts_job",
]
