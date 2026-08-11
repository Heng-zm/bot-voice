"""Framework-independent validation for durable Telegram broadcasts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def normalize_parse_mode(
    value: str | None,
    *,
    aliases: Mapping[str, str],
    default: str = "auto",
) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in aliases:
        return str(aliases[key])
    default_key = str(default or "").strip().lower()
    return str(aliases.get(default_key, "auto"))


def normalize_link_preview(value: Any, *, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "preview", "link_preview", "url_preview"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "nopreview", "no_preview", "no-preview"}:
        return False
    return bool(default)


@dataclass(frozen=True, slots=True)
class BroadcastRequest:
    recipients: tuple[int, ...]
    text: str
    parse_mode: str
    photo_file_id: str | None
    link_preview: bool
    concurrency: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BroadcastRequest:
        raw_recipients = payload.get("recipient_ids")
        if not isinstance(raw_recipients, list) or not raw_recipients:
            raise ValueError("recipient_ids must be a non-empty list.")
        if len(raw_recipients) > 10_000:
            raise ValueError("recipient_ids exceeds 10,000 entries.")
        recipients: list[int] = []
        seen: set[int] = set()
        for value in raw_recipients:
            try:
                recipient = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("recipient_ids contains an invalid ID.") from exc
            if recipient > 0 and recipient not in seen:
                recipients.append(recipient)
                seen.add(recipient)
        if not recipients:
            raise ValueError("recipient_ids contains no valid positive IDs.")
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("text is required.")
        if len(text) > 4_096:
            raise ValueError("text exceeds 4096 characters.")
        try:
            concurrency = int(payload.get("concurrency") or 3)
        except (TypeError, ValueError) as exc:
            raise ValueError("concurrency must be an integer.") from exc
        return cls(
            recipients=tuple(recipients),
            text=text,
            parse_mode=str(payload.get("parse_mode") or "auto").strip().lower(),
            photo_file_id=str(payload.get("photo_file_id") or "").strip() or None,
            link_preview=normalize_link_preview(
                payload.get("link_preview"),
                default=True,
            ),
            concurrency=max(1, min(10, concurrency)),
        )


__all__ = ["BroadcastRequest", "normalize_link_preview", "normalize_parse_mode"]
