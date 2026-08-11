"""Speech generation and Telegram voice conversion helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app._legacy_bridge import exported_dir, exported_getattr


@dataclass(frozen=True, slots=True)
class TTSRequest:
    """Validated speech-generation values independent of bot framework state."""

    text: str
    gender: str
    speed: float
    model: str


def normalize_tts_request(
    payload: Mapping[str, Any],
    *,
    max_chars: int = 20_000,
) -> TTSRequest:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("text is required.")
    if len(text) > max(1, int(max_chars)):
        raise ValueError(f"text exceeds {max_chars} characters.")
    gender = str(payload.get("gender") or "female").strip().lower()
    if gender not in {"female", "male"}:
        raise ValueError("gender must be female or male.")
    try:
        speed = float(payload.get("speed") or 1.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("speed must be numeric.") from exc
    model = str(payload.get("tts_model") or "auto").strip().lower().replace("-", "_")
    model = {
        "hf": "hf_space",
        "khmer_hf": "hf_space",
        "edge_tts": "edge",
    }.get(model, model)
    if model not in {"auto", "hf_space", "edge"}:
        model = "auto"
    return TTSRequest(text, gender, max(0.5, min(2.0, speed)), model)


def normalize_tts_model(
    value: Any,
    *,
    aliases: Mapping[str, str],
    default: str = "auto",
) -> str:
    raw = str(value or default or "auto").strip().lower().replace("-", "_")
    return str(aliases.get(raw, "auto"))


def clean_tts_text(text: str) -> str:
    """Remove hidden/control characters without rewriting user content."""

    cleaned = str(text or "")
    for character in ("\ufeff", "\u200b", "\u200c", "\u200d"):
        cleaned = cleaned.replace(character, "")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", cleaned)
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

__all__ = [
    "TTSRequest",
    "_detect_tts_lang_key",
    "_edge_tts_stream_once",
    "_edge_tts_stream_with_retry",
    "_tts_provider_summary",
    "clean_tts_text",
    "generate_user_voice_limited",
    "generate_voice",
    "generate_voice_limited",
    "normalize_tts_model",
    "normalize_tts_request",
    "resolve_tts_text",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
