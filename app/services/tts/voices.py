"""TTS voice definitions, candidate resolution, and normalization."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from app.services.ai.language import _detect_lang

VOICE_MAP: dict[str, dict[str, str]] = {
    # Khmer
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    # English (United States)
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
    # Chinese / Mandarin
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    # Korean
    "ko": {"female": "ko-KR-SunHiNeural", "male": "ko-KR-InJoonNeural"},
    # Japanese
    "ja": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
    # Hindi (India)
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    # Malay (Malaysia)
    "ms": {"female": "ms-MY-YasminNeural", "male": "ms-MY-OsmanNeural"},
    # Indonesian (Indonesia)
    "id": {"female": "id-ID-GadisNeural", "male": "id-ID-ArdiNeural"},
    # Filipino / Tagalog (Philippines)
    "fil": {"female": "fil-PH-BlessicaNeural", "male": "fil-PH-AngeloNeural"},
    # Arabic (Saudi Arabia)
    "ar": {"female": "ar-SA-ZariyahNeural", "male": "ar-SA-HamedNeural"},
}

TTS_LANGUAGE_LABELS: dict[str, str] = {
    "km": "Khmer",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "hi": "Hindi (India)",
    "ms": "Malay (Malaysia)",
    "id": "Indonesian",
    "fil": "Filipino (Philippines)",
    "ar": "Arabic",
}

TTS_SUPPORTED_LANG_ORDER: tuple[str, ...] = (
    "en",
    "km",
    "zh",
    "ja",
    "ko",
    "hi",
    "ms",
    "id",
    "fil",
    "ar",
)

DEFAULT_SPEED: float = 1.0

SPEED_OPTIONS: dict[str, tuple[str, float]] = {
    "spd_0.5": ("x0.5", 0.5),
    "spd_1.0": ("Normal", 1.0),
    "spd_1.5": ("x1.5", 1.5),
    "spd_2.0": ("x2.0", 2.0),
}

TTS_MODEL_OPTIONS: dict[str, tuple[str, str]] = {
    "auto": ("ស្វ័យប្រវត្តិ", "Kiri → Edge TTS"),
    "gemini": ("សំឡេង Gemini AI", "Google Gemini TTS"),
    "edge": ("Edge TTS ពហុភាសា", ""),
    "hf_space": ("សំឡេងខ្មែរ Kiri", ""),
}

TTS_MODEL_ALIASES: dict[str, str] = {
    "auto": "auto",
    "default": "auto",
    "server": "auto",
    "gemini": "gemini",
    "gemini_tts": "gemini",
    "google": "gemini",
    "google_tts": "gemini",
    "genai": "gemini",
    "hf": "hf_space",
    "hf_space": "hf_space",
    "khmer_hf": "hf_space",
    "khmer_hf_space": "hf_space",
    "khmer-tts": "hf_space",
    "mrrtmob": "hf_space",
    "edge": "edge",
    "edge_tts": "edge",
    "msedge": "edge",
}


DEFAULT_TTS_MODEL: str = (
    os.environ.get("DEFAULT_TTS_MODEL")
    or os.environ.get("USER_DEFAULT_TTS_MODEL")
    or "auto"
).strip().lower()


def normalize_tts_model(value: Any) -> str:
    """Normalize user-supplied TTS model identifier."""
    raw = str(value or DEFAULT_TTS_MODEL or "auto").strip().lower().replace("-", "_")
    return TTS_MODEL_ALIASES.get(raw, "auto")


def get_default_tts_model() -> str:
    """Return runtime default TTS model with dynamic database override."""
    with suppress(Exception):
        from app.services.settings.store import get_settings_store

        val = get_settings_store().get_text_sync("DEFAULT_TTS_MODEL", "")
        if val:
            return normalize_tts_model(val)
    return normalize_tts_model(DEFAULT_TTS_MODEL)


def tts_model_label(value: Any) -> str:
    """Return user-friendly label for a TTS model."""
    key = normalize_tts_model(value)
    label, hint = TTS_MODEL_OPTIONS.get(key, TTS_MODEL_OPTIONS["auto"])
    return f"{label} — {hint}" if hint else label



_ZERO_WIDTH_MAP = {ord(c): None for c in ("\ufeff", "\u200b", "\u200c", "\u200d")}
_CONTROL_MAP = {i: " " for i in (*range(0x00, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20), 0x7F)}
_CLEAN_TRANSLATE_TABLE = str.maketrans({**_ZERO_WIDTH_MAP, **_CONTROL_MAP})

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
_SENTENCE_DELIM_RE = re.compile(r"([។.!?\n]+)")


def clean_tts_text(text: str) -> str:
    """Remove hidden, control, and zero-width characters with fast C-level translation."""
    if not text:
        return ""
    val = text.translate(_CLEAN_TRANSLATE_TABLE)
    val = _WHITESPACE_RE.sub(" ", val)
    val = _EXCESS_NEWLINES_RE.sub("\n\n", val)
    return val.strip()


def resolve_tts_voice_candidates(
    text: str,
    gender: str = "female",
    *,
    cross_lang_fallback: bool = True,
    fallback_order: Sequence[str] = TTS_SUPPORTED_LANG_ORDER,
) -> list[str]:
    """Resolve ordered voice candidates: primary voice, same-language opposite gender, then cross-language."""
    norm_gender = "male" if str(gender).strip().lower() == "male" else "female"
    other_gender = "female" if norm_gender == "male" else "male"

    lang = _detect_lang(text)
    primary_pair = VOICE_MAP.get(lang) or VOICE_MAP["en"]

    candidates: list[str] = [
        primary_pair[norm_gender],
        primary_pair[other_gender],
    ]

    if cross_lang_fallback:
        for other_lang in fallback_order:
            if other_lang == lang:
                continue
            pair = VOICE_MAP.get(other_lang)
            if not pair:
                continue
            candidates.append(pair[norm_gender])
            candidates.append(pair[other_gender])

    seen: set[str] = set()
    unique: list[str] = []
    for voice in candidates:
        if voice and voice not in seen:
            seen.add(voice)
            unique.append(voice)
    return unique


def split_text_chunks(text: str, max_chars: int = 300) -> list[str]:
    """Split text into sentence/clause chunks respecting sentence boundaries."""
    cleaned = clean_tts_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            if current_chunk and len(current_chunk) + len(paragraph) + 1 <= max_chars:
                current_chunk += "\n" + paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        else:
            sentences = _SENTENCE_DELIM_RE.split(paragraph)
            sentence_parts: list[str] = []
            for i in range(0, len(sentences), 2):
                sent = sentences[i]
                delim = sentences[i + 1] if i + 1 < len(sentences) else ""
                combined = (sent + delim).strip()
                if combined:
                    sentence_parts.append(combined)

            for sentence in sentence_parts:
                if current_chunk and len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(sentence) <= max_chars:
                        current_chunk = sentence
                    else:
                        for j in range(0, len(sentence), max_chars):
                            sub = sentence[j : j + max_chars].strip()
                            if sub:
                                chunks.append(sub)
                        current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


__all__ = [
    "DEFAULT_SPEED",
    "DEFAULT_TTS_MODEL",
    "SPEED_OPTIONS",
    "TTS_LANGUAGE_LABELS",
    "TTS_MODEL_ALIASES",
    "TTS_MODEL_OPTIONS",
    "TTS_SUPPORTED_LANG_ORDER",
    "VOICE_MAP",
    "clean_tts_text",
    "normalize_tts_model",
    "resolve_tts_voice_candidates",
    "split_text_chunks",
    "tts_model_label",
]
