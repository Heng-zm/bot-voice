"""Language and writing-system detection helpers."""

from __future__ import annotations

import re

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_ARABIC_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

_LATIN_LANGUAGE_MARKERS: dict[str, dict[str, int]] = {
    "id": {
        "bahwa": 3,
        "karena": 2,
        "adalah": 2,
        "bisa": 2,
        "kalian": 2,
        "nggak": 3,
        "tidak": 1,
        "saya": 1,
        "dari": 1,
        "kepada": 1,
        "sedang": 1,
        "sudah": 1,
        "belum": 1,
        "juga": 1,
    },
    "ms": {
        "bahawa": 3,
        "kerana": 2,
        "ialah": 3,
        "boleh": 2,
        "awak": 2,
        "daripada": 2,
        "tak": 1,
        "tidak": 1,
        "saya": 1,
        "kepada": 1,
        "sedang": 1,
        "sudah": 1,
        "belum": 1,
        "juga": 1,
    },
    "fil": {
        "kumusta": 3,
        "salamat": 2,
        "mga": 3,
        "hindi": 2,
        "mayroon": 2,
        "ngunit": 2,
        "ako": 1,
        "ikaw": 1,
        "natin": 1,
        "ninyo": 2,
        "ito": 1,
        "iyon": 2,
        "opo": 3,
        "po": 1,
    },
}

_LANGUAGE_FLAGS = {
    "km": "🇰🇭",
    "en": "🇺🇸",
    "zh": "🇨🇳",
    "ko": "🇰🇷",
    "ja": "🇯🇵",
    "hi": "🇮🇳",
    "ms": "🇲🇾",
    "id": "🇮🇩",
    "fil": "🇵🇭",
    "ar": "🇸🇦",
}
_LANGUAGE_NAMES = {
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


def _looks_like_japanese_han_phrase(text: str) -> bool:
    """Conservatively recognize common Japanese-only Han phrases."""

    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return any(item in compact for item in ("日本語", "仮名", "片仮名", "平仮名"))


def _detect_latin_tts_language(text: str) -> str:
    """Conservatively detect Indonesian, Malay, or Filipino Latin text."""

    words = [word.lower() for word in _LATIN_WORD_RE.findall(str(text or ""))]
    if not words:
        return ""
    scores = {
        language: sum(markers.get(word, 0) for word in words)
        for language, markers in _LATIN_LANGUAGE_MARKERS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score >= 3 and best_score - second_score >= 2:
        return best_language
    return ""


def _detect_lang(text: str) -> str:
    """Detect the dominant writing system without manual prefix overrides."""

    value = str(text or "")
    if not value:
        return "en"

    khmer = 0
    korean = 0
    japanese = 0
    chinese = 0
    devanagari = 0
    arabic = 0
    latin = 0

    for char in value:
        cp = ord(char)
        if 0x1780 <= cp <= 0x17FF:
            khmer += 1
        elif (
            (0x1100 <= cp <= 0x11FF)
            or (0x3130 <= cp <= 0x318F)
            or (0xAC00 <= cp <= 0xD7AF)
        ):
            korean += 1
        elif (0x3040 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF):
            japanese += 1
        elif (
            (0x3400 <= cp <= 0x4DBF)
            or (0x4E00 <= cp <= 0x9FFF)
            or (0xF900 <= cp <= 0xFAFF)
        ):
            chinese += 1
        elif 0x0900 <= cp <= 0x097F:
            devanagari += 1
        elif (
            (0x0600 <= cp <= 0x06FF)
            or (0x0750 <= cp <= 0x077F)
            or (0x08A0 <= cp <= 0x08FF)
            or (0xFB50 <= cp <= 0xFDFF)
            or (0xFE70 <= cp <= 0xFEFF)
        ):
            arabic += 1
        elif (65 <= cp <= 90) or (97 <= cp <= 122):
            latin += 1

    signal_total = (
        khmer + korean + japanese + chinese + devanagari + arabic + latin
    )
    if signal_total <= 0:
        return "en"
    if khmer and khmer / signal_total >= 0.15:
        return "km"
    if korean and korean / signal_total >= 0.15:
        return "ko"
    if japanese and japanese / signal_total >= 0.08:
        return "ja"
    if _looks_like_japanese_han_phrase(value):
        return "ja"
    if chinese and chinese / signal_total >= 0.15:
        return "zh"
    if devanagari and devanagari / signal_total >= 0.15:
        return "hi"
    if arabic and arabic / signal_total >= 0.15:
        return "ar"
    return _detect_latin_tts_language(value) or "en"


def _language_display(lang_key: str) -> tuple[str, str]:
    normalized = str(lang_key or "en").lower().strip()
    return (
        _LANGUAGE_FLAGS.get(normalized, "🌐"),
        _LANGUAGE_NAMES.get(normalized, normalized.upper() or "Unknown"),
    )


__all__ = [
    "_detect_lang",
    "_detect_latin_tts_language",
    "_language_display",
    "_looks_like_japanese_han_phrase",
]
