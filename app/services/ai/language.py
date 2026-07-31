"""Language and writing-system detection helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_detect_lang",
    "_detect_latin_tts_language",
    "_language_display",
    "_looks_like_japanese_han_phrase",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
