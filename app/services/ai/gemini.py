"""Gemini client, prompt, generation, transcription, and OCR helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_ai_gen_config",
    "_build_gemini_contents",
    "_gemini",
    "_gemini_generate_with_retry",
    "_load_google_genai_sdk",
    "_load_huggingface_sdk",
    "ask_gemini_ocr",
    "ocr_image",
    "transcribe_audio_file",
    "transcribe_voice",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
