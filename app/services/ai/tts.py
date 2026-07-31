"""Speech generation and Telegram voice conversion helpers."""

from app._legacy_bridge import exported_dir, exported_getattr

__all__ = [
    "_detect_tts_lang_key",
    "_edge_tts_stream_once",
    "_edge_tts_stream_with_retry",
    "_generate_voice_voxcpm2",
    "_prepare_voxcpm2_session",
    "_tts_provider_summary",
    "_voxcpm2_api_inputs",
    "_voxcpm2_normalize_profile",
    "_voxcpm2_profile_ready",
    "generate_user_voice_limited",
    "generate_voice",
    "generate_voice_limited",
    "resolve_tts_text",
]

__getattr__ = exported_getattr(__name__, __all__)


def __dir__() -> list[str]:
    return exported_dir(globals(), __all__)
