"""Pure routing helpers for Telegram callback flows.

Keeping callback classification independent from the legacy runtime makes the
dispatcher easy to test and prevents broad prefix guards from swallowing old
or malformed buttons without answering them.
"""

from __future__ import annotations

from collections.abc import Collection


def classify_callback(
    data: str | None,
    *,
    speed_callbacks: Collection[str] = (),
) -> str | None:
    """Return the generic callback action for *data*, or ``None`` if unknown."""

    value = str(data or "").strip()
    exact_actions = {
        "show_speed": "show_speed",
        "hide_speed": "hide_speed",
        "show_tts_model": "show_tts_model",
        "hide_tts_model": "hide_tts_model",
        "tg_female": "gender",
        "tg_male": "gender",
        "welcome_profile": "welcome_profile",
    }
    if value in exact_actions:
        return exact_actions[value]
    if value in speed_callbacks:
        return "speed"

    prefix_actions = (
        ("ttsmodel_", "tts_model"),
        ("tts_transcript:", "tts_transcript"),
        ("del_transcript:", "delete"),
        ("doc_del:", "delete"),
        ("audio_del:", "delete"),
        ("doc_read:", "doc_read"),
        ("audio_tts:", "audio_tts"),
        ("needs_", "needs_admin"),
        ("api_", "api_admin"),
        ("admin_", "admin"),
    )
    for prefix, action in prefix_actions:
        if value.startswith(prefix):
            return action
    return None


def callback_requires_tts_access(action: str, data: str | None) -> bool:
    """Return whether a generic callback changes or generates TTS state."""

    return action in {
        "speed",
        "gender",
        "tts_model",
        "tts_transcript",
        "doc_read",
        "audio_tts",
    }


__all__ = ["callback_requires_tts_access", "classify_callback"]
