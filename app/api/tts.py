"""Text-to-Speech synthesis API endpoints."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from app import legacy
from app.core.security import validate_api_key

logger = logging.getLogger("app.api.tts")

router = APIRouter(tags=["Text to Speech"])

MAX_TTS_TEXT_LENGTH = 4000
DEFAULT_TTS_SPEED = 1.0
VALID_GENDERS = {"female", "male"}
SUPPORTED_MODELS = {
    "auto": "Automatic best-engine selection (Khmer Kiri for Khmer, Edge for others)",
    "kiri": "Khmer Kiri neural engine (mrrtmob/khmer-tts)",
    "edge": "Microsoft Edge Neural TTS (Natural multi-lingual)",
    "gemini": "Google Gemini AI conversational voice synthesis",
}


def _cleanup_temp_file(path: str | None) -> None:
    """Safely delete a temporary file."""
    if path:
        with suppress(Exception):
            if os.path.exists(path):
                os.remove(path)


def _detect_audio_mime(audio_bytes: bytes) -> tuple[str, str]:
    """Detect MIME type and file extension from raw audio magic bytes.

    Returns:
        (mime_type, extension)
    """
    if not audio_bytes:
        return "audio/ogg", ".ogg"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg", ".ogg"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg", ".mp3"
    if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 12 and audio_bytes[8:12] == b"WAVE":
        return "audio/wav", ".wav"
    return "audio/ogg", ".ogg"


def _check_api_auth(
    x_api_key: str | None = None,
    authorization: str | None = None,
    api_key_query: str | None = None,
) -> None:
    """Validate API credentials from headers or query parameters."""
    effective_key = x_api_key or api_key_query
    if not validate_api_key(effective_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")


async def _synthesize_audio_multi_tier(
    text: str,
    gender: str = "female",
    speed: float = 1.0,
    model: str = "auto",
) -> bytes:
    """Multi-tier resilient audio synthesis with fallback across all available engines."""
    # 1. Try modern modular TTS service if available
    with suppress(Exception):
        from app.services.tts.service import synthesize_speech

        audio_bytes = await synthesize_speech(
            text=text,
            gender=gender,
            speed=speed,
            model=model,
        )
        if audio_bytes:
            return audio_bytes

    # 2. Try legacy limited generator with temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".ogg")
    os.close(temp_fd)
    try:
        if hasattr(legacy, "generate_voice_limited"):
            audio_bytes = await legacy.generate_voice_limited(
                text=text,
                gender=gender,
                speed=speed,
                output_path=temp_path,
                tts_model=model,
            )
            if audio_bytes:
                return audio_bytes

        if hasattr(legacy, "generate_voice"):
            audio_bytes = await legacy.generate_voice(
                text=text,
                gender=gender,
                speed=speed,
                output_path=temp_path,
                tts_model=model,
            )
            if audio_bytes:
                return audio_bytes
    finally:
        _cleanup_temp_file(temp_path)

    # 3. Direct Edge-TTS fallback
    try:
        import edge_tts

        voice = "km-KH-PisethNeural" if gender == "male" else "km-KH-SreymomNeural"
        spd_pct = int(round((speed - 1.0) * 100))
        rate_str = f"{spd_pct:+d}%"

        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.extend(chunk["data"])
        if buffer:
            return bytes(buffer)
    except Exception as edge_err:
        logger.warning("Direct Edge-TTS fallback failed: %s", edge_err)

    raise RuntimeError("All Text-to-Speech synthesis engines failed to generate audio.")


@router.get("/tts/models")
@router.get("/api/tts/models")
async def list_tts_models(
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """List available TTS synthesis models and supported parameters."""
    _check_api_auth(x_api_key, authorization, api_key)
    return JSONResponse({
        "ok": True,
        "models": SUPPORTED_MODELS,
        "genders": list(VALID_GENDERS),
        "speed_range": {"min": 0.5, "max": 2.5, "default": 1.0},
        "max_length": MAX_TTS_TEXT_LENGTH,
    })


@router.post("/tts")
@router.post("/api/tts")
async def tts_endpoint(
    request: Request,
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Response:
    """High-quality Text-to-Speech synthesis API endpoint (POST).

    Supports JSON output (base64) or direct audio streaming based on format
    field or Accept headers.
    """
    _check_api_auth(x_api_key, authorization, api_key)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    text = str(body.get("text") or body.get("message") or body.get("input") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: 'text'")

    if len(text) > MAX_TTS_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {MAX_TTS_TEXT_LENGTH} characters (received {len(text)})",
        )

    # Clean and validate parameters
    gender = str(body.get("gender", "female") or "female").lower().strip()
    if gender not in VALID_GENDERS:
        gender = "female"

    try:
        speed = float(body.get("speed", DEFAULT_TTS_SPEED))
        if speed < 0.5 or speed > 2.5 or speed != speed:
            speed = DEFAULT_TTS_SPEED
    except (ValueError, TypeError):
        speed = DEFAULT_TTS_SPEED

    raw_model = str(body.get("model", "auto") or "auto").lower().strip()
    model = raw_model if raw_model in SUPPORTED_MODELS else "auto"

    requested_format = str(body.get("format", "")).lower().strip()
    accept_header = request.headers.get("accept", "").lower()
    stream_audio = (
        requested_format in ("audio", "binary", "stream", "file")
        or "audio/" in accept_header
    )

    try:
        audio_bytes = await _synthesize_audio_multi_tier(
            text=text,
            gender=gender,
            speed=speed,
            model=model,
        )

        mime_type, ext = _detect_audio_mime(audio_bytes)

        if stream_audio:
            filename = f"tts_{gender}_{int(speed * 100)}{ext}"
            return Response(
                content=audio_bytes,
                media_type=mime_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return JSONResponse({
            "ok": True,
            "text": text,
            "gender": gender,
            "speed": speed,
            "model": model,
            "mime_type": mime_type,
            "bytes_length": len(audio_bytes),
            "audio_base64": audio_b64,
        })
    except Exception as exc:
        logger.warning("TTS API synthesis failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@router.get("/tts")
@router.get("/api/tts")
async def tts_get_endpoint(
    request: Request,
    text: str = Query(..., description="Text to synthesize"),
    gender: str = Query(default="female", description="Voice gender: female or male"),
    speed: float = Query(default=1.0, ge=0.5, le=2.5, description="Speech rate"),
    model: str = Query(default="auto", description="Model: auto, kiri, edge, gemini"),
    stream: bool = Query(default=True, description="Return raw audio stream instead of JSON"),
    api_key: str | None = Query(default=None, alias="api_key"),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Response:
    """Browser-friendly GET endpoint for direct audio streaming via URL or audio tags."""
    _check_api_auth(x_api_key, authorization, api_key)

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Missing required parameter: 'text'")

    if len(clean_text) > MAX_TTS_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {MAX_TTS_TEXT_LENGTH} characters",
        )

    clean_gender = gender.lower().strip()
    if clean_gender not in VALID_GENDERS:
        clean_gender = "female"

    clean_model = model.lower().strip()
    if clean_model not in SUPPORTED_MODELS:
        clean_model = "auto"

    try:
        audio_bytes = await _synthesize_audio_multi_tier(
            text=clean_text,
            gender=clean_gender,
            speed=speed,
            model=clean_model,
        )

        mime_type, ext = _detect_audio_mime(audio_bytes)

        if stream:
            filename = f"tts_{clean_gender}_{int(speed * 100)}{ext}"
            return Response(
                content=audio_bytes,
                media_type=mime_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return JSONResponse({
            "ok": True,
            "text": clean_text,
            "gender": clean_gender,
            "speed": speed,
            "model": clean_model,
            "mime_type": mime_type,
            "bytes_length": len(audio_bytes),
            "audio_base64": audio_b64,
        })
    except Exception as exc:
        logger.warning("TTS GET synthesis failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


__all__ = ["router"]