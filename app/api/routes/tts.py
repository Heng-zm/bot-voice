"""Text-to-Speech synthesis API endpoints."""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app import legacy
from app.core.security import validate_api_key
from app.utils.file_io import cleanup_files, make_temp_ogg

logger = logging.getLogger("app.api.tts")

router = APIRouter(tags=["Text to Speech"])


@router.post("/tts")
@router.post("/api/tts")
async def tts_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """High-quality Text-to-Speech synthesis API endpoint."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    text = (body.get("text") or body.get("message") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: 'text'")

    gender = str(body.get("gender", "female") or "female").lower()
    if gender not in ("female", "male"):
        gender = "female"
    try:
        speed = float(body.get("speed", 1.0))
        if speed <= 0.2 or speed > 3.0 or speed != speed:
            speed = 1.0
    except (ValueError, TypeError):
        speed = 1.0
    model = str(body.get("model", "auto") or "auto").lower()

    temp_path = make_temp_ogg()
    try:
        audio_bytes = await legacy.generate_voice_limited(
            text=text,
            gender=gender,
            speed=speed,
            output_path=temp_path,
            tts_model=model,
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return JSONResponse({
            "ok": True,
            "text": text,
            "gender": gender,
            "speed": speed,
            "model": model,
            "mime_type": "audio/ogg",
            "bytes_length": len(audio_bytes),
            "audio_base64": audio_b64,
        })
    except Exception as exc:
        logger.warning("TTS API synthesis failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    finally:
        cleanup_files(temp_path)


__all__ = ["router"]
