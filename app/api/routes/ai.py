"""AI Assistant, Translation, Summarization, and Article Narration API endpoints."""

from __future__ import annotations

import asyncio
import base64
import logging
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app import legacy
from app.core.config import get_detected_webhook_url
from app.core.security import validate_api_key
from app.services.ai.gemini import generate_content_with_fallback
from app.utils.file_io import cleanup_files, make_temp_ogg

logger = logging.getLogger("app.api.ai")

router = APIRouter(tags=["AI Assistant & Intelligence"])


@router.get("/ai-assistant")
async def ai_assistant_info(request: Request) -> JSONResponse:
    """Info and dynamic cURL example for the AI Assistant API."""
    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    base_url = f"{proto}://{host_header}" if host_header and not host_header.startswith("localhost") else (get_detected_webhook_url() or "https://your-domain.anajak.cloud")

    return JSONResponse({
        "service": "AI Assistant API",
        "status": "online",
        "endpoint": f"{base_url}/ai-assistant",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "X-Api-Key": "YOUR_API_KEY",
        },
        "sample_curl": (
            f"curl -X POST {base_url}/ai-assistant \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -H 'X-Api-Key: YOUR_API_KEY' \\\n"
            f"  -d '{{\"message\":\"Hello\"}}'"
        ),
    })


@router.post("/ai-assistant")
@router.post("/api/ai-assistant")
async def ai_assistant_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Generate AI response with auto-captured URL."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    message = (body.get("message") or body.get("prompt") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing required field: 'message'")

    model = body.get("model", getattr(legacy, "GEMINI_MODEL", "gemini-3.6-flash"))
    system_prompt = body.get(
        "system_instruction",
        "You are an intelligent, helpful, and polite multilingual AI assistant. "
        "Answer fluently and accurately in the language requested.",
    )

    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    base_url = f"{proto}://{host_header}" if host_header and not host_header.startswith("localhost") else (get_detected_webhook_url() or "https://your-domain.anajak.cloud")

    # Call Gemini AI
    try:
        gemini_client = getattr(legacy, "_gemini", None)
        if gemini_client is not None:
            loop = asyncio.get_running_loop()

            def _call_ai():
                return generate_content_with_fallback(
                    client=gemini_client,
                    contents=message,
                    preferred_model=model,
                    config={"system_instruction": system_prompt},
                )

            response = await loop.run_in_executor(None, _call_ai)
            ai_text = (getattr(response, "text", "") or "").strip()
        else:
            ai_text = f"Received: {message}"
    except Exception as exc:
        logger.warning("AI generation failed: %s", exc)
        ai_text = f"I received your message: '{message}'."

    return JSONResponse({
        "ok": True,
        "response": ai_text,
        "model": model,
        "api_url": f"{base_url}/ai-assistant",
    })


@router.post("/translate")
@router.post("/api/translate")
async def translate_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Multilingual AI translation endpoint."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    text = (body.get("text") or body.get("message") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: 'text'")

    target_lang = body.get("target_language", "Khmer")
    gemini_client = getattr(legacy, "_gemini", None)
    preferred_model = getattr(legacy, "GEMINI_MODEL", "gemini-3.6-flash")
    if gemini_client is not None:
        loop = asyncio.get_running_loop()
        prompt = f"Translate the following text accurately and naturally into {target_lang}. Return only the translated text without extra explanation:\n\n{text}"

        def _call_ai():
            return generate_content_with_fallback(
                client=gemini_client,
                contents=prompt,
                preferred_model=preferred_model,
            )

        try:
            response = await loop.run_in_executor(None, _call_ai)
            translated = (getattr(response, "text", "") or "").strip()
        except Exception as exc:
            logger.warning("AI translation failed: %s", exc)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    else:
        translated = text

    return JSONResponse({
        "ok": True,
        "original_text": text,
        "translated_text": translated,
        "target_language": target_lang,
    })


@router.post("/summarize")
@router.post("/api/summarize")
async def summarize_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """AI document and text summarization endpoint."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    text = (body.get("text") or body.get("content") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: 'text'")

    gemini_client = getattr(legacy, "_gemini", None)
    preferred_model = getattr(legacy, "GEMINI_MODEL", "gemini-3.6-flash")
    if gemini_client is not None:
        loop = asyncio.get_running_loop()
        prompt = f"Summarize the following text into clear, actionable bullet points preserving key details:\n\n{text}"

        def _call_ai():
            return generate_content_with_fallback(
                client=gemini_client,
                contents=prompt,
                preferred_model=preferred_model,
            )

        try:
            response = await loop.run_in_executor(None, _call_ai)
            summary = (getattr(response, "text", "") or "").strip()
        except Exception as exc:
            logger.warning("AI summarization failed: %s", exc)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    else:
        summary = text[:300] + "..."

    return JSONResponse({
        "ok": True,
        "summary": summary,
        "model": preferred_model,
    })


@router.post("/narrate")
@router.post("/api/narrate")
async def narrate_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Scrape a public web article, summarize it, and return metadata and TTS voice audio."""
    if not validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    url = str(body.get("url") or body.get("link") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing required field: 'url'")

    from app.services.ai.article_reader import (
        MIN_ARTICLE_CHARS,
        extract_article_content,
        fetch_article_html,
        is_safe_public_url,
        summarize_article_with_ai,
    )

    safe, reason = is_safe_public_url(url)
    if not safe:
        raise HTTPException(status_code=400, detail=f"Invalid or forbidden URL: {reason}")

    try:
        html_data = await fetch_article_html(url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch remote article: {exc}") from exc

    title, body_text = extract_article_content(html_data)
    if len(body_text) < MIN_ARTICLE_CHARS:
        raise HTTPException(status_code=422, detail="Could not extract readable article text from URL.")

    gemini_client = getattr(legacy, "_gemini", None)
    preferred_model = body.get("model", getattr(legacy, "GEMINI_MODEL", "gemini-3.6-flash"))
    summary = summarize_article_with_ai(title, body_text, gemini_client, preferred_model)

    gender = str(body.get("gender", "female") or "female").lower()
    if gender not in ("female", "male"):
        gender = "female"
    try:
        speed = float(body.get("speed", 1.0))
        if speed <= 0.2 or speed > 3.0 or speed != speed:
            speed = 1.0
    except (ValueError, TypeError):
        speed = 1.0
    tts_model = str(body.get("tts_model", "auto") or "auto").lower()

    temp_path = make_temp_ogg()
    tts_script = f"{title}. {summary}" if title and not summary.startswith(title) else summary
    try:
        audio_bytes = await legacy.generate_voice_limited(
            text=tts_script,
            gender=gender,
            speed=speed,
            output_path=temp_path,
            tts_model=tts_model,
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return JSONResponse({
            "ok": True,
            "title": title,
            "url": url,
            "summary": summary,
            "audio_base64": audio_b64,
            "bytes_length": len(audio_bytes),
            "mime_type": "audio/ogg",
        })
    except Exception as exc:
        logger.warning("Article narration TTS synthesis failed: %s", exc)
        return JSONResponse({
            "ok": True,
            "title": title,
            "url": url,
            "summary": summary,
            "audio_error": str(exc),
        })
    finally:
        cleanup_files(temp_path)


__all__ = ["router"]
