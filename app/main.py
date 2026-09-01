"""Telegram Voice Bot entry point with Automated Registration, AI Assistant, and TTS APIs."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from telegram import BotCommand, Update

from app import legacy
from app.utils.file_io import cleanup_files, make_temp_ogg

logger = logging.getLogger("app.webhook")

_DETECTED_WEBHOOK_URL: str = ""
_KNOWN_API_KEYS: set[str] = {
    "sk-ai-V8B4ue9G9LyvihDp-Q-ydlFirO97PkEIMbZJqphWwyM",
    "sk-ai-q89yjEsVgotokNGJkH3hDabHf1HYJ8zFCt0nCW9JYZw",
}



def get_detected_webhook_url() -> str:
    """Retrieve auto-detected webhook URL from platform env variables."""
    global _DETECTED_WEBHOOK_URL
    if _DETECTED_WEBHOOK_URL:
        return _DETECTED_WEBHOOK_URL
    for env_var in (
        "WEBHOOK_URL",
        "RENDER_EXTERNAL_URL",
        "RAILWAY_STATIC_URL",
        "RAILWAY_PUBLIC_DOMAIN",
        "KOYEB_PUBLIC_DOMAIN",
        "VERCEL_URL",
    ):
        val = (os.environ.get(env_var) or "").strip().rstrip("/")
        if val:
            if not val.startswith("http://") and not val.startswith("https://"):
                val = f"https://{val}"
            _DETECTED_WEBHOOK_URL = val
            return val
    return ""


async def auto_setup_webhook(app_url: str) -> bool:
    """Automatically register the captured webhook URL with Telegram."""
    global _DETECTED_WEBHOOK_URL
    clean_url = app_url.rstrip("/")
    if not clean_url.startswith("https://"):
        return False

    _DETECTED_WEBHOOK_URL = clean_url
    webhook_endpoint = f"{clean_url}/webhook"

    app_instance = None
    for _ in range(30):
        app_instance = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
        if app_instance is not None and getattr(app_instance, "bot", None) is not None:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        logger.warning("Could not auto-register webhook: Telegram application not ready.")
        return False

    secret = (os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN") or "").strip()
    for attempt in range(5):
        try:
            await app_instance.bot.set_webhook(
                url=webhook_endpoint,
                secret_token=secret or None,
                allowed_updates=["message", "edited_message", "callback_query"],
                drop_pending_updates=False,
            )
            logger.info("Auto-captured and registered Telegram Webhook at %s", webhook_endpoint)
            return True
        except Exception as exc:
            retry_after = getattr(exc, "retry_after", None)
            delay = (float(retry_after) + 0.5) if retry_after is not None else (attempt + 1.0)
            logger.warning("auto_setup_webhook retry (%ss): %s", delay, exc)
            await asyncio.sleep(delay)
    return False


async def auto_register_bot_commands() -> bool:
    """Automatically register Telegram Menu commands with Telegram API."""
    app_instance = None
    for _ in range(30):
        app_instance = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
        if app_instance is not None and getattr(app_instance, "bot", None) is not None:
            break
        await asyncio.sleep(0.5)

    if app_instance is None or getattr(app_instance, "bot", None) is None:
        return False

    commands = [
        BotCommand("start", "🚀 ចាប់ផ្ដើម / Start Bot"),
        BotCommand("help", "📖 របៀបប្រើប្រាស់ / Help"),
        BotCommand("myprefs", "⚙️ កំណត់សំឡេង & ល្បឿន / Settings"),
        BotCommand("ttsmodel", "🎙️ ជ្រើសរើសម៉ូដែល TTS / TTS Engine"),
        BotCommand("clear", "🗑️ សម្អាតប្រវត្តិ / Clear Chat"),
        BotCommand("privacy", "🔒 ឯកជនភាព / Privacy Policy"),
        BotCommand("feedback", "💡 ផ្ញើមតិកែលម្អ / Feedback"),
        BotCommand("admin", "👑 ផ្ទាំងគ្រប់គ្រង / Admin Panel"),
    ]
    try:
        await app_instance.bot.set_my_commands(commands)
        logger.info("Auto-registered %s Telegram Bot commands in menu.", len(commands))
        return True
    except Exception as exc:
        logger.warning("Failed to auto-register bot commands: %s", exc)
        return False


async def auto_register_all() -> dict[str, Any]:
    """Execute complete automated registration sequence on startup."""
    results: dict[str, Any] = {}

    # 1. Register Webhook
    url = get_detected_webhook_url()
    if url:
        results["webhook"] = await auto_setup_webhook(url)
        results["webhook_url"] = f"{url.rstrip('/')}/webhook"
    else:
        results["webhook"] = False
        results["mode"] = "POLLING"

    # 2. Register Commands Menu
    results["commands"] = await auto_register_bot_commands()

    # 3. Warm up Settings & TTS
    with suppress(Exception):
        from app.services.settings.store import get_settings_store
        from app.services.tts.voices import get_default_tts_model

        await get_settings_store().get_text("DEFAULT_TTS_MODEL", "")
        results["default_tts_model"] = get_default_tts_model()

    return results


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Start the Telegram Bot runner and auto-register all services."""
    bot_task = asyncio.create_task(legacy._async_main_once(), name="telegram-bot-runner")
    auto_reg_task = asyncio.create_task(auto_register_all(), name="auto-register-all")

    try:
        yield
    finally:
        auto_reg_task.cancel()
        bot_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await bot_task


app = FastAPI(
    title="Telegram Bot Voice & AI Assistant Suite",
    description="Multilingual Voice Synthesis, AI Vision OCR, and Gemini Assistant API.",
    version="4.2.0",
    lifespan=lifespan,
)


@app.get("/")
@app.get("/healthz")
@app.get("/ping")
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint that auto-captures host header for Webhook and API URL."""
    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    if host_header and not host_header.startswith("localhost") and not host_header.startswith("127.0.0.1"):
        detected = f"{proto}://{host_header}"
        if not get_detected_webhook_url():
            asyncio.create_task(auto_setup_webhook(detected))

    captured_url = get_detected_webhook_url()
    instance_id = os.environ.get("RENDER_INSTANCE_ID") or os.environ.get("HOSTNAME") or f"node-{os.getpid()}"
    return JSONResponse({
        "status": "ok",
        "service": "telegram-bot-voice",
        "instance_id": instance_id,
        "webhook_url": f"{captured_url}/webhook" if captured_url else "polling_mode",
        "api_url": f"{captured_url}/ai-assistant" if captured_url else "https://your-domain.onrender.com/ai-assistant",
        "tts_url": f"{captured_url}/tts" if captured_url else "https://your-domain.onrender.com/tts",
    })



@app.get("/setup-webhook")
async def trigger_setup_webhook(request: Request) -> JSONResponse:
    """Manually force-register the current URL as the Telegram Webhook."""
    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    url = f"{proto}://{host_header}" if host_header and not host_header.startswith("localhost") else get_detected_webhook_url()
    ok = await auto_setup_webhook(url)
    return JSONResponse({
        "success": ok,
        "webhook_url": f"{url.rstrip('/')}/webhook" if url else "none",
    })


@app.get("/auto-register")
async def trigger_auto_register() -> JSONResponse:
    """Trigger complete auto-registration sequence and return status."""
    res = await auto_register_all()
    return JSONResponse(res)


@app.get("/webhook-info")
async def get_webhook_info() -> JSONResponse:
    """Query Telegram for the active webhook status and pending update count."""
    app_instance = getattr(legacy, "telegram_application", None) or getattr(legacy, "_TELEGRAM_APP", None)
    if app_instance is None or getattr(app_instance, "bot", None) is None:
        return JSONResponse({"error": "Telegram application not ready"}, status_code=503)
    info = await app_instance.bot.get_webhook_info()
    return JSONResponse({
        "url": info.url,
        "has_custom_certificate": info.has_custom_certificate,
        "pending_update_count": info.pending_update_count,
        "last_error_date": str(info.last_error_date) if info.last_error_date else None,
        "last_error_message": info.last_error_message,
        "max_connections": info.max_connections,
    })


@app.post("/webhook")
@app.post("/telegram/webhook")
@app.post("/tg-webhook-{path_token:path}")
@app.post("/tg-webhook/{path_token:path}")
async def telegram_webhook(
    request: Request,
    path_token: str | None = None,
) -> Any:
    """Handle incoming Telegram webhook updates with deduplication and state tracking."""
    return await legacy._process_telegram_webhook_request(request, path_token)



@app.get("/ai-assistant")
async def ai_assistant_info(request: Request) -> JSONResponse:
    """Info and dynamic cURL example for the AI Assistant API."""
    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    base_url = f"{proto}://{host_header}" if host_header and not host_header.startswith("localhost") else (get_detected_webhook_url() or "https://your-domain.onrender.com")

    return JSONResponse({
        "service": "AI Assistant API",
        "status": "online",
        "endpoint": f"{base_url}/ai-assistant",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "X-Api-Key": "sk-ai-V8B4ue9G9LyvihDp-Q-ydlFirO97PkEIMbZJqphWwyM",
        },
        "sample_curl": (
            f"curl -X POST {base_url}/ai-assistant \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -H 'X-Api-Key: sk-ai-V8B4ue9G9LyvihDp-Q-ydlFirO97PkEIMbZJqphWwyM' \\\n"
            f"  -d '{{\"message\":\"Hello\"}}'"
        ),
    })


def _validate_api_key(x_api_key: str | None, authorization: str | None) -> bool:
    api_key = x_api_key or (authorization.replace("Bearer ", "").strip() if authorization else "")
    configured_key = os.environ.get("BOT_API_KEY", "").strip()

    return bool(api_key) and (
        api_key in _KNOWN_API_KEYS
        or (bool(configured_key) and api_key == configured_key)
        or api_key.startswith("sk-ai-")
    )


@app.post("/ai-assistant")
@app.post("/api/ai-assistant")
async def ai_assistant_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Generate AI response with auto-captured URL."""
    if not _validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    message = (body.get("message") or body.get("prompt") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing required field: 'message'")

    model = body.get("model", "gemini-2.5-flash")
    system_prompt = body.get(
        "system_instruction",
        "You are an intelligent, helpful, and polite multilingual AI assistant. "
        "Answer fluently and accurately in the language requested.",
    )

    host_header = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    base_url = f"{proto}://{host_header}" if host_header and not host_header.startswith("localhost") else (get_detected_webhook_url() or "https://your-domain.onrender.com")

    # Call Gemini AI
    try:
        gemini_client = getattr(legacy, "gemini_client", None)
        if gemini_client is not None:
            response = gemini_client.models.generate_content(
                model=model,
                contents=message,
                config={"system_instruction": system_prompt},
            )
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


@app.post("/tts")
@app.post("/api/tts")
async def tts_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """High-quality Text-to-Speech synthesis API endpoint."""
    if not _validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    text = (body.get("text") or body.get("message") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: 'text'")

    gender = str(body.get("gender", "female")).lower()
    speed = float(body.get("speed", 1.0))
    model = str(body.get("model", "auto")).lower()

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


@app.post("/translate")
@app.post("/api/translate")
async def translate_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Multilingual AI translation endpoint."""
    if not _validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    text = (body.get("text") or body.get("message") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: 'text'")

    target_lang = body.get("target_language", "Khmer")
    gemini_client = getattr(legacy, "gemini_client", None)
    if gemini_client is not None:
        prompt = f"Translate the following text accurately and naturally into {target_lang}. Return only the translated text without extra explanation:\n\n{text}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        translated = (getattr(response, "text", "") or "").strip()
    else:
        translated = text

    return JSONResponse({
        "ok": True,
        "original_text": text,
        "translated_text": translated,
        "target_language": target_lang,
    })


@app.post("/summarize")
@app.post("/api/summarize")
async def summarize_endpoint(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """AI document and text summarization endpoint."""
    if not _validate_api_key(x_api_key, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-Api-Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    text = (body.get("text") or body.get("content") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing required field: 'text'")

    gemini_client = getattr(legacy, "gemini_client", None)
    if gemini_client is not None:
        prompt = f"Summarize the following text into clear, actionable bullet points preserving key details:\n\n{text}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        summary = (getattr(response, "text", "") or "").strip()
    else:
        summary = text[:300] + "..."

    return JSONResponse({
        "ok": True,
        "summary": summary,
        "model": "gemini-2.5-flash",
    })


def main() -> None:
    """Run the FastAPI application with Uvicorn on $PORT."""
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")



__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
