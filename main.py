import logging
import os
import re
import asyncio
import inspect
import threading
import time
import html
import functools
import tempfile
import glob
import requests
import imageio_ffmpeg as _iio_ffmpeg
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None
from flask import Flask
from supabase import create_client, Client

# ── Flask Web Server ───────────────────────────────────────────────────────
app_flask = Flask(__name__)

@app_flask.route("/")
def health_check():
    return "Bot is running!", 200

@app_flask.route("/ping")
def ping():
    return "pong", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host="0.0.0.0", port=port, threaded=True)

def keep_alive():
    """Self-ping every 4 min as secondary keep-alive layer.

    Tries the base URL first, then /ping. This avoids noisy 404 logs when the
    deployed service does not expose /ping or RENDER_EXTERNAL_URL accidentally
    points to a route instead of the base app URL.
    """
    logger_ka = logging.getLogger("keep_alive")

    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if not render_url:
        logger_ka.warning("RENDER_EXTERNAL_URL not set — self-ping disabled.")
        return

    time.sleep(10)

    headers = {
        "User-Agent": "Mozilla/5.0 (KeepAlive/1.0)",
        "Accept": "text/plain,*/*",
    }

    while True:
        urls = [render_url]
        if not render_url.endswith("/ping"):
            urls.append(f"{render_url}/ping")

        ok = False
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if 200 <= r.status_code < 400:
                    logger_ka.info(f"Keep-alive OK -> {r.status_code} {url}")
                    ok = True
                    break

                logger_ka.warning(f"Keep-alive non-OK -> {r.status_code} {url}")

            except Exception as e:
                logger_ka.warning(f"Keep-alive failed for {url}: {e}")

        if not ok:
            logger_ka.warning(
                "Keep-alive failed for all URLs. Check RENDER_EXTERNAL_URL. "
                "Use the base URL only, e.g. https://your-app.onrender.com"
            )

        time.sleep(240)

# ── AI Assistant REST API ──────────────────────────────────────────────────
#
#  Endpoints:
#
#  POST /ai-assistant               – unified chat (text + optional image/audio)
#  POST /ai-assistant/transcribe    – audio → transcript only
#  POST /ai-assistant/vision        – image analysis / OCR
#  GET  /ai-assistant/info          – model info & capabilities
#
# ── Request formats ───────────────────────────────────────────────────────
#
#  POST /ai-assistant
#  Content-Type: multipart/form-data  OR  application/json
#
#  Multipart fields:
#    message      (str)  – user text prompt
#    image        (file) – image (jpg/png/webp/gif, max 10 MB)
#    audio        (file) – audio (mp3/wav/ogg/mp4/m4a/flac/opus, max 50 MB)
#    history      (json) – JSON array [{role,content}] for multi-turn chat
#    stream       (str)  – "true" to get Server-Sent Events streaming
#
#  JSON body alternative:
#    message      (str)
#    image_base64 (str)  – base64-encoded image bytes
#    image_mime   (str)  – MIME type for image_base64
#    audio_base64 (str)  – base64-encoded audio bytes
#    audio_mime   (str)  – MIME type for audio_base64
#    history      (list) – [{role, content}]
#    stream       (bool)
#
#  Optional auth header: X-Api-Key or Authorization: Bearer <key>
#  Set AI_API_KEY env var to enable key gate (leave empty to disable).
#
# ── Response (non-streaming) ──────────────────────────────────────────────
#  200 { "ok": true, "reply": "...", "detected_language": "km"|"en",
#        "tokens_used": <int|null>, "model": "..." }
#  4xx { "ok": false, "error": "...", "code": <int> }
#
# ── Response (streaming: stream=true) ─────────────────────────────────────
#  text/event-stream  SSE chunks:
#    data: {"ok":true,"delta":"...","done":false}
#    data: {"ok":true,"delta":"","done":true,"reply":"<full>","model":"..."}
#
# ──────────────────────────────────────────────────────────────────────────

import base64
import json as _json
from flask import request, jsonify, Response, stream_with_context

try:
    from huggingface_hub import InferenceClient
except Exception:  # huggingface_hub is optional; install it only when AI_PROVIDER=hf
    InferenceClient = None

_AI_API_MAX_IMAGE_BYTES   = 10 * 1024 * 1024   # 10 MB
_AI_API_MAX_AUDIO_BYTES   = 50 * 1024 * 1024   # 50 MB
_AI_API_MAX_MESSAGE_CHARS = 32_000
_AI_API_MAX_HISTORY_TURNS = 20

_AI_ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_AI_ALLOWED_AUDIO_MIME = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/flac",
    "audio/aac", "audio/opus", "audio/webm", "audio/mp4", "audio/x-m4a",
    "video/mp4", "video/webm",
}

_AI_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that supports both Khmer (ភាសាខ្មែរ) and English. "
    "Always reply in the same language the user uses. "
    "If the user sends an image, describe and analyse it fully. "
    "If the user sends audio, provide an accurate transcription then answer any follow-up. "
    "Be concise, accurate, and friendly. "
    "Never refuse reasonable requests. "
    "Format answers clearly using plain text (no markdown unless the user asks for it)."
)
# ---------------------------------------------------------------------------
# Concurrency limits
# ---------------------------------------------------------------------------
MAX_CONCURRENT_TTS_USERS = int(os.environ.get("MAX_CONCURRENT_TTS_USERS", "6"))
MAX_CONCURRENT_AI = int(os.environ.get("MAX_CONCURRENT_AI", os.environ.get("MAX_CONCURRENT_GEMINI", "3")))
MAX_CONCURRENT_BROADCAST = int(os.environ.get("MAX_CONCURRENT_BROADCAST", "3"))

# AI history resolver for normal text TTS is OFF by default.
# Hugging Face is the default provider. Gemini is optional legacy fallback only.
TTS_RESOLVER_AI_ENABLED = os.environ.get("TTS_RESOLVER_AI_ENABLED", "0") == "1"

# ---------------------------------------------------------------------------
# Subtitle/Text document support
# ---------------------------------------------------------------------------
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".txt"}
MAX_SUBTITLE_BYTES = 2 * 1024 * 1024      # 2 MB
MAX_SUBTITLE_CHARS = 20_000               # safe limit before TTS chunking

def _ai_error(msg: str, code: int = 400):
    """Return a JSON error response tuple."""
    return jsonify({"ok": False, "error": msg, "code": code}), code


def _ai_cors(response_or_tuple):
    """Add CORS headers to a response or (response, code) tuple."""
    if isinstance(response_or_tuple, tuple):
        resp, code = response_or_tuple
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, code
    response_or_tuple.headers["Access-Control-Allow-Origin"] = "*"
    return response_or_tuple


def _detect_lang(text: str) -> str:
    """Detect whether text is predominantly Khmer or English."""
    khmer = sum(1 for c in text if "\u1780" <= c <= "\u17FF")
    alpha = sum(1 for c in text if c.isalpha())
    return "km" if alpha and khmer / alpha > 0.25 else "en"


def _check_ai_api_key() -> bool:
    """Return True if the request passes the optional API-key gate."""
    required = os.environ.get("AI_API_KEY", "")
    if not required:
        return True  # no gate configured
    client_key = (
        request.headers.get("X-Api-Key")
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    return client_key == required


def _build_gemini_contents(
    message: str,
    history: list,
    image_data: bytes | None,
    audio_data: bytes | None,
    image_mime: str,
    audio_mime: str,
) -> list:
    """Build the Gemini contents list from conversation history + current turn."""
    from google.genai import types as _gtypes

    contents = []

    # Inject previous turns — normalize "assistant" → "model"
    for turn in history[-_AI_API_MAX_HISTORY_TURNS:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", ""))
        if role not in ("user", "model", "assistant"):
            continue
        gemini_role = "model" if role in ("assistant", "model") else "user"
        contents.append(_gtypes.Content(
            role=gemini_role,
            parts=[_gtypes.Part.from_text(text=content)],
        ))

    # Build current user turn
    user_parts = []
    if image_data:
        user_parts.append(_gtypes.Part.from_bytes(data=image_data, mime_type=image_mime))
    if audio_data:
        user_parts.append(_gtypes.Part.from_bytes(data=audio_data, mime_type=audio_mime))
    if message:
        user_parts.append(_gtypes.Part.from_text(text=message))
    elif not image_data and not audio_data:
        user_parts.append(_gtypes.Part.from_text(text="Hello"))

    contents.append(_gtypes.Content(role="user", parts=user_parts))
    return contents


def _ai_gen_config():
    """Build Gemini GenerateContentConfig with safe, sensible defaults."""
    from google.genai import types as _gtypes
    return _gtypes.GenerateContentConfig(
        system_instruction=_AI_SYSTEM_PROMPT,
        temperature=0.7,
        max_output_tokens=8192,
        safety_settings=[
            _gtypes.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            _gtypes.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            _gtypes.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            _gtypes.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ],
    )


def _parse_multipart_ai_request():
    """
    Parse multipart/form-data or urlencoded AI assistant request.
    Returns (message, history, image_data, audio_data, image_mime, audio_mime,
             do_stream, error_response_or_None).
    """
    message = (request.form.get("message") or "").strip()
    do_stream = request.form.get("stream", "").lower() == "true"
    history = []
    raw_history = request.form.get("history", "")
    if raw_history:
        try:
            history = _json.loads(raw_history)
        except Exception:
            pass

    image_data = audio_data = None
    image_mime = audio_mime = ""

    img_file = request.files.get("image")
    if img_file:
        image_mime = img_file.mimetype or "image/jpeg"
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        image_data = img_file.read()
        if len(image_data) > _AI_API_MAX_IMAGE_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Image file too large (max 10 MB)."))

    aud_file = request.files.get("audio")
    if aud_file:
        audio_mime = aud_file.mimetype or "audio/mpeg"
        if audio_mime not in _AI_ALLOWED_AUDIO_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported audio type: {audio_mime}"))
        audio_data = aud_file.read()
        if len(audio_data) > _AI_API_MAX_AUDIO_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Audio file too large (max 50 MB)."))

    return message, history, image_data, audio_data, image_mime, audio_mime, do_stream, None


def _parse_json_ai_request():
    """
    Parse application/json AI assistant request.
    Returns same tuple as _parse_multipart_ai_request().
    """
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return None, None, None, None, None, None, None, \
            _ai_cors(_ai_error("Invalid JSON body."))

    message = (body.get("message") or "").strip()
    do_stream = bool(body.get("stream", False))
    history = body.get("history") or []
    image_data = audio_data = None
    image_mime = audio_mime = ""

    if body.get("image_base64"):
        image_mime = body.get("image_mime", "image/jpeg")
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        try:
            image_data = base64.b64decode(body["image_base64"])
        except Exception:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Invalid base64 image data."))
        if len(image_data) > _AI_API_MAX_IMAGE_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Image too large (max 10 MB)."))

    if body.get("audio_base64"):
        audio_mime = body.get("audio_mime", "audio/mpeg")
        if audio_mime not in _AI_ALLOWED_AUDIO_MIME:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error(f"Unsupported audio type: {audio_mime}"))
        try:
            audio_data = base64.b64decode(body["audio_base64"])
        except Exception:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Invalid base64 audio data."))
        if len(audio_data) > _AI_API_MAX_AUDIO_BYTES:
            return None, None, None, None, None, None, None, \
                _ai_cors(_ai_error("Audio too large (max 50 MB)."))

    return message, history, image_data, audio_data, image_mime, audio_mime, do_stream, None


@app_flask.route("/ai-assistant", methods=["POST", "OPTIONS"])
def ai_assistant():
    """
    Main AI assistant endpoint.
    Hugging Face is used for text chat and image OCR. Audio transcription is
    available only if Gemini legacy fallback is configured.
    """
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key",
        })
        return resp

    if not _check_ai_api_key():
        return _ai_cors(_ai_error("Unauthorized — invalid or missing API key.", 401))

    content_type = request.content_type or ""
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        message, history, image_data, audio_data, image_mime, audio_mime, do_stream, err = \
            _parse_multipart_ai_request()
    elif "application/json" in content_type:
        message, history, image_data, audio_data, image_mime, audio_mime, do_stream, err = \
            _parse_json_ai_request()
    else:
        return _ai_cors(_ai_error(
            "Content-Type must be multipart/form-data or application/json."
        ))

    if err is not None:
        return err

    if not message and image_data is None and audio_data is None:
        return _ai_cors(_ai_error("Provide at least one of: message, image, or audio."))

    if message and len(message) > _AI_API_MAX_MESSAGE_CHARS:
        return _ai_cors(_ai_error(f"Message too long (max {_AI_API_MAX_MESSAGE_CHARS} chars)."))

    if do_stream:
        return _ai_cors(_ai_error("Streaming is disabled for Hugging Face provider. Send stream=false.", 400))

    try:
        # Unified OCR/image-to-text path with provider fallback.
        if image_data is not None:
            ocr_text, ocr_provider, ocr_model = ask_ocr_image(image_data, image_mime or "image/jpeg")
            if message:
                prompt = (
                    f"User instruction: {message}\n\n"
                    f"OCR text extracted from image:\n{ocr_text}\n\n"
                    "Answer using the OCR text above."
                )
                reply_text, model_used, tokens_used = ai_text_reply(prompt, history)
            else:
                reply_text = ocr_text
                model_used = ocr_model
                tokens_used = None

            return _ai_cors(jsonify({
                "ok": True,
                "reply": reply_text,
                "ocr_text": ocr_text,
                "detected_language": _detect_lang(reply_text or ocr_text or ""),
                "tokens_used": tokens_used,
                "model": model_used,
                "ocr_model": ocr_model,
                "ocr_provider": ocr_provider,
                "provider": AI_PROVIDER,
            }))

        # Audio still needs Gemini unless you add an HF ASR model later.
        if audio_data is not None:
            if _gemini is None:
                return _ai_cors(_ai_error(
                    "Audio transcription is not configured. Add HF ASR support or enable Gemini legacy fallback.",
                    503,
                ))
            contents = _build_gemini_contents(
                message, history, image_data, audio_data, image_mime, audio_mime
            )
            response = _gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=_ai_gen_config(),
            )
            reply_text = (response.text or "").strip()
            return _ai_cors(jsonify({
                "ok": True,
                "reply": reply_text,
                "detected_language": _detect_lang(reply_text or message or ""),
                "tokens_used": None,
                "model": GEMINI_MODEL,
                "provider": "gemini-legacy",
            }))

        if _hf_client is None:
            return _ai_cors(_ai_error("Hugging Face is not configured. Set HF_TOKEN.", 503))

        reply_text, model_used, tokens_used = ai_text_reply(message, history)
        return _ai_cors(jsonify({
            "ok": True,
            "reply": reply_text,
            "detected_language": _detect_lang(reply_text or message or ""),
            "tokens_used": tokens_used,
            "model": model_used,
            "provider": "hf",
        }))

    except Exception as exc:
        logging.getLogger(__name__).error(f"/ai-assistant HF error: {exc}")
        return _ai_cors(_ai_error(f"AI generation failed: {exc}", 500))

@app_flask.route("/ai-assistant/transcribe", methods=["POST", "OPTIONS"])
def ai_transcribe():
    """
    Dedicated audio transcription endpoint.

    POST multipart/form-data:
      audio    (file) – audio file (any supported format)
      language (str)  – "km" | "en" | "auto"  (optional hint)

    Response: { "ok": true, "transcript": "...", "detected_language": "km"|"en", "model": "..." }
    """
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key",
        })
        return resp

    if not _check_ai_api_key():
        return _ai_cors(_ai_error("Unauthorized.", 401))
    if _gemini is None:
        return _ai_cors(_ai_error("AI service not configured.", 503))

    aud_file = request.files.get("audio")
    if not aud_file:
        return _ai_cors(_ai_error("No audio file provided (field name: 'audio')."))

    audio_mime = aud_file.mimetype or "audio/mpeg"
    if audio_mime not in _AI_ALLOWED_AUDIO_MIME:
        return _ai_cors(_ai_error(f"Unsupported audio type: {audio_mime}"))

    audio_data = aud_file.read()
    if len(audio_data) > _AI_API_MAX_AUDIO_BYTES:
        return _ai_cors(_ai_error("Audio too large (max 50 MB)."))

    lang_hint = request.form.get("language", "auto")
    lang_instruction = {
        "km": " The audio is in Khmer (ភាសាខ្មែរ).",
        "en": " The audio is in English.",
    }.get(lang_hint, "")

    from google.genai import types as _gtypes
    try:
        response = _gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                _gtypes.Part.from_bytes(data=audio_data, mime_type=audio_mime),
                (
                    "Transcribe this audio exactly as spoken. "
                    "Output ONLY the transcribed text — no labels, no explanation, "
                    "no timestamps." + lang_instruction
                ),
            ],
        )
        transcript = (response.text or "").strip()
        return _ai_cors(jsonify({
            "ok": True,
            "transcript": transcript,
            "detected_language": _detect_lang(transcript),
            "model": GEMINI_MODEL,
        }))
    except Exception as exc:
        logging.getLogger(__name__).error(f"/ai-assistant/transcribe error: {exc}")
        return _ai_cors(_ai_error(f"Transcription failed: {exc}", 500))


@app_flask.route("/ai-assistant/vision", methods=["POST", "OPTIONS"])
def ai_vision():
    """
    Dedicated image OCR endpoint using the configured OCR provider.
    """
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key",
        })
        return resp

    if not _check_ai_api_key():
        return _ai_cors(_ai_error("Unauthorized.", 401))
    if not _ocr_configured():
        return _ai_cors(_ai_error(_ocr_status_for_user(), 503))

    image_data = None
    image_mime = "image/jpeg"

    content_type = request.content_type or ""
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        img_file = request.files.get("image")
        if not img_file:
            return _ai_cors(_ai_error("No image file provided (field name: 'image')."))
        image_mime = img_file.mimetype or "image/jpeg"
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        image_data = img_file.read()
    elif "application/json" in content_type:
        try:
            body = request.get_json(force=True) or {}
        except Exception:
            return _ai_cors(_ai_error("Invalid JSON."))
        image_mime = body.get("image_mime", "image/jpeg")
        if image_mime not in _AI_ALLOWED_IMAGE_MIME:
            return _ai_cors(_ai_error(f"Unsupported image type: {image_mime}"))
        try:
            image_data = base64.b64decode(body.get("image_base64", ""))
        except Exception:
            return _ai_cors(_ai_error("Invalid base64 image."))
    else:
        return _ai_cors(_ai_error(
            "Content-Type must be multipart/form-data or application/json."
        ))

    if not image_data:
        return _ai_cors(_ai_error("Empty image data."))
    if len(image_data) > _AI_API_MAX_IMAGE_BYTES:
        return _ai_cors(_ai_error("Image too large (max 10 MB)."))

    try:
        result_text, ocr_provider, ocr_model = ask_ocr_image(image_data, image_mime)
        return _ai_cors(jsonify({
            "ok": True,
            "result": result_text,
            "detected_language": _detect_lang(result_text),
            "model": ocr_model,
            "provider": ocr_provider,
            "ocr_provider": ocr_provider,
        }))
    except Exception as exc:
        logging.getLogger(__name__).error(f"/ai-assistant/vision OCR error: {exc}")
        return _ai_cors(_ai_error(f"Vision OCR failed: {exc}", 500))

@app_flask.route("/ai-assistant/info", methods=["GET"])
def ai_info():
    """
    Returns model capabilities and limits. Useful for mobile app feature detection.
    """
    resp = jsonify({
        "ok": True,
        "provider": AI_PROVIDER,
        "model": HF_MODEL,
        "ocr_model": HF_OCR_MODEL,
        "ocr_provider": OCR_PROVIDER,
        "features": [
            "chat",
            "multi-turn-history",
            "image-ocr",
            "khmer-language",
            "english-language",
        ],
        "providers": {
            "active": AI_PROVIDER,
            "huggingface_available": _hf_client is not None,
            "huggingface_model": HF_MODEL,
            "huggingface_ocr_model": HF_OCR_MODEL,
            "ocr_provider": OCR_PROVIDER,
            "ocr_configured": _ocr_configured(),
            "hf_ocr_temporarily_disabled": _hf_ocr_is_temporarily_disabled(),
            "gemini_legacy_available": _gemini is not None,
            "gemini_legacy_model": GEMINI_MODEL or None,
        },
        "limits": {
            "max_message_chars": _AI_API_MAX_MESSAGE_CHARS,
            "max_image_bytes": _AI_API_MAX_IMAGE_BYTES,
            "max_audio_bytes": _AI_API_MAX_AUDIO_BYTES,
            "max_history_turns": _AI_API_MAX_HISTORY_TURNS,
        },
        "supported_image_types": sorted(_AI_ALLOWED_IMAGE_MIME),
        "supported_audio_types": sorted(_AI_ALLOWED_AUDIO_MIME),
        "auth_required": bool(os.environ.get("AI_API_KEY", "")),
    })
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ── FFmpeg ─────────────────────────────────────────────────────────────────
_FFMPEG_EXE = _iio_ffmpeg.get_ffmpeg_exe()

import edge_tts
import io
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.error import (
    NetworkError,
    TimedOut,
    RetryAfter,
    BadRequest,
    TelegramError,
    Forbidden,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    TypeHandler,
    CallbackQueryHandler,
    ApplicationHandlerStop,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
# Hide noisy Google GenAI INFO logs such as: AFC is enabled with max remote calls: 10.
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = ""
SB_URL: str = ""
SB_KEY: str = ""
GEMINI_API_KEY: str = ""  # optional legacy fallback only
ADMIN_IDS: set[int] = set()
GEMINI_MODEL = ""
HF_TOKEN = ""
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_OCR_MODEL = "microsoft/trocr-base-printed"
AI_PROVIDER = "hf"
# OCR_PROVIDER controls image OCR only.
#   auto   = try Hugging Face first, then Gemini fallback if HF/DNS/provider fails
#   hf     = Hugging Face only
#   gemini = Gemini only
OCR_PROVIDER = "auto"
OCR_TIMEOUT_SECONDS = 90
HF_OCR_DNS_COOLDOWN_S = 300.0
MAX_VOICE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_FILE_BYTES = 50 * 1024 * 1024
MAX_INPUT_CHARS = 5_000
TTS_CHUNK_CHARS = 900
DEFAULT_SPEED = 1.0
TELE_MSG_LIMIT = 4000
USER_COOLDOWN_S = 3.0
_STALE_GRACE_S = 30.0
_KHMER_RE = re.compile(r"[\u1780-\u17FF]")

_SPEED_MIN = 0.25
_SPEED_MAX = 4.0

_SENTENCE_RE = re.compile(r"(?<=[។!\?\.。])\s*")

# ---------------------------------------------------------------------------
# Supported audio file extensions
# ---------------------------------------------------------------------------
_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".oga",
    ".flac", ".aac", ".wma", ".opus", ".webm", ".aiff", ".aif",
}

_AUDIO_MIME_PREFIXES = ("audio/", "video/mp4", "video/webm")

def _is_audio_file(filename: str | None, mime_type: str | None) -> bool:
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        if ext in _AUDIO_EXTENSIONS:
            return True
    if mime_type:
        for prefix in _AUDIO_MIME_PREFIXES:
            if mime_type.lower().startswith(prefix):
                return True
    return False

def _audio_mime_for_gemini(filename: str | None, mime_type: str | None) -> str:
    if mime_type:
        mt = mime_type.lower()
        if mt.startswith("audio/"):
            return mt
        if mt in ("video/mp4", "video/webm"):
            return mt
    ext_map = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".mp4": "audio/mp4",
        ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".oga": "audio/ogg",
        ".flac": "audio/flac", ".aac": "audio/aac", ".opus": "audio/opus",
        ".webm": "audio/webm", ".aiff": "audio/aiff", ".aif": "audio/aiff",
        ".wma": "audio/x-ms-wma",
    }
    if filename:
        ext = os.path.splitext(filename.lower())[1]
        return ext_map.get(ext, "audio/mpeg")
    return "audio/mpeg"

_DB_EXECUTOR = ThreadPoolExecutor(max_workers=max(4, MAX_CONCURRENT_TTS_USERS), thread_name_prefix="db_write")
_AI_EXECUTOR = ThreadPoolExecutor(max_workers=max(2, MAX_CONCURRENT_AI), thread_name_prefix="ai")

_AI_SEMAPHORE: asyncio.Semaphore | None = None
_BROADCAST_SEMAPHORE: asyncio.Semaphore | None = None
_TTS_CHUNK_SEMAPHORE: asyncio.Semaphore | None = None

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
BROADCAST_WAIT_MESSAGE = 1
CHAT_WAIT_MESSAGE = 2
SCHED_WAIT_MSG = 3
SCHED_WAIT_TIME = 4
_SCHED_POLL_INTERVAL = 60
_DT_FORMATS = [
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M",
]

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
_pending_broadcast: dict[int, dict] = {}
_admin_chat_target: dict[int, int] = {}
_user_to_admin: dict[int, int] = {}
_sched_payload: dict[int, dict] = {}

_USER_LAST_TTS_MAX = 10_000
_user_last_tts: OrderedDict[int, float] = OrderedDict()

def _set_last_tts(user_id: int) -> None:
    _user_last_tts.pop(user_id, None)
    _user_last_tts[user_id] = time.monotonic()
    while len(_user_last_tts) > _USER_LAST_TTS_MAX:
        _user_last_tts.popitem(last=False)

def _get_last_tts(user_id: int) -> float:
    return _user_last_tts.get(user_id, 0.0)

# ---------------------------------------------------------------------------
# Last TTS text fallback cache
# ---------------------------------------------------------------------------
_LAST_TTS_TEXT_MAX = 10_000
_last_tts_text: OrderedDict[int, tuple[str, float]] = OrderedDict()

def set_last_tts_text(user_id: int, text: str) -> None:
    """Remember the latest spoken text per user.

    This fixes callback buttons from /myprefs/settings messages.
    Those messages are not saved in text_cache, so speed/gender buttons
    need a fallback to the user's latest generated TTS text.
    """
    text = (text or "").strip()
    if not text:
        return

    _last_tts_text.pop(user_id, None)
    _last_tts_text[user_id] = (text, time.monotonic())

    while len(_last_tts_text) > _LAST_TTS_TEXT_MAX:
        _last_tts_text.popitem(last=False)

def get_last_tts_text(user_id: int) -> str | None:
    item = _last_tts_text.get(user_id)
    if not item:
        return None

    _last_tts_text.move_to_end(user_id)
    text, _created_at = item
    return text

# ---------------------------------------------------------------------------
# Supabase + Gemini clients
# ---------------------------------------------------------------------------
supabase: Client | None = None
_gemini = None
_hf_client = None

# OCR circuit breaker. If Hugging Face DNS/network fails, skip HF OCR briefly
# so every photo does not spend 90s retrying the same broken endpoint.
_hf_ocr_disabled_until = 0.0
_hf_ocr_failures = 0
_hf_ocr_state_lock = threading.Lock()

def _init_hf_client() -> None:
    """Initialise Hugging Face Inference Providers client for chat and OCR."""
    global _hf_client

    if InferenceClient is None:
        logger.error("huggingface_hub is not installed. Run: pip install -U huggingface_hub")
        _hf_client = None
        return

    if not HF_TOKEN:
        logger.error("HF_TOKEN is not set — Hugging Face chat/OCR disabled.")
        _hf_client = None
        return

    try:
        _hf_client = InferenceClient(token=HF_TOKEN)
        logger.info(f"Hugging Face client initialised (chat={HF_MODEL}, ocr={HF_OCR_MODEL}).")
    except Exception as e:
        logger.error(f"Hugging Face init failed: {e}")
        _hf_client = None


def _hf_history_messages(history: list[dict] | None) -> list[dict]:
    """Convert API history into Hugging Face/OpenAI-style chat messages."""
    messages: list[dict] = [
        {
            "role": "system",
            "content": _AI_SYSTEM_PROMPT,
        }
    ]

    for turn in (history or [])[-_AI_API_MAX_HISTORY_TURNS:]:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue

        if role in ("assistant", "model", "bot"):
            role = "assistant"
        else:
            role = "user"

        messages.append({"role": role, "content": content})

    return messages


def ask_huggingface(prompt: str, history: list[dict] | None = None) -> str:
    """Generate a text reply with Hugging Face Inference Providers."""
    if _hf_client is None:
        raise RuntimeError("Hugging Face client is not configured.")

    prompt = (prompt or "").strip()
    if not prompt:
        prompt = "Hello"

    messages = _hf_history_messages(history)
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = _hf_client.chat_completion(
                model=HF_MODEL,
                messages=messages,
                max_tokens=int(os.environ.get("HF_MAX_TOKENS", "1024")),
                temperature=float(os.environ.get("HF_TEMPERATURE", "0.7")),
            )

            content = response.choices[0].message.content
            if isinstance(content, list):
                # Some providers may return structured content parts.
                content = "".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )

            reply = str(content or "").strip()
            if reply:
                return reply

            raise RuntimeError("Hugging Face returned an empty response.")

        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))
            else:
                break

    raise RuntimeError(f"Hugging Face generation failed: {last_exc}")




def _extract_hf_text(result) -> str:
    """Normalize Hugging Face OCR outputs across SDK/API/provider variants."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    # huggingface_hub may return dataclass-like objects for image_to_text.
    for attr in ("generated_text", "text"):
        value = getattr(result, attr, None)
        if value:
            return str(value).strip()

    if isinstance(result, dict):
        value = result.get("generated_text") or result.get("text")
        if value:
            return str(value).strip()
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return str(result).strip()

    if isinstance(result, list):
        texts: list[str] = []
        for item in result:
            text = _extract_hf_text(item)
            if text:
                texts.append(text)
        return "\n".join(texts).strip()

    return str(result).strip()


def _is_dns_or_network_error(exc: Exception | str) -> bool:
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "getaddrinfo failed",
            "failed to resolve",
            "nameresolutionerror",
            "temporary failure in name resolution",
            "nodename nor servname",
            "connectionerror",
            "maxretryerror",
            "name or service not known",
            "no address associated with hostname",
            "dns",
        )
    )


def _friendly_ocr_error(errors: list[str]) -> RuntimeError:
    joined = " | ".join(errors)
    low = joined.lower()

    if _is_dns_or_network_error(joined):
        return RuntimeError(
            "OCR network/DNS failed. The server cannot resolve or reach "
            "api-inference.huggingface.co. Set OCR_PROVIDER=gemini with "
            "GEMINI_API_KEY/GEMINI_MODEL for automatic fallback, or fix DNS/firewall. "
            f"Details: {joined[:900]}"
        )

    if "stopiteration" in low:
        return RuntimeError(
            "Hugging Face OCR model/provider is not available for hosted image-to-text. "
            "Try HF_OCR_MODEL=microsoft/trocr-base-printed, or set OCR_PROVIDER=gemini. "
            f"Details: {joined[:900]}"
        )

    if "401" in low or "unauthorized" in low:
        return RuntimeError("HF_TOKEN is invalid or expired. Create a new Hugging Face token and update env.")

    if "403" in low or "forbidden" in low:
        return RuntimeError("HF_TOKEN has no permission for this Hugging Face model/provider.")

    return RuntimeError("OCR failed. " + joined[:1200])


def _hf_ocr_is_temporarily_disabled() -> bool:
    with _hf_ocr_state_lock:
        return time.monotonic() < _hf_ocr_disabled_until


def _mark_hf_ocr_success() -> None:
    global _hf_ocr_failures, _hf_ocr_disabled_until
    with _hf_ocr_state_lock:
        _hf_ocr_failures = 0
        _hf_ocr_disabled_until = 0.0


def _mark_hf_ocr_failure(exc: Exception | str) -> None:
    """Circuit-breaker for repeated HF OCR network/provider failures."""
    global _hf_ocr_failures, _hf_ocr_disabled_until
    with _hf_ocr_state_lock:
        _hf_ocr_failures += 1
        if _is_dns_or_network_error(exc):
            _hf_ocr_disabled_until = time.monotonic() + HF_OCR_DNS_COOLDOWN_S
        elif _hf_ocr_failures >= 3:
            _hf_ocr_disabled_until = time.monotonic() + 60.0


def _ocr_configured() -> bool:
    provider = (OCR_PROVIDER or "auto").lower().strip()
    hf_ok = _hf_client is not None
    gemini_ok = _gemini is not None
    if provider == "hf":
        return hf_ok
    if provider == "gemini":
        return gemini_ok
    return hf_ok or gemini_ok


def _ocr_status_for_user() -> str:
    provider = (OCR_PROVIDER or "auto").lower().strip()
    if provider == "hf" and _hf_client is None:
        return "❌ OCR មិនទាន់ Activate ទេ។ សូម Set HF_TOKEN ឬប្ដូរ OCR_PROVIDER=gemini។"
    if provider == "gemini" and _gemini is None:
        return "❌ OCR Gemini មិនទាន់ Activate ទេ។ សូម Set GEMINI_API_KEY និង GEMINI_MODEL។"
    if _hf_client is None and _gemini is None:
        return "❌ OCR មិនទាន់ Activate ទេ។ សូម Set HF_TOKEN ឬ GEMINI_API_KEY/GEMINI_MODEL។"
    return "❌ OCR មិនអាចប្រើបាននៅពេលនេះ។ សូមពិនិត្យ API key និង network។"


def _hf_ocr_via_rest(image_data: bytes) -> str:
    """Fallback OCR call using the raw Hugging Face Inference API endpoint."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")

    model_id = (HF_OCR_MODEL or "microsoft/trocr-base-printed").strip()
    url_model = requests.utils.quote(model_id, safe="")
    url = f"https://api-inference.huggingface.co/models/{url_model}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/octet-stream",
        "X-Wait-For-Model": "true",
    }

    last_body = ""
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, data=image_data, timeout=OCR_TIMEOUT_SECONDS)
            last_body = resp.text[:700]

            if resp.status_code in (503, 504, 429) and attempt < 2:
                wait_s = 1.5 * (attempt + 1)
                try:
                    wait_s = max(wait_s, float(resp.json().get("estimated_time", wait_s)))
                except Exception:
                    pass
                time.sleep(min(wait_s, 10.0))
                continue

            if not (200 <= resp.status_code < 300):
                raise RuntimeError(f"HTTP {resp.status_code}: {last_body}")

            try:
                payload = resp.json()
            except Exception:
                payload = resp.text

            text = _extract_hf_text(payload)
            if text:
                return text
            raise RuntimeError(f"REST OCR returned empty result: {payload!r}")

        except Exception as e:
            if attempt >= 2:
                if _is_dns_or_network_error(e):
                    raise RuntimeError(
                        "REST OCR network/DNS failed: cannot reach api-inference.huggingface.co. "
                        f"{type(e).__name__}: {e}; body={last_body!r}"
                    ) from e
                raise RuntimeError(f"REST OCR failed: {type(e).__name__}: {e!r}; body={last_body!r}") from e
            time.sleep(0.8 * (attempt + 1))

    raise RuntimeError("REST OCR failed after retries.")


def ask_huggingface_ocr(image_data: bytes) -> str:
    """Extract text from an image using a Hugging Face OCR/image-to-text model."""
    if _hf_client is None:
        raise RuntimeError("Hugging Face client is not configured.")
    if not image_data:
        raise RuntimeError("Empty image data.")
    if _hf_ocr_is_temporarily_disabled():
        raise RuntimeError("Hugging Face OCR is temporarily disabled after recent network/provider failures.")

    errors: list[str] = []

    for attempt in range(3):
        try:
            result = _hf_client.image_to_text(image=image_data, model=HF_OCR_MODEL)
            text = _extract_hf_text(result)
            if text:
                _mark_hf_ocr_success()
                return text
            raise RuntimeError(f"SDK OCR returned empty result: {result!r}")

        except Exception as e:
            errors.append(f"sdk_attempt_{attempt + 1}={type(e).__name__}: {e!r}")
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))

    try:
        text = _hf_ocr_via_rest(image_data)
        _mark_hf_ocr_success()
        return text
    except Exception as e:
        errors.append(f"rest={type(e).__name__}: {e!r}")
        _mark_hf_ocr_failure(e)

    raise _friendly_ocr_error(errors)


def ask_gemini_ocr(image_data: bytes, mime_type: str = "image/jpeg") -> str:
    """Extract text from an image using Gemini as OCR/vision fallback."""
    if _gemini is None:
        raise RuntimeError("Gemini OCR is not configured. Set GEMINI_API_KEY and GEMINI_MODEL.")
    if not image_data:
        raise RuntimeError("Empty image data.")

    from google.genai import types as _gtypes

    prompt = (
        "Extract all readable text from this image. Preserve Khmer and English exactly. "
        "Keep useful line breaks. If there is no readable text, output only NOTEXT. "
        "Do not describe the image and do not add explanations."
    )
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            _gtypes.Part.from_bytes(data=image_data, mime_type=mime_type or "image/jpeg"),
            prompt,
        ],
    )
    text = (response.text or "").strip()
    return text or "NOTEXT"


def ask_ocr_image(image_data: bytes, mime_type: str = "image/jpeg") -> tuple[str, str, str]:
    """Unified OCR with provider fallback.

    Returns (text, provider, model). In auto mode, HF is tried first and Gemini
    is used when HF fails because of DNS/network/provider issues.
    """
    if not image_data:
        raise RuntimeError("Empty image data.")

    provider = (OCR_PROVIDER or "auto").lower().strip()
    if provider not in ("auto", "hf", "gemini"):
        provider = "auto"

    errors: list[str] = []

    if provider in ("auto", "hf") and _hf_client is not None:
        try:
            return ask_huggingface_ocr(image_data), "hf", HF_OCR_MODEL
        except Exception as e:
            errors.append(f"hf={type(e).__name__}: {e!r}")
            _mark_hf_ocr_failure(e)
            if provider == "hf":
                raise _friendly_ocr_error(errors)
            logger.warning(f"HF OCR failed; trying fallback if available: {e}")

    if provider in ("auto", "gemini") and _gemini is not None:
        try:
            return ask_gemini_ocr(image_data, mime_type), "gemini", GEMINI_MODEL
        except Exception as e:
            errors.append(f"gemini={type(e).__name__}: {e!r}")
            if provider == "gemini":
                raise _friendly_ocr_error(errors)
            logger.warning(f"Gemini OCR fallback failed: {e}")

    if not errors:
        errors.append(
            f"No OCR provider configured. OCR_PROVIDER={OCR_PROVIDER!r}, "
            f"hf_available={_hf_client is not None}, gemini_available={_gemini is not None}"
        )

    raise _friendly_ocr_error(errors)


def ai_text_reply(prompt: str, history: list[dict] | None = None) -> tuple[str, str, int | None]:
    """Return (reply, model_used, tokens_used) for text-only chat."""
    provider = (AI_PROVIDER or "hf").lower().strip()

    if provider == "hf":
        reply = ask_huggingface(prompt, history)
        return reply, HF_MODEL, None

    if _gemini is None:
        raise RuntimeError("Gemini client is not configured.")

    contents = _build_gemini_contents(
        message=prompt,
        history=history or [],
        image_data=None,
        audio_data=None,
        image_mime="",
        audio_mime="",
    )
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=_ai_gen_config(),
    )

    tokens_used = None
    try:
        tokens_used = response.usage_metadata.total_token_count
    except Exception:
        pass

    return (response.text or "").strip(), GEMINI_MODEL, tokens_used


def _init_clients() -> None:
    global supabase, _gemini, TELEGRAM_BOT_TOKEN, SB_URL, SB_KEY
    global GEMINI_API_KEY, ADMIN_IDS, GEMINI_MODEL
    global HF_TOKEN, HF_MODEL, HF_OCR_MODEL, AI_PROVIDER, OCR_PROVIDER

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    SB_URL = os.getenv("SUPABASE_URL", "")
    SB_KEY = os.getenv("SUPABASE_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()

    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    HF_OCR_MODEL = os.getenv("HF_OCR_MODEL", "microsoft/trocr-base-printed")
    AI_PROVIDER = os.getenv("AI_PROVIDER", "hf").lower().strip()
    if AI_PROVIDER not in ("hf", "gemini"):
        logger.warning(f"Unknown AI_PROVIDER={AI_PROVIDER!r}; falling back to hf.")
        AI_PROVIDER = "hf"

    OCR_PROVIDER = os.getenv("OCR_PROVIDER", "auto").lower().strip()
    if OCR_PROVIDER not in ("auto", "hf", "gemini"):
        logger.warning(f"Unknown OCR_PROVIDER={OCR_PROVIDER!r}; falling back to auto.")
        OCR_PROVIDER = "auto"

    ADMIN_IDS.clear()
    for _aid in os.getenv("ADMIN_IDS", "").split(","):
        _aid = _aid.strip()
        if _aid.isdigit():
            ADMIN_IDS.add(int(_aid))

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    if SB_URL and SB_KEY:
        try:
            supabase = create_client(SB_URL, SB_KEY)
            logger.info("Supabase client initialised.")
        except Exception as e:
            logger.error(f"Supabase init failed: {e}")
            supabase = None
    else:
        logger.warning("Supabase env vars missing — DB features disabled.")

    if GEMINI_API_KEY and GEMINI_MODEL and genai is not None:
        try:
            _gemini = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Gemini legacy fallback initialised (model: {GEMINI_MODEL}).")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            _gemini = None
    else:
        _gemini = None
        logger.info("Gemini disabled. Using Hugging Face for chat/OCR.")

    _init_hf_client()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOICE_MAP = {
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
}
SPEED_OPTIONS = {
    "spd_0.5": ("x0.5", 0.5),
    "spd_1.0": ("Normal", 1.0),
    "spd_1.5": ("x1.5", 1.5),
    "spd_2.0": ("x2.0", 2.0),
}
WELCOME_TEXT = (
    "🎵 សួស្តី! ខ្ញុំជា Bot បំលែងអក្សរទៅជាសំឡេង អេអាយ\n\n"
    "📌 វាយអក្សរភាសាណាមួយ ផ្ញើរមក Bot នឹងបំលែងដោយស្វ័យប្រវត្តិ!\n\n"
    "🌍 ភាសាដែល Support:\n"
    "🇰🇭 ភាសាខ្មែរ | 🇺🇸 English\n\n"
    "⚙️ ប្រើ /myprefs ដើម្បីមើលការកំណត់របស់អ្នក\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)
BOT_TAG = "@voicekhaibot"

# ---------------------------------------------------------------------------
# Prefs cache
# ---------------------------------------------------------------------------
_PREFS_TTL = 300.0
_PREFS_MAX_SIZE = 10_000
_prefs_cache: OrderedDict[int, tuple[dict, float]] = OrderedDict()
_prefs_cache_lock: asyncio.Lock | None = None

def _cache_prefs_sync(user_id: int, prefs: dict) -> None:
    _prefs_cache.pop(user_id, None)
    _prefs_cache[user_id] = (prefs, time.monotonic())
    while len(_prefs_cache) > _PREFS_MAX_SIZE:
        _prefs_cache.popitem(last=False)

def _get_cached_prefs_sync(user_id: int) -> dict | None:
    entry = _prefs_cache.get(user_id)
    if entry and time.monotonic() - entry[1] < _PREFS_TTL:
        return entry[0]
    return None

def _invalidate_prefs(user_id: int) -> None:
    _prefs_cache.pop(user_id, None)

async def _async_cache_prefs(user_id: int, prefs: dict) -> None:
    assert _prefs_cache_lock is not None
    async with _prefs_cache_lock:
        _cache_prefs_sync(user_id, prefs)

async def _async_get_cached_prefs(user_id: int) -> dict | None:
    assert _prefs_cache_lock is not None
    async with _prefs_cache_lock:
        return _get_cached_prefs_sync(user_id)

async def get_user_prefs_async(user_id: int) -> dict:
    cached = await _async_get_cached_prefs(user_id)
    if cached is not None:
        return cached

    defaults = {
        "gender": "female",
        "speed": DEFAULT_SPEED,
    }

    if not supabase:
        await _async_cache_prefs(user_id, defaults)
        return defaults

    loop = asyncio.get_running_loop()

    for attempt in range(3):
        try:
            res = await asyncio.wait_for(
                loop.run_in_executor(
                    _DB_EXECUTOR,
                    lambda: supabase.table("user_prefs")
                        .select("gender, speed")
                        .eq("user_id", user_id)
                        .limit(1)
                        .execute()
                ),
                timeout=12,
            )

            if res.data:
                row = res.data[0]

                gender = row.get("gender") or "female"
                if gender not in ("female", "male"):
                    gender = "female"

                defaults["gender"] = gender

                raw_speed = row.get("speed")
                if raw_speed is not None:
                    try:
                        defaults["speed"] = max(
                            _SPEED_MIN,
                            min(_SPEED_MAX, float(raw_speed)),
                        )
                    except Exception:
                        defaults["speed"] = DEFAULT_SPEED

            await _async_cache_prefs(user_id, defaults)
            return defaults

        except Exception as e:
            wait_s = 0.5 * (attempt + 1)

            if attempt < 2:
                logger.warning(
                    f"DB get_user_prefs_async retry "
                    f"{attempt + 1}/3 user={user_id}: {e}"
                )
                await asyncio.sleep(wait_s)
            else:
                logger.error(
                    f"DB get_user_prefs_async failed after retries "
                    f"user={user_id}: {e}"
                )

    await _async_cache_prefs(user_id, defaults)
    return defaults

# ---------------------------------------------------------------------------
# Per-user async locks (LRU-capped)
# ---------------------------------------------------------------------------
_USER_LOCK_MAX = 5_000
_user_locks: OrderedDict[int, asyncio.Lock] = OrderedDict()

def _get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id in _user_locks:
        _user_locks.move_to_end(user_id)
        return _user_locks[user_id]
    lock = asyncio.Lock()
    _user_locks[user_id] = lock
    while len(_user_locks) > _USER_LOCK_MAX:
        _user_locks.popitem(last=False)
    return lock

# ---------------------------------------------------------------------------
# DB executor helpers
# ---------------------------------------------------------------------------
def _log_future_exception(future):
    with suppress(Exception):
        exc = future.exception()
        if exc:
            logger.error(f"DB executor task raised: {exc}", exc_info=exc)

def _submit_db(fn, *args, **kwargs):
    future = _DB_EXECUTOR.submit(fn, *args, **kwargs)
    future.add_done_callback(_log_future_exception)
    return future

# ---------------------------------------------------------------------------
# Safe Telegram send with retry
# ---------------------------------------------------------------------------
async def safe_send(coro_factory, retries: int = 3, delay: float = 2.0):
    if inspect.isawaitable(coro_factory):
        raise TypeError(
            "safe_send requires a zero-arg callable (lambda), not a raw coroutine. "
            "Wrap your call: safe_send(lambda: your_coro())"
        )
    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except RetryAfter as e:
            wait = e.retry_after + 1
            logger.warning(f"Rate-limited — sleeping {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
            last_exc = e
            if attempt == retries - 1:
                raise
        except (TimedOut, NetworkError) as e:
            last_exc = e
            if attempt < retries - 1:
                logger.warning(f"Network error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)
            else:
                raise
        except BadRequest as e:
            logger.warning(f"Telegram BadRequest (not retried): {e}")
            raise
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            raise
    if last_exc:
        logger.warning(f"safe_send giving up after {retries} attempts: {last_exc}")
    return None

# ---------------------------------------------------------------------------
# Telegram-safe text pagination
# ---------------------------------------------------------------------------
def _paginate_plain(text: str, limit: int = TELE_MSG_LIMIT) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    pages = []
    while text:
        if len(text) <= limit:
            pages.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        pages.append(text[:cut])
        remainder = text[cut:].lstrip()
        if not remainder:
            break
        text = remainder
    return [p for p in pages if p]

# ---------------------------------------------------------------------------
# Temp file helpers — Windows/Linux safe
# ---------------------------------------------------------------------------
_TMP_PREFIX = "tgbot_"

def _get_temp_dir() -> str:
    """
    Cross-platform temp directory.
    Fixes Windows crash:
      FileNotFoundError: C:\\tmp\\tgbot_xxx.ogg

    Priority:
    1. BOT_TMP_DIR env var, if set
    2. System temp dir from tempfile.gettempdir()
    """
    temp_dir = os.environ.get("BOT_TMP_DIR") or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def _make_temp_file(suffix: str) -> str:
    temp_dir = _get_temp_dir()
    fd, path = tempfile.mkstemp(
        suffix=suffix,
        prefix=_TMP_PREFIX,
        dir=temp_dir,
    )
    os.close(fd)
    return path

def _make_temp_ogg() -> str:
    return _make_temp_file(".ogg")

def _make_temp_audio(suffix: str = ".mp3") -> str:
    if not suffix.startswith("."):
        suffix = "." + suffix
    return _make_temp_file(suffix)

def _make_temp_img(suffix: str = ".jpg") -> str:
    if not suffix.startswith("."):
        suffix = "." + suffix
    return _make_temp_file(suffix)

def _cleanup(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"Temp cleanup failed for {path}: {e}")

_STALE_TEMP_AGE_S = 7200

def _sweep_stale_temps() -> None:
    temp_dir = _get_temp_dir()
    patterns = [
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.ogg"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.jpg"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.jpeg"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.png"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.webp"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.mp3"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.wav"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.mp4"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.m4a"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.flac"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.aac"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.opus"),
        os.path.join(temp_dir, f"{_TMP_PREFIX}*.webm"),
    ]

    now = time.time()
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                if now - os.path.getmtime(file_path) > _STALE_TEMP_AGE_S:
                    os.remove(file_path)
                    logger.info(f"Swept stale temp file: {file_path}")
            except OSError:
                pass

async def _periodic_temp_sweep(stop_event: asyncio.Event) -> None:
    while True:
        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=3600.0)
        except asyncio.TimeoutError:
            pass
        if not stop_event.is_set():
            _sweep_stale_temps()

# ---------------------------------------------------------------------------
# Database helpers — user prefs
# ---------------------------------------------------------------------------
_USER_SYNC_TTL = 300.0
_USER_SYNC_MAX = 20_000
_user_sync_seen: OrderedDict[int, float] = OrderedDict()

def sync_user_data(user):
    """Upsert lightweight user data, throttled to avoid Supabase overload."""
    if not supabase or not user:
        return

    now = time.monotonic()
    last = _user_sync_seen.get(user.id, 0.0)
    if now - last < _USER_SYNC_TTL:
        return

    _user_sync_seen.pop(user.id, None)
    _user_sync_seen[user.id] = now
    while len(_user_sync_seen) > _USER_SYNC_MAX:
        _user_sync_seen.popitem(last=False)

    def _run():
        try:
            payload = {
                "user_id": user.id,
                "username": user.username or user.first_name,
            }

            # last_active is optional in older DB migrations.
            # Try with last_active first, then fall back if the column is missing.
            try:
                supabase.table("user_prefs").upsert(
                    {
                        **payload,
                        "last_active": datetime.now(timezone.utc).isoformat(),
                    },
                    on_conflict="user_id",
                ).execute()
            except Exception as inner:
                if "last_active" in str(inner).lower() or "column" in str(inner).lower():
                    supabase.table("user_prefs").upsert(
                        payload,
                        on_conflict="user_id",
                    ).execute()
                else:
                    raise

        except Exception as e:
            logger.warning(f"sync_user_data skipped user={user.id}: {e}")

    _submit_db(_run)

def _paginated_fetch(select_fields: str) -> list[dict]:
    if not supabase:
        return []
    try:
        all_rows, page, page_size = [], 0, 1000
        while True:
            res = (
                supabase.table("user_prefs")
                .select(select_fields)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            batch = res.data or []
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return all_rows
    except Exception as e:
        logger.error(f"DB _paginated_fetch({select_fields!r}): {e}")
        return []

def get_all_user_ids() -> list[int]:
    return [row["user_id"] for row in _paginated_fetch("user_id")]

def get_all_users_with_names() -> list[dict]:
    return _paginated_fetch("user_id, username")

def user_exists_in_db(user_id: int) -> bool:
    if not supabase:
        return False
    try:
        res = (
            supabase.table("user_prefs")
            .select("user_id")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        logger.error(f"DB user_exists_in_db: {e}")
        return False

def update_user_gender(user_id: int, gender: str):
    _invalidate_prefs(user_id)
    if not supabase:
        return
    def _run():
        with suppress(Exception):
            supabase.table("user_prefs").update({"gender": gender}).eq(
                "user_id", user_id
            ).execute()
    _submit_db(_run)

def update_user_speed(user_id: int, speed: float):
    speed = round(max(_SPEED_MIN, min(_SPEED_MAX, speed)), 4)
    _invalidate_prefs(user_id)
    if not supabase:
        return
    def _run():
        try:
            supabase.table("user_prefs").update({"speed": speed}).eq(
                "user_id", user_id
            ).execute()
        except Exception as e:
            if "does not exist" in str(e):
                logger.error(
                    "speed column missing — run: "
                    "ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
                )
            else:
                logger.error(f"update_user_speed error: {e}")
    _submit_db(_run)

_rls_warned = False

def save_text_cache(
    msg_id: int,
    text: str,
    chat_id: int = 0,
    user_id: int = None,
    username: str = None,
):
    if not supabase:
        return
    def _run():
        global _rls_warned
        try:
            payload = {"message_id": msg_id, "chat_id": chat_id, "original_text": text}
            if user_id is not None:
                payload["user_id"] = user_id
            if username is not None:
                payload["username"] = username
            supabase.table("text_cache").upsert(
                payload, on_conflict="chat_id,message_id"
            ).execute()
        except Exception as e:
            err_str = str(e)
            if "42501" in err_str or "row-level security" in err_str.lower():
                if not _rls_warned:
                    _rls_warned = True
                    logger.error(
                        "text_cache RLS policy blocking inserts. "
                        "Fix: run in Supabase SQL editor:\n"
                        "  ALTER TABLE text_cache DISABLE ROW LEVEL SECURITY;\n"
                        "  -- or --\n"
                        "  CREATE POLICY \"service_role_all\" ON text_cache "
                        "FOR ALL TO service_role USING (true) WITH CHECK (true);"
                    )
            else:
                logger.error(f"save_text_cache error: {e}")
    _submit_db(_run)

def get_text_cache(msg_id: int, chat_id: int = 0) -> str | None:
    if not supabase:
        return None
    try:
        res = (
            supabase.table("text_cache")
            .select("original_text")
            .eq("message_id", msg_id)
            .eq("chat_id", chat_id)
            .execute()
        )
        if res.data:
            return res.data[0]["original_text"]
    except Exception as e:
        logger.error(f"DB get_text_cache: {e}")
    return None

def ensure_speed_column():
    if not supabase:
        logger.info("ensure_speed_column: Supabase not configured, skipping.")
        return
    try:
        supabase.table("user_prefs").select("speed").limit(1).execute()
        logger.info("speed column present.")
    except Exception as e:
        err = str(e).lower()
        if "does not exist" in err or "column" in err:
            logger.warning(
                "speed column missing. Run in Supabase SQL editor:\n"
                "  ALTER TABLE user_prefs ADD COLUMN speed FLOAT DEFAULT 1.0;"
            )
        else:
            logger.error(f"ensure_speed_column unexpected error: {e}")

# ---------------------------------------------------------------------------
# Database helpers — conversation history
# ---------------------------------------------------------------------------
CONV_HISTORY_LIMIT = 10
CONV_CONTEXT_MAX_CHARS = 3000
CONV_RESOLVE_TIMEOUT_S = 15

_HIST_CACHE_MAX_USERS = 5_000
_HIST_CACHE_TURNS = 10
_hist_cache: OrderedDict[int, deque] = OrderedDict()

def _normalize_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role in ("assistant", "bot", "model"):
        return "assistant"
    return "user"

def _hist_cache_append(user_id: int, role: str, content: str) -> None:
    role = _normalize_role(role)
    content = (content or "").strip()
    if not content:
        return

    if user_id not in _hist_cache:
        while len(_hist_cache) >= _HIST_CACHE_MAX_USERS:
            _hist_cache.popitem(last=False)
        _hist_cache[user_id] = deque(maxlen=_HIST_CACHE_TURNS)

    _hist_cache.move_to_end(user_id)
    _hist_cache[user_id].append({
        "role": role,
        "content": content,
    })

def _hist_cache_get(user_id: int) -> list[dict] | None:
    d = _hist_cache.get(user_id)
    if d is None:
        return None
    _hist_cache.move_to_end(user_id)
    return list(d)

def _hist_cache_clear(user_id: int) -> None:
    _hist_cache.pop(user_id, None)

def db_history_append(user_id: int, role: str, content: str) -> None:
    if not supabase:
        return

    role = _normalize_role(role)
    content = (content or "").strip()
    if not content:
        return

    def _run():
        try:
            supabase.table("conversation_history").insert({
                "user_id": user_id,
                "role": role,
                "content": content,
            }).execute()
        except Exception as e:
            logger.error(
                f"db_history_append error "
                f"(user={user_id}, role={role}): {e}"
            )

    _submit_db(_run)

def db_history_fetch(user_id: int, limit: int = CONV_HISTORY_LIMIT) -> list[dict]:
    if not supabase:
        return []

    try:
        res = (
            supabase.table("conversation_history")
            .select("role, content, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        rows = res.data or []
        rows.reverse()

        clean_rows = []
        for row in rows:
            content = (row.get("content") or "").strip()
            if not content:
                continue
            clean_rows.append({
                "role": _normalize_role(row.get("role")),
                "content": content,
            })

        return clean_rows

    except Exception as e:
        logger.error(f"db_history_fetch error user={user_id}: {e}")
        return []

def db_history_clear(user_id: int) -> None:
    _hist_cache_clear(user_id)

    if not supabase:
        return

    def _run():
        try:
            supabase.table("conversation_history").delete().eq(
                "user_id", user_id
            ).execute()
        except Exception as e:
            logger.error(f"db_history_clear error user={user_id}: {e}")

    _submit_db(_run)

def _build_context_block(history: list[dict]) -> str:
    if not history:
        return ""

    lines = []
    for row in history:
        role = _normalize_role(row.get("role"))
        content = (row.get("content") or "").strip()
        if not content:
            continue

        prefix = "User" if role == "user" else "Bot"
        lines.append(f"{prefix}: {content}")

    block = "\n".join(lines)

    if len(block) > CONV_CONTEXT_MAX_CHARS:
        block = block[-CONV_CONTEXT_MAX_CHARS:]
        nl = block.find("\n")
        if nl != -1:
            block = block[nl + 1:]

    return block

def record_turn(user_id: int, role: str, content: str) -> None:
    role = _normalize_role(role)
    content = (content or "").strip()
    if not content:
        return

    _hist_cache_append(user_id, role, content)
    db_history_append(user_id, role, content)

# ---------------------------------------------------------------------------
# Conversation history: Gemini-powered text resolution
# ---------------------------------------------------------------------------
async def resolve_tts_text(
    user_id: int,
    raw_text: str,
    loop: asyncio.AbstractEventLoop,
) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return raw_text

    # 1. Always load history FIRST.
    # Do not depend on Gemini for normal history commands.
    history = _hist_cache_get(user_id)

    if history is None:
        try:
            db_rows = await loop.run_in_executor(
                _DB_EXECUTOR,
                db_history_fetch,
                user_id,
            )

            for row in db_rows:
                _hist_cache_append(
                    user_id,
                    row.get("role", "user"),
                    row.get("content", ""),
                )

            history = db_rows

        except Exception as e:
            logger.error(f"resolve_tts_text history fetch failed for user {user_id}: {e}")
            history = []

    if not history:
        logger.info(f"conversation_history empty for user {user_id}")
        return raw_text

    # 2. Normalize helper
    def _role_of(row: dict) -> str:
        role = str(row.get("role", "")).strip().lower()
        if role in ("assistant", "bot", "model"):
            return "assistant"
        return "user"

    def _content_of(row: dict) -> str:
        return str(row.get("content", "") or "").strip()

    def _last_by_role(role_name: str) -> str | None:
        for row in reversed(history):
            if _role_of(row) == role_name:
                content = _content_of(row)
                if content:
                    return content
        return None

    # 3. Fast local history commands.
    # These work even when Gemini returns 500 INTERNAL.
    lower = raw_text.lower()
    compact = lower.replace(" ", "")

    repeat_phrases = (
        "read again",
        "read that again",
        "say again",
        "repeat",
        "again",
        "អានម្តងទៀត",
        "អានម្ដងទៀត",
        "ម្តងទៀត",
        "ម្ដងទៀត",
        "អានឡើងវិញ",
        "និយាយម្តងទៀត",
        "និយាយម្ដងទៀត",
    )

    if any(p in lower for p in repeat_phrases) or any(p in compact for p in (
        "អានម្តងទៀត",
        "អានម្ដងទៀត",
        "ម្តងទៀត",
        "ម្ដងទៀត",
    )):
        last_bot = _last_by_role("assistant")
        if last_bot:
            logger.info(f"resolve_tts_text local repeat matched for user {user_id}")
            return last_bot

    what_user_said_phrases = (
        "what did i say",
        "what i said",
        "my last message",
        "អ្វីដែលខ្ញុំនិយាយ",
        "ខ្ញុំនិយាយអ្វី",
        "សារចុងក្រោយរបស់ខ្ញុំ",
    )

    if any(p in lower for p in what_user_said_phrases) or any(p in compact for p in (
        "ខ្ញុំនិយាយអ្វី",
        "អ្វីដែលខ្ញុំនិយាយ",
    )):
        last_user = _last_by_role("user")
        if last_user:
            logger.info(f"resolve_tts_text local last-user matched for user {user_id}")
            return last_user

    # 4. Optional Gemini resolver.
    # Disabled by default so normal text TTS does not call Gemini and does not print AFC logs.
    # Image OCR, audio transcription, and /ai-assistant still use Gemini normally.
    if not TTS_RESOLVER_AI_ENABLED:
        logger.debug(f"TTS resolver AI disabled for user {user_id}; using raw text after local history checks.")
        return raw_text

    if not _gemini:
        logger.info("resolve_tts_text: Gemini not available, using raw text after local history checks.")
        return raw_text

    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        logger.warning("resolve_tts_text: _AI_SEMAPHORE not initialised, using raw text.")
        return raw_text

    context_block = _build_context_block(history)

    if not context_block.strip():
        logger.info(f"resolve_tts_text context block empty for user {user_id}")
        return raw_text

    system_prompt = (
        "You are a text pre-processor for a Khmer/English text-to-speech bot. "
        "You receive a conversation history and the user's latest message. "
        "Your ONLY job is to output the exact text that should be spoken aloud. "
        "Output nothing else — no labels, no explanation, no markdown.\n\n"
        "Rules:\n"
        "1. If the message is a normal sentence or paragraph to be read, output it verbatim.\n"
        "2. If the message is a follow-up like 'read that again' or 'អានម្តងទៀត', "
        "output the last Bot turn verbatim.\n"
        "3. If the message asks to translate something, output the translated text in the target language.\n"
        "4. If the message references previous content using pronouns like 'that', 'it', "
        "'the last thing', resolve the reference and output the resolved text.\n"
        "5. If the message asks 'what did I say?' or similar, output the last User turn.\n"
        "6. Preserve the original language Khmer or English unless translation is requested.\n"
        "7. If you cannot determine what to speak, output the user's message verbatim."
    )

    combined_prompt = (
        f"{system_prompt}\n\n"
        f"Conversation history:\n{context_block}\n\n"
        f"User's new message: {raw_text}\n\n"
        "Output only the text to speak:"
    )

    async def _guarded_call():
        async with semaphore:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=combined_prompt,
                )

            return await loop.run_in_executor(_AI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(
            _guarded_call(),
            timeout=CONV_RESOLVE_TIMEOUT_S,
        )

        resolved = (response.text or "").strip()

        if resolved:
            logger.info(f"resolve_tts_text Gemini resolved text for user {user_id}")
            return resolved

        return raw_text

    except asyncio.TimeoutError:
        logger.warning(f"resolve_tts_text timed out for user {user_id} — using raw text")
        return raw_text

    except Exception as e:
        logger.warning(f"resolve_tts_text error for user {user_id}: {e} — using raw text")
        return raw_text

# ---------------------------------------------------------------------------
# Database helpers — scheduled broadcasts
# ---------------------------------------------------------------------------
def db_sched_insert(payload: dict, admin_id: int, broadcast_at: datetime) -> dict:
    if not supabase:
        raise RuntimeError("Supabase not configured.")
    row = {
        "admin_id": admin_id,
        "photo_file_id": payload.get("photo_file_id"),
        "caption": payload.get("caption"),
        "plain_text": payload.get("text"),
        "broadcast_at": broadcast_at.isoformat(),
        "status": "pending",
    }
    res = supabase.table("scheduled_broadcasts").insert(row).execute()
    return res.data[0]

def db_sched_fetch_due() -> list[dict]:
    if not supabase:
        return []
    try:
        now = datetime.now(timezone.utc).isoformat()
        res = (
            supabase.table("scheduled_broadcasts")
            .select("*")
            .eq("status", "pending")
            .lte("broadcast_at", now)
            .order("broadcast_at")
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"db_sched_fetch_due: {e}")
        return []

def db_sched_claim(row_id: int) -> bool:
    if not supabase:
        return False
    try:
        res = (
            supabase.table("scheduled_broadcasts")
            .update({"status": "sending"})
            .eq("id", row_id)
            .eq("status", "pending")
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        logger.error(f"db_sched_claim #{row_id}: {e}")
        return False

def db_sched_set_status(row_id: int, status: str, **extra) -> None:
    if not supabase:
        return
    with suppress(Exception):
        patch = {"status": status, **extra}
        supabase.table("scheduled_broadcasts").update(patch).eq("id", row_id).execute()

def db_sched_fetch_admin_pending(admin_id: int) -> list[dict]:
    if not supabase:
        return []
    try:
        res = (
            supabase.table("scheduled_broadcasts")
            .select("id, broadcast_at, plain_text, caption, photo_file_id, status")
            .eq("admin_id", admin_id)
            .eq("status", "pending")
            .order("broadcast_at")
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"db_sched_fetch_admin_pending: {e}")
        return []

def db_sched_fetch_one(row_id: int) -> dict | None:
    if not supabase:
        return None
    try:
        res = (
            supabase.table("scheduled_broadcasts")
            .select("*")
            .eq("id", row_id)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"db_sched_fetch_one: {e}")
        return None

# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------
def _parse_dt(text: str) -> datetime | None:
    text = text.strip()
    for fmt in _DT_FORMATS:
        try:
            naive = datetime.strptime(text, fmt)
            return naive.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")

# ---------------------------------------------------------------------------
# Keyboard helpers — scheduled broadcast
# ---------------------------------------------------------------------------
def get_sched_confirm_kb(row_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ បញ្ជាក់ Schedule", callback_data=f"sched_ok:{row_id}"),
            InlineKeyboardButton("❌ បោះបង់", callback_data=f"sched_no:{row_id}"),
        ]]
    )

def get_admin_dashboard_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Stats", callback_data="admin_stats"),
                InlineKeyboardButton("🩺 Health", callback_data="admin_health"),
            ],
            [
                InlineKeyboardButton("📢 Broadcast", callback_data="admin_broadcast"),
                InlineKeyboardButton("⏰ Schedules", callback_data="admin_schedules"),
            ],
            [
                InlineKeyboardButton("👥 Users", callback_data="admin_users"),
                InlineKeyboardButton("❌ Close", callback_data="admin_close"),
            ],
        ]
    )

def get_schedules_list_kb(
    rows: list[dict], page: int, page_size: int = 5
) -> InlineKeyboardMarkup:
    total = max(1, (len(rows) + page_size - 1) // page_size)
    chunk = rows[page * page_size : (page + 1) * page_size]
    kbd_rows = []
    for r in chunk:
        try:
            dt_str = _fmt_dt(datetime.fromisoformat(r["broadcast_at"]))
        except Exception:
            dt_str = str(r.get("broadcast_at", "?"))
        label = f"#{r['id']}  {dt_str}"
        kbd_rows.append([InlineKeyboardButton(label, callback_data=f"sched_view:{r['id']}")])
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"sched_page:{page - 1}"))
    nav.append(InlineKeyboardButton(f"{page + 1}/{total}", callback_data="sched_noop"))
    if page < total - 1:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"sched_page:{page + 1}"))
    if nav:
        kbd_rows.append(nav)
    kbd_rows.append([InlineKeyboardButton("❌ បិទ", callback_data="sched_close")])
    return InlineKeyboardMarkup(kbd_rows)

# ---------------------------------------------------------------------------
# Status timer
# ---------------------------------------------------------------------------
_STATUS_FRAMES = [
    "⏳ កំពុងបង្កើតសំឡេង ·",
    "⏳ កំពុងបង្កើតសំឡេង ··",
    "⏳ កំពុងបង្កើតសំឡេង ···",
    "⏳ កំពុងបង្កើតសំឡេង ····",
]
_TRANSCRIBE_FRAMES = [
    "🎙️ កំពុង Transcribe ·",
    "🎙️ កំពុង Transcribe ··",
    "🎙️ កំពុង Transcribe ···",
    "🎙️ កំពុង Transcribe ····",
]
_OCR_FRAMES = [
    "🔍 កំពុង OCR រូបភាព ·",
    "🔍 កំពុង OCR រូបភាព ··",
    "🔍 កំពុង OCR រូបភាព ···",
    "🔍 កំពុង OCR រូបភាព ····",
]
_AUDIO_FILE_FRAMES = [
    "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ·",
    "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ··",
    "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ···",
    "🎵 កំពុង Transcribe ឯកសារអូឌីយ៉ូ ····",
]

async def send_status_timer(
    chat_id: int, bot, stop_event: asyncio.Event, frames=None
):
    frames = frames or _STATUS_FRAMES
    msg = None
    try:
        msg = await safe_send(lambda: bot.send_message(chat_id=chat_id, text=frames[0]))
        if msg is None:
            return
        frame_idx = 1
        while not stop_event.is_set():
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            if stop_event.is_set():
                break
            current_frame = frames[frame_idx % len(frames)]
            with suppress(BadRequest, TelegramError):
                await msg.edit_text(current_frame)
            frame_idx += 1
    except (asyncio.CancelledError, Exception) as e:
        if not isinstance(e, asyncio.CancelledError):
            logger.warning(f"Status timer error: {e}")
    finally:
        if msg:
            with suppress(Exception):
                await msg.delete()

async def _stop_timer(stop_event: asyncio.Event, timer_task: asyncio.Task):
    stop_event.set()
    if not timer_task.done():
        timer_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await timer_task

# ---------------------------------------------------------------------------
# Audio pipeline
# ---------------------------------------------------------------------------
def _rounded_speed(speed: float) -> float:
    return round(max(_SPEED_MIN, min(_SPEED_MAX, speed)), 4)

@functools.lru_cache(maxsize=64)
def _build_atempo_chain(speed: float) -> str:
    """Build an FFmpeg atempo filter chain for the given speed multiplier.

    Uses pre-rounded speed values as cache keys to avoid float equality issues.
    atempo only supports [0.5, 2.0], so chain multiple stages for values outside.
    """
    if abs(speed - 1.0) < 1e-6:
        return "atempo=1.0"
    stages: list[str] = []
    r = speed
    if r < 1.0:
        while r < 0.5 - 1e-9 and r > 1e-6:
            stages.append("atempo=0.5")
            r = round(r / 0.5, 6)
        stages.append(f"atempo={max(0.5, r):.6f}")
    else:
        while r > 2.0 + 1e-9:
            stages.append("atempo=2.0")
            r = round(r / 2.0, 6)
        stages.append(f"atempo={min(2.0, r):.6f}")
    return ",".join(stages)

def _detect_voice(text: str, gender: str) -> str:
    khmer_chars = len(_KHMER_RE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())
    is_khmer = khmer_chars > (total_alpha * 0.3) if total_alpha else False
    return VOICE_MAP["km" if is_khmer else "en"][gender]

async def generate_voice(text: str, gender: str, speed: float, output_path: str) -> bytes:
    text = text.strip()
    if not text:
        raise ValueError("generate_voice: text must not be empty")

    voice = _detect_voice(text, gender)
    mp3_chunks: list[bytes] = []
    try:
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])
    except Exception as e:
        raise RuntimeError(f"edge-tts failed: {e}") from e

    mp3_data = b"".join(mp3_chunks)
    if not mp3_data:
        raise RuntimeError("edge-tts returned empty audio")

    speed_key = _rounded_speed(speed)
    af = _build_atempo_chain(speed_key) if abs(speed_key - DEFAULT_SPEED) > 1e-4 else None
    cmd = [_FFMPEG_EXE, "-y", "-f", "mp3", "-i", "pipe:0"]
    if af:
        cmd += ["-filter:a", af]
    cmd += ["-c:a", "libopus", "-b:a", "32k", output_path]

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5,  # subprocess creation should be near-instant
        )
        _, stderr_data = await asyncio.wait_for(proc.communicate(input=mp3_data), timeout=60)
    except asyncio.TimeoutError:
        raise RuntimeError("FFmpeg timed out after 60s")

    if proc.returncode != 0:
        stderr_snippet = (stderr_data or b"").decode(errors="replace")[-400:]
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode}): {stderr_snippet}")

    try:
        loop = asyncio.get_running_loop()
        audio_bytes = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: open(output_path, "rb").read()),
            timeout=10,
        )
    except asyncio.TimeoutError:
        raise RuntimeError("Timed out reading output audio file")
    except OSError as e:
        raise RuntimeError(f"Failed to read output audio file: {e}") from e
    return audio_bytes

async def generate_voice_limited(text: str, gender: str, speed: float, output_path: str) -> bytes:
    """Generate TTS using the global TTS semaphore.

    This keeps the bot able to serve multiple users while preventing FFmpeg/Edge-TTS
    from overloading the server.
    """
    sem = _TTS_CHUNK_SEMAPHORE
    if sem is None:
        return await generate_voice(text, gender, speed, output_path)

    async with sem:
        return await generate_voice(text, gender, speed, output_path)


# ---------------------------------------------------------------------------
# Gemini transcription helpers
# ---------------------------------------------------------------------------
async def transcribe_voice(ogg_path: str) -> str:
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(
            None, lambda: open(ogg_path, "rb").read()
        )
    except OSError as e:
        raise RuntimeError(f"Cannot read voice file: {e}") from e

    prompt = (
        "Transcribe this audio exactly as spoken. "
        "Output ONLY the transcribed text — no labels, no explanation. "
        "Support both Khmer and English."
    )
    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        raise RuntimeError("Gemini semaphore not initialised.")

    async def _guarded_call():
        async with semaphore:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
                        prompt,
                    ],
                )
            return await loop.run_in_executor(_AI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(_guarded_call(), timeout=60)
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini transcription timed out after 60s")

async def transcribe_audio_file(file_path: str, mime_type: str) -> str:
    if not _gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(
            None, lambda: open(file_path, "rb").read()
        )
    except OSError as e:
        raise RuntimeError(f"Cannot read audio file: {e}") from e

    prompt = (
        "Transcribe this audio exactly as spoken. "
        "Output ONLY the transcribed text — no labels, no explanation. "
        "Support both Khmer and English."
    )
    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        raise RuntimeError("Gemini semaphore not initialised.")

    async def _guarded_call():
        async with semaphore:
            def _call():
                return _gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
            return await loop.run_in_executor(_AI_EXECUTOR, _call)

    try:
        response = await asyncio.wait_for(_guarded_call(), timeout=90)
        return (response.text or "").strip()
    except asyncio.TimeoutError:
        raise RuntimeError("Gemini audio transcription timed out after 90s")

# ---------------------------------------------------------------------------
# Text chunking for TTS
# ---------------------------------------------------------------------------
def _split_text_chunks(text: str, max_chars: int = TTS_CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    chunks, current = [], ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            if " " in sent:
                for word in sent.split():
                    if not word:
                        continue
                    if len(current) + len(word) + (1 if current else 0) > max_chars:
                        if current:
                            chunks.append(current.strip())
                        current = word
                    else:
                        current = (current + " " + word).strip() if current else word
            else:
                pos = 0
                while pos < len(sent):
                    space = max_chars - len(current)
                    current += sent[pos : pos + space]
                    pos += space
                    if len(current) >= max_chars:
                        chunks.append(current)
                        current = ""
        elif len(current) + len(sent) + (1 if current else 0) > max_chars:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c and c.strip()]

# ---------------------------------------------------------------------------
# Detect image MIME type from magic bytes
# ---------------------------------------------------------------------------
def _detect_image_mime(path: str) -> str:
    try:
        with open(path, "rb") as f:
            header = f.read(12)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
        if header[:2] == b"\xff\xd8":
            return "image/jpeg"
        if header[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
    except Exception:
        pass
    return "image/jpeg"

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
async def ocr_image(image_path: str, mime_type: str = "image/jpeg") -> str:
    """Read an image from disk and extract text using the configured OCR provider."""
    if not _ocr_configured():
        raise RuntimeError(_ocr_status_for_user())

    loop = asyncio.get_running_loop()
    try:
        image_bytes = await loop.run_in_executor(
            None, lambda: open(image_path, "rb").read()
        )
    except OSError as e:
        raise RuntimeError(f"Cannot read image file: {e}") from e

    semaphore = _AI_SEMAPHORE
    if semaphore is None:
        raise RuntimeError("AI semaphore not initialised.")

    async def _guarded_call():
        async with semaphore:
            return await loop.run_in_executor(
                _AI_EXECUTOR,
                lambda: ask_ocr_image(image_bytes, mime_type)[0],
            )

    try:
        return await asyncio.wait_for(_guarded_call(), timeout=OCR_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise RuntimeError(f"OCR timed out after {OCR_TIMEOUT_SECONDS}s")

# ---------------------------------------------------------------------------
# Cooldown check
# ---------------------------------------------------------------------------
async def _check_cooldown(reply_target, user_id: int) -> bool:
    """Return True (and notify user) if the user should be rate-limited."""
    lock = _get_user_lock(user_id)
    if lock.locked():
        await safe_send(
            lambda: reply_target.reply_text("⏳ សូមរង់ចាំ TTS មុននៅក្នុងដំណើរការ...")
        )
        return True
    now = time.monotonic()
    last = _get_last_tts(user_id)
    if now - last < USER_COOLDOWN_S:
        rem = round(USER_COOLDOWN_S - (now - last), 1)
        await safe_send(
            lambda r=rem: reply_target.reply_text(
                f"⏳ សូមរង់ចាំ {r}s មុននឹងផ្ញើម្តងទៀត។"
            )
        )
        return True
    return False

# ---------------------------------------------------------------------------
# Paged TTS delivery
# ---------------------------------------------------------------------------
async def _deliver_paged_tts(
    chat_id: int,
    bot,
    text: str,
    gender: str,
    speed: float,
    user_id: int,
    username: str,
) -> None:
    """Generate and send long TTS text in ordered chunks.

    The global TTS semaphore lets up to MAX_CONCURRENT_TTS_USERS different users
    run at the same time, while each user's chunks are generated sequentially to
    avoid high memory/CPU spikes from multiple FFmpeg processes for one request.
    """
    chunks = _split_text_chunks(text)

    if not chunks:
        await safe_send(
            lambda: bot.send_message(chat_id=chat_id, text="❌ រកអត្ថបទមិនឃើញ។")
        )
        return

    set_last_tts_text(user_id, text)
    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        file_path = _make_temp_ogg()
        try:
            audio_bytes = await generate_voice_limited(chunk, gender, speed, file_path)

            sent = await safe_send(
                lambda ab=audio_bytes, ci=i, ct=total: bot.send_voice(
                    chat_id=chat_id,
                    voice=io.BytesIO(ab),
                    caption=f"🗣️ {BOT_TAG}  [{ci}/{ct}]",
                    reply_markup=get_main_kb(gender),
                )
            )

            if sent:
                save_text_cache(
                    sent.message_id,
                    chunk,
                    chat_id=chat_id,
                    user_id=user_id,
                    username=username,
                )
                set_last_tts_text(user_id, chunk)

        except Exception as e:
            logger.error(f"paged TTS chunk {i}/{total} error: {e}", exc_info=True)
            await safe_send(
                lambda ci=i, ct=total: bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ មានបញ្ហាក្នុង chunk {ci}/{ct}។",
                )
            )

        finally:
            _cleanup(file_path)

        if i < total:
            await asyncio.sleep(0.25)

    _set_last_tts(user_id)

# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------
def get_main_kb(gender: str) -> InlineKeyboardMarkup:
    f_btn = "👩 សំឡេងស្រី" + (" ✅" if gender == "female" else "")
    m_btn = "👨 សំឡេងប្រុស" + (" ✅" if gender == "male" else "")
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(f_btn, callback_data="tg_female"),
                InlineKeyboardButton(m_btn, callback_data="tg_male"),
            ],
            [InlineKeyboardButton("🎚️ ល្បឿនសំឡេង", callback_data="show_speed")],
        ]
    )

def get_speed_kb(current_speed: float) -> InlineKeyboardMarkup:
    speed_row = []
    for cb, (lbl, val) in SPEED_OPTIONS.items():
        mark = " ✅" if abs(val - current_speed) < 0.01 else ""
        speed_row.append(InlineKeyboardButton(lbl + mark, callback_data=cb))
    return InlineKeyboardMarkup(
        [
            speed_row,
            [InlineKeyboardButton("🔙 ត្រឡប់", callback_data="hide_speed")],
        ]
    )

def get_transcription_kb(transcript_msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("📢 AI អាន", callback_data=f"tts_transcript:{transcript_msg_id}"),
            InlineKeyboardButton("🗑️ លុប", callback_data=f"del_transcript:{transcript_msg_id}"),
        ]]
    )

def get_audio_file_kb(msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("📢 បំលែងទៅសំឡេង TTS", callback_data=f"audio_tts:{msg_id}"),
            InlineKeyboardButton("🗑️ លុប", callback_data=f"audio_del:{msg_id}"),
        ]]
    )

def get_ocr_confirm_kb(msg_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("▶️ អាន", callback_data=f"doc_read:{msg_id}"),
            InlineKeyboardButton("🗑️ លុប", callback_data=f"doc_del:{msg_id}"),
        ]]
    )

def get_broadcast_confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ បញ្ជាក់ Broadcast", callback_data="bc_confirm"),
            InlineKeyboardButton("❌ បោះបង់", callback_data="bc_cancel"),
        ]]
    )

def get_users_page_kb(
    users: list[dict], page: int, page_size: int = 8
) -> InlineKeyboardMarkup:
    total_pages = max(1, (len(users) + page_size - 1) // page_size)
    chunk = users[page * page_size : page * page_size + page_size]
    rows = []
    for u in chunk:
        uid = u["user_id"]
        uname = (u.get("username") or str(uid))[:20]
        rows.append(
            [InlineKeyboardButton(f"👤 {uname}  ({uid})", callback_data=f"chat_open:{uid}")]
        )
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"users_page:{page - 1}"))
    nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"users_page:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("❌ បិទ", callback_data="users_close")])
    return InlineKeyboardMarkup(rows)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            with suppress(OSError):
                os.remove(p)

# ---------------------------------------------------------------------------
# Admin guard
# ---------------------------------------------------------------------------
def admin_only(handler):
    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        if uid not in ADMIN_IDS:
            if update.message:
                await safe_send(
                    lambda: update.message.reply_text(
                        "⛔ អ្នកមិនមានសិទ្ធិប្រើពាក្យបញ្ជានេះទេ។"
                    )
                )
            return
        return await handler(update, context)
    return wrapper

def _is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


# ---------------------------------------------------------------------------
# Chat session helpers
# ---------------------------------------------------------------------------
def _open_session(admin_id: int, target_id: int) -> None:
    old_target = _admin_chat_target.get(admin_id)
    if old_target is not None:
        _user_to_admin.pop(old_target, None)
    _admin_chat_target[admin_id] = target_id
    _user_to_admin[target_id] = admin_id

def _close_session(admin_id: int) -> int | None:
    target_id = _admin_chat_target.pop(admin_id, None)
    if target_id is not None:
        _user_to_admin.pop(target_id, None)
    return target_id

def _get_admin_for_user(user_id: int) -> int | None:
    return _user_to_admin.get(user_id)

# ===========================================================================
# _run_broadcast_to_all
# ===========================================================================
async def _run_broadcast_to_all(
    bot,
    admin_id: int,
    pending: dict,
    label: str = "Broadcast",
) -> tuple[int, int, int]:
    broadcast_semaphore = _BROADCAST_SEMAPHORE
    if broadcast_semaphore is None:
        logger.error(f"{label}: _BROADCAST_SEMAPHORE not initialised.")
        return (0, 0, 0)

    loop = asyncio.get_running_loop()
    user_ids = await loop.run_in_executor(None, get_all_user_ids)
    total = len(user_ids)
    if total == 0:
        await safe_send(
            lambda: bot.send_message(
                chat_id=admin_id,
                text=f"⚠️ {label}: មិនមានអ្នកប្រើប្រាស់ registered ណាមួយទេ។",
            )
        )
        return (0, 0, 0)

    sent = failed = blocked = 0
    photo_file_id = pending.get("photo_file_id")
    caption = pending.get("caption") or ""
    plain_text = pending.get("text") or ""

    progress_msg = await safe_send(
        lambda: bot.send_message(
            chat_id=admin_id, text=f"📡 {label} — កំពុង Broadcast ទៅ {total} នាក់..."
        )
    )

    async def _send_one(uid: int) -> str:
        async with broadcast_semaphore:
            for attempt in range(2):
                try:
                    if photo_file_id:
                        await bot.send_photo(
                            chat_id=uid,
                            photo=photo_file_id,
                            caption=caption or None,
                            parse_mode="HTML" if caption else None,
                        )
                    else:
                        await bot.send_message(
                            chat_id=uid, text=plain_text, parse_mode="HTML"
                        )
                    return "sent"
                except Forbidden:
                    return "blocked"
                except RetryAfter as e:
                    await asyncio.sleep(e.retry_after + 1)
                    if attempt == 1:
                        return "failed"
                except Exception as e:
                    logger.error(f"{label} error uid={uid} attempt={attempt}: {e}")
                    if attempt == 1:
                        return "failed"
        return "failed"

    for i, uid in enumerate(user_ids):
        result = await _send_one(uid)
        if result == "sent":
            sent += 1
        elif result == "blocked":
            blocked += 1
        else:
            failed += 1
        await asyncio.sleep(0.034)
        if (i + 1) % 25 == 0 and progress_msg:
            with suppress(Exception):
                pct = int((i + 1) / total * 100) if total else 0
                await progress_msg.edit_text(
                    f"📡 {label}: {pct}% ({i + 1}/{total})\n"
                    f"✅ {sent}  🚫 {blocked}  ❌ {failed}"
                )

    report = (
        f"✅ <b>{label} រួចរាល់!</b>\n\n"
        f"👥 សរុប: {total}\n"
        f"📨 បានផ្ញើ: {sent}\n"
        f"🚫 Blocked: {blocked}\n"
        f"❌ Failed: {failed}"
    )
    try:
        if progress_msg:
            await safe_send(lambda: progress_msg.edit_text(report, parse_mode="HTML"))
        else:
            await safe_send(
                lambda: bot.send_message(chat_id=admin_id, text=report, parse_mode="HTML")
            )
    except Exception as e:
        logger.error(f"{label} report error: {e}")

    return (sent, failed, blocked)

# ===========================================================================
# BROADCAST (immediate)
# ===========================================================================
@admin_only
async def broadcast_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _pending_broadcast.pop(update.effective_user.id, None)
    context.user_data["bc_state"] = BROADCAST_WAIT_MESSAGE
    await safe_send(
        lambda: update.message.reply_text(
            "📡 <b>Admin Broadcast</b>\n\n"
            "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Broadcast ។\n"
            "👉 អាចផ្ញើរូបភាព + Caption រួមគ្នា ឬ តែ text ។\n\n"
            "វាយ /cancel ដើម្បីបោះបង់។",
            parse_mode="HTML",
        )
    )

@admin_only
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🛠️ <b>Admin Dashboard</b>\n\n"
        "ជ្រើសរើសមុខងារខាងក្រោម៖\n\n"
        "📊 Stats - មើលស្ថិតិ Bot\n"
        "🩺 Health - ពិនិត្យ Bot/Supabase/Gemini/FFmpeg\n"
        "📢 Broadcast - ផ្ញើសារទៅ users\n"
        "⏰ Schedules - មើល scheduled broadcasts\n"
        "👥 Users - មើល users"
    )

    await safe_send(
        lambda: update.message.reply_text(
            text,
            parse_mode="HTML",
            reply_markup=get_admin_dashboard_kb(),
        )
    )
    
@admin_only
async def broadcast_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if context.user_data.get("bc_state") != BROADCAST_WAIT_MESSAGE:
        return

    msg = update.message
    photo_file_id: str | None = None
    caption_text: str | None = None
    plain_text: str | None = None

    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text = msg.caption or ""
    elif msg.text:
        plain_text = msg.text.strip()
        if not plain_text:
            await safe_send(lambda: msg.reply_text("⚠️ អត្ថបទមិនអាចទទេបាន។ សូមវាយសារ។"))
            return
    else:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return

    _pending_broadcast[user_id] = {
        "photo_file_id": photo_file_id,
        "caption": caption_text,
        "text": plain_text,
    }

    if photo_file_id:
        preview_cap = html.escape(caption_text) if caption_text else "<i>(គ្មាន Caption)</i>"
        await safe_send(
            lambda: msg.reply_photo(
                photo=photo_file_id,
                caption=(
                    f"👁️ <b>Preview Broadcast</b>\n\n{preview_cap}\n\n"
                    "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?"
                ),
                parse_mode="HTML",
                reply_markup=get_broadcast_confirm_kb(),
            )
        )
    else:
        await safe_send(
            lambda: msg.reply_text(
                f"👁️ <b>Preview Broadcast</b>\n\n{html.escape(plain_text)}\n\n"
                "តើចង់ Broadcast ដល់អ្នកប្រើប្រាស់ទាំងអស់?",
                parse_mode="HTML",
                reply_markup=get_broadcast_confirm_kb(),
            )
        )

async def broadcast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    if data == "bc_cancel":
        _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(lambda: query.message.reply_text("❌ Broadcast បានបោះបង់។"))
        return

    if data == "bc_confirm":
        pending = _pending_broadcast.pop(user_id, None)
        context.user_data.pop("bc_state", None)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        if not pending:
            await safe_send(
                lambda: query.message.reply_text(
                    "⚠️ រកទិន្នន័យ Broadcast មិនឃើញ។ សូមចាប់ផ្ដើមថ្មី។"
                )
            )
            return
        context.application.create_task(
            _run_broadcast_to_all(context.bot, user_id, pending, label="Broadcast")
        )

# ===========================================================================
# SCHEDULED BROADCAST
# ===========================================================================
@admin_only
async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    _sched_payload.pop(admin_id, None)
    context.user_data["sched_state"] = SCHED_WAIT_MSG
    await safe_send(
        lambda: update.message.reply_text(
            "📅 <b>Scheduled Broadcast</b>\n\n"
            "ផ្ញើ <b>សារ</b> ឬ <b>រូបភាព + Caption</b> ដែលចង់ Schedule ។\n\n"
            "វាយ /cancel ដើម្បីបោះបង់។",
            parse_mode="HTML",
        )
    )

@admin_only
async def cmd_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, db_sched_fetch_admin_pending, admin_id)
    if not rows:
        await safe_send(
            lambda: update.message.reply_text("📭 មិនមាន Scheduled Broadcast ណាមួយទេ។")
        )
        return
    await safe_send(
        lambda: update.message.reply_text(
            f"📋 <b>Scheduled Broadcasts ({len(rows)} pending)</b>\n"
            "ចុចលើ Schedule ដើម្បីមើលលម្អិត ឬ Cancel ។",
            parse_mode="HTML",
            reply_markup=get_schedules_list_kb(rows, page=0),
        )
    )

@admin_only
async def cmd_cancelschedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    args = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(
            lambda: update.message.reply_text(
                "❌ Usage: /cancelschedule &lt;id&gt;\nឬប្រើ /schedules ដើម្បីជ្រើស។",
                parse_mode="HTML",
            )
        )
        return
    row_id = int(args[0])
    loop = asyncio.get_running_loop()
    row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
    if not row:
        await safe_send(lambda: update.message.reply_text(f"❌ រកមិនឃើញ Schedule #{row_id}។"))
        return
    if row["admin_id"] != admin_id:
        await safe_send(lambda: update.message.reply_text("⛔ Schedule នេះមិនមែនជារបស់អ្នកទេ។"))
        return
    if row["status"] != "pending":
        st = row["status"]
        await safe_send(
            lambda: update.message.reply_text(
                f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
                parse_mode="HTML",
            )
        )
        return
    await loop.run_in_executor(None, db_sched_set_status, row_id, "cancelled")
    await safe_send(
        lambda: update.message.reply_text(
            f"✅ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
        )
    )

async def _handle_sched_content(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        return False
    if context.user_data.get("sched_state") != SCHED_WAIT_MSG:
        return False

    msg = update.message
    photo_file_id: str | None = None
    caption_text: str | None = None
    plain_text: str | None = None

    if msg.photo:
        photo_file_id = msg.photo[-1].file_id
        caption_text = msg.caption or ""
    elif msg.text:
        plain_text = msg.text.strip()
        if not plain_text:
            await safe_send(
                lambda: msg.reply_text("⚠️ អត្ថបទមិនអាចទទេបាន។ សូមវាយសារ ឬ ផ្ញើរូបភាព។")
            )
            return True
    else:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើតែ Text ឬ រូបភាព + Caption ប៉ុណ្ណោះ។"))
        return True

    _sched_payload[user_id] = {
        "photo_file_id": photo_file_id,
        "caption": caption_text,
        "text": plain_text,
    }
    context.user_data["sched_state"] = SCHED_WAIT_TIME
    await safe_send(
        lambda: msg.reply_text(
            "🕐 <b>ពេលវេលា Broadcast (UTC)</b>\n\n"
            "វាយកាលបរិច្ឆេទ និងម៉ោង ។\n"
            "ទម្រង់: <code>YYYY-MM-DD HH:MM</code>\n"
            "ឧទាហរណ៍: <code>2025-12-25 09:00</code>",
            parse_mode="HTML",
        )
    )
    return True

async def _handle_sched_datetime(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        return False
    if context.user_data.get("sched_state") != SCHED_WAIT_TIME:
        return False

    msg = update.message
    if not msg.text:
        await safe_send(lambda: msg.reply_text("⚠️ ផ្ញើ Text ពេលវេលា (UTC)។"))
        return True

    broadcast_at = _parse_dt(msg.text)
    if broadcast_at is None:
        await safe_send(
            lambda: msg.reply_text(
                "❌ ទម្រង់ពេលវេលាខុស។\nឧទាហរណ៍ត្រឹមត្រូវ: <code>2025-12-25 09:00</code>",
                parse_mode="HTML",
            )
        )
        return True

    now = datetime.now(timezone.utc)
    if broadcast_at <= now:
        await safe_send(
            lambda: msg.reply_text(
                "❌ ពេលវេលាត្រូវតែជាអនាគត (UTC) ។\n"
                f"ឥឡូវ: <code>{_fmt_dt(now)}</code>",
                parse_mode="HTML",
            )
        )
        return True

    payload = _sched_payload.pop(user_id, None)
    if not payload:
        context.user_data.pop("sched_state", None)
        await safe_send(
            lambda: msg.reply_text(
                "❌ រកទិន្នន័យ Schedule មិនឃើញ (session expired)។\n"
                "សូមចាប់ផ្ដើម /schedule ម្តងទៀត។"
            )
        )
        return True

    context.user_data.pop("sched_state", None)
    loop = asyncio.get_running_loop()
    try:
        row = await loop.run_in_executor(None, db_sched_insert, payload, user_id, broadcast_at)
    except Exception as e:
        logger.error(f"db_sched_insert failed: {e}")
        await safe_send(
            lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ Save Schedule ។ សូមព្យាយាមម្តងទៀត។")
        )
        return True

    row_id = row["id"]
    dt_str = _fmt_dt(broadcast_at)

    if payload.get("photo_file_id"):
        cap_preview = (
            html.escape(payload["caption"])
            if payload.get("caption")
            else "<i>(គ្មាន Caption)</i>"
        )
        await safe_send(
            lambda: msg.reply_photo(
                photo=payload["photo_file_id"],
                caption=(
                    f"📅 <b>Preview Schedule #{row_id}</b>\n"
                    f"⏰ {dt_str}\n\n{cap_preview}"
                ),
                parse_mode="HTML",
                reply_markup=get_sched_confirm_kb(row_id),
            )
        )
    else:
        await safe_send(
            lambda: msg.reply_text(
                f"📅 <b>Preview Schedule #{row_id}</b>\n"
                f"⏰ {dt_str}\n\n"
                f"{html.escape(payload.get('text') or '')}",
                parse_mode="HTML",
                reply_markup=get_sched_confirm_kb(row_id),
            )
        )
    return True

async def _cb_admin_dashboard(query, user_id: int, context, data: str):
    if not _is_admin(user_id):
        await safe_send(lambda: query.message.reply_text("⛔ Admin only."))
        return

    if data == "admin_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "admin_stats":
        # Reuse your existing /stats command logic if possible.
        # This sends a short stats view directly from DB.
        total_users = 0
        pending_sched = 0
        blocked_users = 0

        loop = asyncio.get_running_loop()

        if supabase:
            try:
                def _fetch_stats():
                    users = supabase.table("user_prefs").select("user_id").execute()
                    sched = (
                        supabase.table("scheduled_broadcasts")
                        .select("id")
                        .eq("status", "pending")
                        .execute()
                    )
                    blocked = supabase.table("blocked_users").select("user_id").execute()
                    return (
                        len(users.data or []),
                        len(sched.data or []),
                        len(blocked.data or []),
                    )

                total_users, pending_sched, blocked_users = await loop.run_in_executor(
                    _DB_EXECUTOR,
                    _fetch_stats,
                )
            except Exception as e:
                logger.error(f"admin dashboard stats error: {e}", exc_info=True)

        text = (
            "📊 <b>Bot Stats</b>\n\n"
            f"👥 Total users: <b>{total_users}</b>\n"
            f"⏰ Pending schedules: <b>{pending_sched}</b>\n"
            f"🚫 Blocked users: <b>{blocked_users}</b>\n"
            f"🤗 Hugging Face: <b>{'OK' if _hf_client else 'OFF'}</b>\n"
            f"🗄️ Supabase: <b>{'OK' if supabase else 'OFF'}</b>"
        )

        await safe_send(
            lambda: query.message.edit_text(
                text,
                parse_mode="HTML",
                reply_markup=get_admin_dashboard_kb(),
            )
        )
        return

    if data == "admin_health":
        temp_ok = False
        ffmpeg_ok = bool(_FFMPEG_EXE and os.path.exists(_FFMPEG_EXE))

        try:
            temp_dir = _get_temp_dir()
            temp_ok = os.path.isdir(temp_dir) and os.access(temp_dir, os.W_OK)
        except Exception:
            temp_dir = "ERROR"

        text = (
            "🩺 <b>Bot Health</b>\n\n"
            f"🤖 Telegram bot: <b>OK</b>\n"
            f"🗄️ Supabase: <b>{'OK' if supabase else 'OFF'}</b>\n"
            f"🧠 Hugging Face: <b>{'OK' if _hf_client else 'OFF'}</b>\n"
            f"🎧 FFmpeg: <b>{'OK' if ffmpeg_ok else 'ERROR'}</b>\n"
            f"📁 Temp folder: <b>{'OK' if temp_ok else 'ERROR'}</b>\n"
            f"<code>{html.escape(str(temp_dir))}</code>"
        )

        await safe_send(
            lambda: query.message.edit_text(
                text,
                parse_mode="HTML",
                reply_markup=get_admin_dashboard_kb(),
            )
        )
        return

    if data == "admin_broadcast":
        await safe_send(
            lambda: query.message.reply_text(
                "📢 ប្រើ command:\n\n<code>/broadcast</code>",
                parse_mode="HTML",
            )
        )
        return

    if data == "admin_schedules":
        await safe_send(
            lambda: query.message.reply_text(
                "⏰ ប្រើ command:\n\n<code>/schedules</code>",
                parse_mode="HTML",
            )
        )
        return

    if data == "admin_users":
        await safe_send(
            lambda: query.message.reply_text(
                "👥 ប្រើ command:\n\n<code>/users</code>",
                parse_mode="HTML",
            )
        )
        return
    
async def sched_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    loop = asyncio.get_running_loop()

    if data.startswith("sched_ok:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text("❌ រកមិនឃើញ Schedule ។"))
            return
        if row["status"] != "pending":
            st = row["status"]
            await safe_send(
                lambda: query.message.reply_text(
                    f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាចបញ្ជាក់ទៀតទេ។",
                    parse_mode="HTML",
                )
            )
            return
        dt_str = _fmt_dt(datetime.fromisoformat(row["broadcast_at"]))
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(
            lambda: query.message.reply_text(
                f"✅ <b>Schedule #{row_id} បានបញ្ជាក់!</b>\n⏰ នឹង Broadcast នៅ {dt_str}",
                parse_mode="HTML",
            )
        )
        return

    if data.startswith("sched_no:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if row and row["status"] == "pending":
            await loop.run_in_executor(None, db_sched_set_status, row_id, "cancelled")
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(
            lambda: query.message.reply_text(
                f"❌ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
            )
        )
        return

    if data == "sched_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "sched_noop":
        return

    if data.startswith("sched_page:"):
        page = int(data.split(":")[1])
        rows = await loop.run_in_executor(None, db_sched_fetch_admin_pending, user_id)
        with suppress(Exception):
            await query.message.edit_reply_markup(
                reply_markup=get_schedules_list_kb(rows, page=page)
            )
        return

    if data.startswith("sched_view:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row:
            await safe_send(lambda: query.message.reply_text("❌ រកមិនឃើញ Schedule ។"))
            return
        try:
            dt_str = _fmt_dt(datetime.fromisoformat(row["broadcast_at"]))
        except Exception:
            dt_str = str(row.get("broadcast_at", "?"))
        content = (row.get("plain_text") or row.get("caption") or "(photo)")[:300]
        cancel_kb = (
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("🗑️ Cancel Schedule", callback_data=f"sched_cancel_confirm:{row_id}")]]
            )
            if row["status"] == "pending"
            else None
        )
        await safe_send(
            lambda: query.message.reply_text(
                f"📋 <b>Schedule #{row_id}</b>\n"
                f"⏰ {dt_str}\n"
                f"ស្ថានភាព: <b>{row['status']}</b>\n\n"
                f"{html.escape(content)}",
                parse_mode="HTML",
                reply_markup=cancel_kb,
            )
        )
        return

    if data.startswith("sched_cancel_confirm:"):
        row_id = int(data.split(":")[1])
        row = await loop.run_in_executor(None, db_sched_fetch_one, row_id)
        if not row or row.get("admin_id") != user_id:
            await safe_send(
                lambda: query.message.reply_text("⛔ អ្នកមិនមានសិទ្ធិ cancel Schedule នេះ។")
            )
            return
        if row["status"] != "pending":
            st = row["status"]
            await safe_send(
                lambda: query.message.reply_text(
                    f"⚠️ Schedule #{row_id} មានស្ថានភាព <b>{st}</b> — មិនអាច cancel ។",
                    parse_mode="HTML",
                )
            )
            return
        await loop.run_in_executor(None, db_sched_set_status, row_id, "cancelled")
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)
        await safe_send(
            lambda: query.message.reply_text(
                f"✅ Schedule <b>#{row_id}</b> បានបោះបង់។", parse_mode="HTML"
            )
        )

def _is_subtitle_file(filename: str | None) -> bool:
    if not filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in SUBTITLE_EXTENSIONS


def _decode_text_bytes(data: bytes) -> str:
    if not data:
        return ""

    # UTF-8 BOM
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]

    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="ignore")


def _clean_subtitle_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")

    # Remove WEBVTT header and metadata
    text = re.sub(r"^\ufeff?WEBVTT.*?(?:\n\n|\Z)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(NOTE|STYLE|REGION)\b.*?(?:\n\n|\Z)", "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    lines: list[str] = []
    previous = ""

    timestamp_re = re.compile(
        r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[\.,]\d{1,3}\s*-->\s*"
        r"(?:\d{1,2}:)?\d{1,2}:\d{2}[\.,]\d{1,3}.*$"
    )

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Remove subtitle index numbers
        if line.isdigit():
            continue

        # Remove timestamp lines
        if timestamp_re.match(line):
            continue

        # Remove common subtitle tags
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\{\\.*?\}", "", line)
        line = re.sub(r"^\s*[-–—]\s*", "", line)

        # Remove sound labels like [music], (laughing)
        line = re.sub(r"^\[[^\]]{1,40}\]$", "", line)
        line = re.sub(r"^\([^\)]{1,40}\)$", "", line)

        line = html.unescape(line).strip()

        if not line:
            continue

        # Avoid duplicate consecutive subtitle lines
        if line == previous:
            continue

        lines.append(line)
        previous = line

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if len(cleaned) > MAX_SUBTITLE_CHARS:
        cleaned = cleaned[:MAX_SUBTITLE_CHARS].rsplit("\n", 1)[0].strip()

    return cleaned


async def _download_telegram_file_to_bytes(tg_file, max_bytes: int) -> bytes:
    bio = io.BytesIO()
    await tg_file.download_to_memory(out=bio)
    data = bio.getvalue()

    if len(data) > max_bytes:
        raise ValueError(f"File too large. Max {max_bytes // 1024 // 1024} MB.")

    return data

async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user

    if not msg or not user or not msg.document:
        return

    sync_user_data(user)

    user_id = user.id
    chat_id = msg.chat_id
    document = msg.document

    filename = document.file_name or ""
    mime_type = document.mime_type or ""

    # Subtitle / text reader: .srt / .vtt / .txt
    if _is_subtitle_file(filename):
        status_msg = await safe_send(
            lambda: msg.reply_text("🎬 កំពុងអាន subtitle/text file...")
        )

        try:
            if document.file_size and document.file_size > MAX_SUBTITLE_BYTES:
                await safe_send(
                    lambda: msg.reply_text(
                        f"❌ File ធំពេក។ Max {MAX_SUBTITLE_BYTES // 1024 // 1024} MB."
                    )
                )
                return

            tg_file = await context.bot.get_file(document.file_id)
            file_bytes = await _download_telegram_file_to_bytes(
                tg_file,
                MAX_SUBTITLE_BYTES,
            )

            raw_text = _decode_text_bytes(file_bytes)
            cleaned_text = _clean_subtitle_text(raw_text)

            if not cleaned_text:
                await safe_send(
                    lambda: msg.reply_text("❌ មិនមានអត្ថបទអាចអានបានក្នុង file នេះទេ។")
                )
                return

            preview = cleaned_text[:1200]
            if len(cleaned_text) > 1200:
                preview += "\n\n..."

            sent = await safe_send(
                lambda: msg.reply_text(
                    "🎬 <b>Subtitle/Text detected</b>\n\n"
                    f"📄 File: <code>{html.escape(filename)}</code>\n"
                    f"🔤 Characters: <b>{len(cleaned_text)}</b>\n\n"
                    f"<blockquote>{html.escape(preview)}</blockquote>\n\n"
                    "ចុច ▶️ អាន ដើម្បីបំលែងទៅសំឡេង។",
                    parse_mode="HTML",
                    reply_markup=get_ocr_confirm_kb(msg.message_id),
                )
            )

            # Save cleaned text using original document message_id.
            # Button doc_read:<msg.message_id> will read this cache.
            save_text_cache(
                msg.message_id,
                cleaned_text,
                chat_id=chat_id,
                user_id=user_id,
                username=user.username or user.first_name,
            )

            set_last_tts_text(user_id, cleaned_text)
            record_turn(user_id, "user", f"[subtitle file] {filename}")
            record_turn(user_id, "assistant", cleaned_text[:CONV_CONTEXT_MAX_CHARS])

        except Exception as e:
            logger.error(f"subtitle reader error: {e}", exc_info=True)
            await safe_send(
                lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការអាន subtitle/text file។")
            )

        finally:
            if status_msg:
                with suppress(Exception):
                    await status_msg.delete()

        return

    # Keep your old document/audio/image logic below this line.
    # If this function replaces your old on_document fully, unsupported files show this:
    await safe_send(
        lambda: msg.reply_text(
            "❌ File នេះមិនទាន់ support ទេ។\n\n"
            "Support subtitle/text:\n"
            "• .srt\n"
            "• .vtt\n"
            "• .txt"
        )
    )
async def _cb_doc_read(query, user_id: int, context, data: str):
    try:
        src_msg_id = int(data.split(":")[1])
    except Exception:
        await safe_send(lambda: query.message.reply_text("❌ Invalid document cache id."))
        return

    if query.message is None:
        return

    chat_id = query.message.chat.id
    loop = asyncio.get_running_loop()

    full_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, src_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not full_text:
        fallback = get_last_tts_text(user_id)
        full_text = fallback or ""

    if not full_text:
        await safe_send(
            lambda: query.message.reply_text(
                "❌ រកអត្ថបទមិនឃើញ។ សូមផ្ញើ file ម្តងទៀត។"
            )
        )
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    speed = prefs["speed"]
    uname = query.from_user.username or query.from_user.first_name or str(user_id)

    with suppress(Exception):
        await query.message.delete()

    set_last_tts_text(user_id, full_text)

    tts_stop = asyncio.Event()
    tts_timer = asyncio.create_task(
        send_status_timer(chat_id, context.bot, tts_stop)
    )

    lock = _get_user_lock(user_id)

    try:
        async with lock:
            await _deliver_paged_tts(
                chat_id=chat_id,
                bot=context.bot,
                text=full_text,
                gender=gender,
                speed=speed,
                user_id=user_id,
                username=uname,
            )

            record_turn(user_id, "assistant", full_text[:CONV_CONTEXT_MAX_CHARS])
            _set_last_tts(user_id)

    except Exception as e:
        logger.error(f"_cb_doc_read delivery error: {e}", exc_info=True)
        with suppress(Exception):
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។",
            )

    finally:
        await _stop_timer(tts_stop, tts_timer)
async def _fire_scheduled_broadcast(bot, row: dict) -> None:
    row_id = row["id"]
    admin_id = row["admin_id"]
    logger.info(f"Firing scheduled broadcast #{row_id} for admin {admin_id}")
    loop = asyncio.get_running_loop()

    claimed = await loop.run_in_executor(None, db_sched_claim, row_id)
    if not claimed:
        logger.warning(f"Scheduled broadcast #{row_id} already claimed — skipping.")
        return

    pending = {
        "photo_file_id": row.get("photo_file_id"),
        "caption": row.get("caption") or "",
        "text": row.get("plain_text") or "",
    }

    sent, failed, blocked = await _run_broadcast_to_all(
        bot, admin_id, pending, label=f"Scheduled #{row_id}"
    )

    try:
        if supabase:
            await loop.run_in_executor(
                None,
                lambda: supabase.table("scheduled_broadcasts")
                .update({
                    "status": "done",
                    "sent_count": sent,
                    "failed_count": failed,
                    "blocked_count": blocked,
                })
                .eq("id", row_id)
                .execute(),
            )
    except Exception as e:
        logger.error(f"Could not mark scheduled broadcast #{row_id} done: {e}")

_scheduler_tasks: set[asyncio.Task] = set()

async def _scheduler_loop(bot, stop_event: asyncio.Event) -> None:
    logger.info("Scheduled broadcast loop started.")
    while not stop_event.is_set():
        try:
            loop = asyncio.get_running_loop()
            due = await loop.run_in_executor(None, db_sched_fetch_due)
            for row in due:
                task = asyncio.create_task(_fire_scheduled_broadcast(bot, row))
                _scheduler_tasks.add(task)
                task.add_done_callback(_scheduler_tasks.discard)
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=float(_SCHED_POLL_INTERVAL))
        except asyncio.TimeoutError:
            pass
    logger.info("Scheduled broadcast loop stopped.")

# ===========================================================================
# PER-USER CHAT
# ===========================================================================
@admin_only
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users = await asyncio.get_running_loop().run_in_executor(None, get_all_users_with_names)
    if not users:
        await safe_send(
            lambda: update.message.reply_text("❌ គ្មានអ្នកប្រើប្រាស់ registered ទេ។")
        )
        return
    await safe_send(
        lambda: update.message.reply_text(
            f"👥 <b>អ្នកប្រើប្រាស់ ({len(users)} នាក់)</b>\nចុចលើឈ្មោះ ដើម្បីចាប់ផ្ដើម Chat ។",
            parse_mode="HTML",
            reply_markup=get_users_page_kb(users, page=0),
        )
    )

@admin_only
async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    args = context.args or []
    if not args or not args[0].isdigit():
        await safe_send(
            lambda: update.message.reply_text(
                "❌ Usage: /chat <user_id>\nឬប្រើ /users ដើម្បីជ្រើស user ។"
            )
        )
        return
    target_id = int(args[0])
    exists = await asyncio.get_running_loop().run_in_executor(None, user_exists_in_db, target_id)
    if not exists:
        await safe_send(
            lambda: update.message.reply_text(
                f"❌ User <code>{target_id}</code> មិនមាននៅក្នុង Database ។",
                parse_mode="HTML",
            )
        )
        return
    await _open_chat_session(context.bot, admin_id, target_id, context)
    await safe_send(
        lambda: update.message.reply_text(
            f"💬 <b>Chat Mode បើក</b>\n\n"
            f"កំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
            "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
            "វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។",
            parse_mode="HTML",
        )
    )

@admin_only
async def cmd_endchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = update.effective_user.id
    target_id = _close_session(admin_id)
    context.user_data.pop("chat_state", None)
    if target_id is None:
        await safe_send(lambda: update.message.reply_text("ℹ️ អ្នកមិនទាន់ open Chat ណាមួយទេ។"))
        return
    await safe_send(
        lambda: update.message.reply_text(
            f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។",
            parse_mode="HTML",
        )
    )
    with suppress(Exception):
        await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")

async def _open_chat_session(bot, admin_id: int, target_id: int, context):
    _open_session(admin_id, target_id)
    context.user_data["chat_state"] = CHAT_WAIT_MESSAGE
    with suppress(Exception):
        await bot.send_message(
            chat_id=target_id, text="🔔 Admin ចង់ Chat ជាមួយអ្នក។ ផ្ញើសារតបមកបាន!"
        )

async def users_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if not _is_admin(user_id):
        with suppress(Exception):
            await query.answer("⛔ អ្នកមិនមានសិទ្ធិ។", show_alert=True)
        return

    with suppress(Exception):
        await query.answer()

    if data == "users_close":
        with suppress(Exception):
            await query.message.delete()
        return

    if data == "noop":
        return

    if data.startswith("users_page:"):
        page = int(data.split(":")[1])
        users = await asyncio.get_running_loop().run_in_executor(None, get_all_users_with_names)
        with suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=get_users_page_kb(users, page=page))
        return

    if data.startswith("chat_open:"):
        target_id = int(data.split(":")[1])
        admin_id = user_id
        with suppress(Exception):
            await query.message.delete()
        await _open_chat_session(context.bot, admin_id, target_id, context)
        await safe_send(
            lambda: context.bot.send_message(
                chat_id=admin_id,
                text=(
                    f"💬 <b>Chat Mode បើក</b>\n\n"
                    f"កំពុង Chat ជាមួយ User <code>{target_id}</code>\n"
                    "សារ/រូបភាព/Voice ផ្ញើនឹងទៅដល់ User ។\n\n"
                    "វាយ /endchat ឬ /cancel ដើម្បីបញ្ចប់។"
                ),
                parse_mode="HTML",
            )
        )

async def _fwd_admin_to_user(bot, admin_id: int, target_id: int, msg) -> bool:
    async def _do():
        if msg.text:
            await bot.send_message(
                chat_id=target_id,
                text=f"📩 <b>Admin:</b>\n{html.escape(msg.text)}",
                parse_mode="HTML",
            )
        elif msg.photo:
            cap = (f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>")
            await bot.send_photo(chat_id=target_id, photo=msg.photo[-1].file_id, caption=cap, parse_mode="HTML")
        elif msg.voice:
            await bot.send_voice(chat_id=target_id, voice=msg.voice.file_id, caption="📩 Admin voice message")
        elif msg.video_note:
            await bot.send_video_note(chat_id=target_id, video_note=msg.video_note.file_id)
        elif msg.sticker:
            await bot.send_sticker(chat_id=target_id, sticker=msg.sticker.file_id)
        elif msg.document:
            cap = (f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>")
            await bot.send_document(chat_id=target_id, document=msg.document.file_id, caption=cap, parse_mode="HTML")
        elif msg.video:
            cap = (f"📩 <b>Admin:</b>\n{html.escape(msg.caption)}" if msg.caption else "📩 <b>Admin:</b>")
            await bot.send_video(chat_id=target_id, video=msg.video.file_id, caption=cap, parse_mode="HTML")
        elif msg.audio:
            await bot.send_audio(chat_id=target_id, audio=msg.audio.file_id)
        else:
            await bot.forward_message(chat_id=target_id, from_chat_id=admin_id, message_id=msg.message_id)

    try:
        await safe_send(_do)
        return True
    except Forbidden:
        return False
    except Exception as e:
        logger.error(f"_fwd_admin_to_user error: {e}")
        return False

async def _fwd_user_to_admin(bot, admin_id: int, user_id: int, username: str, msg) -> bool:
    banner = f"💬 <b>{html.escape(username)} ({user_id}):</b>"

    async def _do():
        if msg.text:
            await bot.send_message(chat_id=admin_id, text=f"{banner}\n{html.escape(msg.text)}", parse_mode="HTML")
        elif msg.photo:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_photo(chat_id=admin_id, photo=msg.photo[-1].file_id, caption=cap, parse_mode="HTML")
        elif msg.voice:
            await bot.send_voice(chat_id=admin_id, voice=msg.voice.file_id, caption=banner, parse_mode="HTML")
        elif msg.video_note:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_video_note(chat_id=admin_id, video_note=msg.video_note.file_id)
        elif msg.sticker:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_sticker(chat_id=admin_id, sticker=msg.sticker.file_id)
        elif msg.document:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_document(chat_id=admin_id, document=msg.document.file_id, caption=cap, parse_mode="HTML")
        elif msg.video:
            cap = f"{banner}\n{html.escape(msg.caption)}" if msg.caption else banner
            await bot.send_video(chat_id=admin_id, video=msg.video.file_id, caption=cap, parse_mode="HTML")
        elif msg.audio:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.send_audio(chat_id=admin_id, audio=msg.audio.file_id)
        else:
            await bot.send_message(chat_id=admin_id, text=banner, parse_mode="HTML")
            await bot.forward_message(chat_id=admin_id, from_chat_id=user_id, message_id=msg.message_id)

    try:
        await safe_send(_do)
        return True
    except Exception as e:
        logger.error(f"_fwd_user_to_admin error: {e}")
        return False

# ===========================================================================
# STATS + CANCEL
# ===========================================================================
@admin_only
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loop = asyncio.get_running_loop()
    user_ids, pending_scheds = await asyncio.gather(
        loop.run_in_executor(None, get_all_user_ids),
        loop.run_in_executor(None, db_sched_fetch_admin_pending, update.effective_user.id),
    )
    active_chats = len(_admin_chat_target)
    await safe_send(
        lambda: update.message.reply_text(
            f"📊 <b>Bot Statistics</b>\n\n"
            f"👥 អ្នកប្រើប្រាស់សរុប: <b>{len(user_ids)}</b>\n"
            f"💬 Active Admin Chats: <b>{active_chats}</b>\n"
            f"📅 Scheduled (pending): <b>{len(pending_scheds)}</b>\n"
            f"🔒 Active user locks: <b>{len(_user_locks)}</b>\n"
            f"💭 History cache entries: <b>{len(_hist_cache)}</b>\n"
            f"🤗 HF Model: <b>{HF_MODEL}</b>\nOCR: <b>{HF_OCR_MODEL}</b>",
            parse_mode="HTML",
        )
    )

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not _is_admin(uid):
        return
    cleared = False

    if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
        _pending_broadcast.pop(uid, None)
        context.user_data.pop("bc_state", None)
        await safe_send(lambda: update.message.reply_text("❌ Broadcast បានបោះបង់។"))
        cleared = True

    if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _close_session(uid)
        context.user_data.pop("chat_state", None)
        if target_id:
            await safe_send(
                lambda: update.message.reply_text(
                    f"✅ Chat ជាមួយ User <code>{target_id}</code> បានបញ្ចប់។",
                    parse_mode="HTML",
                )
            )
            with suppress(Exception):
                await context.bot.send_message(chat_id=target_id, text="ℹ️ Admin បានបញ្ចប់ Session Chat ។")
        cleared = True

    if context.user_data.get("sched_state") in (SCHED_WAIT_MSG, SCHED_WAIT_TIME):
        _sched_payload.pop(uid, None)
        context.user_data.pop("sched_state", None)
        await safe_send(lambda: update.message.reply_text("❌ Schedule flow បានបោះបង់។"))
        cleared = True

    if not cleared:
        await safe_send(lambda: update.message.reply_text("ℹ️ មិនមាន operation ត្រូវ cancel ទេ។"))

# ===========================================================================
# REGULAR HANDLERS
# ===========================================================================
_BOT_START_TIME: float = 0.0

async def _drop_stale_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _BOT_START_TIME == 0.0:
        return
    msg = update.message or update.edited_message
    if msg and hasattr(msg, "date") and msg.date:
        update_ts = msg.date.timestamp()
        if update_ts < (_BOT_START_TIME - _STALE_GRACE_S):
            logger.debug(
                f"Dropping stale update (id={update.update_id}, "
                f"age={_BOT_START_TIME - update_ts:.1f}s)"
            )
            raise ApplicationHandlerStop

async def _drop_stale_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query:
        return
    if _BOT_START_TIME == 0.0:
        return
    msg = update.callback_query.message
    if msg is None:
        return
    if msg.date and msg.date.timestamp() < (_BOT_START_TIME - _STALE_GRACE_S):
        logger.debug(f"Dropping stale callback (id={update.update_id})")
        raise ApplicationHandlerStop

async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sync_user_data(update.effective_user)
        await safe_send(
            lambda: update.message.reply_text(
                WELCOME_TEXT,
                reply_markup=ReplyKeyboardRemove(),
                disable_web_page_preview=True,
            )
        )
    except Exception as e:
        logger.error(f"on_start error: {e}")

async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await on_start(update, context)

async def cmd_myprefs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    prefs = await get_user_prefs_async(user_id)
    gender_label = "👩 ស្រី (Female)" if prefs["gender"] == "female" else "👨 ប្រុស (Male)"
    speed_label = next(
        (lbl for _, (lbl, val) in SPEED_OPTIONS.items() if abs(val - prefs["speed"]) < 0.01),
        f"{prefs['speed']}x",
    )
    await safe_send(
        lambda: update.message.reply_text(
            f"⚙️ <b>ការកំណត់របស់អ្នក</b>\n\n"
            f"🗣️ សំឡេង: <b>{gender_label}</b>\n"
            f"🎚️ ល្បឿន: <b>{speed_label}</b>\n\n"
            "ផ្ញើ text ណាមួយ ហើយប្រើប៊ូតុងក្រោមសំឡេង ដើម្បីប្តូរ។",
            parse_mode="HTML",
            reply_markup=get_main_kb(prefs["gender"]),
        )
    )

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _hist_cache_clear(user_id)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_DB_EXECUTOR, db_history_clear, user_id)
    await safe_send(
        lambda: update.message.reply_text(
            "🗑️ ប្រវត្តិការសន្ទនារបស់អ្នកបានលុបចេញហើយ។\nBot នឹងចាប់ផ្ដើមការសន្ទនាថ្មី។"
        )
    )

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    if _is_admin(user_id):
        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        elif context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        elif context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                reply = (
                    f"✅ Photo ផ្ញើដល់ User <code>{target_id}</code> ។"
                    if ok
                    else f"❌ User <code>{target_id}</code> blocked bot ។"
                )
                await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
                if not ok:
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ រូបភាពបានផ្ញើដល់ Admin ។"))
        return

    if not _ocr_configured():
        await safe_send(lambda: msg.reply_text(_ocr_status_for_user()))
        return

    if await _check_cooldown(msg, user_id):
        return

    sync_user_data(user)
    uname = user.username or user.first_name or str(user_id)
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_OCR_FRAMES)
    )
    best_photo = msg.photo[-1]
    img_path = None
    try:
        img_path = _make_temp_img(suffix=".jpg")
        tg_file = await safe_send(lambda: context.bot.get_file(best_photo.file_id))
        if not tg_file:
            raise RuntimeError("Could not download photo.")
        await tg_file.download_to_drive(img_path)
        mime_type = _detect_image_mime(img_path)
        ocr_text = await ocr_image(img_path, mime_type=mime_type)

        if not ocr_text or ocr_text.upper() == "NOTEXT":
            await safe_send(lambda: msg.reply_text("🖼️ រូបភាពនេះមិនមានអត្ថបទដែលអាចអានបាន។"))
            return

        record_turn(user_id, "user", f"[Image OCR]: {ocr_text[:500]}")

        is_khmer = bool(_KHMER_RE.search(ocr_text))
        lang_flag = "🇰🇭" if is_khmer else "🇺🇸"
        header = f"🔍 <b>OCR {lang_flag}</b>\n\n"
        plain_pages = _paginate_plain(ocr_text, limit=TELE_MSG_LIMIT - len(header))
        sent_pages = []
        for page_idx, plain_page in enumerate(plain_pages):
            page_body = (header if page_idx == 0 else "") + html.escape(plain_page)
            sent = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, ocr_text,
                chat_id=msg.chat_id, user_id=user_id, username=uname,
            )
            await safe_send(
                lambda: last_sent.edit_reply_markup(
                    reply_markup=get_ocr_confirm_kb(last_sent.message_id)
                )
            )
    except Exception as e:
        err_msg = str(e) or repr(e)
        logger.error(f"on_photo OCR error: {type(e).__name__}: {e!r}", exc_info=True)
        if _is_dns_or_network_error(err_msg):
            user_msg = (
                "❌ OCR network/DNS មានបញ្ហា។ Server មិនអាចភ្ជាប់ទៅ Provider OCR បាន។\n"
                "✅ Code នេះនឹងសាក fallback provider ដោយស្វ័យប្រវត្តិ បើអ្នកបាន Set OCR_PROVIDER=auto និង GEMINI_API_KEY/GEMINI_MODEL។"
            )
        elif "model/provider is not available" in err_msg:
            user_msg = (
                "❌ Hugging Face OCR model នេះមិន support hosted inference ទេ។\n"
                "សូមសាក HF_OCR_MODEL=microsoft/trocr-base-printed ជាមុនសិន។"
            )
        else:
            user_msg = f"❌ មានបញ្ហាក្នុងការ OCR រូបភាព។\n{html.escape(err_msg)[:1000]}"
        await safe_send(lambda: msg.reply_text(user_msg, parse_mode=None))
    finally:
        await _stop_timer(stop_event, timer_task)
        if img_path:
            _cleanup(img_path)

async def on_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    # ── Admin flow intercepts ───────────────────────────────────────────
    if _is_admin(user_id):
        # Admin is in a scheduled broadcast/content flow — non-audio docs
        # should not be intercepted here; only audio files get transcribed.
        # Forward to user if in chat session regardless of file type.
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                reply = (
                    f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។"
                    if ok
                    else f"❌ User <code>{target_id}</code> blocked bot ។"
                )
                await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
                if not ok:
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    # ── User in admin chat session — forward back ───────────────────────
    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ ឯកសារបានផ្ញើដល់ Admin ។"))
        return

    doc = msg.document
    audio = msg.audio

    if doc is not None:
        filename = doc.file_name or ""
        mime_type = doc.mime_type or ""
        file_id = doc.file_id
        file_size = doc.file_size or 0
    elif audio is not None:
        filename = audio.file_name or ""
        mime_type = audio.mime_type or ""
        file_id = audio.file_id
        file_size = audio.file_size or 0
    else:
        return

    # Subtitle/Text reader: .srt / .vtt / .txt
    # This must run before the audio-only check because subtitle files arrive as documents.
    if doc is not None and _is_subtitle_file(filename):
        await on_document(update, context)
        return

    # Only process recognised audio files; silently skip everything else
    if not _is_audio_file(filename, mime_type):
        return

    if not _gemini:
        await safe_send(
            lambda: msg.reply_text(
                "❌ Gemini API មិន Activate ទេ។ មុខងារ Transcribe ត្រូវការ GEMINI_API_KEY ។"
            )
        )
        return

    if file_size > MAX_AUDIO_FILE_BYTES:
        await safe_send(
            lambda: msg.reply_text(
                f"❌ ឯកសារអូឌីយ៉ូធំពេក (អតិបរមា {MAX_AUDIO_FILE_BYTES // 1024 // 1024}MB)។"
            )
        )
        return

    if await _check_cooldown(msg, user_id):
        return

    sync_user_data(user)
    uname = user.username or user.first_name or str(user_id)

    ext = os.path.splitext(filename)[1].lower() if filename else ".mp3"
    if ext not in _AUDIO_EXTENSIONS:
        ext = ".mp3"

    gemini_mime = _audio_mime_for_gemini(filename, mime_type)
    audio_path = _make_temp_audio(suffix=ext)

    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_AUDIO_FILE_FRAMES)
    )

    try:
        tg_file = await safe_send(lambda: context.bot.get_file(file_id))
        if not tg_file:
            raise RuntimeError("Could not download audio file.")
        await tg_file.download_to_drive(audio_path)

        transcript = await transcribe_audio_file(audio_path, gemini_mime)

        if not transcript:
            await safe_send(
                lambda: msg.reply_text("❌ រក Transcript មិនឃើញ — ឯកសារប្រហែលជាស្ងាត់ ឬ មិន Support ។")
            )
            return

        record_turn(user_id, "user", f"[Audio File Transcript]: {transcript[:500]}")

        is_khmer = bool(_KHMER_RE.search(transcript))
        lang_flag = "🇰🇭" if is_khmer else "🇺🇸"
        fname_display = html.escape(filename[:50]) if filename else "audio"
        header = f"🎵 <b>Transcript</b> {lang_flag} — <code>{fname_display}</code>\n\n"

        plain_pages = _paginate_plain(transcript, limit=TELE_MSG_LIMIT - len(header))
        sent_pages = []
        for page_idx, plain_page in enumerate(plain_pages):
            page_body = (header if page_idx == 0 else "") + html.escape(plain_page)
            sent = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, transcript,
                chat_id=msg.chat_id, user_id=user_id, username=uname,
            )
            await safe_send(
                lambda: last_sent.edit_reply_markup(
                    reply_markup=get_audio_file_kb(last_sent.message_id)
                )
            )

    except Exception as e:
        logger.error(f"on_audio_file error: {e}", exc_info=True)
        await safe_send(
            lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ Transcribe ឯកសារអូឌីយ៉ូ។")
        )
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(audio_path)


async def on_any_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok
                else f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)
                context.user_data.pop("chat_state", None)
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ ផ្ញើដល់ Admin ។"))

async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = update.effective_user
    user_id = user.id

    if _is_admin(user_id) and context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
        target_id = _admin_chat_target.get(user_id)
        if target_id:
            ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
            reply = (
                f"✅ Voice ផ្ញើដល់ User <code>{target_id}</code> ។"
                if ok
                else f"❌ User <code>{target_id}</code> blocked bot ។"
            )
            await safe_send(lambda: msg.reply_text(reply, parse_mode="HTML"))
            if not ok:
                _close_session(user_id)
                context.user_data.pop("chat_state", None)
        return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ Voice ផ្ញើដល់ Admin ។"))
        return

    if not _gemini:
        await safe_send(
            lambda: msg.reply_text("❌ Gemini API មិន Activate ទេ។ សូម Set GEMINI_API_KEY ។")
        )
        return

    if msg.voice.file_size and msg.voice.file_size > MAX_VOICE_BYTES:
        await safe_send(lambda: msg.reply_text("❌ ឯកសារសំឡេងធំពេក (អតិបរមា 20MB)។"))
        return

    if await _check_cooldown(msg, user_id):
        return

    sync_user_data(user)
    ogg_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(msg.chat_id, context.bot, stop_event, frames=_TRANSCRIBE_FRAMES)
    )
    try:
        voice_file = await safe_send(lambda: context.bot.get_file(msg.voice.file_id))
        if not voice_file:
            raise RuntimeError("Could not get voice file")
        await voice_file.download_to_drive(ogg_path)
        transcript = await transcribe_voice(ogg_path)

        if not transcript:
            await safe_send(lambda: msg.reply_text("❌ រក Transcript មិនឃើញ។"))
            return

        record_turn(user_id, "user", f"[Voice Transcript]: {transcript[:500]}")

        is_khmer = bool(_KHMER_RE.search(transcript))
        lang_flag = "🇰🇭" if is_khmer else "🇺🇸"
        header = f"📝 <b>Transcript</b> {lang_flag}\n\n"
        plain_pages = _paginate_plain(transcript, limit=TELE_MSG_LIMIT - len(header))
        sent_pages = []
        for page_idx, plain_page in enumerate(plain_pages):
            page_body = (header if page_idx == 0 else "") + html.escape(plain_page)
            sent = await safe_send(lambda pb=page_body: msg.reply_text(pb, parse_mode="HTML"))
            sent_pages.append(sent)
            await asyncio.sleep(0.2)

        last_sent = sent_pages[-1] if sent_pages else None
        if last_sent:
            save_text_cache(
                last_sent.message_id, transcript,
                chat_id=msg.chat_id, user_id=user_id,
                username=user.username or user.first_name,
            )
            await safe_send(
                lambda: last_sent.edit_reply_markup(
                    reply_markup=get_transcription_kb(last_sent.message_id)
                )
            )
    except Exception as e:
        logger.error(f"on_voice error: {e}")
        with suppress(Exception):
            await safe_send(lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការ Transcribe។"))
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(ogg_path)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return

    text = msg.text
    user = update.effective_user
    if not user:
        return

    user_id = user.id

    if _is_admin(user_id):
        sched_state = context.user_data.get("sched_state")
        if sched_state == SCHED_WAIT_MSG:
            await _handle_sched_content(update, context)
            return
        if sched_state == SCHED_WAIT_TIME:
            await _handle_sched_datetime(update, context)
            return
        if context.user_data.get("bc_state") == BROADCAST_WAIT_MESSAGE:
            await broadcast_receive(update, context)
            return
        if context.user_data.get("chat_state") == CHAT_WAIT_MESSAGE:
            target_id = _admin_chat_target.get(user_id)
            if target_id:
                ok = await _fwd_admin_to_user(context.bot, user_id, target_id, msg)
                if ok:
                    await safe_send(
                        lambda: msg.reply_text(
                            f"✅ ផ្ញើដល់ User <code>{target_id}</code> ។",
                            parse_mode="HTML",
                        )
                    )
                else:
                    await safe_send(
                        lambda: msg.reply_text(
                            f"❌ User <code>{target_id}</code> blocked bot ។ Chat session បានបិទ។",
                            parse_mode="HTML",
                        )
                    )
                    _close_session(user_id)
                    context.user_data.pop("chat_state", None)
            return

    admin_id = _get_admin_for_user(user_id)
    if admin_id is not None:
        uname = user.username or user.first_name or str(user_id)
        await _fwd_user_to_admin(context.bot, admin_id, user_id, uname, msg)
        await safe_send(lambda: msg.reply_text("✅ សាររបស់អ្នកបានផ្ញើដល់ Admin ។"))
        return

    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    stripped = text.strip()
    if not stripped:
        return

    if len(stripped) > MAX_INPUT_CHARS:
        await safe_send(
            lambda: msg.reply_text(
                f"❌ អត្ថបទវែងពេក។ អតិបរមា {MAX_INPUT_CHARS} តួអក្សរ។\n"
                f"(អ្នកបានផ្ញើ {len(stripped)} តួ)"
            )
        )
        return

    if await _check_cooldown(msg, user_id):
        return

    sync_user_data(user)

    loop = asyncio.get_running_loop()
    prefs, tts_text = await asyncio.gather(
        get_user_prefs_async(user_id),
        resolve_tts_text(user_id, stripped, loop),
    )

    gender = prefs["gender"]
    speed = prefs["speed"]

    if not tts_text or not tts_text.strip():
        tts_text = stripped

    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(update.effective_chat.id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)

    async with lock:
        try:
            audio_bytes = await generate_voice_limited(
                tts_text,
                gender,
                speed,
                file_path,
            )

            sent_msg = await safe_send(
                lambda ab=audio_bytes: msg.reply_voice(
                    voice=io.BytesIO(ab),
                    caption=f"🗣️ {BOT_TAG}",
                    reply_markup=get_main_kb(gender),
                )
            )

            if sent_msg:
                save_text_cache(
                    sent_msg.message_id,
                    tts_text,
                    chat_id=msg.chat_id,
                    user_id=user_id,
                    username=user.username or user.first_name,
                )

                set_last_tts_text(user_id, tts_text)

                record_turn(user_id, "user", stripped)
                record_turn(user_id, "assistant", tts_text)

            _set_last_tts(user_id)

        except Exception as e:
            logger.error(f"on_text TTS error: {e}", exc_info=True)
            with suppress(Exception):
                await safe_send(
                    lambda: msg.reply_text("❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។")
                )

        finally:
            await _stop_timer(stop_event, timer_task)
            _cleanup(file_path)


# ---------------------------------------------------------------------------
# Cache key helper
# ---------------------------------------------------------------------------
def _cache_key_from_query(query) -> tuple[int, int]:
    return query.message.chat.id, query.message.message_id

# ---------------------------------------------------------------------------
# Callback dispatch helpers
# ---------------------------------------------------------------------------
async def _cb_show_speed(query, user_id: int, context):
    prefs = await get_user_prefs_async(user_id)
    await safe_send(
        lambda: query.message.edit_reply_markup(reply_markup=get_speed_kb(prefs["speed"]))
    )
async def get_callback_original_text(query, user_id: int) -> str | None:
    """Get original TTS text for callback buttons.

    Order:
    1. Exact text_cache by clicked message id
    2. In-memory latest TTS text
    3. conversation_history latest assistant message

    This fixes buttons shown on /myprefs/settings messages because those
    messages do not have an entry in text_cache.
    """
    if query.message is None:
        return get_last_tts_text(user_id)

    chat_id = query.message.chat.id
    msg_id = query.message.message_id
    loop = asyncio.get_running_loop()

    # 1. Exact cache for voice/transcript messages
    try:
        original_text = await loop.run_in_executor(
            None,
            get_text_cache,
            msg_id,
            chat_id,
        )
        if original_text:
            return original_text
    except Exception as e:
        logger.warning(
            f"get_callback_original_text text_cache failed "
            f"user={user_id} chat={chat_id} msg={msg_id}: {e}"
        )

    # 2. Latest generated TTS from memory
    original_text = get_last_tts_text(user_id)
    if original_text:
        return original_text

    # 3. Latest assistant message from conversation_history
    try:
        history = _hist_cache_get(user_id)

        if history is None:
            history = await loop.run_in_executor(
                _DB_EXECUTOR,
                db_history_fetch,
                user_id,
            )

            for row in history or []:
                _hist_cache_append(
                    user_id,
                    row.get("role", "user"),
                    row.get("content", ""),
                )

        for row in reversed(history or []):
            role = str(row.get("role", "") or "").strip().lower()
            if role in ("assistant", "bot", "model"):
                content = str(row.get("content", "") or "").strip()
                if content:
                    set_last_tts_text(user_id, content)
                    return content

    except Exception as e:
        logger.error(f"get_callback_original_text history fallback failed user={user_id}: {e}")

    return None

async def _cb_hide_speed(query, user_id: int, context):
    prefs = await get_user_prefs_async(user_id)
    await safe_send(
        lambda: query.message.edit_reply_markup(reply_markup=get_main_kb(prefs["gender"]))
    )

async def _cb_speed(query, user_id: int, context, data: str):
    _, new_speed = SPEED_OPTIONS[data]

    if query.message is None:
        return

    chat_id = query.message.chat.id

    original_text, prefs = await asyncio.gather(
        get_callback_original_text(query, user_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(
            lambda: query.message.reply_text(
                "❌ រកអត្ថបទដើមមិនឃើញ។\n"
                "សូមផ្ញើអត្ថបទម្តងទៀត រួចប្តូរល្បឿន។"
            )
        )
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    update_user_speed(user_id, new_speed)

    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(chat_id, context.bot, stop_event)
    )

    lock = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(
                    original_text,
                    gender,
                    new_speed,
                    file_path,
                )

                with suppress(Exception):
                    await query.message.delete()

                new_msg = await safe_send(
                    lambda ab=audio_bytes, g=gender: query.message.chat.send_voice(
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(g),
                    )
                )

                if new_msg:
                    save_text_cache(
                        new_msg.message_id,
                        original_text,
                        chat_id=chat_id,
                        user_id=user_id,
                        username=query.from_user.username or query.from_user.first_name,
                    )

                    set_last_tts_text(user_id, original_text)

                    record_turn(user_id, "assistant", original_text)

                _set_last_tts(user_id)

            except Exception as e:
                logger.error(f"speed regen error: {e}", exc_info=True)
                with suppress(Exception):
                    await safe_send(
                        lambda: query.message.reply_text(
                            "❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"
                        )
                    )

    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(file_path)

async def _cb_gender(query, user_id: int, context, data: str):
    new_gender = data.replace("tg_", "")

    if query.message is None:
        return

    chat_id = query.message.chat.id

    original_text, prefs = await asyncio.gather(
        get_callback_original_text(query, user_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(
            lambda: query.message.reply_text(
                "❌ រកអត្ថបទដើមមិនឃើញ។\n"
                "សូមផ្ញើអត្ថបទម្តងទៀត រួចប្តូរសំឡេង។"
            )
        )
        return

    if await _check_cooldown(query.message, user_id):
        return

    speed = prefs["speed"]
    update_user_gender(user_id, new_gender)

    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(chat_id, context.bot, stop_event)
    )

    lock = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(
                    original_text,
                    new_gender,
                    speed,
                    file_path,
                )

                with suppress(Exception):
                    await query.message.delete()

                new_msg = await safe_send(
                    lambda ab=audio_bytes, ng=new_gender: query.message.chat.send_voice(
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(ng),
                    )
                )

                if new_msg:
                    save_text_cache(
                        new_msg.message_id,
                        original_text,
                        chat_id=chat_id,
                        user_id=user_id,
                        username=query.from_user.username or query.from_user.first_name,
                    )

                    set_last_tts_text(user_id, original_text)

                    record_turn(user_id, "assistant", original_text)

                _set_last_tts(user_id)

            except Exception as e:
                logger.error(f"gender regen error: {e}", exc_info=True)
                with suppress(Exception):
                    await safe_send(
                        lambda: query.message.reply_text(
                            "❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"
                        )
                    )

    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(file_path)

async def _cb_tts_transcript(query, user_id: int, context, data: str):
    transcript_msg_id = int(data.split(":")[1])

    if query.message is None:
        return

    chat_id = query.message.chat.id
    loop = asyncio.get_running_loop()

    original_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, transcript_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not original_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    speed = prefs["speed"]
    file_path = _make_temp_ogg()
    stop_event = asyncio.Event()
    timer_task = asyncio.create_task(
        send_status_timer(chat_id, context.bot, stop_event)
    )
    lock = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                audio_bytes = await generate_voice_limited(original_text, gender, speed, file_path)
                new_msg = await safe_send(
                    lambda ab=audio_bytes, g=gender: query.message.chat.send_voice(
                        voice=io.BytesIO(ab),
                        caption=f"🗣️ {BOT_TAG}",
                        reply_markup=get_main_kb(g),
                    )
                )

                if new_msg:
                    save_text_cache(
                        new_msg.message_id,
                        original_text,
                        chat_id=chat_id,
                        user_id=user_id,
                        username=query.from_user.username or query.from_user.first_name,
                    )
                    set_last_tts_text(user_id, original_text)
                    record_turn(user_id, "assistant", original_text)

                _set_last_tts(user_id)

            except Exception as e:
                logger.error(f"transcript TTS error: {e}", exc_info=True)
                with suppress(Exception):
                    await safe_send(
                        lambda: query.message.reply_text(
                            "❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"
                        )
                    )
    finally:
        await _stop_timer(stop_event, timer_task)
        _cleanup(file_path)

async def _cb_audio_tts(query, user_id: int, context, data: str):
    src_msg_id = int(data.split(":")[1])
    chat_id = query.message.chat.id

    loop = asyncio.get_running_loop()
    full_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, src_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not full_text:
        await safe_send(
            lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ (cache expired)។")
        )
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    speed = prefs["speed"]
    uname = query.from_user.username or query.from_user.first_name or str(user_id)

    with suppress(Exception):
        await query.message.edit_reply_markup(reply_markup=None)

    tts_stop = asyncio.Event()
    tts_timer = asyncio.create_task(send_status_timer(chat_id, context.bot, tts_stop))

    lock = _get_user_lock(user_id)
    try:
        async with lock:
            try:
                await _deliver_paged_tts(
                    chat_id=chat_id, bot=context.bot, text=full_text,
                    gender=gender, speed=speed, user_id=user_id, username=uname,
                )
                record_turn(user_id, "assistant", full_text[:CONV_CONTEXT_MAX_CHARS])
            except Exception as e:
                logger.error(f"_cb_audio_tts delivery error: {e}")
                with suppress(Exception):
                    await context.bot.send_message(
                        chat_id=chat_id, text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។"
                    )
    finally:
        await _stop_timer(tts_stop, tts_timer)

async def _cb_doc_read(query, user_id: int, context, data: str):
    src_msg_id = int(data.split(":")[1])

    if query.message is None:
        return

    chat_id = query.message.chat.id
    loop = asyncio.get_running_loop()

    full_text, prefs = await asyncio.gather(
        loop.run_in_executor(None, get_text_cache, src_msg_id, chat_id),
        get_user_prefs_async(user_id),
    )

    if not full_text:
        await safe_send(lambda: query.message.reply_text("❌ រកអត្ថបទមិនឃើញ (cache expired)។"))
        return

    if await _check_cooldown(query.message, user_id):
        return

    gender = prefs["gender"]
    speed = prefs["speed"]
    uname = query.from_user.username or query.from_user.first_name or str(user_id)

    with suppress(Exception):
        await query.message.delete()

    tts_stop = asyncio.Event()
    tts_timer = asyncio.create_task(send_status_timer(chat_id, context.bot, tts_stop))
    lock = _get_user_lock(user_id)

    try:
        async with lock:
            try:
                set_last_tts_text(user_id, full_text)
                await _deliver_paged_tts(
                    chat_id=chat_id,
                    bot=context.bot,
                    text=full_text,
                    gender=gender,
                    speed=speed,
                    user_id=user_id,
                    username=uname,
                )
                record_turn(user_id, "assistant", full_text[:CONV_CONTEXT_MAX_CHARS])
            except Exception as e:
                logger.error(f"_cb_doc_read delivery error: {e}", exc_info=True)
                with suppress(Exception):
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="❌ មានបញ្ហាក្នុងការបង្កើតសំឡេង។",
                    )
    finally:
        await _stop_timer(tts_stop, tts_timer)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    # Answer first so Telegram doesn't show a loading spinner indefinitely.
    # Specific handlers (broadcast_callback, sched_callback, etc.) also call
    # query.answer() — that's fine; duplicate answers are silently ignored.
    with suppress(Exception):
        await query.answer()

    if query.message is None:
        logger.debug(f"on_callback: no message for data={data!r} (inline mode?)")
        return

    # These prefixes are fully handled by dedicated CallbackQueryHandlers
    # registered with higher priority — skip to avoid double processing.
    _HANDLED_PREFIXES = (
        "bc_confirm", "bc_cancel",
        "sched_",
        "users_page:", "users_close", "chat_open:", "noop",
    )
    if any(data == p or data.startswith(p) for p in _HANDLED_PREFIXES):
        return

    try:
        if data == "show_speed":
            await _cb_show_speed(query, user_id, context)
        elif data == "hide_speed":
            await _cb_hide_speed(query, user_id, context)
        elif data in SPEED_OPTIONS:
            await _cb_speed(query, user_id, context, data)
        elif data in ("tg_female", "tg_male"):
            await _cb_gender(query, user_id, context, data)
        elif data.startswith("tts_transcript:"):
            await _cb_tts_transcript(query, user_id, context, data)
        elif data.startswith("del_transcript:"):
            with suppress(Exception):
                await query.message.delete()
        elif data.startswith("doc_del:"):
            with suppress(Exception):
                await query.message.delete()
        elif data.startswith("doc_read:"):
            await _cb_doc_read(query, user_id, context, data)
        elif data.startswith("audio_tts:"):
            await _cb_audio_tts(query, user_id, context, data)
        elif data.startswith("audio_del:"):
            with suppress(Exception):
                await query.message.delete()
        elif data.startswith("admin_"):
            await _cb_admin_dashboard(query, user_id, context, data)
        else:
            logger.debug(f"on_callback: unhandled data={data!r}")
    except Exception as e:
        logger.error(f"on_callback unhandled [data={data}]: {e}", exc_info=True)

# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Unhandled exception: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        with suppress(Exception):
            await safe_send(
                lambda: update.effective_message.reply_text(
                    "⚠️ មានបញ្ហាបច្ចេកទេស។ Bot នៅដំណើរការ — សូមព្យាយាមម្តងទៀត។"
                )
            )

# ---------------------------------------------------------------------------
# Bot runner
# ---------------------------------------------------------------------------
async def _run_bot():
    global _BOT_START_TIME, _AI_SEMAPHORE, _BROADCAST_SEMAPHORE
    global _prefs_cache_lock, _TTS_CHUNK_SEMAPHORE

    _AI_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_AI)
    _BROADCAST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_BROADCAST)
    _prefs_cache_lock = asyncio.Lock()
    _TTS_CHUNK_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_TTS_USERS)

    _admin_chat_target.clear()
    _user_to_admin.clear()
    _pending_broadcast.clear()
    _prefs_cache.clear()
    _user_last_tts.clear()
    _sched_payload.clear()
    _scheduler_tasks.clear()
    _hist_cache.clear()
    _user_locks.clear()

    _BOT_START_TIME = time.time()

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    app.add_handler(TypeHandler(Update, _drop_stale_updates), group=-1)
    app.add_handler(CallbackQueryHandler(_drop_stale_callback), group=-1)

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("myprefs", cmd_myprefs))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("broadcast", broadcast_start))
    app.add_handler(CommandHandler("schedule", cmd_schedule))
    app.add_handler(CommandHandler("schedules", cmd_schedules))
    app.add_handler(CommandHandler("cancelschedule", cmd_cancelschedule))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("stats", admin_stats))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("users", cmd_users))
    app.add_handler(CommandHandler("chat", cmd_chat))
    app.add_handler(CommandHandler("endchat", cmd_endchat))

    app.add_handler(
        CallbackQueryHandler(broadcast_callback, pattern=r"^bc_(confirm|cancel)$")
    )
    app.add_handler(
        CallbackQueryHandler(
            users_page_callback,
            pattern=r"^(users_page:\d+|users_close|chat_open:\d+|noop)$",
        )
    )
    app.add_handler(CallbackQueryHandler(sched_callback, pattern=r"^sched_"))
    app.add_handler(CallbackQueryHandler(on_callback))

    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(
        MessageHandler(
            (filters.Document.ALL | filters.AUDIO) & ~filters.VOICE,
            on_audio_file,
        )
    )
    app.add_handler(
        MessageHandler(
            filters.Sticker.ALL | filters.VIDEO | filters.VIDEO_NOTE,
            on_any_media,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    logger.info(f"Bot polling started. Admins: {ADMIN_IDS or 'none configured'}")
    logger.info(f"HF model: {HF_MODEL}; OCR provider: {OCR_PROVIDER}; HF OCR model: {HF_OCR_MODEL}; Gemini OCR available: {_gemini is not None}")

    sched_stop = asyncio.Event()
    sweep_stop = asyncio.Event()
    sched_task = asyncio.create_task(_scheduler_loop(app.bot, sched_stop))
    sweep_task = asyncio.create_task(_periodic_temp_sweep(sweep_stop))

    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )
        try:
            await asyncio.Event().wait()
        finally:
            sched_stop.set()
            sweep_stop.set()
            for task in (sched_task, sweep_task):
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    load_dotenv()
    _init_clients()

    threading.Thread(target=run_flask, daemon=True).start()
    print("Flask health-check server started.")

    threading.Thread(target=keep_alive, daemon=True).start()
    print("Keep-alive thread started.")

    _sweep_stale_temps()
    ensure_speed_column()

    if not ADMIN_IDS:
        logger.warning(
            "No ADMIN_IDS configured. "
            "Set ADMIN_IDS=123456,789012 in your environment."
        )

    print(f"Bot is starting... (AI provider: {AI_PROVIDER}, HF: {HF_MODEL}, OCR provider: {OCR_PROVIDER}, HF OCR: {HF_OCR_MODEL})")
    while True:
        try:
            asyncio.run(_run_bot())
        except Exception as e:
            logger.error(f"Bot crashed: {e} — restarting in 5s...", exc_info=True)
        else:
            logger.warning("Polling stopped — restarting in 5s...")
        time.sleep(5)

if __name__ == "__main__":
    main()
