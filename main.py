import logging
import os
import re
import asyncio
import base64
import threading
import imageio_ffmpeg as _iio_ffmpeg
from flask import Flask
from supabase import create_client, Client

# ── Flask Web Server for Render Health Checks ──────────────────────────────
# Render requires a web server to stay alive.
app_flask = Flask(__name__)

@app_flask.route('/')
def health_check():
    return "Bot is running!", 200

def run_flask():
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host='0.0.0.0', port=port)

# ── FFmpeg Configuration ──────────────────────────────────────────────────
_FFMPEG_EXE = _iio_ffmpeg.get_ffmpeg_exe()

from google import genai
from google.genai import types
import edge_tts
from dotenv import load_dotenv
from telegram import (
    Update, constants,
    InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, CallbackQueryHandler,
)

# ---------------------------------------------------------------------------
# Configuration & Database Init
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")

SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SB_URL, SB_KEY)

VOICE_MAP = {
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    "en": {"female": "Kore", "male": "Puck"}, 
}

WELCOME_TEXT = (
    "🎵 សួស្តី! ខ្ញុំជា Bot បំលែងអក្សរទៅជាសំឡេង\n\n"
    "📌 វាយអក្សរភាសាណាមួយ ផ្ញើរមក Bot នឹងបំលែងដោយស្វ័យប្រវត្តិ!\n\n"
    "🌍 ភាសាដែល Support:\n"
    "🇰🇭 ភាសាខ្មែរ | 🇺🇸 English\n\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)

# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------
def sync_user_data(user):
    user_id = user.id
    username = user.username or user.first_name
    try:
        supabase.table("user_prefs").upsert({"user_id": user_id, "username": username}, on_conflict="user_id").execute()
    except Exception as e:
        logger.error(f"DB Sync Error: {e}")

def get_user_gender(user_id: int) -> str:
    try:
        res = supabase.table("user_prefs").select("gender").eq("user_id", user_id).execute()
        if res.data and res.data[0]['gender']:
            return res.data[0]['gender']
    except Exception as e:
        logger.error(f"Error fetching gender: {e}")
    return "female"

def update_user_gender(user_id: int, gender: str):
    supabase.table("user_prefs").update({"gender": gender}).eq("user_id", user_id).execute()

def save_text_cache(msg_id: int, text: str):
    supabase.table("text_cache").upsert({"message_id": msg_id, "original_text": text}).execute()

def get_text_cache(msg_id: int) -> str:
    res = supabase.table("text_cache").select("original_text").eq("message_id", msg_id).execute()
    return res.data[0]['original_text'] if res.data else None

# ---------------------------------------------------------------------------
# TTS & Audio (Async FFmpeg)
# ---------------------------------------------------------------------------
async def async_convert_audio(input_data: bytes | str, output_path: str, is_pcm: bool):
    cmd = [_FFMPEG_EXE, "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0" if is_pcm else input_data,
           "-c:a", "libopus", "-b:a", "32k", output_path]
    if not is_pcm: cmd = [_FFMPEG_EXE, "-y", "-i", input_data, "-c:a", "libopus", "-b:a", "32k", output_path]
    process = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE if is_pcm else None,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    if is_pcm: await process.communicate(input=input_data)
    else: await process.wait()

_gemini_client = None
def get_gemini():
    global _gemini_client
    if _gemini_client is None: _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

async def generate_voice(text: str, gender: str, output_path: str):
    is_khmer = bool(re.search(r"[\u1780-\u17FF]", text))
    if is_khmer:
        tmp_mp3 = f"{output_path}.mp3"
        await edge_tts.Communicate(text, VOICE_MAP["km"][gender]).save(tmp_mp3)
        await async_convert_audio(tmp_mp3, output_path, False)
        if os.path.exists(tmp_mp3): os.remove(tmp_mp3)
        return "🇰🇭 ភាសាខ្មែរ"
    else:
        resp = get_gemini().models.generate_content(model="gemini-2.5-flash-preview-tts",
            contents=[{"role": "user", "parts": [{"text": text}]}],
            config=types.GenerateContentConfig(response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_MAP["en"][gender])))),)
        pcm = base64.b64decode(resp.candidates[0].content.parts[0].inline_data.data)
        await async_convert_audio(pcm, output_path, True)
        return "🇺🇸 English"

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def get_voice_kb(gender: str):
    f_btn = "👩 សំឡេងស្រី" + (" ✅" if gender == "female" else "")
    m_btn = "👨 សំឡេងប្រុស" + (" ✅" if gender == "male" else "")
    return InlineKeyboardMarkup([[InlineKeyboardButton(f_btn, callback_data="tg_female"),
                                  InlineKeyboardButton(m_btn, callback_data="tg_male")]])

async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sync_user_data(update.effective_user)
    await update.message.reply_text(WELCOME_TEXT, reply_markup=ReplyKeyboardRemove(), disable_web_page_preview=True)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text: return
    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context)
        return

    sync_user_data(update.effective_user)
    user_id = update.effective_user.id
    gender = get_user_gender(user_id)
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.RECORD_VOICE)
    file_path = f"v_{user_id}_{update.message.message_id}.ogg"
    
    try:
        label = await generate_voice(text, gender, file_path)
        with open(file_path, "rb") as audio:
            sent_msg = await update.message.reply_voice(voice=audio, caption=f"🗣️ {label}", reply_markup=get_voice_kb(gender))
            save_text_cache(sent_msg.message_id, text)
    except Exception as e:
        logger.error(f"TTS Error: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    msg_id = query.message.message_id
    new_gender = query.data.replace("tg_", "")
    
    original_text = get_text_cache(msg_id)
    if not original_text:
        await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True)
        return

    update_user_gender(user_id, new_gender)
    await query.answer("🔄 កំពុងប្តូរសំឡេង...")
    
    file_path = f"rev_{user_id}_{msg_id}.ogg"
    try:
        label = await generate_voice(original_text, new_gender, file_path)
        try: await query.message.delete()
        except: pass
        with open(file_path, "rb") as audio:
            new_msg = await query.message.chat.send_voice(voice=audio, caption=f"🗣️ {label}", reply_markup=get_voice_kb(new_gender))
            save_text_cache(new_msg.message_id, original_text)
    except Exception as e:
        logger.error(f"Regen Error: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

# ---------------------------------------------------------------------------
# Main Deployment Logic
# ---------------------------------------------------------------------------
def main():
    # 1. Start Flask in a background thread
    threading.Thread(target=run_flask, daemon=True).start()
    print("✅ Health Check server started on background thread.")

    # 2. Start Telegram Bot
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    
    print("🚀 Bot is running and waiting for messages...")
    app.run_polling()

if __name__ == "__main__":
    main()
