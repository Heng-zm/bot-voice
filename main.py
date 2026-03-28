import logging
import os
import re
import asyncio
import base64
import threading
import imageio_ffmpeg as _iio_ffmpeg
from flask import Flask
from supabase import create_client, Client

# ── Flask Web Server សម្រាប់ Render Health Checks ──────────────────────────
app_flask = Flask(__name__)

@app_flask.route('/')
def health_check():
    return "Bot is running online!", 200

def run_flask():
    # Render ផ្ដល់ Port តាមរយៈ Environment Variable
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
# ការកំណត់ Configuration & Database
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")

# Supabase Connection
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
# មុខងារជំនួយ Database (Database Helpers)
# ---------------------------------------------------------------------------
def sync_user_data(user):
    try:
        supabase.table("user_prefs").upsert({
            "user_id": user.id, 
            "username": user.username or user.first_name
        }, on_conflict="user_id").execute()
    except Exception as e:
        logger.error(f"DB Sync Error (401?): {e}")

def get_user_gender(user_id: int) -> str:
    try:
        res = supabase.table("user_prefs").select("gender").eq("user_id", user_id).execute()
        if res.data and res.data[0].get('gender'):
            return res.data[0]['gender']
    except:
        pass
    return "female"

def update_user_gender(user_id: int, gender: str):
    try:
        supabase.table("user_prefs").update({"gender": gender}).eq("user_id", user_id).execute()
    except: pass

def save_text_cache(msg_id: int, text: str):
    try:
        supabase.table("text_cache").upsert({"message_id": msg_id, "original_text": text}).execute()
    except: pass

def get_text_cache(msg_id: int) -> str:
    try:
        res = supabase.table("text_cache").select("original_text").eq("message_id", msg_id).execute()
        return res.data[0]['original_text'] if res.data else None
    except: return None

def delete_text_cache(msg_id: int):
    try:
        supabase.table("text_cache").delete().eq("message_id", msg_id).execute()
    except: pass

# ---------------------------------------------------------------------------
# ម៉ាស៊ីនបំប្លែងសំឡេង (Async Audio Engine)
# ---------------------------------------------------------------------------
async def async_convert_audio(input_data: bytes | str, output_path: str, is_pcm: bool):
    """បំប្លែងអូឌីយ៉ូដោយប្រើ FFmpeg ផ្ទាល់ (លឿន និងមានស្ថេរភាពលើ Cloud)"""
    if is_pcm:
        cmd = [_FFMPEG_EXE, "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0", 
               "-c:a", "libopus", "-b:a", "32k", output_path]
    else:
        cmd = [_FFMPEG_EXE, "-y", "-i", input_data, "-c:a", "libopus", "-b:a", "32k", output_path]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE if is_pcm else None,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    if is_pcm:
        await process.communicate(input=input_data)
    else:
        await process.wait()

# ---------------------------------------------------------------------------
# TTS Routing Logic
# ---------------------------------------------------------------------------
_gemini_client = None
def get_gemini():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

async def generate_voice(text: str, gender: str, output_path: str):
    is_khmer = bool(re.search(r"[\u1780-\u17FF]", text))
    
    if is_khmer:
        # Khmer Flow (Edge-TTS)
        tmp_mp3 = f"{output_path}.mp3"
        await edge_tts.Communicate(text, VOICE_MAP["km"][gender]).save(tmp_mp3)
        await async_convert_audio(tmp_mp3, output_path, False)
        if os.path.exists(tmp_mp3): os.remove(tmp_mp3)
        return "🇰🇭 ភាសាខ្មែរ"
    else:
        # English Flow (Gemini AI with Retry logic and strict System Instruction)
        client = get_gemini()
        for attempt in range(2): # Retry up to 2 times for 500 error
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=[{"role": "user", "parts": [{"text": text}]}],
                    config=types.GenerateContentConfig(
                        system_instruction="You are a professional text-to-speech engine. Your only task is to read the provided text transcript exactly as it is. Do NOT generate any conversational text, comments, or answers. Output MUST be audio only.",
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_MAP["en"][gender])
                            )
                        ),
                    ),
                )
                if not resp.candidates or not resp.candidates[0].content.parts:
                    raise ValueError("Gemini Response is empty.")

                raw = resp.candidates[0].content.parts[0].inline_data.data
                pcm = base64.b64decode(raw) if isinstance(raw, str) else raw
                await async_convert_audio(pcm, output_path, True)
                return "🇺🇸 English"
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                raise e

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
            sent_msg = await update.message.reply_voice(
                voice=audio, 
                caption=f"🗣️ {label}",
                reply_markup=get_voice_kb(gender)
            )
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
        await query.answer("❌ រកអត្ថបទដើមមិនឃើញ (Cache Expired)។", show_alert=True)
        return

    if get_user_gender(user_id) == new_gender:
        await query.answer()
        return

    update_user_gender(user_id, new_gender)
    await query.answer("🔄 កំពុងប្តូរសំឡេង...")
    
    file_path = f"rev_{user_id}_{msg_id}.ogg"
    try:
        label = await generate_voice(original_text, new_gender, file_path)
        try: await query.message.delete()
        except: pass
        with open(file_path, "rb") as audio:
            new_msg = await query.message.chat.send_voice(
                voice=audio, 
                caption=f"🗣️ {label}",
                reply_markup=get_voice_kb(new_gender)
            )
            save_text_cache(new_msg.message_id, original_text)
            delete_text_cache(msg_id)
    except Exception as e:
        logger.error(f"Regen Error: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

# ---------------------------------------------------------------------------
# Main Deployment Logic
# ---------------------------------------------------------------------------
def main():
    # ១. ដំណើរការ Flask ក្នុង background thread (សម្រាប់ Render)
    threading.Thread(target=run_flask, daemon=True).start()

    # ២. ដំណើរការ Telegram Bot
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    
    print("🚀 Bot is Online & Fixed Bugs...")
    
    # ប្រើ drop_pending_updates=True ដើម្បីការពារ Error 409 Conflict
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
