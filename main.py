import logging
import os
import re
import asyncio
import threading
import imageio_ffmpeg as _iio_ffmpeg
from flask import Flask
from supabase import create_client, Client

# ── Flask Web Server ──────────────────────────────────────────────────────
app_flask = Flask(__name__)
@app_flask.route('/')
def health_check(): return "Bot is running online!", 200
def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app_flask.run(host='0.0.0.0', port=port)

# ── FFmpeg Configuration ──────────────────────────────────────────────────
_FFMPEG_EXE = _iio_ffmpeg.get_ffmpeg_exe()

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
# Configuration & Database
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SB_URL, SB_KEY)

# កំណត់សំឡេង (ប្រើតែ Edge-TTS)
VOICE_MAP = {
    "km": {"female": "km-KH-SreymomNeural", "male": "km-KH-PisethNeural"},
    "en": {"female": "en-US-AvaNeural", "male": "en-US-AndrewNeural"},
}

# ម៉េបល្បឿន (Rate Mapping)
RATE_MAP = {
    "rate_0.5": "-50%",
    "rate_1.0": "+0%",
    "rate_1.5": "+50%",
    "rate_2.0": "+100%"
}
RATE_LABELS = {"-50%": "x0.5", "+0%": "Normal", "+50%": "x1.5", "+100%": "x2.0"}

WELCOME_TEXT = (
    "🎵 សួស្តី! ខ្ញុំជា Bot បំលែងអក្សរទៅជាសំឡេង\n\n"
    "📌 វាយអក្សរភាសាណាមួយ ផ្ញើរមក Bot នឹងបំលែងដោយស្វ័យប្រវត្តិ!\n\n"
    "📢 Join My Channel: https://t.me/m11mmm112"
)

# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------
def sync_user_data(user):
    try:
        supabase.table("user_prefs").upsert({
            "user_id": user.id, 
            "username": user.username or user.first_name
        }, on_conflict="user_id").execute()
    except: pass

def get_user_prefs(user_id: int) -> dict:
    try:
        res = supabase.table("user_prefs").select("*").eq("user_id", user_id).execute()
        if res.data:
            return {
                "gender": res.data[0].get('gender') or "female",
                "rate": res.data[0].get('rate') or "+0%"
            }
    except: pass
    return {"gender": "female", "rate": "+0%"}

def update_user_pref(user_id: int, data: dict):
    try: supabase.table("user_prefs").update(data).eq("user_id", user_id).execute()
    except: pass

def save_text_cache(msg_id: int, text: str):
    try: supabase.table("text_cache").upsert({"message_id": msg_id, "original_text": text}).execute()
    except: pass

def get_text_cache(msg_id: int) -> str:
    try:
        res = supabase.table("text_cache").select("original_text").eq("message_id", msg_id).execute()
        return res.data[0]['original_text'] if res.data else None
    except: return None

def delete_text_cache(msg_id: int):
    try: supabase.table("text_cache").delete().eq("message_id", msg_id).execute()
    except: pass

# ---------------------------------------------------------------------------
# Audio Engine (Direct Edge-TTS + FFmpeg)
# ---------------------------------------------------------------------------
async def async_convert_audio(mp3_path: str, output_path: str):
    """បំប្លែង MP3 ទៅ OGG Opus ឱ្យត្រូវស្តង់ដារ Telegram"""
    cmd = [_FFMPEG_EXE, "-y", "-i", mp3_path, "-c:a", "libopus", "-b:a", "32k", output_path]
    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await process.wait()

async def generate_voice(text: str, gender: str, rate: str, output_path: str):
    is_khmer = bool(re.search(r"[\u1780-\u17FF]", text))
    lang_key = "km" if is_khmer else "en"
    voice_id = VOICE_MAP[lang_key][gender]
    
    tmp_mp3 = f"{output_path}.mp3"
    # ប្រើ Edge-TTS សម្រាប់គ្រប់ភាសា និងដាក់ល្បឿន (Rate)
    communicate = edge_tts.Communicate(text, voice_id, rate=rate)
    await communicate.save(tmp_mp3)
    
    await async_convert_audio(tmp_mp3, output_path)
    if os.path.exists(tmp_mp3): os.remove(tmp_mp3)
    return f"{'🇰🇭 ខ្មែរ' if is_khmer else '🇺🇸 English'} | {RATE_LABELS.get(rate, 'Normal')}"

# ---------------------------------------------------------------------------
# UI & Handlers
# ---------------------------------------------------------------------------
def get_control_kb(gender: str, current_rate: str):
    # ជួរទី១: ភេទ
    f_btn = "👩 ស្រី" + (" ✅" if gender == "female" else "")
    m_btn = "👨 ប្រុស" + (" ✅" if gender == "male" else "")
    
    # ជួរទី២: ល្បឿន
    rates = []
    for k, v in RATE_MAP.items():
        label = RATE_LABELS[v] + (" ✅" if v == current_rate else "")
        rates.append(InlineKeyboardButton(label, callback_data=k))
        
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f_btn, callback_data="tg_female"), InlineKeyboardButton(m_btn, callback_data="tg_male")],
        rates # ប៊ូតុងល្បឿនទាំង ៤
    ])

async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sync_user_data(update.effective_user)
    await update.message.reply_text(WELCOME_TEXT, reply_markup=ReplyKeyboardRemove(), disable_web_page_preview=True)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text: return
    if text.strip() == "🎵 សួស្តី!":
        await on_start(update, context); return

    sync_user_data(update.effective_user)
    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.RECORD_VOICE)
    file_path = f"v_{user_id}_{update.message.message_id}.ogg"
    
    try:
        label = await generate_voice(text, prefs["gender"], prefs["rate"], file_path)
        with open(file_path, "rb") as audio:
            sent_msg = await update.message.reply_voice(
                voice=audio, caption=f"🗣️ {label}", reply_markup=get_control_kb(prefs["gender"], prefs["rate"])
            )
            save_text_cache(sent_msg.message_id, text)
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    msg_id = query.message.message_id
    data = query.data
    
    original_text = get_text_cache(msg_id)
    if not original_text:
        await query.answer("❌ រកអត្ថបទដើមមិនឃើញ។", show_alert=True); return

    prefs = get_user_prefs(user_id)
    
    # ពិនិត្យថា User ចុចលើ ភេទ ឬ ល្បឿន
    if data.startswith("tg_"):
        new_gender = data.replace("tg_", "")
        if prefs["gender"] == new_gender: await query.answer(); return
        prefs["gender"] = new_gender
    elif data.startswith("rate_"):
        new_rate = RATE_MAP[data]
        if prefs["rate"] == new_rate: await query.answer(); return
        prefs["rate"] = new_rate

    # Update Database
    update_user_pref(user_id, prefs)
    await query.answer("🔄 កំពុងកែសម្រួលសំឡេង...")
    
    # បង្កើតសំឡេងថ្មី
    file_path = f"rev_{user_id}_{msg_id}.ogg"
    try:
        label = await generate_voice(original_text, prefs["gender"], prefs["rate"], file_path)
        try: await query.message.delete()
        except: pass
        
        with open(file_path, "rb") as audio:
            new_msg = await query.message.chat.send_voice(
                voice=audio, caption=f"🗣️ {label}", reply_markup=get_control_kb(prefs["gender"], prefs["rate"])
            )
            save_text_cache(new_msg.message_id, original_text)
            delete_text_cache(msg_id)
    except Exception as e:
        logger.error(f"Regen Error: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    threading.Thread(target=run_flask, daemon=True).start()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("🚀 Bot running (Edge-TTS only + Speed Control)...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
