# Bot Voice

Telegram-only AI, OCR, transcription, and text-to-speech bot. The FastAPI
backend and Telegram Mini App have been removed; the process now runs through
Telegram long polling with its internal broadcast scheduler.

## Features

- AI chat with Gemini and optional Hugging Face fallback
- Khmer and multilingual text-to-speech
- OCR and audio transcription
- User profiles, preferences, history, and welcome messages
- Telegram `/admin` controls
- Immediate, scheduled, and recurring daily broadcasts
- Supabase persistence with bounded outage retries
- Provider health, workload limits, caches, and maintenance mode

## Requirements

- Python 3.11 or newer
- FFmpeg
- Telegram bot token
- Supabase project and service-role key

## Install

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, replace the placeholder credentials, and run
`supabase_bot_setup.sql` in the Supabase SQL Editor.

Minimum configuration:

```env
TELEGRAM_BOT_TOKEN=123456789:REPLACE_ME
ADMIN_IDS=123456789
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
GEMINI_API_KEY=REPLACE_ME
AI_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
BOT_MODE=POLLING
```

`SUPABASE_KEY` remains accepted as a compatibility alias. Keep the service-role
key private and never send it through Telegram.

## 🚀 Easy Server Deployment

### Method 1: Docker Compose (Recommended)

1. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
2. Start the bot in the background:
   ```bash
   docker compose up -d --build
   ```
3. View live logs:
   ```bash
   docker compose logs -f
   ```

---

### Method 2: Automated 1-Click Script (`deploy.sh`)

On any Linux VPS (Ubuntu / Debian / CentOS / RHEL):
```bash
chmod +x deploy.sh
./deploy.sh
```
*The script automatically detects Docker (or configures a Python virtual environment with dependencies) and launches the bot.*

---

### Method 3: Linux Systemd Daemon (24/7 Background Service)

1. Copy the service file template:
   ```bash
   sudo cp bot-voice.service.example /etc/systemd/system/bot-voice.service
   ```
2. Edit `/etc/systemd/system/bot-voice.service` with your user and working directory.
3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now bot-voice
   ```
4. Check status & logs:
   ```bash
   sudo systemctl status bot-voice
   journalctl -u bot-voice -f
   ```

### Method 4: Anajak Cloud (https://anajak.cloud/)

**For Anajak Cloud VPS (Ubuntu/Debian):**
```bash
git clone https://github.com/Heng-zm/bot-voice.git
cd bot-voice
cp .env.example .env
# Edit .env with your tokens
docker compose up -d --build
```

**For Anajak Cloud Bot / Pterodactyl Panel:**
1. Upload files and `.env` via Web File Manager / SFTP.
2. In Panel **Startup**: select **Python 3.11+** and set Startup Command to `./start.sh` (or `python main.py`).
3. Set panel environment variable `INSTALL_REQUIREMENTS_ON_START=true`.
4. Click **Start**.

---

## 💻 Manual Run & Development

Direct launchers:

```bash
python -m app.main
python main.py
python app/main.py
```

`start.sh` checks Python 3.11+, optionally installs requirements, compiles the
application, and starts the polling bot:

```bash
chmod +x start.sh
./start.sh
```


Only one process may poll a Telegram token unless distributed ownership is
enabled. For a normal single-process deployment:

```env
TELEGRAM_MULTI_SERVER_ENABLED=false
TELEGRAM_ACTIVE_LOCK_ENABLED=false
TELEGRAM_ACTIVE_LOCK_REQUIRED=false
SCHED_LOCK_ENABLED=false
```

For multiple services using the same bot token, run the `bot_locks` section of
`supabase_bot_setup.sql` and enable both ownership locks:

```env
TELEGRAM_MULTI_SERVER_ENABLED=true
TELEGRAM_ACTIVE_LOCK_ENABLED=true
TELEGRAM_ACTIVE_LOCK_REQUIRED=true
SCHED_LOCK_ENABLED=true
SCHED_LOCK_REQUIRED=true
```

Supabase timeouts are bounded so a DNS/database outage cannot occupy all bot
workers:

```env
SUPABASE_CONNECT_TIMEOUT_S=5
SUPABASE_HTTP_TIMEOUT_S=12
```

## Administration and broadcasts

Open `/admin` inside Telegram for settings, users, provider status, cache
controls, welcome messages, broadcast previews, scheduling, and reports.

Daily broadcasts use Phnom Penh time and automatically move to the next day
after completion. Test messages with an administrator account before sending to
all users. Failed scheduled broadcasts remain available through the Telegram
admin controls for inspection and retry.

## Verification

```bash
python -m compileall -q app
ruff check app --output-format concise
python -m pip check
```

The Docker image starts `python -m app.main` as a Telegram-only process and no
longer exposes a port or HTTP health check.
