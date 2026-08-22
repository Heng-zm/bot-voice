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

## Run

Linux or hosting panel:

```bash
chmod +x start.sh
./start.sh
```

Direct launchers:

```bash
python -m app.main
python main.py
python app/main.py
```

`start.sh` checks Python 3.11+, optionally installs requirements, compiles the
application, and starts the polling bot. Dependency installation on every
restart is disabled by default; enable it only when required:

```env
INSTALL_REQUIREMENTS_ON_START=true
STARTUP_COMPILE_CHECK=true
```

## Deployment

Deploy this project as a background worker or persistent process. It does not
open an HTTP port and does not provide `/readyz`, webhook, REST API, dashboard,
or Mini App routes.

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
