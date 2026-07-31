# bot-voice

Telegram voice/AI bot with a FastAPI HTTP surface, Gemini integrations,
speech generation, Supabase persistence, Redis coordination, and admin tools.

## Project structure

```text
app/
├── main.py                  # ASGI and combined-process entry point
├── config.py                # Settings/environment compatibility boundary
├── api/
│   ├── dependencies.py
│   └── v1/
│       ├── health.py
│       ├── telegram.py
│       ├── ai.py
│       └── admin_cors.py
├── core/
│   ├── async_loop.py
│   ├── security.py
│   ├── cors.py
│   ├── middleware.py
│   └── json.py
├── services/
│   ├── telegram/
│   │   ├── client.py
│   │   ├── handlers.py
│   │   └── deduplication.py
│   ├── ai/
│   │   ├── gemini.py
│   │   ├── tts.py
│   │   └── language.py
│   └── db/
│       ├── supabase.py
│       └── redis.py
├── utils/
│   ├── time.py
│   └── file_io.py
├── compat/
│   └── flask.py
└── legacy.py                # Preserved runtime during staged extraction
tests/
├── test_core.py
├── test_telegram.py
├── test_ai.py
├── test_voxcpm2.py
├── test_runtime_security.py
└── test_dynamic_cors.py
```

The runtime was previously a single 28k-line root module. `app/legacy.py`
preserves that behavior while the new modules provide stable ownership
boundaries for incremental extraction. Avoid adding new features to
`app/legacy.py`; put them in the matching package above.

## Run

Copy `.env.example` to `.env`, configure the required values, then:

```bash
pip install -r requirements.txt
python -m app.main
```

Never commit `.env`. If a populated `.env` was committed previously, remove it
from Git tracking and rotate every exposed credential; adding `.gitignore`
cannot erase secrets from existing history. `.dockerignore` prevents the local
environment file from being copied into container images.

The ASGI app is also exposed as `app.main:app`. Running only Uvicorn serves the
HTTP application but does not start the combined Telegram/scheduler lifecycle.

## Runtime secrets and CORS

`.env` contains only the five core connection values:

```dotenv
REDIS_URL=
SUPABASE_URL=
SUPABASE_KEY=
TELEGRAM_BOT_TOKEN=
GEMINI_API_KEY=
```

At startup the application atomically loads or creates 64-character
`TELEGRAM_WEBHOOK_SECRET_TOKEN`, `WEB_SECRET_KEY`, and `FLASK_SECRET_KEY`
values in Redis. They have no expiry and their raw values are never logged.
Startup fails closed when Redis is unavailable. When a new Telegram secret
needs registration and the bot is in webhook mode, startup registers it with
Telegram before accepting traffic.

CORS origins are exact HTTP(S) origins stored in Redis and mirrored to the
Supabase `bot_settings` row named `frontend_allowed_origins`. Wildcards, URL
paths, credentials, query strings, and fragments are rejected. The initial
production policy is an empty list.

Authenticated administrators can manage it using:

```text
GET    /api/admin/cors
POST   /api/admin/cors       {"origin":"https://admin.example.com"}
DELETE /api/admin/cors       {"origin":"https://admin.example.com"}
```

Bearer-authenticated writes do not need CSRF. Cookie-authenticated writes must
send the `X-CSRF-Token` returned by the admin login/bootstrap API. Each process
caches the Redis policy for five seconds and updates its local cache
immediately after an edit.

## VoxCPM2 voice cloning

In Telegram, run `/voxcpm2` and follow the setup panel:

1. Choose **Controllable Clone** to preserve the reference voice while changing
   emotion, pace, or style, or choose **Ultimate Clone** for transcript-guided
   continuation that preserves more of the original vocal detail.
2. Upload a clean Telegram voice message or WAV/MP3/OGG/FLAC file. A 5–30
   second clip is recommended; the default configured maximum is 50 seconds.
3. For Controllable Clone, optionally enter a style instruction. For Ultimate
   Clone, enter the exact transcript spoken in the reference clip.
4. Select VoxCPM2 and send the text that should be spoken.

Only clone a voice you own or have permission to use.

## Test

```bash
python -m unittest discover -s tests -v
```
