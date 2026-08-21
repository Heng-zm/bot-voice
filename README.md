# bot-voice v4.2.2

Telegram AI, text-to-speech, OCR, and audio bot with a FastAPI backend and a
Telegram Mini App administration console.

The supported architecture is one Python process running FastAPI, Telegram
ingestion, provider calls, schedulers, and bounded background work. Redis and
the dedicated worker service are no longer part of the runtime.

## Highlights

- Telegram commands, callbacks, photos, voice, audio, and text handlers
- Gemini-first AI chat with health-aware provider fallback
- Gemini and Hugging Face OCR routing
- Edge TTS 7.2.8+ with optional Hugging Face Khmer TTS
- Broadcast and recurring schedule management
- Supabase-backed preferences, history, administrators, CORS, and runtime settings
- Signed Telegram Mini App administrator authentication
- Admin-editable `/start` welcome text and image with Support, Channel, and User Profile actions
- Process-local webhook replay protection with ownership-aware leases
- Bounded OCR, transcription, and audio workload admission
- Batched runtime-setting startup reads and cache-first blocked-user checks
- Dedicated bounded database execution for Telegram admin and schedule actions
- English and Khmer administration interface
- Polling startup that preserves queued Telegram updates by default

## Architecture

```text
Telegram / Browser
        |
        v
     app.main
        |
        +-- FastAPI and native admin routers
        +-- Telegram bot lifecycle and handlers
        +-- Gemini / Hugging Face / Edge TTS providers
        +-- schedulers and bounded background tasks
        |
        v
     Supabase
        +-- bot_settings
        +-- user_prefs
        +-- conversation_history
        +-- blocked_users
        +-- scheduled_broadcasts
        +-- feature_requests
        +-- text_cache
        +-- ai_api_keys
```

Deploy one active application instance. Webhook replay state is process-local,
so horizontal scaling requires a shared replay and coordination design.

## Requirements

- Python 3.11 or 3.12
- FFmpeg and Opus runtime libraries
- A Telegram bot token
- A Supabase project for persistent features
- A Gemini API key for the default AI and OCR route

Runtime dependencies are installed from `requirements.txt`. Edge TTS is bounded
to `>=7.2.8,<8` because older endpoint implementations can receive HTTP 403 from
Microsoft's synthesis service.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and replace the placeholder values. Run
`supabase_bot_setup.sql` in the target Supabase project if the application
tables have not been created yet.

## Minimum environment

```env
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
TELEGRAM_BOT_TOKEN=123456789:REPLACE_ME
GEMINI_API_KEY=REPLACE_ME
AI_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
ADMIN_IDS=123456789
```

`SUPABASE_KEY` remains accepted as a compatibility alias, but a server-side
service-role key is recommended for administrator tables protected by RLS.
Never expose this key to the Mini App or another frontend.

`WEB_SECRET_KEY`, `FLASK_SECRET_KEY`, and
`TELEGRAM_WEBHOOK_SECRET_TOKEN` are optional explicit overrides. Stable,
domain-separated values are derived from the Telegram bot token when possible,
and the generated webhook token is persisted through `bot_settings`.

## AI, OCR, and TTS providers

Gemini is the default chat and OCR provider:

```env
AI_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
OCR_PROVIDER=gemini
```

Hugging Face is optional. Before setting `AI_PROVIDER=hf`, select an `HF_MODEL`
that is currently served by an inference provider enabled for the account.
Provider availability can change independently of this application. When an
eligible provider fails, the internal provider manager can route to another
healthy provider.

Edge TTS is the default general speech path. Optional provider credentials and
models can be configured with `HF_TOKEN`, `HF_TTS_SPACE`, and `GRADIO_TOKEN`.

## Telegram runtime modes

### Polling

Polling is suitable for one active bot process and local deployments:

```env
BOT_MODE=POLLING
TELEGRAM_POLLING_DROP_PENDING_UPDATES=false
```

Pending updates are preserved by default across startup and runtime mode
transitions. Set `TELEGRAM_POLLING_DROP_PENDING_UPDATES=true` only when an
operator intentionally wants to discard a stale or broken backlog.

Only one process may poll a Telegram bot token at a time. Running multiple
pollers causes Telegram `409 Conflict` responses.

### Webhook

Webhook mode requires a public HTTPS base URL:

```env
BOT_MODE=WEBHOOK
TELEGRAM_WEBHOOK_URL=https://bot.example.com
# TELEGRAM_WEBHOOK_SECRET_TOKEN=optional-explicit-secret
```

`RENDER_EXTERNAL_URL` can provide the webhook base URL when
`TELEGRAM_WEBHOOK_URL` is not set. Webhook updates use process-local replay
protection, short processing leases, completed-update TTLs, and ownership
tokens. Keep `TELEGRAM_WEBHOOK_DROP_PENDING_UPDATES=false` unless deliberately
clearing Telegram's queue.

## Run

Preferred command for container panels and Linux hosts:

```bash
chmod +x start.sh
./start.sh
```

The launcher selects `python3` (falling back to `python`), requires Python
3.11+, checks application syntax before startup, enables unbuffered logs, and
runs from the repository directory. Optional startup controls are:

```env
PYTHON_BIN=/path/to/python
INSTALL_REQUIREMENTS_ON_START=false
STARTUP_COMPILE_CHECK=true
```

Dependency installation is disabled by default to avoid reinstalling packages
during every automatic restart. Enable it only when the host does not install
`requirements.txt` during its build step. The direct command remains supported:

```bash
python -m app.main
```

The application does not self-ping. Configure the hosting platform to check
`/readyz` when an external uptime or readiness probe is required.

Compatibility launchers are also supported:

```bash
python main.py
python app/main.py
# From the parent directory (for panels fixed to /home/container):
python -m deploy.main
```

The Docker image exposes port `8080` and runs `python -m app.main`.

Do not deploy `PROCESS_ROLE=worker`. If an older environment still defines
`REDIS_URL`, remove it; v4.2.2 ignores Redis.

## Health and administration

`GET /readyz` reports whether the single-process runtime has started and shows
the settings backend, runtime role, provider scope, workload pressure, and
webhook replay state.

```bash
python -m app.healthcheck
```

The Telegram Mini App is served at `/miniapp/admin`. Its data routes require a
trusted administrator and accept Telegram init data through either:

- `X-Telegram-Init-Data: <initData>`
- `Authorization: Bearer <initData>`

The existing opaque administrator API bearer token remains supported. Cookie
writes require the CSRF header generated for the administrator session.

Dynamic CORS accepts only exact HTTP or HTTPS origins. Wildcards, URL paths,
credentials, queries, fragments, malformed ports, and invalid host syntax are
rejected.

## Workload and replay controls

Conservative defaults are provided in `.env.example`:

```env
TELEGRAM_OCR_MAX_CONCURRENT=2
TELEGRAM_TRANSCRIBE_MAX_CONCURRENT=2
TELEGRAM_AUDIO_MAX_CONCURRENT=2
TELEGRAM_WORKLOAD_QUEUE_TIMEOUT_S=6

WEBHOOK_PROCESSING_TTL_S=120
WEBHOOK_REPLAY_TTL_S=600
WEBHOOK_REPLAY_MAX_ENTRIES=50000
```

The `/runtime` Telegram panel and Mini App show accepted, active, waiting, and
rejected workloads plus webhook replay diagnostics.

## Validation

Use pytest as the canonical runner. Some extraction-boundary tests are
pytest-style functions and are not collected by `unittest discover`.

```bash
python -m pytest -q
python -m ruff check app tests
python -m compileall -q app tests
node --check static/admin/app.js
```

Current verified result:

```text
85 passed
Ruff passed
Python compilation passed
Admin JavaScript syntax passed
Edge TTS 7.2.8 live synthesis passed
Gemini chat and OCR live checks passed
Telegram identity and Supabase schema checks passed
```

Live broadcasts, destructive user-data actions, and scheduler mutations should
be tested only in a staging bot/project because they affect real users or
persistent data.

## Migration status

Runtime ownership, security, Mini App authorization, administrator management,
dynamic CORS, settings storage, and native admin APIs live outside the legacy
monolith. Telegram commands, callbacks, media handlers, guards, routing,
workload admission, and webhook replay protection are also extracted into
native modules under `app/services/telegram`.

`app/legacy.py` remains a staged compatibility layer for database helpers,
caches, broadcast primitives, and older administration surfaces. Thin wrappers
preserve older imports while migration continues.

Removed runtime components include:

- `app/worker.py`
- `app/services/jobs/`
- Redis-backed Telegram delivery state
- Redis readiness and worker heartbeat requirements
- administrator queue and worker controls

See `FINAL_RELEASE_V4_2_2.md`, `CODE_REVIEW_V4_2.md`, and
`UPDATE_V4_2_NOTES.md` for release and migration history.
