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
│       ├── admin.py
│       └── admin_cors.py
├── core/
│   ├── async_loop.py
│   ├── security.py
│   ├── telegram_auth.py
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
├── test_runtime_security.py
├── test_dynamic_cors.py
├── test_telegram_auth.py
└── test_admin_api.py
static/
└── admin/
    ├── index.html
    ├── styles.css
    └── app.js
```

The runtime was previously a single 28k-line root module. `app/legacy.py`
preserves that behavior while the new modules provide stable ownership
boundaries for incremental extraction. Avoid adding new features to
`app/legacy.py`; put them in the matching package above.

## Run

Copy `.env.example` to `.env`, configure the required values, then:

```bash
pip install -r requirements.lock
python -m app.main
```

For local development, use an isolated virtual environment and install the
quality tools as well:

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.lock -r requirements-dev.txt
python -m ruff check .
python -m unittest discover -s tests -v
```

The lint baseline intentionally excludes `app/legacy.py` during the staged
extraction. New or extracted modules remain checked in CI.

`requirements.lock` is the pinned runtime dependency set used by Docker and
CI. Regenerate it only when intentionally updating dependencies:

```bash
python -m pip install pip-tools
pip-compile --output-file requirements.lock requirements.txt
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
SUPABASE_SERVICE_ROLE_KEY=
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

## Telegram Admin Mini App

The dashboard is served at:

```text
https://YOUR_PUBLIC_HOST/miniapp/admin
```

It validates raw `Telegram.WebApp.initData` on every API request and accepts it
through `X-Telegram-Init-Data` or `Authorization: Bearer <initData>`. The
signature, timestamp, user JSON, and Redis administrator membership are all
checked server-side. The browser's `initDataUnsafe` object is never trusted for
authorization.

For Hugging Face chat models, `HF_MODEL` accepts repository IDs or compact
aliases: `qwen2.5:3b` maps to `Qwen/Qwen2.5-3B-Instruct`, and `llama3.2:3b`
maps to `meta-llama/Llama-3.2-3B-Instruct`.

To launch the Mini App from a separate Bot 2 while keeping Bot 1 as the main
bot, set `TELEGRAM_ADMIN_BOT_TOKEN` to Bot 2's token. Normal bot operations
continue using `TELEGRAM_BOT_TOKEN`; if the admin token is omitted, it falls
back to the main bot token.

Administrator IDs are stored in this Redis set, not `.env`:

```text
tgbot:security:admin_user_ids:v1
```

Bootstrap the first administrator once using your Redis provider console:

```bash
redis-cli -u "YOUR_REDIS_URL" SADD tgbot:security:admin_user_ids:v1 123456789
```

Use the numeric Telegram user ID. Existing optional `ADMIN_IDS` deployment
values are migrated into this set on startup for backward compatibility, after
which they can be removed. Configure the same HTTPS Mini App URL in BotFather.
When the bot has a public webhook/base URL, `/admin` also includes an
**Open Admin Mini App** button automatically.

Protected Mini App endpoints:

```text
GET    /api/admin/me
GET    /api/admin/stats
GET    /api/admin/settings
POST   /api/admin/settings
GET    /api/admin/cors
POST   /api/admin/cors
DELETE /api/admin/cors
```

`POST /api/admin/settings` exposes only validated, bounded hot-runtime
controls and maintenance mode. Deployment-level settings and secrets are not
editable from the Mini App.

## Test

```bash
python -m unittest discover -s tests -v
```

## Runtime reliability update

See [`UPDATE_NOTES.md`](UPDATE_NOTES.md) for the durable worker lifecycle,
provider timeout consistency, unified `RuntimeContext`, queue backpressure,
admin operations UI, and staged migration instructions.

## Runtime reliability update v2

See [`UPDATE_V2_NOTES.md`](UPDATE_V2_NOTES.md) for job progress, terminal job
history, worker drain/resume, extracted utility modules, CI, and Docker health
checks.

## Runtime reliability update v3

V3 migrates the real Telegram OCR and transcription entry points to the
Redis queue. The web process now validates the request, creates one editable
progress message, and enqueues durable Telegram `file_id` references. A worker
downloads the source, runs OCR/transcription, stores the full text as an
artifact, and edits the same progress message through Redis-backed idempotent
delivery.

### Separate web and worker services

Production deployments should run both commands from the same release:

```bash
# Web/API + Telegram webhook ingestion. Does not consume jobs.
PROCESS_ROLE=web uvicorn app.main:app --host 0.0.0.0 --port 8080

# Background OCR/transcription/TTS/broadcast workers. Does not bind a port.
PROCESS_ROLE=worker python -m app.worker
```

`python -m app.main` remains the backward-compatible combined process for local
runs and one-service deployments.

### Artifact storage

Supabase Storage is selected automatically when a Supabase client is
configured. Create a private bucket named `bot-job-artifacts`, or set
`BOT_ARTIFACT_STORAGE_BUCKET` to an existing private bucket. The server key
must have upload, download, and delete access to that bucket.

Optional deployment controls:

```text
BOT_ARTIFACT_STORAGE_MODE=auto     # auto | supabase | local
BOT_ARTIFACT_STORAGE_BUCKET=bot-job-artifacts
BOT_ARTIFACT_LOCAL_DIRECTORY=data/job-artifacts
BOT_ARTIFACT_MAX_BYTES=52428800
BOT_JOB_WORKERS=2
BOT_JOB_QUEUE_MAX=1000
DURABLE_OCR_ENABLED=true
DURABLE_TRANSCRIPTION_ENABLED=true
```

Local artifact storage is for development or a single shared host. Use
Supabase mode when workers can run on different machines or containers.

See [`UPDATE_V3_NOTES.md`](UPDATE_V3_NOTES.md) for migration and rollback
instructions.

## Reliability and observability update v4

V4 makes Telegram voice delivery retry-safe, automatically restarts workers
whose outer loop stops unexpectedly, and deletes expired local or Supabase
artifacts through a Redis expiration registry. The admin Mini App now exposes
queue age, hourly throughput, final failure rate, job-type filters, and search.

Pure validation and normalization logic now lives in focused TTS, OCR,
broadcast, and runtime-settings service modules. `app/legacy.py` keeps thin
compatibility wrappers while extraction continues without breaking imports.

Artifact cleanup runs in worker and combined roles every five minutes by
default. `BOT_ARTIFACT_CLEANUP_SECONDS` is an optional deployment override; it
does not belong in the minimal `.env.example`.

The V4 hardening pass also pipelines queue statistics, preserves artifact
cleanup records across transient storage outages, resets stale worker backoff
after healthy operation, and prevents overlapping Mini App pagination calls.

See [`UPDATE_V4_NOTES.md`](UPDATE_V4_NOTES.md) for operational details and
[`VALIDATION_V4.md`](VALIDATION_V4.md) for the verification record.
