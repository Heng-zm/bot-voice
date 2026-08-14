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

Create `.env`, configure the required values, then:

```bash
python -m pip install -r requirements.lock
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
pytest -q
```

The lint baseline intentionally excludes `app/legacy.py` during the staged
extraction. New or extracted modules remain checked in CI.

`requirements.txt` defines the direct runtime dependencies used to regenerate
`requirements.lock`. Docker, CI, and local setup install the lock file so they
all run the same tested versions.

Never commit `.env`. If a populated `.env` was committed previously, remove it
from Git tracking and rotate every exposed credential; adding `.gitignore`
cannot erase secrets from existing history. `.dockerignore` prevents the local
environment file from being copied into container images.

The ASGI app is also exposed as `app.main:app`. Running only Uvicorn serves the
HTTP application but does not start the combined Telegram/scheduler lifecycle.

Optional release metadata can be injected at deploy time for status reporting:

```dotenv
BOT_BUILD_VERSION=2026.08.12
RELEASE_SHA=abcdef1234567890
RELEASE_CREATED_AT=2026-08-12T10:30:00Z
```

The bot exposes that metadata through:

```text
Telegram command: /version
HTTP endpoints:   /version and /api/version
Admin API:        GET /api/admin/stats
```

## Runtime secrets and CORS

`.env` contains the core connection values and the Redis feature switch:

```dotenv
REDIS_URL=
REDIS_ENABLED=false
ADMIN_IDS=123456789
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
TELEGRAM_BOT_TOKEN=
GEMINI_API_KEY=
```

With `REDIS_ENABLED=false`, the application does not connect to Redis even if
`REDIS_URL` is present. Run the application as one combined process (the
default `PROCESS_ROLE=combined`). Jobs, delivery idempotency, caches, and
generated session secrets are held in memory and are lost on restart; separate
web and worker processes and multi-instance coordination require Redis.
Set `ADMIN_IDS` to a comma-separated list of Telegram user IDs when Redis is
disabled. Runtime administrator allowlist edits require Redis.

Set `REDIS_ENABLED=true` and configure `REDIS_URL` to restore durable,
multi-process operation.

### Server resource profile

The bot defaults to `BOT_RESOURCE_PROFILE=efficient`. This keeps two queue
workers for responsive TTS delivery while using smaller HTTP/Telegram pools,
provider thread pools, memory caches, and less frequent artifact cleanup. It is
the recommended profile for a single small Wispbyte/Render-style server.

Hosts with more CPU and memory can opt out of the efficient caps:

```dotenv
BOT_RESOURCE_PROFILE=balanced
```

Use `performance` only for a larger dedicated host. Environment variables and
admin performance controls still tune individual values; in `efficient` mode,
oversized persisted pool/concurrency values are safely capped on startup.

### Automatic Wispbyte updates

In the Wispbyte client panel, open the server's **GitHub** page, set the
repository to `https://github.com/Heng-zm/bot-voice`, select branch `main`, and
enable **Auto-update on startup**. Wispbyte documents this as its supported
repository update path.

`start.sh` also performs a safe fast-forward check on Wispbyte before launching
the bot. It never resets files or overwrites tracked server edits: a dirty or
diverged checkout starts its existing version and prints a warning. Controls:

```dotenv
AUTO_UPDATE_ON_START=true
AUTO_UPDATE_BRANCH=main
```

The Wispbyte default is enabled; set `AUTO_UPDATE_ON_START=false` to disable the
fallback. Updates apply when the server starts or restarts. A push alone does
not force a running Wispbyte server to restart because the host does not publish
a documented deployment/restart API.

When Redis is enabled, the application atomically loads or creates 64-character
`TELEGRAM_WEBHOOK_SECRET_TOKEN`, `WEB_SECRET_KEY`, and `FLASK_SECRET_KEY`
values in Redis. They have no expiry and their raw values are never logged.
Startup fails closed if Redis was enabled but is unavailable. In Redis-disabled
mode, these values are generated in memory and sessions reset on restart. When
a new Telegram secret needs registration and the bot is in webhook mode,
startup registers it with Telegram before accepting traffic.

CORS origins are exact HTTP(S) origins stored in Redis when enabled and mirrored
to the Supabase `bot_settings` row named `frontend_allowed_origins`. Without
Redis, Supabase is the persistent source. Wildcards, URL paths, credentials,
query strings, and fragments are rejected. The initial production policy is an
empty list.

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

## Atomic Supabase bot locks

Scheduler and Telegram leader leases use the `public.bot_locks` table. Run
[`scripts/supabase_bot_locks.sql`](scripts/supabase_bot_locks.sql) once in the
Supabase SQL Editor to install the atomic `acquire_bot_lock` function. The SQL
is idempotent and grants execution only to `service_role`.

The application prefers that single-statement database function. During a
rolling deployment where the SQL has not been installed yet, it falls back to
conditional updates plus `ON CONFLICT DO NOTHING`; concurrent first acquisition
therefore returns `false` for the losing instance instead of raising
`bot_locks_pkey` / PostgreSQL `23505`.

## Test

```bash
python -m unittest discover -s tests -v
```

## Runtime reliability update

The runtime includes durable worker lifecycle management, provider timeout
consistency, a unified `RuntimeContext`, queue backpressure, and admin
operations UI.

## Runtime reliability update v2

The v2 reliability work added job progress, terminal job history, worker
drain/resume, extracted utility modules, CI checks, and Docker health checks.

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

Use `/version` in Telegram or `GET /api/version` after deploy to confirm the
running release metadata and active process role.

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

Deploy the worker and web processes from the same release when using the split
web/worker topology.

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

The v4 hardening pass added retry-safe Telegram voice delivery, worker
supervision, artifact expiry cleanup, and richer admin queue telemetry.
