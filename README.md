# bot-voice — single-process v4.2

Telegram AI / TTS / OCR bot with a FastAPI backend and Telegram Mini App admin console.

This update removes the **dedicated Redis + worker architecture**. The supported production topology is now one application process running FastAPI, Telegram ingestion, provider execution, and small in-process background tasks. Persistent control-plane state uses the existing Supabase `bot_settings` table, with an in-memory fallback for local development.


## V4.2 production optimizations

V4.2 hardens the single-process topology introduced in v4 and the Telegram handler extraction from v4.1:

- native process-local webhook replay protection replaces the remaining Redis-era webhook lease implementation
- separate processing and completed-update TTLs let interrupted webhook updates retry quickly without weakening duplicate suppression
- ownership tokens prevent an expired/stale webhook handler from completing or releasing a newer lease
- bounded admission for OCR, transcription, and audio conversion prevents traffic bursts from exhausting CPU/RAM/provider capacity
- the Admin Mini App and Telegram `/runtime` panel show active/waiting/rejected workload pressure and webhook replay state
- the temporary Telegram legacy bridge now inspects actual `LOAD_GLOBAL` bytecode instead of treating attribute names as dependencies, reducing per-update compatibility overhead
- the combined-process supervisor restarts when either FastAPI or Telegram stops unexpectedly, including a normal return with no exception
- the Telegram Admin `WEB_KEY` action now matches the no-Redis architecture and generates a private explicit environment-secret candidate instead of attempting a broken Redis write

The default admission limits are conservative and configurable through `.env.example`.

## Architecture

```text
Telegram / Browser
       │
       ▼
   app.main
       │
       ├── FastAPI + native admin routers
       ├── Telegram bot lifecycle
       ├── AI / TTS / OCR providers
       ├── Supabase persistence
       └── in-process background tasks

Supabase
  ├── bot_settings   runtime overrides, admin IDs, CORS policy
  ├── user_prefs
  ├── conversation_history
  ├── blocked_users
  └── other existing bot tables
```

There is no separate `app.worker` process and no Redis package/runtime dependency.

## Run

Python 3.12 is the deployment target.

```bash
python -m pip install -r requirements.txt
python -m app.main
```

For container deployment, the included Dockerfile starts the same command and exposes port `8080`.

Do **not** deploy a second `PROCESS_ROLE=worker` service. If an old deployment still sets `REDIS_URL`, remove it; the v4 runtime ignores it.

## Minimum environment

Copy `.env.example` and configure at least the integrations you use:

```env
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=YOUR_SERVICE_ROLE_KEY
TELEGRAM_BOT_TOKEN=123456789:REPLACE_ME
GEMINI_API_KEY=REPLACE_ME
ADMIN_IDS=123456789
```

`WEB_SECRET_KEY`, `FLASK_SECRET_KEY`, and `TELEGRAM_WEBHOOK_SECRET_TOKEN` are optional explicit overrides. Without them, stable domain-separated values are derived from the Telegram bot token. A generated webhook token is persisted to Supabase when possible.

### Supabase requirement

The existing `public.bot_settings` table is used as the small persistent settings store. It must have at least:

```sql
create table if not exists public.bot_settings (
  key text primary key,
  value text not null,
  updated_by bigint,
  updated_at timestamptz not null default now()
);
```

Use a server-side service-role/secret key. Never expose it to the Mini App or frontend.

## Admin Mini App

Open `/miniapp/admin` from the configured Telegram Mini App launcher. The v4 UI adds:

- English / Khmer switching with persisted preference
- Telegram theme integration and haptic feedback
- request timeout and degraded/offline status
- partial refresh via `Promise.allSettled` so one failed endpoint does not blank the page
- safe DOM rendering instead of HTML string injection
- maintenance, runtime limits, provider reset, administrator management, and exact-origin CORS management
- architecture status showing `Single process · No Redis · No worker`
- live OCR/transcription/audio workload pressure and webhook replay diagnostics

Telegram init data works through both `X-Telegram-Init-Data` and the Mini App `Authorization: Bearer <initData>` fallback.

## Health

`GET /readyz` returns readiness for the single-process runtime and reports the settings-store backend. It no longer treats Redis or a worker heartbeat as readiness requirements.

Docker healthcheck:

```bash
python -m app.healthcheck
```

## Migration away from `legacy.py`

The migration is intentionally staged. v4 moves these active responsibilities out of the monolith:

- runtime ownership: `app/runtime.py`
- runtime secrets: `app/core/security.py`
- Telegram Mini App admin authorization: `app/core/telegram_auth.py`
- administrator persistence/confirmation: `app/core/admin_management.py`
- dynamic CORS policy: `app/core/cors.py`
- persistent control-plane settings: `app/services/settings/store.py`
- native admin runtime API: `app/api/v1/admin_runtime.py`

V4.1 moves the **live Telegram handler layer** out of `app/legacy.py` as well:

- command handlers: `app/services/telegram/commands.py`
- callback handlers: `app/services/telegram/callbacks.py`
- photo/voice/audio/text handlers: `app/services/telegram/media.py`
- rate-limit/security/stale-update/error guards: `app/services/telegram/guards.py`
- deterministic handler registration: `app/services/telegram/routing.py`

`app/legacy.py` now keeps thin compatibility wrappers for those public names so older imports continue to work. Shared helper functions, caches, broadcast primitives, database helpers, and some older admin surfaces still live in the monolith. Extracted handlers use a narrow `app/services/telegram/_legacy_runtime.py` bridge to resolve those remaining dependencies until they are migrated to native services. No normal Telegram handler is registered directly from the monolith anymore.

V4.2 additionally moves Telegram webhook replay ownership to `app/services/telegram/deduplication.py` and adds `app/services/telegram/workloads.py` for bounded expensive-work admission. The webhook replay store is intentionally **process-local**. Deploy one application instance for this architecture; multi-replica deployments need a shared database-backed replay/coordination design before horizontal scaling.

## Removed in v4

- `app/worker.py`
- `app/services/jobs/`
- Redis-backed Telegram delivery state
- Redis dependency / Redis readiness checks
- Admin job queue / worker controls
- separate worker deployment instructions

OCR and transcription execute inline in the application process and keep the existing Telegram progress flow.

## Validation

Run:

```bash
python -m compileall -q app tests
python -m unittest discover -s tests -v
node --check static/admin/app.js
```

The v4.2 update package was validated in the available container with **37 focused tests plus 5 subtests**, along with Python compile checks and `node --check` for the Admin Mini App. Those tests cover authentication, admin management, CORS, runtime security, provider timeout/fallback, Telegram extraction boundaries, workload admission, webhook lease expiry/ownership, UI architecture checks, and single-process supervision.

The container used for this review does not currently have the `python-telegram-bot` package installed, so tests that import the full Telegram SDK cannot be collected here. After applying the updater to the real repository/environment, install `requirements.txt` and run the complete project test suite before deployment.

See `CODE_REVIEW_V4_2.md`, `UPDATE_V4_2_NOTES.md`, and the earlier v4/v4.1 notes for review findings and migration details.
