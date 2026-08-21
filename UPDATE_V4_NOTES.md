# v4 — Single-process / No-Redis migration

## Scope

This release is a focused architecture and UX cleanup based on the v3.1 update line. It removes the dedicated worker/Redis topology while preserving current Telegram, AI, TTS, OCR, Supabase, and Mini App behavior.

## Runtime changes

- One supported runtime process: `python -m app.main`.
- `PROCESS_ROLE=worker` now fails with a migration message instead of silently starting an incomplete service.
- Removed `app.worker` and `app/services/jobs` durable Redis queue.
- Removed Redis from runtime requirements.
- Runtime overrides now use Supabase `bot_settings` with memory fallback.
- Admin IDs and dynamic CORS policy now use the same settings store.
- Redis is no longer a readiness dependency.
- Old `REDIS_URL` is ignored by the compatibility monolith; remove it from deployment configuration.

## Bug fixes

1. **Mini App Bearer auth mismatch** — the frontend sent a Bearer fallback, while the backend only treated TMA as Telegram init data. Both are now accepted.
2. **Stale CORS store capture** — middleware could keep the pre-start global store instead of the configured store. It now resolves the current store at request time.
3. **Provider timeout clamp** — very small configured timeouts were silently raised to 100ms. The lower bound is now 10ms, so timeout fallback behaves as configured and the next provider can run.
4. **Redis startup coupling** — session/webhook security no longer fails because Redis is absent. Stable secrets are explicit or token-derived, with webhook persistence in Supabase.
5. **Queue deletion import risk** — normal OCR/transcription paths run inline; queue files are removed from the supported architecture.

## UX improvements

- English/Khmer translation for the Mini App.
- Telegram theme CSS variables and haptic feedback.
- 12-second API timeout with clear degraded/offline state.
- Partial dashboard refresh when one endpoint fails.
- Correct four-column metric layout on desktop and responsive mobile behavior.
- Removed obsolete Redis/worker/job controls.
- Safer DOM updates using `textContent` and element APIs.
- Refresh pauses while the Mini App is hidden and refreshes when visible again.
- Architecture and settings-store status are visible in the dashboard.

## `legacy.py` migration status

Moved out of the monolith in this release:

- runtime lifecycle
- runtime security policy
- Telegram Mini App administrator authorization
- administrator management
- CORS persistence/policy
- runtime settings persistence
- provider/runtime admin endpoints

Still intentionally retained in `legacy.py`:

- compatibility FastAPI shell
- Telegram command/message handlers
- older cache/database helper functions
- in-process broadcast helpers
- old advanced admin compatibility surfaces

Dead Redis-oriented helper names may still exist in `legacy.py`, but no Redis package/client is started by the supported runtime. Removing those remaining helpers should be a separate extraction pass with handler-level regression tests.

## Deployment migration

1. Remove the Redis service and `REDIS_URL` environment variable.
2. Remove the dedicated worker service/command.
3. Keep one service running `python -m app.main`.
4. Ensure Supabase `bot_settings` exists.
5. Keep `ADMIN_IDS` for first bootstrap; administrator changes made in the Mini App persist to Supabase.
6. Run the validation commands from README before deployment.
